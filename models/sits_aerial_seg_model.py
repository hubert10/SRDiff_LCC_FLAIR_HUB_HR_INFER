import torch
import timm
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from utils.hparams import hparams
from timm.layers import create_conv2d
from models.sits_branch import SITSSegmenter
from models.fusion_module.aer_cross_sat_atts import FFCA
from models.decoders.unet_former_decoder import UNetFormerDecoder


class SITSAerialSegmenter(nn.Module):
    def __init__(self, gaussian, config):
        super().__init__()
        self.gaussian = gaussian
        self.config = config

        # 1. SITS Network
        self.sits_network = SITSSegmenter(
            img_size=config["inputs"]["sr_patch_size"],
            in_chans=config["inputs"]["num_channels_sat"],
            embed_dim=config["models"]["t_convformer"]["embed_dim"],
            uper_head_dim=config["models"]["t_convformer"]["uper_head_dim"],
            depths=config["models"]["t_convformer"]["depths"],
            num_heads=config["models"]["t_convformer"]["num_heads"],
            mlp_ratio=config["models"]["t_convformer"]["mlp_ratio"],
            num_classes=config["inputs"]["num_classes"],
            nbts=config["inputs"]["nbts"],
            pool_scales=config["models"]["t_convformer"]["pool_scales"],
            spa_temp_att=config["models"]["t_convformer"]["spa_temp_att"],
            conv_spa_att=config["models"]["t_convformer"]["conv_spa_att"],
            decoder_channels=config["models"]["maxvit"]["decoder_channels"],
            window_size=config["models"]["maxvit"]["window_size"],
            d_model=config["models"]["t_convformer"]["d_model"],
            config=config,
        )

        if not hparams["infer"]:
        # if not hparams["resume"] and not hparams["infer"]:
            if hparams["cond_net_ckpt"] != "" and os.path.exists(hparams["cond_net_ckpt"]):
                weights_path = hparams["cond_net_ckpt"]
                if torch.cuda.is_available():
                    old_dict = torch.load(weights_path, weights_only=False)
                else:
                    old_dict = torch.load(weights_path, map_location=torch.device("cpu"))
                model_dict = self.sits_network.state_dict()
                old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
                model_dict.update(old_dict)
                self.sits_network.load_state_dict(model_dict)

        # # 2. Aerial Network
        self.aerial_net = timm.create_model(
            "maxvit_tiny_tf_512.in1k",
            pretrained=True,
            features_only=True,
            num_classes=config["inputs"]["num_classes"],
        )

        # Get first conv layer (usually called 'stem.conv' in MaxViT)
        conv1 = (
            self.aerial_net.stem.conv1
        )  # <-- sometimes it's model.stem.conv or model.conv_stem, check print(model)

        # Create new conv with 5 input channels instead of 3
        new_conv = create_conv2d(
            in_channels=config["inputs"][
                "num_channels_aer"
            ],  # Use num_channels from config
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=1,  # original padding was None, but we set it to 1 for compatibility
            bias=conv1.bias is not None,
        )

        # Initialize the first 3 channels with pretrained weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv1.weight  # copy RGB weights
            # Initialize the extra channels randomly (e.g., Kaiming normal)
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

        # Replace the old conv with the new one
        self.aerial_net.stem.conv1 = new_conv

        # give latent_diff access
        self.gaussian.aerial_net = self.aerial_net

        encoder_channels = [
            config["models"]["t_convformer"]["embed_dim"],
            config["models"]["t_convformer"]["embed_dim"] * 2,
            config["models"]["t_convformer"]["embed_dim"] * 4,
            config["models"]["t_convformer"]["embed_dim"] * 8,
        ]

        # 3. Decoder from U-Net Former paper
        self.decoder = UNetFormerDecoder(
            encoder_channels,
            config["models"]["maxvit"]["decoder_channels"],
            config["models"]["maxvit"]["dropout"],
            config["models"]["maxvit"]["window_size"],
            config["inputs"]["num_classes"],
        )
        self.fusion_module = FFCA(
            aer_channels_list=config["models"]["t_convformer"]["aer_channels_list"],
            sits_channels_list=config["models"]["t_convformer"]["sits_channels_list"],
            num_heads=8,
        )

    def forward(
        self,
        img: torch.FloatTensor,
        img_sr: torch.FloatTensor,
        dates: torch.FloatTensor,
    ):

        h, w = img.size()[-2:]
        # Aerial branch
        hr_0, hr_1, hr_2, hr_3, hr_4 = self.aerial_net(img)

        # SITS branch
        sits_logit, multi_lvls_cls, red_temp_feats, _ = self.sits_network(img_sr, dates)
        # print()
        # print("res 0:", res2.shape)
        # print("res 1:", res3.shape)
        # print("res 2:", res4.shape)

        # print()
        # print("red_temp_feats:", red_temp_feats[0].shape)
        # print("red_temp_feats:", red_temp_feats[1].shape)
        # print("red_temp_feats:", red_temp_feats[2].shape)

        # print("multi_lvls_outs:", multi_lvls_outs[3].shape)
        # Fusion FFCA
        fus_2, fus_3, fus_4 = self.fusion_module([hr_2, hr_3, hr_4], red_temp_feats)

        # Decoder
        logits = self.decoder(hr_0, hr_1, fus_2, fus_3, fus_4, h, w)
        return sits_logit, multi_lvls_cls, logits


