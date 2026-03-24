import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.encoders.t_convformer import TConvFormer
from models.decoders.unet_decoder import UNetDecoder
from models.decoders.uper_head import UPerHead

# II.TConvFormer like Swin Architecture (Encoder + Decoder)
# An encoder is implemented here;
# 1. ConvFormerSits(For timeseries)
# A decoder is implemented here;
# 1. UPerHead
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# Description: It uses shifted window approach for computing self-attention
# Adapated from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Paper associated to it https://ieeexplore.ieee.org/document/9710580


class SITSSegmenter(nn.Module):
    def __init__(
        self,
        config,
        img_size,
        in_chans,
        embed_dim,
        uper_head_dim,
        depths,
        num_heads,
        mlp_ratio,
        num_classes,
        nbts,
        pool_scales,
        spa_temp_att,
        conv_spa_att,
        decoder_channels,
        window_size,
        d_model,
        dropout_ratio=0.1,
    ):
        super().__init__()
        self.backbone_dims = [embed_dim * 2**i for i in range(len(depths))]
        self.img_size = img_size
        self.num_classes = num_classes
        self.nbts = nbts
        self.d_model = d_model
        self.pool_scales = pool_scales
        self.spa_temp_att = spa_temp_att
        self.conv_spa_att = conv_spa_att
        self.decoder_channels = decoder_channels
        self.config = config
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.dropout_ratio = dropout_ratio
        # self.partition_size = partition_size
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.sits_encoder = TConvFormer(
            input_size=(self.img_size, self.img_size),
            stem_channels=64,
            block_channels=config["models"]["t_convformer"]["block_channels"],  # [64, 128, 256, 512]
            block_layers=config["models"]["t_convformer"]["block_layers"],  # [2, 2, 5, 2]
            head_dim=32,
            stochastic_depth_prob=0.2,
            partition_size=config["models"]["maxvit"]["window_cond_size"],
        )

        self.encoder_channels = [
            config["models"]["t_convformer"]["embed_dim"],
            config["models"]["t_convformer"]["embed_dim"] * 2,
            config["models"]["t_convformer"]["embed_dim"] * 4,
            config["models"]["t_convformer"]["embed_dim"] * 8,
        ]

        self.decoder_head = UPerHead(
            self.backbone_dims, # [64, 128, 256]
            uper_head_dim,  # 512
            num_classes,
            pool_scales,
            dropout_ratio=0.1,
        )

    def forward(self, x, batch_positions=None):
        # print("Swin Segmentation inputs:", x.shape)
        # x_enc = self.backbone(x, batch_positions)
        h, w = x.size()[-2:]
        red_temp_feats, enc_temp_feats = self.sits_encoder(x, batch_positions)
        
        # print("backbone_dims:", self.backbone_dims)

        # print("red_temp_feats 0:", red_temp_feats[0].shape)
        # print("red_temp_feats 1:", red_temp_feats[1].shape)
        # print("red_temp_feats 2:", red_temp_feats[2].shape)
        # print("red_temp_feats 3:", red_temp_feats[3].shape)

        # red_temp_feats 0: torch.Size([2, 64, 64, 64])
        # red_temp_feats 1: torch.Size([2, 128, 32, 32])
        # red_temp_feats 2: torch.Size([2, 256, 16, 16])
        # red_temp_feats 3: torch.Size([2, 512, 8, 8])

        # sits_logits, cls_sits_feats, multi_lvls_cls = self.decode_head(
        #     res0, res1, res2, res3, h, w
        # )
        sits_logit, multi_lvls_cls = self.decoder_head(red_temp_feats)

        # print("enc_features:", enc_temp_feats[0].shape)
        # print("enc_features:", enc_temp_feats[1].shape)
        # print("enc_features:", enc_temp_feats[2].shape)
        # print("enc_features:", enc_temp_feats[3].shape)
        # print()
        # print("multi_lvls_cls:", multi_lvls_cls[0].shape)
        # print("multi_lvls_cls:", multi_lvls_cls[1].shape)
        # print("multi_lvls_cls:", multi_lvls_cls[2].shape)
        # print("multi_lvls_cls:", multi_lvls_cls[3].shape)
        # print()
        # print("sits_logit:", sits_logit.shape)

        # enc_temp_feats: torch.Size([2, 12, 64, 64, 64])
        # enc_temp_feats: torch.Size([2, 12, 128, 32, 32])
        # enc_temp_feats: torch.Size([2, 12, 256, 16, 16])
        # enc_temp_feats: torch.Size([2, 12, 512, 8, 8])

        # sits_logits: torch.Size([2, 19, 10, 10])
        # cls_sits_feats:
        # sits_logit: sits logits, multi_lvls_cls: multi-res features, 
        # red_temp_feats: reduced sits features, 
        # enc_temp_feats: temporal sits features
        return sits_logit, multi_lvls_cls, red_temp_feats, enc_temp_feats
