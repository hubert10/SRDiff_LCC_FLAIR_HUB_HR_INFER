import os.path
import json
import torch
import time
import pathlib
from typing import Union
import requests
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Literal
from trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt
from models.denoiser.unet import Unet
from models.sits_aerial_seg_model import SITSAerialSegmenter
from losses.focal_smooth import FocalLossWithSmoothing
from models.diffusion.latent_diffusion import LatentDiffusion
import torch.utils.checkpoint as checkpoint
from models.encoders.t_convformer import TConvFormer


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams["hidden_size"]
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        self.criterion_aer = FocalLossWithSmoothing(
            hparams["inputs"]["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )
        self.criterion_sat = FocalLossWithSmoothing(
            hparams["inputs"]["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )

        self.loss_aux_sat_weight = hparams["hyperparams"]["loss_aux_sat_weight"]
        self.loss_main_sat_weight = hparams["hyperparams"]["loss_main_sat_weight"]

        # 1. Denoising Net

        self.denoise_net = Unet(
            hidden_size,
            out_dim=hparams["inputs"]["num_channels_sat"],
            cond_dim=hparams["rrdb_num_feat"],
            dim_mults=dim_mults,
        )

        first_stage_config = {
            "embed_dim": 4,
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 4,
            "out_ch": 4,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
        cond_stage_config = {
            "image_size": 64,
            "in_channels": 8,
            "model_channels": 160,
            "out_channels": 4,
            "num_res_blocks": 2,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 2, 2, 4],
            "num_head_channels": 32,
        }

        # 2. Cond Encoder with ONLY two encoding block layers
        self.cond_net = TConvFormer(
            input_size=(
                hparams["inputs"]["lr_patch_size"],
                hparams["inputs"]["lr_patch_size"],
            ),
            stem_channels=64,
            partition_size=hparams["models"]["maxvit"]["window_cond_size"],
            block_channels=hparams["models"]["t_convformer"]["block_channels"],  # [128, 256, 512],  # [64, 128, 256, 512]
            block_layers=hparams["models"]["t_convformer"]["block_layers"],  # [2, 2, 5],  # [2, 2, 5, 2]
            head_dim=32,
            stochastic_depth_prob=0.2,
        )

        # The idea here is that if you load pretrained weights for
        # inference, they will be overwitten by the full trained model

        # Load pretrained weights (if provided)
        if (
            hparams["cond_net_ckpt"]
            and os.path.exists(hparams["cond_net_ckpt"])
            and not hparams["infer"]
        ):
            self.load_pretrained_weights(self.cond_net, hparams["cond_net_ckpt"])


        # 3. Latent Diff Model

        latent_diff = LatentDiffusion(
            denoise_net=self.denoise_net,
            cond_net=self.cond_net,
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            timesteps=hparams["timesteps"],
        )

        # 4. CLS Net
        self.model = SITSAerialSegmenter(latent_diff=latent_diff, config=hparams)

        if hparams["infer"]:
            if hparams["diff_net_ckpt"] != "" and os.path.exists(
                hparams["diff_net_ckpt"]
            ):
                load_ckpt(self.model, hparams["diff_net_ckpt"])

        # what is used for?
        self.global_step = 0
        return self.model

    # =========================
    # Utility Functions
    # =========================

    def _strip_prefix(self, state_dict, prefix="model.backbone."):
        """Remove prefix from checkpoint keys."""
        return {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }

    def _load_checkpoint(self, path):
        """Load checkpoint and extract state_dict."""
        ckpt = torch.load(path, map_location="cpu")
        return ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    def load_pretrained_weights(self, model, weights_path):
        print(f"\nLoading pretrained weights from: {weights_path}")

        pretrained_dict = self._load_checkpoint(weights_path)
        pretrained_dict = self._strip_prefix(pretrained_dict)

        model_dict = model.state_dict()

        # 🔍 Analyze BEFORE loading
        self.analyze_weight_loading(model_dict, pretrained_dict)

        # ✅ Keep only safe weights
        safe_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(safe_dict)
        model.load_state_dict(model_dict)

        print(
            f"Loaded {len(safe_dict)}/{len(model_dict)} "
            f"({len(safe_dict)/len(model_dict):.2%}) parameters"
        )

    def analyze_weight_loading(self, model_dict, pretrained_dict):
        matched = []
        shape_mismatch = []
        missing = []
        unexpected = []

        # Check pretrained → model

        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    matched.append(k)
                else:
                    shape_mismatch.append((k, v.shape, model_dict[k].shape))
            else:
                unexpected.append(k)

        # Check model → pretrained
        for k in model_dict.keys():
            if k not in pretrained_dict:
                missing.append(k)

        # ===== PRINT REPORT =====
        print("\n========== WEIGHT LOADING REPORT FOR COND-Net ==========")
        print(f"Matched: {len(matched)}")
        print(f"Shape mismatch: {len(shape_mismatch)}")
        print(f"Missing: {len(missing)}")
        print(f"Unexpected: {len(unexpected)}")

        # ---- Shape mismatch details ----
        if shape_mismatch:
            print("\n--- Shape Mismatches ---")
            for k, s1, s2 in shape_mismatch[:20]:
                print(f"{k}")
                print(f"  checkpoint: {s1}")
                print(f"  model     : {s2}")

        # ---- Missing keys ----
        if missing:
            print("\n--- Missing Keys (not found in checkpoint) ---")
            for k in missing[:20]:
                print(k)

        # ---- Unexpected keys ----
        if unexpected:
            print("\n--- Unexpected Keys (not used) ---")
            for k in unexpected[:20]:
                print(k)

        print("===========================================\n")
        return matched, shape_mismatch, missing, unexpected

    def build_optimizer(self, model):
        params = list(model.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=hparams["lr"])
        return optimizer

    def build_scheduler(self, optimizer):
        # 1. Scheduler type
        # It uses torch.optim.lr_scheduler.MultiStepLR, which reduces the learning
        # rate (LR) by a factor (gamma) at specific training steps (called milestones)

        # 2. Milestones
        # This means the LR will drop twice:
        # First at 50% of decay_steps
        # Then at 90% of decay_steps
        # Example: if decay_steps = 100000, milestones = [50000, 90000].

        # 3. Gamma factor
        # At each milestone, the LR is multiplied by 0.1 (reduced by 10×).
        # Example: if LR starts at 0.001,
        # at step 50k → LR becomes 0.0001
        # at step 90k → LR becomes 0.00001.

        # 4. Effect
        # This creates a piecewise-constant decay schedule:
        # LR stays constant in between milestones.
        # At the milestone steps, LR suddenly drops.

        scheduler_param = {
            "milestones": [
                np.floor(hparams["decay_steps"] * 0.5),
                np.floor(hparams["decay_steps"] * 0.9),
            ],
            "gamma": 0.1,
        }
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)

    def training_step(self, batch):
        img = batch["img"]  # torch.Size([4, 5, 512, 512]): Aerial image
        img_hr = batch["img_hr"]  # torch.Size([4, 5, 64, 64]): SPOT6 image
        img_lr = batch["img_lr"]  # torch.Size([4, 12, 3, 16, 16]): SENT-2 images
        labels = batch["labels"]  # torch.Size([4, 19, 512, 512])
        labels_sr = batch["labels_sr"]  # torch.Size([4, 19, 64, 64])
        dates = batch["dates_encoding"]
        closest_idx = batch["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        # Call latent diffusion model for SR-prediction this should also
        # return the SR-SITS images alongside the diffusion losses
        losses, img_sr = self.model.latent_diff(
            sc_img_hr,  # HR image
            img_lr,  # LR image for conditioning
            dates=dates,
            closest_idx=closest_idx,
        )
        # print("---------------------img_sr training_step------------------------:", img_sr.shape)

        # for classification branches
        cls_sits, multi_outputs, aer_outputs = self.model(
            img, img_sr, labels, dates, hparams
        )

        labels_sr = torch.argmax(labels_sr, dim=1) if labels_sr.ndim == 4 else labels_sr
        labels = torch.argmax(labels, dim=1) if labels.ndim == 4 else labels

        # Auxiliary losses
        # The CE loss for the SITS classification branch is done at 1m GSD
        # print("multi_outputs 2:", multi_outputs[2].shape)
        # print("multi_outputs 1:", multi_outputs[1].shape)
        # print("multi_outputs 0:", multi_outputs[0].shape)
        # print("labels_sr:", labels_sr.shape)

        aux_loss1 = self.criterion_sat(multi_outputs[2], labels_sr)
        aux_loss2 = self.criterion_sat(multi_outputs[1], labels_sr)
        aux_loss3 = self.criterion_sat(multi_outputs[0], labels_sr)

        # print("cls_sits:", cls_sits.shape)
        # print("labels_sr:", labels_sr.shape)

        # loss for main SITS classification branch
        loss_main_sat = self.criterion_sat(cls_sits, labels_sr)

        # Total loss for SITS branch
        loss_sat = self.loss_main_sat_weight * loss_main_sat + (
            self.loss_aux_sat_weight * aux_loss1
            + self.loss_aux_sat_weight * aux_loss2
            + self.loss_aux_sat_weight * aux_loss3
        )

        # print("img:", img.shape)
        # print("img_hr:", img_hr.shape)
        # print("img_lr:", img_lr.shape)
        # print("img_lr_up:", img_lr_up.shape)
        # print("img_sr:", img_sr.shape)
        # print("labels_sr:", labels_sr.shape)

        # img: torch.Size([2, 5, 512, 512])
        # img_hr: torch.Size([2, 5, 64, 64])
        # img_lr: torch.Size([2, 2, 4, 16, 16])
        # img_lr_up: torch.Size([2, 2, 4, 64, 64])
        # img_sr: torch.Size([2, 2, 4, 64, 64])
        # labels_sr: torch.Size([2, 64, 64])

        # labels: torch.Size([2, 512, 512])
        # aer_outputs: torch.Size([2, 13, 512, 512])

        # Loss for AER branch
        loss_aer = self.criterion_aer(aer_outputs, labels.long())

        # The CE loss for the SITS classification branch is done at 1.6m GSD
        # that combines the loss from the SR-diffusion model and the SITS
        #  segmentation branch

        losses["sr"] = hparams["hyperparams"]["loss_weights_aer_sat"][1] * (
            losses["sr"] + loss_sat
        )

        # The CE loss for the AER classification branch is done at 20cm GSD
        losses["aer"] = hparams["hyperparams"]["loss_weights_aer_sat"][0] * loss_aer
        total_loss = sum(losses.values())
        return losses, total_loss

    def sample_and_test(self, sample):
        # Sample images and calculate evaluation metrics
        # Used for inference mode
        ret = {k: [] for k in self.metric_keys}
        ret["n_samples"] = 0
        img = sample["img"]
        img_hr = sample["img_hr"]
        img_lr = sample["img_lr"]
        labels = sample["labels"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        torch.cuda.synchronize()
        start_time = time.time()

        # Sampling or inference or sr generation
        img_sr, final_loss = self.model.latent_diff.sample(
            img,
            sc_img_hr,
            img_lr,
            dates=dates,
            closest_idx=closest_idx,
            custom_steps=hparams[
                "ddim_infer_steps"
            ],  # HR imahe  # LR image for conditioning
        )

        torch.cuda.synchronize()
        end_time = time.time()

        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.4f} seconds")

        # during inference, only the aer branch is used
        _, _, aer_outputs = self.model(img, img_sr, labels, dates, hparams)

        proba = torch.softmax(aer_outputs, dim=1)
        preds = torch.argmax(proba, dim=1)
        labels = torch.argmax(labels, dim=1)

        # print("preds:", preds.shape)
        # print("labels:", labels.shape)

        # preds: torch.Size([1, 512, 512])
        # labels: torch.Size([1, 512, 512])

        # Loop over batch
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b][int(closest_idx[b].item()), :, :, :],  # SR image at t
                sc_img_hr[b],  # reference HR image
                img_lr[b][int(closest_idx[b].item()), :, :, :],  # LR input at t
                preds[b],
                labels[b],
            )
            ret["psnr"].append(s["psnr"])
            ret["ssim"].append(s["ssim"])
            ret["lpips"].append(s["lpips"])
            ret["mae"].append(s["mae"])
            ret["mse"].append(s["mse"])
            ret["shift_mae"].append(s["shift_mae"])
            ret["miou"].append(s["miou"])

            ret["n_samples"] += 1
        return img_sr, preds, ret, final_loss
        # return img_sr, preds, rrdb_out, ret, ret


# ---------------Conditioning Temp Feats-------------------
# cond 0: torch.Size([1, 4, 64, 16, 16])
# cond 1: torch.Size([1, 4, 128, 8, 8])
# cond 2: torch.Size([1, 4, 256, 4, 4])
