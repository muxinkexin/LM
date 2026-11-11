from __future__ import absolute_import, division, print_function


import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *
from checkpoint_utils import load_pretraining_checkpoint

import datasets
import networks
import random
import numpy as np
import os


# torch.backends.cudnn.benchmark = True


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.profile = self.opt.profile

        self.num_scales = len(self.opt.scales)
        self.frame_ids = len(self.opt.frame_ids)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                     self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models_pose["pose_encoder"].to(self.device)
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        # The repository did not include its imported third-party scheduler.
        # Native cosine schedule, common to baseline and RA for a controlled comparison.
        self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, T_max=int(self.opt.lr[2]), eta_min=self.opt.lr[1])
        if self.use_pose_net:
            self.model_pose_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.model_pose_optimizer, T_max=int(self.opt.lr[5]), eta_min=self.opt.lr[4])

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mypretrain is not None:
            self.load_pretrain()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext,
            ra_options=self.opt if self.opt.ra_depth else None)
        train_dataset.load_depth = False  # Self-supervision needs RGB only.
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=not self.opt.no_cuda, drop_last=True,
            generator=torch.Generator().manual_seed(self.opt.seed))
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        val_dataset.load_depth = False  # Full metrics use evaluate_depth.py and bundled GT.
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=not self.opt.no_cuda, drop_last=True,
            generator=torch.Generator().manual_seed(self.opt.seed + 1))
        self.val_iter = iter(self.val_loader)

        if len(self.train_loader) == 0 or len(self.val_loader) == 0:
            raise ValueError("batch_size exceeds available train/validation samples")
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in list(self.models.values()) + list(self.models_pose.values()):
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in list(self.models.values()) + list(self.models_pose.values()):
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            self.model_lr_scheduler.step()
            if self.use_pose_net:
                self.model_pose_lr_scheduler.step()
            if self.opt.max_train_steps and self.step >= self.opt.max_train_steps:
                self.save_model()
                break
        for writer in self.writers.values():
            writer.close()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("non-finite total loss")
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            if self.opt.max_train_steps and self.step >= self.opt.max_train_steps:
                break

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder

            features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        if self.opt.ra_depth and self.models["encoder"].training:
            self.add_ra_losses(inputs, outputs, losses)
        return outputs, losses

    def add_ra_losses(self, inputs, outputs, losses):
        branches = ["high", "low"] if self.opt.ra_views == "both" else [self.opt.ra_views]
        photo, consistency = [], []
        mid_mask = outputs.get("identity_selection/0")
        if mid_mask is not None:
            mid_mask = mid_mask.unsqueeze(1)
        for branch in branches:
            aux = self.models["depth"](self.models["encoder"](inputs[("ra_aug", branch, 0)]))
            disp = aux[("disp", 0)]
            lp = self.ra_photometric(inputs, outputs, disp, branch)
            lc = ra_consistency_loss(outputs[("disp", 0)], disp,
                                     inputs[("ra_meta", branch)], branch,
                                     None if self.opt.no_ssim else self.ssim, mid_mask)
            losses["ra/photo_" + branch] = lp
            losses["ra/consistency_" + branch] = lc
            photo.append(lp)
            consistency.append(lc)
        ramp = 1.0 if self.opt.ra_warmup_epochs == 0 else min(
            1.0, (self.epoch + 1) / self.opt.ra_warmup_epochs)
        losses["ra/weight"] = outputs[("disp", 0)].new_tensor(
            self.opt.ra_consistency_weight * ramp)
        losses["loss_base"] = losses["loss"]
        losses["loss"] = (losses["loss"] + self.opt.ra_photo_weight * sum(photo) / len(photo)
                          + losses["ra/weight"] * sum(consistency) / len(consistency))

    def ra_photometric(self, inputs, outputs, disp, branch):
        """Share physical poses across views; transform K and mask stitched context.
        The low-view duplicated region supplies context only, never geometry supervision.
        """
        h, w = disp.shape[-2:]
        meta = inputs[("ra_meta", branch)]
        K = ra_intrinsics(inputs[("K", 0)], meta, h, w)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        points = self.backproject_depth[0](depth, torch.linalg.inv(K))
        target = inputs[("ra_color", branch, 0)]
        yy, xx = torch.meshgrid(torch.arange(h, device=disp.device),
                                torch.arange(w, device=disp.device), indexing="ij")
        rh = meta[:, 0, None, None] if branch == "low" else h
        rw = meta[:, 1, None, None] if branch == "low" else w
        target_valid = ((xx[None] >= 1) & (yy[None] >= 1)
                        & (xx[None] < rw - 1) & (yy[None] < rh - 1))
        reprojections, identities, valid_sources = [], [], []
        for frame_id in self.opt.frame_ids[1:]:
            T = outputs[("cam_T_cam", 0, frame_id)]
            grid = self.project_3d[0](points, K, T)
            px = (grid[..., 0] + 1) * (w - 1) / 2
            py = (grid[..., 1] + 1) * (h - 1) / 2
            front = (T @ points)[:, 2].reshape(-1, h, w) > 1e-7
            valid = (target_valid & front & (px >= 1) & (py >= 1)
                     & (px < rw - 1) & (py < rh - 1)).unsqueeze(1)
            # Require valid 3x3 support for the photometric SSIM window.
            valid = (1 - F.max_pool2d(1 - valid.float(), 3, 1, 1)) > 0.5
            source = inputs[("ra_color", branch, frame_id)]
            warped = F.grid_sample(source, grid, padding_mode="border", align_corners=True)
            reprojections.append(self.compute_reprojection_loss(warped, target).masked_fill(~valid, 1e3))
            identities.append(self.compute_reprojection_loss(source, target))
            valid_sources.append(valid)
        reproj = torch.cat(reprojections, 1)
        valid = torch.cat(valid_sources, 1)
        any_valid = valid.any(1, keepdim=True)
        if self.opt.avg_reprojection:
            reproj = (reproj.masked_fill(~valid, 0).sum(1, keepdim=True)
                      / valid.sum(1, keepdim=True).clamp_min(1))
        best = reproj.min(1, keepdim=True).values
        if not self.opt.disable_automasking:
            identity = torch.cat(identities, 1)
            if self.opt.avg_reprojection:
                identity = identity.mean(1, keepdim=True)
            best_identity = (identity + torch.randn_like(identity) * 1e-5).min(1, keepdim=True).values
            best = torch.minimum(best, best_identity)
        photo = best.masked_fill(~any_valid, 0).sum() / any_valid.sum().clamp_min(1)
        smooth = disp.new_zeros(())
        meta_rows = meta.detach().cpu().tolist()
        for i in range(disp.shape[0]):
            ih, iw = meta_rows[i][:2] if branch == "low" else [h, w]
            d = disp[i:i+1, :, :ih, :iw]
            smooth = smooth + get_smooth_loss(d / (d.mean() + 1e-7), target[i:i+1, :, :ih, :iw])
        return photo + self.opt.disparity_smoothness * smooth / disp.shape[0]

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models_pose["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones_like(mask))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'] if self.use_pose_net else 0.,
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
                to_save['model'] = self.opt.model
                to_save['scales'] = self.opt.scales
                to_save['ra_depth'] = self.opt.ra_depth
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = load_pretraining_checkpoint(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        if not pretrained_dict:
            raise ValueError("no matching encoder keys in ImageNet checkpoint")
        if any(v.shape != model_dict[k].shape for k, v in pretrained_dict.items()):
            raise ValueError("ImageNet checkpoint architecture does not match --model")
        print("ImageNet encoder tensors matched: {} / {}".format(len(pretrained_dict), len(model_dict)))
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            model = self.models_pose[n] if n in self.models_pose else self.models[n]
            model_dict = model.state_dict()
            pretrained_dict = torch.load(os.path.join(self.opt.load_weights_folder, n + '.pth'),
                                         map_location='cpu', weights_only=True)
            model.load_state_dict({k: v for k, v in pretrained_dict.items() if k in model_dict}, strict=True)

        print("Loaded model weights; optimizer, epoch and scheduler start fresh (fine-tuning).")


def check_data(opt):
    """Check all required RGB paths, without starting training or loading weights."""
    from pathlib import Path
    root = Path(opt.data_path)
    ext = '.png' if opt.png else '.jpg'
    missing, checked = [], 0
    for split, name, frames in [(opt.split, 'train', opt.frame_ids),
                                 (opt.split, 'val', opt.frame_ids),
                                 ('eigen', 'test', [0])]:
        lines = readlines(str(Path(__file__).parent / 'splits' / split / (name + '_files.txt')))
        for line in lines:
            folder, frame, side = line.split()
            for offset in frames:
                camera = {'l': 2, 'r': 3, '2': 2, '3': 3}[side]
                path = root / folder / ('image_0{}/data'.format(camera)) / ('{:010d}{}'.format(int(frame)+offset, ext))
                checked += 1
                if not path.is_file():
                    if len(missing) < 20:
                        missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing RGB files (first 20):\n" + '\n'.join(missing))
    gt_file = Path(__file__).parent / 'splits/eigen/gt_depths.npz'
    with np.load(gt_file, allow_pickle=True, encoding='latin1') as data:
        count = len(data['data'])
    expected = len(readlines(str(Path(__file__).parent / 'splits/eigen/test_files.txt')))
    if count != expected:
        raise ValueError("Bundled GT count does not match test split")
    print("PASS data paths: {} references; bundled GT: {} images".format(checked, count))


def self_test(opt):
    """Synthetic geometry, gradients, on/off isolation and variable-resolution tests.
    No KITTI or pretrained downloads; CPU or CUDA selected by --no_cuda.
    """
    from PIL import Image
    from types import SimpleNamespace
    from datasets.mono_dataset import MonoDataset
    from torch.utils.data._utils.collate import default_collate
    import copy
    torch.set_num_threads(2)
    device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --no_cuda for CPU tests')
    h, w, batch = 64, 128, 2
    test_opt = copy.deepcopy(opt)
    test_opt.height, test_opt.width, test_opt.batch_size = h, w, batch
    test_opt.scales, test_opt.frame_ids = [0], [0, -1, 1]
    test_opt.pose_model_type, test_opt.pose_model_input = 'separate_resnet', 'pairs'
    test_opt.predictive_mask = test_opt.v1_multiscale = test_opt.use_stereo = False
    test_opt.no_ssim = test_opt.disable_automasking = test_opt.avg_reprojection = False
    test_opt.ra_views, test_opt.ra_depth = 'both', True
    test_opt.ra_high_range, test_opt.ra_low_range = [1.1, 2.0], [0.7, 0.9]
    test_opt.ra_consistency_weight, test_opt.ra_photo_weight, test_opt.ra_warmup_epochs = .1, 1., 5

    class Synthetic(MonoDataset):
        def __init__(self, ra_options):
            super().__init__('', ['synthetic 1 l'] * batch, h, w, [0, -1, 1], 4,
                             is_train=True, ra_options=ra_options)
            self.K = np.array([[.58, 0, .5, 0], [0, 1.92, .5, 0],
                               [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        def check_depth(self): return False
        def get_color(self, folder, frame_index, side, do_flip):
            rng = np.random.default_rng(7)
            a = rng.integers(0, 256, (h*2, w*2, 3), dtype=np.uint8)
            a = np.roll(a, frame_index - 1, axis=1)
            im = Image.fromarray(a)
            return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if do_flip else im

    data = Synthetic(test_opt)
    inputs = default_collate([data[i] for i in range(batch)])
    for branch in ['high', 'low']:
        assert inputs[('ra_color', branch, 0)].shape == (batch, 3, h, w)
    base_data = Synthetic(None)
    assert not any(isinstance(k, tuple) and k[0].startswith('ra_') for k in base_data[0])
    print('PASS dataset: native resize/crop/stitch, fixed canvas; off has no RA tensors')

    mid = torch.rand(batch, 1, h, w, device=device, requires_grad=True)
    for branch in ['high', 'low']:
        meta = inputs[('ra_meta', branch)].to(device)
        view = torch.zeros_like(mid)
        for i in range(batch):
            rh, rw, dy, dx = meta[i].tolist()
            resized = F.interpolate(mid[i:i+1], (rh, rw), mode='bilinear', align_corners=False)
            if branch == 'high': view[i:i+1] = resized[:, :, dy:dy+h, dx:dx+w]
            else: view[i:i+1, :, :rh, :rw] = resized
        assert ra_consistency_loss(mid, view, meta, branch).abs() < 1e-7
        # Perturbing stitched context must not enter the consistency supervision.
        if branch == 'low':
            for i in range(batch):
                rh, rw = meta[i, :2].tolist()
                view[i, :, rh:, :] = 100
                view[i, :, :, rw:] = 100
            assert ra_consistency_loss(mid, view, meta, branch).abs() < 1e-7
        view = (view.detach() + .01).requires_grad_()
        lc = ra_consistency_loss(mid, view, meta, branch, SSIM().to(device))
        grads = torch.autograd.grad(lc, (mid, view), retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
        assert ra_consistency_loss(mid, view, meta, branch, mid_mask=torch.zeros_like(mid)) == 0
    print('PASS consistency: exact crop alignment, seam exclusion, zero mask, bilateral gradients')

    K = inputs[('K', 0)].to(device)
    meta = inputs[('ra_meta', 'high')].to(device)
    Ka = ra_intrinsics(K, meta, h, w)
    points = torch.tensor([.2, .1, 1, 1.], device=device).repeat(batch, 1)
    px = (K @ points.unsqueeze(-1)).squeeze(-1)
    actual = (Ka @ points.unsqueeze(-1)).squeeze(-1)
    sx, sy = meta[:, 1] / w, meta[:, 0] / h
    torch.testing.assert_close(actual[:, 0], (px[:, 0] + .5) * sx - .5 - meta[:, 3])
    torch.testing.assert_close(actual[:, 1], (px[:, 1] + .5) * sy - .5 - meta[:, 2])
    torch.testing.assert_close(Ka @ torch.linalg.inv(Ka), torch.eye(4, device=device).expand(batch, -1, -1), atol=1e-5, rtol=1e-5)
    print('PASS intrinsics: pixel-center resize/crop and inverse projection')

    t = Trainer.__new__(Trainer)
    t.opt, t.device, t.epoch = test_opt, device, 0
    t.num_scales, t.num_pose_frames, t.use_pose_net = 1, 2, True
    t.models = {'encoder': networks.LiteMono(model=opt.model, height=h, width=w).to(device)}
    t.models['depth'] = networks.DepthDecoder(t.models['encoder'].num_ch_enc, [0]).to(device)
    t.models_pose = {'pose_encoder': networks.ResnetEncoder(18, False, 2).to(device)}
    t.models_pose['pose'] = networks.PoseDecoder(t.models_pose['pose_encoder'].num_ch_enc, 1, 2).to(device)
    t.ssim = SSIM().to(device)
    t.backproject_depth = {0: BackprojectDepth(batch, h, w).to(device)}
    t.project_3d = {0: Project3D(batch, h, w).to(device)}
    calls = []
    hook = t.models['encoder'].register_forward_hook(lambda *args: calls.append(1))
    param_count = sum(p.numel() for m in t.models.values() for p in m.parameters())
    for enabled, training, expected in [(False, True, 1), (True, True, 3), (True, False, 1)]:
        t.opt.ra_depth = enabled
        t.set_train() if training else t.set_eval()
        assert all(m.training == training for m in t.models_pose.values())
        calls.clear()
        for m in list(t.models.values()) + list(t.models_pose.values()): m.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out, losses = t.process_batch(dict(inputs))
            assert torch.isfinite(losses['loss'])
            assert ('ra/consistency_high' in losses) == (enabled and training)
            if training:
                losses['loss'].backward()
                for name, model in list(t.models.items()) + list(t.models_pose.items()):
                    gs = [p.grad for p in model.parameters() if p.grad is not None]
                    assert gs and all(torch.isfinite(g).all() for g in gs), name
                    assert sum(float(g.abs().sum()) for g in gs) > 0, name
        assert len(calls) == expected, calls
        assert param_count == sum(p.numel() for m in t.models.values() for p in m.parameters())
    hook.remove()
    print('PASS process_batch: off=1 / on=3 / validation=1 depth forward; finite encoder/depth/pose gradients')
    t.set_eval()
    with torch.no_grad():
        for ih, iw in [(128, 416), (192, 640), (320, 1024), (375, 1242)]:
            x = torch.rand(1, 3, ih, iw, device=device)
            x = F.pad(x, (0, (-iw) % 32, 0, (-ih) % 32), mode='replicate')
            y = t.models['depth'](t.models['encoder'](x))[('disp', 0)][:, :, :ih, :iw]
            assert y.shape == (1, 1, ih, iw) and torch.isfinite(y).all()
            print('PASS variable resolution: {}x{}'.format(ih, iw))
    print('PASS all synthetic checks; depth inference parameters:', param_count)
    print('These checks do not establish KITTI accuracy or RTX 5090 performance.')
