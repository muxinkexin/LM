from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class LiteMonoOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Lite-Mono options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "/home/zy/KITTI"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="/home/tzq/tzq/3/checkpoint/LM_runs")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="E0_pretrained_s0")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--model",
                                 type=str,
                                 help="which model to load",
                                 choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"],
                                 default="lite-mono-tiny")
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay in AdamW",
                                 default=1e-2)
        self.parser.add_argument("--drop_path",
                                 type=float,
                                 help="drop path rate",
                                 default=0.2)
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        self.parser.add_argument("--profile",
                                 action=argparse.BooleanOptionalAction,
                                 help="profile a single evaluation batch with THOP",
                                 default=False)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        self.parser.add_argument("--lr",
                                 nargs="+",
                                 type=float,
                                 help="learning rates of DepthNet and PoseNet. "
                                      "Initial learning rate, "
                                      "minimum learning rate, "
                                      "First cycle step size.",
                                 default=[0.0001, 5e-6, 31, 0.0001, 1e-5, 31])
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--mypretrain",
                                 type=str,
                                 help="if set, use my pretrained encoder")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        # RA-Depth training mechanisms; no inference-time modules.
        self.parser.add_argument("--ra_depth", action="store_true")
        self.parser.add_argument("--ra_views", choices=["both", "high", "low"], default="both")
        self.parser.add_argument("--ra_high_range", nargs=2, type=float, default=[1.1, 2.0])
        self.parser.add_argument("--ra_low_range", nargs=2, type=float, default=[0.7, 0.9])
        self.parser.add_argument("--ra_consistency_weight", type=float, default=0.1)
        self.parser.add_argument("--ra_photo_weight", type=float, default=1.0)
        self.parser.add_argument("--ra_warmup_epochs", type=int, default=5)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--max_train_steps", type=int, default=0,
                                 help="0: full training; positive: smoke run with checkpoint")
        self.parser.add_argument("--self_test", action="store_true",
                                 help="synthetic mechanism/gradient tests; no KITTI required")
        self.parser.add_argument("--check_data", action="store_true",
                                 help="check all split image paths, then exit")
        self.parser.add_argument("--eval_height", type=int, default=None)
        self.parser.add_argument("--eval_width", type=int, default=None)
        self.parser.add_argument("--eval_batch_size", type=int, default=4)
        self.parser.add_argument("--eval_files", type=str, default=None,
                                 help="image list for full depth evaluation")
        self.parser.add_argument("--eval_gt_path", type=str, default=None,
                                 help="matching GT NPZ/cache for full depth evaluation")
        self.parser.add_argument("--eval_role", choices=["val", "test"], default="test")
        self.parser.add_argument("--eval_crop", choices=["garg", "none"], default="garg")
        self.parser.add_argument("--metrics_json", type=str, default=None)
        self.parser.add_argument("--compare_metrics", nargs=2, metavar=("BASE_DIR", "RA_DIR"),
                                 help="compare paired per-resolution evaluation JSON files")
        self.parser.add_argument("--allow_tf32", action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        o = self.options
        if o.height < 64 or o.width < 64 or o.height % 32 or o.width % 32:
            self.parser.error("training height/width must be >=64 and multiples of 32")
        if (o.eval_height is None) != (o.eval_width is None):
            self.parser.error("set both --eval_height and --eval_width")
        if o.eval_height is not None and min(o.eval_height, o.eval_width) < 64:
            self.parser.error("evaluation dimensions must be >=64")
        if o.eval_role == "val" and not o.eval_gt_path:
            self.parser.error("--eval_role val requires --eval_gt_path")
        if len(o.lr) != 6 or min(o.lr[2], o.lr[5]) < 1:
            self.parser.error("--lr requires six numbers; cycle lengths must be positive")
        if not o.scales or 0 not in o.scales or any(v not in [0, 1, 2] for v in o.scales):
            self.parser.error("--scales must contain 0 and be a subset of 0 1 2")
        if min(o.batch_size, o.eval_batch_size, o.num_epochs, o.log_frequency, o.save_frequency) < 1:
            self.parser.error("batch sizes, epochs and logging/save intervals must be positive")
        if o.num_workers < 0 or o.max_train_steps < 0:
            self.parser.error("workers and max_train_steps must be nonnegative")
        if o.load_weights_folder and o.mypretrain:
            self.parser.error("choose load_weights_folder fine-tuning OR mypretrain initialization")
        if len(o.scales) != len(set(o.scales)):
            self.parser.error("duplicate scales are not supported")
        if o.ra_depth:
            if o.scales != [0] or o.v1_multiscale or o.predictive_mask:
                self.parser.error("RA migration requires --scales 0, without v1_multiscale/predictive_mask")
            if o.use_stereo or o.pose_model_type != "separate_resnet" or o.pose_model_input != "pairs":
                self.parser.error("RA migration supports monocular separate_resnet/pairs only")
            if len(o.frame_ids) < 2 or o.frame_ids[0] != 0:
                self.parser.error("RA requires target frame 0 and temporal source frames")
        if not 1 <= o.ra_high_range[0] <= o.ra_high_range[1] <= 3:
            self.parser.error("high range must satisfy 1 <= min <= max <= 3")
        if not 0.5 <= o.ra_low_range[0] <= o.ra_low_range[1] <= 1:
            self.parser.error("low range must satisfy .5 <= min <= max <= 1")
        if min(o.ra_photo_weight, o.ra_consistency_weight, o.ra_warmup_epochs) < 0:
            self.parser.error("RA weights and warmup must be nonnegative")
        return self.options
