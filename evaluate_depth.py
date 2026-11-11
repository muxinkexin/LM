from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets
import networks
import time
import json
import hashlib
import torch.nn.functional as F
from thop import clever_format
from thop import profile


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    torch.backends.cuda.matmul.allow_tf32 = opt.allow_tf32
    torch.backends.cudnn.allow_tf32 = opt.allow_tf32
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    if opt.ext_disp_to_eval is None:

        if not opt.load_weights_folder:
            raise ValueError("--load_weights_folder is required")
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location='cpu', weights_only=True)
        decoder_dict = torch.load(decoder_path, map_location='cpu', weights_only=True)

        feed_h = opt.eval_height or encoder_dict['height']
        feed_w = opt.eval_width or encoder_dict['width']
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, feed_h, feed_w,
                                           [0], 4, is_train=False,
                                           img_ext='.png' if opt.png else '.jpg')
        # Metrics use the bundled GT; do not recompute optional online LiDAR depth.
        dataset.load_depth = False
        dataloader = DataLoader(dataset, opt.eval_batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=not opt.no_cuda,
                                drop_last=False)
        device = torch.device('cpu' if opt.no_cuda else 'cuda')
        # Dilation is architecture metadata: always construct with TRAINING size.
        encoder = networks.LiteMono(model=encoder_dict.get('model', opt.model),
                                    height=encoder_dict['height'], width=encoder_dict['width'])
        # Legacy checkpoints have one dispconv per decoder index >=6.
        heads = sorted({int(k.split('.')[1]) - 6 for k in decoder_dict
                        if k.startswith('decoder.') and k.endswith('.conv.weight')
                        and int(k.split('.')[1]) >= 6})
        scales = encoder_dict.get('scales', heads)
        if not scales or 0 not in scales:
            raise ValueError("checkpoint does not have a scale-0 depth head")
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=scales)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict}, strict=True)
        depth_decoder.load_state_dict(decoder_dict, strict=True)
        encoder.to(device).eval()
        depth_decoder.to(device).eval()
        profiled = False
        profile_info = None
        num_images, elapsed = 0, 0.0

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            feed_w, feed_h))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].to(device)
                # Any requested size >=64: only pad the right/bottom to encoder stride.
                input_color = F.pad(input_color, (0, (-feed_w) % 32, 0, (-feed_h) % 32), mode='replicate')

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                if opt.profile and not profiled:
                    profile_info = profile_once(encoder, depth_decoder, input_color)
                    profiled = True
                t1 = time_sync()
                output = depth_decoder(encoder(input_color))
                t2 = time_sync()

                elapsed += t2 - t1
                num_images += data[("color", 0, 0)].shape[0]
                pred_disp, _ = disp_to_depth(output[("disp", 0)][:, :, :feed_h, :feed_w],
                                             opt.min_depth, opt.max_depth)
                if not torch.isfinite(pred_disp).all():
                    raise FloatingPointError("non-finite inverse-depth prediction")
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if getattr(opt, 'eval_eigen_to_benchmark', False):
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        out_dir = opt.eval_out_dir or opt.load_weights_folder or '.'
        os.makedirs(out_dir, exist_ok=True)
        suffix = "{}x{}".format(feed_h, feed_w) if opt.ext_disp_to_eval is None else 'external'
        output_path = os.path.join(out_dir, "disps_{}_{}.npy".format(opt.eval_split, suffix))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")
    print("   Mono evaluation - median scaling: {}".format(not opt.disable_median_scaling))

    if len(gt_depths) != len(pred_disps):
        raise ValueError("GT/prediction count mismatch: {} vs {}".format(len(gt_depths), len(pred_disps)))
    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    if opt.ext_disp_to_eval is None:
        print("Measured forward throughput (no loader/profile): {:.2f} images/s".format(num_images / max(elapsed, 1e-9)))
        if profile_info is not None:
            print("THOP estimates (MACs, params; unsupported ops excluded):", profile_info)
    if opt.metrics_json:
        result = {
            "metrics": dict(zip(["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"], mean_errors.tolist())),
            "per_image_errors": np.asarray(errors).tolist(),
            "input_hw": [feed_h, feed_w] if opt.ext_disp_to_eval is None else None,
            "weights": opt.load_weights_folder, "post_process": opt.post_process,
            "median_scaling": not opt.disable_median_scaling,
            "depth_scale_factor": opt.pred_depth_scale_factor,
            "split": opt.eval_split, "num_images": len(errors),
            "test_files_sha256": hashlib.sha256('\n'.join(readlines(os.path.join(
                splits_dir, opt.eval_split, 'test_files.txt'))).encode()).hexdigest()}
        os.makedirs(os.path.dirname(os.path.abspath(opt.metrics_json)), exist_ok=True)
        with open(opt.metrics_json, 'w') as f:
            json.dump(result, f, indent=2)
    print("\n-> Done!")


def compare_metrics(directories):
    """Paired image-bootstrap intervals; use multiple training seeds for conclusions."""
    from pathlib import Path
    base_dir, ra_dir = map(Path, directories)
    left = {p.name: p for p in base_dir.glob('*.json')}
    right = {p.name: p for p in ra_dir.glob('*.json')}
    if set(left) != set(right) or not left:
        raise ValueError("directories must contain the same nonempty set of metric JSON filenames")
    print("Input HxW | Baseline AbsRel | RA AbsRel | RA-Base (95% image bootstrap CI) | Delta a1")
    rows = []
    rng = np.random.default_rng(42)
    for name in sorted(left):
        with open(left[name]) as f: b = json.load(f)
        with open(right[name]) as f: r = json.load(f)
        for key in ['input_hw', 'split', 'num_images', 'post_process', 'median_scaling',
                    'depth_scale_factor', 'test_files_sha256']:
            if key not in b or b[key] != r.get(key):
                raise ValueError("protocol mismatch or missing {} in {}".format(key, name))
        be, re = np.asarray(b['per_image_errors']), np.asarray(r['per_image_errors'])
        if be.shape != re.shape or be.shape != (b['num_images'], 7):
            raise ValueError("invalid paired error arrays")
        if not np.isfinite(be).all() or not np.isfinite(re).all():
            raise ValueError("metric arrays contain NaN/Inf")
        delta = re[:, 0] - be[:, 0]
        boot = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        hw = b['input_hw']
        print("{} | {:.5f} | {:.5f} | {:+.5f} [{:+.5f}, {:+.5f}] | {:+.5f}".format(
            hw, be[:, 0].mean(), re[:, 0].mean(), delta.mean(), lo, hi,
            (re[:, 4] - be[:, 4]).mean()))
        rows.append((hw, be[:, 0].mean(), re[:, 0].mean(), delta.mean()))
    print("Negative AbsRel delta is better. Image-bootstrap ignores correlation across nearby frames;")
    print("it does not replace repeated training seeds or prove resolution adaptation by itself.")
    print("Check off-training resolutions, in-resolution regression, and model selection on held-out val.")
    return rows


if __name__ == "__main__":
    options = LiteMonoOptions()
    opt = options.parse()
    if opt.compare_metrics:
        compare_metrics(opt.compare_metrics)
    else:
        evaluate(opt)
