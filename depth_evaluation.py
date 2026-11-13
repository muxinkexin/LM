"""Shared, streaming KITTI evaluation for training and checkpoint sweeps.

Metrics: per-image Garg crop, valid GT in (1e-3,80), optional per-image
median scaling. Complexity: Conv/Linear + both XCA matrix products only;
FLOPs=2*MACs. Norm/activation/pooling/resize/softmax are explicitly excluded.
"""
from contextlib import contextmanager
from pathlib import Path
import csv
import copy
import sys
import traceback
import uuid
import importlib.metadata
import hashlib
import json
import os
import re
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from layers import disp_to_depth
import networks

ROOT = Path(__file__).resolve().parent
METRICS = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
HEADERS = ('Abs Rel', 'Sq Rel', 'RMSE', 'RMSE log', 'a1', 'a2', 'a3')
COMPLEXITY_CONVENTION = 'FLOPs=2*MACs; Conv2d+Linear+XCA_matmul; other_ops_excluded; B=1'
CSV_FIELDS = ['weights', 'checkpoint', 'epoch', 'global_step', 'role', 'height', 'width',
              'padded_height', 'padded_width', 'num_images', *HEADERS, 'Params', 'MACs',
              'FLOPs', 'Params_M', 'GMACs', 'GFLOPs', 'SinglePassFLOPs',
              'post_process', 'median_scaling', 'crop', 'files_sha256', 'GT_fingerprint',
              'complexity_convention']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def parse_resolutions(values):
    result = []
    for value in values:
        match = re.fullmatch(r'(\d+)[xX](\d+)', value)
        if not match:
            raise ValueError('Resolution must be HxW, e.g. 192x640: ' + value)
        hw = tuple(map(int, match.groups()))
        if min(hw) < 64 or hw in result:
            raise ValueError('Dimensions must be >=64 and resolutions unique: ' + value)
        result.append(hw)
    if not result:
        raise ValueError('At least one resolution is required')
    return result


def read_files(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError('Empty image list: ' + str(path))
    for line in lines:
        parse_line(line)
    if len(set(map(canonical_line, lines))) != len(lines):
        raise ValueError('Duplicate targets in image list: ' + str(path))
    return lines


def parse_line(line):
    folder, frame, side = line.split()
    camera = {'l': 2, 'r': 3, '2': 2, '3': 3}[side]
    return folder, int(frame), camera


def canonical_line(line):
    folder, frame, camera = parse_line(line)
    return '{} {} {}'.format(folder, frame, camera)


def list_hash(lines):
    return hashlib.sha256('\n'.join(map(canonical_line, lines)).encode()).hexdigest()


def check_validation_split(lines, train_lines=None):
    """Check target-frame disjointness; this does not imply drive independence."""
    keys = set(map(canonical_line, lines))
    test_file = ROOT / 'splits/eigen/test_files.txt'
    if test_file.exists() and keys.intersection(map(canonical_line, read_files(test_file))):
        raise ValueError('Validation targets overlap Eigen TEST; refuse best-model selection')
    if train_lines is not None and keys.intersection(map(canonical_line, train_lines)):
        raise ValueError('Validation targets overlap TRAIN targets')


def image_path(root, line, png=False):
    folder, frame, camera = parse_line(line)
    return Path(root, folder, 'image_0{}'.format(camera), 'data',
                '{:010d}{}'.format(frame, '.png' if png else '.jpg'))


class GroundTruth:
    """Lazy per-frame cache, or trusted NPZ with an explicitly matched image list."""
    def __init__(self, path, lines, role='val'):
        self.path, self.lines = Path(path).expanduser().resolve(), lines
        self.depths = None
        if self.path.is_dir():
            self.manifest = json.loads((self.path / 'manifest.json').read_text())
            if self.manifest.get('format') != 'LM_KITTI_DEPTH_CACHE_V1':
                raise ValueError('Unknown GT cache format')
            if self.manifest['files_sha256'] != list_hash(lines):
                raise ValueError('GT cache image order/list mismatch')
            if list(map(canonical_line, self.manifest['filenames'])) != list(map(canonical_line, lines)):
                raise ValueError('GT cache filenames mismatch')
            if len(self.manifest['entries']) != len(lines):
                raise ValueError('GT cache entry count mismatch')
            for entry in self.manifest['entries']:
                name = Path(entry['file'])
                if name.is_absolute() or '..' in name.parts or not (self.path / name).is_file():
                    raise FileNotFoundError('Incomplete or unsafe GT cache entry: ' + str(name))
                if sha256_file(self.path / name) != entry['file_sha256']:
                    raise ValueError('GT cache compressed-file checksum mismatch: ' + str(name))
            self.fingerprint = sha256_file(self.path / 'manifest.json')
        else:
            # allow_pickle is required by the original variable-size KITTI GT archive.
            # Use only locally generated or explicitly trusted GT archives.
            with np.load(self.path, allow_pickle=True, encoding='latin1') as archive:
                if 'filenames' in archive:
                    names = archive['filenames'].astype(str).tolist()
                    if list(map(canonical_line, names)) != list(map(canonical_line, lines)):
                        raise ValueError('GT NPZ filenames/order mismatch')
                else:
                    builtin = (ROOT / 'splits/eigen/gt_depths.npz').resolve()
                    tests = read_files(ROOT / 'splits/eigen/test_files.txt')
                    if role != 'test' or self.path != builtin or list_hash(lines) != list_hash(tests):
                        raise ValueError('Custom/validation GT NPZ must contain matching filenames; do not reuse test GT')
                self.depths = archive['data']
            if len(self.depths) != len(lines):
                raise ValueError('GT count does not match image list')
            self.fingerprint = sha256_file(self.path)

    def get(self, index):
        if self.depths is not None:
            depth = np.asarray(self.depths[index], dtype=np.float32)
        else:
            entry = self.manifest['entries'][index]
            with np.load(self.path / entry['file'], allow_pickle=False) as archive:
                depth = archive['depth'].astype(np.float32)
            # Each compressed file is checked as it is consumed; never silently skip corruption.
            if hashlib.sha256(depth.tobytes()).hexdigest() != entry['depth_sha256']:
                raise ValueError('GT cache checksum mismatch at index {}'.format(index))
        if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError('Invalid GT depth at index {}'.format(index))
        return depth


def valid_mask(depth, crop='garg'):
    h, w = depth.shape
    mask = (depth > 1e-3) & (depth < 80)
    if crop == 'garg':
        top, bottom, left, right = np.array([.40810811*h, .99189189*h,
                                            .03594771*w, .96405229*w]).astype(int)
        cropped = np.zeros_like(mask)
        cropped[top:bottom, left:right] = True
        mask &= cropped
    elif crop != 'none':
        raise ValueError('Unknown evaluation crop')
    if not mask.any():
        raise ValueError('No valid GT pixels after depth range/crop; fix GT instead of skipping images')
    return mask


def prepare_gt(data_path, files_path, output, png=False, role='val'):
    """Export lazily readable raw-resolution GT; all source files preflight first."""
    from kitti_utils import generate_depth_map
    data_path = Path(data_path).expanduser().resolve()
    lines = read_files(files_path)
    if role == 'val':
        check_validation_split(lines)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('GT output must be absent or empty; use a new directory: ' + str(output))
    sources = []
    for line in lines:
        folder, frame, camera = parse_line(line)
        calib = Path(data_path) / folder.split('/')[0]
        velo = Path(data_path) / folder / 'velodyne_points/data' / ('{:010d}.bin'.format(frame))
        rgb = image_path(data_path, line, png)
        for file in [rgb, velo, calib/'calib_cam_to_cam.txt', calib/'calib_velo_to_cam.txt']:
            if not file.is_file():
                raise FileNotFoundError('GT requires RGB + raw LiDAR + calibration; missing: ' + str(file))
        sources.append((calib, velo, camera, rgb))
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (calib, velo, camera, rgb) in enumerate(sources):
        depth = generate_depth_map(str(calib), str(velo), camera, vel_depth=True).astype(np.float32)
        with Image.open(rgb) as image:
            if depth.shape != (image.height, image.width):
                raise ValueError('Native RGB/calibration dimensions mismatch: ' + str(rgb))
        valid_mask(depth)
        name = '{:06d}.npz'.format(i)
        tmp = output / (name + '.tmp')
        with open(tmp, 'wb') as stream:
            np.savez_compressed(stream, depth=depth)
        os.replace(tmp, output / name)
        entries.append({'file': name, 'shape': list(depth.shape), 'file_sha256':sha256_file(output/name),
                        'depth_sha256': hashlib.sha256(depth.tobytes()).hexdigest()})
        if (i+1) % 100 == 0 or i+1 == len(lines):
            print('GT export: {}/{}'.format(i+1, len(lines)), flush=True)
    manifest = {'format': 'LM_KITTI_DEPTH_CACHE_V1', 'filenames': lines,
                'files_sha256': list_hash(lines), 'entries': entries,
                'projection': 'kitti_utils.generate_depth_map(cam=side, vel_depth=True)',
                'data_path': str(Path(data_path).expanduser().resolve()), 'role': role}
    atomic_json(output/'manifest.json', manifest)
    print('Complete GT cache:', output)
    return output


class RGBDataset(Dataset):
    """A resolution-specific view of paths already checked by FullEvaluator."""
    def __init__(self, paths, height, width):
        self.paths = tuple(paths)
        self.hw = height, width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        h, w = self.hw
        with Image.open(self.paths[index]) as image:
            image = image.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32).copy() / 255
        return torch.from_numpy(array).permute(2, 0, 1), index


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    return (np.mean(np.abs(gt-pred)/gt), np.mean((gt-pred)**2/gt),
            np.sqrt(np.mean((gt-pred)**2)), np.sqrt(np.mean((np.log(gt)-np.log(pred))**2)),
            (thresh < 1.25).mean(), (thresh < 1.25**2).mean(), (thresh < 1.25**3).mean())


def score_prediction(gt, inverse_depth, median_scaling=True, depth_scale=1., crop='garg'):
    if not np.isfinite(inverse_depth).all() or (inverse_depth <= 0).any():
        raise FloatingPointError('Invalid inverse-depth prediction')
    pred = 1 / cv2.resize(inverse_depth, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = valid_mask(gt, crop)
    target, pred = gt[mask].astype(np.float64), pred[mask].astype(np.float64) * depth_scale
    ratio = 1.
    if median_scaling:
        ratio = np.median(target) / np.median(pred)
        pred *= ratio
    errors = np.asarray(compute_errors(target, np.clip(pred, 1e-3, 80)), dtype=np.float64)
    if not np.isfinite(errors).all():
        raise FloatingPointError('Non-finite depth metrics')
    return errors, float(ratio)


@contextmanager
def evaluation_mode(*models):
    flags = [(module, module.training) for model in models for module in model.modules()]
    try:
        for model in models:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, flag in flags:
            module.training = flag


def _profile_depth_inplace(encoder, decoder, height, width, device, post_process=False):
    """Exact parameter count; explicitly limited MAC/FLOP convention. No THOP state mutation."""
    macs = [0]
    handles = []
    def conv(module, args, output):
        macs[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
    def linear(module, args, output):
        macs[0] += output.numel() * module.in_features
    def xca(module, args, output):
        b, n, c = args[0].shape
        macs[0] += 2 * b * n * c * c // module.num_heads
    try:
        for model in [encoder, decoder]:
            for module in model.modules():
                hook = conv if isinstance(module, torch.nn.Conv2d) else linear if isinstance(module, torch.nn.Linear) else xca if isinstance(module, networks.depth_encoder.XCA) else None
                if hook is not None:
                    handles.append(module.register_forward_hook(hook))
        with evaluation_mode(encoder, decoder):
            x = torch.zeros(1, 3, (height+31)//32*32, (width+31)//32*32, device=device)
            decoder(encoder(x))
    finally:
        for handle in handles:
            handle.remove()
    params = sum(p.numel() for model in [encoder, decoder] for p in model.parameters())
    passes = 2 if post_process else 1
    return {'Params': params, 'MACs': macs[0]*passes, 'FLOPs': 2*macs[0]*passes,
            'Params_M': params/1e6, 'GMACs': macs[0]*passes/1e9,
            'GFLOPs': 2*macs[0]*passes/1e9, 'SinglePassFLOPs': 2*macs[0],
            'complexity_convention': COMPLEXITY_CONVENTION}


def module_health(*named_models):
    issues = []
    for prefix, model in named_models:
        for name, module in model.named_modules():
            path = prefix + ("." + name if name else "")
            if not callable(getattr(module, 'forward', None)):
                issues.append({'module':path, 'type':type(module).__name__,
                               'problem':'forward is not callable',
                               'instance_forward_override':'forward' in module.__dict__})
            for attribute in ('_forward_hooks','_forward_pre_hooks'):
                for key, value in getattr(module, attribute, {}).items():
                    if not callable(value):
                        issues.append({'module':path, 'problem':attribute+' contains non-callable',
                                       'hook_id':str(key),'value_type':type(value).__name__})
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                for key, child in module._modules.items():
                    if not callable(child):
                        issues.append({'module':path+'.'+key,'problem':'container child is not callable'})
    return issues


def profile_depth(encoder, decoder, height, width, device, post_process=False):
    """Profile detached CPU replicas, never register counting hooks on the live model.

    Memoize known Lite-Mono runtime caches as empty: the decoder's last training
    outputs can be non-leaf tensors and cannot be deep-copied after backward.
    """
    problems = module_health(('encoder',encoder),('decoder',decoder))
    if problems:
        raise TypeError('Model is not callable before profiling: '+json.dumps(problems))
    replicas = []
    for model in (encoder,decoder):
        memo = {}
        for module in model.modules():
            for attribute in ('outputs','features'):
                value = module.__dict__.get(attribute)
                if isinstance(value, (dict,list)):
                    memo[id(value)] = {} if isinstance(value,dict) else []
        replica = copy.deepcopy(model,memo).cpu()
        for module in replica.modules():
            # External hooks are not part of the MAC convention; clear replicas only.
            for attribute in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks'):
                getattr(module,attribute,{}).clear()
            module.__dict__.pop('_compiled_call_impl',None)
        replicas.append(replica)
    return _profile_depth_inplace(*replicas,height,width,torch.device('cpu'),post_process)


@contextmanager
def forward_trace(named_models, enabled):
    state = {'last_entered':None,'stack':[],'recent':[]}
    handles = []
    def enter(name):
        def hook(module, args):
            state['last_entered'] = name
            state['stack'].append(name)
            state['recent'] = (state['recent']+[name])[-24:]
        return hook
    def leave(name):
        def hook(module, args, output):
            if state['stack'] and state['stack'][-1]==name:
                state['stack'].pop()
        return hook
    try:
        if enabled:
            for prefix, model in named_models:
                for name, module in model.named_modules():
                    path = prefix + ('.'+name if name else '')
                    handles.append(module.register_forward_pre_hook(enter(path)))
                    handles.append(module.register_forward_hook(leave(path)))
        yield state
    finally:
        for handle in handles:
            handle.remove()



def post_process(l_disp, r_disp):
    _, h, w = l_disp.shape
    middle = (l_disp + r_disp) / 2
    left = (1 - np.clip(20*(np.linspace(0, 1, w)-.05), 0, 1))[None, None, :]
    right = left[:, :, ::-1]
    return right*l_disp + left*r_disp + (1-left-right)*middle


class FullEvaluator:
    def __init__(self, data_path, files_path, gt_path, resolutions, png=False,
                 batch_size=4, workers=4, device='cuda', role='val', crop='garg',
                 median_scaling=True, depth_scale=1., min_depth=.1, max_depth=100.,
                 post_process_enabled=False, progress_every=100, diagnostics=False, diagnostic_dir=None):
        if batch_size < 1 or workers < 0 or min_depth <= 0 or max_depth <= min_depth or depth_scale <= 0:
            raise ValueError('Invalid evaluation configuration')
        self.lines = read_files(files_path)
        if role == 'val':
            check_validation_split(self.lines)
        self.gt = GroundTruth(gt_path, self.lines, role)
        self.resolutions = resolutions
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --no_cuda')
        self.config = {'data_path': str(Path(data_path).expanduser().resolve()),
                       'files_sha256': list_hash(self.lines), 'GT_fingerprint': self.gt.fingerprint,
                       'role': role, 'crop': crop, 'median_scaling': median_scaling,
                       'depth_scale': depth_scale, 'min_depth': min_depth, 'max_depth': max_depth,
                       'post_process': post_process_enabled, 'png': png,
                       'resolutions': [list(hw) for hw in resolutions]}
        self.batch_size, self.workers, self.progress_every = batch_size, workers, progress_every
        self.diagnostics = diagnostics
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir is not None else None
        self.evaluation_context = {}
        # Validate paths once, not during the first expensive training epoch.
        self.dataset_paths = tuple(image_path(self.config['data_path'], line, png) for line in self.lines)
        for path in self.dataset_paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    def _predict(self, encoder, decoder, x, h, w, batch_index, indices):
        named = [('encoder',encoder),('decoder',decoder)]
        with forward_trace(named,self.diagnostics) as trace:
            try:
                return decoder(encoder(x))[('disp',0)][:,:,:h,:w]
            except Exception as error:
                versions = {}
                for package in ('torch','torchvision','timm','numpy'):
                    try: versions[package] = importlib.metadata.version(package)
                    except importlib.metadata.PackageNotFoundError: pass
                report = {'exception':type(error).__name__, 'message':str(error),
                    'traceback':traceback.format_exc(),'context':self.evaluation_context,
                    'input_hw':[h,w],'network_shape':list(x.shape),
                    'batch_index_zero_based':batch_index,'image_indices':indices.tolist(),
                    'image_lines':[self.lines[i] for i in indices.tolist()],
                    'last_entered_module':trace['last_entered'], 'active_module_stack':trace['stack'],
                    'recent_modules':trace['recent'], 'module_issues':module_health(*named),
                    'versions':versions, 'python':sys.version, 'device':str(self.device),
                    'cuda_runtime':torch.version.cuda, 'cudnn_version':torch.backends.cudnn.version(),
                    'encoder_source':str(Path(networks.depth_encoder.__file__).resolve()),
                    'encoder_source_sha256':sha256_file(networks.depth_encoder.__file__)}
                print('FORWARD FAILURE: batch {} at {}x{}; last module {}; issues {}'.format(
                    batch_index,h,w,trace['last_entered'],report['module_issues']),flush=True)
                if self.diagnostic_dir is not None:
                    path = self.diagnostic_dir/('forward_failure_'+uuid.uuid4().hex[:12]+'.json')
                    atomic_json(path,report)
                    print('Diagnostic report:',path,flush=True)
                raise

    def evaluate(self, encoder, decoder, profile=True, on_resolution=None,
                 prediction_callback=None, per_image=False, completed_results=None):
        results = []
        with evaluation_mode(encoder, decoder):
            for h, w in self.resolutions:
                prior = next((r for r in (completed_results or []) if (r['height'],r['width'])==(h,w)),None)
                if prior is not None:
                    if prior['num_images'] != len(self.lines) or not all(np.isfinite(prior[k]) for k in METRICS):
                        raise ValueError('Invalid completed resolution cache')
                    if profile and not all(k in prior for k in ('Params','MACs','FLOPs')):
                        raise ValueError('Completed cache lacks complexity fields')
                    if per_image and len(prior.get('per_image_errors',[]))!=len(self.lines):
                        raise ValueError('Completed cache lacks per-image metrics')
                    print('Reusing complete resolution: {}x{}'.format(h,w),flush=True)
                    results.append(prior)
                    if on_resolution: on_resolution(prior)
                    continue
                problems = module_health(('encoder',encoder),('decoder',decoder))
                if problems:
                    raise TypeError('Model is not callable before inference: '+json.dumps(problems))
                dataset = RGBDataset(self.dataset_paths, h, w)
                loader = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=False,
                                    num_workers=self.workers, pin_memory=self.device.type=='cuda',
                                    generator=torch.Generator().manual_seed(42))
                total, count, errors = np.zeros(7, dtype=np.float64), 0, []
                complexity = profile_depth(encoder, decoder, h, w, self.device, self.config['post_process']) if profile else {}
                for batch_index, (color, indices) in enumerate(loader):
                    x = color.to(self.device, non_blocking=True)
                    if self.config['post_process']:
                        x = torch.cat((x, torch.flip(x, [3])), 0)
                    x = F.pad(x, (0, (-w)%32, 0, (-h)%32), mode='replicate')
                    disp = self._predict(encoder,decoder,x,h,w,batch_index,indices)
                    inv, _ = disp_to_depth(disp, self.config['min_depth'], self.config['max_depth'])
                    inv = inv[:, 0].float().cpu().numpy()
                    if self.config['post_process']:
                        n = len(indices)
                        inv = post_process(inv[:n], inv[n:, :, ::-1])
                    for j, index in enumerate(indices.tolist()):
                        e, _ = score_prediction(self.gt.get(index), inv[j],
                            self.config['median_scaling'], self.config['depth_scale'], self.config['crop'])
                        total += e
                        count += 1
                        if per_image:
                            errors.append(e.tolist())
                        if prediction_callback:
                            prediction_callback(h, w, index, inv[j])
                    if self.progress_every and ((batch_index+1)%self.progress_every==0 or count==len(self.lines)):
                        print('Full {} {}x{}: {}/{} images, running AbsRel={:.5f}'.format(
                            self.config['role'], h, w, count, len(self.lines), total[0]/count), flush=True)
                if count != len(self.lines):
                    raise RuntimeError('Full evaluation did not consume every target')
                result = {'height': h, 'width': w, 'padded_height': (h+31)//32*32,
                          'padded_width': (w+31)//32*32, 'num_images': count,
                          **dict(zip(METRICS, (total/count).tolist())), **complexity}
                if per_image:
                    result['per_image_errors'] = errors
                results.append(result)
                print('{}x{} '.format(h, w) + ' '.join('{}={:.5f}'.format(k, result[k]) for k in METRICS), flush=True)
                if on_resolution:
                    on_resolution(result)
        return results


def load_checkpoint(folder, device, legacy_model='lite-mono-tiny'):
    folder = Path(folder)
    enc_state = torch.load(folder/'encoder.pth', map_location='cpu', weights_only=True)
    dec_state = torch.load(folder/'depth.pth', map_location='cpu', weights_only=True)
    model = enc_state.get('model', legacy_model)
    encoder = networks.LiteMono(model=model, height=int(enc_state['height']), width=int(enc_state['width']))
    heads = sorted({int(k.split('.')[1])-6 for k in dec_state
                    if k.startswith('decoder.') and k.endswith('.conv.weight') and int(k.split('.')[1])>=6})
    scales = enc_state.get('scales', heads)
    if 0 not in scales:
        raise ValueError('Checkpoint must have a scale-0 depth head')
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=scales)
    encoder.load_state_dict({k:v for k,v in enc_state.items() if k in encoder.state_dict()}, strict=True)
    decoder.load_state_dict(dec_state, strict=True)
    encoder.to(device).eval()
    decoder.to(device).eval()
    meta = {k:v for k,v in enc_state.items() if k not in encoder.state_dict()}
    meta['model'], meta['scales'] = model, list(scales)
    return encoder, decoder, meta


def metrics_mean(results):
    return {name: float(np.mean([row[name] for row in results])) for name in METRICS}


def selection_key(results, metric='abs_rel'):
    means = metrics_mean(results)
    return (-means[metric] if metric in ('a1','a2','a3') else means[metric], -means['a1'])


def csv_rows(results, checkpoint, epoch, step, config):
    rows = []
    for result in results:
        row = {k:v for k,v in result.items() if k in CSV_FIELDS}
        row.update(dict(zip(HEADERS, [result[k] for k in METRICS])))
        row.update({'weights': Path(checkpoint).name, 'checkpoint': str(Path(checkpoint).resolve()),
                    'epoch': epoch, 'global_step': step,
                    **{k:config[k] for k in ['role','post_process','median_scaling','crop','files_sha256','GT_fingerprint']}})
        rows.append(row)
    return rows


def write_csv(path, rows, fields=CSV_FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def publish_best(output, checkpoint, results, metric, epoch, step, config, link=False):
    output, checkpoint = Path(output), Path(checkpoint).resolve()
    if config['role'] != 'val':
        raise ValueError('Best-model selection is allowed on validation only')
    summary = {'weights': checkpoint.name, 'checkpoint': str(checkpoint), 'epoch': epoch,
               'global_step': step, 'selection_metric': metric, 'aggregation': 'equal-resolution mean',
               'score': metrics_mean(results)[metric], 'mean_metrics': metrics_mean(results),
               'per_resolution': results, 'protocol': config}
    if not (checkpoint/'encoder.pth').is_file() or not (checkpoint/'depth.pth').is_file():
        raise FileNotFoundError('Best checkpoint was not saved completely')
    atomic_json(output/'best_weights.json', summary)
    (output/'best_weights.txt').write_text(str(checkpoint)+'\n')
    if link:
        destination = output/'best'
        if destination.exists() and not destination.is_symlink():
            raise FileExistsError('Will not replace a non-symlink best directory: '+str(destination))
        tmp = output/'best.tmp'
        if tmp.is_symlink():
            tmp.unlink()
        tmp.symlink_to(os.path.relpath(checkpoint, output.resolve()), target_is_directory=True)
        os.replace(tmp, destination)
    print('BEST validation checkpoint: {} | mean {}={:.6f}'.format(checkpoint.name, metric, summary['score']), flush=True)
    return summary
