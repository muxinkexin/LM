"""Evaluate all weights_* on the 697-image Eigen TEST set and export CSV by default.

Training validation/best selection is handled by Trainer --full_val.
Explicit --role val remains available for validation audits and GT export.
No training weights or source files are modified by this script.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
from depth_evaluation import (ROOT, METRICS, HEADERS, FullEvaluator, atomic_json,
    check_validation_split, csv_rows, load_checkpoint, metrics_mean, parse_resolutions,
    prepare_gt, publish_best, read_files, selection_key, sha256_file, write_csv)


# Only these exact previous engines are eligible for explicitly requested migration.
# fc0d9b0... predates path reuse; image loading, inference and metrics are unchanged.
COMPATIBLE_ENGINE_SHA256 = {
    '81475ce16f9e55ce6da4277639c0e1571cb30ef1b0639baea4c3b5434b778ff6',
    'fc0d9b0ed825feb0c5d818ad6ecd62c107c100169580761b6313e52c1a601a02',
}


def compatible_config(previous, current, allow_compatible=False):
    if previous==current:
        return True
    if allow_compatible and previous.get('engine_sha256') in COMPATIBLE_ENGINE_SHA256:
        return {**previous,'engine_sha256':current['engine_sha256']}==current
    return False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path', required=True)
    p.add_argument('--weights_root', help='run folder or its models folder')
    p.add_argument('--output_dir', help='CSV/ranking/cache output directory')
    p.add_argument('--eval_files', '--val_files', dest='eval_files')
    p.add_argument('--gt_path', help='GT cache directory or trusted NPZ with filenames')
    p.add_argument('--role', choices=['val','test'], default='test',
                   help='default: 697-image Eigen test report; val is explicit validation audit/export')
    p.add_argument('--prepare_gt', action='store_true')
    p.add_argument('--png', action='store_true')
    p.add_argument('--resolutions', nargs='+', default=['128x416','192x640','320x1024'])
    p.add_argument('--model', default='lite-mono-tiny', help='fallback for old checkpoints lacking model metadata')
    p.add_argument('--eval_batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--allow_tf32', action='store_true')
    p.add_argument('--disable_median_scaling', action='store_true')
    p.add_argument('--pred_depth_scale_factor', type=float, default=1.)
    p.add_argument('--min_depth', type=float, default=.1)
    p.add_argument('--max_depth', type=float, default=100.)
    p.add_argument('--crop', choices=['garg','none'], default='garg')
    p.add_argument('--post_process', action='store_true')
    p.add_argument('--select_metric', choices=METRICS, default='abs_rel')
    p.add_argument('--tensorboard_dir', help='optional retrospective metric curves for existing checkpoints')
    p.add_argument('--diagnostics', action='store_true', help='trace exact module entry on forward failure')
    p.add_argument('--forward_retries', type=int, default=0, help='reload weights and retry NoneType-call failures N times')
    p.add_argument('--reuse_compatible_results', action='store_true', help='explicitly reuse complete records from the known previous engine')
    p.add_argument('--resume', action='store_true', help='reuse only matching, fingerprinted evaluation records')
    p.add_argument('--progress_every', type=int, default=100)
    return p


def discover(root):
    root = Path(root).expanduser().resolve()
    if (root/'models').is_dir():
        root = root/'models'
    found = []
    for path in root.glob('weights_*'):
        match = re.fullmatch(r'weights_(\d+)(?:_step_(\d+))?', path.name)
        if match and path.is_dir() and not path.is_symlink():
            for name in ['encoder.pth','depth.pth']:
                if not (path/name).is_file():
                    raise FileNotFoundError('Incomplete checkpoint: '+str(path/name))
            epoch, step = match.groups()
            found.append((int(epoch), int(step) if step else sys.maxsize, path))
    if not found:
        raise FileNotFoundError('No complete weights_* directories found in '+str(root))
    return sorted(found)


def run(args):
    if args.forward_retries < 0:
        raise ValueError('--forward_retries must be nonnegative')
    files = args.eval_files or str(ROOT / ('splits/eigen_zhou/val_files.txt' if args.role=='val' else 'splits/eigen/test_files.txt'))
    gt = args.gt_path or (str(ROOT/'splits/eigen/gt_depths.npz') if args.role=='test' else None)
    if not gt:
        raise ValueError('Validation needs --gt_path. Export it first with --prepare_gt; test GT is not validation GT.')
    if args.prepare_gt:
        return prepare_gt(args.data_path, files, gt, args.png, args.role)
    if not args.weights_root or not args.output_dir:
        raise ValueError('--weights_root and --output_dir are required for a sweep')
    resolutions = parse_resolutions(args.resolutions)
    if args.role=='val':
        train_file = ROOT/'splits/eigen_zhou/train_files.txt'
        check_validation_split(read_files(files), read_files(train_file) if train_file.exists() else None)
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    evaluator = FullEvaluator(args.data_path, files, gt, resolutions, args.png,
        args.eval_batch_size, args.num_workers, device, args.role, args.crop,
        not args.disable_median_scaling, args.pred_depth_scale_factor,
        args.min_depth, args.max_depth, args.post_process, args.progress_every,
        diagnostics=args.diagnostics, diagnostic_dir=Path(args.output_dir).expanduser().resolve()/'diagnostics')
    candidates = discover(args.weights_root)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {**evaluator.config, 'engine_sha256': sha256_file(ROOT/'depth_evaluation.py'),
              'torch_version': str(torch.__version__), 'allow_tf32': args.allow_tf32,
              'device': str(device), 'legacy_model': args.model,
              'batch_size': args.eval_batch_size, 'select_metric': args.select_metric}
    manifest_path = output/'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not args.resume:
            raise FileExistsError('Output already contains a run; use a new directory or --resume')
        if not compatible_config(previous['config'],config,args.reuse_compatible_results):
            raise ValueError('Resume evaluation protocol/engine differs; use a new output directory')
    elif any(output.iterdir()):
        raise FileExistsError('Output directory contains unrelated files; choose a new directory')
    atomic_json(manifest_path, {'status':'running','config':config,'candidate_count':len(candidates)})
    rows, records = [], []
    writer = None
    if args.tensorboard_dir:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.tensorboard_dir)
    try:
        for index, (epoch, _, folder) in enumerate(candidates):
            fingerprint = {name:sha256_file(folder/name) for name in ['encoder.pth','depth.pth']}
            cache = output/'records'/(folder.name+'.json')
            record = json.loads(cache.read_text()) if args.resume and cache.is_file() else None
            if record and record.get('fingerprint')==fingerprint and compatible_config(record.get('config',{}),config,args.reuse_compatible_results):
                print('Reusing verified evaluation record:', folder.name, flush=True)
                if record['config']!=config:
                    record = {**record,'config':config,'compatible_engine_source':record['config']['engine_sha256']}
                    atomic_json(cache,record)
            else:
                print('Checkpoint {}/{}: {}'.format(index+1,len(candidates),folder), flush=True)
                partial_path = output/'records'/(folder.name+'.partial.json')
                partial = json.loads(partial_path.read_text()) if args.resume and partial_path.is_file() else None
                completed = []
                if partial and partial.get('fingerprint')==fingerprint and compatible_config(partial.get('config',{}),config,args.reuse_compatible_results):
                    completed = partial['results']
                def resolution_done(result):
                    nonlocal completed
                    completed = [r for r in completed if (r['height'],r['width'])!=(result['height'],result['width'])]+[result]
                    completed.sort(key=lambda r:resolutions.index((r['height'],r['width'])))
                    atomic_json(partial_path, {'weights':folder.name,'fingerprint':fingerprint,
                                              'config':config,'results':completed})
                    # Also publish completed resolution rows before the entire checkpoint finishes.
                    write_csv(output/'all_weights.csv',rows+csv_rows(completed,folder,epoch,current_step,evaluator.config))
                for attempt in range(args.forward_retries+1):
                    encoder, decoder, meta = load_checkpoint(folder, device, args.model)
                    current_step = int(meta.get('global_step',-1))
                    evaluator.evaluation_context = {'checkpoint':str(folder),'attempt':attempt}
                    for name in ['min_depth','max_depth']:
                        if name in meta and abs(float(meta[name])-float(getattr(args,name))) > 1e-12:
                            raise ValueError('Checkpoint {} differs; pass --{} {}'.format(name,name,meta[name]))
                    try:
                        results = evaluator.evaluate(encoder,decoder,profile=True,
                            on_resolution=resolution_done,completed_results=completed)
                        break
                    except TypeError as error:
                        if attempt>=args.forward_retries or "'NoneType' object is not callable" not in str(error):
                            raise
                        print('Reloading full checkpoint after NoneType-call failure; retry {}/{}'.format(
                            attempt+1,args.forward_retries),flush=True)
                    finally:
                        del encoder, decoder
                        gc.collect()
                        if device.type=='cuda':torch.cuda.empty_cache()
                record = {'weights':folder.name,'checkpoint':str(folder),'epoch':epoch,
                          'global_step':current_step,'fingerprint':fingerprint,
                          'config':config,'results':results}
                atomic_json(cache, record)
                if partial_path.exists():partial_path.unlink()
            records.append(record)
            if writer is not None:
                step = record['global_step'] if record['global_step'] >= 0 else record['epoch']
                for result in record['results']:
                    for metric in METRICS:
                        writer.add_scalar('checkpoints/{}/{}x{}/{}'.format(args.role,result['height'],result['width'],metric),result[metric],step)
                for metric,value in metrics_mean(record['results']).items():
                    writer.add_scalar('checkpoints/'+args.role+'/mean/'+metric,value,step)
                writer.flush()
            rows.extend(csv_rows(record['results'], folder, epoch, record['global_step'], evaluator.config))
            write_csv(output/'all_weights.csv', rows)
        if args.role=='val':
            ordered = sorted(records, key=lambda r: (*selection_key(r['results'],args.select_metric),r['epoch'],r['global_step']))
            ranking = []
            for rank, record in enumerate(ordered,1):
                means = metrics_mean(record['results'])
                ranking.append({'rank':rank,'weights':record['weights'],'checkpoint':record['checkpoint'],
                                'epoch':record['epoch'],'global_step':record['global_step'],
                                'selection_metric':args.select_metric,'score':means[args.select_metric],
                                **{'mean '+h:means[k] for h,k in zip(HEADERS,METRICS)}})
            write_csv(output/'ranking.csv',ranking,list(ranking[0]))
            best = ordered[0]
            publish_best(output,best['checkpoint'],best['results'],args.select_metric,
                         best['epoch'],best['global_step'],evaluator.config)
            individual = {}
            for h,w in resolutions:
                ranked = sorted(records,key=lambda r: selection_key(
                    [next(row for row in r['results'] if row['height']==h and row['width']==w)],args.select_metric))
                individual['{}x{}'.format(h,w)] = {'weights':ranked[0]['weights'],'checkpoint':ranked[0]['checkpoint']}
            atomic_json(output/'best_by_resolution.json',individual)
        else:
            print('TEST report only: no best-model selection or ranking on test data.',flush=True)
        atomic_json(manifest_path, {'status':'complete','config':config,
                    'candidate_count':len(candidates),'evaluated_count':len(records)})
        print('CSV:',output/'all_weights.csv')
        return records
    except BaseException as error:
        atomic_json(manifest_path, {'status':'failed','config':config,
                    'candidate_count':len(candidates),'evaluated_count':len(records),'error':str(error)})
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__=='__main__':
    run(parser().parse_args())
