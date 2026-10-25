"""Root-owned request materializer; creates no admission and runs no numerical code."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'c-rl-20261024-r1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pin(p):
    p = Path(p).resolve()
    return {'path': p.as_posix(), 'sha256': sha(p)}


def checkpoint(training):
    training = Path(training)
    d = read(training)
    assert d['state'] == 'completed' and d['updates'] == 1000
    entry = next(x for x in d['checkpoints'] if x['step'] == 1000)
    assert sha(entry['path']) == entry['sha256']
    return {**entry, 'training': pin(training)}


def build(mode, cap):
    base = OLD / ('request-live-prefix-r1.json' if mode == 'rollout' else 'request-evaluate-selected-r1.json')
    q = read(base)
    q.update(operation={'screen': 'screen-scientific3-r1', 'full': 'full-scientific3-r1', 'rollout': 'rollout-scientific3-r1'}[mode],
             deadline_utc='2026-10-25T05:45:00+00:00', maximum_seconds=3600,
             cap_gib=cap, reserve_gib=2, cpu_threads=4,
             stage='live_prefix' if mode == 'rollout' else 'evaluate',
             module_path=(ROOT / 'policy' / ('rollout.py' if mode == 'rollout' else 'evaluate.py')).as_posix(),
             function={'screen': 'score', 'full': 'evaluate', 'rollout': 'run'}[mode],
             supervised_updates=0, new_rm_fits=0, final_test_reads=0, rl_updates=0,
             owned_output_roots=[(ROOT / 'policy').as_posix()],
             policy_module=pin(OLD / 'policy/policy_v2.py'),
             residual_module=pin(ROOT / 'policy/residual.py'),
             utility_module=pin(ROOT / 'reward/utility.py'),
             scientific_checkpoints={
                 'RL-completion': checkpoint(OLD / 'policy/rl-r2/training.json'),
                 'RL-residual': checkpoint(ROOT / 'policy/residual-r1/training.json')})
    for old in ['evaluation_checkpoints', 'rollout_checkpoints', 'frontier_output', 'frontier_sha256']:
        q.pop(old, None)
    new_sources = [ROOT/'policy/evaluate.py', ROOT/'policy/residual.py', ROOT/'reward/utility.py']
    new_sources += [ROOT/'control'/x for x in ['common.py', 'launch.py', 'supervise.py', 'worker.py']]
    if mode == 'rollout':
        new_sources += [ROOT/'policy/rollout.py']
        q['evaluate_module'] = pin(ROOT/'policy/evaluate.py')
    for p in new_sources:
        q['source_hashes'][p.as_posix()] = sha(p)
    input_paths = [ROOT/'protocols/evaluation.json', ROOT/'protocols/preregistration.json']
    for c in q['scientific_checkpoints'].values():
        input_paths += [Path(c['path']), Path(c['training']['path'])]
    if mode in ('screen', 'rollout'):
        q.update(cache_dir=(OLD/'policy/screen-cache-r1').as_posix(),
                 cohort_name='screen', rows=256,
                 reward_dir=(OLD/'reward/screen256-r1').as_posix(),
                 utility_dir=(ROOT/'reward/residual-r1/screen').as_posix())
        input_paths += [OLD/'policy/screen-cache-r1/cache.json', OLD/'reward/screen256-r1/reward.json',
                        ROOT/'reward/residual-r1/screen/utility.json']
    if mode == 'screen':
        output = ROOT/'policy/screen-scientific3-r1'
        q.update(screen_evaluation_dir=output.as_posix(), maximum_seconds=600,
                 legacy_screen_outputs={'C': pin(OLD/'policy/screen-0-r1.json'),
                                        'RL-completion': pin(OLD/'policy/screen-1000-r1.json')})
        input_paths += [OLD/'policy/screen-0-r1.json', OLD/'policy/screen-1000-r1.json']
    elif mode == 'full':
        output = ROOT/'policy/full-scientific3-r1'
        q.update(full_evaluation_dir=output.as_posix())
    else:
        output = ROOT/'policy/rollout-scientific3-r1.json'
        q.update(rollout_output=output.as_posix(),
                 rollout_selection_pin=pin(OLD/'policy/rollout-selection-r1.json'),
                 preregistration=pin(ROOT/'protocols/preregistration.json'))
        input_paths += [OLD/'policy/rollout-selection-r1.json']
    q['output_paths'] = [output.as_posix()]
    q['required_output_bytes'] = {'D:': 150000000 if mode == 'full' else 50000000}
    for p in input_paths:
        q['input_hashes'][p.as_posix()] = sha(p)
    assert not output.exists(), 'Output must be fresh'
    target = ROOT / ('request-' + q['operation'] + '.json')
    with target.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(q, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(target.as_posix(), sha(target))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['screen', 'full', 'rollout'])
    parser.add_argument('--cap', type=int, choices=[8, 12], default=8)
    args = parser.parse_args()
    build(args.mode, args.cap)
