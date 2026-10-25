"""Root-only minimal postprocessing request builder; never admits or executes jobs."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'c-rl-20261024-r1'
PAS = Path('D:/personal-agent-system')
RUNTIME = ROOT.parent/'ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pin(p):
    p = Path(p).resolve()
    return {'path': p.as_posix(), 'sha256': sha(p)}


def build(mode):
    if mode in ('screen', 'full'):
        module = ROOT/'interpretation/analyze.py'
        result = ROOT/('policy/'+mode+'-scientific3-r1/evaluation.json')
        protocol = ROOT/'protocols/evaluation.json'
        output = ROOT/('interpretation/'+mode+'-analysis-r1')
        fields = {'evaluation': pin(result), 'protocol': pin(protocol), 'output': output.as_posix()}
        inputs = [result, protocol]
        if mode == 'screen':
            curve = [{'step': s, **pin(OLD/('policy/screen-'+str(s)+'-r1.json'))} for s in [0, 250, 500, 1000]]
            fields['checkpoint_curve'] = curve
            inputs += [Path(p['path']) for p in curve]
    elif mode == 'rollout':
        module = ROOT/'interpretation/rollout_summary.py'
        result = ROOT/'policy/rollout-scientific3-r1.json'
        catalog = Path('D:/tlab-recsys-2026/data/raw/kion/full/items.csv')
        selection = json.loads((OLD/'policy/rollout-selection-r1.json').read_text(encoding='utf-8'))
        # The fixed request carries the already-registered order; never select by outcomes.
        positions = json.loads((ROOT/'request-rollout-scientific3-r1.json').read_text(encoding='utf-8'))['rollout_positions']
        output = ROOT/'interpretation/rollout-summary-r1'
        fields = {'rollout': pin(result), 'items': pin(catalog), 'expected_positions': positions,
                  'output': output.as_posix(), 'seed': 20261024}
        inputs = [result, catalog, OLD/'policy/rollout-selection-r1.json']
    else:
        raise ValueError(mode)
    assert not output.exists(), 'Fresh output required'
    sources = [module, PAS/'scripts/portability.py', RUNTIME]
    sources += [ROOT/'control'/n for n in ['common.py', 'launch.py', 'supervise.py', 'worker.py']]
    op = mode+'-analysis-r1'
    req = dict(contract='kion-c-rl-native-v1', stage='analyze', operation=op,
               deadline_utc='2026-10-25T05:45:00+00:00', maximum_seconds=900,
               cpu_threads=4, reserve_gib=2, cap_gib=4,
               supervised_updates=0, new_rm_fits=0, final_test_reads=0, rl_updates=0,
               module_path=module.as_posix(), function='run',
               owned_output_roots=[(ROOT/'interpretation').as_posix()], output_paths=[output.as_posix()],
               required_output_bytes={'D:': 15000000},
               source_hashes={p.as_posix(): sha(p) for p in sources},
               input_hashes={p.as_posix(): sha(p) for p in inputs}, **fields)
    path = ROOT/('request-'+op+'.json')
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(req, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(path.as_posix(), sha(path))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['screen', 'full', 'rollout'])
    build(parser.parse_args().mode)
