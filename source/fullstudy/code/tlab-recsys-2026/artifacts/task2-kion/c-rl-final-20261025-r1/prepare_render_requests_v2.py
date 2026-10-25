"""Root metadata-only final layout requests; original figures and sources preserved."""
import hashlib
import json
from pathlib import Path

Z = Path(__file__).resolve().parent
PAS = Path('D:/personal-agent-system')
RUNTIME = Z.parent / 'ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return {'path': path.as_posix(), 'sha256': sha(path)}


for mode in ('screen', 'full'):
    module = Z / 'interpretation/render_figures_v2.py'
    renderer = Z / 'interpretation/plot_results_v3.py'
    data = Z / ('interpretation/screen-analysis-r2/plot-data.json' if mode == 'screen' else 'interpretation/full-analysis-r1/plot-data.json')
    output = Z / 'reports/figures-v2' / mode
    items = Path('D:/tlab-recsys-2026/data/raw/kion/full/items.csv')
    assert not output.exists()
    sources = [module, renderer, Z / 'interpretation/plot_results.py', PAS / 'scripts/portability.py', RUNTIME]
    sources += [Z / 'control' / name for name in ('common.py', 'launch.py', 'supervise.py', 'worker.py')]
    operation = f'render-{mode}-r2'
    request = dict(contract='kion-c-rl-native-v1', stage='analyze', operation=operation,
        deadline_utc='2026-10-25T05:45:00+00:00', maximum_seconds=900,
        cpu_threads=4, reserve_gib=2, cap_gib=4,
        supervised_updates=0, new_rm_fits=0, final_test_reads=0, rl_updates=0,
        module_path=module.as_posix(), function='run',
        owned_output_roots=[(Z / 'reports/figures-v2').as_posix()], output_paths=[output.as_posix()],
        required_output_bytes={'D:': 40000000},
        source_hashes={p.as_posix(): sha(p) for p in sources},
        input_hashes={p.as_posix(): sha(p) for p in [data, items]},
        data=pin(data), plot_module=pin(renderer), items=pin(items), output=output.as_posix())
    path = Z / f'request-{operation}.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(mode, sha(path))
