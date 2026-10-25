"""Root-only metadata builder, called after source bundle and compact evidence freeze.

Requests pin the finalized manifest; replay requests stay outside that manifest
to avoid a self-referential hash cycle. Never admits or launches any operation.
"""
import hashlib
import json
from pathlib import Path

Z = Path(__file__).resolve().parent
REPORT = Z / 'reports'
SOURCE = REPORT / 'code/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation'
PAS = Path('D:/personal-agent-system')
RUNTIME = Z.parent / 'ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return {'path': path.as_posix(), 'sha256': sha(path)}


for mode in ('screen', 'full', 'rollout'):
    module = Z / 'interpretation/replay_portable_v3.py'
    renderer = SOURCE / ('plot_rollout.py' if mode == 'rollout' else 'plot_results_v4.py')
    verifier = REPORT / 'code/tools/verify_bundle.py'
    manifest = REPORT / 'code-manifest.json'
    data = REPORT / 'evidence' / {'screen': 'plot-data-screen.json', 'full': 'plot-data-full.json',
                                'rollout': 'rollout-summary.json'}[mode]
    output = REPORT / 'replay-figures' / mode
    assert not output.exists()
    sources = [module, renderer, verifier, PAS / 'scripts/portability.py', RUNTIME]
    sources += [Z / 'control' / name for name in ('common.py', 'launch.py', 'supervise.py', 'worker.py')]
    inputs = [manifest, data]
    fields = dict(report_root=REPORT.as_posix(), bundle_manifest=pin(manifest), verifier=pin(verifier),
                  plot_module=pin(renderer), data=pin(data), output=output.as_posix())
    if mode != 'rollout':
        sources.append(SOURCE / 'plot_results.py')
        labels = REPORT / 'figures-v3' / mode / 'labels.json'
        fields['labels'] = pin(labels)
        inputs.append(labels)
    operation = f'replay-{mode}-r1'
    request = dict(contract='kion-c-rl-native-v1', stage='analyze', operation=operation,
        deadline_utc='2026-10-25T05:45:00+00:00', maximum_seconds=900,
        cpu_threads=4, reserve_gib=2, cap_gib=4,
        supervised_updates=0, new_rm_fits=0, final_test_reads=0, rl_updates=0,
        module_path=module.as_posix(), function='run',
        owned_output_roots=[(REPORT / 'replay-figures').as_posix()], output_paths=[output.as_posix()],
        required_output_bytes={'D:': 40000000},
        source_hashes={p.as_posix(): sha(p) for p in sources},
        input_hashes={p.as_posix(): sha(p) for p in inputs}, **fields)
    path = Z / f'request-{operation}.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(mode, sha(path))
