"""Root-native Level A replay of the packaged sources and compact evidence only.

stage=analyze, function=run. Numerical plotting happens only inside run after all
pins and the bundled stdlib verifier pass. No raw CSV, model or array is an input.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PREFIX = 'artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/'
MAX_JSON = 8 << 20
FORBIDDEN_KEYS = {'user_id', 'user_ids', 'userid', 'userids', 'users', 'row_index',
                  'row_indices', 'rows_by_user', 'per_query'}
FORBIDDEN_MODULES = ('torch', 'catboost', 'transformers')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError('Nonfinite JSON value: ' + value)


def read(path, maximum=MAX_JSON):
    path = Path(path)
    require(path.is_file() and path.suffix == '.json' and path.stat().st_size <= maximum,
            'Compact JSON input required')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def aggregate_only(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.casefold().replace('-', '_') not in FORBIDDEN_KEYS,
                    'Individual user/query data forbidden: ' + key)
            aggregate_only(child)
    elif isinstance(value, list):
        for child in value:
            aggregate_only(child)


def no_ml_modules():
    require(not any(name == root or name.startswith(root + '.') for name in sys.modules
                    for root in FORBIDDEN_MODULES), 'ML module present in aggregate-only worker')


def no_bytecode(report):
    require(not any(path.is_file() and (path.suffix.casefold() in ('.pyc', '.pyo') or
                    any(part.casefold() == '__pycache__' for part in path.parts))
                    for path in (report / 'code').rglob('*')), 'Bytecode cache is forbidden in packaged source')


def pin(entry, bindings, report, suffix, maximum):
    path = Path(entry['path']).resolve(strict=True)
    require(path.is_relative_to(report) and path.is_file() and path.suffix == suffix and
            path.stat().st_size <= maximum, 'Pinned file is not a small packaged input')
    digest = entry['sha256']
    require(re.fullmatch('[a-f0-9]{64}', digest) and bindings.get(path.as_posix()) == digest and
            sha(path) == digest, 'Pinned content or monitored binding differs: ' + path.name)
    return path, {'path': path.as_posix(), 'sha256': digest, 'bytes': path.stat().st_size}


def load(path, digest, name):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, 'Source changed before import')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load packaged source')
    module = importlib.util.module_from_spec(spec)
    # SourceFileLoader.exec_module may consume a timestamp/size-valid stale .pyc.
    # Execute exactly the same bytes whose digest was checked above.
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


def run(request, check=None):
    require(callable(check), 'Native Budget callback required')
    require(sys.dont_write_bytecode, 'Packaged replay requires Python -B')
    require(sys.pycache_prefix is None, 'External bytecode cache prefix is forbidden for packaged replay')
    require(all(request.get(key) == 0 for key in
                ('supervised_updates', 'rl_updates', 'new_rm_fits', 'final_test_reads')), 'Excluded ML work')
    no_ml_modules()
    check(stage='packaged-replay-validate')
    report = Path(request['report_root']).resolve(strict=True)
    require(report == (ROOT / 'reports').resolve(), 'Native replay report root differs')
    no_bytecode(report)
    manifest_path, manifest_pin = pin(request['bundle_manifest'], request['input_hashes'], report, '.json', 2 << 20)
    require(manifest_path == report / 'code-manifest.json', 'Expected packaged code manifest')
    manifest = read(manifest_path, 2 << 20)
    require(manifest.get('contract') == 'kion-small-source-bundle-v1', 'Unknown bundle manifest')
    by_path = {}
    for row in manifest['files']:
        relative = row['bundle_path']
        require(isinstance(relative, str) and not Path(relative).is_absolute() and '\\' not in relative,
                'Relative POSIX code path required')
        path = (report / relative).resolve()
        require(path.is_relative_to(report / 'code') and path not in by_path, 'Invalid/duplicate code member')
        by_path[path] = row

    verifier_path, verifier_pin = pin(request['verifier'], request['source_hashes'], report, '.py', 128 << 10)
    require(verifier_path == report / 'code/tools/verify_bundle.py' and
            by_path.get(verifier_path, {}).get('sha256') == verifier_pin['sha256'], 'Bundled verifier differs')
    own = Path(__file__).resolve()
    own_bundle = report / 'code/tlab-recsys-2026' / (SOURCE_PREFIX + own.name)
    own_digest = sha(own)
    require(request['source_hashes'].get(own.as_posix()) == own_digest and
            by_path.get(own_bundle, {}).get('sha256') == own_digest, 'Replay source must match packaged snapshot')
    verifier = load(verifier_path, verifier_pin['sha256'], '_kion_packaged_verifier')
    package_check = verifier.verify(report)
    require(package_check.get('state') == 'passed', 'Package verification failed')

    data_path, data_pin = pin(request['data'], request['input_hashes'], report, '.json', MAX_JSON)
    require(data_path.is_relative_to(report / 'evidence'), 'Aggregate must be in packaged evidence')
    data = read(data_path)
    aggregate_only(data)
    require(isinstance(data, dict) and data.get('state') == 'completed', 'Completed compact aggregate required')
    if data.get('contract') == 'kion-scientific3-analysis-v1':
        stage = data.get('stage')
        require(stage in ('screen', 'full4096'), 'Unknown aggregate cohort')
        renderer_name, contract = 'plot_results_v2.py', 'kion-scientific3-figures-v1'
        rows = data['rows']
    else:
        require(data.get('contract') == 'kion-final3-saved-liveprefix-summary-v1', 'Unknown compact aggregate contract')
        stage, renderer_name, contract, rows = 'rollout', 'plot_rollout.py', 'kion-rollout-figures-v1', data['position_count']
    folder = {'screen': 'screen', 'full4096': 'full', 'rollout': 'rollout'}[stage]
    output = Path(request['output']).resolve()
    require(output == report / 'replay-figures' / folder and not output.exists(), 'Fresh native cohort replay output required')
    plot_path, plot_pin = pin(request['plot_module'], request['source_hashes'], report, '.py', 128 << 10)
    expected_plot = report / 'code/tlab-recsys-2026' / (SOURCE_PREFIX + renderer_name)
    require(plot_path == expected_plot and by_path.get(plot_path, {}).get('sha256') == plot_pin['sha256'],
            'Only the packaged cohort renderer is allowed')
    dependency_pins = []
    if stage != 'rollout':
        base_path = plot_path.with_name('plot_results.py')
        expected = by_path.get(base_path, {}).get('sha256')
        require(isinstance(expected, str), 'Packaged frozen base renderer absent')
        _, base_pin = pin({'path': base_path.as_posix(), 'sha256': expected},
                          request['source_hashes'], report, '.py', 128 << 10)
        dependency_pins.append(base_pin)
    module = load(plot_path, plot_pin['sha256'], '_kion_packaged_cohort_plot')
    declared = getattr(module, 'DEPENDENCIES', {})
    require(declared == {row['path']: row['sha256'] for row in dependency_pins}, 'Renderer dependency closure differs')
    module.validate(data)
    labels, label_pin = None, None
    needed = set(module.needed_raw_ids(data))
    require(all(type(value) is int for value in needed) and len(needed) <= 100, 'Plotted-title scope is not compact')
    if stage != 'rollout':
        require('labels' in request and callable(getattr(module, 'load_labels', None)), 'Compact title labels required')
        label_path, label_pin = pin(request['labels'], request['input_hashes'], report, '.json', 1 << 20)
        require(label_path.is_relative_to(report / 'evidence') or label_path.is_relative_to(report / 'figures'),
                'Labels must be packaged compact evidence')
        label_data = read(label_path, 1 << 20)
        aggregate_only(label_data)
        labels = module.load_labels(label_path, data, input_path=data_path)
        require(isinstance(labels, dict) and set(labels) == {str(value) for value in needed} and
                all(value is None or isinstance(value, str) and value.strip() for value in labels.values()),
                'Title map differs from plotted IDs')
        require(not needed or any(value is not None for value in labels.values()), 'All plotted titles are absent')
    else:
        require(not needed and 'labels' not in request, 'Rollout renderer does not need an item catalog')
    no_ml_modules()
    no_bytecode(report)
    check(stage='packaged-replay-render-before', cohort=stage)
    result = module.render(data, output, labels=labels)
    check(stage='packaged-replay-render-after', cohort=stage)
    no_ml_modules()
    no_bytecode(report)
    require(result.get('contract') == contract and result.get('state') == 'completed' and
            result.get('stage') == stage and result.get('rows') == rows, 'Renderer result/cohort differs')
    versions = {name: getattr(sys.modules.get(name), '__version__', None) for name in ('numpy', 'matplotlib')}
    require(all(isinstance(value, str) and value for value in versions.values()), 'Actual plot versions unavailable')
    files = []
    for figure in result['figures']:
        stem = figure['file']
        require(re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}', stem) and set(figure['formats']) == {'svg', 'png'},
                'Unexpected figure path/format')
        for extension in ('svg', 'png'):
            path = (output / (stem + '.' + extension)).resolve(strict=True)
            require(path.parent == output and path.is_file() and path.stat().st_size <= 64 << 20,
                    'Figure output leaves replay folder or is oversized')
            files.append({'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    index = output / 'figures.json'
    index_data = read(index, 2 << 20)
    require(index_data.get('contract') == contract and index_data.get('state') == 'completed' and
            index_data.get('stage') == stage and index_data.get('rows') == rows and
            index_data.get('figures') == result['figures'], 'Figure index differs')
    require({path.name for path in output.iterdir()} == {row['path'] for row in files} | {'figures.json'},
            'Unexpected replay output member')
    require(sha(manifest_path) == manifest_pin['sha256'] and sha(data_path) == data_pin['sha256'] and
            (label_pin is None or sha(label_pin['path']) == label_pin['sha256']), 'Replay input changed')
    require(verifier.verify(report) == package_check and sha(own) == own_digest, 'Package/source changed during replay')
    record = {'contract': 'kion-packaged-aggregate-replay-v1', 'state': 'completed', 'stage': stage,
              'rows': rows, 'bundle_manifest': manifest_pin, 'package_verification': package_check,
              'data': data_pin, 'aggregate_canonical_sha256': canonical(data), 'labels': label_pin,
              'titles_requested': len(needed), 'titles_present': sum(value is not None for value in (labels or {}).values()),
              'plot_module': plot_pin, 'dependencies': dependency_pins, 'verifier': verifier_pin,
              'source': {'path': own.as_posix(), 'sha256': own_digest}, 'versions': versions,
              'output': output.as_posix(), 'files': files,
              'figures_index': {'path': index.as_posix(), 'sha256': sha(index), 'bytes': index.stat().st_size},
              'requires_visual_QA': True, 'rendered_visual_QA': False,
              'scope': 'Packaged code and compact aggregate/title-map replay only; no model inference or raw-data training.'}
    receipt = output / 'replay.json'
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return {**record, 'receipt': {'path': receipt.as_posix(), 'sha256': sha(receipt), 'bytes': receipt.stat().st_size}}
