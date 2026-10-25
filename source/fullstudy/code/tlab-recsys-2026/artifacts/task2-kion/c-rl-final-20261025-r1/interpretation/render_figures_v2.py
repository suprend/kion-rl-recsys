"""Root-native aggregate figure entrypoint; no numerical import on module load.

Controller: stage=analyze, function=run. Request adds data={path,sha256},
plot_module={path,sha256}, output=ROOT/reports/figures-v2/{screen,full}; matching
input_hashes/source_hashes and the controller's Budget callback are mandatory.
Optional items={path,sha256} enriches only plotted IDs from public item_id/title CSV.
Actual versions, title evidence and pins go into native-render.json and the receipt.
"""
from __future__ import annotations

import hashlib
import importlib.util
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PLOT_SOURCE = Path(__file__).resolve().with_name('plot_results.py')
V2_PLOT_SOURCE = Path(__file__).resolve().with_name('plot_results_v2.py')
V3_PLOT_SOURCE = Path(__file__).resolve().with_name('plot_results_v3.py')
ROLLOUT_PLOT_SOURCE = Path(__file__).resolve().with_name('plot_rollout.py')
MAX_JSON_BYTES = 8 * 1024 * 1024
FORBIDDEN_KEYS = {'user_id', 'user_ids', 'userid', 'userids', 'users', 'row_index',
                  'row_indices', 'rows_by_user', 'per_query'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(entry, bindings, suffix, maximum=None):
    path = Path(entry['path']).resolve(strict=True)
    require(path.is_file() and path.suffix.lower() == suffix, 'Unexpected pinned file type')
    require(maximum is None or path.stat().st_size <= maximum, 'Pinned input exceeds compact input limit')
    digest = entry['sha256']
    require(bindings.get(path.as_posix()) == digest and sha(path) == digest,
            'Pin or monitored binding differs: ' + path.as_posix())
    return path, {'path': path.as_posix(), 'sha256': digest, 'bytes': path.stat().st_size}


def reject_constant(value):
    raise ValueError('Nonfinite JSON value: ' + value)


def aggregate_only(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.casefold().replace('-', '_') not in FORBIDDEN_KEYS, 'Individual row/user binding forbidden: ' + key)
            aggregate_only(child)
    elif isinstance(value, list):
        for child in value:
            aggregate_only(child)


def title_lookup(path, needed):
    """Stream public CSV; retain exactly the selected raw-ID/title fields."""
    titles = {str(raw_id): None for raw_id in needed}
    found = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream)
        header = next(rows)
        require(header.count('item_id') == header.count('title') == 1, 'Unique item_id/title CSV columns required')
        id_column, title_column = header.index('item_id'), header.index('title')
        for row in rows:
            require(len(row) > max(id_column, title_column), 'Malformed item CSV row')
            raw_id = int(row[id_column])
            if raw_id in needed:
                require(raw_id not in found, 'Duplicate selected item_id in title CSV')
                found.add(raw_id)
                title = row[title_column]
                titles[str(raw_id)] = title if title.strip() else None
    return titles


def run(request, check=None):
    require(callable(check), 'Native Budget callback required')
    check(stage='native-figure-input-integrity')
    data_path, data_pin = pinned(request['data'], request['input_hashes'], '.json', MAX_JSON_BYTES)
    plot_path, plot_pin = pinned(request['plot_module'], request['source_hashes'], '.py', 128 * 1024)
    data = json.loads(data_path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    require(isinstance(data, dict) and data.get('state') == 'completed', 'Completed aggregate required')
    aggregate_only(data)
    if data.get('contract') == 'kion-scientific3-analysis-v1':
        stage = data.get('stage')
        require(stage in ('screen', 'full4096'), 'Unknown aggregate cohort')
        allowed = {V3_PLOT_SOURCE.resolve()}
        figure_contract = 'kion-scientific3-figures-v1'
        rows, unit = data['rows'], 'queries'
    else:
        require(data.get('contract') == 'kion-final3-saved-liveprefix-summary-v1', 'Unknown aggregate contract')
        stage, figure_contract = 'rollout', 'kion-rollout-figures-v1'
        allowed = {ROLLOUT_PLOT_SOURCE.resolve()}
        rows, unit = data['position_count'], 'contexts'
    require(plot_path in allowed and plot_path.is_relative_to(ROOT.resolve()), 'Only root-local reviewed plot source allowed')
    require(stage in ('screen', 'full4096'), 'Layout-v2 render supports only screen/full; rollout figures remain unchanged')
    output = Path(request['output']).resolve()
    folder = {'screen': 'screen', 'full4096': 'full', 'rollout': 'rollout'}[stage]
    expected = (ROOT / 'reports' / 'figures-v2' / folder).resolve()
    require(output == expected and output.is_relative_to(ROOT.resolve()) and not output.exists(),
            'Fresh root-local cohort figure output required')
    spec = importlib.util.spec_from_file_location('_kion_native_pinned_plot_' + plot_pin['sha256'][:12], plot_path)
    require(spec is not None and spec.loader is not None, 'Cannot load pinned plot source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dependencies = [pinned({'path': path, 'sha256': digest}, request['source_hashes'], '.py', 128 * 1024)[1]
                    for path, digest in getattr(module, 'DEPENDENCIES', {}).items()]
    module.validate(data)
    labels, item_pin = None, None
    if 'items' in request:
        require(callable(getattr(module, 'needed_raw_ids', None)), 'Renderer does not support title enrichment')
        needed = set(module.needed_raw_ids(data))
        require(all(type(raw_id) is int for raw_id in needed) and len(needed) <= 100, 'Title scope is not compact plotted IDs')
        item_path, item_pin = pinned(request['items'], request['input_hashes'], '.csv', 40 * 1024 * 1024)
        labels = title_lookup(item_path, needed)
    check(stage='native-figure-render-before', cohort=stage)
    result = module.render(data, output, labels=labels) if labels is not None else module.render(data, output)
    check(stage='native-figure-render-after', cohort=stage)
    require(result.get('contract') == figure_contract and result.get('state') == 'completed' and
            result.get('stage') == stage and result.get('rows') == rows, 'Figure result/cohort differs')
    versions = {name: getattr(sys.modules.get(name), '__version__', None) for name in ('numpy', 'matplotlib')}
    require(all(isinstance(version, str) and version for version in versions.values()), 'Actual numerical/plot versions unavailable')
    # Rehash small inputs after render in addition to native Budget's stat checks.
    require(sha(data_path) == data_pin['sha256'] and sha(plot_path) == plot_pin['sha256'], 'Pinned render input changed')
    require(all(sha(pin['path']) == pin['sha256'] for pin in dependencies) and
            (item_pin is None or sha(item_pin['path']) == item_pin['sha256']), 'Title/dependency input changed')
    index = output / 'figures.json'
    require(index.is_file(), 'Renderer did not save figures.json')
    enrichment = {'enabled': labels is not None, 'items': item_pin,
                  'requested_count': len(labels or {}),
                  'found_title_count': sum(title is not None for title in (labels or {}).values()),
                  'labels': [{'raw_id': int(raw_id), 'title': title,
                              'missing_title': title is None,
                              'display_label': module.format_label(int(raw_id), title) if hasattr(module, 'format_label') else None}
                             for raw_id, title in sorted((labels or {}).items(), key=lambda pair: int(pair[0]))]}
    enrichment['canonical_sha256'] = hashlib.sha256(json.dumps(enrichment, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    record = {'contract': 'kion-native-figure-render-v1', 'state': 'completed', 'stage': stage,
            'rows': rows, 'row_unit': unit, 'unique_users': data.get('unique_users'), 'output': output.as_posix(),
            'data': data_pin, 'plot_module': plot_pin, 'dependencies': dependencies, 'title_enrichment': enrichment,
            'aggregate_canonical_sha256': hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest(),
            'source': {'path': Path(__file__).resolve().as_posix(), 'sha256': sha(__file__)},
            'versions': versions, 'figures_index': {'path': index.as_posix(), 'sha256': sha(index), 'bytes': index.stat().st_size},
            'figure_count': len(result['figures']), 'requires_visual_QA': True}
    if stage == 'rollout':
        record.update(position_count=data['position_count'], trajectory_count=data['trajectory_count'])
    labels_path = output / 'labels.json'
    with labels_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'contract': 'kion-plot-title-labels-v1', 'state': 'completed', 'stage': stage,
                   'aggregate': data_pin, 'aggregate_canonical_sha256': record['aggregate_canonical_sha256'],
                   'title_enrichment': enrichment}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    record['labels_artifact'] = {'path': labels_path.as_posix(), 'sha256': sha(labels_path), 'bytes': labels_path.stat().st_size}
    manifest = output / 'native-render.json'
    require(not manifest.exists(), 'Native render manifest already exists')
    with manifest.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return {**record, 'manifest': {'path': manifest.as_posix(), 'sha256': sha(manifest), 'bytes': manifest.stat().st_size}}
