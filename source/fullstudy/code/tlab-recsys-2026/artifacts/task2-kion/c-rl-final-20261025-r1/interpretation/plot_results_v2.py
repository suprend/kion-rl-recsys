"""Title-label adapter over the frozen original renderer; no numerical changes.

Only raw item-ID y-axis tick text changes before the original save/layout step.
Full exact titles and their CSV evidence belong to native-render.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
from pathlib import Path
import textwrap

BASE_SOURCE = Path(__file__).resolve().with_name('plot_results.py')
BASE_SHA256 = 'f6be0387a947dd33f065deae54fe98d14881e30271040389d743602afcd024f4'
DEPENDENCIES = {BASE_SOURCE.as_posix(): BASE_SHA256}
ARMS = ('C', 'RL-completion', 'RL-residual')


def base_module():
    if hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError('Frozen numerical plot source differs')
    spec = importlib.util.spec_from_file_location('_kion_frozen_plot_base', BASE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(data):
    return base_module().validate(data)


def needed_raw_ids(data):
    """Exactly the IDs displayed by original figures04/07; no catalog copy."""
    result = set()
    for weighting in ('query_weighted', 'equal_user_weighted'):
        mappings = {arm: {int(item['raw_id']): item['share'] for item in
                         data['summaries'][arm]['exposure'][weighting]['sorted_top10_exposure']} for arm in ARMS}
        ids = sorted(set().union(*(mapping for mapping in mappings.values())),
                     key=lambda raw_id: (-max(mapping.get(raw_id, 0.) for mapping in mappings.values()), raw_id))[:12]
        result.update(ids)
    if data['stage'] == 'screen':
        for comparison in ('RL-completion-minus-C', 'RL-residual-minus-C'):
            for utility in ('completion', 'residual'):
                result.update(int(item['raw_id']) for item in data['item_decomposition'][comparison][utility]
                              ['top_items_by_absolute_contribution'][:12])
    return result


def format_label(raw_id, title, width=34):
    prefix = str(int(raw_id)) + ' · '
    if not isinstance(title, str) or not title.strip():
        return prefix + 'название отсутствует'
    clean = ' '.join(title.split())
    short = textwrap.shorten(clean, width=max(10, 2 * width - len(prefix) - 3), placeholder='…')
    return textwrap.fill(prefix + short, width=width, max_lines=2, placeholder='…')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def load_labels(path, data, input_path=None):
    """Load only compact public labels; no catalog or pinned source files opened."""
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    if not (artifact.get('contract') == 'kion-plot-title-labels-v1' and artifact.get('state') == 'completed' and
            artifact.get('stage') == data['stage'] and artifact.get('aggregate_canonical_sha256') == canonical_sha(data)):
        raise ValueError('Compact labels/aggregate binding differs')
    if input_path is not None and hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != artifact['aggregate']['sha256']:
        raise ValueError('Original aggregate byte SHA differs')
    enrichment = dict(artifact['title_enrichment'])
    digest = enrichment.pop('canonical_sha256')
    if canonical_sha(enrichment) != digest:
        raise ValueError('Compact title enrichment SHA differs')
    if not enrichment['enabled']:
        if enrichment['labels']:
            raise ValueError('Disabled title enrichment contains labels')
        return None
    labels = {}
    for item in enrichment['labels']:
        raw_id, title = item['raw_id'], item['title']
        if not (type(raw_id) is int and (title is None or isinstance(title, str)) and
                item['missing_title'] == (title is None) and str(raw_id) not in labels):
            raise ValueError('Ambiguous compact title label')
        labels[str(raw_id)] = title
    if enrichment['requested_count'] != len(labels) or enrichment['found_title_count'] != sum(title is not None for title in labels.values()):
        raise ValueError('Compact title counts differ')
    if set(map(int, labels)) != needed_raw_ids(data):
        raise ValueError('Compact title scope differs from plotted IDs')
    return labels


def render(data, output, labels=None):
    base = base_module()
    original_save = base._save

    def title_save(plt, fig, folder, name, title, caption):
        if labels is not None:
            for axis in fig.axes:
                if axis.get_ylabel() == 'raw item ID':
                    raw_ids = [int(tick.get_text()) for tick in axis.get_yticklabels()]
                    axis.set_yticks(axis.get_yticks(), [format_label(raw_id, labels.get(str(raw_id))) for raw_id in raw_ids], fontsize=7.5)
                    axis.set_ylabel('Фильм · raw ID')
        return original_save(plt, fig, folder, name, title, caption)

    base._save = title_save
    result = base.render(data, output)
    return {**result, 'title_adapter': {'path': Path(__file__).resolve().as_posix(),
                                      'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                      'base_source': {'path': BASE_SOURCE.as_posix(), 'sha256': BASE_SHA256},
                                      'labels_enabled': labels is not None}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original aggregate plot-data.json')
    parser.add_argument('--output', type=Path, required=True, help='Fresh/empty figure directory')
    parser.add_argument('--labels', type=Path, help='Optional compact labels.json; no raw CSV needed')
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    labels = load_labels(args.labels, data, args.input) if args.labels else None
    result = render(data, args.output, labels=labels)
    print(json.dumps({'state': result['state'], 'figures': len(result['figures'])}, ensure_ascii=False))
