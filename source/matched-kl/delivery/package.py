"""Stdlib-only, explicit local delivery; no imports of experiment producers.

The draft/freeze commands never create archives. Only root may run build after
reviewing and approving a frozen selection. All numerical replay is external.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
from urllib.parse import unquote, urlsplit
import zipfile

Z = Path(__file__).resolve().parent.parent
HISTORY = Z.parent / 'c-rl-final-20261025-r1/delivery/kion-task2-research-20261025.zip'
HISTORY_SHA = '1fc9f26a3078465b430bcaa7da1454517577b532a0711234cf4dab154efc1a59'
HISTORY_MEMBER = 'historical/kion-task2-research-20261025.zip'
CONTRACT = 'kion-kl-seeds-delivery-selection-v1'
APPROVED = 'approved-by-root-after-final-QA'
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_MEMBERS = 500
PAIRED_HEADER = ['comparison', 'metric', 'difference', 'ci_low', 'ci_high',
                 'gains', 'losses', 'ties', 'queries', 'users']
REQUIRED = (
    'reports/reader.md', 'reports/technical.md', 'reports/reproduce.md',
    'reports/report-build.py', 'reports/figures.py', 'reports/FIGURES.md',
    'reports/test_figures.py', 'reports/figures-r1/train-kl-and-map.png',
    'reports/figures-r1/train-kl-and-map.svg', 'reports/figures-r1/manifest.json',
    'protocols/preregistration.json',
    'protocols/target.json', 'protocols/frozen-beta.json', 'protocols/matching.json',
    'protocols/scientific-arms.json', 'protocols/sources.json',
    'protocols/proposal-method.md', 'fit-ledger.json', 'campaign.json', 'state.json',
    'JOURNAL.md', 'prepare.py', 'next_request.py', 'final_requests.py', 'post_requests.py',
    'policy/policy.py', 'policy/evaluate.py', 'policy/METHOD.md',
    'policy/METHOD-evaluation.md', 'interpretation/analyze.py',
    'interpretation/verify.py', 'interpretation/test_verify.py', 'interpretation/VERIFY.md',
    'review/numeric-full4096.json', 'review/numeric-screen.json', 'review/visual-final.json',
    'control/common.py', 'control/launch.py', 'control/supervise.py', 'control/worker.py',
    'policy/train-target-r1.json', 'policy/train-final-six-r1.json',
    'policy/screen-six-r1/evaluation.json', 'policy/full-six-r1/evaluation.json',
    'interpretation/screen-r1/analysis.json', 'interpretation/screen-r1/plot-data.json',
    'interpretation/screen-r1/paired-metrics.csv',
    'interpretation/full-r1/analysis.json', 'interpretation/full-r1/plot-data.json',
    'interpretation/full-r1/paired-metrics.csv', 'delivery/package.py',
    'delivery/DELIVERY.md',
)
OPTIONAL = (
    'reports/reader.pdf', 'reports/reader-r1.pdf', 'reports/supplement.pdf', 'reports/technical.pdf',
    'reports/reproduce.pdf', 'reports/report.pdf',
    'policy/test_policy.py', 'policy/test_evaluate.py', 'interpretation/test_analyze.py',
    'protocols/pilot1-decision.json',
)
FINAL_OPS = ('train-target-r1', 'train-final-six-r1', 'screen-six-r1',
             'full-six-r1', 'analyze-screen-r1', 'analyze-full-r1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def read_json(path):
    require(path.stat().st_size <= MAX_FILE, 'JSON metadata too large: ' + path.as_posix())
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def relative(value):
    require(isinstance(value, str) and value and '\\' not in value, 'POSIX member path required')
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(p not in ('', '.', '..') for p in value.split('/')),
            'Unsafe member path: ' + value)
    require(not any(':' in part or part.endswith((' ', '.')) for part in path.parts),
            'Unsafe Windows member path: ' + value)
    require(path.as_posix() == value, 'Noncanonical member path: ' + value)
    return value


def checked_source(member, kind='followup'):
    relative(member)
    if kind == 'historical_archive':
        require(member == HISTORY_MEMBER, 'Only the fixed historical archive is allowed')
        source = HISTORY
        boundary = HISTORY.parent
    else:
        require(kind == 'followup', 'Unknown source kind')
        source, boundary = Z / member, Z
        require(source.suffix.lower() in ('.md', '.json', '.py', '.csv', '.pdf', '.png', '.svg'),
                'Forbidden source suffix: ' + member)
        require(source.suffix.lower() not in ('.png', '.svg') or member in (
            'reports/figures-r1/train-kl-and-map.png',
            'reports/figures-r1/train-kl-and-map.svg'), 'Only the exact compact report figures are allowed')
        require(not any(part.casefold() in ('models', 'cache', 'caches', '__pycache__', '.git',
                    'datasets', 'raw', 'secrets') for part in PurePosixPath(member).parts),
                'Forbidden source tree: ' + member)
        require(source.suffix.lower() != '.csv' or member in (
            'interpretation/screen-r1/paired-metrics.csv',
            'interpretation/full-r1/paired-metrics.csv'), 'Only aggregate paired CSV is allowed')
    # Resolve containment as well as rejecting linked parent directories/junctions.
    require(source.resolve().is_relative_to(boundary.resolve()), 'Source escapes its root: ' + member)
    for part in (source, *source.parents):
        if part == boundary.parent:
            break
        require(not part.is_symlink() and not (hasattr(part, 'is_junction') and part.is_junction()),
                'Linked source path: ' + part.as_posix())
    return source


def payload_for(entry):
    source = checked_source(entry['path'], entry['source_kind'])
    require(source.is_file(), 'Missing source: ' + entry['path'])
    require(source.stat().st_size <= MAX_FILE, 'Source too large: ' + entry['path'])
    payload = source.read_bytes()
    require(len(payload) == entry['bytes'] and digest(payload) == entry['sha256'],
            'Pinned source changed: ' + entry['path'])
    if entry['source_kind'] == 'historical_archive':
        require(digest(payload) == HISTORY_SHA, 'Historical archive differs from its immutable pin')
    if source.suffix.lower() == '.csv':
        rows = list(csv.reader(io.StringIO(payload.decode('utf-8'))))
        require(rows and rows[0] == PAIRED_HEADER and len(rows) <= 1000,
                'Expected aggregate paired-metrics CSV header/size')
        require(all(len(row) == len(PAIRED_HEADER) for row in rows), 'Ragged paired-metrics CSV')
    return payload


def operation_name(value):
    require(isinstance(value, str) and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', value),
            'Unsafe operation name')
    return value


def source_json(member):
    source = checked_source(member)
    require(source.is_file(), 'Missing metadata: ' + member)
    return read_json(source)


def candidates():
    """Bounded metadata names; never recurse through data/model/cache trees."""
    paths = {value: True for value in REQUIRED}
    paths.update({value: False for value in OPTIONAL})
    campaign = source_json('campaign.json')
    ledger = source_json('fit-ledger.json')
    admitted = campaign.get('admitted_requests', {})
    require(isinstance(admitted, dict), 'Invalid admitted request ledger')
    operations = set(FINAL_OPS) | {operation_name(op) for op in admitted}
    reservations = ledger.get('reservations', [])
    require(isinstance(reservations, list) and len(reservations) <= 8, 'Invalid fit reservation count')
    for ordinal, item in enumerate(reservations, 1):
        op = operation_name(item['operation'])
        require(item['fit_ordinal'] == ordinal, 'Nonsequential fit ledger')
        operations.add(op)
        paths[f'protocols/fit-ledger-{ordinal:02d}.json'] = True
        receipt = checked_source(f'operations/{op}/receipt.json')
        succeeded = receipt.is_file() and source_json(f'operations/{op}/receipt.json').get('state') == 'succeeded'
        paths[f'policy/{op}/training.json'] = succeeded  # Failed fits may have no result.
    for op in sorted(operations):
        mandatory = op in admitted or op in FINAL_OPS
        paths[f'request-{op}.json'] = mandatory
        paths[f'operations/{op}/receipt.json'] = mandatory
        receipt = checked_source(f'operations/{op}/receipt.json')
        succeeded = receipt.is_file() and source_json(f'operations/{op}/receipt.json').get('state') == 'succeeded'
        # A killed/failed worker may never write its own completion; retain its
        # supervisor receipt without inventing a completion file.
        paths[f'operations/{op}/stage-completion.json'] = op in FINAL_OPS or succeeded
        if op.startswith('kl-'):
            paths[f'policy/{op}.json'] = mandatory
    # Reviews are compact metadata, selected by a bounded, explicit filename list.
    for name in ('preregistration-science.json', 'target-science.json', 'metadata-planner.json',
                 'control-clone.json', 'train-target-admission.json', 'method-skeptic.md',
                 'report-source.json', 'report-final.json', 'report-visual.json', 'pdf-visual.json',
                 'delivery-source-review.json', 'pilot1-science.json', 'pilot2-science.json',
                 'pilot1-admission.json', 'pilot2-admission.json', 'final-factory.json',
                 'eval-factory.json', 'root-source-tests.json', 'post-factory.json',
                 'prospective-admission.json', 'evaluation-admission.json',
                 'beta-freeze-science.json', 'matching-science.json', 'results-science.json',
                 'source-validation.json'):
        paths['review/' + name] = False
    return paths, operations


def markdown_links(files):
    delivered = {entry['path'] for entry in files}
    unresolved, external, checked = [], [], 0
    for entry in files:
        if not entry['path'].endswith('.md'):
            continue
        text = payload_for(entry).decode('utf-8')
        text = re.sub(r'```[^\n]*\n.*?```', '', text, flags=re.S)
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', text):
            target = match[1].strip().strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                external.append({'document': entry['path'], 'target': target})
                continue
            if not parsed.path:
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(entry['path']), unquote(parsed.path)))
            checked += 1
            if resolved not in delivered:
                unresolved.append({'document': entry['path'], 'target': target, 'resolved_member': resolved,
                                   'reason': 'Local workspace dependency; not a delivered member'})
    return {'scope': 'inline Markdown links outside fenced code; anchors and code-path mentions not validated',
            'checked_relative_links': checked, 'unresolved': unresolved, 'external_links': external}


def readiness(files, operations):
    """Metadata completion checks, never a numerical correctness/replay claim."""
    errors = []
    paths = {entry['path']: entry for entry in files}
    def check(test, message):
        if not test:
            errors.append(message)
    for name in ('reader', 'technical', 'reproduce'):
        member = f'reports/{name}.md'
        if member in paths:
            text = payload_for(paths[member]).decode('utf-8')
            check(not re.search(r'\bPENDING\b|\{\{[^}]+\}\}', text), 'Pending report: ' + member)
    for op in sorted(operations):
        member = f'operations/{op}/receipt.json'
        if member not in paths:
            continue
        receipt = json.loads(payload_for(paths[member]))
        check(receipt.get('state') in ('succeeded', 'failed', 'stop_requested', 'time_limit',
                    'host_reserve', 'memory_cap', 'disk_or_pin_guard', 'supervisor_error'),
              'Nonterminal receipt: ' + op)
        check(receipt.get('cleanup_quiescent') is True, 'Unconfirmed cleanup: ' + op)
        if op in FINAL_OPS:
            check(receipt.get('state') == 'succeeded', 'Required operation did not succeed: ' + op)
        completion_member = f'operations/{op}/stage-completion.json'
        if receipt.get('state') == 'succeeded' and completion_member in paths:
            completion = json.loads(payload_for(paths[completion_member]))
            check(completion.get('state') == 'completed', 'Worker completion differs: ' + op)
    for member in ('policy/train-target-r1.json', 'policy/train-final-six-r1.json',
                   'policy/screen-six-r1/evaluation.json', 'policy/full-six-r1/evaluation.json',
                   'interpretation/screen-r1/analysis.json', 'interpretation/full-r1/analysis.json'):
        if member in paths:
            data = json.loads(payload_for(paths[member]))
            check(data.get('state') == 'completed', 'Incomplete evidence: ' + member)
    for stage, rows, member in (
            ('full4096', 4096, 'review/numeric-full4096.json'),
            ('screen', 256, 'review/numeric-screen.json')):
        if member in paths:
            proof = json.loads(payload_for(paths[member]))
            check(proof.get('contract') == 'kion-independent-csv-numeric-verification-v1'
                  and proof.get('state') == 'completed' and proof.get('verified') is True
                  and proof.get('stage') == stage and proof.get('rows') == rows
                  and proof.get('independent_metric_recomputation') is True
                  and proof.get('bootstrap_intervals_independently_resampled') is False,
                  'Required independent compact numeric proof is not accepted: ' + member)
            part = 'full' if stage == 'full4096' else 'screen'
            bindings = {
                'analysis': f'interpretation/{part}-r1/analysis.json',
                'evaluation_metadata': f'policy/{part}-six-r1/evaluation.json',
                'matching_receipt': 'protocols/matching.json',
                'train_reference_metadata': 'policy/train-target-r1.json',
                'six_arm_train_diagnostics': 'policy/train-final-six-r1.json',
            }
            pins = proof.get('inputs') if isinstance(proof.get('inputs'), dict) else {}
            for key, delivered in bindings.items():
                pin = pins.get(key)
                expected = paths.get(delivered)
                check(isinstance(pin, dict) and expected is not None
                      and pin.get('path') == (Z / delivered).as_posix()
                      and pin.get('sha256') == expected['sha256']
                      and pin.get('bytes') == expected['bytes'],
                      f'Numeric proof does not bind current delivered {key}: {member}')
    return errors


def selection(draft):
    paths, operations = candidates()
    files, missing = [], []
    for member, required in sorted(paths.items()):
        source = checked_source(member)
        if not source.is_file():
            if required:
                missing.append(member)
            continue
        require(source.stat().st_size <= MAX_FILE, 'Source too large: ' + member)
        payload = source.read_bytes()
        files.append({'path': member, 'source_kind': 'followup', 'bytes': len(payload),
                      'sha256': digest(payload)})
    source = checked_source(HISTORY_MEMBER, 'historical_archive')
    require(source.is_file() and source.stat().st_size <= MAX_FILE, 'Historical archive unavailable')
    payload = source.read_bytes()
    require(digest(payload) == HISTORY_SHA, 'Immutable historical ZIP differs')
    files.append({'path': HISTORY_MEMBER, 'source_kind': 'historical_archive',
                  'bytes': len(payload), 'sha256': HISTORY_SHA})
    files.sort(key=lambda entry: entry['path'])
    errors = readiness(files, operations)
    require(len(files) + 2 <= MAX_MEMBERS and sum(f['bytes'] for f in files) <= MAX_TOTAL,
            'Delivery exceeds compact size/member limits')
    if not draft:
        require(not missing, 'Missing required evidence: ' + ', '.join(missing))
        require(not errors, 'Not ready: ' + '; '.join(errors))
    return {'contract': CONTRACT, 'state': 'draft-results-pending' if draft else 'frozen-awaiting-root-approval',
            'followup_root': Z.as_posix(), 'archive_name': 'kion-task2-kl-seeds-20261025.zip',
            'scope': 'Level A compact evidence/report rebuild; unchanged historical support/code ZIP. '
                     'Not portable numerical ML replay. Absolute request pins are provenance only.',
            'operations': sorted(operations), 'missing_required': missing,
            'readiness_errors': errors, 'markdown_references': markdown_links(files),
            'external_dependencies': ['Numerical replay requires separately available frozen weights, '
                'raw datasets/features, caches/arrays and pinned native workspace/runtime.',
                'PDF rebuild requires ReportLab and an available Cyrillic TTF font.',
                'Historical support/code is nested unchanged; its internal manifest describes its own scope.'],
            'files': files}


def validate_selection(value):
    require(value.get('contract') == CONTRACT and value.get('state') == APPROVED,
            'Root approval of the final frozen selection is required')
    require(Path(value['followup_root']).resolve() == Z, 'Selection belongs to another followup root')
    require(not value.get('missing_required') and not value.get('readiness_errors'), 'Selection is not ready')
    require(value.get('root_review') and isinstance(value['root_review'], str), 'Root review note required')
    current = selection(False)
    require(set(value) == set(current) | {'root_review'}, 'Unexpected frozen selection fields')
    for key in current.keys() - {'state'}:
        require(value.get(key) == current[key], 'Frozen selection differs from current ' + key)
    # Only these exact names may be built; selection cannot opt in extra sources.
    require(value.get('archive_name') == current['archive_name'], 'Unexpected archive filename')
    return current


def build(selection_path, output_dir):
    require(selection_path.is_file(), 'Selection file missing')
    selection_payload = selection_path.read_bytes()
    selected = json.loads(selection_payload)
    current = validate_selection(selected)
    files = selected['files']
    payloads = {entry['path']: payload_for(entry) for entry in files}
    require(not output_dir.exists() and output_dir.parent.is_dir(), 'Fresh output directory required')
    require(output_dir.resolve().is_relative_to((Z / 'delivery').resolve()),
            'Build directory must remain below this followup delivery directory')
    selection_member = {'path': 'DELIVERY-SELECTION.json', 'bytes': len(selection_payload),
                        'sha256': digest(selection_payload)}
    manifest = {'contract': 'kion-kl-seeds-package-manifest-v1',
                'scope': current['scope'], 'selection_sha256': digest(selection_payload),
                'files': [{key: item[key] for key in ('path', 'bytes', 'sha256')} for item in files] + [selection_member],
                'markdown_references': current['markdown_references'],
                'external_dependencies': current['external_dependencies'],
                'excluded': ['models/weights', '.npy/.npz arrays', 'per-query.csv/raw user IDs',
                             'raw datasets/features', 'caches', 'secrets', 'Git trees'],
                'historical_archive_sha256': HISTORY_SHA}
    manifest_payload = encode(manifest)
    payloads['DELIVERY-SELECTION.json'] = selection_payload
    payloads['PACKAGE-MANIFEST.json'] = manifest_payload
    require(sum(len(payload) for payload in payloads.values()) <= MAX_TOTAL,
            'Delivery plus embedded manifests exceeds compact size limit')
    require(selection_path.read_bytes() == selection_payload, 'Selection changed before build')
    output_dir.mkdir(exist_ok=False)
    archive = output_dir / selected['archive_name']
    # Exclusive creation plus fixed timestamps and ordering: same sources => same ZIP bytes.
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(archive, 'r') as bundle:
        require(len(bundle.infolist()) == len(payloads) and set(bundle.namelist()) == set(payloads),
                'ZIP member set differs or duplicates exist')
        require(bundle.testzip() is None, 'ZIP CRC check failed')
        for name, payload in payloads.items():
            actual = bundle.read(name)  # Reads each member fully; zipfile also checks its CRC.
            require(len(actual) == len(payload) and digest(actual) == digest(payload),
                    'ZIP readback differs: ' + name)
    for entry in files:
        payload_for(entry)  # Source changes during write/readback invalidate this run.
    require(selection_path.read_bytes() == selection_payload, 'Selection changed during build')
    (output_dir / 'PACKAGE-MANIFEST.json').write_bytes(manifest_payload)
    receipt = {'contract': 'kion-kl-seeds-delivery-verification-v1', 'state': 'passed',
               'archive': {'path': archive.as_posix(), 'bytes': archive.stat().st_size,
                           'sha256': digest(archive.read_bytes())},
               'selection_sha256': digest(selection_payload), 'member_count': len(payloads),
               'all_members_read_back_crc_and_sha256': True,
               'historical_archive_unchanged': True, 'no_upload_email_or_publish': True,
               'numerical_replay_verified': False}
    (output_dir / 'delivery.json').write_bytes(encode(receipt))
    return receipt


def write_new(path, payload):
    require(path.parent.is_dir(), 'Output parent directory missing')
    with path.open('xb') as stream:
        stream.write(payload)


def main():
    require(sys.dont_write_bytecode, 'Run with python -B')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('draft', 'freeze'):
        command = sub.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('check')
    command.add_argument('--selection', type=Path, required=True)
    command = sub.add_parser('build')
    command.add_argument('--selection', type=Path, required=True)
    command.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.action in ('draft', 'freeze'):
        result = selection(args.action == 'draft')
        write_new(args.output.resolve(), encode(result))
        print(json.dumps({'state': result['state'], 'files': len(result['files']),
                          'missing_required': result['missing_required'],
                          'readiness_errors': result['readiness_errors']}, ensure_ascii=False))
    elif args.action == 'check':
        validate_selection(read_json(args.selection))
        print(json.dumps({'state': 'approved-selection-current', 'archive_created': False}))
    else:
        print(json.dumps(build(args.selection.resolve(), args.output_dir.resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
