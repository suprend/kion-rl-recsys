"""Bounded, stdlib-only GPT input inventory. Never loads an ML payload.

The checked-in profile is a finite list of JSON pointers into byte-pinned
historical metadata. Bindings are exact file identities, never prefix rewrites.
Reports/transfer recipes contain paths and pins only, not input contents.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import stat
import sys

from input_inventory import checked_path, relative_parts

LIMIT = 16 * 1024 * 1024
MAX_ENTRIES = 4096


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_small(path, limit=LIMIT):
    path = checked_path(Path(path))
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError('not a bounded regular metadata/source file')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    after = path.stat()
    if len(data) > limit or (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file changed during read')
    return data


def json_bytes(data):
    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs)


def pointer(obj, parts):
    if not isinstance(parts, list) or not 1 <= len(parts) <= 20:
        raise ValueError('invalid finite selector')
    for part in parts:
        if isinstance(obj, list):
            if type(part) is not int or part < 0:
                raise ValueError('invalid array selector')
        elif not isinstance(obj, dict) or not isinstance(part, str):
            raise ValueError('invalid object selector')
        obj = obj[part]
    return obj


def identity(value):
    if not isinstance(value, str) or '\\' in value or not PureWindowsPath(value).is_absolute():
        raise ValueError('logical identity must be absolute Windows POSIX path')
    relative_parts(value.split(':/', 1)[1] if ':/' in value else '')
    return value


def sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid SHA256')
    return value


def inspect(profile, bindings):
    """Return a new plan; original documents, profile and bindings stay unchanged."""
    if set(profile) != {'schema_version', 'documents', 'entries', 'blocked_stages'} or profile['schema_version'] != 1:
        raise ValueError('unknown profile schema/fields')
    if not 1 <= len(profile['documents']) <= 16 or not 1 <= len(profile['entries']) <= MAX_ENTRIES:
        raise ValueError('finite document/entry limit exceeded')
    if not isinstance(bindings, dict):
        raise ValueError('bindings must map exact logical files to absolute local files')
    docs, rows, allowed, pins = {}, [], set(), {}
    # Hash before parsing or interpreting any original paths.
    for doc in profile['documents']:
        if set(doc) != {'id', 'logical', 'sha256', 'signature'} or doc['id'] in docs:
            raise ValueError('unknown/duplicate document declaration')
        logical = identity(doc['logical'])
        expected = sha(doc['sha256'])
        allowed.add(logical)
        row = {'logical': logical, 'sha256': expected, 'kind': 'metadata', 'declarations': [doc['id']]}
        rows.append(row)
        docs[doc['id']] = None
        if logical not in bindings:
            row['status'] = 'blocked_binding'
            continue
        local = checked_path(Path(bindings[logical]))
        row['physical'] = local.as_posix()
        try:
            data = read_small(local)
        except FileNotFoundError:
            row['status'] = 'blocked_missing'
            continue
        if digest(data) != expected:
            raise ValueError('original metadata SHA mismatch: ' + doc['id'])
        value = json_bytes(data)
        if doc['signature']:
            canonical = json.dumps({k: v for k, v in value.items() if k != 'signature'}, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            if digest(canonical) != value.get('signature'):
                raise ValueError('original canonical signature mismatch')
        docs[doc['id']] = value
        row.update(status='verified_metadata', actual_bytes=len(data))
    ids = set()
    for entry in profile['entries']:
        if set(entry) != {'id', 'document', 'selector', 'logical', 'sha256', 'kind'} or entry['id'] in ids:
            raise ValueError('unknown/duplicate entry')
        ids.add(entry['id'])
        logical, expected = identity(entry['logical']), sha(entry['sha256'])
        if entry['kind'] not in ('source', 'payload'):
            raise ValueError('only explicitly bounded sources may be hashed')
        if logical in pins and pins[logical] != expected:
            raise ValueError('conflicting pins for logical identity')
        pins[logical] = expected
        allowed.add(logical)
        original = docs[entry['document']]
        row = dict(entry)
        rows.append(row)
        if original is None:
            row['status'] = 'blocked_metadata'
            continue
        declaration = pointer(original, entry['selector'])
        if declaration != expected:
            raise ValueError('entry pin differs from original metadata')
        parent = pointer(original, entry['selector'][:-1])
        field = entry['selector'][-1]
        if field in ('sha256', 'manifest_sha256'):
            recorded = parent['path']
            if field == 'manifest_sha256':
                recorded += '/manifest.json'
        elif entry['selector'][0] == 'files' and entry['document'] == 'recovery':
            recorded = original['input_job'] + '/' + field
        else:
            raise ValueError('unknown pin declaration shape')
        if recorded != logical:
            raise ValueError('logical identity differs from original metadata')
        if logical not in bindings:
            row['status'] = 'blocked_binding'
            continue
        local = checked_path(Path(bindings[logical]))
        row['physical'] = local.as_posix()
        try:
            info = local.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('input is not a regular file')
            row['actual_bytes'] = info.st_size
            row['status'] = 'present_unverified_payload'
            if entry['kind'] == 'source':
                row['status'] = 'verified_source' if digest(read_small(local, 1024 * 1024)) == expected else 'blocked_source_hash'
        except FileNotFoundError:
            row['status'] = 'blocked_missing'
    if set(bindings) - allowed:
        raise ValueError('undeclared binding')
    physical = [str(checked_path(Path(v))).casefold() for v in bindings.values()]
    if len(physical) != len(set(physical)):
        raise ValueError('ambiguous physical binding')
    # Metadata-path entries may recur, but a transfer recipe has one row/file.
    transfer = {}
    for row in rows:
        logical = row['logical']
        if logical in transfer and transfer[logical]['sha256'] != row['sha256']:
            raise ValueError('conflicting document/entry pins')
        transfer.setdefault(logical, {k: row[k] for k in ('logical', 'sha256', 'kind')})
        transfer[logical]['destination'] = 'inputs/' + digest(logical.encode())[:20] + '/' + PureWindowsPath(logical).name
    return {'schema_version': 1, 'ready': False, 'payloads_hashed': False,
            'closure_complete': False, 'counts': dict(Counter(r['status'] for r in rows)),
            'blocked_stages': profile['blocked_stages'], 'entries': rows,
            'transfer_recipe': list(transfer.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path(__file__).with_name('profiles') / 'gpt-inputs.json')
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        profile = json_bytes(read_small(args.profile.absolute()))
        bindings = json_bytes(read_small(args.bindings.absolute()))
        output = checked_path(args.output.absolute())
        if output.exists() or any(output == checked_path(Path(v)) for v in bindings.values()):
            raise ValueError('output must be a fresh file')
        report = inspect(profile, bindings)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(json.dumps({'ready': False, 'counts': report['counts']}))
        return 1  # inventory never grants ML execution admission
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        print('gpt inputs: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
