"""Finite historical child-metadata inventory; no payload is ever opened.

The profile is a reviewed selector list, not a recursive JSON crawler. A child
can only be opened after its byte pin is established by an already verified
parent. Exact logical identities and original path spellings are retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path, PureWindowsPath
import stat
import sys

import gpt_inputs as direct

CONTRACTS = {'kion-all-local-pack-v1', 'kion-text-embeddings-v1',
             'kion-related-interest-v1', 'kion-completion-event-store-v1',
             'kion-user-diagnostics-v1', 'kion-exact-outcome-threshold-facts-v1',
             'kion-audience-exact-loo-v1', 'kion-complete-scoped-assessment-v1'}
PROVIDER_SOURCE_SHA = '7cbe284dfbd9d133bd98e2d363a9ee8b8d8530ffe7b977cbe13c6146d0f4fe5e'


def canonical_hash(value):
    return direct.digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))

def logical_path(raw, base):
    if not isinstance(raw, str) or '\\' in raw:
        raise ValueError('invalid original path spelling')
    if PureWindowsPath(raw).is_absolute():
        return direct.identity(raw)
    direct.relative_parts(raw)
    return direct.identity(base + '/' + raw)


def transfer_recipe(rows):
    """Deduplicate already-classified rows without dropping transfer roles."""
    transfer = {}
    for row in rows:
        key = row['logical']
        if row['kind'] not in ('metadata', 'source', 'payload'):
            raise ValueError('unknown transfer kind')
        if key in transfer:
            if transfer[key]['sha256'] != row['sha256']:
                raise ValueError('conflicting transfer pins')
            if transfer[key]['kind'] != row['kind']:
                raise ValueError('conflicting transfer kinds')
        transfer.setdefault(key, dict(logical=key, sha256=row['sha256'], kind=row['kind'],
                                     destination='inputs/' + direct.digest(key.encode())[:20] + '/' + PureWindowsPath(key).name))
    return list(transfer.values())


def inspect(profile, bindings, direct_profile_bytes):
    """Authenticate original direct-profile bytes before interpreting any bindings.

    A parsed dictionary is intentionally insufficient: JSON serialization would
    lose the original byte identity pinned by the transitive profile.
    """
    if set(profile) != {'schema_version', 'direct_profile_sha256', 'documents', 'entries', 'unexpanded'} or profile['schema_version'] != 1:
        raise ValueError('unknown transitive profile')
    if not isinstance(direct_profile_bytes, bytes) or len(direct_profile_bytes) > direct.LIMIT:
        raise ValueError('bounded original direct-profile bytes required')
    if direct.digest(direct_profile_bytes) != direct.sha(profile['direct_profile_sha256']):
        raise ValueError('direct profile pin mismatch')
    direct_profile = direct.json_bytes(direct_profile_bytes)
    if not 1 <= len(profile['documents']) <= 16 or len(profile['entries']) > 4096:
        raise ValueError('finite metadata inventory exceeded')
    doc_ids = {d['id'] for d in profile['documents']}
    for e in profile['entries']:
        if set(e) != {'document', 'selector', 'path_selector', 'implicit_name', 'original_path', 'base', 'logical', 'sha256'} or e['document'] not in doc_ids:
            raise ValueError('unknown child entry schema/document')
        if not e['path_selector'] and (e['selector'][:1] != ['outputs'] or len(e['selector']) != 3 or e['implicit_name'] != e['selector'][1]):
            raise ValueError('unknown implicit filename schema')
    allowed = {r['logical'] for r in direct_profile['documents'] + direct_profile['entries']}
    root = direct.inspect(direct_profile, {k: v for k, v in bindings.items() if k in allowed})
    rows = list(root['entries'])
    established = {r['logical']: r['sha256'] for r in rows if 'document' in r and r['status'] != 'blocked_metadata'}
    docs = {}
    for doc in profile['documents']:
        fields = {'id', 'logical', 'sha256', 'contract', 'keys', 'signature_exclude', 'limit'}
        if set(doc) not in (fields, fields | {'derived_from'}) or doc['id'] in docs:
            raise ValueError('invalid child declaration')
        logical, pin = direct.identity(doc['logical']), direct.sha(doc['sha256'])
        derived = doc.get('derived_from')
        name = 'assessment.json' if derived else 'manifest.json'
        if PureWindowsPath(logical).name != name or doc['contract'] not in CONTRACTS:
            raise ValueError('only allowlisted child metadata schemas may be opened')
        maximum = 32 * 1024 * 1024 if doc['contract'] == 'kion-all-local-pack-v1' else 16 * 1024 * 1024
        if type(doc['limit']) is not int or not 0 < doc['limit'] <= maximum:
            raise ValueError('invalid metadata size limit')
        allowed.add(logical)
        row = dict(logical=logical, sha256=pin, kind='metadata', status='blocked_parent')
        rows.append(row)
        docs[doc['id']] = None
        canonical_pin = None
        if derived:
            if derived != {'document': 'provider', 'selector': ['assembly', 'assessment_sha256'], 'consumer_sha256': PROVIDER_SOURCE_SHA} or doc['contract'] != 'kion-complete-scoped-assessment-v1':
                raise ValueError('unknown derived metadata rule')
            parent = next((d for d in direct_profile['documents'] if d['id'] == 'provider'), None)
            if parent is None or logical != str(PureWindowsPath(parent['logical']).parent / 'assessment.json').replace('\\', '/'):
                raise ValueError('derived metadata identity differs from pinned consumer rule')
            if not any(r['logical'] == parent['logical'] and r['status'] == 'verified_metadata' for r in rows) or not any(r['sha256'] == PROVIDER_SOURCE_SHA and r['status'] == 'verified_source' for r in rows):
                continue
            parent_bytes = direct.read_small(Path(bindings[parent['logical']]))
            if direct.digest(parent_bytes) != parent['sha256']:
                raise ValueError('derived metadata parent changed')
            canonical_pin = direct.sha(direct.pointer(direct.json_bytes(parent_bytes), derived['selector']))
        elif established.get(logical) != pin:
            continue
        if logical not in bindings:
            row['status'] = 'blocked_binding'
            continue
        local = direct.checked_path(Path(bindings[logical]))
        row['physical'] = local.as_posix()
        try:
            data = direct.read_small(local, doc['limit'])
        except FileNotFoundError:
            row['status'] = 'blocked_missing'
            continue
        if direct.digest(data) != pin:
            raise ValueError('child metadata SHA mismatch: ' + doc['id'])
        obj = direct.json_bytes(data)
        if canonical_pin is not None and canonical_hash(obj) != canonical_pin:
            raise ValueError('derived metadata canonical parent pin mismatch')
        if sorted(obj) != doc['keys'] or obj.get('contract') != doc['contract']:
            raise ValueError('unseen child schema: ' + doc['id'])
        if doc['signature_exclude'] is not None:
            canonical = json.dumps({k: v for k, v in obj.items() if k not in doc['signature_exclude']}, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            if direct.digest(canonical) != obj.get('signature'):
                raise ValueError('child canonical signature mismatch')
        docs[doc['id']] = obj
        row.update(status='verified_metadata', actual_bytes=len(data))
        # Only this document's finite, reviewed selectors are expanded.
        for entry in profile['entries']:
            if entry['document'] != doc['id']:
                continue
            pin_value = direct.pointer(obj, entry['selector'])
            if pin_value != entry['sha256']:
                raise ValueError('child pin differs from original')
            raw = direct.pointer(obj, entry['path_selector']) if entry['path_selector'] else entry['implicit_name']
            if raw != entry['original_path'] or logical_path(raw, entry['base']) != entry['logical']:
                raise ValueError('child identity differs from original')
            if entry['logical'] in established and established[entry['logical']] != pin_value:
                raise ValueError('conflicting child pins')
            established[entry['logical']] = pin_value
    for entry in profile['entries']:
        logical, pin = direct.identity(entry['logical']), direct.sha(entry['sha256'])
        allowed.add(logical)
        row = dict(entry, kind='payload', status='blocked_metadata')
        rows.append(row)
        if docs.get(entry['document']) is None:
            continue
        row['status'] = 'blocked_binding'
        if logical in bindings:
            local = direct.checked_path(Path(bindings[logical]))
            row['physical'] = local.as_posix()
            try:
                info = local.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('payload is not a regular file')
                row.update(status='present_unverified_payload', actual_bytes=info.st_size)
            except FileNotFoundError:
                row['status'] = 'blocked_missing'
    if set(bindings) - allowed:
        raise ValueError('undeclared binding')
    physical = [str(direct.checked_path(Path(v))).casefold() for v in bindings.values()]
    if len(set(physical)) != len(physical):
        raise ValueError('ambiguous physical binding')
    # Direct opaque rows become metadata only for the explicit child allowlist.
    # Child source provenance remains stat-only even when its transfer role is
    # source; status, not kind, records whether bytes were actually verified.
    metadata = {d['logical'] for d in direct_profile['documents'] + profile['documents']}
    sources = {e['logical'] for e in direct_profile['entries'] if e['kind'] == 'source'}
    if metadata & sources:
        raise ValueError('conflicting metadata/source declarations')
    for row in rows:
        row['kind'] = 'metadata' if row['logical'] in metadata else 'source' if row['logical'] in sources else 'payload'
    return dict(schema_version=1, ready=False, closure_complete=False, payloads_hashed=False,
                entries=rows, counts=dict(Counter(r['status'] for r in rows)),
                transfer_recipe=transfer_recipe(rows), unexpanded=profile['unexpanded'], blocked_stages=root['blocked_stages'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, default=Path(__file__).with_name('profiles') / 'gpt-transitive.json')
    p.add_argument('--direct-profile', type=Path, default=Path(__file__).with_name('profiles') / 'gpt-inputs.json')
    p.add_argument('--bindings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    try:
        profile = direct.json_bytes(direct.read_small(a.profile.absolute()))
        source = direct.read_small(a.direct_profile.absolute())
        bindings = direct.json_bytes(direct.read_small(a.bindings.absolute()))
        output = direct.checked_path(a.output.absolute())
        if output.exists() or any(output == direct.checked_path(Path(v)) for v in bindings.values()):
            raise ValueError('output must be fresh')
        result = inspect(profile, bindings, source)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(json.dumps({'ready': False, 'counts': result['counts'], 'unique_files': len(result['transfer_recipe'])}))
        return 1
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        print('gpt transitive: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
