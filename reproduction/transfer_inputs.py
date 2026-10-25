"""Plan finite external-file transfers; copying requires explicit --copy-inputs.

Stdlib only. Default operations stat files, but never open payloads. A transfer
receipt certifies listed bytes only, never scientific dependency closure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import sys

from gpt_inputs import json_bytes, read_small, sha
from input_inventory import checked_path, relative_parts

CONTRACT = 'kion-external-file-transfer-v1'
ROOT = Path(__file__).resolve().parents[1]
MAX_FILES = 4096
CHUNK = 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def physical(value):
    require(isinstance(value, str) and '..' not in Path(value).parts, 'Invalid absolute physical path')
    return checked_path(Path(value))


def logical_name(value):
    require(isinstance(value, str) and '\\' not in value and
            str(PurePosixPath(value)) == value and
            (PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()),
            'Canonical absolute logical identity required')
    tail = value.split(':/', 1)[1] if ':/' in value else value.lstrip('/')
    relative_parts(tail)
    return value


def logical_destination(logical):
    logical_name(logical)
    if ':/' in logical:
        drive, tail = logical.split(':/', 1)
        return 'inputs/' + drive + '/' + tail
    return 'inputs/posix/' + logical.lstrip('/')


def envelope(entries):
    return {'contract': CONTRACT, 'schema_version': 1, 'entries': entries}


def convert(document, format_name, bindings=None):
    """Explicit adapters only. GPT sizes come from stats, not unpinned report claims."""
    if format_name == 'manifest':
        require(bindings is None, 'Bindings are only accepted for gpt-report')
        return document
    entries = []
    if format_name == 'gpt-report':
        require(document.get('schema_version') == 1 and isinstance(bindings, dict),
                'GPT report v1 and exact bindings object required')
        recipe = document['transfer_recipe']
        require(isinstance(recipe, list) and 1 <= len(recipe) <= MAX_FILES, 'Finite GPT recipe required')
        require(set(bindings) == {row['logical'] for row in recipe}, 'GPT bindings must exactly cover recipe')
        for row in recipe:
            require(set(row) == {'logical', 'sha256', 'kind', 'destination'}, 'Unknown GPT recipe fields')
            require(row['kind'] in ('metadata', 'source', 'payload'), 'Unknown GPT kind')
            file = physical(bindings[row['logical']])
            entries.append(dict(logical=row['logical'], physical=file.as_posix(),
                                destination=row['destination'], sha256=row['sha256'], bytes=file.stat().st_size))
    elif format_name == 'rl-report':
        require(bindings is None, 'RL report already contains bindings')
        require(document.get('contract') == 'kion-historical-cache-rl-preparation-v1', 'Unknown RL report contract')
        rows = document['bindings']
        require(isinstance(rows, list) and 1 <= len(rows) <= MAX_FILES, 'Finite RL bindings required')
        for row in rows:
            require(set(row) == {'logical', 'physical', 'sha256', 'bytes', 'role', 'content_verified'},
                    'Unknown RL binding fields')
            entries.append({**{key: row[key] for key in ('logical', 'physical', 'sha256', 'bytes')},
                            'destination': logical_destination(row['logical'])})
    else:
        raise ValueError('Unknown explicit format')
    return envelope(entries)


def snapshot(info):
    # Windows path.stat and handle fstat can report different ctime semantics.
    # Identity, size and mtime are comparable; content is pinned by SHA256.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def plan(manifest):
    require(isinstance(manifest, dict) and set(manifest) == {'contract', 'schema_version', 'entries'} and
            manifest['contract'] == CONTRACT and type(manifest['schema_version']) is int and
            manifest['schema_version'] == 1, 'Unknown manifest schema')
    entries = manifest['entries']
    require(isinstance(entries, list) and 1 <= len(entries) <= MAX_FILES, 'Expected 1..4096 files')
    result, logicals, destinations, inodes = [], set(), set(), set()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {'logical', 'physical', 'destination', 'sha256', 'bytes'},
                'Unknown entry fields')
        logical = logical_name(entry['logical'])
        require(logical.casefold() not in logicals, 'Duplicate logical identity')
        logicals.add(logical.casefold())
        sha(entry['sha256'])
        require(type(entry['bytes']) is int and entry['bytes'] >= 0, 'Invalid byte count')
        parts = relative_parts(entry['destination'])
        require(len(parts) > 1 and parts[0] == 'inputs', 'Destination must be inside inputs/')
        dest = entry['destination'].casefold()
        require(all(dest != old and not dest.startswith(old + '/') and not old.startswith(dest + '/')
                    for old in destinations), 'Destination collision or file/directory alias')
        destinations.add(dest)
        file = physical(entry['physical'])
        info = file.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_size == entry['bytes'], 'Missing regular input or size mismatch')
        require(info.st_nlink == 1, 'Hardlinked sources are forbidden')
        inode = (info.st_dev, info.st_ino)
        require(inode not in inodes, 'Physical alias')
        inodes.add(inode)
        result.append({**entry, 'physical': file.as_posix()})
    return {'contract': CONTRACT, 'state': 'planned', 'payloads_hashed': False,
            'closure_complete': False, 'file_count': len(result),
            'total_bytes': sum(row['bytes'] for row in result), 'manifest': envelope(result)}


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def check_output(output, rows):
    output = physical(str(output))
    require(not output.exists(), 'Output must be a fresh directory/file')
    require(not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output), 'Output must be external to repository')
    for row in rows:
        source = physical(row['physical'])
        require(not source.is_relative_to(output) and not output.is_relative_to(source), 'Input/output overlap')
    require(output.parent.is_dir(), 'Output parent must already exist')
    return output


def copy_file(row, output):
    source = physical(row['physical'])
    before = source.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == row['bytes'], 'Source changed')
    destination = checked_path(output.joinpath(*relative_parts(row['destination'])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    checked_path(destination)
    digest, size = hashlib.sha256(), 0
    with source.open('rb') as reader, destination.open('xb') as writer:
        require(snapshot(os.fstat(reader.fileno())) == snapshot(before), 'Source replaced during open')
        for block in iter(lambda: reader.read(CHUNK), b''):
            size += len(block)
            require(size <= row['bytes'], 'Source grew during copy')
            digest.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
        require(snapshot(os.fstat(reader.fileno())) == snapshot(before), 'Source changed during copy')
    require(snapshot(physical(row['physical']).stat()) == snapshot(before), 'Source replaced during copy')
    require(size == row['bytes'] and digest.hexdigest() == row['sha256'], 'Source SHA256/size mismatch')
    copied = hashlib.sha256()
    with checked_path(destination).open('rb') as reader:
        for block in iter(lambda: reader.read(CHUNK), b''):
            copied.update(block)
    require(destination.stat().st_size == size and copied.hexdigest() == row['sha256'], 'Copied SHA256/size mismatch')
    return destination.as_posix()


def copy_inputs(planned, output, receiver_root=None):
    # Revalidate the finite plan immediately before making any output.
    planned = plan(planned['manifest'])
    rows = planned['manifest']['entries']
    output = check_output(output, rows)
    receiver_root = logical_name(receiver_root) if receiver_root is not None else output.as_posix()
    output.mkdir()
    receipt = {**planned, 'state': 'incomplete', 'verified_files': 0, 'receiver_root': receiver_root}
    write_json(output / 'started.json', receipt)
    try:
        bindings = {}
        receiver = []
        for row in rows:
            copy_file(row, output)
            new = str(PurePosixPath(receiver_root) / row['destination'])
            bindings[row['logical']] = new
            receiver.append({**row, 'physical': new})
            receipt['verified_files'] += 1
        write_json(output / 'bindings.json', bindings)
        write_json(output / 'manifest.json', envelope(receiver))
        write_json(output / 'recipe.json', envelope([{**row, 'physical': bindings[row['logical']]} for row in rows]))
        receipt.update(state='completed', payloads_hashed=True)
        # A final receipt is the completion marker. No completion file on failure.
        write_json(output / 'completion.pending.json', receipt)
        (output / 'completion.pending.json').rename(output / 'completed.json')
        return receipt
    except BaseException as exc:
        receipt.update(state='failed', error=str(exc), payloads_hashed=False)
        write_json(output / 'failed.json', receipt)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--format', choices=('manifest', 'gpt-report', 'rl-report'), default='manifest')
    parser.add_argument('--bindings', type=Path)
    parser.add_argument('--report', type=Path, help='Optional fresh plan JSON file; otherwise stdout only')
    parser.add_argument('--copy-inputs', action='store_true')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--receiver-root', help='Optional final absolute bundle path on recipient host')
    args = parser.parse_args(argv)
    try:
        require(args.copy_inputs == (args.destination is not None), '--destination and --copy-inputs must be supplied together')
        require(args.receiver_root is None or args.copy_inputs, '--receiver-root requires --copy-inputs')
        if args.receiver_root is not None:
            logical_name(args.receiver_root)
        document = json_bytes(read_small(args.input.absolute()))
        bindings = json_bytes(read_small(args.bindings.absolute())) if args.bindings else None
        planned = plan(convert(document, args.format, bindings))
        rows = planned['manifest']['entries']
        # Include metadata input files in all overlap checks, not only payloads.
        protected = rows + [{'physical': args.input.absolute().as_posix()}]
        if args.bindings:
            protected.append({'physical': args.bindings.absolute().as_posix()})
        report = check_output(args.report.absolute(), protected) if args.report else None
        output = check_output(args.destination.absolute(), protected) if args.destination else None
        if report and output:
            require(not report.is_relative_to(output) and not output.is_relative_to(report), 'Report/output overlap')
        if report:
            write_json(report, planned)
        if args.copy_inputs:
            planned = copy_inputs(planned, output, args.receiver_root)
        print(json.dumps(planned, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('input transfer: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
