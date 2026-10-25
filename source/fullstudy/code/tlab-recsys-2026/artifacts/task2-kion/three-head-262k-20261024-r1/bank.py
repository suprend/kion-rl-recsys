"""Frozen E2 nested 262k bank. Import is stdlib-only; root exclusively admits execution."""
from collections import Counter
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys

HERE = Path(__file__).resolve().parent
ART = HERE.parent
CONTRACT = 'kion-three-head-262k-bank-v1'
ROWS, SEED = 262144, 20261024
REUSE_ROWS = 32768
REUSE_DIR = ART / 'three-head-pilot-20261024-r1/cache/train'
REUSE_MANIFEST_SHA = 'a64decc0990a4644e878d3ab830ea81035d431ca6ff6d664aaff9a03d8c8cf36'
BANK_OUTPUT = Path('C:/Users/Михаил/.cache/codex-kion/three-head-262k-20261024-r1/train')
MAP_WINDOW_ROWS = 1024
TEACHER_SHA = '5e7101857c2dbe100fb4e1587c0369abc95c108303ed149123ee183f62aeac26'
CONFIG_SHA = 'e3b4b1730895fceb1b7d4ea6357aea783c046f5e87cacd7d4ff933cf94d45d45'
SCHEMA_SHA = '71d90eee774d30db804d1ab32c4d82995d1320baf26aaf36443a6498fd85fc5d'
CANDIDATE_SHA = '6923f4e63ce6278de3297449c8d59fa498b9117f2cdc85537d7e9bd4e0acc09d'
Q_SHA = '5e3c76195479ec0b31d5356176dc1f7df15a784cf50d675a7d4d57f687736a5c'
EVENT_SHA = '382b410d564bb73eba268b5e2e49d092c25d2af4d1df677020ce46f546dc0cbf'
EVENT_PINS = {
    'user_id': ('int64', '26d00139e72b998092ccf9bc6cf24a1bcb1e38a807fce38976609ad73fefc8a7'),
    'date': ('int32', '2b236ebb5a4d3b5057a4f763c25f412d1cc0902a5610a9d2972a4f364af78c9f'),
    'token': ('int32', '23078cdedb0c6aea4da9356d4b0572fdaf14db7eb56fb0e2834b11e40677cf21')}
EVENT_MANIFEST = ART / 'all-agreed-20261020/completion-diagnostics/event-store-r1/full-r1/manifest.json'
DATA = Path('D:/tlab-recsys-2026/data/processed/kion-full-context64')
ORIGINAL = ART / 'sft-e5-20261023-r1/admission/request-r1.json'
ORIGINAL_SHA = 'a8ffeb9cb94146c9ee29e24ef814ff5951d0a81184a5daf0c669ffb1fb9fdb4f'
RUNTIME = ART / 'ready-pack-20261020/provider/runtime-native-r5/scripts'
SOURCE_COLUMNS = ['current_detached_gpt_logit', 'log1p_strict7d_count']
LAYOUT = {'encoded': ('float32', (ROWS, 256)),
    'candidate_ids': ('int32', (ROWS, 32)), 'base_logits': ('float32', (ROWS, 32)),
    'raw_features': ('float32', (ROWS, 32, 1068)), 'source_values': ('float32', (ROWS, 32, 2)),
    'row_indices': ('int64', (ROWS,)), 'user_ids': ('int64', (ROWS,)),
    'target_ids': ('int32', (ROWS,)), 'dates': ('int32', (ROWS,))}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path, maximum=8 << 20):
    path = Path(path)
    require(path.stat().st_size <= maximum, 'Bounded JSON exceeded: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(request, path, name):
    path = Path(path).resolve()
    require(path.stat().st_size <= 8 << 20 and request['source_hashes'].get(path.as_posix()) == digest(path),
            'Unpinned/changed source: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def proportional_quotas(counts, maximum):
    """Largest remainder allocation, exact total; ties use ascending date."""
    require(type(maximum) is int and maximum > 0 and all(n > 0 for n in counts.values()), 'Invalid strata')
    total = sum(counts.values())
    require(total >= maximum, 'Too few known train rows')
    quotas = {day: n * maximum // total for day, n in counts.items()}
    remainder = maximum - sum(quotas.values())
    order = sorted(counts, key=lambda day: (-(counts[day] * maximum % total), day))
    for day in order[:remainder]:
        quotas[day] += 1
    require(sum(quotas.values()) == maximum and all(quotas[d] <= counts[d] for d in counts), 'Allocation differs')
    return quotas


def select_rows(targets, dates, vocab, maximum=ROWS, seed=SEED, reuse_rows=()):
    """Exact date quotas with an explicit nested old-selection constraint.

    Retain every old row, then draw only the remaining quota from other known
    rows using fixed per-date RNG. This is not a fresh unconstrained sample.
    """
    require(len(targets) == len(dates), 'Train date/target alignment differs')
    counts = Counter(int(day) for target, day in zip(targets, dates) if 3 <= int(target) < vocab)
    quotas = proportional_quotas(counts, maximum)
    reused = set()
    reused_counts = Counter()
    for row in reuse_rows:
        row = int(row)
        require(0 <= row < len(targets) and row not in reused
                and 3 <= int(targets[row]) < vocab, 'Invalid/duplicate reused known train row')
        reused.add(row)
        reused_counts[int(dates[row])] += 1
    require(all(reused_counts[d] <= quotas.get(d, 0) for d in reused_counts),
            'Old selection exceeds new date quota')
    positions = {}
    for day in sorted(counts):
        key = hashlib.sha256(f'{seed}:date:{day}'.encode('ascii')).digest()
        positions[day] = set(random.Random(int.from_bytes(key, 'little')).sample(
            range(counts[day] - reused_counts[day]), quotas[day] - reused_counts[day]))
    cursors = Counter()
    selected = [(int(dates[i]), i) for i in reused]
    for index, (target, day) in enumerate(zip(targets, dates)):
        if 3 <= int(target) < vocab and index not in reused:
            day = int(day)
            if cursors[day] in positions[day]:
                selected.append((day, index))
            cursors[day] += 1
    selected.sort()
    require(len(selected) == maximum and len({i for _, i in selected}) == maximum, 'Exact unique train sample required')
    require(Counter(d for d, _ in selected) == Counter(quotas), 'Nested date quota mismatch')
    return [index for _, index in selected], counts, quotas


def context_signature(request):
    inputs = request['input_hashes']
    return canonical({'metadata_sha256': inputs[(DATA / 'metadata.json').as_posix()],
        'q_sha256': Q_SHA, 'event_manifest_sha256': EVENT_SHA,
        'event_arrays_sha256': {key: pin for key, (_, pin) in EVENT_PINS.items()}})


def denied(*args, **kwargs):
    raise RuntimeError('Frozen bank context forbids training/optimizer and cache reconstruction')


def _raw_ids(provider, np):
    vocab = int(provider.metadata['vocab_size'])
    result = np.empty(vocab - 3, dtype=np.int64)
    assigned = set()
    for raw, token in provider.metadata['item_id_to_token'].items():
        token = int(token)
        require(3 <= token < vocab and token not in assigned, 'Non-bijective known token mapping')
        result[token - 3] = int(raw)
        assigned.add(token)
    require(len(assigned) == len(result) and bool(np.all(np.diff(result) > 0)), 'Incomplete/noncanonical raw-ID mapping')
    result.flags.writeable = False
    return result


def configure_validation_graph(candidates, request, check):
    """Reuse train paths read-only; allow only the admitted val graph in our root."""
    candidates._build = denied
    if request.get('validation_graph_cache'):
        root = Path(request['validation_graph_cache']).resolve()
        require(request.get('stage') == 'evaluate' and request.get('allow_validation_graph_build', False) is False
                and root == (REUSE_DIR.parent.parent / 'cache/validation-graph').resolve()
                and root.is_dir(), 'Exact existing readonly validation graph required')
        shadow = copy.copy(candidates)
        shadow.root = root
        original_paths, validation_paths = candidates._paths, shadow._paths
        candidates._paths = lambda origin: validation_paths(origin) if int(origin) == 738011 else original_paths(origin)
        return dict(root=root.as_posix(), origin=738011, train_cache_read_only=True,
                    validation_cache_read_only=True, rebuild_forbidden=True)
    if not request.get('allow_validation_graph_build', False):
        return None
    require(request['allow_validation_graph_build'] is True and request.get('stage') == 'evaluate'
            and candidates.validation_origin == 738011, 'Validation graph build requires evaluate-only admission')
    root = (HERE / 'cache/validation-graph').resolve()
    require(root.is_relative_to(HERE) and root != Path(request['graph_cache']).resolve(), 'Distinct owned validation graph root required')
    root.mkdir(parents=True, exist_ok=True)
    shadow = copy.copy(candidates)
    shadow.root, shadow.max_disk_bytes = root, 64 << 20
    shadow._disk_check()
    original_paths = candidates._paths
    validation_paths = shadow._paths

    def paths(origin):
        return validation_paths(origin) if int(origin) == 738011 else original_paths(origin)

    def build(origin, descriptor, arrays):
        require(int(origin) == 738011, 'Training graph cache is read-only')
        wanted_descriptor, wanted_arrays = validation_paths(738011)
        require(descriptor == wanted_descriptor and arrays == wanted_arrays
                and descriptor.parent.resolve() == root
                and all(path.parent.resolve() == root for path in arrays.values()),
                'Validation graph write escaped owned root')
        check(stage='validation-only-graph-build', origin=738011)
        # The original corrected implementation owns arithmetic and atomic publication.
        type(candidates)._build(shadow, 738011, descriptor, arrays)
        shadow._disk_check()

    candidates._paths, candidates._build = paths, build
    return dict(root=root.as_posix(), origin=738011, maximum_bytes=64 << 20,
                train_cache_read_only=True, original_build_implementation=True)


def open_context(request, check):
    """Heavy root-only stage: one original provider and immutable old1068 E2."""
    ctx = {'events': {}, 'check': check, 'closed': False}
    try:
        check(stage='frozen-context-open')
        require(digest(ORIGINAL) == ORIGINAL_SHA, 'Original provider provenance changed')
        original = read(ORIGINAL)
        require(all(request['source_hashes'].get(p) == h for p, h in original['source_hashes'].items()),
                'Original runtime source closure must be retained')
        teacher = dict(request['teacher'])
        require(teacher['sha256'] == TEACHER_SHA and teacher['config_sha256'] == CONFIG_SHA,
                'Exact frozen old1068 E2 required')
        checkpoint, config_path = Path(teacher['path']), Path(teacher['config_path'])
        require(checkpoint.resolve() == ART / 'night-20261023-r1/sft-recovery/run-recovery-r1/worker/model/latest.pt'
                and config_path.resolve() == checkpoint.parent / 'config.json', 'E2 teacher path changed')
        check(stage='teacher-integrity')
        require(digest(checkpoint) == TEACHER_SHA and digest(config_path) == CONFIG_SHA, 'Frozen E2 payload/config changed')
        # The >8MiB model configuration is only opened inside this root-admitted heavy stage.
        config = read(config_path, maximum=16 << 20)
        require(Path(sys.executable).resolve() == Path(original['interpreter']['path']).resolve(), 'Pinned primary interpreter required')
        sys.path.insert(0, RUNTIME.as_posix())
        runtime = load(request, RUNTIME / 'kion_ready_runtime.py', '_pilot_ready_runtime')
        runtime.BASE_EXE = Path(original['interpreter']['path'])
        runtime.configure_paths()
        train = load(request, RUNTIME / 'kion_complete_train.py', '_pilot_original_train')
        ctx.update(train=train, np=train.np, torch=train.torch)
        require(train.np.__version__ == '2.2.6' and train.torch.__version__ == '2.8.0+cu126'
                and train.torch.cuda.is_available(), 'Original CUDA runtime required')
        for name in ('update', 'run', 'main'):
            setattr(train, name, denied)
        args = train.parser().parse_args(original['train_argv'])
        train.torch.set_num_threads(4)
        train.torch.manual_seed(SEED)
        train.torch.backends.cudnn.deterministic, train.torch.backends.cudnn.benchmark = True, False
        hook = original['provider_hook']
        require(request['provider_hook'] == hook and hook['options']['enabled'] is True
                and hook['options'].get('allow_missing_fallback', False) is False, 'Original exact readonly store required')
        store_module = load(request, hook['path'], '_pilot_original_store')
        ctx['store'] = store_module.install(train, copy.deepcopy(hook['options']))
        check(stage='provider-start')
        module_name, factory_name = args.provider.split(':', 1)
        provider = getattr(importlib.import_module(module_name), factory_name)(args)
        ctx['provider'] = provider
        require(len(provider.splits['train']['y']) == 3477437 and len(provider.splits['val']['y']) == 158760,
                'Original full train/validation provider required')
        perf = original['performance_hook']
        require(request['performance_hook'] == perf and digest(perf['acceptance_receipt']['path']) == perf['acceptance_receipt']['sha256']
                and read(perf['acceptance_receipt']['path'])['state'] == 'accepted', 'Original performance hook changed')
        perf_module = load(request, perf['path'], '_pilot_original_perf')
        ctx['perf'] = perf_module.install(provider, **perf['options'])
        require(provider.audience is None and provider.people_dynamics is None, 'Original deferred feature contract differs')
        dynamic_module = load(request, request['dynamic_cache_path'], '_pilot_dynamic_cache')
        ctx['dynamic'] = dynamic_module.install_dynamic_cache(provider, max_bytes=64 << 20, max_origins=2)
        q_path = Path(request['q_source_path'])
        require(digest(q_path) == Q_SHA, 'Original Q source changed')
        q = load(request, q_path, '_pilot_frozen_q')
        # Optional adapter is a separately admitted source pin, never implicit.
        if request.get('q_adapter_path'):
            adapter = load(request, request['q_adapter_path'], '_pilot_q_adapter')
            ctx['q_adapter'] = adapter.install(q)
        require(digest(EVENT_MANIFEST) == EVENT_SHA, 'Original event store changed')
        event = read(EVENT_MANIFEST)
        require(event['state'] == 'completed' and event['events_count'] == 4649162, 'Incomplete event store')
        require(digest(DATA / 'metadata.json') == request['input_hashes'][(DATA / 'metadata.json').as_posix()], 'Changed metadata')
        require(request['input_hashes'].get((DATA / 'train.npz').as_posix()) == event['prepared_pins']['train.npz']['sha256'],
                'Train prepared pin differs')
        for key, (dtype, pin) in EVENT_PINS.items():
            check(stage='event-mmap', array=key)
            entry = event['arrays'][key]
            path = EVENT_MANIFEST.parent / entry['path']
            require(entry == dict(path=key + '.npy', sha256=pin, dtype=dtype, shape=[4649162])
                    and request['input_hashes'].get(path.as_posix()) == pin, 'Event source pin differs')
            # Root verifies payload pins before admission; read-only mmap enforces bounded access here.
            value = train.np.load(path, mmap_mode='r', allow_pickle=False)
            ctx['events'][key] = value
            require(value.shape == (4649162,) and str(value.dtype) == dtype and not value.flags.writeable, 'Event header differs')
        candidate_path = Path(request['candidate_source_path'])
        require(digest(candidate_path) == CANDIDATE_SHA, 'Correct current LiveCandidates required')
        candidate_module = load(request, candidate_path, '_pilot_live_candidates')
        require(Path(request['graph_cache']).is_dir(), 'Existing train graph cache required')
        candidates = candidate_module.LiveCandidates(provider, train.np, q, ctx['events'], _raw_ids(provider, train.np),
            request['graph_cache'], context_signature(request), check=check, seed=SEED)
        ctx['candidates'] = candidates
        validation_graph = configure_validation_graph(candidates, request, check)
        schedule = train.TrainingSchedule(provider, args)
        require(schedule.updates_per_epoch == 27235 and len(schedule.eligible) == 3477437, 'Original schedule identity differs')
        device = train.torch.device('cuda')
        check(stage='teacher-model-create')
        model = train.make_model(provider, args).to(device)
        ctx['model'] = model
        require(train.make_config(provider, model, args, device, schedule=schedule) == config, 'Original E2 model config differs')
        saved = train.load_checkpoint(checkpoint, config, map_location='cpu')
        require(saved['state']['epoch'] == 2 and saved['state']['offset'] == 0 and saved['state']['updates'] == 54470,
                'Completed E2 state required')
        model.load_state_dict(saved['state_dict'])
        del saved, schedule, config
        gc.collect()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        columns = model.feature_columns['candidate']
        schema = hashlib.sha256(json.dumps(columns, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
        require(model.candidate_feature_dim == 1068 and schema == SCHEMA_SHA, 'Original full1068 feature schema required')
        require(list(candidates.source_feature_names[12:14]) == SOURCE_COLUMNS, 'Original numeric metadata12/13 required')
        with train.torch.inference_mode(), train.legacy.autocast(device):
            table = model.candidate_table()
        ctx.update(args=args, device=device, table=table, metadata=provider.metadata, teacher=teacher,
            feature_columns=columns, feature_schema_sha256=schema, source_columns=SOURCE_COLUMNS,
            candidate_contract=candidates.contract())
        if validation_graph is not None:
            ctx['candidate_contract']['validation_graph'] = validation_graph
        check(stage='frozen-context-ready')
        return ctx
    except BaseException:
        close_context(ctx)
        raise


def close_context(ctx):
    """Attempt every teardown; errors prevent a completed bank descriptor."""
    if ctx.get('closed'):
        return ctx['cleanup']
    cleanup = {'teardown_completed': False, 'errors': []}
    for key, method in (('candidates', 'close'), ('perf', 'uninstall'), ('dynamic', 'uninstall'),
                        ('provider', 'close'), ('store', 'uninstall'), ('q_adapter', 'uninstall')):
        handle = ctx.pop(key, None)
        if handle is not None:
            try:
                getattr(handle, method)()
                if key == 'store':
                    cleanup['query_store'] = handle.evidence()
            except BaseException as error:
                cleanup['errors'].append({'owner': key, 'error': repr(error)})
    for value in ctx.pop('events', {}).values():
        try:
            mapping = getattr(value, '_mmap', None)
            if mapping is not None:
                mapping.close()
        except BaseException as error:
            cleanup['errors'].append({'owner': 'event_mmap', 'error': repr(error)})
    for key in ('table', 'model', 'provider', 'np', 'metadata', 'train', 'args'):
        ctx.pop(key, None)
    torch = ctx.pop('torch', None)
    gc.collect()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except BaseException as error:
            cleanup['errors'].append({'owner': 'cuda_release', 'error': repr(error)})
    cleanup['teardown_completed'] = not cleanup['errors']
    ctx.update(closed=True, cleanup=cleanup)
    return cleanup


def _entry(path, values, root):
    return dict(path=path.relative_to(root).as_posix(), shape=list(values.shape), dtype=str(values.dtype),
                bytes=path.stat().st_size, payload_bytes=int(values.nbytes), sha256=digest(path))


def validate_reuse_manifest(manifest, request):
    """Small JSON contract checks; payloads are checked only in admitted run."""
    require(manifest.get('contract') == 'kion-three-head-pilot-bank-v1'
            and manifest.get('state') == 'completed'
            and manifest.get('rows') == manifest.get('train_rows') == REUSE_ROWS
            and manifest.get('split') == 'train' and manifest.get('seed') == SEED
            and manifest.get('teardown_completed') is True
            and manifest.get('model_training') is False and manifest.get('optimizer_created') is False
            and manifest.get('final_test') is False, 'Completed exact old 32768 bank required')
    require(manifest.get('teacher') == request['teacher']
            and manifest['teacher']['sha256'] == TEACHER_SHA
            and manifest['teacher']['config_sha256'] == CONFIG_SHA,
            'Reuse teacher/config lineage differs')
    require(manifest.get('feature_schema_sha256') == SCHEMA_SHA
            and manifest.get('source_columns') == SOURCE_COLUMNS
            and manifest.get('candidate_width') == 32 and manifest.get('feature_width') == 1068
            and manifest.get('context_width') == 256
            and manifest.get('natural_covered_only') is False
            and manifest.get('same_day_positives_excluded') is True
            and manifest.get('context_signature') == context_signature(request), 'Reuse feature/context lineage differs')
    require(set(manifest.get('arrays', {})) == set(LAYOUT) | {'item_table'}, 'Complete old bank arrays required')


def open_reuse(request, np, ctx, check, maps):
    root = Path(request['reuse_bank_dir']).resolve()
    require(root == REUSE_DIR.resolve() and request['reuse_bank_manifest_sha256'] == REUSE_MANIFEST_SHA,
            'Exact old bank directory/manifest pin required')
    require(digest(root / 'bank.json') == REUSE_MANIFEST_SHA, 'Old bank manifest changed')
    manifest = read(root / 'bank.json')
    validate_reuse_manifest(manifest, request)
    require(manifest['feature_columns'] == ctx['feature_columns']
            and manifest['candidate_contract'] == ctx['candidate_contract'], 'Reuse schema/candidate configuration differs')
    for name, entry in manifest['arrays'].items():
        check(stage='reuse-array-integrity', array=name)
        path = root / (name + '.npy')
        require(entry['path'] == name + '.npy' and path.is_file()
                and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                'Old array payload pin differs: ' + name)
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        maps[name] = values
        dtype, shape = ('float32', (ctx['model'].vocab_size, 256)) if name == 'item_table' else LAYOUT[name]
        if name != 'item_table':
            shape = (REUSE_ROWS, *shape[1:])
        require(values.shape == shape and list(values.shape) == entry['shape']
                and str(values.dtype) == dtype == entry['dtype']
                and int(values.nbytes) == entry['payload_bytes'] and not values.flags.writeable,
                'Old array header differs: ' + name)
    return manifest


def remap_maps(np, maps, root, mode):
    """Release touched mapped pages; flush alone does not bound native RSS."""
    for name, value in list(maps.items()):
        if mode == 'r+':
            value.flush()
            with (root / (name + '.npy')).open('r+b') as stream:
                os.fsync(stream.fileno())
        value._mmap.close()
        maps.pop(name)
        maps[name] = np.load(root / (name + '.npy'), mmap_mode=mode, allow_pickle=False)


def close_maps(maps):
    for name, value in list(maps.items()):
        value._mmap.close()
        maps.pop(name)


def run(request, check):
    """Root-only builder. Fresh NPY maps are flushed per bounded batch, published last."""
    require(request.get('seed', SEED) == SEED and request.get('train_rows', ROWS) == ROWS, 'Fixed 262k seed/rows required')
    output = Path(request.get('bank_output', request.get('bank_dir', ''))).resolve()
    require(output == BANK_OUTPUT.resolve() and not output.exists(), 'Fresh exact admitted C cache bank directory required')
    output.mkdir(parents=True)
    ctx, arrays, reused, descriptor = None, {}, {}, None
    try:
        ctx = open_context(request, check)
        np, torch, train = ctx['np'], ctx['torch'], ctx['train']
        provider, model, table = ctx['provider'], ctx['model'], ctx['table']
        candidates, device = ctx['candidates'], ctx['device']
        origins = candidates.origins
        old = open_reuse(request, np, ctx, check, reused)
        old_rows = [int(i) for i in reused['row_indices']]
        check(stage='nested-proportional-selection')
        rows, counts, quotas = select_rows(provider.splits['train']['y'], origins, model.vocab_size, reuse_rows=old_rows)
        require(old['date_counts'] == {str(d): n for d, n in sorted(counts.items())}, 'Old/full known date census differs')
        require(Counter(int(d) for d in reused['dates']) == Counter({int(d): n for d, n in old['date_quotas'].items()}),
                'Old rows/date quotas differ')
        require(all(int(origins[i]) < candidates.validation_origin for i in rows), 'Train origin crosses validation')
        # Validate every daily descriptor before allocating the ~33.4GiB raw bank.
        for day in sorted(d for d, count in quotas.items() if count):
            check(stage='pilot-cache-signature', origin=day)
            path, _ = candidates._paths(day)
            require(path.exists(), 'Missing exact admitted daily graph cache; rebuild forbidden')
            document = read(path)
            require(document.get('signature') == candidates.signature and document.get('cache_key') == candidates.cache_key
                    and document.get('state') == 'completed' and document.get('origin') == document.get('cutoff') == day,
                    'Exact daily cache signature differs')
        for name, (dtype, shape) in LAYOUT.items():
            check(stage='pilot-array-allocate', array=name)
            arrays[name] = np.lib.format.open_memmap(output / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
        # Verify and copy the exact frozen table before releasing old maps.
        table_cpu = table.detach().float().cpu().numpy()
        require(table_cpu.shape == (model.vocab_size, 256) and bool(np.isfinite(table_cpu).all()), 'Item table invalid')
        for start in range(0, model.vocab_size, 32):
            require(bool(np.array_equal(table_cpu[start:start + 32], reused['item_table'][start:start + 32])),
                    'Frozen item table differs from reused table')
        with (output / 'item_table.npy').open('xb') as stream:
            with (REUSE_DIR / 'item_table.npy').open('rb') as source:
                shutil.copyfileobj(source, stream, length=1 << 20)
            stream.flush()
            os.fsync(stream.fileno())
        output_positions = {row: index for index, row in enumerate(rows)}
        old_row_set = set(old_rows)
        # Copy all old row arrays verbatim, never call teacher/features/sampler for them.
        for start in range(0, REUSE_ROWS, 32):
            stop = min(start + 32, REUSE_ROWS)
            check(stage='bank-reuse-copy', reused_rows_done=start, reused_rows_total=REUSE_ROWS)
            if start and start % MAP_WINDOW_ROWS == 0:
                remap_maps(np, arrays, output, 'r+')
                remap_maps(np, reused, REUSE_DIR, 'r')
            batch_rows = np.asarray(old_rows[start:stop], dtype=np.int64)
            destinations = [output_positions[int(row)] for row in batch_rows]
            require(bool(np.array_equal(reused['user_ids'][start:stop], provider.splits['train']['user_id'][batch_rows]))
                    and bool(np.array_equal(reused['target_ids'][start:stop], provider.splits['train']['y'][batch_rows]))
                    and bool(np.array_equal(reused['dates'][start:stop], origins[batch_rows])), 'Old train row identity differs')
            for name, value in arrays.items():
                value[destinations] = reused[name][start:stop]
                value.flush()
        remap_maps(np, arrays, output, 'r+')
        close_maps(reused)
        new_work = [(index, row) for index, row in enumerate(rows) if row not in old_row_set]
        require(len(new_work) == ROWS - REUSE_ROWS, 'Exact newly computed row count differs')
        start = 0
        mapped_start = 0
        while start < len(new_work):
            # Homogeneous date is required by strict-origin LiveCandidates.
            stop = start + 1
            while stop < min(start + 32, len(new_work)) and int(origins[new_work[stop][1]]) == int(origins[new_work[start][1]]):
                stop += 1
            if stop - mapped_start > MAP_WINDOW_ROWS:
                remap_maps(np, arrays, output, 'r+')
                mapped_start = start
            check(stage='bank-new-batch', rows_done=REUSE_ROWS + start, rows_total=ROWS,
                  new_rows_done=start, new_rows_total=ROWS - REUSE_ROWS, reused_rows=REUSE_ROWS)
            destinations = [index for index, _ in new_work[start:stop]]
            batch_rows = np.asarray([row for _, row in new_work[start:stop]], dtype=np.int64)
            arrays['row_indices'][destinations] = batch_rows
            arrays['user_ids'][destinations] = provider.splits['train']['user_id'][batch_rows]
            arrays['target_ids'][destinations] = provider.splits['train']['y'][batch_rows]
            arrays['dates'][destinations] = origins[batch_rows]
            with torch.inference_mode():
                x, lengths, y, user = train.batch(provider, batch_rows, 'train', device)
                with train.legacy.autocast(device):
                    encoded = model.encode_context(x, lengths, user, candidate_table=table)
                    logits = model.retrieval_logits(encoded, table)
                ids, source = candidates.sample(batch_rows, 'train', logits, y)
                require(ids.shape == (stop - start, 32) and bool(np.array_equal(ids[:, 0], provider.splits['train']['y'][batch_rows])),
                        'Positive-first fixed32 sample differs')
                raw = np.asarray(provider.candidate_features(batch_rows, ids, 'train'), dtype=np.float32)
                require(raw.shape == (stop - start, 32, 1068) and bool(np.isfinite(raw).all()), 'Raw full1068 invalid')
                base = logits.gather(1, torch.as_tensor(ids, dtype=torch.long, device=device)).float().cpu().numpy()
                encoded_cpu = encoded.float().cpu().numpy()
                require(bool(np.isfinite(base).all()) and bool(np.isfinite(encoded_cpu).all())
                        and bool(np.isfinite(source[:, :, 12:14]).all()), 'Nonfinite bank values')
                arrays['encoded'][destinations] = encoded_cpu
                arrays['candidate_ids'][destinations] = ids
                arrays['base_logits'][destinations] = base
                arrays['raw_features'][destinations] = raw
                arrays['source_values'][destinations] = source[:, :, 12:14]
            for value in arrays.values():
                value.flush()
            start = stop
            del raw, base, encoded_cpu, encoded, logits, ids, source, x, lengths, y, user
        entries = {}
        for name, value in {**arrays, 'item_table': table_cpu}.items():
            check(stage='pilot-array-final-hash', array=name)
            entries[name] = _entry(output / (name + '.npy'), value, output)
        descriptor = dict(contract=CONTRACT, state='completed', rows=ROWS, train_rows=ROWS,
            split='train', seed=SEED, natural_covered_only=False, same_day_positives_excluded=True,
            candidate_width=32, feature_width=1068, context_width=256, arrays=entries,
            teacher=ctx['teacher'], feature_schema_sha256=ctx['feature_schema_sha256'],
            feature_columns=ctx['feature_columns'], source_columns=SOURCE_COLUMNS,
            source_values_policy='Exact original LiveCandidates metadata columns12 and13; frozen E2 logits and log1p strict7d count only',
            sample_policy='All known train rows eligible; target first including natural misses;31 distinct natural mixed1024 negatives excluding all same-user/same-origin positives',
            selection_policy='Nested constrained sample: retain every old32768 row; largest remainder proportional final date quotas; fixed per-date residual draw without replacement; sorted(date,row_index); no coverage filter',
            selection_constraint='All old32768 rows mandatory; this is not an unconstrained fresh262144 sample',
            reused_rows=REUSE_ROWS, new_rows=ROWS - REUSE_ROWS,
            mapping_policy=dict(batch_rows=32, maximum_touched_rows_between_remaps=MAP_WINDOW_ROWS,
                                close_flush_fsync_reopen=True, old_maps_closed_before_new_computation=True,
                                reuse_destination_rows_may_be_noncontiguous=True),
            reuse=dict(bank_dir=REUSE_DIR.as_posix(), bank_manifest_sha256=REUSE_MANIFEST_SHA,
                       contract=old['contract'], reused_rows=REUSE_ROWS, newly_computed_rows=ROWS - REUSE_ROWS,
                       arrays_copied=list(LAYOUT), item_table_copied_verbatim=True,
                       old_arrays_payload_and_headers_verified=True,
                       original_negative_sampler='Pinned original draw uses SHA256(seed:original_train_row); no bank-position or batch RNG',
                       candidate_source_sha256=CANDIDATE_SHA,
                       old_date_quotas=old['date_quotas']),
            date_counts={str(d): n for d, n in sorted(counts.items())},
            date_quotas={str(d): n for d, n in sorted(quotas.items())}, candidate_contract=ctx['candidate_contract'],
            context_signature=context_signature(request), model_training=False, optimizer_created=False, final_test=False)
    finally:
        array_errors = []
        for name, value in list(arrays.items()) + [('reuse_' + n, v) for n, v in reused.items()]:
            try:
                if not name.startswith('reuse_'):
                    value.flush()
            except BaseException as error:
                array_errors.append({'owner': name + '_flush', 'error': repr(error)})
            try:
                mapping = getattr(value, '_mmap', None)
                if mapping is not None:
                    mapping.close()
            except BaseException as error:
                array_errors.append({'owner': name + '_mmap', 'error': repr(error)})
        arrays.clear()
        reused.clear()
        cleanup = close_context(ctx) if ctx is not None else {'teardown_completed': True, 'errors': []}
        cleanup['errors'].extend(array_errors)
        cleanup['teardown_completed'] = not cleanup['errors']
    require(cleanup['teardown_completed'], 'Context teardown failed: ' + repr(cleanup))
    check(stage='pilot-bank-publish')
    descriptor.update(teardown_completed=True, cleanup=cleanup)
    write_new(output / 'bank.json', descriptor)
    return descriptor
