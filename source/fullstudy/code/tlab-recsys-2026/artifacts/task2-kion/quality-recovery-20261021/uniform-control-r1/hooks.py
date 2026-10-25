"""One process-local E1 sidecar hook; no ML imports or alternate training code."""
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError('Uniform singlepass hook: ' + message)


class SinglePass:
    def __init__(self, train, sidecar, snapshot):
        self.train, self.sidecar, self.snapshot = train, Path(sidecar), snapshot
        self.originals = {name: getattr(train, name) for name in ('update', 'evaluate', 'publish_selected')}
        self.updates, self.examples, self.evaluations, self.publications = 0, 0, 0, 0
        self.objects, self.evaluated, self.metrics = None, None, None
        self.evidence = {'evaluate_calls': 0, 'selected_publications': 0, 'evaluation_succeeded': False}

    def install(self):
        require(self.train.run.__globals__ is self.train.__dict__, 'original run belongs to this exact imported module')
        for name in self.originals:
            require(self.train.run.__globals__[name] is self.originals[name], 'original actual run global: ' + name)
        self.train.update, self.train.evaluate, self.train.publish_selected = self.update, self.evaluate, self.publish_selected
        require(self.train.run.__globals__['evaluate'] == self.evaluate
            and self.train.run.__globals__['update'] == self.update
            and self.train.run.__globals__['publish_selected'] == self.publish_selected, 'actual original run globals use installed wrappers')
        self.evidence['actual_run_global_binding_verified'] = True

    def restore(self):
        for name, value in self.originals.items():
            setattr(self.train, name, value)

    def update(self, *args, **kwargs):
        require(self.evaluations == 0, 'no optimizer update after sole evaluation')
        require(len(args) == 6 and set(kwargs) == {'hard_negatives', 'uniform_negatives', 'auxiliary_weight', 'candidate_chunk_size', 'collect_loss'},
            'original exact update argument form')
        model, provider, batches, optimizer, scaler, device = args
        require(kwargs['hard_negatives'] == 0 and kwargs['uniform_negatives'] == 3 and kwargs['auxiliary_weight'] == 1.
            and kwargs['candidate_chunk_size'] == 64 and type(kwargs['collect_loss']) is bool and str(device) == 'cuda', 'only admitted H0/U3/lambda1/chunk64/CUDA update')
        require(len(batches) == 1 and 1 <= len(batches[0]) <= 128, 'originalB128/accum1 including origin tails')
        objects = (model, provider, optimizer, scaler)
        if self.objects is None:
            self.objects = objects
        require(all(a is b for a, b in zip(objects, self.objects)), 'same model/provider/optimizer/scaler throughout original run')
        result = self.originals['update'](*args, **kwargs)
        self.updates += 1
        self.examples += sum(map(len, batches))
        if self.updates == 1:
            loss, retries = result
            origins = provider.origin_keys('train')
            require(len(batches[0]) == 128 and all(int(origins[int(row)]) == 737869 for row in batches[0]), 'first origin737869 and128rows')
            golden = 11.080821990966797
            require(isinstance(loss, dict) and all(type(v) in (int, float) and math.isfinite(v) for v in loss.values())
                and abs(loss['retrieval_ce'] - golden) <= max(.01, golden * .001), 'firstforward retrievalCE golden; new auxiliary/total only require finite')
            self.evidence['firstforward'] = {'origin': 737869, 'examples': 128, 'golden_retrieval_ce': golden,
                'actual_loss': dict(loss), 'amp_retries': retries, 'absolute_tolerance': .01, 'relative_tolerance': .001}
        return result

    def evaluate(self, *args, **kwargs):
        require(self.evaluations == 0 and self.updates == 27235 and self.examples == 3477437, 'sole evaluate only after exact fullE1 updates/rows')
        require(len(args) == 4 and self.objects is not None, 'original evaluator positional arguments')
        model, provider, batch_size, device = args
        require(model is self.objects[0] and provider is self.objects[1] and type(batch_size) is int and batch_size == 128
            and str(device) == 'cuda' and len(provider.splits['val']['y']) == 158760, 'same objects, fullvalidation andB128/CUDA')
        require(set(kwargs) == {'retrieval_topk', 'candidate_chunk_size', 'scoring_mode', 'should_stop'}
            and kwargs['retrieval_topk'] == 128 and kwargs['candidate_chunk_size'] == 64
            and kwargs['scoring_mode'] == 'rerank' and callable(kwargs['should_stop']), 'no subset/other path/changed evaluator arguments')
        require(not self.sidecar.exists() and not self.sidecar.with_suffix('.npz.tmp').exists(), 'fresh single writer sidecar and temporary path')
        self.evaluations += 1
        self.evidence['evaluate_calls'] = self.evaluations
        before = self.snapshot(*self.objects)
        forwarded = dict(kwargs)
        forwarded['predictions_path'] = self.sidecar
        metrics = self.originals['evaluate'](*args, **forwarded)
        after = self.snapshot(*self.objects)
        require(before == after, 'evaluated weights/optimizer/scaler/RNG unchanged by original evaluator plus serialization')
        require(type(metrics.get('examples')) is int and metrics['examples'] == 158760
            and type(metrics.get('known_targets')) is int and metrics['known_targets'] == 154564
            and type(metrics.get('unknown_target_misses')) is int and metrics['unknown_target_misses'] == 4196,
            'exact complete validation rows/known/unknown counts')
        require(self.sidecar.is_file() and not self.sidecar.with_suffix('.npz.tmp').exists(), 'original exporter atomically committed sidecar')
        self.evaluated, self.metrics = before, metrics
        self.evidence.update(evaluation_succeeded=True, evaluated_snapshot=before,
            evaluator_arguments={'batch_size': 128, 'retrieval_topk': 128, 'candidate_chunk_size': 64, 'scoring_mode': 'rerank', 'indices': None},
            extra_argument_only='predictions_path', no_extra_forward=True)
        return metrics

    def publish_selected(self, *args, **kwargs):
        require(self.publications == 0 and self.evaluations == 1 and self.evaluated is not None and not kwargs and len(args) == 6,
            'one original selected publication after successful hooked evaluation')
        path, model, optimizer, scaler, state, config = args
        require(Path(path).name == 'selected-epoch0001.pt' and model is self.objects[0] and optimizer is self.objects[2]
            and scaler is self.objects[3], 'same original selected model/optimizer/scaler')
        require(all(type(state.get(k)) is int and state[k] == v for k, v in
            (('epoch', 1), ('offset', 0), ('updates', 27235), ('examples', 3477437), ('selected_epoch', 1)))
            and state['history'] == [{'epoch': 1, 'validation': self.metrics}], 'exact selected E1 cursor/history')
        before = self.snapshot(*self.objects)
        require(before == self.evaluated, 'selected publication state equals evaluated weights/optimizer/scaler/RNG')
        result = self.originals['publish_selected'](*args, **kwargs)
        require(self.snapshot(*self.objects) == before, 'original selected save is read-only for evaluated state')
        self.publications += 1
        self.evidence.update(selected_publications=1, selected_save_snapshot=before)
        return result

    def finished(self):
        require(self.updates == 27235 and self.examples == 3477437 and self.evaluations == 1
            and self.publications == 1 and self.evaluated is not None, 'full trained/validated/selected E1 hook lifecycle')
        return {**self.evidence, 'successful_updates': self.updates, 'train_examples': self.examples,
            'optimizer_updates_after_evaluation': 0, 'timing_label': 'validation-plus-sidecar-export-plus-readonly-state-digests'}
