"""One C4 training proposal: uniform distinct negatives from actual retrieval128.

No ML import occurs here. The admitted runtime supplies its already-loaded torch.
The frozen update owns RNG checkpoint/retry and gradient/loss computation.
"""

CONTRACT = 'retrieval128-uniform3-forced-training-positive-v1'


def specification():
    return {'contract': CONTRACT, 'candidate_pool': 'unforced_retrieval_top128',
            'retrieval_pool_dtype': 'float32_after_detach_matching_evaluate',
            'sample_without_replacement': 3, 'sampled_width': 4,
            'force_target_train_only': True,
            'outside_positive_policy': 'positive_column0_plus3_from_original_unforced128',
            'inside_positive_policy': 'exclude_target_from_original128_without_rank129_refill',
            'rng': 'global_Torch_rand_Bx128; frozen_update_restores_CPU_CUDA_RNG_on_AMP_retry',
            'same_seed_does_not_imply_same_baseline_training_rng_trajectory': True,
            'evaluation_target_insertion': False, 'importance_correction': False,
            'raw_config_proposal_description': 'generic_frozen_metadata_only; actual_proposal_bound_by_this_contract'}


def make_sampler(torch, first_item):
    """Bind the runtime's tensor backend without loading a provider or model."""
    if type(first_item) is not int or first_item < 0:
        raise ValueError('Known-catalog start must be an integer')
    audit = {'calls_including_amp_retries': 0, 'rows_including_amp_retries': 0,
             'candidate_width': 4, 'random_width': 128}

    def sample(logits, targets, *, hard_negatives=0, uniform_negatives=3):
        if type(hard_negatives) is not int or type(uniform_negatives) is not int or (hard_negatives, uniform_negatives) != (0, 3):
            raise ValueError('Only H0/U3 at C4 is admitted')
        if logits.ndim != 2 or logits.shape[1] - first_item < 128 or targets.shape != (len(logits),):
            raise ValueError('Need BxV logits, B known targets, and at least128 known items')
        if targets.dtype != torch.long or torch.any(targets < first_item) or torch.any(targets >= logits.shape[1]):
            raise ValueError('Auxiliary positive must be a known int64 token')
        # Match evaluate exactly: retrieve first; the target does not alter support.
        pool = logits[:, first_item:].detach().float().topk(128, dim=1).indices + first_item
        noise = torch.rand(pool.shape, device=logits.device)
        noise.masked_fill_(pool.eq(targets[:, None]), -1.)
        negatives = pool.gather(1, noise.topk(3, dim=1).indices)
        audit['calls_including_amp_retries'] += 1
        audit['rows_including_amp_retries'] += len(targets)
        return torch.cat((targets[:, None], negatives), dim=1)

    sample.sampler_contract = CONTRACT
    sample.audit = audit
    return sample


class InstalledSampler:
    """Replace just the exact loaded train.update global, then restore it."""
    def __init__(self, train):
        self.train = train
        self.original = train.sample_training_candidates
        self.replacement = make_sampler(train.torch, train.FIRST_ITEM)
        self.installed = False

    def install(self):
        if self.installed or self.train.update.__globals__ is not self.train.__dict__:
            raise ValueError('Install once in actual frozen update globals before SinglePass')
        if self.train.update.__globals__.get('sample_training_candidates') is not self.original:
            raise ValueError('Frozen update sampler binding changed')
        self.train.sample_training_candidates = self.replacement
        self.installed = True

    def restore(self):
        if self.installed:
            if self.train.sample_training_candidates is not self.replacement:
                raise ValueError('Installed sampler was replaced by another owner')
            self.train.sample_training_candidates = self.original
            self.installed = False


def install_benchmark_sampler(train):
    """Only the benchmark compares unchanged H2/U1 against the new H0/U3.

    Called by the root probe's pinned loader for the exact frozen train module.
    No alternative full training variant is launched by this adapter.
    """
    if train.update.__globals__ is not train.__dict__:
        raise ValueError('Benchmark must patch actual frozen update globals')
    original = train.sample_training_candidates
    treatment = make_sampler(train.torch, train.FIRST_ITEM)
    evidence = {'contract': CONTRACT, 'baseline_calls': 0, 'treatment_calls': 0,
                'treatment_rows': 0, 'outside_positive_rows': 0, 'support_checks_passed': True}

    def sample(logits, targets, *, hard_negatives, uniform_negatives):
        if (hard_negatives, uniform_negatives) == (2, 1):
            evidence['baseline_calls'] += 1
            return original(logits, targets, hard_negatives=2, uniform_negatives=1)
        if (hard_negatives, uniform_negatives) != (0, 3):
            raise ValueError('Benchmark admits only baseline H2/U1 and retrieval128 H0/U3')
        result = treatment(logits, targets, hard_negatives=0, uniform_negatives=3)
        pool = logits[:, train.FIRST_ITEM:].detach().float().topk(128, dim=1).indices + train.FIRST_ITEM
        # Timing explicitly includes these benchmark-only source correctness checks.
        if not result[:, 1:, None].eq(pool[:, None, :]).any(dim=2).all().item():
            evidence['support_checks_passed'] = False
            raise ValueError('Treatment negative escaped original retrieval128')
        evidence['treatment_calls'] += 1
        evidence['treatment_rows'] += len(targets)
        evidence['outside_positive_rows'] += int((~pool.eq(targets[:, None]).any(dim=1)).sum().item())
        return result

    train.sample_training_candidates = sample
    return evidence
