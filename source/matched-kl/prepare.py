"""Root-only metadata preparation. No numerical/model imports or implicit launches."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import math

Z = Path(__file__).resolve().parent
F = Z.parent / 'c-rl-final-20261025-r1'
O = Z.parent / 'c-rl-20261024-r1'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def utc():
    return datetime.now(timezone.utc).isoformat()

def bootstrap():
    assert not (Z/'campaign.json').exists()
    protocol = dict(
        contract='kion-C1070-train-KL-seeds-preregistered-v1', registered_utc=utc(),
        authority='User: Ставь goal, following matched-KL, multi-seed, short-report proposal',
        objective='Bounded head-only comparison of two frozen rewards at similar TRAIN512 candidate-pool forward KL to original C; no causal claim',
        seeds=[20261024,20261025,20261026], calibration_seed=20261024,
        seed_scope='seed24 reused completion plus selected calibration residual; seeds25/26 prospectively paired. Three seeds are descriptive, not three independent confirmations.',
        primary_metric='MAP@10 on all fixed dev4096 states, latest1000 endpoint',
        secondary_metrics=['NDCG@10','Recall@10','screen256 completion and residual utilities','KL','exposure'],
        reference='original frozen C1070; unchanged encoder/table/provider and frozen RM-r2',
        TRAIN_KL='equal mean across all512 fixed train states of sum over all1024 actions pi*(log(pi)-log(C)); temperature1; same producer-slot FP16 packing',
        target='T = TRAIN_KL(old completion seed20261024 head-1000); finite T>1e-8 required',
        initial_measurement='Before any new fit evaluate C, old completion1000, old residual1000 on all512; exact C/cache/top10 parity required.',
        calibration=dict(
            beta1='clip(0.5*sqrt(K_old_residual/T),0.01,2.0)',
            first_pilot='residual seed24 original-C start, exactly1000 updates',
            stop_if_ratio_in=[0.8,1.2],
            beta2='only if first pilot ratio outside tolerance: clip(beta1*sqrt(K1/T),0.01,2.0); do not repeat if identical beta',
            second_pilot='at most one further residual seed24 original-C fit, exactly1000 updates',
            selection='among finite positive pilot TRAIN KL choose smallest abs(log(K/T)); tie larger beta; if none valid stop branch',
            freeze='Before any seed25/26 fit freeze residual beta and selected seed24 pilot checkpoint. Reuse this pilot as final seed24 arm; no duplicate fit.',
            forbidden='No dev MAP/reward selection, no third beta, no per-seed retuning, no best checkpoint/seed selection'),
        final_arms='completion beta0.5 and residual frozen beta, each seeds24/25/26, latest1000. Reuse historical completion24 and selected pilot24.',
        matched_claim='Only if for EACH seed TRAIN_KL(residual)/TRAIN_KL(completion) is in [0.8,1.2]. Otherwise whole attempt labelled not matched; report all outcomes without adaptive retuning.',
        match_limit='Operational scalar candidate-pool KL tolerance, not statistical equivalence, per-state/fullcatalog matching or causal isolation.',
        fit_budget=dict(maximum_new_fits=8, maximum_scientific_fits=6, pilot_fits_max=2, prospective_fits=4,
                        technical_retries_max=2, maximum_updates_per_fit=1000,
                        retries='Only diagnosed technical failure; every admitted launched attempt counts including failed attempts; no retry to improve metrics.'),
        optimizer=dict(start_from='original-C', updates=1000, logical_batch=16, sampler='Python random.Random(seed).choice over supported train states, with replacement',
                       optimizer='Adam', learning_rate=1e-5, weight_decay=0, grad_clip=1, precision='cuda-fp16', checkpoints=[0,250,500,1000],
                       AMP='Initial scale1024; skipped update is failure', temperature=1,
                       utility='Frozen support; C-weighted supported centering; unsupported advantage0; all1024 actions normalized'),
        inference='One shared raw-feature dev4096 pass for C plus exactly six final arms on identical natural1024; cached screen256 diagnostics separately; no target injection/seen filter.',
        inference_statistics=dict(bootstrap='2000 paired user-cluster draws shared across arms, fixed seed20261024; conditional on fitted heads, reused dev, unadjusted',
                                  seed_stats='Show each paired seed delta, mean/range/sampleSD across3, and separate descriptive newseed25/26 mean; no seed-level significance from n3'),
        exclusions=['new supervised epochs','new RM fits','final-test reads','external downloads','paid operations','uploads/publication/submission','commit/push'],
        execution=dict(owner='root only actual native numerical/model jobs', global_heavy_jobs=1, cpu_threads=4, cap_gib=[4,8], reserve_gib=2, disk_reserve_gib=15,
                       no_new_fit_after='2026-10-25T09:25:00+03:00', numerical_deadline='2026-10-25T09:45:00+03:00', report_deadline='2026-10-25T10:00:00+03:00',
                       old_roots='F/O STOP and completed source/report/archive immutable',
                       honesty='If incomplete by cutoff save checkpoint, do not mark goal complete; no invented result'),
        amendments_to_proposal=['MAP@10 primary for consistency with user/research; NDCG secondary',
                                'Reuse selected calibration seed24 pilot as final arm, avoiding redundant identical fit',
                                'Maximum6 scientific fits plus up to2 technical retries within total8',
                                '1000 optimizer updates and1024 candidate actions are different dimensions; no missing24 updates'])
    write(Z/'protocols/preregistration.json', protocol)
    old = read(F/'campaign.json')
    campaign = dict(contract=old['contract'], state='running', started_utc=utc(), deadline_utc='2026-10-25T09:45:00+03:00',
                    handoff_deadline='2026-10-25T10:00:00+03:00', authorization=protocol['authority'], coordinator='root', round_limit=3,
                    plan=dict(path=(Z/'protocols/preregistration.json').as_posix(),sha256=sha(Z/'protocols/preregistration.json')),
                    admitted_requests={}, limits=old['limits'], fit_limit_global=8)
    write(Z/'campaign.json', campaign)
    write(Z/'state.json',dict(state='running',stage='preregistered-awaiting-review',updated_utc=utc(),next='Independent protocol review, then nonfit train512 probe',pending_operations=[]))
    write(Z/'fit-ledger.json',dict(contract='root-global-fit-reservations-v1',maximum_new_fits=8,reservations=[]))
    (Z/'operations').mkdir(exist_ok=True)
    (Z/'JOURNAL.md').write_text('# Matched TRAIN KL / fixed seeds follow-up\n\n'+utc()+': User requested goal. New namespace created; old completed F/O remain read-only with STOP. Preregistered before new numeric results.\n', encoding='utf-8', newline='\n')
    queue_path=O/'queue.json'
    queue=read(queue_path)
    queue['active_execution_root']=Z.as_posix()
    queue['execution_state']='running-matched-KL-followup'
    queue['updated_utc']=utc()
    queue['matched_KL_followup_steps']=[dict(id='K'+str(i+1),question=q,state='ready' if i==0 else 'candidate') for i,q in enumerate([
        'Exact train512 target and bounded residual beta calibration', 'Four prospective paired seed fits with frozen beta',
        'One shared dev4096 and cached screen evaluation plus conditional bootstrap', 'Polished short report and independently verified reproducible supplement'])]
    write(queue_path,queue)
    plan=Z.parents[2]/'docs/task2-next-plan.md'
    existing=plan.read_text(encoding='utf-8')
    pre='# Активное дополнение: сопоставимый KL и три seed, 25 октября 2026\n\nПользователь явно попросил новый goal («Ставь goal») после завершённого отчёта. Новый локальный execution root: `artifacts/task2-kion/c-kl-seeds-20261025-r1/`. Протокол фиксируется до численных запусков: две награды при близком TRAIN512 KL, seed24 для калибровки и два новых seed25/26, максимум8 новых RL fit по1000 updates (научных≤6, остальные только технические повторы). Никаких новых supervised-эпох/RM fit/final-test reads. Один heavy job, CPU4, cap4/8+reserve2GiB. Не начинать fit после09:25, численные jobs до09:45, отчёт до10:00МСК25октября; при нехватке времени честный checkpoint. Старый завершённый отчёт, ZIP и STOP не изменяются. Метрика MAP@10; reused development4096 и screen256 показаны отдельно. Сопоставимость KL — операционный допуск[0.8,1.2] на каждой паре seed, без обещания статистической/каузальной эквивалентности. Подробный новый preregistration — локальное необязательное дополнение в указанном root; может отсутствовать в Git.\n\n---\n\n'
    plan.write_text(pre+existing,encoding='utf-8',newline='\n')

def base_request(op):
    r=read(F/'request-residual-fit-r1.json')
    r.update(operation=op, deadline_utc='2026-10-25T09:45:00+03:00', maximum_seconds=900,
             stage='evaluate', module_path=(Z/'policy/policy.py').as_posix(),function='run', mode='evaluate_train512',
             utility='completion',beta=.5,seed=20261024,rl_updates=0,
             owned_output_roots=[(Z/'policy').as_posix()], required_output_bytes={'D:':10000000},cap_gib=8)
    for p in (Z/'policy/policy.py', *(Z/'control'/name for name in ['common.py','launch.py','supervise.py','worker.py'])):
        r['source_hashes'][p.as_posix()]=sha(p)
    for p in (Z/'protocols/preregistration.json',):
        r['input_hashes'][p.as_posix()]=sha(p)
    return r

def descriptor(name,path,utility):
    return dict(name=name,path=path.as_posix(),sha256=sha(path),step=1000,utility=utility)

def save_request(r):
    p=Z/('request-'+r['operation']+'.json')
    assert not p.exists()
    write(p,r)
    print(json.dumps(dict(request=p.as_posix(),sha256=sha(p)),ensure_ascii=False))

def probe():
    r=base_request('train-target-r1')
    r['checkpoints']=[descriptor('completion-20261024',O/'policy/rl-r2/head-1000.pt','completion'),descriptor('residual-old-20261024',F/'policy/residual-r1/head-1000.pt','residual')]
    r['evaluation_output']=(Z/'policy/train-target-r1.json').as_posix()
    r['output_paths']=[r['evaluation_output']]
    for d in r['checkpoints']:
        r['input_hashes'][d['path']]=d['sha256']
    save_request(r)

def admit(op):
    # Called by root ONLY after review and before actual launch, with no worker active.
    path=Z/('request-'+op+'.json')
    request=read(path)
    campaign=read(Z/'campaign.json')
    assert op not in campaign['admitted_requests']
    campaign['admitted_requests'][op]=dict(request_path=path.as_posix(),sha256=sha(path),admitted_utc=utc(),review='root+independent actual request review')
    write(Z/'campaign.json',campaign)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['bootstrap','probe','admit']);p.add_argument('operation',nargs='?');a=p.parse_args()
    if a.action=='admit':admit(a.operation)
    else:globals()[a.action]()
