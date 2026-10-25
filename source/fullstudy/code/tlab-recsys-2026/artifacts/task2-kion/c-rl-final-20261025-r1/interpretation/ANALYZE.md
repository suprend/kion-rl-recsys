# Сопоставленный анализ трёх политик

`analyze.py` реализует анализ сохранённых outputs C, RL-completion latest1000 и
RL-residual latest1000. Код и stdlib tests подготовлены; их наличие не означает,
что реальные массивы, bootstrap или рисунки уже рассчитаны. Только завершённый
`analysis.json` доказывает numerical run; визуальная проверка рисунков проводится
после их реального рендера.

## Вход и запуск владельцем numerical jobs

`run(request, check=None)` совместим с контролируемым Python entrypoint. CLI:

```powershell
.\pas.ps1 python 'D:\tlab-recsys-2026\artifacts\task2-kion\c-rl-final-20261025-r1\interpretation\analyze.py' --request '<analysis-request.json>'
```

Запуск из `D:/personal-agent-system`. CLI не заменяет допуск общего контроллера,
ресурсный лимит, deadline и единственного владельца numerical jobs. `check` при
контролируемом запуске получает стадии integrity/rank/bootstrap/decomposition.
NumPy импортируется только внутри `run`; моделей, provider, raw dataset, GPU,
training, network и final-test reads в анализаторе нет.

Request содержит три обязательных поля и необязательную кривую checkpoints:

```json
{
  "protocol": {
    "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/protocols/evaluation.json",
    "sha256": "ee0e1e06f060f27e4dd149a8c26c02827f9b1c9fb67a57d688d84e166fddfd25"
  },
  "evaluation": {"path": "<completed cohort>/evaluation.json", "sha256": "<actual SHA256>"},
  "output": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/<fresh result directory>",
  "checkpoint_curve": [
    {"step": 0, "path": "<legacy>/policy/screen-0-r1.json", "sha256": "<actual SHA256>"},
    {"step": 250, "path": "<legacy>/policy/screen-250-r1.json", "sha256": "<actual SHA256>"},
    {"step": 500, "path": "<legacy>/policy/screen-500-r1.json", "sha256": "<actual SHA256>"},
    {"step": 1000, "path": "<legacy>/policy/screen-1000-r1.json", "sha256": "<actual SHA256>"}
  ]
}
```

Пути placeholders нужно заменить реальными pinned files. Запись выполняется
только в новый output внутри `interpretation`; существующий результат не
перезаписывается. Отдельный request используется для каждого cohort. Отсутствие
full4096 допускает честный screen256 результат, но не подмену metadata full4096.

Evaluation contract — `kion-scientific3-evaluation-v1`, state `completed`, stage
`screen` (ровно256) или `full4096` (ровно4096). Обязательны denominator=rows,
natural1024, temperature1, score-desc/raw-ID-asc ties, no target injection,
no seen filter, all unknown/off-pool misses и успешная original-C parity.
Каждый RL checkpoint должен быть фактическим latest1000, а не selected0.
Дескрипторы `.npy` проверяются по size/SHA/shape/dtype; pickle запрещён.

Common arrays: `user_id,row_index,target_token,known_target,candidate_coverage,pool_ids`,
а также `<arm>_scores` и `<arm>_top10`. Full получает `catalog_raw_ids` и
проверяет122 unknown/616 known off-pool. Screen получает `candidate_raw_ids`,
`raw_reward,utility_residual,utility_prior,support,fallback,history_tokens,strict7d_count`;
`candidate_sources` (14 каналов) опционален. Screen `catalog_size` задаётся producer
из полного item-table, а не числом уникальных кандидатов в выборке.

## Определения

Ранжирование независимо восстанавливается по всем1024 scores. Сохранённый top10
сверяется точно; для каждого known covered target сохраняется полный ранг1–1024.
Unknown и known off-pool имеют пустой rank и нулевые MAP/NDCG/Recall. При одном
observed target MAP@10 здесь равен RR@10; это явно ограничивает интерпретацию.

Bootstrap использует2000 общих multinomial resamples уникальных пользователей,
seed20261024, с переносом всех запросов выбранного пользователя. Point estimator
и каждый resample — query-weighted ratio sums/counts. Девять колонок = три метрики
для трёх пар; draws общие. CI — percentile2.5/97.5 с NumPy linear quantiles,
без multiple-comparison correction. Это не CI между повторными fits и не
причинный inference. Screen и full имеют отдельные bootstrap schedules/cohorts.

Для каждой screen utility `u` один раз замораживается baseline:

`a_u(s,i) = support(s,i) * (u(s,i) - sum_j p_C(j|s)*support(s,j)*u(s,j) / sum_j p_C(j|s)*support(s,j))`.

Если state не имеет support, все его advantages равны0. Все1024 actions остаются
в softmax; unsupported не получают выдуманный outcome0. Proxy политики —
`mean_s sum_i p_arm(i|s)*a_u(s,i)`. Это считается для raw completion, residual и
observed-fit prior отдельно; prior — диагностическая величина, не третий fit.

Точное разложение для любой пары:
`contribution(item) = mean_s sum_{i:rawID=item} (p_right-p_left)*a_u`.
Сумма всех item contributions проверяется против разности двух centered proxies
с absolute tolerance1e−12. Дополнительно сохраняются полный probability shift
и supported-probability shift. Разность conditional raw means имеет другие
arm-specific denominators и не используется как ожидаемый прирост полезности.

Query exposure даёт каждому запросу вес1; equal-user exposure — вес1/n_queries
каждому запросу пользователя. Top10 HHI считается по всем рекомендованным слотам,
effective catalog=1/HHI. Полные ranked shares доступны в compact JSON. Это
рекомендованная distribution, не реальные показы. H63-seen — membership в исходном
observed64-token prefix после исключения BOS/PAD/UNK (<3); это усечённая история.
Strict7d — `log1p` локального count в исходном origin. Invalid/missing counts не
становятся нулями. Full-vocabulary denominator для self-information отсутствует:
novelty остаётся явно unavailable. Source channels0–11 показываются как mean values
в выбранном top10 с именами producer metadata; source12/13 отдельно не называются
owner flags. Все14 каналов не сжимаются в выдуманный единичный owner.

## Результаты и переносимые рисунки

- `analysis.json`: complete provenance, cohort, metrics, paired CI, exposure,
  screen diagnostics и exact decomposition.
- `plot-data.json`: тот же aggregate без user IDs; достаточный Level A вход.
- `paired-metrics.csv`: compact paired differences/CI/gain-loss-tie.
- `per-query.csv`: row/user bindings, ranks, metrics; локальное проверочное
  приложение, не обязательный переносимый пакет.
- `item-shifts.csv`: полный каталог contributions для всех трёх пар и utilities;
  только для screen. В JSON — top20 по |contribution| и точный остаток.

Рендер из compact JSON:

```powershell
.\pas.ps1 python 'D:\tlab-recsys-2026\artifacts\task2-kion\c-rl-final-20261025-r1\interpretation\plot_results.py' --input '<result>/plot-data.json' --output '<fresh figures directory>'
```

`plot_results.py` читает только JSON; Matplotlib импортируется только в `render`.
Поддерживается установленный Matplotlib3.11.2; SVG сохраняет редактируемый текст,
PNG —190dpi, шрифт DejaVu Sans поддерживает русский. Один cohort рендерится за
один запуск. Фигуры: legacy checkpoint/gates (если вход есть), matched relevance
и concentration, paired CI, exposure two weightings, full target ranks;
screen дополнительно utilities/behavior и exact item contributions. Legacy curve
всегда подписана как screen256, даже если расположена рядом с full4096 figures.
`figures.json` содержит source hash, canonical aggregate hash, версию Matplotlib,
подписи и `requires_visual_QA=true`: визуальный осмотр
PNG после root-run обязателен перед заявлением о финальной читаемости.

Нужны NumPy для actual analysis и Matplotlib для actual rendering; stdlib tests
не требуют ни одной из этих библиотек:

```powershell
.\pas.ps1 python -m unittest discover -s 'D:\tlab-recsys-2026\artifacts\task2-kion\c-rl-final-20261025-r1\interpretation' -p 'test_analyze.py' -v
.\pas.ps1 python -m unittest discover -s 'D:\tlab-recsys-2026\artifacts\task2-kion\c-rl-final-20261025-r1\interpretation' -p 'test_plot_source.py' -v
```

Ограничения: reused development, один fit на utility, fixedβ, не matched KL;
RM support-limited proxy, observed-fit prior и synthetic prefixes не устанавливают
персонализацию, reward hacking, causal utility или долгосрочный retention. В full4096
нет RM scoring и нельзя приписывать ему screen utility values.
