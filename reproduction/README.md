# Воспроизведение решения KION

Все команды выполняются из корня распакованного комплекта. Нужен обычный Python 3.12; локальные launchers автора и исходные диски не требуются. Рабочие окружения описаны в [ENVIRONMENTS.md](ENVIRONMENTS.md), состав входов — в [INPUTS.md](INPUTS.md), карта передачи — в [HAND_IN.md](HAND_IN.md).

| Уровень | Что повторяется | Что это подтверждает |
|---|---|---|
| A | Проверка включённого кода и компактных результатов; построение рисунков из агрегатов | Целостность поставки, synthetic ranking arithmetic и рисунки из сохранённых агрегатов |
| B | Inference замороженных моделей на ограниченном fixture | Только явно проверенные модель, входы, устройство и precision; без обучения |
| C | Рецепт подготовки raw KION, GPT, supervised heads, RM и RL | Описанное происхождение исследования; новый полный прогон в этой поставке не выполнялся |

Успешный A не означает успешный B. Численный fixture не заменяет всю выборку development 4096, повторное обучение или оценку на новом тесте.

## A. Начать без моделей и данных

Проверь сохранённые source snapshots стандартной библиотекой:

```text
python -B source/verify_sources.py
python -B source/fullstudy/code/tools/verify_bundle.py --report-root source/fullstudy
python -B reproduction/kion_replay verify --bundle .
```

Первый checker проверяет SHA256, размеры и Python syntax 284 сохранённых файлов (257 Python). Второй независимо проверяет вложенный historical bundle из 261 файла и MAP/Recall/NDCG на трёх вымышленных запросах. Третий проверяет новый portable runtime по `reproduction/runtime-manifest.json`. Эти команды не импортируют producer modules и не открывают ML-массивы. Ожидаемые synthetic значения: MAP@10=0,5, Recall@10=2/3, NDCG@10=0,5436432511904858.

Точные версии, внешние входы и original requests сохранены в [source manifest](../source/manifest.json), [dependencies](../source/fullstudy/dependencies.json) и [recipes](../source/fullstudy/code/recipes/source-requests.json). Historical requests фиксируют происхождение; старые deadlines, пути и output directories не являются командами для повторного запуска.

Для пересборки screen/full рисунков из компактных агрегатов выбери figures environment из [ENVIRONMENTS.md](ENVIRONMENTS.md). Следующие outputs должны быть новыми каталогами:

```text
python -B source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/plot_results_v4.py --input evidence/fullstudy/plot-data-screen.json --labels evidence/fullstudy/figure-provenance/screen/labels.json --output replay-figures/screen
python -B source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/plot_results_v4.py --input evidence/fullstudy/plot-data-full.json --labels evidence/fullstudy/figure-provenance/full/labels.json --output replay-figures/full
```

Здесь читаются сохранённые aggregate JSON и компактные названия фильмов; модели и raw CSV не нужны. Набор чисел сохранится, но шрифты и изображения могут отличаться на другой render-среде. Готовые rollout/matched-KL рисунки и их manifests входят в комплект; их historical builders имеют отдельные input/request contracts.

## B. Ограниченный numerical replay

**Frozen GPT + три головы фактически проверены.** В новой распаковке проверочного runtime+inputs архива, в пути с пробелами на исходной Windows-машине, `frozen-encoder / CUDA FP16 / exact` завершился за 9,781 с: q, retrieval logits и все scores/top10 C/completion/residual совпали с historical oracle; maximum error 0. Процесс завершён и quiescent, optimizer updates 0. [Summary](../evidence/reproduction/actor-replay-summary.json), [runtime receipt](../evidence/reproduction/actor-replay-receipt.json), [native receipt](../evidence/reproduction/actor-replay-native.json), [extraction check](../evidence/reproduction/actor-extraction.json).

Проверены именно новые portable source bytes и включённые model inputs. Оба replay выполнены из новых минимальных проверочных архивов на исходной машине с уже установленными раздельными средами. Установка зависимостей на новой машине и численный повтор всего final delivery archive отдельно не заявляются; финальная упаковка проверяется отдельно. Независимые запуски команды `cached-features`, CPU FP32, full-cohort inference и новое обучение не выполнялись. Отдельный frozen RM replay также принят: настоящая CBM prediction для четырёх contexts × 1024 candidates завершилась за 4,407 с, float64 maximum error 0, float32 reward и support совпали точно (3976 из 4096 pairs supported); RM fits/updates 0, процесс quiescent. [RM summary](../evidence/reproduction/rm-replay-summary.json), [runtime receipt](../evidence/reproduction/rm-replay-receipt.json), [native receipt](../evidence/reproduction/rm-replay-native.json), [extraction check](../evidence/reproduction/rm-extraction.json).

| Путь | Входы и пересчитываемая часть | Граница |
|---|---|---|
| `cached-features` | Четыре сохранённых q, item table, raw 1068 + source 2, три C/RL головы → scores и top10 | GPT q и feature provider не пересчитываются |
| `frozen-encoder` | Полные 16 producer token-prefix peers и user vectors → GPT q/retrieval; первые четыре slots → C/completion/residual scores | Frozen item table и raw features уже экспортированы; raw-to-feature pipeline и E5 не запускаются |
| `reward` | Для тех же четырёх contexts × 1024 candidates (4096 candidate pairs) сохранённые RM features → настоящие CatBoost predictions и calibration | Отдельный CPU process; feature generation и RM fit не выполняются |

Три replay heads — исходная C, completion seed 20261024 и исходная residual seed 20261024 с beta 0,5. Они не являются шестью головами matched-KL продолжения.

Fixture содержит преобразованные истории item-token и user vectors. Удаление source user IDs не является гарантией анонимности. В нём нет observed target/reward labels, поэтому этот replay проверяет score/rank parity; reported development MAP/CI на 4096 queries остаются сохранёнными результатами исходной оценки и не пересчитываются этим fixture. Five-step diagnostic и six-head matched-KL evaluation остаются результатами полного исследования, а не новыми командами этого ограниченного replay.

Сначала выполните stdlib проверку runtime и fixture. [Runtime manifest](runtime-manifest.json), [actor inputs](inputs/manifest.json) и [RM inputs](inputs-rm-r1/manifest.json) фиксируют точные bytes; финальный archive allowlist находится в `PACKAGE_MANIFEST.json`:

```text
python -B reproduction/kion_replay verify --bundle . --inputs reproduction/inputs
python -B reproduction/kion_replay verify --bundle . --inputs reproduction/inputs-rm-r1
```

Для всех команд `infer` ниже `python` означает интерпретатор actor environment из [ENVIRONMENTS.md](ENVIRONMENTS.md), например `./.venv-actor/Scripts/python.exe` на Windows или `./.venv-actor/bin/python` на Linux. В PowerShell полный путь вызывается с `&`. Каждый output — отдельный новый каталог с существующим parent:

```text
python -B reproduction/kion_replay infer --bundle . --inputs reproduction/inputs --output replay-cached-cuda --tier cached-features --device cuda --precision cuda-fp16 --oracle exact --cpu-threads 4
python -B reproduction/kion_replay infer --bundle . --inputs reproduction/inputs --output replay-encoder-cuda --tier frozen-encoder --device cuda --precision cuda-fp16 --oracle exact --cpu-threads 4
```

`exact` требует совпадения всех score values и, для encoder tier, q/base. Полный producer batch B16 и slots 0–3 сохранены. При расхождении процесс завершается ошибкой; автоматической смены устройства или ослабления oracle нет. Это сравнение ограниченного fixture с historical CUDA oracle на подходящем runtime.

На CPU FP32 проверяется отдельное исполнение, без утверждения exact historical parity:

```text
python -B reproduction/kion_replay infer --bundle . --inputs reproduction/inputs --output replay-encoder-cpu --tier frozen-encoder --device cpu --precision fp32 --oracle none --cpu-threads 4
```

`none` сохраняет описательные score errors и top10 agreement с oracle после настоящего inference, но не назначает допустимые пределы; `oracle_passed` не становится `true`. Даже если какие-то значения совпали, это не заменяет отдельный gate для нового backend.

Для команды `reward` замени `python` на интерпретатор отдельного reward-worker environment: `./.venv-rm/Scripts/python.exe` на Windows или `./.venv-rm/bin/python` на Linux; в PowerShell добавь `&` перед путём. Scorer выполняет настоящую CBM prediction, затем clip → affine calibration → clip и cold fallback; residual utility использует frozen fit prior. Точные формула/schema и source pins — в [RM_PORTABILITY.md](../research/RM_PORTABILITY.md):

```text
python -B reproduction/kion_replay reward --bundle . --inputs reproduction/inputs-rm-r1 --output replay-rm
```

Outputs actor: `scores-C.npy`, `scores-RL-completion.npy`, `scores-RL-residual.npy`, `summary.json` и `run-receipt.json`; top10 tokens/columns записаны в summary. Encoder tier дополнительно сохраняет `encoded.npy`/`base_logits.npy`. RM сохраняет `rm_reward64.npy`, `rm_reward.npy`, `rm_support.npy`, `rm_prior.npy`, `rm_utility.npy`, `rm-summary.json`, `run-receipt.json`. Проверяйте exit code, статус, input/runtime hashes, precision и oracle mode в receipt; факт появления output directory не означает pass.

Результаты записываются в новые outputs, fixtures проверяются до и после inference. Portable `.pt` exports загружаются с `weights_only=True`; модели остаются в eval/inference mode, optimizer не создаётся и updates равны нулю. Ни одна команда не скачивает model weights и не обращается к raw KION.

## C. Полный путь от raw: рецепт, без нового обучения

Ниже — последовательность исходного исследования. Полные параметры и source pins находятся в `source/fullstudy/code/recipes/source-requests.json`; matched-KL продолжение — в `source/matched-kl/`. Historical runtime содержит машинные guard/constants, поэтому простая замена JSON-путей не делает весь train pipeline переносимым. Этот уровень требует отдельных данных, восстановления caches, адаптации runtime и ресурсного плана.

1. Получить три raw CSV KION и проверить hashes из [INPUTS.md](INPUTS.md). Не превращать ненаблюдённые пары в RM label0.
2. Подготовить item-token mapping и temporal splits через `kion_prepare.py`: явно передать `--context 64 --max-train 1000000000 --max-eval 1000000000 --seed 2026` (defaults context 32 / 150 000 / 3 000 не соответствуют full recipe), train до 9 августа 2021; development 9–15 августа; определённое metadata test-окно 16–22 августа. Сохранённые размеры: train 3 477 437, development 158 760, test 167 432. Development многократно использован, исторически открытый test не является новым untouched test.
3. Восстановить feature providers, strict-past query clocks, frozen candidate graph, raw 1068 schema и source 2. Сохранить mapping/schema hashes и fixed cohorts. Точные source paths перечислены в [SOURCE_CLOSURE.md](../research/SOURCE_CLOSURE.md). Для external schema161 сохранён [metadata index](external-provenance/INDEX.json); dated banks и original MovieLens/pageview payloads исключены, поэтому данный recipe не является самодостаточным. Для текстовых caches отдельно нужны pinned multilingual-E5 weights/tokenizer из [INPUTS.md](INPUTS.md).
4. Обучить ancestral item-ID GPT: hidden 256, layers 4, heads 4, bottleneck 32, person_dim 32, context 64, batch 128, AdamW lr 1e-4, weight decay 0,01, seed 2026. Сохранить horizon 5 epochs: принятая E2-линия завершила две эпохи, 54 470 updates и 6 954 874 предъявления, продолжая полный optimizer/scaler/RNG state. Двухэпоховый scheduler не эквивалентен этой истории; fresh retraining не обещан bitwise равным.
5. На frozen E2 сформировать nested bank 262 144, содержащий прежние 32 768 плюс 229 376 примеров. Обучить A/B/C по три эпохи, 6144 updates на голову, B128; positive + 31 fixed natural negatives. C использует raw 1068 + source 2; q/v/GPT заморожены. Evaluation использует natural 1024 без target insertion; unknown/off-pool остаются misses в общем знаменателе.
6. Построить observed `watched_pct/100` RM target: fit до 2 августа 2021, select/calibration targets 2–8 августа с user-disjoint cohorts и history до 2 августа. Fit/select/calibration: 100 000 / 20 000 / 20 000, две history/nohistory CatBoost конфигурации depth 4, 160 iterations, lr 0,05, L2 leaf 10, seed 2026, CPU 2 threads. RM-r2 изменяет calibration, сохраняя те же деревья. RM schema 2079 context + 2008 candidate отличается от C raw 1068.
7. Зафиксировать train 512 / screen 256 с исключением RM select/calibration users/context keys. Получить RM scores, проверить support, mapping, scale и update0/reference. Выполнить completion H1 head-only RL: `E_pi[r_eff] − beta KL(pi||C)`, beta 0,5, Adam lr 1e-5, weight decay 0, clip 1, logical B16, 1000 updates, seed 20261024, checkpoints 0/250/500/1000; encoder/reference/RM заморожены.
8. Для второй utility использовать `r_RM(s,i) − m_fit(i)` с frozen observed-fit prior до 2 августа, coefficient 1; unsupported advantage 0, softmax/KL по всем 1024 actions. Исходный residual fit имеет тот же beta 0,5. В matched-KL продолжении residual beta 0,1627403 выбран только по train; три пары sampler seeds 20261024/25/26 проверяются отдельно. Это новая utility, а не один beta sweep.
9. Выполнить screen 256 / full 4096 evaluation и paired user-cluster uncertainty для исходных и matched-KL policies. Отдельный five-step live-prefix diagnostic выполнен для исходных C, completion seed24 и residual с beta 0,5; six-head matched-KL rollout не выполнялся. На шагах 2–5 нет наблюдаемых counterfactual relevance labels; pool и RM observed state фиксированы, GPT q меняется после synthetic append. No-repeat обеспечивает уникальность механически.

Retrospective metadata и агрегированные last-watch строки сохраняют временные ограничения, описанные в актуальном [REPORT.md](../reports/REPORT.md); этот recipe не доказывает строгую contemporaneous или causal оценку.

Эти пункты не запускают train, RM fit или RL fit. В ходе финальной подготовки не добавляются optimizer updates и не скачиваются данные. Исторические успешные receipts подтверждают исходное исследование; актуальный отчёт соединяет полный опыт и matched-KL продолжение.
