# Воспроизведение

Все команды ниже выполняются из корня этого репозитория. Для проверки кода нужен Python 3.12+. Для чтения готового PDF Python не нужен.

## 1. Выберите уровень проверки

| Уровень | Что есть в Git | Что получится проверить |
|---|---|---|
| Исходники и компактные результаты | Код, manifests, агрегаты, рисунки, отчёт и receipts | Целостность snapshot, синтаксис и synthetic ranking arithmetic |
| Ограниченный frozen-model replay | Runtime и описатели входов; численные входы и веса отсутствуют | После отдельного получения ZIP — GPT/heads и RM на четырёх контекстах × 1024 кандидата |
| Полный inference или обучение от raw | Исторический код, параметры и рецепты | Нужны дополнительные данные, веса, feature stores и адаптация runtime; готового полного окружения здесь нет |

Сохранённые [reproduction/README.md](reproduction/README.md), [INPUTS.md](reproduction/INPUTS.md), [HAND_IN.md](reproduction/HAND_IN.md), [delivery/PACKAGING.md](delivery/PACKAGING.md) и научный отчёт описывают исходный **полный hand-in ZIP**. Их состав и формулировки сохранены без изменения. В Git численные payloads удалены из проекции; исторические manifests и receipts остаются доказательствами ожидаемых файлов и прежних запусков. Границы клона определяют этот документ и [PROJECTION.md](PROJECTION.md).

## 2. Проверка клона стандартной библиотекой

```text
python -B verify_repository.py
python -B source/verify_sources.py
python -B source/fullstudy/code/tools/verify_bundle.py --report-root source/fullstudy
python -B reproduction/kion_replay verify --bundle .
```

Эти команды не требуют NumPy, Torch, CatBoost или данных и не выполняют model inference:

- `verify_repository.py` сверяет точную Git-проекцию с `projection.json` и отсутствие исключённых payloads.
- `source/verify_sources.py` проверяет SHA256, размеры и Python syntax 284 сохранённых исторических файлов.
- `verify_bundle.py` проверяет вложенный исторический bundle из 261 файла и метрики на трёх вымышленных запросах. Ожидаются MAP@10 = 0,5, Recall@10 = 2/3 и NDCG@10 = 0,5436432511904858.
- `kion_replay verify --bundle .` проверяет portable runtime по `reproduction/runtime-manifest.json`, без открытия model-input файлов.

После добавления локальных численных входов, environments или replay outputs корневой `verify_repository.py` намеренно перестанет проходить проверку точного состава: он предназначен для чистой проекции исходников. Для проверки восстановленных входов используйте команды с `--inputs` из раздела 3. Исторические source/runtime checkers от них независимы.

## 3. Отдельные входы для numerical replay

В Git сохранены только описатели:

| Fixture | Описатель в Git | Отсутствующие payloads |
|---|---|---|
| GPT и C/completion/residual | [reproduction/inputs/manifest.json](reproduction/inputs/manifest.json) | 17 файлов: преобразованные token-prefix histories, user/item representations, features, candidate IDs, frozen веса и validation oracle |
| Reward model | [reproduction/inputs-rm-r1/manifest.json](reproduction/inputs-rm-r1/manifest.json) | 9 файлов: отдельные RM features, fit priors/counts, frozen CatBoost и validation references |

Полный hand-in архив подготовлен локально и передаётся отдельно. В этом репозитории нет опубликованного URL для его скачивания; доступность архива получателю не подразумевается.

| Поле | Значение |
|---|---|
| Имя | `kion-task2-final-20261025.zip` |
| Размер | 62 719 299 байт |
| SHA256 | `1b03322432d8e0d612abfb507bdd6b9dd4edd53e0f31ac3c9826408718e0577e` |

Это идентичность локально собранного полного архива. После отдельного получения сверяйте байты до распаковки. Пример для PowerShell, заменив путь своим:

```powershell
$Archive = 'C:/path/to/kion-task2-final-20261025.zip'
(Get-Item -LiteralPath $Archive).Length
(Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
```

Вручную распакуйте архив в новый каталог. Из него восстановите содержимое `reproduction/inputs/` и `reproduction/inputs-rm-r1/` в одноимённые каталоги клона. Сохранённые в Git `manifest.json` должны совпасть с архивными; менять descriptors под другие файлы нельзя. Ожидаемые SHA256 самих manifests:

```text
reproduction/inputs/manifest.json
  e680dd145b2bfdc812107fb9a257bf16f6d1ef5a53c07845f7dc21838146a52c
reproduction/inputs-rm-r1/manifest.json
  7b3e7dbefde3a6d0e113c0340dbbfc425d223574a7c4cd13db7927fb29b30380
```

Затем проверьте fixture-файлы стандартной библиотекой:

```text
python -B reproduction/kion_replay verify --bundle . --inputs reproduction/inputs
python -B reproduction/kion_replay verify --bundle . --inputs reproduction/inputs-rm-r1
```

Проверка включает hashes, размеры и контракты массивов/весов до numerical loading. Локальные payloads исключены из Git. Преобразованные histories и representations с удалёнными user IDs не следует считать гарантированно анонимными.

[external-provenance/INDEX.json](reproduction/external-provenance/INDEX.json) сохранён как metadata. Описанные им 13 JSON payloads в Git отсутствуют; даже в полном hand-in ZIP это provenance, а не полный набор внешних feature banks. Для ограниченного replay они не нужны. Подробные raw/E5/feature prerequisites и historical source pins перечислены в [INPUTS.md](reproduction/INPUTS.md) и [excluded-inputs.json](source/fullstudy/code/recipes/excluded-inputs.json).

## 4. Раздельные actor и RM окружения

Исторический numerical replay использовал Windows x64 и Python 3.12.14. Верхние версии закреплены в requirements; это не полный transitive binary lock и не гарантия установки того же backend на другой машине.

| Среда | Верхние зависимости |
|---|---|
| Actor | NumPy 2.2.6, Torch 2.8.0+cu126, Transformers 4.55.4, psutil 7.0.0 |
| Отдельный RM worker | NumPy 2.5.4, CatBoost 1.2.10, SciPy 1.18.1 |
| Рисунки, необязательно | NumPy 2.2.6, Matplotlib 3.11.2 |

Команды установки для получателя в PowerShell:

```powershell
python -m venv .venv-actor
python -m venv .venv-rm
& ./.venv-actor/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/actor.txt --extra-index-url https://download.pytorch.org/whl/cu126
& ./.venv-rm/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/reward-worker.txt
```

Actor и RM требуют разных NumPy. Устанавливайте их в отдельные environments и запускайте RM отдельным интерпретатором/процессом. На Linux/macOS пути интерпретаторов — `.venv-actor/bin/python` и `.venv-rm/bin/python`, без PowerShell `&`. Совместимость CUDA wheel, GPU и драйвера нужно проверить в выбранной среде; silent CPU fallback отсутствует. Дополнительные детали — [ENVIRONMENTS.md](reproduction/ENVIRONMENTS.md).

## 5. Настоящий frozen inference

После успешной проверки входов запустите actor с новым output-каталогом:

```powershell
& ./.venv-actor/Scripts/python.exe -B reproduction/kion_replay infer --bundle . --inputs reproduction/inputs --output replay-encoder-cuda --tier frozen-encoder --device cuda --precision cuda-fp16 --oracle exact --cpu-threads 4
```

Этот режим пересчитывает GPT q/retrieval для полного producer batch из 16 prefix peers. Slots 0–3 проходят C и две **исходные** RL-головы seed20261024 с β=0,5. Это не шесть голов matched-KL продолжения. Frozen item table, raw features и candidate pool уже экспортированы; raw-to-feature pipeline не выполняется.

`--oracle exact` требует точного совпадения q, base logits, scores и top10 с historical CUDA FP16 oracle. Batch shape, slots и precision существенны; при расхождении процесс завершается ошибкой. Менять oracle или устройство для получения pass нельзя.

Отдельная reward-команда выполняет настоящую CatBoost prediction, calibration и cold fallback на RM-признаках:

```powershell
& ./.venv-rm/Scripts/python.exe -B reproduction/kion_replay reward --bundle . --inputs reproduction/inputs-rm-r1 --output replay-rm
```

Проверяйте exit code и `summary.json`/`rm-summary.json`, затем `run-receipt.json`: manifests, устройство, precision, oracle mode и результат. Создание output-каталога само по себе не означает pass. Outputs должны быть новыми; runtime не перезаписывает fixtures. В обоих режимах отсутствуют обучение, optimizer updates, загрузки моделей из сети и обращения к raw KION.

В runtime также есть `cached-features` и CPU FP32. Их команды приведены в [сохранённой инструкции](reproduction/README.md). CPU требует явных `--device cpu --precision fp32 --oracle none`: это отдельное исполнение с описательными расхождениями, без pass historical parity.

## 6. Что уже выполнено, а что остаётся рецептом

[Индекс доказательств](evidence/reproduction/INDEX.json) связывает отдельные export, extraction и replay receipts.

| Проверка | Фактический статус |
|---|---|
| Frozen GPT + три головы | Принято: новая проверочная распаковка, CUDA FP16, 4 × 1024, exact q/base/scores/top10, maximum error 0 |
| Frozen RM | Принято: отдельный CPU process, 4096 candidate pairs, float64 error 0, exact float32 reward/support; supported 3976/4096 |
| Среда этих двух запусков | Исходная Windows-машина с уже установленными раздельными environments; без fits/updates |
| Новая машина или numerical replay всего финального ZIP | Отдельно не проверены |
| Отдельный `cached-features` CLI, actor CPU FP32 | Не запускались |
| Полный frozen inference на development4096 | Этим fixture не повторён; нужны дополнительные inputs |
| Новое обучение GPT/heads/RM/RL от raw | Не выполнялось при подготовке комплекта или Git-проекции |

У четырёх replay-контекстов нет observed target/reward labels. Поэтому они проверяют mechanics и score/rank parity, не пересчитывая MAP/CI исследования. Matched-KL сравнение трёх sampler-пар и пятишаговая диагностика остаются сохранёнными результатами полного исследования.

Полный raw-рецепт находится в разделе C [reproduction/README.md](reproduction/README.md), точные requests — в [source-requests.json](source/fullstudy/code/recipes/source-requests.json), продолжение — в [source/matched-kl/](source/matched-kl/). Он описывает GPT 3 477 437 × 2 эпохи, A/B/C 262 144 × 3 и H1 head-only RL на train512. Это исторический код с machine-bound constants/guards; замена путей в JSON не делает весь pipeline переносимым. Raw KION, внешние banks, E5-файлы, полные cohorts/caches и исходные training checkpoints требуется получать или восстанавливать отдельно. Их наличие на машине получателя здесь не обещано.

## 7. Рисунки и PDF

Готовые PNG/SVG и 16-страничный PDF уже включены. PDF содержит встроенные шрифты; [независимая визуальная проверка](evidence/reviews/pdf-visual-final.json) приняла все 16 страниц. Пересборка не нужна для чтения или model replay.

Статическая проверка Markdown/builder использует standard library:

```text
python -B reports/render_report.py --source reports/REPORT.md --check
```

Для пересборки screen/full рисунков из сохранённых агрегатов создайте отдельную figures environment:

```powershell
python -m venv .venv-figures
& ./.venv-figures/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/figures.txt
& ./.venv-figures/Scripts/python.exe -B source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/plot_results_v4.py --input evidence/fullstudy/plot-data-screen.json --labels evidence/fullstudy/figure-provenance/screen/labels.json --output replay-figures/screen
& ./.venv-figures/Scripts/python.exe -B source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/plot_results_v4.py --input evidence/fullstudy/plot-data-full.json --labels evidence/fullstudy/figure-provenance/full/labels.json --output replay-figures/full
```

Outputs должны быть новыми каталогами. Эти команды читают aggregate JSON, а не модели или raw CSV. Исторически 13 PNG совпали побайтно на исходной render-среде; на другой font/render среде такое совпадение не обещается. Optional PDF rebuild с ReportLab 4.4.9 и DejaVu Sans описан в [HAND_IN.md](reproduction/HAND_IN.md); изменённый PDF требует новой визуальной проверки.
