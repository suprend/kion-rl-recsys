# Воспроизведение

## Прочесть результаты

Начните с [REPORT.md](REPORT.md) или [REPORT.pdf](REPORT.pdf). Это основной отчёт: постановка задачи, устройство моделей, результаты и ограничения исследования. Для чтения Python не нужен. Прежний [reports/REPORT.md](reports/REPORT.md) с [PDF](reports/REPORT.pdf) сохранён как техническое приложение.

Новый PDF собирается небольшой обёрткой [make_report.py](make_report.py). Она использует сохранённый сборщик `reports/render_report.py` и заменяет локальные ссылки ссылками на GitHub.

В Git и новом архиве исходников с отчётом есть код, компактные результаты, рисунки и свидетельства прежних проверок. Численных входов и весов для запуска моделей там нет. Состав клона описан в [PROJECTION.md](PROJECTION.md).

Документы [reproduction/README.md](reproduction/README.md), [INPUTS.md](reproduction/INPUTS.md), [ENVIRONMENTS.md](reproduction/ENVIRONMENTS.md) и [HAND_IN.md](reproduction/HAND_IN.md) сохранены без изменений. Они относятся к прежнему полному архиву. Их указания на включённые входы и актуальный отчёт нужно читать в этих исторических границах; текущая инструкция уточняет доступность файлов.

### Пересобрать PDF при необходимости

Готовый PDF не нужно пересобирать для чтения. Для собственной сборки нужны ReportLab 4.4.9 и два TTF-шрифта DejaVu Sans: обычный и жирный. Из корня клона выполните команду, заменив пути к шрифтам своими:

```text
python -B make_report.py --source REPORT.md --output ../kion-report-rebuilt.pdf --font C:/path/to/DejaVuSans.ttf --font-bold C:/path/to/DejaVuSans-Bold.ttf
```

Выходной файл должен быть новым. Новую вёрстку нужно визуально проверить; побайтовое совпадение PDF в другой среде не гарантируется.

## Проверить исходники без модельных зависимостей

Нужен Python 3.12+. Выполните из корня чистого клона:

```text
python -B verify_repository.py
python -B source/verify_sources.py
python -B source/fullstudy/code/tools/verify_bundle.py --report-root source/fullstudy
python -B reproduction/kion_replay verify --bundle .
```

Первая команда сверяет состав клона с `projection.json` и проверяет отсутствие исключённых численных файлов. Вторая проверяет хеши, размеры и синтаксис сохранённых исходников. Третья проверяет вложенный исторический комплект и расчёт метрик на трёх вымышленных запросах. Ожидаются MAP@10 = 0,5, Recall@10 = 2/3 и NDCG@10 = 0,5436432511904858. Последняя сверяет код переносимого replay с `reproduction/runtime-manifest.json`.

Все четыре команды используют стандартную библиотеку. NumPy, Torch, CatBoost и данные им не нужны; модели они не запускают. Успех подтверждает целостность исходников и простую проверку арифметики метрик, но не повторяет исследование.

Корневой `verify_repository.py` рассчитан на точный состав чистого клона. После добавления входов, окружений или результатов replay эта проверка перестанет проходить. Для восстановленных входов используйте отдельные команды `verify --inputs` ниже.

## Повторить ограниченный запуск замороженных моделей

Для этого уровня нужен отдельно предоставленный старый полный архив. Он подготовлен локально, не загружен и не входит в новый архив исходников с отчётом. Опубликованного URL для скачивания нет; наличие файла у получателя не подразумевается.

| Поле | Значение |
|---|---|
| Имя | `kion-task2-final-20261025.zip` |
| Размер | 62 719 299 байт |
| SHA256 | `1b03322432d8e0d612abfb507bdd6b9dd4edd53e0f31ac3c9826408718e0577e` |

После получения сверьте архив до распаковки. В PowerShell замените путь своим:

```powershell
$Archive = 'C:/path/to/kion-task2-final-20261025.zip'
(Get-Item -LiteralPath $Archive).Length
(Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
```

Вручную распакуйте его в новый каталог. Скопируйте оттуда содержимое `reproduction/inputs/` и `reproduction/inputs-rm-r1/` в одноимённые каталоги клона. Помимо `manifest.json`, первый набор содержит 17 файлов для GPT и трёх голов, второй содержит 9 файлов для модели награды, далее RM. В Git оставлены только описатели `manifest.json`; они должны совпасть с архивными. Ожидаемые SHA256:

```text
reproduction/inputs/manifest.json
  e680dd145b2bfdc812107fb9a257bf16f6d1ef5a53c07845f7dc21838146a52c
reproduction/inputs-rm-r1/manifest.json
  7b3e7dbefde3a6d0e113c0340dbbfc425d223574a7c4cd13db7927fb29b30380
```

Проверьте восстановленные файлы стандартной библиотекой:

```text
python -B reproduction/kion_replay verify --bundle . --inputs reproduction/inputs
python -B reproduction/kion_replay verify --bundle . --inputs reproduction/inputs-rm-r1
```

Проверка сверяет хеши, размеры и контракты массивов и весов до численной загрузки. Входы содержат преобразованные истории и представления пользователей. Удаление исходных user IDs не гарантирует анонимность.

### Подготовить раздельные окружения

Actor, то есть GPT и головы ранжирования, требует NumPy 2.2.6, Torch 2.8.0+cu126 и Transformers 4.55.4. RM требует NumPy 2.5.4 и CatBoost 1.2.10. Используйте отдельные окружения и процессы. Команды установки для получателя в PowerShell:

```powershell
python -m venv .venv-actor
python -m venv .venv-rm
& ./.venv-actor/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/actor.txt --extra-index-url https://download.pytorch.org/whl/cu126
& ./.venv-rm/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/reward-worker.txt
```

На Linux/macOS пути интерпретаторов имеют вид `.venv-actor/bin/python` и `.venv-rm/bin/python`, без PowerShell `&`. Requirements закрепляют верхние версии, но не все транзитивные зависимости и бинарники. Совместимость CUDA wheel, GPU и драйвера нужно проверить в выбранной среде. Прежние запуски использовали Windows x64 и Python 3.12.14; установка на новой машине не проверена.

### Запустить replay и прочесть результат

После успешной проверки входов выберите новые output-каталоги:

```powershell
& ./.venv-actor/Scripts/python.exe -B reproduction/kion_replay infer --bundle . --inputs reproduction/inputs --output replay-encoder-cuda --tier frozen-encoder --device cuda --precision cuda-fp16 --oracle exact --cpu-threads 4
& ./.venv-rm/Scripts/python.exe -B reproduction/kion_replay reward --bundle . --inputs reproduction/inputs-rm-r1 --output replay-rm
```

Actor пересчитывает GPT для всех 16 историй в исходном пакете. Для первых четырёх историй головы C, исходная completion и исходная residual ранжируют по 1024 кандидата. Две RL-головы используют β=0,5 и seed20261024. Это три исходные головы; шесть голов matched-KL продолжения сюда не входят. RM рассчитывает прогноз CatBoost по отдельным сохранённым признакам, калибрует его и для фильмов без обучающих наблюдений использует общий средний досмотр. Подготовка признаков от raw, обучение и обновление весов не выполняются.

`--oracle exact` требует точного совпадения q, base logits, scores и top10 с историческим CUDA FP16 oracle. Форма batch, позиции и precision существенны. При расхождении процесс завершается ошибкой; автоматической смены устройства нет. Менять oracle для получения pass нельзя. CPU FP32 требует явных `--device cpu --precision fp32 --oracle none` и проверяет отдельное исполнение без подтверждения исторического совпадения. Команды CPU и `cached-features` приведены в историческом [reproduction/README.md](reproduction/README.md); отдельно они не запускались.

Проверьте exit code, `summary.json` или `rm-summary.json`, затем `run-receipt.json`: входные и runtime manifests, устройство, precision, oracle и статус. Наличие output-каталога само по себе не означает успех.

Прежние проверки из свежих проверочных архивов на исходной машине с готовыми окружениями дали нулевые ошибки actor и RM; [свидетельства](evidence/reproduction/INDEX.json) сохранены. Финальный полный ZIP прошёл проверку стандартной библиотекой, но его численный replay целиком не повторялся. Текущая редактура также не является новым численным запуском.

У этих четырёх контекстов нет observed target/reward labels. Replay проверяет вычисления и совпадение ранжирования, но не пересчитывает MAP и доверительные интервалы исследования. Для полного переобучения есть исторический [рецепт](reproduction/README.md#c-полный-путь-от-raw-рецепт-без-нового-обучения) и [точные requests](source/fullstudy/code/recipes/source-requests.json). Потребуются отдельно полученные raw KION, E5, внешние feature banks, полные cohorts/caches и исходные checkpoints, а также адаптация машинных путей и guards. Это описание последовательности исследования, а не готовая команда полного воспроизведения.
