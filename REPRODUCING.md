# Воспроизведение

Основной отчёт: [REPORT.pdf](REPORT.pdf), исходник [report-latex/main.tex](report-latex/main.tex). [REPORT.md](REPORT.md) является текстовой производной. Для чтения Python не нужен. Техническое приложение [reports/REPORT.md](reports/REPORT.md) и прежний [PDF](reports/REPORT.pdf) сохранены отдельно. Команда сборки текущего PDF приведена в [README](README.md).

Инструкция описывает текущую поставку; старые снимки исходников и протоколы запусков сохранены для истории. [Краткая карта](reproduction/README.md), [входы](reproduction/INPUTS.md), [окружения](reproduction/ENVIRONMENTS.md), [передача](reproduction/HAND_IN.md) обновлены вместе с ней.

| Путь | Что получает читатель | Что пересчитывается |
|---|---|---|
| Проверка исходников | Текущий клон/архив репозитория | Целостность кода и арифметика метрик на тестовых примерах |
| Текущая таблица и интервалы | `saved-metrics64k/`, Python и NumPy | 65536 запросов, метрики и парный bootstrap по пользователям |
| Историческая таблица | `saved-metrics/`, стандартная библиотека Python | Прежние 4096 запросов |
| Проверка сохранённых моделей | Отдельно полученный прежний полный ZIP плюс текущий код запуска | GPT, головы C/completion/residual и RM для четырёх контекстов |
| Полное обучение | Исторический рецепт и список недостающих входов | Готового переносимого запуска нет |

Для отдельных этапов подготовлены команды. По умолчанию они только проверяют входы и показывают план; вычисления требуют отдельного флага.

| Этап | Инструкция |
|---|---|
| Исходные CSV → временное разбиение | [Подготовка данных](reproduction/PREPARE_DATA.md) |
| История → признаки и обучающие записи RM | [Подготовка входов RM](reproduction/PREPARE_REWARD_DATA.md) |
| Готовые записи RM → CatBoost и калибровка | [Обучение RM](reproduction/TRAIN_REWARD_MODEL.md) |
| Полные входы GPT → новый непрерывный E2 | [Обучение GPT](reproduction/GPT_TRAINING.md) |
| Готовый банк → головы A/B/C | [Обучение ранкеров](reproduction/HEAD_TRAINING.md) |
| Сохранённые входы и C → RL | [Повторение RL](reproduction/TRAIN_RL.md) |

При подготовке этой поставки проверены режимы плана, метаданные входов и тесты на искусственных данных. Полные этапы подготовки данных и обучения повторно не запускались.

[Проверка входов GPT](reproduction/GPT_INPUTS.md) строит список переноса из 370 файлов; массивы и веса при этом не читаются. `train_gpt.py` уже соединяет загрузчик с исходным обучением и отдельно разрешаемым выполнением до E2; порядок подготовки запроса приведён в [GPT_TRAINING.md](reproduction/GPT_TRAINING.md).

Для [повторного построения банка](reproduction/PREPARE_HEAD_BANK.md) нужен сохранённый GPT после второй эпохи, прежний банк и готовые вспомогательные данные. Исполнитель проверен на искусственных примерах; полного прогона не было. Новые веса GPT нельзя автоматически подставить вместо исторических. Подготовка полного запроса для этого этапа описана в отдельной инструкции.

Для копирования перечисленных входов с проверкой контрольных сумм есть [команда переноса](reproduction/TRANSFER_INPUTS.md). Перед последовательным запуском этапов нужно подготовить перечисленные файлы и проверить совместимость их версий.

## Проверить исходники

Нужен Python 3.12+. Все команды выполняются из корня чистого клона:

```text
python -B verify_repository.py
python -B source/verify_sources.py
python -B source/fullstudy/code/tools/verify_bundle.py --report-root source/fullstudy
python -B reproduction/kion_replay verify --bundle .
```

Первые две сверяют состав, SHA256, размеры и синтаксис. Третья дополнительно проверяет метрики на трёх вымышленных запросах: MAP@10 = 0,5, Recall@10 = 2/3, NDCG@10 = 0,5436432511904858. Последняя проверяет текущий код проверки моделей. Это стандартная библиотека Python, без NumPy, моделей и их входов.

`verify_repository.py` проверяет точный состав по `projection.json`. Окружения, распаковку входов и новые результаты ниже держите рядом с клоном. Добавленные внутрь клона файлы изменят его состав; это не следует исправлять удалением пользовательских результатов. Для внешних входов есть отдельный `verify --inputs`.

## Пересчитать текущую таблицу и интервалы

В `reproduction/saved-metrics64k/` включены ранги и плотные метки групп пользователей без исходных ID для 65536 запросов / 53024 пользователей. Создайте окружение с NumPy 2.5.4 по [ENVIRONMENTS.md](reproduction/ENVIRONMENTS.md). Из корня клона в PowerShell:

```powershell
& ../kion-env-metrics/Scripts/python.exe -B reproduction/saved_metrics64k.py verify --data reproduction/saved-metrics64k
& ../kion-env-metrics/Scripts/python.exe -B reproduction/saved_metrics64k.py recompute --data reproduction/saved-metrics64k --output ../kion-table64k.json
& ../kion-env-metrics/Scripts/python.exe -B reproduction/saved_metrics64k.py bootstrap --data reproduction/saved-metrics64k --output ../kion-bootstrap64k.json
```

Выходные JSON должны отсутствовать. `verify` проверяет входы; `recompute` вычисляет метрики из рангов; `bootstrap` заново строит 2000 парных выборок пользователей с seed 20261024 и 95% интервалы по процентилям. В знаменателе остаются 1746 запросов с неизвестным целевым фильмом и 10183 с известным фильмом вне списка кандидатов. При повторном выборе пользователя повторяются все его строки; среднее взвешено числом строк, одни и те же выборки используются для всех сравнений. Метки групп сохраняют связь запросов одного пользователя, без исходных ID пользователей.

Программа проверяет метрики и интервалы относительно закреплённых результатов. Она не пересортировывает оценки моделей и не запускает модели. Bootstrap оценивает неопределённость на этой выборке разработки; ограничения многократного выбора моделей по этой выборке сохраняются.

## Историческая таблица 4096 запросов

Оба входа уже находятся в `reproduction/saved-metrics/`: `manifest.json` и `rankings.json`. Общий размер ограничен 1 MiB; точные байты и SHA256 данных указаны в описателе. Ничего дополнительно скачивать не нужно.

```text
python -B reproduction/saved_metrics.py verify --data reproduction/saved-metrics
python -B reproduction/saved_metrics.py recompute --data reproduction/saved-metrics --output ../kion-table4096.json
```

Выходной JSON должен отсутствовать. Команда независимо рассчитывает MAP@10, Recall@10 и NDCG@10 по сохранённым рангам 0–10 с общим знаменателем 4096. В нём остаются 122 запроса с неизвестной целью и 616 с известной целью вне списка кандидатов. При единственной наблюдаемой цели MAP равен обратному рангу с отсечением на позиции 10. GPT/A/B/C берутся из прежней оценки обученных моделей; completion/residual являются средними метрик трёх отдельных политик с согласованным KL. Это не ансамбль.

Ожидаемая основная колонка после округления до четырёх знаков: GPT 0,0723; A 0,0829; B 0,0865; C 0,0969; RL completion 0,0944; RL residual 0,0963. Пересчёт проверяет арифметику сохранённого ранжирования. Он не запускает модели, не перепроверяет сортировку полных оценок моделей и не повторяет построение признаков или обучение. Исходные ID пользователей, фильмов и запросов, истории и веса в компактный экспорт не входят.

Этот исторический экспорт не содержит групп пользователей и не позволяет пересчитать доверительные интервалы. Для текущих интервалов используется отдельный путь 65536 запросов выше.

## Получить входы для проверки моделей

Этот путь необязателен для чтения, текущих метрик и bootstrap. Нужен отдельно предоставленный прежний полный архив. Он существует локально у владельца; публичного URL нет, факт получения внешним адресатом не установлен. Получатель должен отдельно получить именно этот файл от владельца:

| Имя | Байты | SHA256 |
|---|---:|---|
| `kion-task2-final-20261025.zip` | 62719299 | `1b03322432d8e0d612abfb507bdd6b9dd4edd53e0f31ac3c9826408718e0577e` |

Из корня текущего клона в PowerShell, заменив только путь к полученному архиву:

```powershell
$Archive = 'C:/path/to/kion-task2-final-20261025.zip'
if ((Get-Item -LiteralPath $Archive).Length -ne 62719299) { throw 'Archive size differs' }
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne '1b03322432d8e0d612abfb507bdd6b9dd4edd53e0f31ac3c9826408718e0577e') { throw 'Archive SHA256 differs' }
python -B delivery/package.py verify --archive $Archive --extract-to ../kion-old-full
if ($LASTEXITCODE -ne 0) { throw 'Archive verification/extraction failed' }
$ActorInputs = (Resolve-Path -LiteralPath '../kion-old-full/reproduction/inputs').Path
$RmInputs = (Resolve-Path -LiteralPath '../kion-old-full/reproduction/inputs-rm-r1').Path
python -B reproduction/kion_replay verify --bundle . --inputs $ActorInputs
if ($LASTEXITCODE -ne 0) { throw 'Actor input verification failed' }
python -B reproduction/kion_replay verify --bundle . --inputs $RmInputs
if ($LASTEXITCODE -ne 0) { throw 'RM input verification failed' }
```

`../kion-old-full` должен отсутствовать. Архив проверяется по встроенному описателю и безопасно распаковывается в новый каталог. Его старые код и отчёт не заменяют текущий клон. Команды ниже используют только два каталога входов оттуда. В Linux/macOS используйте тот же `package.py`, задайте абсолютные пути к этим каталогам и передайте их в `--inputs` обычным синтаксисом своей оболочки.

Actor: 17 файлов, 68375375 байт плюс описатель 7578 байт, SHA256 описателя `e680dd145b2bfdc812107fb9a257bf16f6d1ef5a53c07845f7dc21838146a52c`. RM: 9 файлов, 33248280 байт плюс описатель 4381 байт, SHA256 описателя `7b3e7dbefde3a6d0e113c0340dbbfc425d223574a7c4cd13db7927fb29b30380`. В репозитории есть только описатели; перед вычислениями программа сверяет хеши, размеры и заголовки массивов.

## Проверить сохранённые actor и RM

Создайте две среды вне клона по [ENVIRONMENTS.md](reproduction/ENVIRONMENTS.md). Историческая среда: Windows x64, Python 3.12.14; actor NumPy 2.2.6 / Torch 2.8.0+cu126 / Transformers 4.55.4, RM NumPy 2.5.4 / CatBoost 1.2.10. Установка с нуля и другая машина не проверены. Версии прямых зависимостей закреплены; полный состав транзитивных и системных компонентов не зафиксирован.

После успешной проверки входов, из той же PowerShell-сессии:

```powershell
& ../kion-env-actor/Scripts/python.exe -B reproduction/kion_replay infer --bundle . --inputs $ActorInputs --output ../kion-replay-actor --tier frozen-encoder --device cuda --precision cuda-fp16 --oracle exact --cpu-threads 4
if ($LASTEXITCODE -ne 0) { throw 'Actor replay failed' }
& ../kion-env-rm/Scripts/python.exe -B reproduction/kion_replay reward --bundle . --inputs $RmInputs --output ../kion-replay-rm
if ($LASTEXITCODE -ne 0) { throw 'RM replay failed' }
```

Выходные каталоги должны отсутствовать, родительский каталог должен существовать. GPT пересчитывается для всех 16 исходных примеров; C и две исходные RL-головы ранжируют по 1024 кандидата для первых четырёх позиций. У RL beta 0,5 и seed 20261024; шесть голов с согласованным KL в этот набор не входят. RM вычисляет предсказания CatBoost, калибровку и значения для неподдержанных кандидатов. Обучения и обновления весов нет.

Режим exact требует совпадения q, базовых оценок, оценок голов и top10 с эталоном CUDA FP16. При несовпадении или недоступной CUDA процесс завершается ошибкой. Для отдельной проверки на CPU задайте `--device cpu --precision fp32 --oracle none`; она не подтверждает историческое совпадение. Режим `cached-features` заменяет `--tier frozen-encoder` на `--tier cached-features` и использует сохранённый GPT q.

После кода завершения 0 прочитайте `summary.json` или `rm-summary.json`, затем `run-receipt.json`: состояние, хеши входов и кода, устройство, точность и результат сравнения. Для actor exact требуется `oracle_passed: true`. Само появление каталога не означает успех. Ограничения среды и статус прежних запусков описаны в [ENVIRONMENTS.md](reproduction/ENVIRONMENTS.md).

## Полное обучение

Для полного обучения нужны исходные данные, промежуточные признаки и настройка путей к ним. Состав входов и команды перечислены в [историческом рецепте](reproduction/README.md#исторический-рецепт-обучения), [INPUTS.md](reproduction/INPUTS.md) и [SOURCE_CLOSURE.md](research/SOURCE_CLOSURE.md). Полный цикл от исходных CSV до обученных моделей на новой машине не запускался. Проверенные здесь команды пересчитывают сохранённые метрики и выполняют отдельную проверку моделей на четырёх контекстах.
