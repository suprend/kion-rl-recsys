# GPT с нуля до E2

Адаптер готовит новый непрерывный запуск. По умолчанию обучение выключено.
Нужны полный набор входов из [GPT_INPUTS.md](GPT_INPUTS.md), проверенный
транзитивный inventory, файл привязок и ожидаемый config с SHA256.
Bindings сопоставляют исходные логические имена с файлами вашей машины.
Пропущенные признаки не заменяются нулями, файлы из сети не скачиваются.

## Получить ожидаемый config

Нужен исходный config завершённой E1, который использовался для проверенного
плана этого адаптера. Это внешний вход; в Git-клоне его нет. Владелец исходных
артефактов должен передать следующие байты вместе с metadata-пакетом:

| Поле переноса | Значение |
|---|---|
| Файл относительно исходного каталога `task2-kion` | `deadline-20261022/control-r2/job/model/config.json` |
| Имя в передаваемом пакете | `metadata/expected-config.json` |
| Размер | 8947263 bytes |
| SHA256 | `0b00d2d2a5bbbdb6798f4e3cda762e6feb98e96806bc486e0bdcb9b89da817ab` |

Команда экспорта для владельца артефактов; корни выбираются на его машине:

```powershell
$OriginalArtifacts = 'E:/original-artifacts/task2-kion'
$TransferMetadata = 'E:/reproduction/metadata'
$SourceConfig = Join-Path $OriginalArtifacts 'deadline-20261022/control-r2/job/model/config.json'
$ExpectedConfig = Join-Path $TransferMetadata 'expected-config.json'
$ExpectedConfigSha = '0b00d2d2a5bbbdb6798f4e3cda762e6feb98e96806bc486e0bdcb9b89da817ab'
if ((Get-Item -LiteralPath $SourceConfig).Length -ne 8947263 -or
    (Get-FileHash -LiteralPath $SourceConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedConfigSha) {
    throw 'Исходный E1 config не совпадает с закреплённым файлом'
}
if (Test-Path -LiteralPath $ExpectedConfig) { throw 'Файл назначения уже существует' }
New-Item -ItemType Directory -Path $TransferMetadata -Force | Out-Null
Copy-Item -LiteralPath $SourceConfig -Destination $ExpectedConfig
```

Получатель использует `metadata/expected-config.json` и указанный SHA в команде
ниже. Его `--expected-config-sha256` повторно проверит переданные байты.
Не заменяйте этот файл config от E2, уменьшенной выборки или заново составленным
JSON. Если исходный файл недоступен, этап заблокирован этим конкретным входом.

Config закрепляет runtime: PyTorch `2.8.0+cu126`, NumPy `2.2.6`, Transformers
`4.55.4`, CUDA `12.6`, device `cuda`, CPU2, GPU `NVIDIA GeForce RTX 3070`.
Эти значения входят в полное сравнение. Другая GPU или версии библиотек будут
отвергнуты до optimizer; их поддержку нужно согласовать отдельным изменением
контракта, не редактировать переданный config ради прохождения проверки.

## Подготовить запрос

В корне репозитория выполните следующую команду, подставив свои пути и SHA256
из проверенных записей переноса. Каталог `E:/reproduction` здесь является примером.
Не нужно вручную перечислять сотни входов в запросе.

```powershell
python -B reproduction/train_gpt.py --prepare-request `
  --inventory E:/reproduction/gpt-inventory.json --inventory-sha256 '<SHA256-inventory>' `
  --bindings E:/reproduction/gpt-bindings.json --bindings-sha256 '<SHA256-bindings>' `
  --expected-config E:/reproduction/metadata/expected-config.json --expected-config-sha256 0b00d2d2a5bbbdb6798f4e3cda762e6feb98e96806bc486e0bdcb9b89da817ab `
  --output E:/reproduction/fresh-gpt-E2 --write-request E:/reproduction/fresh-gpt-request.json
```

Команда проверяет указанные SHA256 и согласованность inventory с bindings,
отдельно проверяет байты исходного feature manifest по фиксированному SHA.
Для scope сверяется объявленный SHA; сами байты проверяются при admission provider.
Читает только ограниченные JSON и исходники, не читает массивы.
Создаёт новый запрос и соседний `fresh-gpt-request.preparation.json`.
Существующие файлы и каталог обучения перезаписывать запрещено.

Границы по умолчанию: 604800 секунд, RSS до 12 GiB, свободное место не менее
10 GiB. Доступны параметры `--max-seconds`, `--max-rss-gib`,
`--min-free-disk-gib`; допустимый диапазон проверяется. RSS cap увеличить нельзя.

## Проверить и запустить позднее

Проверка запроса без обучения:

```powershell
python -B reproduction/train_gpt.py --request E:/reproduction/fresh-gpt-request.json
```

Она проверяет известные файлы и ожидаемую конфигурацию. Полноту provider и
байты всех входных массивов проверяет отдельный исполняемый admission.
Наличие файлов в плане само по себе не разрешает считать этот этап пройденным.

После отдельного разрешения на настоящее обучение:

```powershell
python -B reproduction/train_gpt.py --request E:/reproduction/fresh-gpt-request.json --execute-training
```

Нужен интерпретатор с NumPy, PyTorch CUDA и Transformers. Фактически построенный
config сравнивается целиком с ожидаемым до создания optimizer, включая runtime,
схемы признаков и provider. Неподходящее окружение или изменённые входы отвергаются.
Запуски здесь не выполнялись.

## Что сохраняется

Используются исходные `kion_complete_train.run` и negative-worker-r2 sampler:
context64, train3477437, val158760, B128/accumulation1, seed2026,
origin-grouped-v1, CE+auxiliary1, три разных uniform negatives из retrieval top128,
rerank, hidden256/layers4/heads4/bottleneck32/person32, text-all и три people variants.
H5 остаётся горизонтом LR: 136175 updates, warmup0.05, lr0.0001, weight_decay0.01.
Остановка после второй валидации: 54470 updates и 6954874 предъявленных примера.
CPU2 действует весь новый запуск.

Историческая миграция CPU2→CPU4, авария и восстановление partial E2 не воспроизводятся.
Новый `latest.pt` содержит E2, а `selected.json` может выбрать E1 по исходному правилу
NDCG. Следующий этап должен явно закрепить новый checkpoint, config и запись
выполнения. Совпадение старых весов и численная эквивалентность не заявляются.

`admission.json` и `execution.json` сохраняют входы, logical/physical paths,
SHA исходников и адаптеров, состояние и проверки ресурсов. Отмена: создать
`<output>/stop-request.json`. Guard кооперативный: он проверяет deadline, RSS,
свободное место и отмену между операциями, но не прерывает уже выполняющийся
CUDA kernel или системный вызов. Частичные файлы после ошибки сохраняются;
автоматического продолжения нет.

Восемь тестов проходят без ML, включая исходный `run` на искусственном backend:

```powershell
python -B -m unittest discover -s reproduction -p test_train_gpt.py -v
```

Эти тесты проверяют адаптер, H5, остановку E2, отмену, SHA и отсутствие payload
reads при подготовке. Полное provider/CUDA выполнение, качество и ресурсная
пригодность остаются непроверенными до отдельно разрешённого запуска.
