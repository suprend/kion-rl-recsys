# Окружения

Для чтения Markdown/PDF Python не нужен. Для stdlib `verify` нужен Python 3.12+. Команды установки ниже предназначены для получателя; финальная подготовка сама не устанавливает пакеты и не обращается к model hubs.

## Выбор среды

| Среда | Закреплённые верхние зависимости | Назначение |
|---|---|---|
| Source/fixture check | Только Python standard library | Hashes, sizes, AST, metadata/headers; без model inference |
| Aggregate figures | NumPy 2.2.6, Matplotlib 3.11.2 | Рисунки из компактных сохранённых агрегатов |
| Actor | NumPy 2.2.6, Torch 2.8.0+cu126, Transformers 4.55.4, psutil 7.0.0 | Frozen GPT и C/RL heads |
| Reward worker | NumPy 2.5.4, CatBoost 1.2.10, SciPy 1.18.1 | Отдельный CPU process для frozen RM-r2 |
| PDF (по желанию) | ReportLab 4.4.9 — версия source/synthetic проверки builder | Пересборка Markdown в PDF; модельные пакеты не нужны |

Исторические runs использовали Python 3.12.14, Windows x64. [dependencies.json](../source/fullstudy/dependencies.json) сохраняет их происхождение и native backend hashes; [actor requirements](../source/fullstudy/code/requirements/actor.txt), [reward-worker requirements](../source/fullstudy/code/requirements/reward-worker.txt) и [figures requirements](../source/fullstudy/code/requirements/figures.txt) фиксируют верхние версии. Это не полный transitive lock, не wheel archive и не гарантия установки тех же бинарников на другой ОС. Historical driver version этим dependency audit не установлена.

## Создать отдельные окружения

Все пути ниже относительны корню распакованного комплекта. `python` в дальнейших командах означает интерпретатор выбранной среды. Его можно использовать через полное имя без activation.

```text
python -m venv .venv-actor
python -m venv .venv-rm
python -m venv .venv-figures
```

В Windows интерпретатор среды — `.venv-actor/Scripts/python.exe`, в Linux/macOS — `.venv-actor/bin/python`; аналогично для других двух. Например, PowerShell:

```powershell
& ./.venv-actor/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/actor.txt --extra-index-url https://download.pytorch.org/whl/cu126
& ./.venv-rm/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/reward-worker.txt
& ./.venv-figures/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/figures.txt
```

Для Linux/macOS замени только путь интерпретатора и синтаксис его вызова. Torch wheel `+cu126` относится к исторической CUDA 12.6 сборке. На платформе без совместимого wheel нужен отдельный CPU-compatible runtime; он не становится историческим FP16 runtime; его исполнение и численные расхождения оцениваются отдельно. Не заменяй dependency versions молча.

Actor и RM используют разные NumPy. Не устанавливай оба набора в одно окружение и не импортируй CatBoost bridge после actor numerical imports в том же процессе. Запускай RM отдельным выбранным интерпретатором. Наличие CUDA wheel не означает, что CUDA доступна: устройство, backend versions и precision фиксируются в run receipt.

## Численный контракт

Primary numerical gate — CUDA FP16 с exact oracle и historical producer precision. Форма полного producer batch, реальные slots и исходный trim существенны. Пересчёт одного примера вместо полного B16 может изменить близкие ranks и не считается exact historical parity. CPU FP32 запускается отдельно с `--oracle none`: такой run проверяет исполнение forward, а измеренные ошибки относительно historical oracle остаются описательными; допустимая граница для CPU из них не выводится.

Изменённый runtime или precision записываются как новая проверка; exact mismatch должен завершать процесс ошибкой, а не скрываться fallback. CLI принимает только `--oracle exact` и `--oracle none`. Второй режим всё равно сохраняет описательные score errors и top10 agreement после inference, но не назначает допустимый предел и не даёт pass historical parity. Выбрать CPU при недоступной CUDA нужно явно; silent fallback отсутствует.

Команды replay выполняют только inference с `eval`/`no_grad` или `inference_mode`, без optimizer, backward, model download, provider calls и raw preprocessing. Output должен быть новым каталогом; оригинальные fixtures и сохранённые результаты не перезаписываются.

## Фактически проверенная среда

Actual `frozen-encoder` replay из свежего минимального runtime+inputs архива принят на исходной Windows-машине: Python 3.12.14, NumPy 2.2.6, Torch 2.8.0+cu126, Transformers 4.55.4, CUDA FP16/SDPA. [Receipt](../evidence/reproduction/actor-replay-receipt.json) и [summary](../evidence/reproduction/actor-replay-summary.json) связывают точные runtime/input manifests и zero-error parity. Среда уже существовала; fresh dependency installation на новой машине не проверена. CPU FP32 и самостоятельный cached-features CLI не запускались. Отдельный [RM replay](../evidence/reproduction/rm-replay-summary.json) принят из новой проверочной распаковки в fresh CPU process: Python 3.12.14, NumPy 2.5.4, CatBoost 1.2.10; actual prediction/calibration, float64 error 0 и exact float32/support, без RM fit или updates. [RM receipt](../evidence/reproduction/rm-replay-receipt.json) связывает тот же runtime и отдельный RM fixture manifest.

Historical aggregate replay принят на исходной Windows-машине: NumPy 2.2.6 / Matplotlib 3.11.2, 13 PNG совпали побайтно. Это историческое свидетельство уровня A; новый figure replay текущей поставки отдельно не заявляется. Совпадение PNG на другой font/render среде не обещается.
