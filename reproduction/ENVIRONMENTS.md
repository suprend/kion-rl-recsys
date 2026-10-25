# Окружения

Актуальные команды: [REPRODUCING.md](../REPRODUCING.md). Для чтения PDF/Markdown Python не нужен. Подготовка этой редакции не устанавливает пакеты, не скачивает модели и не выполняет обучение.

| Среда | Зависимости | Назначение |
|---|---|---|
| Проверка исходников и метаданных | Python 3.12+, стандартная библиотека Python | Хеши, размеры, синтаксис и заголовки массивов |
| Текущие 65536 запросов и bootstrap | Python 3.12+, NumPy 2.5.4 | `saved_metrics64k.py verify/recompute/bootstrap` |
| Исторические 4096 запросов | Python 3.12+, стандартная библиотека Python | `saved_metrics.py verify/recompute`; без NumPy и моделей |
| Actor | NumPy 2.2.6, Torch 2.8.0+cu126, Transformers 4.55.4, psutil 7.0.0 | Сохранённый GPT и три головы C/RL |
| RM | Python 3.12.14, NumPy 2.5.4, CatBoost 1.2.10, SciPy 1.18.1 | Отдельный CPU-процесс для RM и команд её подготовки, обучения и калибровки |
| Прежние рисунки | NumPy 2.2.6, Matplotlib 3.11.2 | Необязательная пересборка рисунков из JSON |
| Текущий основной PDF | Исходник LaTeX `report-latex/main.tex`; сборка описана в корневом README | PDF из LaTeX; сборщик ReportLab относится к прежнему отчёту |

Прежние запуски использовали Windows x64 и Python 3.12.14. [dependencies.json](../source/fullstudy/dependencies.json) сохраняет версии пакетов. Требования для [actor](../source/fullstudy/code/requirements/actor.txt), [RM](../source/fullstudy/code/requirements/reward-worker.txt) и [рисунков](../source/fullstudy/code/requirements/figures.txt) закрепляют прямые зависимости, но не все транзитивные и системные компоненты. Установка с нуля и точное совпадение вычислений на другой машине не проверены; версия прежнего драйвера не установлена.

## Среда текущих метрик

Из корня клона в PowerShell:

```powershell
python -m venv ../kion-env-metrics
& ../kion-env-metrics/Scripts/python.exe -m pip install numpy==2.5.4
```

В Linux/macOS путь к интерпретатору: `../kion-env-metrics/bin/python`, без PowerShell `&`. GPU, Torch, CatBoost и исходные данные KION для этого пути не нужны. Версия NumPy закреплена для повторения генератора bootstrap. Пересчёт 4096 запросов использует только стандартную библиотеку Python.

## Раздельные среды actor/RM вне клона

Команды подготовки, обучения и калибровки RM проверяют точную версию Python 3.12.14. Python 3.12+ достаточно только для проверок исходников и пересчёта сохранённых метрик. Из корня клона в PowerShell укажите путь к установленному Python 3.12.14 и создайте среды рядом с клоном:

```powershell
$Python31214 = 'C:/path/to/Python31214/python.exe'
& $Python31214 --version
& $Python31214 -m venv ../kion-env-actor
& $Python31214 -m venv ../kion-env-rm
```

Установка в PowerShell:

```powershell
& ../kion-env-actor/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/actor.txt --extra-index-url https://download.pytorch.org/whl/cu126
& ../kion-env-rm/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/reward-worker.txt
```

На Linux/macOS используйте `../kion-env-actor/bin/python` и `../kion-env-rm/bin/python`, без PowerShell `&`. Для пакетов CUDA 12.6 нужны совместимые GPU и драйвер. Другая платформа или точность вычислений проверяется отдельно.

Actor и RM используют разные NumPy. Не объединяйте их в одну среду и не загружайте CatBoost в процесс actor. RM запускается отдельным интерпретатором.

Для необязательных рисунков создайте третью среду `python -m venv ../kion-env-figures`, затем в PowerShell:

```powershell
& ../kion-env-figures/Scripts/python.exe -m pip install -r source/fullstudy/code/requirements/figures.txt
```

В командах рисунков из [README](README.md) `python` означает этот интерпретатор.

## Условия численного совпадения

Основной режим проверки: CUDA FP16, исходный пакет из 16 примеров, позиции 0–3 и `--oracle exact`. Размер пакета влияет на близкие ранги. Требуется совпадение q, базовых оценок, оценок голов и top10 с эталоном; несовпадение завершает процесс ошибкой. При недоступной CUDA автоматического перехода на CPU нет.

CPU FP32 выбирается явно: `--device cpu --precision fp32 --oracle none`. Он проверяет предсказания в другой среде и сохраняет величину расхождений, но не устанавливает `oracle_passed: true`.

До загрузки численных библиотек CLI проверяет описатели кода и входов, границы путей, SHA256, размеры и типы массивов. NPY/NPZ читаются без pickle, веса: через `weights_only=True`. Выходной каталог должен отсутствовать, родительский каталог должен существовать; он не должен пересекаться с кодом или входами. Actor принимает `--cpu-threads` от 1 до 4, RM использует два потока. После вычислений хеши проверяются повторно. CLI не задаёт системные квоты RAM/VRAM и времени.

## Прежние проверки

[Протоколы](../evidence/reproduction/INDEX.json) подтверждают CUDA FP16 exact для actor и нулевую ошибку RM на исходной машине с готовыми средами. Установка с нуля, самостоятельные CPU FP32 и cached-features запуски не проверялись.

После этих запусков усилена проверка метаданных RM: текущий `runtime-manifest.json` отличается от записанного в прежних протоколах. Поэтому они не подтверждают новый запуск текущей версии. Для новой проверки используются её хеши и новый протокол.
