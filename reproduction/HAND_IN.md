# Карта комплекта для передачи

Комплект подготовлен для локальной проверки. Факт загрузки, публикации или отправки решения не заявляется.

| Путь от корня распаковки | Для чего нужен |
|---|---|
| `reports/REPORT.pdf` | PDF-версия отчёта; включение определяется финальным manifest |
| `reports/REPORT.md` | Единый актуальный отчёт по полному исследованию и matched-KL/seed продолжению |
| `reports/LIMITATIONS.md` | Сводка методологических границ и доступности воспроизведения |
| `reports/figures/` | Рисунки в PNG/SVG |
| `reports/render_report.py` | Переносимый checker и optional PDF builder |
| `evidence/fullstudy/` | Компактные результаты полного исследования и historical execution receipts |
| `evidence/matched-kl/` | Компактные результаты matched-KL/seed продолжения |
| `source/fullstudy/` | Точный исторический snapshot: 261 текстовый файл, manifests, dependency inventory и recipes |
| `source/matched-kl/` | Дополнительный код matched-KL/seed продолжения |
| `source/manifest.json` | Общий source allowlist с hashes/размерами |
| `PACKAGE_MANIFEST.json` | Финальный точный allowlist содержимого архива и hashes |
| `delivery/package.py`, `delivery/PACKAGING.md` | Проверка ZIP и безопасная распаковка в новый каталог |
| `reproduction/kion_replay/`, `reproduction/runtime-manifest.json` | Portable verify/infer/reward CLI и точные source pins |
| `reproduction/inputs/manifest.json` | GPT + три heads, реальные model-input histories/features и per-file hashes |
| `reproduction/inputs-rm-r1/manifest.json` | RM weights/features/references для четырёх contexts × 1024 candidates |
| `evidence/reproduction/INDEX.json` | Actual export/extraction/inference proof, с раздельным статусом каждого этапа |
| `reproduction/README.md` | Порядок проверки и границы A/B/C |
| `reproduction/external-provenance/INDEX.json` | Compact external metadata exports и список исключённых feature prerequisites; для bounded inference не нужны |
| `reproduction/INPUTS.md` | Доступность входов, provenance и excluded raw/model inputs |
| `reproduction/ENVIRONMENTS.md` | Разделённые actor/RM/figures среды и precision contract |
| `research/SOURCE_CLOSURE.md` | Закрытие local source dependencies и historical runtime assumptions |

Начни с актуального отчёта, затем stdlib source verification из [README](README.md). Frozen GPT + три heads прошли CUDA FP16 exact replay из новой проверочной распаковки; [summary](../evidence/reproduction/actor-replay-summary.json) фиксирует нулевые q/base/score errors и точные top10. Отдельный [RM replay](../evidence/reproduction/rm-replay-summary.json) также принят: actual CBM prediction для 4096 candidate pairs, error 0 / exact float32/support, без fits/updates. Численная приёмка относится к новым минимальным проверочным распаковкам на исходной машине; финальные проверки hand-in ZIP и PDF фиксируются отдельно. Для numerical replay нужны именно export manifest и соответствующая модельная среда; имена файлов без hashes не заменяют входной контракт. Финальный archive manifest определяет реально включённый состав; этот документ не обещает отсутствующие внешние файлы.

Практическая граница: полный raw KION, все feature stores и исходные training checkpoints не входят в компактное решение. Ограниченный fixture проверяет frozen inference на выбранных exported model inputs; воспроизводимость raw-to-feature pipeline и новое обучение оцениваются отдельно. Negative или uncertain RL outcomes являются результатом исследования, а не незавершённым заданием.

Перед передачей получатель может повторить source/fixture verifier, численный replay в новом output directory и сверить receipts с archive manifest. Проверки не требуют внутренних инструментов Codex, PAS, доступов к машине автора или запуска ночной кампании. Для самого чтения достаточно распакованного комплекта.

## Проверить или пересобрать PDF

Готовый PDF читается без Python. Пересборка не нужна для просмотра Markdown и numerical replay. Проверка документа использует standard library:

```text
python -B reports/render_report.py --source reports/REPORT.md --check
```

Для собственной сборки нужны ReportLab 4.4.9 и DejaVu Sans regular/bold. Финальный отчёт содержит математические Unicode glyphs, которые Arial на проверенной машине покрывает не полностью. DejaVu из Matplotlib прошёл полный font/table preflight; шрифты встраиваются в PDF. TTF в комплект не входят: их можно взять из установленной figures environment.

После создания figures environment из [ENVIRONMENTS.md](ENVIRONMENTS.md) в PowerShell:

```powershell
$FigurePython = './.venv-figures/Scripts/python.exe'
& $FigurePython -m pip install reportlab==4.4.9
$DejaVuDir = & $FigurePython -B -c "import matplotlib; from pathlib import Path; print(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf')"
& $FigurePython -B reports/render_report.py --source reports/REPORT.md --output reports/REPORT-rebuilt.pdf --font (Join-Path $DejaVuDir 'DejaVuSans.ttf') --font-bold (Join-Path $DejaVuDir 'DejaVuSans-Bold.ttf')
```

Это команды установки/сборки для получателя; автоматически они не выполняются. На Linux/macOS используй `.venv-figures/bin/python`: команда `-c` выше выводит тот же portable font directory, затем передай полные пути к двум TTF через `--font` и `--font-bold`. Выбери новое имя output. ReportLab/Matplotlib — верхние зависимости, а не полный transitive lock. Изменённый PDF нужно визуально проверить по всем страницам: переносы таблиц, кириллица, рисунки и отсутствие clipping. Successful build сам по себе не доказывает качество layout или numerical correctness.
