# Входы и происхождение

## Что доступно после распаковки

Компактные результаты и source snapshot включены в комплект. [source/manifest.json](../source/manifest.json) перечисляет точные сохранённые байты исходников; [excluded-inputs.json](../source/fullstudy/code/recipes/excluded-inputs.json) — исторические исключённые входы. Hash в историческом descriptor подтверждает ожидаемую идентичность, но не означает, что файл поставляется или был заново прочитан при упаковке.

Создан actor/GPT export [inputs/manifest.json](inputs/manifest.json): 17 payload files, 68 375 375 байт плюс 7578 байт manifest; SHA256 manifest `e680dd145b2bfdc812107fb9a257bf16f6d1ef5a53c07845f7dc21838146a52c`. [Export receipt](../evidence/reproduction/actor-export-native.json) фиксирует только извлечение: fresh inference проверяется отдельными receipts, а не export-флагом.

| Exported inputs | Dtype / shape |
|---|---|
| GPT item-token prefixes, lengths | int64 `[16,64]`, `[16]`; original trim 23 |
| GPT user feature vectors | float32 `[16,3252]` |
| Frozen item table | float32 `[15418,256]` |
| Cached q / retrieval logits | float32 `[4,256]` / `[4,1024]` |
| Candidate token/raw IDs | int32/int64 `[4,1024]` |
| Head raw/source features | float32 `[4,1024,1068]` / `[4,1024,2]` |
| GPT and C/completion/residual weights | Pure tensor state dictionaries; per-file hashes в manifest |
| Historical oracle scores | float32 `[4,1024]` для каждой головы; validation-only |

Четыре строки выбраны как первые позиции frozen screen cache независимо от outcome; GPT сохраняет полный producer B16 и slots 0–3. Fixture содержит преобразованные истории. Source user IDs, имена, observed target/reward labels и title mapping исключены; это не гарантия анонимности. Actual actor inference принят отдельными [summary/receipts](../evidence/reproduction/INDEX.json); export manifest намеренно описывает только export stage. Точный final archive состав задаёт `PACKAGE_MANIFEST.json`.

RM export [inputs-rm-r1/manifest.json](inputs-rm-r1/manifest.json): 9 payload files, 33 248 280 байт плюс 4381 байт manifest; SHA256 manifest `7b3e7dbefde3a6d0e113c0340dbbfc425d223574a7c4cd13db7927fb29b30380`. Он содержит float32 context `[4,2079]`, candidate features `[4,1024,2008]`, candidate raw IDs, frozen fit counts/prior, validation-only references и исходную `model.cbm` (163 368 байт, SHA в таблице ниже). Это четыре contexts × 1024 candidates = 4096 candidate pairs. [RM export receipt](../evidence/reproduction/rm-export-native.json) и [extraction check](../evidence/reproduction/rm-extraction.json) фиксируют export/перенос. Отдельный [fresh RM replay](../evidence/reproduction/rm-replay-summary.json) выполнил CBM prediction: float64 maximum error 0, historical float32 reward/support exact; 3976 из 4096 pairs supported, RM fits/updates 0. [Все actual proofs](../evidence/reproduction/INDEX.json) имеют собственные hashes; SHA256 индекса `0a478130fa43a8a8b4c1aa9e5480395683dbac0386fcf61166fb3b33d905d3c1`.
Полный frozen pipeline дополнительно требует E2 config/weights, подготовленные history tokens и user features, feature provider/cache, mapping, cohorts, graph и RM-r2 inputs. RM inference имеет отдельную schema 2079 context + 2008 candidate и модель CatBoost. Synthetic inputs полезны для проверки кода, но не воспроизводят исторические пользовательские predictions.

## Внешние признаки полного recipe

[external-provenance/INDEX.json](external-provenance/INDEX.json) связывает 13 historical metadata exports (17 540 011 байт): 12 прямых копий и один явно обозначенный sanitized derivative acquisition receipt. SHA256 самого индекса: `a0bbbc4f7939a51d92815b7adb99755db36e28338402cee6dba5e095d718900e`. Из derivative удалены все 1321 response-header objects для исключения служебных client/cookie headers; он не выдаётся за native exact input. Оригинальный source hash и hash включённой копии различаются явно. Включены public film/person identity ledger, MovieLens receipt, pageview query/acquisition plan, ordered external schema 161 и declarations. Они позволяют изучить provenance и исходную population без доступа к дискам автора; embedded absolute paths — исторические ссылки.

Это не полный feature-input bundle. Исключены 149 dated external banks по 9 929 320 байт (итого 1 479 468 680 байт), Wikimedia response bodies, MovieLens archive и per-work aggregate/prior payloads. Их ожидаемые identities описаны сохранёнными receipts, но bytes не считаются заново проверенными этой копией metadata. Полный raw recipe потребует восстановления этих входов; ограниченный GPT/head/RM fixture не читает данный каталог.
## Raw KION: получить отдельно

Raw `interactions.csv`, `users.csv`, `items.csv` в компактную поставку не входят. Описание и условия получения следует проверить у [организаторов RecSys Challenge 2021](https://ods.ai/competitions/competition-recsys-21/data); первичное описание релиза — [статья KION](https://arxiv.org/abs/2209.00325). Доступность URL не предоставляет новых прав на данные, веса или производные представления. Данные не скачиваются командами проверки и при финальной подготовке не загружались.

Исторические [download manifest](../evidence/raw-provenance/manifest.json) и [verification metadata](../evidence/raw-provenance/verified-download.json) указывают pinned сторонний mirror: [data_original.zip, revision f69775be](https://raw.githubusercontent.com/irsafilo/KION_DATASET/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip). Это provenance конкретных использованных байтов, а не обозначение mirror официальным источником. Архив: 78 795 295 байт, SHA256 `50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b`.

| Файл после разрешённого получения и распаковки | Байты | Ожидаемый SHA256 |
|---|---:|---|
| `interactions.csv` |180 032 258|`201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf`|
| `users.csv` |29 945 389|`37b2945240628a3f215e1fb5f79b4cff386d8988b21fa61e5b135aef4541182c`|
| `items.csv` |31 836 561|`027c3a0027b4fe4a8db9d5fa626ab40bab507c96aac81d462db0771eb2cb4a8e`|

Metadata предупреждает, что отдельный `interactions.csv.zip` был усечён: для исторического recipe использовался полный архив и каталог `data/raw/kion/full/`. Сравнивать следует сами bytes CSV, не только имена. Перекодирование, другой line ending или пересохранение меняет hash.


Для восстановления текстовых признаков полного pipeline нужны отдельно полученные файлы `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Они не входят в поставку. Точные пять имён, размеры и SHA256 находятся в `MODEL_FILES` сохранённого `source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts/kion_text_embeddings.py`; загрузчик использует `local_files_only=True`. Frozen table в ограниченном fixture заменяет эту стадию только для inference уже обученной модели.
## Идентичность исторических моделей

Это references к оригинальным входам, а не обещание их поставки. Полный список — в `excluded-inputs.json`; matched-KL pins сохранены вместе с его stage sources и selection metadata.

| Роль | Исторический SHA256 |
|---|---|
| GPT E2 `latest.pt` / selected epoch 2 |`5e7101857c2dbe100fb4e1587c0369abc95c108303ed149123ee183f62aeac26`|
| E2 `config.json` |`e3b4b1730895fceb1b7d4ea6357aea783c046f5e87cacd7d4ff933cf94d45d45`|
| C `head.pt` |`9c42a75a0c3092ceef3ee8f94b5a62b9b69808a81a42be4ed8b1c3acec277ea7`|
| Original completion latest1000 |`436d735d67fd861e6b925719b9e3597582ec405eed1b13ede3db3671e1ce37b7`|
| Original residual latest1000, beta 0,5 |`29d20cd3d058b4ccac52d94f3ad946891f3839601fe935817f7c7b89f09b1ee3`|
| RM-r2 `model.cbm` — те же деревья, что r1 |`b6ed4c6427129afcddec017e2f64d9c3fb59fb31fc300e5c9d074964a21ddd62`|
| RM-r2 `scorer.npz` |`d5df55586a9df54b5e121d494a149764d956750ac3caaf2b4c1ada27a93243d4`|
| Prepared context 64 `metadata.json` |`f02acaa9a5e5532e79b060279ba8a5027f268f202132e85ebab97cd4912fbd11`|

Новые минимальные exports могут иметь другие hashes: checkpoint с optimizer/RNG и чистый inference state dictionary — разные файлы. Их соответствие должно быть явно связано source hash, экспортным manifest и проверкой inference; одинаковое имя не доказывает происхождение.

## Безопасное использование весов

Используй только доверенные собственные exports либо отдельно полученные входы с проверяемым происхождением. Portable replay должен проверять manifest, границы путей, SHA256, размеры/dtypes/shapes и source pins до numerical import/загрузки. NPY/NPZ читаются без pickle; state dictionaries — через `torch.load(..., weights_only=True)`. Не следует обходить эти проверки загрузкой произвольного checkpoint с `weights_only=False`.

Fixture с удалёнными user IDs не следует автоматически считать анонимным: representations и features могут сохранять информацию об исходных примерах. Право распространять исходные данные или производные входы отдельно не выводится из наличия source bundle. Комплект не вводит новую лицензию; публикация/отправка не выполнялась.
