# Ограничения и точный scope

Этот документ дополняет [единый отчёт](REPORT.md). Он относится ко всему исследованию, включая завершённую matched-train-KL проверку трёх sampler-пар. Старые утверждения «matched-KL ещё не выполнен» и «вся работа имеет только один seed» к текущей версии не относятся. При сборке этой версии новые модели не обучались.

## Данные и время

| Граница | Практическое последствие |
|---|---|
| Reused development9–15 августа | Screen256 и full4096 использовались в разработке. Bootstrap не устраняет adaptation к этим данным и не даёт нового holdout. |
| Ранее открытый test16–22 августа | Его нельзя задним числом считать pristine. Новый untouched test в этом комплекте не заявляется. |
| Агрегированные user-item строки KION | Last-watch дата и completion, агрегированный по просмотрам пары, не описывают отдельную сессию или полный порядок событий. Формула агрегации процента в статье не раскрыта. |
| Same-day ordering | Target того же дня исключён из prefix; произвольный порядок хранения внутри дня не считается пользовательским порядком. |
| Ретроспективные признаки | Историческая доступность внешней metadata проверена не полностью; real-time поступление агрегированного completion не восстановлено. |
| Chronology frozen models | RM calibrator из2–8 августа и E2 полного prepared train используются для actor-состояний2–8 августа. Это frozen-proxy offline comparison, а не contemporaneous replay. |
| Нет описанных exposure/propensity/session logs в релизе | Нельзя восстановить causal outcome непоказанного фильма или полностью наблюдаемый retention. Это граница доступного набора, без вывода о внутренних логах платформы. |

## Модели, признаки и candidate boundary

| Граница | Практическое последствие |
|---|---|
| GPT E2 обучен3477437×2 | Пятиэпоховый schedule не завершён; два завершённых прохода не называются пятью. |
| A/B/C262144×3, frozen E2 | Heads не обучались на всех3,5млн примеров. Везде сохранена разница между GPT ancestry и head training. |
| Independently fitted A/B/C | C−B — сравнение heads с разными raw masks, без causal attribution513 добавленных координат или объяснения текущих параметров C. |
| A сохраняет GPT representations/source2 | «Без raw» не означает «без истории» или «неперсонализированная». |
| Natural1024 | Результат ограничен retrieval pool.122 unknown и616 off-pool targets остаются промахами из4096. |
| Supervised train32 / eval1024 | Positive вставлен в обучающий32-way set; в оценке target injection отсутствует. Это разные условия. |
| Нет popularity-only common baseline | Popular участвует в retrieval и RRF, что не доказывает, что C победила отдельную popularity-only модель. |
| Нет historical-seen mask | Повторы просмотренных item разрешены в основном pool. Generated-repeat mask в rollout решает отдельный вопрос. |
| Разные concentration metrics | Top1 share, top10 HHI, unique coverage и entropy описывают разные свойства. Одно число не доказывает персонализацию или общее разнообразие. |

## Reward model и смысл proxy

RM-r2 имеет полезный предсказательный сигнал на20 000 **наблюдаемых** development исходов. Это не доказательство качества произвольных counterfactual пар: randomized exposure, показ именно выбранного кандидата и его causal outcome отсутствуют.

| Граница | Практическое последствие |
|---|---|
| Continuous watched_pct/100 | RM предсказывает долю досмотра, а не вероятность полного досмотра. |
| Calibration | r2 меняет affine calibration прежних деревьев; поздний RMSE0,352039 немного хуже uncalibrated0,352010. Идеальная calibration не заявляется. |
| Item support | Одно fit-наблюдение item — техническая знакомость, без гарантии user-level accuracy или propensity overlap. |
| Cold fallback | Frozen mean≈0,478028 против observed cold mean≈0,277632; cold RMSE0,418036. Unsupported utility обнулена алгебраически, без приписывания реального нулевого досмотра. |
| Меняющаяся support mass | Conditional raw mean, supported-top10 mean и centered effective utility имеют разные знаменатели; их разность не является единым эффектом. |
| Completion prior | `m_fit` — сглаженный средний досмотр фильма, а не popularity count. Декомпозиция proxy не доказывает устранение popularity bias. |
| Reward decomposition | Точная item-арифметика показывает изменение заданного score, без causal attribution пользы, raw-признаков C или единственного механизма reward hacking. |

## RL, train-KL match и неопределённость

Исходный опыт имеет по одному bounded fit на utility при β0,5. Последующий completed follow-up содержит **три matched-KL sampler-пары**, причём seed20261024 калибровочный,20261025/26 prospective. Это расширяет проверку, но не превращает её в независимую репликацию на новых данных.

| Граница | Практическое последствие |
|---|---|
| Train512 и B16 с возвращением | 1000 updates не означают512000 разных состояний и не представляют всю train-популяцию. |
| Head-only от original C | GPT encoder/item table/RM/reference frozen. Полный GPT fine-tuning не проверен. |
| One-step exact objective | Нет PPO/critic, обучения transition model или пятишагового return. |
| Исторический fixed-β confound | Residual advantage RMS≈половина completion; меньший KL приβ0,5 не изолировал semantics. Это историческая граница, дополнительно проверенная matched-KL follow-up. |
| Train-only match3/3 | Ratio0,910646/0,953189/0,914389 попадает в[0,8;1,2] у каждой пары. Match контролирует средний endpoint KL natural1024, без уравнивания каждого состояния или Adam trajectory. |
| Изменение β | Completion0,5 и residual0,1627403014 имеют разные objectives. Сопоставимый KL не доказывает causal effect вычитания prior. |
| Calibration seed24 | Пара участвовала в выбореβ; её включение в descriptive mean3 явно названо. Отдельно приведено mean25-26. |
| Только sampler-вариативность | Меняется minibatch choice stream, но не данные, initialization C, RM или optimizer. Три пары не дают population-level seed CI/p-value. |
| User-bootstrap | 2000 paired user-cluster draws условны на конечных fitted heads. Они не оценивают training-seed uncertainty и не устраняют повторное использование development. |
| Secondary intervals | Exploratory, без multiplicity correction. Исторический completion−C MAP CI лишь едва ниже0. |
| Primary result | Matched residual−completion MAP mean+0,001893, CI[−0,000908;+0,004520]; residual−C−0,000602, CI[−0,002536;+0,001291]. Превосходство не установлено, C point-best. |
| Перенос screen→full | RM utilities рассчитаны наscreen256; они не подставляются как reward full4096. Full relevance/exposure и screen proxy различаются явно. |

## Пятишаговая диагностика и долгосрочная польза

Фактически выполнены192 траектории/960 шагов на16 заранее выбранных начальных позициях **для исходных** C/completionβ0,5/residualβ0,5. Для новых matched-KL residual heads rollout не выполнялся.

| Граница | Практическое последствие |
|---|---|
| Synthetic append | GPT-prefix пересчитывается, но item-ID не становится реальным просмотром, RM-меткой, датой или feedback. |
| Fixed initial natural1024 | Fresh retrieval после каждого шага отсутствует; raw1068, strict7d source/count, clock и RM state s₀ фиксированы. Текущий GPT source logit пересчитывается. |
| Conditional current-prefix KL | C и arm сравниваются на текущем prefix данного arm и маске. Это не trajectory KL и не отдельная C-траектория. |
| Frozen-s₀ discounted proxy | γ0,9 return не является watchtime, наблюдаемым completion или retention. Baseline взят в начальном unmasked s₀. |
| n16 и один исходный rollout seed | Диагностика descriptive, без population CI. Матched-KL seed-проверка эту малую rollout-выборку не расширяет. |
| No-repeat control | Пять уникальных item с exclude-generated обеспечены маской, а не обучением разнообразию. |
| Raw return только all-supported | Categorical средние с разным n нельзя трактовать как matched effect; null outcome не заменяется реальным нулём. |
| Нет outcome на шагах2–5 | Нельзя объявить relevance в несуществующих counterfactual targets, обученную multi-step environment или causal retention gain. |
| Выбранные примеры | Первые три predeclared позиции иллюстрируют поведение, без оценки частоты в популяции. Старые I0 gain/loss кейсы outcome-enriched и private payload исключён. |
| Нет online A/B | Рост реального досмотра, удовлетворённости, personalization и возврата пользователя не доказан. |

## Воспроизводимость и границы поставки

Актуальные команды и actual acceptance текущего архива определяет [reproduction/README.md](../reproduction/README.md). Статическая проверка кода, synthetic tests, aggregate replay, bounded genuine numerical replay, полный inference и raw retraining — разные уровни. Завершение одного не доказывает другие.

Исторический aggregate replay основного отчёта принят на исходной машине:13 PNG побайтово совпали с проверенными originals. Дополнительно **фактически выполнены** два bounded numerical replay из новых минимальных проверочных архивов, распакованных в пути с пробелами на исходной Windows-машине с установленными раздельными средами:

- **B-frozen-encoder:** настоящий GPT forward в producer B16 и три головы для четырёх slots × 1024 кандидата; CUDA FP16 q/base/scores/top10 совпали точно, maximum error 0; native 9,781 с, quiescent, optimizer updates 0. Головы — C и исходные completion/residual seed20261024 с β0,5; это не replay шести matched-KL голов. [Summary](../evidence/reproduction/actor-replay-summary.json), [native receipt](../evidence/reproduction/actor-replay-native.json), [распаковка](../evidence/reproduction/actor-extraction.json).
- **B-frozen-RM:** настоящая CBM prediction и calibration для 4 × 1024 candidate pairs; float64 maximum error 0, historical float32 reward и вся support mask совпали точно, supported 3976/4096; native 4,407 с, quiescent, RM fits/updates 0. [Summary](../evidence/reproduction/rm-replay-summary.json), [native receipt](../evidence/reproduction/rm-replay-native.json), [распаковка](../evidence/reproduction/rm-extraction.json).

Это реальные predictions из включённых frozen inputs/weights, без нового обучения. У четырёх контекстов отсутствуют observed target/reward labels: replay проверяет score/rank parity, без новой MAP или CI. Features, candidate pool и frozen item table не перестраиваются от raw. Установка среды на новой машине, численный повтор всего final delivery archive, отдельный `cached-features` CLI, actor CPU FP32, полный4096 inference и обучение от raw заново не выполнялись. [Все replay receipts и pins](../evidence/reproduction/INDEX.json).

Reader-часть содержит отчёт, агрегаты и рисунки; воспроизводимая часть дополнительно включает frozen weights и ограниченные model-input fixtures с преобразованными item-token histories и user feature vectors. Удалённые user IDs не гарантируют анонимность. Исходные user IDs, observed target/reward labels, raw KION и большие исходные feature caches не включены. Состав определяют [описание входов](../reproduction/INPUTS.md) и manifests. Полный inference требует отдельно доступных дополнительных inputs с проверенными bindings. Их доступность читателю не подразумевается названием пути. Source recipes сохраняют происхождение и параметры; они не заменяют отсутствующие данные.

FP16 scores чувствительны к batch shape/slot. Actor и RM использовали разные NumPy environments; верхние version pins не образуют полного transitive binary lock. В historical sources есть machine-bound constants и native guards. Переносимый небольшой adapter и его доказанная численная паритетность не означают перенос всего pipeline. Полный запуск на другой OS/CUDA/driver не проверен.

Первый исторический screen post-processing завершился strict-parity ошибкой на границе float32 reduction/float64 export. Принятый r2 проверил точный float32 roundtrip262144 значений и воспроизвёл producer reduction; научные utilities остались float64. Failed receipts сохранены, point metrics не менялись. Остановленная старая full4096 попытка также остаётся отдельной неуспешной попыткой; результаты взяты из поздних завершённых shared-feature passes.

Один Git commit не описывает все фактические source bytes: часть кода была untracked/ignored. Для проверки нужны included SHA256 manifests и input/request pins. Отчёт и архив не подтверждают upload, публикацию или фактическую отправку решения.

## Что проверило бы переносимость вывода

Новый временной holdout, более широкий train и независимая вариативность данных/инициализаций проверили бы переносимость. На causal mechanism нужны корректный ablation/sensitivity design и exposure/overlap; на retention — session/return данные и censoring либо online рандомизация. Эти расширения не объявлены выполненными и не включены в текущую сборку без обучения.
