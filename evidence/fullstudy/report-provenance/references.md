# Первоисточники и границы метода

Проверено 25 октября 2026. Это библиография и терминологическая опора отчёта; она не подтверждает статусы запусков или экспериментальные числа. Ниже разделены прямое содержание источника и применение к нашей постановке. Использованы восемь работ; ссылки на авторский код и описание релиза дополняют соответствующую работу.

1. **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). _Language Models are Unsupervised Multitask Learners_.** [Технический отчёт OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), §2 и §2.3; [авторская реализация causal attention mask](https://github.com/openai/gpt-2/blob/master/src/model.py#L52-L87).

   Прямая опора: авторегрессионная факторизация вероятности последовательности и Transformer GPT-2; код ограничивает внимание доступным префиксом. В нашем отчёте корректно «каузальный decoder-only backbone архитектуры GPT-2, обученный предсказывать следующий item-ID». Замена текстовых токенов на ID и feature-augmented конфигурация — наша реализация, а не результат этой статьи. Термин GPT-2 не означает использование её pretrained текстовых весов или совпадение с исходными размерами модели.

2. **Covington, P., Adams, J., Sargin, E. (2016). _Deep Neural Networks for YouTube Recommendations_. RecSys 2016.** [Страница Google Research](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/), [авторский PDF](https://research.google.com/pubs/archive/45530.pdf), §2 и Figure 2.

   Прямая опора: отдельные стадии candidate generation и ranking; ранжирование использует более подробные user/item признаки, допускается смешение источников кандидатов. Это методологический контекст для нашего mixed pool и residual ranker. Работа не доказывает оптимальность наших квот, размера пула, признаков или головы C. При отсутствии target в pool reranker не может вернуть его: это следствие нашей двухстадийной постановки. Авторы также отделяют offline диагностику от окончательной online A/B проверки.

3. **Li, L., Chu, W., Langford, J., Schapire, R. E. (2010). _A Contextual-Bandit Approach to Personalized News Article Recommendation_. WWW 2010, 661–670.** [Авторский PDF](https://www.schapire.net/papers/www10.pdf), §2 и §4; [DOI](https://doi.org/10.1145/1772690.1772758).

   Прямая опора: контекст, выбор одного действия и обратная связь о выбранном действии; offline replay в статье требует специально собранного случайного трафика. Наше обозначение H1 как contextual policy optimization — классификация одношагового objective, а не заявление о воспроизведении LinUCB или работе с пользователями. Здесь RM предоставляет модельные scores всему фиксированному pool; это не logged bandit feedback и не unbiased replay. Точная сумма по кандидатам точна относительно заданного RM objective, но не относительно неизвестной пользовательской пользы.

4. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). _Proximal Policy Optimization Algorithms_. arXiv:1707.06347v2, 28 августа 2017.** [Метаданные](https://arxiv.org/abs/1707.06347v2), [PDF](https://arxiv.org/pdf/1707.06347v2), §3–4, equations (7)–(8).

   Прямая опора: PPO использует sampled surrogate с отношением текущей policy к old policy; помимо clipped objective статья рассматривает adaptive KL penalty. Поэтому отсутствие clipping само по себе не определяет «не PPO». В нашем H1 непосредственно дифференцируется ожидаемая utility по всем действиям, а KL направлен от текущей policy к фиксированной supervised reference. Здесь нет соответствующего old-policy importance-ratio surrogate. Это сравнение методов по их objective, а не экспериментальное сравнение качества с PPO.

5. **Petrov, A. V., Macdonald, C. (2024). _Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning_. GenRec@WWW 2024; arXiv:2403.04875v1, 7 марта 2024.** [Метаданные](https://arxiv.org/abs/2403.04875v1), [полный текст](https://arxiv.org/html/2403.04875v1), §3.1–3.2, Appendix A.3 и B. Рекомендована в задании.

   Прямая опора: Next-K генерирует список по одному item, включая уже сгенерированную часть в контекст; после teacher–student стадии авторы применяют PPO с отдельной value model к целям качества списка. Работа мотивирует исследование reward–accuracy tradeoff. Наши supervised ID next-item, head-only H1 и synthetic append являются более узкой постановкой: они не воспроизводят teacher–student/PPO обучение GPTRec и не устанавливают оптимизацию целого slate во время fit.

6. **Matveev, A., Makeev, S., Krasilnikov, A., Baikalov, V., Liamaev, S., Khrylchenko, K. (2026). _Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction_. KDD 2026 workshop; arXiv:2607.02818v2, 27 июля 2026.** [Метаданные](https://arxiv.org/abs/2607.02818v2), [полный текст](https://arxiv.org/html/2607.02818v2), §2–3 и §4.3. Рекомендована в задании.

   Прямая опора: session-level off-policy REINFORCE на Yambda-5B, оценка behavior policy, поправки importance weights и feedback model для модельных переходов. Для test-time rollouts состояние пополняется действием и предсказанным feedback. Авторы явно ограничивают вывод: выбор и оценка по одной feedback model не дают независимого доказательства улучшения реальной policy (§4.3). Наш append только item-ID при фиксированном observed RM state не воспроизводит эти переходы или off-policy correction; его surrogate return не является session retention.

7. **Gao, L., Schulman, J., Hilton, J. (2023). _Scaling Laws for Reward Model Overoptimization_. ICML 2023, PMLR 202:10835–10866.** [Публикация PMLR](https://proceedings.mlr.press/v202/gao23h.html), [PDF](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf), §1–2. Препринт arXiv:2210.10760 впервые опубликован в 2022 году.

   Прямая опора: в синтетической схеме с gold RM и proxy RM усиление оптимизации proxy может ухудшать gold score; изучены RL и best-of-n. Это основание проверять пределы learned proxy, KL и независимые метрики. Перенос механизма на наш KION — гипотеза. Рост RM score одновременно с падением next-item метрик устанавливает конфликт измеренных целей; он сам по себе не доказывает ни причинный reward hacking, ни ухудшение истинной пользовательской удовлетворённости. KL — ограничение сдвига, а не сертификат корректности RM.

8. **Petrov, A., Safilo, I., Tikhonovich, D., Ignatov, D. (2022). _MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation_. arXiv:2209.00325v1, 1 сентября 2022.** [Статья](https://arxiv.org/html/2209.00325v1), §2.2; [официальный README релиза](https://github.com/MobileTeleSystems/RecTools/blob/main/datasets/KION/README.md).

   Прямая опора: при повторных взаимодействиях user–item сохраняется последнее взаимодействие, а длительность и процент просмотра агрегируются по всем просмотрам; формула агрегации процента не раскрыта. Поэтому дата относится к последнему просмотру, но `watched_pct` нельзя называть откликом только этого просмотра. Статья указывает период 13.03–22.08.2021; окончание 22.08.2022 в README противоречит статье. Описанный релиз не предоставляет exposure/propensity logs или полного журнала визитов: невозможность восстановить causal retention — методологический вывод из состава и агрегации данных, а не утверждение о том, что платформа никогда не собирала такие логи.

## Краткое позиционирование для основного текста

Рекомендованные заданием работы задают два более широких направления: Petrov и Macdonald обучают Next-K GPTRec с PPO и value model для качества целого списка; Matveev и соавторы оптимизируют session-level return с off-policy correction и feedback model. Наш опыт проверяет более узкий вопрос: изменение головы рекомендательной policy над frozen GPT при точном одношаговом ожидании learned utility и KL к фиксированной supervised reference. Пятишаговый append исследует генерацию условного списка и сдвиг synthetic context; пользовательские переходы и feedback он не моделирует. [GPTRec](https://arxiv.org/html/2403.04875v1), [session-level retrieval](https://arxiv.org/html/2607.02818v2).

Для проверки proxy сохраняем next-item метрики, концентрацию и support рядом с RM score. Исследование overoptimization мотивирует такую проверку, но обнаруженный конфликт целей не идентифицирует его механизм в KION. Вывод о пользе реальным пользователям потребует самостоятельного evidence. [Gao et al.](https://proceedings.mlr.press/v202/gao23h.html).

## Точная редакционная поправка к objective

Если `b(s)` задан как reference-взвешенное среднее supported utility, он обычно ненулевой. Корректная фраза: «Центрирование относительно reference обеспечивает `E_pi0[a(s,i)] ≈ 0`». При support mask `a=m·(u−b)` — именно эффективная utility полного action space; unsupported действия имеют ноль. Это не наблюдённый advantage реальной среды. Пересказ формулы — наш аналитический вывод, не отдельный результат цитируемой статьи.
