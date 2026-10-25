"""Source-bound Russian character/occupation relations for KION (CPU, no models).

Fixed profession ontology; explicit grammatical templates only. Keyword and title
mentions never establish a character relation. Zero means no accepted evidence,
not proof of absence. The catalogue is retrospective, not historical metadata.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

_spec = importlib.util.spec_from_file_location("_profession_base", Path(__file__).with_name("kion_pack_semantics.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
CONTRACT = "kion-character-profession-relations-v1"


def _man(stem):
    return stem + r"(?:|а|у|ом|е|ы|и|ов|ам|ами|ах)"


def _woman(stem):
    return stem + r"(?:|а|ы|е|у|ой|ою|ей|ею|и|ам|ами|ах)"


def _tel(stem):
    return stem + r"(?:ь|я|ю|ем|е|и|ей|ям|ями|ях)"


# Closed, inspectable word forms: no arbitrary stems/automatic ontology closure.
# Rare aliases are grounded in the approved table; plovchikha is explicitly in
# its description-positive example. Bare doctor, actor, clergy, astronaut absent.
FORMS = {
    "medical_practitioner": [_man(s) for s in ("врач", "медик", "хирург", "нейрохирург", "акушер", "фельдшер", "парамедик", "стоматолог")] + [_woman("акушерк")],
    "nurse": [r"медсестр(?:а|ы|е|у|ой|ою|ам|ами|ах)", r"медсестер", _woman("медсестричк")],
    "mental_health_specialist": [_man(s) for s in ("психолог", "психиатр", "психотерапевт")],
    "veterinarian": [_man("ветеринар")],
    "teacher": [_tel("учител"), _woman("учительниц"), r"школьн(?:ый|ого|ому|ым|ом|ые|ых|ыми)\s+" + _tel("преподавател")],
    "scientist": [r"учен(?:ый|ого|ому|ым|ом|ая|ой|ую|ые|ых|ыми)"] ,
    "engineer": [_man("инженер")],
    "lawyer": [_man("юрист"), _man("адвокат")],
    "police": [r"полицейск(?:ий|ого|ому|им|ом|ая|ой|ую|ие|их|ими)", r"(?:офицер|инспектор|капитан|лейтенант)(?:а|у|ом|е|ы|ов|ами|ах)?\s+полиции"],
    "investigator": [_tel("следовател"), r"(?:частн(?:ый|ого|ому|ым|ом|ые|ых|ыми)|полицейск(?:ий|ого|ому|им|ом|ие|их|ими))\s+" + _man("детектив"), _man("детектив")],
    "military": [_man("солдат"), r"военнослужащ(?:ий|его|ему|им|ем|ая|ей|ую|ие|их|ими)", r"военн(?:ый|ого|ому|ым|ом|ые|ых|ыми)", r"(?:военн(?:ый|ого|ым)|морск(?:ой|ого|им)|военно-морск(?:ой|ого|им))\s+" + _man("офицер")],
    "firefighter": [r"пожарн(?:ый|ого|ому|ым|ом|ые|ых|ыми)", _man("пожарник"), r"начальник(?:а|у|ом|е)?\s+пожарной\s+охраны"],
    "writer": [_tel("писател"), _woman("писательниц")],
    "journalist": [_man("журналист"), _woman("журналистк"), r"(?:бывш(?:ий|его|им)|военн(?:ый|ого|ым))\s+" + _man("корреспондент")],
    "musician_singer": [_man("музыкант"), _woman("музыкантк"), r"пев(?:ец|ца|цу|цом|це|цы|цов|цам|цами|цах)", _woman("певиц")],
    "visual_artist": [_man("художник"), _woman("художниц")],
    "cook": [_man("повар"), _man("шеф-повар"), _man("кулинар")],
    "professional_driver": [_tel("водител"), _man("дальнобойщик")],
    "pilot": [_man("пилот")],
    "athlete": [_man("спортсмен"), _woman("спортсменк"), _man("боксер"), _man("кикбоксер"), _man("футболист"), _woman("пловчих")],
    "bar_waiter": [_man("бармен"), _woman("барменш"), _man("официант"), _woman("официантк")],
    "accountant": [_man("бухгалтер"), _woman("бухгалтерш")],
    "spy_intelligence": [_man("шпион"), _woman("шпионк"), r"(?:секретн(?:ый|ого|ому|ым|ые|ых|ыми)|тайн(?:ый|ого|ому|ым|ые|ых|ыми)|двойн(?:ой|ого|ому|ым|ые|ых|ыми))\s+" + _man("агент"), _man("агент") + r"\s+(?:цру|(?:советской\s+)?разведки)"],
}
FLAGS = {key: ("role_spy_intelligence" if key == "spy_intelligence" else "profession_" + key) for key in FORMS}
PATTERNS = {key: re.compile(r"(?<!\w)(?:" + "|".join(sorted(forms, key=len, reverse=True)) + r")(?!\w)", re.I) for key, forms in FORMS.items()}
NAME_WORD = r"[А-ЯЁ][а-яё]{1,}(?:[-'][А-ЯЁа-яё][а-яё]*)?"
NAME = NAME_WORD + r"(?:\s+" + NAME_WORD + r"){0,2}"
_NAME = re.compile(NAME)
_SUBJECT = re.compile(r"(?:" + NAME + r"|\b(?:он|она|они|его|ее|него|нее|ему|ей|ним|ней|герой|герои|героиня|героини|героинь|персонаж|персонажи|муж|жена|сын|дочь|парень|девушка|женщина|мужчина|люди)\b)")
_NAME_STOP = set("главная главный молодой молодая бывший бывшая вскоре история фильм сериал драма этот это все два три после чтобы если когда теперь однако оказывается там здесь вместе однажды сначала раньше прежде впрочем оказавшись потеряв для по на до много мало например обычной обычная обычный".split())
_MODIFIERS = r"(?:(?:весьма|очень|довольно|крайне)\s+)?(?:(?:[а-яё-]+(?:ый|ий|ой|ая|ое|ые|ого|его|ей|ую|ых|ым|им|ом|ем|ому|ему|ими|ыми|имся|ымся|егося|ейся))\s+|и\s+|преданный\s+делу\s+){0,5}"
_ACTION = re.compile(r"^\s*(?:(?:" + NAME + r")\s+)?(?:котор(?:ый|ая|ые)\s+)?(?:[а-яё]+\s+){0,2}(?:живет|живет|работает|служит|получает|берет|расследует|ищет|решает|помогает|спасает|обнаруживает|оказывается|возвращается|знакомится|пытается|ведет|отправляется|влюбляется|становится|встречает|попадает|приезжает|узнает|выполняет|выполняют|работают|служат|живут|возвращаются|пытаются|получают|отправляются|встречают|начинает|начинают|выписывает|ставят|вынужден|вынуждена|вынуждены|пропадает|погибает|теряет|ведут|приходит|проводит|создает|выясняет|поступает|должен|должна|должны)\b", re.I)
_EMPLOYMENT = re.compile(r"\b(?:(?:начинает|продолжает)\s+работать|работает|работают|работал|работала|работали|служит|служат|служил|служила|служили|трудится|трудился|трудилась|устроился|устроилась|устраивается|становится|стал|стала|был|была|были|является|нанят|нанята|нанимает[^,;.!?]{0,35}?в\s+качестве|занимает\s+должность)\s+(?P<link>(?:[а-яё-]+\s+){0,2})$", re.I)
_PAST = re.compile(r"(?:\bбывш\w+|\bэкс|\bпрежде|\bраньше|\bкогда-то|\bбыл[аи]?|\bработал[аи]?|\bслужил[аи]?|\bтрудил(?:ся|ась))\b", re.I)
_FUTURE = re.compile(r"\b(?:станет|станут|будет(?:\s+работать)?|будут(?:\s+работать)?)\s+(?:[а-яё-]+\s+){0,2}$", re.I)
_ASPIRATION = re.compile(r"(?:\b(?:мечта\w*|хочет|хотят|желает|намерен\w*|собирается|планирует|надеется|решил|решила|поможет|удастся)\b[^,;.!?]{0,55}\bстать|\bчтобы\s+стать)\s+(?:[а-яё-]+\s+){0,2}$", re.I)
_TRAINING = re.compile(r"\b(?:учится|учился|училась|учатся|обучается|учеба|школ[ауые]|курс\w*)\b[^,;.!?]{0,45}(?:на\s+)?(?:[а-яё-]+\s+){0,1}$", re.I)
_NEG = re.compile(r"(?:\bне\s+(?:был[аи]?|является|работает|работал[аи]?|служит|служат|служил[аи]?|стал[аи]?|станет|будет|хочет|хотят|мечтает|собирается|учится|обучается|играет|сыграл[аи]?)(?:\s+(?:в|на|стать|работать|настоящ\w*|опытн\w*|успешн\w*)){0,3}\s+|\b(?:не|никакой\s+не)\s+(?:настоящ\w*\s+)?)$", re.I)
_HYPOTHETICAL = re.compile(r"\b(?:если\s+бы|будто|якобы|притворяется|выдает\s+себя|выдавая\s+себя|под\s+видом|переодел\w*|играет\s+в\s+|представляет\s+себя|как\s+(?:заправский|настоящий))\b", re.I)
_FOREIGN = re.compile(r"\b(?:в\s+(?:другом|предыдущем|прошлом)\s+фильме|ранее\s+сыграл|извест\w+\s+ролью|фильмограф\w*)\b", re.I)
_PRODUCTION = re.compile(r"\b(?:экранизаци\w*|по\s+(?:мотивам|роману|книг(?:е|ам)|повести|рассказу|сказк\w*|произведени\w*)|роман\w*\s+(?:известного\s+)?писател\w*|книг\w*\s+писател\w*|автор\s+сценария|сценарист\w*|режиссер\w*|актер\w*|актрис\w*|продюсер\w*|композитор\w*|съемочн\w*|съемк\w*|эксперт\w*|комментатор\w*|снял[аи]?|сняли|снимает|снявш\w*)\b", re.I)
_PROMO = re.compile(r"\b(?:смотрите|посмотрите|рекомендуем|предлагаем|приглашаем|любителям|поклонникам|лучший\s+(?:фильм|сценарий)|преми\w*|номинаци\w*)\b", re.I)
_ATTRIBUTIVE = {
    "scientist": re.compile(r"^\s+(?:совет|степен\w*|звание|труд\w*|мир|сообщество)\b", re.I),
    "police": re.compile(r"^\s+(?:участ\w*|академи\w*|машин\w*|отдел\w*|управлен\w*|операци\w*|расследован\w*|сериал\w*|драм\w*|комеди\w*|триллер\w*)\b", re.I),
    "military": re.compile(r"^\s+(?:учен\w*|корреспондент\w*|драм\w*|фильм\w*|баз\w*|лагер\w*|положен\w*|врем\w*|корабл\w*|операци\w*|школ\w*|врач\w*|госпитал\w*|тайн\w*|истори\w*|советник\w*|техник\w*|развед\w*)\b", re.I),
    "firefighter": re.compile(r"^\s+(?:част\w*|машин\w*|станци\w*|служб\w*|охран\w*|безопасност\w*)\b", re.I),
}
_ROLE_UNION = "(?:" + "|".join(form for forms in FORMS.values() for form in forms) + ")"
_COORDINATION = re.compile(r"^(?:\s*(?:,|и)\s*" + _ROLE_UNION + r"){1,3}", re.I)
_APPOSITION = re.compile(r"((?-i:" + NAME + r")|\b(?:он|она|они)\b|главн\w*\s+геро\w*(?:\s+(?-i:" + NAME + r"))?)\s*(?:[,—–-]|является|стал|стала)\s*(?:" + _MODIFIERS + _ROLE_UNION + r"\s+и\s+){0,2}" + _MODIFIERS + "$", re.I)
_EMPLOYMENT_LINK = re.compile(r"(?:(?:в\s+качестве|на\s+должности|на\s+работу|работать|там|здесь|сейчас|тогда|(?:одним|одной|одними)\s+из|в\s+(?:полиции|больнице|клинике|школе|ресторане|кафе|баре|оркестре|фирме|газете|журнале|театре|музее|лаборатории))\s+)?(?:" + _MODIFIERS + _ROLE_UNION + r"\s+и\s+){0,1}" + _MODIFIERS + "$", re.I)
METRICS = ("current", "past", "future", "aspirational", "training", "central", "character_mention", "tag_mention", "title_mention", "uncertain", "conflict", "unknown", "relation_support", "source_support")


def _normalize_with_offsets(raw):
    chars, offsets = [], []
    for i, char in enumerate(raw):
        folded = char.casefold().replace("ё", "е")
        chars.extend(folded); offsets.extend([i] * len(folded))
    return "".join(chars), offsets


def _quoted(raw, start, end):
    return any(m.start() <= start and end <= m.end() for m in re.finditer(r'«[^»]*»|“[^”]*”|"[^"\n]*"', raw))


def _subjects(text):
    return [m for m in _SUBJECT.finditer(text)
        if m.group().casefold() not in _NAME_STOP
        and not _quoted(text, m.start(), m.end())
        and not (_NAME.fullmatch(m.group()) and re.search(r"\b(?:в|во|на|из|до|под|над|около)\s+$", text[:m.start()], re.I))]


def _last_search(pattern, text):
    found, position = None, 0
    while position <= len(text):
        match = pattern.search(text, position)
        if not match:
            return found
        found, position = match, match.start() + 1
    return found


def _fact(iid, family, field, raw, start, end, **extra):
    return {"item_id": int(iid), "flag": FLAGS[family], "source_field": field,
        "start": start, "end": end, "literal_span": raw[start:end],
        "field_sha256": base.hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "rules_version": CONTRACT, "source_known_at": None, "status": "uncertain",
        "temporal_role": "unknown", "centrality": "unknown", "work_relation": "unknown",
        "rule_id": "unresolved-role-mention", "review_status": "deterministic_rule_not_human_verified", **extra}


def _relation(family, raw_sentence, start, end):
    """Return status and explicit relation spans relative to the sentence."""
    sentence = raw_sentence.replace("ё", "е").replace("Ё", "Е")
    before, after = sentence[:start], sentence[end:]
    recent = before[-100:]
    # Casefold is used only for lexical classification; original offsets survive.
    result = {"status": "uncertain", "reason": "No supported explicit character/occupation relation", "rule_id": "unresolved-role-mention", "temporal_role": "unknown", "centrality": "unknown", "work_relation": "unknown", "character_start": None, "character_end": None, "relation_start": None, "relation_end": None}
    def finish(status, rule, reason, char_start=None, char_end=None, temporal="unknown", central=False, rel_start=None, rel_end=None):
        result.update(status=status, rule_id=rule, reason=reason, character_start=char_start, character_end=char_end,
            temporal_role=temporal, centrality="central_explicit" if central else "character_mention" if status == "accepted" else "unknown",
            work_relation="current" if status == "accepted" else "other" if rule == "foreign-work" else "unknown",
            relation_start=rel_start, relation_end=rel_end)
        return result
    if _quoted(sentence, start, end):
        return finish("rejected", "quoted-title-or-speech", "Quoted profession/title does not bind a character")
    if family in _ATTRIBUTIVE and _ATTRIBUTIVE[family].search(after):
        return finish("rejected", "attributive-non-role", "Adjective denotes an institution, production or another occupation")
    education = re.search(r"\b(?:студент|ученик)\w*-$", before, re.I)
    if family == "medical_practitioner" and (re.match(r"-ветеринар", after, re.I) or re.search(r"\bветеринарн\w*\s+$", before, re.I)):
        return finish("rejected", "veterinary-not-human-medicine", "Veterinary compound does not entail the human medical family")
    if re.search(r"\b(?:ассистент\w*|помощник\w*|ученик\w*|студент\w*|пациент\w*|клиент\w*)\s+(?:[а-яё-]+\s+){0,2}$", before, re.I):
        return finish("uncertain", "embedded-dependent-occupation", "Occupation is an embedded employer/teacher/client role, not the subject's job")
    if re.search(r"\b(?:хотя|но)\b[^.!?]{0,80}\b(?:это\s+не\s+так|неправда|ошибал\w*)\b", after, re.I):
        return finish("uncertain", "narrator-retracted-role", "Following narrative explicitly disputes the role claim")
    if family == "visual_artist" and re.search(r"\b(?:размышлен\w*|смысл\w*\s+человеческ\w*|кинематограф\w*)\b", sentence, re.I):
        return finish("uncertain", "abstract-artist-concept", "Abstract artist concept does not identify a visual-artist character")
    if family == "teacher" and re.search(r"\b(?:своим|его|ее|их)\s+бывш\w*\s*$", before, re.I):
        return finish("uncertain", "past-teacher-of-relation", "Past teacher-of relationship does not prove former teaching career")
    if re.search(r"\b(?:роман(?:а|у|ом|е|ы|ов)?|книг(?:а|и|е|у|ой|ам|ами|ах)?|сказк(?:а|и|е|у|ой|ам|ами|ах)|повест(?:ь|и|ью|ей)|сценари(?:й|я|ю|ем|и|ев))\b[^;.!?]{0,120}$", before, re.I):
        return finish("rejected", "source-author-or-nested-work", "Role belongs to a source author or nested book/screenplay reference")
    if family in ("writer", "visual_artist") and re.search(r"\b(?:придуман|создан|нарисован|написан)(?:а|о|ы)?\b[^.!?]{0,55}$", before, re.I):
        return finish("rejected", "source-character-creator", "Passive creation credit does not establish a narrative character profession")
    if _FOREIGN.search(sentence):
        return finish("rejected", "foreign-work", "Explicit other-work/filmography scope")
    if _HYPOTHETICAL.search(recent):
        return finish("uncertain", "hypothetical-or-disguise", "Hypothetical, reported or pretended role")
    # Negation binds the actual supported predicate, including plural forms,
    # embedded employee names and prepositions. "Not only works" is affirmative.
    negative = _NEG.search(recent)
    denied_predicate = None
    predicate_candidates = [_last_search(pattern, before) for pattern in (_EMPLOYMENT, _FUTURE, _ASPIRATION, _TRAINING)]
    nearest = max((m for m in predicate_candidates if m), key=lambda m:m.start(), default=None)
    if nearest and re.search(r"\bне\s*$", before[:nearest.start()], re.I):
        denied_predicate = nearest
    if negative or denied_predicate:
        subject = _subjects(before)
        entity = subject[-1] if subject else None
        denied = negative.group() if negative else denied_predicate.group()
        denied_time = "past" if _PAST.search(denied) else "aspirational" if re.search(r"хочет|хотят|мечтает|собирается", denied, re.I) else "future" if re.search(r"станет|будет", denied, re.I) else "training" if re.search(r"учится|обучается", denied, re.I) else "current"
        return finish("negated", "explicit-role-negation", "Locally negated occupation; never catalogue-wide absence", entity.start() if entity else None, entity.end() if entity else None, denied_time)
    if education:
        if re.search(r"\bне\s*$", before[:education.start()], re.I):
            return finish("negated", "negated-student-profession", "Student-profession relation is explicitly denied", temporal="training")
        trainee = re.match(r"\s+(" + NAME + r")", after)
        return finish("accepted", "student-professional-training", "Student-profession compound identifies education, not current employment", end + trainee.start(1) if trainee else start, end + trainee.end(1) if trainee else end, "training", False, education.start(), end + trainee.end(1) if trainee else end)
    # Screen/crew biographies are not characters. An explicit cast-to-character
    # relation is admitted only if both actor and the portrayed character exist.
    cast = re.search(r"(?:играет|сыграл|сыграла|в\s+роли)\s+(?:[а-яё-]+\s+){0,2}$", before, re.I)
    following_name = re.match(r"(?:-[а-яё]+)?\s*(?:(?:скорой\s+помощи\s*,?\s*|своей\s+эпохи\s*[-—–]\s*|грузовичок\s+)|,\s*(?:сержант|капитан|лейтенант)(?:а|у|ом|е)?\s+|(?:по\s+имени\s+)?)\s*(" + NAME + r")", after)
    if cast and following_name:
        actor_prefix = before[:cast.start()].rstrip()
        actor_candidates = _subjects(actor_prefix)
        actor = actor_candidates[-1] if actor_candidates else None
        actor = actor if actor and actor.end() == len(actor_prefix) and _NAME.fullmatch(actor.group()) and actor.group().casefold() not in ("он", "она", "они", "его", "ее") else None
        if not actor:
            return finish("uncertain", "unresolved-cast-direction", "Named professional after played may be the performer; no independently bound actor/character pair")
        cast_time = "past" if re.search(r"\bбывш\w*\s+(?:[а-яё-]+\s+){0,3}$", before, re.I) else "future" if re.search(r"\bбудущ\w*\s+(?:[а-яё-]+\s+){0,3}$", before, re.I) else "current"
        return finish("accepted", "cast-portrayed-named-character", "Explicit portrayed character with its own name", end + following_name.start(1), end + following_name.end(1), cast_time, rel_start=cast.start(), rel_end=end)
    # Documentary biography of a professional is an allowed real subject;
    # generic interviews/commentators/crew are not inferred as subjects.
    subject_story = re.search(r"\b(?:истори\w*|судьб\w*|приключени\w*|жизн\w*|будн\w*|биографи\w*)\s+(?:[а-яё-]+\s+){0,3}$", before, re.I)
    doc_subject = bool(subject_story and (not _PRODUCTION.search(sentence) or re.search(r"\b(?:биографи\w*|документальн\w*)\b", sentence, re.I)))
    if _PRODUCTION.search(sentence) and not doc_subject:
        return finish("rejected", "production-or-crew", "Production, source author or cast biography without resolved character relation")
    if re.search(r"\b(?:преми\w*|гильди\w*|номинаци\w*)\b", before, re.I):
        return finish("rejected", "award-or-professional-association", "Award/professional association does not name a character")
    if _PROMO.search(sentence) and not (subject_story or following_name):
        return finish("rejected", "promotion-or-award", "Promotional/award wording does not establish character employment")
    subjects = _subjects(before)
    prior = subjects[-1] if subjects else None
    employment, future, aspiration, training = predicate_candidates
    other_starts = [m.start() for m in (future, aspiration, training) if m]
    if employment and employment.start() >= max(other_starts, default=-1):
        link = employment.group("link")
        police_service = family == "police" and sentence[start:end].casefold() == "полиции" and link.strip().casefold() == "в"
        if not police_service and not _EMPLOYMENT_LINK.fullmatch(link):
            named_collaborator = following_name and re.fullmatch(r"(?:с|со)\s*", link, re.I) and re.search(r"(?:ом|ем|ой|ым|им)$", sentence[start:end], re.I)
            if named_collaborator:
                # A named collaborator's occupation may be explicit, but it is
                # not the employment of the pronominal worker before the verb.
                employment = future = aspiration = training = None
            else:
                return finish("uncertain", "employment-complement-not-occupation", "Works with/for/near a professional or other unsupported complement; does not establish works-as")
    trailing_trainee = re.match(r"\s*,\s*(" + NAME + r")", after) if training else None
    if training and trailing_trainee:
        return finish("accepted", "named-training-subject", "Explicit school/training relation names its trainee", end + trailing_trainee.start(1), end + trailing_trainee.end(1), "training", False, training.start(), end + trailing_trainee.end(1))
    dream = re.search(r"\b(?:в\s+мечтах|мечтает|мечтая)\b[^.!?]{0,55}$", before, re.I)
    if dream and prior and not any((employment, future, aspiration, training)):
        return finish("accepted", "explicit-imagined-occupation", "Named/pronominal character explicitly imagines the occupation", prior.start(), prior.end(), "aspirational", False, dream.start(), end)
    nominal_aliases = base._aliases(base.PROFESSIONS[family]) if family in base.PROFESSIONS else ()
    if not prior and employment and employment.start() >= max(other_starts, default=-1) and base.normal(sentence[start:end]) in nominal_aliases and re.match(r"(?:работает|работают|работали|служит|служат|служили)\b", employment.group(), re.I):
        return finish("accepted", "inverted-professional-subject", "Approved nominal professional noun is the explicit subject after employment verb", start, end, "past" if _PAST.search(employment.group()) else "current", False, employment.start(), end)
    if prior and (employment or future or aspiration or training):
        candidates = [(match, kind) for match, kind in ((aspiration, "aspirational"), (training, "training"), (future, "future"), (employment, "current")) if match]
        match, temporal = max(candidates, key=lambda entry: entry[0].start())
        if temporal == "current" and _PAST.search(match.group()):
            temporal = "past"
        finish("accepted", "explicit-" + temporal + "-occupation", "Named/pronominal character and explicit occupation predicate", prior.start(), prior.end(), temporal, bool(re.search(r"главн\w*\s+геро\w*", before, re.I)), match.start(), end)
        if temporal == "aspirational" and re.search(r"\b(?:поступает\s+в\s+университет|учится|обучается)\b", before, re.I):
            result["secondary_temporal_role"] = "training"
        return result
    nominal_past = bool(re.search(r"\b(?:бывш\w+|экс)\s+(?:[а-яё-]+\s+){0,3}$", before, re.I) or re.match(r"\s*,?\s*отслуживш\w*", after, re.I) or re.search(r"\bпосле\s+дембеля\b", before, re.I))
    temporal = "past" if nominal_past else "future" if re.search(r"\bбудущ\w*\s+(?:[а-яё-]+\s+){0,3}$", before, re.I) else "current"
    if following_name:
        # Profession + proper name is an apposition. Words such as Doctor No
        # never match the approved medical lexicon in the first place.
        return finish("accepted", "occupation-named-character", "Occupation apposition to a proper character name", end + following_name.start(1), end + following_name.end(1), temporal, bool(subject_story or re.search(r"главн\w*\s+геро\w*", before, re.I)), start, end + following_name.end(1))
    apposition = _APPOSITION.search(before)
    if family == "veterinarian" and not apposition:
        apposition = re.search(r"(" + NAME + r")\s+врач-$", before)
    if apposition:
        return finish("accepted", "named-character-occupation", "Named character/protagonist has explicit occupation apposition", apposition.start(1), apposition.end(1), temporal, "геро" in apposition.group(1).casefold(), apposition.start(), end)
    if re.search(r"\b(?:влюбляется|влюблен|влюблена)\b[^.!?]{0,65}\bв\s*$", before, re.I):
        return finish("accepted", "professional-character-love-object", "Professional person is explicitly a participant in the narrative relationship", start, end, temporal, False, 0, end)
    if family == "investigator" and re.search(r"\bоба\s*$", before, re.I) and re.match(r"\s*[-—–]\s*(?:[а-яё]+\s+){0,2}профессионалы\b", after, re.I):
        return finish("accepted", "explicit-two-professional-detectives", "Two detectives explicitly identified as professional people", start, end, temporal, False, start - len("оба "), len(sentence))
    if nominal_past and re.match(r"\s*,\s*отслуживш\w*[^.!?]{0,75}\bвозвращается\b", after, re.I):
        return finish("accepted", "completed-service-subject", "Explicit completed service belongs to the returning soldier", start, end, "past", False, start, len(sentence))
    if re.search(r"\bв\s+их\b[^.!?]{0,45}\b(?:компании|отряде|группе|команде)\b[^.!?]{0,45}\b(?:присутствуют|есть|состоят|входят)\b[^.!?]{0,40}$", before, re.I):
        return finish("accepted", "explicit-story-group-member", "Explicit profession is a member of the narrative group", start, end, temporal, False, 0, end)
    if re.match(r"\s+смогут\s+дать\s+\w*\s*отпор\b", after, re.I):
        return finish("accepted", "professional-group-narrative-action", "Professional group performs an explicit narrative action", start, end, temporal, False, start, len(sentence))
    if family == "investigator" and re.fullmatch(r"детектив\w*", sentence[start:end], re.I):
        # The bare noun is also a film genre. Only an unambiguous investigation
        # action can replace the missing name/qualified detective apposition.
        if not re.match(r"\s+(?:расследует|расследуют|разыскивает|разыскивают|разгадывает|разгадывают)\b", after, re.I):
            return finish("uncertain", "bare-detective-role-or-genre", "Unqualified detective noun needs name or explicit investigation action")
    if subject_story:
        return finish("accepted", "story-professional-subject", "Occupation is an explicit story/biography subject", start, end, temporal, True, subject_story.start(), end)
    action = _ACTION.search(after)
    coordinated = _COORDINATION.search(after)
    if not action and coordinated:
        subsequent = _ACTION.search(after[coordinated.end():])
        if subsequent:
            return finish("accepted", "coordinated-professional-actions", "Professional group is explicitly the narrative acting subject", start, end, temporal, False, start, end + coordinated.end() + subsequent.end())
    if action and not re.search(r"\b(?:роль|роли|образ|изображение|памятник|портрет|картина|книг\w*)\s+(?:[а-яё-]+\s+){0,2}$", before, re.I):
        return finish("accepted", "professional-character-action", "Professional noun is the acting narrative character", start, end, temporal, False, start, end + action.end())
    return result


def extract_item(iid, row):
    facts = []
    raw = row.get("description", "")
    for sentence_match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", raw):
        sentence = sentence_match.group()
        normalized, offsets = _normalize_with_offsets(sentence)
        for family, pattern in PATTERNS.items():
            for match in pattern.finditer(normalized):
                lo, hi = offsets[match.start()], offsets[match.end() - 1] + 1
                result = _relation(family, sentence, lo, hi)
                for kind in ("character", "relation"):
                    a, b = result.get(kind + "_start"), result.get(kind + "_end")
                    if a is not None:
                        result[kind + "_span"] = sentence[a:b]
                        result[kind + "_start"] = a + sentence_match.start()
                        result[kind + "_end"] = b + sentence_match.start()
                facts.append(_fact(iid, family, "description", raw, sentence_match.start() + lo, sentence_match.start() + hi,
                    sentence_start=sentence_match.start(), sentence_end=sentence_match.end(), sentence_literal=sentence, **result))
        # An institution noun is insufficient on its own. Explicit service in
        # police is a separately closed grammatical alias, not a police setting.
        for match in re.finditer(r"\bслуж(?:ит|ат|ил|ила|или)\s+в\s+(полиции)\b", normalized):
            lo, hi = offsets[match.start(1)], offsets[match.end(1) - 1] + 1
            result = _relation("police", sentence, lo, hi)
            for kind in ("character", "relation"):
                a, b = result.get(kind + "_start"), result.get(kind + "_end")
                if a is not None:
                    result[kind + "_span"] = sentence[a:b]
                    result[kind + "_start"] = a + sentence_match.start()
                    result[kind + "_end"] = b + sentence_match.start()
            facts.append(_fact(iid, "police", "description", raw, sentence_match.start() + lo, sentence_match.start() + hi,
                sentence_start=sentence_match.start(), sentence_end=sentence_match.end(), sentence_literal=sentence, **result))
    for family, pattern in PATTERNS.items():
        for field in ("title", "title_orig"):
            source = row.get(field, "")
            normalized, offsets = _normalize_with_offsets(source)
            for match in pattern.finditer(normalized):
                facts.append(_fact(iid, family, field, source, offsets[match.start()], offsets[match.end() - 1] + 1,
                    status="mention", rule_id="title-role-mention", reason="Title is lexical evidence only"))
    flag_to_family = {value: key for key, value in FLAGS.items()}
    for fact in base.keyword_evidence(iid, row):
        if fact["flag"] in flag_to_family:
            fact = dict(fact)
            fact.update(status="mention" if fact["decision"] == "accepted" else "rejected" if fact["decision"] == "rejected" else "uncertain",
                temporal_role="past" if re.search(r"\bбывш", fact["literal_span"], re.I) else "unknown",
                reason="Approved exact keyword alias; character relation not inferred", rules_version=CONTRACT,
                rule_id="keyword-evidence-only:" + fact["rule_id"], review_status="unreviewed_lexical_evidence")
            facts.append(fact)
    for fact in facts:
        fact["evidence_id"] = base.canonical_hash({k: fact[k] for k in ("item_id", "flag", "source_field", "start", "end", "field_sha256", "rule_id")})[:24]
    return facts


def build_item_features(items, metadata, temporal_mode="retrospective_current", metadata_cutoff=None):
    metadata, mapping = base._metadata(metadata)
    if temporal_mode not in ("retrospective_current", "historical_as_of"):
        raise ValueError("Unknown temporal mode")
    if temporal_mode == "historical_as_of" and metadata_cutoff is None:
        raise ValueError("Historical mode requires a cutoff; source availability remains unknown")
    columns = [flag + ":" + metric for flag in FLAGS.values() for metric in METRICS]
    columns += ["profession_relations:description_present", "profession_relations:source_missing"]
    index = {name: i for i, name in enumerate(columns)}
    matrix = np.zeros((metadata["vocab_size"], len(columns)), dtype=np.float32)
    evidence, role_counts, rules, status_counts = [], defaultdict(Counter), Counter(), Counter()
    for iid, token in sorted(mapping.items(), key=lambda kv: kv[1]):
        row = items.get(iid, {})
        facts = extract_item(iid, row)
        if temporal_mode == "historical_as_of":
            for fact in facts:
                if fact["status"] in ("accepted", "mention"):
                    fact.update(status="uncertain", reason="source_known_at unknown at requested historical cutoff")
        evidence.extend(facts)
        grouped = defaultdict(list)
        for fact in facts:
            grouped[fact["flag"]].append(fact)
            rules[fact["rule_id"]] += 1; status_counts[fact["status"]] += 1
        for flag in FLAGS.values():
            observations = grouped[flag]
            accepted = [f for f in observations if f["status"] == "accepted"]
            # Only same explicit character string+role can conflict. A different
            # character's denial cannot negate another character's occupation.
            positives = {base.normal(f.get("character_span", "")) for f in accepted if f["temporal_role"] == "current" and f.get("character_span")}
            negatives = {base.normal(f.get("character_span", "")) for f in observations if f["status"] == "negated" and f["temporal_role"] == "current" and f.get("character_span")}
            conflict_entities = positives & negatives
            usable = [f for f in accepted if base.normal(f.get("character_span", "")) not in conflict_entities]
            vals = {metric: 0.0 for metric in METRICS}
            for temporal in ("current", "past", "future", "aspirational", "training"):
                vals[temporal] = float(any(f["temporal_role"] == temporal or f.get("secondary_temporal_role") == temporal for f in usable))
            vals["central"] = float(any(f["centrality"] == "central_explicit" for f in usable))
            vals["character_mention"] = float(bool(usable))
            vals["tag_mention"] = float(any(f["source_field"] == "keywords" and f["status"] == "mention" for f in observations))
            vals["title_mention"] = float(any(f["source_field"] in ("title", "title_orig") and f["status"] == "mention" for f in observations))
            vals["uncertain"] = float(any(f["status"] == "uncertain" for f in observations))
            vals["conflict"] = float(bool(conflict_entities))
            vals["unknown"] = float(not usable)
            vals["relation_support"] = float(len({(f.get("character_span"), f["temporal_role"], f["sentence_start"]) for f in usable}))
            vals["source_support"] = float(len({f["source_field"] for f in observations if f["status"] in ("accepted", "mention")}))
            for metric, value in vals.items():
                matrix[token, index[flag + ":" + metric]] = value
                if value and metric not in ("source_support", "relation_support"):
                    role_counts[flag][metric] += 1
            if conflict_entities:
                for fact in observations:
                    if base.normal(fact.get("character_span", "")) in conflict_entities:
                        fact["conflict"] = True
        matrix[token, index["profession_relations:description_present"]] = bool(row.get("description", "").strip())
        matrix[token, index["profession_relations:source_missing"]] = not any(row.get(f, "").strip() for f in ("description", "keywords", "title", "title_orig"))
    groups = {"profession_character_relations": {"scope": "item", "start": 0, "stop": len(columns), "columns": columns, "flags": list(FLAGS.values())}}
    schema = {"columns": columns, "groups": groups, "forms": FORMS, "metrics": METRICS}
    coverage = {"train_known_items": len(mapping), "raw_items": len(items), "evidence_records": len(evidence),
        "role_item_support": {flag: dict(role_counts[flag]) for flag in FLAGS.values()},
        "rule_evidence_support": dict(sorted(rules.items())), "evidence_status_counts": dict(status_counts),
        "accepted_relation_items": int(np.any(matrix[:, [index[flag + ":character_mention"] for flag in FLAGS.values()]], axis=1).sum()),
        "all_unknown_families": [flag for flag in FLAGS.values() if not role_counts[flag]["character_mention"]],
        "precision_recall": "Independent stratified literal audit; population precision/recall not estimated",
        "supported_scope": ["22 approved profession families and optional intelligence role", "closed Russian noun inflections", "name apposition, explicit employment, acting professional subject, story/biography subject", "named cast-to-character relation", "past/current/future/aspirational/training separated", "exact approved keyword evidence and title mentions kept separate"],
        "out_of_scope": ["unrestricted coreference or unrestricted NLP recall", "professions absent from the approved ontology; rare professions without an explicit mapping basis", "doctor/actor/model/clergy/astronaut as new categories", "generic agent and genre detective", "unconfirmed historical source availability", "population precision or recommendation improvement claims"]}
    provenance = {"contract": CONTRACT, "temporal_mode": temporal_mode, "metadata_cutoff": str(metadata_cutoff) if metadata_cutoff else None,
        "source_known_at": None, "retrieved_at": datetime.now(timezone.utc).isoformat(), "mapping_sha256": base.canonical_hash(mapping),
        "schema_sha256": base.canonical_hash(schema), "ontology_sha256": base.canonical_hash(FORMS),
        "fit": {"learned_transforms": False, "interactions_read": False, "scope": "train-known catalogue only"},
        "special_item_rows": "all zero", "review_status": "deterministic evidence, not externally verified truth", "evidence_sha256": base.canonical_hash(evidence)}
    return {"matrix": matrix, "item_features": matrix, "columns": columns, "item_columns": columns,
        "groups": groups, "coverage": coverage, "provenance": provenance, "evidence": evidence}


def build(raw, metadata, **kwargs):
    """Return matrix/columns/coverage/provenance/evidence from frozen local files."""
    path = Path(raw) / "items.csv" if Path(raw).is_dir() else Path(raw)
    paths = {"items": path, "builder": Path(__file__), "keyword_base": Path(base.__file__)}
    if isinstance(metadata, (str, Path)):
        paths["metadata"] = Path(metadata)
    before = {key: base.file_hash(value) for key, value in paths.items()}
    result = build_item_features(base.read_items(path), metadata, **kwargs)
    if any(base.file_hash(paths[key]) != value for key, value in before.items()):
        raise ValueError("Profession input changed during build")
    result["provenance"]["inputs"] = {key: {"path": value.resolve().as_posix(), "sha256": before[key]} for key, value in paths.items()}
    return result
