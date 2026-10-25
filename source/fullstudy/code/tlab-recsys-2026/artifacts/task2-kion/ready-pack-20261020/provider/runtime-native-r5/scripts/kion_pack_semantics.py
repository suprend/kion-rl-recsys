"""Conservative CPU-only lexical semantics for the frozen KION item catalogue.

No zero below is a verified semantic negative. Every flag has positive, unknown
and conflict coordinates. Exact keyword evidence and narrow, grammatical
current-work description rules are kept distinct. This is a retrospective
metadata experiment, not full NLP or historically available as-of annotation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np


CONTRACT = "kion-local-pack-semantics-v1"
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}
ITEM_FIELDS = ("title", "title_orig", "keywords", "description", "release_year", "countries", "content_type")
STATES = ("matched_accepted", "source_missing", "present_no_evidence", "matched_rejected", "needs_review", "conflict")


def normal(value):
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aliases(text):
    return tuple(normal(part) for part in text.split(";") if part.strip())


# E dictionaries are deliberately literal. F rules are recorded separately;
# profession aliases must never acquire automatically generated F rules.
THEMES = {
    "theme_overcoming_difficulties": "преодоление трудностей",
    "theme_priority_change": "смена жизненных приоритетов",
    "theme_strong_women_characters": "сильная женщина;сильная героиня;сильная женщина-лидер;сильный женский персонаж;сильные женщины",
    "theme_love_triangle": "любовный треугольник;романтический треугольник;любовные треугольники",
    "theme_family_conflict": "семейный конфликт",
    "theme_loss_close_person": "потеря близкого человека;потеря любимого человека;потеря близких",
    "theme_female_friendship": "женская дружба", "theme_male_friendship": "мужская дружба",
    "theme_teen_romance": "подростковая любовь;подростковая влюбленность;романтические отношения подростков",
    "theme_romantic_relationship": "романтические отношения", "theme_good_deeds": "добрые дела",
    "theme_marital_infidelity": "супружеская измена;прелюбодеяние",
}
THEME_F = dict(zip(list(THEMES)[:-1], [
    ("преодоление", "трудностей"), ("смена", "жизненных", "приоритетов"),
    ("сильные", "женщины"), ("любовные", "треугольники"), ("семейный", "конфликт"),
    ("потеря", "близких"), ("женская", "дружба"), ("мужская", "дружба"),
    ("подростковая", "любовь"), ("романтические", "отношения"), ("добрые", "дела")]))
NARRATIVE = {
    "has_flashback": "флешбек;флешбэк;флэшбэк;повествование флешбэками;аудио-флешбэк;недостоверный флешбэк;рассказанный во флэшбэке;фильм начинается с флешбека;флешбэк во флешбэке;флешбэк во флешбэке флешбэка",
    "has_voiceover": "закадровый голос;голос за кадром;закадровый комментарий в эпилоге;вступительное закадровое повествование;закадровый текст;закадровый текст от титульного героя;закадровый текст из могилы;закадровое чтение;закадровое повествование;окончание фильма с закадровым голосом",
    "has_time_travel": "путешествие во времени;путешествия во времени;комедия о путешествии во времени;путешественник во времени;романтика путешествий по времени",
    "has_time_loop": "временная петля;петля времени", "has_surrealism": "сюрреализм;сюрреалистичное",
    "has_absurdism": "абсурдизм;абсурдный юмор;комедия абсурда;абсурд",
}
SETTINGS = {
    "school": "школа;старшая школа;средняя школа;начальная школа;школа-интернат;подготовительная школа;школьная жизнь;неполная средняя школа;частная школа;католическая школа;заброшенная школа;новая школа;вечерняя школа;международная школа;школа для мальчиков;школьный класс",
    "university": "университет;колледж;колумбийский университет;гарвардский университет;оксфордский университет;нью-йоркский университет;кембриджский университет;йельский университет;калифорнийский университет беркли;северо-западный университет;принстонский университет;калифорнийский университет;мичиганский университет;массачусетский технологический институт;университетский городок;кампус;жизнь в колледже;обучение в колледже;вечеринка в колледже",
    "village": "деревня;сельская местность;английская сельская местность;сельская обстановка;сельская жизнь;английская деревня;отдаленная деревня;французская деревня;небольшая деревня;сельская местность соединенных штатов;сельский район;сельская ирландия;горная деревня;деревенская глубинка",
    "province": "провинция;провинциальная жизнь;провинциальные декорации;австралийская глубинка;деревенская глубинка;глубинка;русская глубинка",
    "metropolis": "большой город", "forest_jungle": "лес;джунгли;тропический лес",
    "island": "остров;необитаемый остров;тропический остров", "coast": "побережье;морское побережье;пляж;курорт",
    "mountain": "горы;горная местность;горный район", "desert": "пустыня",
    "polar": "арктика;антарктида;северный полюс;южный полюс",
    "hospital": "больница;госпиталь;клиника;психиатрическая больница",
    "prison": "тюрьма;женская тюрьма;исправительная колония", "hotel": "отель;гостиница;мотель",
    "office": "офис;офисная жизнь", "military_base": "военная база;военный лагерь",
    "summer_camp": "летний лагерь;детский лагерь;пионерский лагерь",
    "marine_watercraft": "корабль;морское путешествие;подводная лодка;яхта;круизный лайнер",
    "roadtrip": "дорожное путешествие;роуд-муви;путешествие на автомобиле",
    "space": "космос;открытый космос;ловушка в космосе;выход в открытый космос;вымышленная планета;галактика;планета марс;чужая планета;замерзшая планета;заснеженная планета;инопланетная планета;в ловушке в космосе;пустынная планета;планета-тюрьма;запертый в космосе",
    "christmas": "рождество;рождества;рождеством;рождественская елка;рождественская вечеринка;рождественский подарок;рождественский фильм;рождественские украшения;спасение рождества;рождество в опасности;рождественский обед;рождественские огни;канун рождества",
    "newyear": "новый год;нового года;с новым годом!;канун нового года;в канун нового года;новогодняя вечеринка;отсчет времени перед наступлением нового года;новогодний;новогодняя;новогоднее;новогодние;новогодних;новогоднем",
}
PROFESSIONS = {
    "medical_practitioner": "врач;врачи;женщина-врач;женщина врач;медик;женщина-медик;хирург;пластический хирург;нейрохирург;женщина-хирург;акушер;акушерка;фельдшер;парамедик;стоматолог;санитарный врач",
    "nurse": "медсестра;медсестры;медсестричка", "mental_health_specialist": "психологи;психолог;психиатр;психотерапевты;психотерапевт;детский психолог;судебный психолог;семейный психолог",
    "veterinarian": "ветеринар", "teacher": "учитель;учителя;учительница;учительницы;женщина-учитель;школьный учитель;школьный преподаватель;учитель музыки;учитель танцев;учитель английского языка;учитель гуманитарных наук",
    "scientist": "ученые;ученый;женщина-ученый;привлекательная женщина-ученый;злой ученый;сумасшедший ученый;безумный ученый;аутичный ученый;военный ученый;ученый-компьютерщик", "engineer": "инженер",
    "lawyer": "юрист;адвокат;адвокаты;женщина-адвокат;женщина-юрист;адвокат по бракоразводным процессам;адвокат по разводам",
    "police": "полицейский;женщина-полицейский;женщина полицейский;офицер полиции;женщина - офицер полиции;инспектор полиции;капитан полиции;лейтенант полиции;полицейский детектив;бывший полицейский;бывший офицер полиции;друг-полицейский;продажный полицейский;полицейский под прикрытием;грязный полицейский",
    "investigator": "частный детектив;полицейский детектив;женщина-детектив;следователь;частный следователь;детектив-инспектор",
    "military": "солдат;солдаты;бывший солдат;женщина-солдат;военный офицер;военные;военный;британский солдат;американский солдат;немецкий солдат;морской офицер;военно-морской офицер;бывший военный;военный по контракту;военнослужащие",
    "firefighter": "пожарный;пожарник;начальник пожарной охраны", "writer": "писатель;писатели;писатель-женщина;писатель-романист;русские писатели",
    "journalist": "журналист;журналисты;женщина-журналист;гонзо-журналист;бывший военный корреспондент",
    "musician_singer": "музыкант;музыканты;музыкантка;женщина-музыкант;рок-музыканты;уличный музыкант;певец;певица;оперный певец;джазовый певец;джазовая певица;французская певица;уличный певец;певец в ночном клубе;певица ночного клуба",
    "visual_artist": "художник;художники;художница", "cook": "повар;шеф-повар;кулинар",
    "professional_driver": "водитель;водитель грузовика;водитель автобуса;водитель лимузина;водитель такси;дальнобойщики",
    "pilot": "пилот;женщина-пилот;второй пилот;пилот самолета;пилот вертолета;пилот-истребитель;пилот реактивного истребителя",
    "athlete": "спортсмен;спортсмены;боксер;боксер-тяжеловес;футболист;кикбоксер",
    "bar_waiter": "бармен;барменша;женщина-официант;официант;официантка", "accountant": "бухгалтер;бухгалтер мафии",
}
SPORTS = {
    "football": "футбол;футболист;футболисты;футбольный матч", "american_football": "американский футбол",
    "basketball": "баскетбол;баскетболист;баскетболисты", "boxing": "бокс;боксер;боксер-тяжеловес;боксеры",
    "martial_arts": "мма;mma;смешанные единоборства;карате;каратэ;кун-фу;кунг-фу;дзюдо;кикбоксинг;тхэквондо;джиу-джитсу;боевые искусства",
    "running": "бег;бегун;бегуны;бегунья;марафонский бег", "swimming": "плавание;пловец;пловчиха;спортивное плавание",
    "cycling": "велоспорт;велогонка;велогонки", "winter_sports": "хоккей;фигурное катание;лыжный спорт;биатлон;сноуборд;конькобежный спорт;бобслей;керлинг",
    "tennis": "теннис;теннисист;теннисистка", "surfing": "серфинг;серфер", "chess": "шахматы;шахматист;гроссмейстер",
}
ACTIVITIES = {"fitness": "фитнес;фитнесс;аэробика;воркаут;бодибилдинг", "yoga": "йога", "pilates": "пилатес",
    "stretching": "растяжка;стретчинг", "dance_ballet": "танцы;танец;балет;бальные танцы;хип-хоп танцы",
    "cooking": "кулинария;приготовление пищи;кулинарный рецепт", "fishing": "рыбалка;рыбная ловля;рыболовство",
    "crafts": "рукоделие;вязание;вышивка;оригами;поделки;гончарное дело"}
STORY = {
    "literature_adaptation": "по мотивам романа или книги;основано на романе или книге;литературная адаптация;по мотивам романа;по роману или книге;по мотивам книги;по мотивам рассказа;по мотивам детской книги;по детской книге;по мотивам новеллы;по мотивам пьесы",
    "comic_adaptation": "по мотивам комикса;по мотивам комиксов;на основе комикса;на основе комиксов;основано на комиксе;основано на комиксах;основанный на комиксе;основанный на комиксах;по комиксу;по комиксам;по мотивам графического романа",
    "manga_adaptation": "по мотивам манги;на основе манги;основано на манге;экранизация манги;по манге",
    "based_on_real_events": "основано на реальных событиях;основанный на реальных событиях;основанная на реальных событиях;основанное на реальных событиях;основанные на реальных событиях;основано на реальной истории;на основе реальной истории;по мотивам реальной истории;вдохновлено реальными событиями",
}
FORMATS = ("miniseries", "standalone_short", "anthology", "anthology_film", "anthology_series", "compilation",
    "recorded_performance", "performance_concert", "performance_theatre", "performance_standup", "instructional_intent")
TIME_FLAGS = ("story_time_explicit_year", "story_time_explicit_interval", "story_time_explicit_century",
    "story_time_explicit_decade", "story_time_named_era", "story_time_relative_present", "story_time_relative_future", "story_time_multiple_plans")


def _rules():
    result = {}
    def add(group, name, aliases, sequences=()):
        result[name] = {"group": group, "E": _aliases(aliases), "F": tuple(sequences)}
    for name, aliases in THEMES.items():
        add("themes", name, aliases, [THEME_F[name]] if name in THEME_F else [])
    for name, aliases in NARRATIVE.items():
        add("narrative", name, aliases)
    for name, aliases in SETTINGS.items():
        sequences = {"school": [("средняя", "школа")], "space": [("открытый", "космос")],
            "christmas": [("канун", "рождества")], "newyear": [("новый", "год"), ("нового", "года")]}.get(name, [])
        add("setting_topics", "setting_" + name + "_tag_topic", aliases, sequences)
    for name, aliases in PROFESSIONS.items():
        add("professions", "profession_" + name, aliases)
    add("professions", "role_spy_intelligence", "шпион;шпионы;герой-шпион;шпион-герой;женщина-шпион;бывший шпион;шпион под прикрытием;секретный агент;тайный агент;агент разведки;агент цру;бывший агент цру;двойной агент")
    for group, terms in (("sport_mentions", SPORTS), ("activity_mentions", ACTIVITIES)):
        for name, aliases in terms.items():
            add(group, group.removesuffix("_mentions") + "_" + name + "_tag_mention", aliases)
    for name, aliases in STORY.items():
        add("story_source", "story_source_" + name, aliases)
    return result


RULES = _rules()
KEYWORD_INDEX = defaultdict(list)
FRAGMENT_INDEX = defaultdict(list)
NARRATIVE_FRAGMENT_INDEX = defaultdict(list)
for _flag, _rule in RULES.items():
    for _number, _literal in enumerate(_rule["E"], 1):
        KEYWORD_INDEX[_literal].append((_flag, f"{_flag}.E{_number}"))
    for _number, _sequence in enumerate(_rule["F"], 1):
        FRAGMENT_INDEX[_sequence[0]].append((_flag, f"{_flag}.F{_number}", _sequence))
    if _flag in NARRATIVE:
        for _number, _literal in enumerate(_rule["E"], 1):
            _words_in_literal = _literal.split()
            for _boundary in range(1, len(_words_in_literal)):
                NARRATIVE_FRAGMENT_INDEX[" ".join(_words_in_literal[:_boundary])].append((_flag, f"{_flag}.F{_number}", _literal))


def chunks(text):
    """Keep empty chunks, duplicate chunks, Unicode offsets and unaltered spans."""
    result, offset = [], 0
    for part in text.split(","):
        result.append({"text": part, "normal": normal(part), "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return result


def _words(text):
    return re.findall(r"[^\W_]+(?:-[^\W_]+)*", normal(text), re.UNICODE)


_STOP = set("в во на по с со за к ко из у и а о об от для как при под до не над без или".split())


def metadata_chunks(row, parts):
    """Quarantine only a title sequence bound to a matching release-year edge.

    Content occurrences elsewhere survive. Country aliases are only boundary
    parsing aids; they never become geography features.
    """
    norms = [part["normal"] for part in parts]
    try:
        year = str(int(float(row.get("release_year", ""))))
    except (ValueError, OverflowError):
        year = None
    country_words = set(_words(row.get("countries", "")))
    if "сша" in country_words:
        country_words.update(("соединенные", "штаты"))
    if {"республика", "корея"} <= country_words:
        country_words.add("южная")
    blocked, title_like, claimed_years = set(), set(), set()
    for field in ("title", "title_orig"):
        words = _words(row.get(field, ""))
        variants = {tuple(words), tuple(w for w in words if w not in _STOP), (normal(row.get(field, "")),)}
        for seq in sorted(variants, key=lambda sequence: (-len(sequence), sequence)):
            if not seq or not all(seq):
                continue
            for start in range(len(norms) - len(seq) + 1):
                stop = start + len(seq)
                if tuple(norms[start:stop]) != seq:
                    continue
                indices = set(range(start, stop))
                title_like.update(indices)
                # Prefix: title + matching year, optionally country words.
                left_ok = start == 0 and year is not None and stop < len(norms) and norms[stop] == year
                if left_ok:
                    claimed_years.add(stop)
                # Suffix: matching year + verified country tokens + title.
                right_ok = False
                if stop == len(norms) and year is not None:
                    for pos in range(max(0, start - 8), start):
                        between = norms[pos + 1:start]
                        if pos not in claimed_years and norms[pos] == year and between and all(_words(t) and set(_words(t)) <= country_words for t in between):
                            right_ok = True
                if left_ok or right_ok:
                    blocked.update(indices)
    # A separately repeated tag outside a verified metadata occurrence is an
    # independent content occurrence, not another copy of the metadata edge.
    return blocked, title_like if not blocked else title_like & blocked


def _fact(iid, flag, field, text, start, end, rule, **extra):
    return {"item_id": int(iid), "flag": flag, "source_field": field, "literal_span": text[start:end],
        "start": start, "end": end, "field_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "rule_id": rule, "rules_version": CONTRACT, "decision": "accepted", "work_relation": "current",
        "assertion_status": "positive", "source_known_at": None, **extra}


def keyword_evidence(iid, row):
    text = row.get("keywords", "")
    parts = chunks(text)
    blocked, title_like = metadata_chunks(row, parts)
    evidence = []
    for index, part in enumerate(parts):
        hits = [(flag, rule, index + 1) for flag, rule in KEYWORD_INDEX.get(part["normal"], ())]
        for flag, rule, seq in FRAGMENT_INDEX.get(part["normal"], ()):
            stop = index + len(seq)
            if tuple(p["normal"] for p in parts[index:stop]) == seq:
                hits.append((flag, rule, stop))
        for flag, rule, literal in NARRATIVE_FRAGMENT_INDEX.get(part["normal"], ()):
            joined = part["normal"]
            for stop in range(index + 1, len(parts)):
                if not parts[stop]["normal"]:
                    break
                joined += " " + parts[stop]["normal"]
                if joined == literal:
                    hits.append((flag, rule, stop + 1))
                    break
                if not literal.startswith(joined + " "):
                    break
        for flag, rule, stop in hits:
            span = set(range(index, stop))
            status = "verified_metadata_only" if span & blocked else ("unresolved" if span & title_like else "verified_content")
            decision = {"verified_metadata_only": "rejected", "unresolved": "needs_review", "verified_content": "accepted"}[status]
            evidence.append(_fact(iid, flag, "keywords", text, part["start"], parts[stop - 1]["end"], rule,
                chunk_indices=list(range(index, stop)), chunk_boundaries=[[p["start"], p["end"]] for p in parts[index:stop]],
                normalized_sequence=[p["normal"] for p in parts[index:stop]], metadata_prefix_status=status,
                metadata_reason="matching title bound to release-year catalogue edge" if status == "verified_metadata_only" else
                    ("title overlap without verified edge" if status == "unresolved" else "exact semantic whitelist outside title metadata candidates"),
                decision=decision, centrality="unknown", review_status="unreviewed_lexical_evidence"))
    pseudo = {"по мотивам псевдореальной истории"}
    for index, part in enumerate(parts):
        if part["normal"] in pseudo:
            evidence.append(_fact(iid, "story_source_based_on_real_events", "keywords", text, part["start"], part["end"],
                "real-events-pseudoreal-quarantine-v1", decision="rejected", assertion_status="uncertain",
                chunk_indices=[index], reason="pseudoreal source is not real-events evidence"))
    return evidence


@lru_cache(maxsize=2048)
def _literal_re(literals):
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(s).replace(r"\ ", r"\s+").replace("е", "[её]") for s in literals) + r")(?!\w)", re.I)


_RELATION = re.compile(r"\b(?:действие(?:\s+(?:фильма|картины|сериала|которого))?|события(?:\s+(?:фильма|картины|сериала))?)\s+(?:происходят|происходит|разворачиваются|разворачивается|начинается|начинаются)\b", re.I)
_UNSAFE = re.compile(r"\b(?:если|возможно|вероятно|может быть|мечтает|снилось|могло бы|не происходит|не разворачивается|другого фильма|первой части|вторая часть|третья часть|первая часть)\b", re.I)
_OTHER = re.compile(r"\b(?:ранее|прежде|известн[а-яё]*\s+ролью|снял|снимался|фильмограф|режиссер|актриса|актер)\b", re.I)
_EXPLICIT_OTHER_WORK = re.compile(r"\b(?:другой|другая|другое|другого|другом|другую|чужой|предыдущий|предыдущего|любимый|любимом)\s+(?:фильм[а-яё]*|сериал[а-яё]*|картин[а-яё]*)\b", re.I)
_NON_STORY_TIME_PREFIX = re.compile(r"\b(?:а\s+не|не\s+в|после\s+выхода|до\s+выхода|роман[а-яё]*|книг[а-яё]*|публикаци[а-яё]*)\b", re.I)
_LOCATIONS = {
    "school": ("в школе", "в старшей школе", "в средней школе"), "university": ("в университете", "в колледже", "в кампусе"),
    "village": ("в деревне", "в селе", "в сельской местности"), "province": ("в провинции", "в провинциальном городке", "в глубинке"),
    "metropolis": ("в большом городе",), "forest_jungle": ("в лесу", "в джунглях", "в тропическом лесу"),
    "island": ("на острове", "на необитаемом острове"), "coast": ("на побережье", "на морском побережье", "на пляже"),
    "mountain": ("в горах",), "desert": ("в пустыне",), "polar": ("в арктике", "в антарктиде", "на северном полюсе", "на южном полюсе"),
    "hospital": ("в больнице", "в госпитале", "в клинике", "в психиатрической больнице"), "prison": ("в тюрьме", "в исправительной колонии"),
    "hotel": ("в отеле", "в гостинице", "в мотеле"), "office": ("в офисе",), "military_base": ("на военной базе", "в военном лагере"),
    "summer_camp": ("в летнем лагере", "в детском лагере", "в пионерском лагере"),
    "marine_watercraft": ("на борту яхты", "на борту подводной лодки", "на круизном лайнере"),
    "roadtrip": ("во время дорожного путешествия", "во время путешествия на автомобиле"),
    "space": ("в космосе", "в открытом космосе", "на космической станции", "на луне", "на марсе"),
    "christmas": ("накануне рождества", "в канун рождества", "на рождество"), "newyear": ("в канун нового года", "накануне нового года", "в новый год"),
}
_PLACE_ALIASES = {
    "city:moscow": ("москве",), "city:new_york": ("нью-йорке", "нью — йорке"), "city:paris": ("париже",),
    "city:london": ("лондоне",), "city:saint_petersburg": ("петербурге", "санкт-петербурге"),
    "city:leningrad": ("ленинграде",), "country:japan": ("японии",), "country:indonesia": ("индонезии",),
    "country:russia": ("россии",), "country:ussr": ("ссср",), "country:france": ("франции",),
    "country:usa": ("сша", "соединенных штатах"), "country:uk": ("великобритании",), "country:england": ("англии",),
    "region:siberia": ("сибири",), "region:africa": ("африке",), "region:europe": ("европе",),
}
_FORMAT_LITERALS = {
    "miniseries": ("мини-сериал", "минисериал", "мини сериал", "сериал мини-формата"),
    "standalone_short": ("короткометражный фильм", "короткометражный мультфильм", "короткометражный документальный фильм", "короткометражка"),
    "anthology": ("киноальманах", "фильм-антология", "сериал-антология"),
    "compilation": ("сборник мультфильмов", "сборник короткометражных фильмов", "коллекция короткометражных фильмов", "подборка стендап-выступлений", "сборник визуальных уроков", "сборник веселых визуальных уроков"),
    "recorded_performance": ("запись концерта", "трансляция концерта", "фильм-спектакль", "запись спектакля", "запись стендап-выступления"),
}
_YEAR = re.compile(r"(?<!\d)([1-9]\d{2,3})(?:\s*[-–—]\s*([1-9]\d{2,3}))?\s*(?:год(?:у|а|ах|ы|ов)?|г\.)(?:\s*(до\s+(?:нашей эры|н\.\s*э\.)))?", re.I)
_CENTURY = re.compile(r"(?<!\w)([IVX]{1,8}|[1-9]\d?)\s*(?:век(?:а|е|ов)?|в\.)(?:\s*(до\s+(?:нашей эры|н\.\s*э\.)))?", re.I)
_DECADE = re.compile(r"(?<!\d)(\d{2}|\d{4})\s*[-–—]\s*(?:е|х|ые)\s*(?:годы|годах|годов)\b", re.I)


def _sentences(text):
    # Protect calendar abbreviation dots without changing string length/offsets.
    protected = list(text)
    for abbreviation in re.finditer(r"\bдо\s+н\.\s*э\.", text, re.I):
        for index in range(abbreviation.start(), abbreviation.end()):
            if protected[index] == ".":
                protected[index] = "\ue000"
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", "".join(protected)):
        if match.group().strip():
            yield match.start(), match.end(), text[match.start():match.end()]


def description_evidence(iid, row):
    text = row.get("description", "")
    facts = []
    title = normal(row.get("title", ""))
    for start, end, sentence in _sentences(text):
        normalized = normal(sentence)
        relation = _RELATION.search(sentence)
        safe = not _UNSAFE.search(sentence)
        if relation:
            clause = sentence[relation.end():]
            relation_safe = safe and not re.match(r"\s*(?:не|если|возможно)\b", clause, re.I)
            prefix = sentence[:relation.start()].strip()
            current_scope = not prefix or bool(title and ("«" + title + "»") in normal(prefix))
            previous = text[:start].strip()
            component_scope = bool(re.search(r"\b(?:первая|вторая|третья|четвертая)\s+часть[^.!?]*[.!?]?\s*$", previous, re.I))
            relation_safe = relation_safe and current_scope and not component_scope
            decision = "accepted" if relation_safe else "needs_review"
            for name, locations in _LOCATIONS.items():
                match = _literal_re(locations).match(clause.lstrip())
                if match:
                    facts.append(_fact(iid, "setting_" + name + "_description_explicit_setting", "description", text, start, end,
                        "current-work-action-location-v1", decision=decision, relation_scope="story_event_time" if name in ("christmas", "newyear") else "story_location", centrality="unknown"))
            for place, aliases in _PLACE_ALIASES.items():
                match = re.match(r"\s*(?:в|на)\s+" + _literal_re(aliases).pattern, clause, re.I)
                if match:
                    facts.append(_fact(iid, "geography_story_setting:" + place, "description", text, start, end,
                        "current-work-action-place-v1", decision=decision, place_id=place, place_type=place.split(":")[0],
                        role="story_setting", direct_or_inferred="direct", entity_resolution_status="literal_local_dictionary",
                        role_status="explicit_relation", centrality="unknown"))
            time_facts = []
            # Subordinate plot events, source publication and a negated later
            # assertion are not the explicit action-time relation being parsed.
            temporal_clause = re.split(r"\b(?:где|который|которая|которые|но|однако|хотя)\b", clause, maxsplit=1, flags=re.I)[0]
            for match in _YEAR.finditer(temporal_clause):
                era = "BCE" if match[3] else "CE"
                y1, y2 = int(match[1]), int(match[2]) if match[2] else None
                if _NON_STORY_TIME_PREFIX.search(temporal_clause[:match.start()]) or (y2 is not None and ((era == "CE" and y1 > y2) or (era == "BCE" and y1 < y2))):
                    status = "needs_review"
                else:
                    status = decision
                time_facts.append(_fact(iid, "story_time_explicit_interval" if y2 else "story_time_explicit_year", "description", text, start, end,
                    "explicit-story-calendar-year-v1", decision=status, temporal_role="story_setting", narrative_scope="unknown",
                    temporal_kind="year_interval" if y2 else "year", calendar_era=era, year_start=y1, year_end=y2, calendar_year_zero=False))
            for regex, flag, kind in ((_CENTURY, "story_time_explicit_century", "century"), (_DECADE, "story_time_explicit_decade", "decade")):
                for match in regex.finditer(temporal_clause):
                    time_facts.append(_fact(iid, flag, "description", text, start, end, "explicit-story-" + kind + "-v1",
                        decision="needs_review" if _NON_STORY_TIME_PREFIX.search(temporal_clause[:match.start()]) else decision, temporal_role="story_setting", narrative_scope="unknown", temporal_kind=kind,
                        temporal_literal=match.group(), absolute_anchor="unknown" if kind == "decade" and len(match[1]) == 2 else "explicit"))
            for literals, flag, kind in ((("в наши дни", "в настоящее время"), "story_time_relative_present", "relative_present"),
                (("в недалеком будущем", "в далеком будущем", "в ближайшем будущем", "в будущем"), "story_time_relative_future", "relative_future"),
                (("в средневековье", "в эпоху античности", "в доисторические времена"), "story_time_named_era", "named_era")):
                if _literal_re(literals).search(temporal_clause):
                    time_facts.append(_fact(iid, flag, "description", text, start, end, "explicit-story-" + kind + "-v1",
                        decision=decision, temporal_role="story_setting", narrative_scope="unknown", temporal_kind=kind, absolute_anchor="unknown"))
            facts.extend(time_facts)
            if len(time_facts) >= 2 and re.search(r"\b(?:а\s+затем|потом)\s+переносится\b", clause, re.I):
                facts.append(_fact(iid, "story_time_multiple_plans", "description", text, start, end,
                    "explicit-linked-story-time-transition-v1", decision=decision, temporal_role="story_setting",
                    narrative_scope="unknown", temporal_kind="linked_plans", absolute_envelope=None))
        # Format scopes: an explicit current title or an opening descriptive noun
        # phrase. Filmography never becomes the current card just by proximity.
        for flag, literals in _FORMAT_LITERALS.items():
            match = _literal_re(literals).search(sentence)
            if not match:
                continue
            after = normal(sentence[match.end():]).lstrip(" «\"“")
            title_bound = bool(title and after.startswith(title) and (len(after) == len(title) or not after[len(title)].isalnum()))
            opening_prefix = normal(sentence[:match.start()])
            opening = start == 0 and opening_prefix in ("", "это", "совместный", "историко-приключенческий", "драматический", "американский", "британский", "российский", "французский", "советский", "новый", "документальный", "анимационный", "это совместный")
            accepted = (title_bound or opening) and safe and not _OTHER.search(sentence[:match.start()])
            accepted = accepted and not re.search(r"\b(?:не|без|вместо)\s*$", sentence[:match.start()], re.I)
            quoted_work = re.match(r"\s*[«\"“]([^»\"”]+)[»\"”]", sentence[match.end():])
            if quoted_work and normal(quoted_work[1]) != title:
                accepted = False
            status = "accepted" if accepted else "needs_review"
            if re.search(r"\b(?:сборник|коллекция|альманах|подборка)\b", sentence[:match.start()], re.I) and flag == "standalone_short":
                status = "rejected"
            facts.append(_fact(iid, flag, "description", text, start, end, "explicit-current-format-v1", decision=status,
                mention_role="current_format" if accepted else "unresolved_format_scope", scope_method="current_title" if title_bound else "opening_format_noun_phrase"))
            if accepted and flag == "anthology":
                subtype = "series" if "сериал-антология" in normal(match.group()) else "film" if "фильм-антология" in normal(match.group()) or "киноальманах" in normal(match.group()) else None
                if subtype:
                    facts.append(_fact(iid, "anthology_" + subtype, "description", text, start, end, "explicit-anthology-subtype-v1"))
                if re.search(r"\bиз\s+(?:\d+|пяти|трех|четырех|шести|семи)\s+короткометражных\s+фильмов\b", normalized):
                    facts.append(_fact(iid, "compilation", "description", text, start, end, "anthology-explicit-component-collection-v1"))
            if accepted and flag == "recorded_performance":
                subtype = "concert" if "концерта" in normal(match.group()) else "standup" if "стендап" in normal(match.group()) else "theatre"
                facts.append(_fact(iid, "performance_" + subtype, "description", text, start, end, "explicit-performance-subtype-v1"))
        # An audience-directed imperative or explicitly identified current lesson.
        instructional = re.search(r"\b(?:включай(?:те)?\s+видео\s+и\s+повторяй(?:те)?\s+за|в\s+этом\s+уроке\s+(?:вы\s+)?(?:узнаете|научитесь)|на\s+стартовом\s+уроке\s+юные\s+ученики\s+смогут\s+узнать|учат\s+маленьких\s+зрителей\s+мастерить\s+поделки)\b", sentence, re.I)
        if instructional and safe:
            facts.append(_fact(iid, "instructional_intent", "description", text, start, end, "current-audience-instruction-v1", mention_role="audience_instruction"))
        broadcast = re.search(r"(?:^\s*Трансляция\s+(?:соревнований|матча|турнира|поединка)|\bСмотрите\s+трансляцию\s+турнира)\b", sentence, re.I)
        if broadcast and safe:
            facts.append(_fact(iid, "sport_format_broadcast", "description", text, start, end, "explicit-current-sport-broadcast-v1"))
        # Origin must be a grammatical assertion about the current film, not
        # mere mention of books or their adaptation in a character's plans.
        origin = re.search(r"\b(?:этот\s+)?(?:фильм|картина|сериал|аниме)\s+(?:снят[ао]?|основан[ао]?|является\s+экранизацией)\s+(?:на|по|по\s+мотивам)\s+(?:(?:одноименном|одноименной|одноименного|популярных)\s+)?(книг[а-яё]*|роман[а-яё]*|повест[а-яё]*|рассказ[а-яё]*|комикс[а-яё]*|манг[а-яё]*)\b", sentence, re.I)
        if origin:
            source = normal(origin[1])
            source_flag = "manga_adaptation" if source.startswith("манг") else "comic_adaptation" if source.startswith("комикс") else "literature_adaptation"
            facts.append(_fact(iid, "story_source_" + source_flag, "description", text, start, end, "explicit-current-work-origin-v1", scope="origin", specificity_status="literal_source_type",
                decision="accepted" if safe and not _OTHER.search(sentence[:origin.start()]) and not _EXPLICIT_OTHER_WORK.search(sentence) else "needs_review"))
        if re.search(r"\bВ\s+основу\s+сюжета\s+легла\s+реальная\s+история\b", sentence, re.I) and safe:
            facts.append(_fact(iid, "story_source_based_on_real_events", "description", text, start, end, "explicit-plot-real-origin-v1", scope="origin"))
        real_origin = re.search(r"\b(?:этот\s+)?(?:фильм|картина|сериал)\s+(?:основан[ао]?\s+на\s+реальных\s+событиях|вдохновлен[ао]?\s+реальными\s+событиями)\b", sentence, re.I)
        if real_origin:
            facts.append(_fact(iid, "story_source_based_on_real_events", "description", text, start, end,
                "explicit-current-real-events-origin-v1", scope="origin", decision="accepted" if safe and not _OTHER.search(sentence[:real_origin.start()]) and not _EXPLICIT_OTHER_WORK.search(sentence) else "needs_review"))
    return facts


def title_evidence(iid, row):
    """Only explicitly named practical lessons, with audience-directed support.

    Generic plot titles (Оригами, Машенькин концерт) remain unknown. The title
    source is retained and these subject labels are separate from keyword tags.
    """
    title, desc = row.get("title", ""), row.get("description", "")
    lesson = re.search(r"\b(?:урок\s*\d+|тренировка(?:\s+дома)?|упражнения|для\s+начинающих|основы\s+бокса)\b", title, re.I)
    audience = re.search(r"\b(?:повторяй(?:те)?\s+за|чтобы\s+вы\s+научились|вы\s+научитесь|вы\s+узнаете|включай\s+видео|начни\s+с|настрой\s+тренажер)\b", desc, re.I)
    if not lesson or not audience:
        return []
    # Food and nutritional recipes are a different activity; a fitness adjective
    # attached to a recipe does not establish physical exercise.
    if re.search(r"\b(?:рецепт[а-яё]*|кулинар[а-яё]*|филе)\b", title, re.I):
        return []
    subjects = []
    categories = {
        "activity_yoga_instructional_subject": ("йога", "йоги"),
        "activity_pilates_instructional_subject": ("пилатес",),
        "activity_stretching_instructional_subject": ("растяжка", "стретчинг"),
        "activity_fitness_instructional_subject": ("фитнес", "фитнесс", "аэробика", "кардио-тренировка"),
        "sport_running_instructional_subject": ("беговые упражнения", "техника бега", "беговая тренировка"),
        "sport_cycling_instructional_subject": ("тренировка на велосипеде",),
        "sport_boxing_instructional_subject": ("основы бокса", "тренировка по боксу"),
        "activity_dance_ballet_instructional_subject": ("уроки танцев", "танцы для начинающих", "балет для начинающих"),
    }
    for flag, aliases in categories.items():
        if _literal_re(aliases).search(title):
            subjects.append(_fact(iid, flag, "title", title, 0, len(title), "explicit-practical-title-subject-v1",
                relation_status="central", mention_role="taught_activity", supporting_description=desc))
    if subjects:
        subjects.append(_fact(iid, "instructional_intent", "title", title, 0, len(title), "explicit-practical-title-with-audience-support-v1",
            mention_role="audience_instruction", supporting_description=desc))
    return subjects


def _conflicts(facts, row):
    positives = {fact["flag"] for fact in facts if fact["decision"] == "accepted"}
    conflicts = set()
    if "story_source_based_on_real_events" in positives and any(f["rule_id"] == "real-events-pseudoreal-quarantine-v1" for f in facts):
        conflicts.add("story_source_based_on_real_events")
    if "miniseries" in positives and normal(row.get("content_type")) == "film":
        conflicts.add("miniseries")
    if "standalone_short" in positives and "compilation" in positives:
        conflicts.add("standalone_short")
    # Exact, local negation of accepted semantic literals. An unrelated "не"
    # elsewhere does not suppress a card or a different assertion.
    desc = row.get("description", "")
    negation_sentences = [(start, sentence) for start, _, sentence in _sentences(desc) if not _EXPLICIT_OTHER_WORK.search(sentence)]
    specific_negations = {
        "has_voiceover": ("нет закадрового текста", "нет закадрового голоса", "без закадрового текста"),
        "has_time_travel": ("нет путешествий во времени", "без путешествий во времени"),
        "has_time_loop": ("нет временной петли", "без временной петли"),
        "story_source_based_on_real_events": ("не основан на реальных событиях", "не основана на реальных событиях", "не основано на реальных событиях"),
    }
    for flag in positives:
        if flag in specific_negations:
            for sentence_start, sentence in negation_sentences:
                match = _literal_re(specific_negations[flag]).search(sentence)
                if match:
                    conflicts.add(flag)
                    facts.append(_fact(facts[0]["item_id"], flag, "description", desc, sentence_start + match.start(), sentence_start + match.end(),
                        "exact-local-semantic-negation-v1", decision="conflict", assertion_status="negated"))
        aliases = RULES.get(flag, {}).get("E", ())
        for alias in aliases:
            pattern = re.compile(r"\b(?:нет|без|не содержит)\s+" + _literal_re((alias,)).pattern, re.I)
            for sentence_start, sentence in negation_sentences:
                match = pattern.search(sentence)
                if match:
                    conflicts.add(flag)
                    facts.append(_fact(facts[0]["item_id"], flag, "description", desc, sentence_start + match.start(), sentence_start + match.end(),
                        "local-explicit-negation-conflict-v1", decision="conflict", assertion_status="negated"))
    for fact in facts:
        if fact["flag"] in conflicts and fact["decision"] == "accepted":
            fact["decision"] = "conflict"
            fact["reason"] = "accepted evidence contradicts explicit source assertion or raw format"
    return conflicts


def _metadata(value):
    metadata = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else dict(value)
    mapping, size = metadata.get("item_id_to_token"), metadata.get("vocab_size")
    if metadata.get("special_tokens", SPECIAL_TOKENS) != SPECIAL_TOKENS:
        raise ValueError("Unexpected special tokens")
    if (not isinstance(mapping, dict) or isinstance(size, bool) or not isinstance(size, int) or size < 3
        or any(not isinstance(k, str) or not re.fullmatch(r"0|[1-9]\d*", k) or isinstance(v, bool) or not isinstance(v, int) for k, v in mapping.items())
        or sorted(mapping.values()) != list(range(3, size))):
        raise ValueError("Expected canonical item IDs and contiguous train-known vocabulary after specials")
    return metadata, mapping


def read_items(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not {"item_id", *ITEM_FIELDS} <= set(reader.fieldnames):
            raise ValueError("Missing or duplicate semantic metadata headers")
        for row in reader:
            if None in row or any(row.get(key) is None for key in ("item_id", *ITEM_FIELDS)):
                raise ValueError("Malformed semantic metadata row")
            iid = row["item_id"].strip()
            if not re.fullmatch(r"0|[1-9]\d*", iid) or iid in result:
                raise ValueError("Invalid or duplicate item_id")
            result[iid] = {field: row[field] for field in ITEM_FIELDS}
    return result


def build_item_features(items, metadata):
    """Pure in-memory builder; fixed dictionaries and geography vocabulary fit U only."""
    metadata, mapping = _metadata(metadata)
    all_facts, per_item = [], {}
    for iid in sorted(mapping, key=int):
        row = items.get(iid, {})
        facts = keyword_evidence(iid, row) + description_evidence(iid, row) + title_evidence(iid, row)
        _conflicts(facts, row)
        all_facts.extend(facts)
        per_item[iid] = facts
    # No raw-only row affects the schema. These are direct resolved place labels,
    # not countries of production or an inferred place hierarchy.
    places = sorted({f["flag"] for f in all_facts if f["flag"].startswith("geography_") and f["decision"] == "accepted"})
    families = {}
    for flag, rule in RULES.items():
        families.setdefault(rule["group"], []).append(flag)
    families["setting_explicit"] = ["setting_" + n + "_description_explicit_setting" for n in SETTINGS]
    families["content_format"] = list(FORMATS)
    families["sport_format"] = ["sport_format_broadcast"]
    families["instructional_subjects"] = ["activity_" + n + "_instructional_subject" for n in ("yoga", "pilates", "stretching", "fitness", "dance_ballet")]
    families["instructional_subjects"] += ["sport_" + n + "_instructional_subject" for n in ("running", "cycling", "boxing")]
    families["setting_time"] = list(TIME_FLAGS)
    families["geography"] = places + ["geography_story_setting_observed"]
    columns, groups = [], {}
    for group, flags in families.items():
        start = len(columns)
        columns.extend(flag + ":" + state for flag in flags for state in ("positive", "unknown", "conflict"))
        # Per-family source/extraction masks retain absent versus unmatched,
        # rejected, unresolved and accepted without hundreds of redundant bits.
        columns.extend(group + ":state:" + state for state in STATES)
        groups[group] = {"scope": "item", "start": start, "stop": len(columns), "columns": columns[start:], "flags": flags}
    values = np.zeros((metadata["vocab_size"], len(columns)), np.float32)
    lookup = {name: index for index, name in enumerate(columns)}
    state_counts, support, surface, rejected = defaultdict(Counter), Counter(), Counter(), Counter()
    for iid, token in mapping.items():
        row = items.get(iid, {})
        by_flag = defaultdict(list)
        for fact in per_item[iid]:
            by_flag[fact["flag"]].append(fact)
        if any(f["decision"] == "accepted" for f in per_item[iid] if f["flag"].startswith("geography_")):
            by_flag["geography_story_setting_observed"].append({"decision": "accepted"})
        for group, block in groups.items():
            decisions = []
            for flag in block["flags"]:
                observations = by_flag[flag]
                ds = {f["decision"] for f in observations}
                positive, conflict = "accepted" in ds and "conflict" not in ds, "conflict" in ds
                values[token, lookup[flag + ":positive"]] = positive
                values[token, lookup[flag + ":unknown"]] = not positive
                values[token, lookup[flag + ":conflict"]] = conflict
                if positive:
                    support[flag] += 1
                if any(f.get("source_field") == "keywords" for f in observations):
                    surface[flag] += 1
                if "rejected" in ds:
                    rejected[flag] += 1
                decisions.extend(ds)
            if group == "geography":
                decisions.extend(f["decision"] for f in per_item[iid] if f["flag"].startswith("geography_"))
            source_fields = ("keywords",) if group in ("themes", "narrative", "setting_topics", "professions", "sport_mentions", "activity_mentions") else ("description", "keywords") if group == "story_source" else ("title", "description") if group in ("content_format", "instructional_subjects") else ("description",)
            source_missing = not any(row.get(field, "").strip() for field in source_fields)
            state = "conflict" if "conflict" in decisions else "matched_accepted" if "accepted" in decisions else "needs_review" if "needs_review" in decisions else "matched_rejected" if "rejected" in decisions else "source_missing" if source_missing else "present_no_evidence"
            values[token, lookup[group + ":state:" + state]] = 1
            state_counts[group][state] += 1
    coverage = {"train_known_items": len(mapping), "raw_items": len(items), "missing_item_rows": sum(iid not in items for iid in mapping),
        "accepted_item_support": dict(sorted(support.items())), "keyword_surface_item_support": dict(sorted(surface.items())),
        "rejected_item_support": dict(sorted(rejected.items())), "family_state_counts": {k: dict(v) for k, v in state_counts.items()},
        "evidence_records": len(all_facts), "geography_vocabulary": places,
        "false_positive_boundaries": ["title-only Петля времени and Голос за кадром are rejected", "женская нагота,дружба is not female friendship", "боксерские трусы is not boxing", "машина времени is not time travel", "псевдореальная история conflicts with real-events origin", "Колумбийский университет is not Colombia", "filming in London is not story geography", "filmography miniseries is not current format", "absence of a time mention is not present-day setting"],
        "not_implemented": ["general description NLP, character profession relations, general sport centrality", "description theme/narrative expansion without per-rule scope audit", "ambiguous title-only labels and broad instructional/sport title inference", "documentary geography, inferred place hierarchy, unrestricted gazetteer", "calendar dates, word-form centuries, Roman-century ranges, unanchored relative offsets", "era bins (unapproved), arbitrary earliest/latest story year, implicit contemporary setting", "unlisted keyword aliases, fuzzy matching and automated semantic synonyms"],
        "precision_recall": "not measured; lexical support is not accuracy or recall"}
    provenance = {"contract": CONTRACT, "metadata_availability_mode": "static_retrospective_catalog_snapshot", "source_known_at": None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "mapping_sha256": canonical_hash(mapping),
        "schema_sha256": canonical_hash({"columns": columns, "groups": groups}), "rules_sha256": canonical_hash(RULES),
        "fit": {"scope": "unique train-known item catalog only", "item_count": len(mapping), "item_ids_sha256": canonical_hash(sorted(mapping, key=int)), "learned_transforms": False},
        "unknown_semantics": "positive=accepted textual evidence; zero=not accepted, never verified absence; unknown remains 1 for conflicts",
        "state_encoding": "per-flag triples plus family-state masks; detailed per-flag rejected/review states recoverable from evidence",
        "evidence": all_facts, "safety": {"interactions_read": False, "outcomes_read": False, "splits_read": [], "external_models_or_network": False, "special_item_rows": "all zero"}}
    if values.dtype != np.float32 or not np.isfinite(values).all() or values[:3].any():
        raise ValueError("Invalid semantic feature matrix")
    return {"item_features": values, "columns": columns, "item_columns": columns, "groups": groups, "provenance": provenance, "coverage": coverage}


def build(raw_dir, metadata):
    """Read only items.csv plus optional metadata JSON and return an in-memory pack."""
    path = Path(raw_dir).resolve() / "items.csv"
    metadata_path = Path(metadata).resolve() if isinstance(metadata, (str, Path)) else None
    paths = {"items": path, "builder": Path(__file__).resolve()}
    if metadata_path:
        paths["metadata"] = metadata_path
    hashes = {name: file_hash(value) for name, value in paths.items()}
    meta, _ = _metadata(metadata)
    result = build_item_features(read_items(path), meta)
    if any(file_hash(value) != hashes[name] for name, value in paths.items()):
        raise ValueError("Semantic source changed while building")
    result["provenance"]["inputs"] = {name: {"path": value.as_posix(), "sha256": hashes[name]} for name, value in paths.items()}
    return result
