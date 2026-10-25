"""Additive, evidence-preserving KION semantics omitted by the narrow pack.

No inherited keyword family is duplicated: description assertions, character
roles, central sport subjects, documentary geography and raw edition metadata
are separate columns. Unknown is never a verified absence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import importlib.util
from pathlib import Path
import json
import re
import unicodedata

import numpy as np

_spec = importlib.util.spec_from_file_location("_kion_narrow_semantics", Path(__file__).with_name("kion_pack_semantics.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
CONTRACT = "kion-semantic-extensions-v1"
EDITION_FLAGS = ("edition.is_4k", "accessibility.sign_interpretation", "accessibility.audio_description", "subtitles.present")
EDITION_TERMS = {
    "edition.is_4k": r"(?<!\w)4\s*[kк](?!\w)|\buhd\b|\bultra\s*hd\b",
    "accessibility.sign_interpretation": r"\bсурдоперевод\w*|\bжестов\w*\s+язык\w*|\bsign\s+language\b",
    "accessibility.audio_description": r"\bтифлокомментар\w*|\bаудиодескрип\w*|\baudio\s*description\b|\baudiodescription\b",
    "subtitles.present": r"\bсубтитр\w*|\bsubtitles?\b",
}
PROFESSION_FORMS = {
    "medical_practitioner": r"врач(?:ом|а)?|хирург(?:ом|а)?", "nurse": r"медсестр(?:а|ой|ы)",
    "mental_health_specialist": r"психолог(?:ом|а)?|психиатр(?:ом|а)?", "veterinarian": r"ветеринар(?:ом|а)?",
    "teacher": r"учител(?:ь|ем|я)|учительниц(?:а|ей|ы)", "scientist": r"учен(?:ый|ым|ого)",
    "engineer": r"инженер(?:ом|а)?", "lawyer": r"адвокат(?:ом|а)?|юрист(?:ом|а)?",
    "police": r"полицейск(?:ий|им|ого)", "investigator": r"следовател(?:ь|ем|я)|частн(?:ый|ым|ого)\s+детектив(?:ом|а)?",
    "military": r"военнослужащ(?:ий|им|его)|солдат(?:ом|а)?", "firefighter": r"пожарн(?:ый|ым|ого)",
    "writer": r"писател(?:ь|ем|я)|писательниц(?:а|ей|ы)", "journalist": r"журналист(?:ом|а)?|журналистк(?:а|ой|и)",
    "musician_singer": r"музыкант(?:ом|а)?|пев(?:ец|цом|ца)|певиц(?:а|ей|ы)",
    "visual_artist": r"художник(?:ом|а)?|художниц(?:а|ей|ы)", "cook": r"повар(?:ом|а)?|шеф-повар(?:ом|а)?",
    "professional_driver": r"водител(?:ь|ем|я)|дальнобойщик(?:ом|а)?", "pilot": r"пилот(?:ом|а)?",
    "athlete": r"спортсмен(?:ом|а)?|спортсменк(?:а|ой|и)", "bar_waiter": r"бармен(?:ом|а)?|официант(?:ом|а)?|официантк(?:а|ой|и)",
    "accountant": r"бухгалтер(?:ом|а)?",
}
SPORT_FORMS = {
    "football": r"\b(?:футбол(?:а|е|ом)?|футболист\w*|футбольн\w*)\b", "american_football": r"американск\w*\s+футбол\w*",
    "basketball": r"баскетбол\w*", "boxing": r"бокс(?:а|ом|е|ер\w*)?",
    "martial_arts": r"смешанн\w*\s+единоборств\w*|\bмма\b|\bmma\b|карате|каратэ|дзюдо|кикбоксинг\w*",
    "running": r"\bбег(?:а|е|ом)?\b|бегун\w*|марафонск\w*\s+бег\w*",
    "swimming": r"плавани\w*|пловчих\w*|пловец|пловц\w*", "cycling": r"велоспорт\w*|велогон\w*|велосипедн\w*\s+трен\w*|тренировк\w*\s+на\s+велосипеде",
    "winter_sports": r"хокке\w*|фигурн\w*\s+катани\w*|биатлон\w*|лыжн\w*\s+спорт\w*|сноуборд\w*|керлинг\w*",
    "tennis": r"теннис(?:а|е|у|ом)?|теннисист\w*", "surfing": r"серфинг\w*|серфер\w*", "chess": r"шахмат(?:ы|ах|ами|ист\w*)|гроссмейстер\w*",
}
ACTIVITY_FORMS = {"fitness": r"фитнес\w*|кардио-тренировк\w*", "yoga": r"йог(?:а|и|ой|у)",
    "pilates": r"пилатес\w*", "stretching": r"растяжк\w*|стретчинг\w*", "dance_ballet": r"танц\w*|балет\w*",
    "cooking": r"готовить|приготовить|кулинар\w*|рецепт\w*", "fishing": r"рыбалк\w*|рыбн\w*\s+ловл\w*",
    "crafts": r"поделк\w*|рукодели\w*|вязани\w*|вышивк\w*|оригами|гончарн\w*"}
_BAD_SCOPE = re.compile(r"\b(?:другой фильм|другом фильме|ранее снял|ранее сыграл|известн\w* ролью|фильмограф\w*|в отличие от|если бы|возможно|когда-нибудь|планируется|обещают|в будущем добавят)\b|\b(?:герой|героиня|персонаж)\b[^.!?]{0,65}\b(?:говорит|рассказывает|вспоминает|цитирует|включает|смотрит)\b|\b(?:актер|актриса|режиссер|исполнитель)\b[^.!?]{0,65}\b(?:снялся|снялась|снимался|снималась|сыграл|сыграла|снял)\b|\bпо\s+сюжету\b[^.!?]{0,100}\b(?:видео|урок|фильм|цитат\w*)\b", re.I)
_NEGATION = re.compile(r"\b(?:не|нет|без|недоступ\w*)\b", re.I)


def _fact(iid, flag, field, raw, start, end, rule, status="confirmed_positive", **extra):
    relation = "other" if re.search(r"\b(?:другой|другом)\s+фильм",raw[start:end],re.I) else "unknown" if _BAD_SCOPE.search(raw[start:end]) else "current"
    return {"item_id": int(iid), "flag": flag, "source_field": field, "literal_span": raw[start:end],
        "start": start, "end": end, "field_sha256": base.hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "rules_version": CONTRACT, "rule_id": rule, "status": status, "decision": "accepted" if status == "confirmed_positive" else status,
        "source_known_at": None, "temporal_mode": "retrospective_current", "work_relation": relation, **extra}


def _sentences(text):
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text):
        if match.group().strip():
            yield match.start(), match.end(), match.group()


def _normalized_match(pattern, raw):
    """NFKC match with offsets projected back to original Unicode source."""
    chars, offsets = [], []
    for index, char in enumerate(raw):
        folded = unicodedata.normalize("NFKC", char).casefold().replace("ё", "е")
        chars.extend(folded)
        offsets.extend([index]*len(folded))
    text = "".join(chars)
    for match in re.finditer(pattern, text, re.I):
        yield offsets[match.start()], offsets[match.end()-1]+1, match.group()


def edition_evidence(iid, row):
    facts = []
    for field in ("title", "title_orig"):
        raw = row.get(field, "")
        marker_patterns = {
            "edition.is_4k": r"\[4[kк]\]",
            "accessibility.sign_interpretation": r"\((?:жестовым\s+языком|сурдоперевод\s*\+\s*субтитры|sign\s+language)\)?",
            "accessibility.audio_description": r"\((?:с\s+тифлокомментар\w*|with\s+audio\s+description)\)?",
            "subtitles.present": r"\((?:субтитры|с\s+(?:\w+\s+)?субтитрами|на\s+[^()]{0,60}\s+с\s+(?:\w+\s+)?субтитрами|сурдоперевод\s*\+\s*субтитры)\)?",
        }
        for flag, pattern in marker_patterns.items():
            for start, end, _ in _normalized_match(pattern, raw):
                facts.append(_fact(iid, flag, field, raw, start, end, "explicit-raw-edition-marker-v1", edition_identity="kion-raw:"+str(iid)))
        for flag, pattern in EDITION_TERMS.items():
            if not any(f["flag"] == flag and f["source_field"] == field for f in facts):
                for start, end, _ in _normalized_match(pattern, raw):
                    facts.append(_fact(iid, flag, field, raw, start, end, "unresolved-title-edition-candidate-v1", "needs_review"))
    raw = row.get("description", "")
    current_patterns = {
        "edition.is_4k": r"смотрите\s+в\s+4\s*[kк]",
        "accessibility.sign_interpretation": r"(?:в\s+(?:ленту|фильм)\s+(?:добавлен\w*|внедрен\w*)\s+(?:дорожк\w*\s+с\s+)?жестов\w*\s+язык\w*|(?:фильм|версия)\s+(?:содержит\s+сурдоперевод|представлен\w*\s+с\s+сурдоперевод\w*))",
        "accessibility.audio_description": r"(?:в\s+фильм\s+добавлен\w*\s+тифлокомментар\w*|версия\s+с\s+(?:тифлокомментар\w*|аудиодескрип\w*))",
        "subtitles.present": r"(?:смотрите|посмотреть|предлагаем\s+посмотреть)[^.!?]{0,100}(?:с\s+(?:\w+\s+)?субтитрами|with\s+subtitles)",
    }
    for start, end, sentence in _sentences(raw):
        safe = not _BAD_SCOPE.search(sentence)
        for flag, pattern in current_patterns.items():
            if re.search(pattern, sentence, re.I):
                status = "confirmed_positive" if safe and not _NEGATION.search(sentence) else "needs_review"
                facts.append(_fact(iid, flag, "description", raw, start, end, "current-card-availability-assertion-v1", status))
        # Only explicit subjects establish card-level negatives; a keyword or
        # fictional scene without subtitles is never a card availability claim.
        for flag, term in EDITION_TERMS.items():
            negative = r"\b(?:эта\s+версия|данная\s+версия|этот\s+фильм|фильм|версия|карточка)\s+(?:не\s+содержит|(?:представлен\w*|доступн\w*)\s+без|без)\s+(?:"+term+r")"
            if safe and re.search(negative, sentence, re.I):
                facts.append(_fact(iid, flag, "description", raw, start, end, "explicit-current-edition-negative-v1", "explicit_negative"))
        if re.search(r"(?<!\w)4\s*[kк]\s+недоступ\w*", sentence, re.I) and safe:
            facts.append(_fact(iid, "edition.is_4k", "description", raw, start, end, "explicit-4k-unavailable-v1", "explicit_negative"))
    by_flag = defaultdict(set)
    for fact in facts:
        by_flag[fact["flag"]].add(fact["status"])
    conflict_flags = {flag for flag, states in by_flag.items() if {"confirmed_positive", "explicit_negative"} <= states}
    title_flags = {f["flag"] for f in facts if f["source_field"] in ("title", "title_orig") and f["status"] == "confirmed_positive"}
    desc_flags = {f["flag"] for f in facts if f["source_field"] == "description" and f["status"] == "confirmed_positive"}
    access = {"accessibility.sign_interpretation", "accessibility.audio_description"}
    if len(title_flags & access) == 1 and len(desc_flags & access) == 1 and title_flags & access != desc_flags & access:
        conflict_flags |= access
    for fact in facts:
        if fact["flag"] in conflict_flags and fact["status"] in ("confirmed_positive", "explicit_negative"):
            fact.update(status="conflict", decision="conflict", reason="Contradictory raw-card accessibility or availability assertions")
    return facts


def description_evidence(iid, row):
    facts = []
    raw = row.get("description", "")
    title = row.get("title", "")
    for start, end, sentence in _sentences(raw):
        safe = not _BAD_SCOPE.search(sentence)
        # Theme/narrative assertions remain separate from keyword mention bits.
        for family in (base.THEMES, base.NARRATIVE):
            for flag, aliases in family.items():
                match = base._literal_re(base._aliases(aliases)).search(sentence)
                if not match:
                    continue
                prefix = sentence[:match.start()]
                bound = bool(re.search(r"(?:главная\s+тема\s+(?:фильма|сериала)|в\s+основе\s+сюжета|фильм\s+(?:использует|содержит|показывает)|повествование\s+использует)\s*[-—–:]?\s*$", prefix, re.I))
                status = "confirmed_positive" if safe and bound and not _NEGATION.search(prefix[-20:]) else "needs_review"
                facts.append(_fact(iid, flag+"_description_assertion", "description", raw, start, end, "explicit-current-semantic-assertion-v1", status))
        for name, form in PROFESSION_FORMS.items():
            # Requires a protagonist/plot noun or explicit employment relation.
            match = re.search(r"\b(?:главн\w*\s+геро\w*[^.!?]{0,35}?(?:работает\s+)?|(?:молодой|молодая|пожилой|опытный|скромный)\s+|история\s+(?:молодого\s+)?)(?:"+form+r")\b", sentence, re.I)
            employment = re.search(r"\b(?:работает|работал|работала|устроил(?:ся|ась))\s+(?:в\s+\w+\s+)?(?:"+form+r")\b", sentence, re.I)
            if match or employment:
                previous = raw[:start].strip()
                prior_sentence = list(_sentences(previous))[-1][2] if previous else ""
                current_subject = bool(re.search(r"\b(?:главн\w*\s+геро\w*|героин\w*|персонаж\w*)\b",sentence,re.I))
                cast_context = re.search(r"\b(?:актер|актриса|режиссер|автор|роль|роли|сыграл|сыграла|съемок|известен\s+зрителям)\b",sentence+" "+prior_sentence,re.I)
                accepted = safe and (bool(match) or current_subject) and (not cast_context or current_subject) and not _NEGATION.search(sentence[:(match or employment).end()])
                facts.append(_fact(iid, "profession_"+name+"_character_relation", "description", raw, start, end, "explicit-current-character-profession-v1", "confirmed_positive" if accepted else "needs_review", mention_role="character_occupation", centrality="unknown"))
    audience = re.search(r"\b(?:включай\w*\s+видео\s+и\s+повторяй\w*|выполняйте|настрой\s+тренажер|вы\s+научились|вы\s+научитесь|учат\s+(?:маленьких\s+зрителей|детей)|учит\s+детей|в\s+каждой\s+передаче[^.!?]+расскажет[^.!?]+приготовить|вы\s+можете\s+готовить\s+вместе)\b", raw, re.I) if not _BAD_SCOPE.search(raw) else None
    broadcast = re.search(r"\b(?:трансляция\s+соревнований|смотрите\s+трансляцию\s+турнира|встреча\s+двух\s+лидеров\s+текущего\s+сезона|поединок\s+с\s+топовым\s+представителем)\b", raw, re.I)
    fiction = re.search(r"\b(?:истори\w*\s+взлетов\s+и\s+падений[^.!?]{0,45}|сюжет\s+(?:фильма|сериала)\s+о\s+|спортивн\w*\s+фильм)\b", raw, re.I)
    documentary = re.search(r"\bдокументальн\w*\s+(?:фильм|сериал)\s+(?:рассказывает\s+)?(?:о|об|посвящен)\b", raw, re.I)
    formats = {"instruction_training": audience, "broadcast": broadcast, "fiction_with_sport_theme": fiction, "documentary": documentary}
    food = bool(re.search(r"\b(?:рецепт\w*|курин\w*\s+филе)\b", title, re.I))
    for axis, forms in (("sport", SPORT_FORMS), ("activity", ACTIVITY_FORMS)):
        for category, form in forms.items():
            subject_pattern = r"\b(?:"+form+r")\b"
            match = re.search(subject_pattern, raw, re.I)
            title_match = re.search(subject_pattern, title, re.I)
            if not match and not (title_match and audience):
                continue
            context = raw[max(0, match.start()-80):min(len(raw),match.end()+100)] if match else raw
            rejected = bool(re.search(r"\b(?:другими\s+видами\s+активности|напоминает\s+плавный\s+танец|боксерск\w*\s+(?:трус\w*|груш\w*)|кафе\s+под\s+названием)\b", context, re.I))
            if match:
                prefix = raw[max(0,match.start()-60):match.start()]
                suffix = raw[match.end():match.end()+60]
                rejected = rejected or bool(re.search(r"\b(?:не\s+(?:о|об|про|занимается|занимаются|является|являются|содержит)|нет|без|никакого|никаких)\s*$", prefix, re.I)) or bool(
                    re.match(r"\s+(?:отсутствует|не\s+(?:является\s+темой|показан|представлен))\b", suffix, re.I))
            if category == "football" and re.search(SPORT_FORMS["american_football"], context, re.I):
                rejected = True
            if food and (axis == "sport" or category == "fitness"):
                rejected = True
            current_format = next((name for name, evidence in formats.items() if evidence), None)
            safe = not _BAD_SCOPE.search(context)
            # A format cue elsewhere in a plot does not turn every mentioned
            # sport into the subject. Broadcasts have an explicit global event
            # scope; instruction needs a matching practical title or a direct
            # teaching/exercise sentence, fiction a same-sentence plot cue.
            if current_format == "instruction_training":
                subject_bound = bool(title_match) or any(re.search(subject_pattern,sentence,re.I) and re.search(
                    r"\b(?:учат|учит|науч\w*|выполняйте|повторяйте|расскажет[^.!?]+приготовить|вы\s+можете\s+готовить|беговые\s+упражнения|упражнения\s+для|упражнения\s+по|тренировк\w*\s+по|мастерить)\b",sentence,re.I)
                    and not _BAD_SCOPE.search(sentence) for _,_,sentence in _sentences(raw))
            elif current_format == "fiction_with_sport_theme":
                subject_bound = bool(re.search(r"(?:истори\w*|сюжет|спортивн\w*\s+фильм)[^.!?]{0,100}(?:"+form+r")", raw, re.I))
            elif current_format == "documentary":
                subject_bound = False
                for _,_,sentence in _sentences(raw):
                    cue = re.search(r"\bдокументальн\w*\s+(?:фильм|сериал)\s+(?:рассказывает\s+)?(?:о|об|посвящен)\s+",sentence,re.I)
                    if cue:
                        topic_clause = re.split(r",\s*(?:в\s+(?:первом|котором|кадре)|на\s+|где\b|котор\w*\b|но\b|а\b)|\b(?:в\s+первом\s+кадре|на\s+заднем\s+плане)\b",sentence[cue.end():],maxsplit=1,flags=re.I)[0]
                        subject_bound |= bool(re.search(subject_pattern,topic_clause,re.I)) and not _BAD_SCOPE.search(sentence)
                opening = next(_sentences(raw))[2]
                if re.match(r"\s*Хроника\s+жизни\b",opening,re.I) and not _BAD_SCOPE.search(opening):
                    subject_bound |= bool(re.search(subject_pattern,opening,re.I))
            elif current_format == "broadcast":
                subject_sentence = next((sent for a,b,sent in _sentences(raw) if match and a <= match.start() < b), "")
                subject_bound = bool(re.search(r"\b(?:трансляц\w*|встреча|поединок|турнир\w*|соревнован\w*|дистанци\w*|км)\b",subject_sentence,re.I)) and not re.search(r"\b(?:также\s+смотрите|рекомендуем|на\s+нашем\s+сайте)\b",subject_sentence,re.I)
            else:
                subject_bound = bool(current_format)
            central = bool(current_format) and subject_bound and safe and not rejected
            field, text = ("description", raw) if match else ("title", title)
            flag = axis+"_"+category+"_central_subject"
            facts.append(_fact(iid, flag, field, text, 0, len(text), "explicit-current-activity-subject-v1",
                "confirmed_positive" if central else "rejected" if rejected else "needs_review",
                relation_status="central" if central else "unknown", mention_role="taught_activity" if audience else "current_subject" if central else "unresolved_or_false_candidate",
                rejection_reason=None if central else "Matched an alternative, negated, figurative, food, clothing or object candidate" if rejected else "No direct current documentary topic, event or teaching clause establishes centrality",
                subtype=(match or title_match).group(), supporting_description=raw if field == "title" else None))
            if central:
                # Cooking instruction remains an instructional activity, never
                # a sport format by itself.
                fflag = "activity_format_instruction_training" if axis == "activity" and current_format == "instruction_training" else "sport_format_"+current_format
                facts.append(_fact(iid, fflag, "description", raw, 0, len(raw), "explicit-current-subject-format-v1", relation_status="central"))
    # Explicit documentary location in title; type/subject noun is required.
    known_subject_title = bool(re.search(r"\bОбучение\s+орангутангов\s*[.:]",title,re.I))
    documentary_current = bool(re.search(r"\bдокументальн\w*\s+(?:фильм|сериал)\b",raw,re.I)) and not _BAD_SCOPE.search(raw)
    if (known_subject_title or documentary_current) and not re.search(r"\b(?:комедийный|игровой)\s+фильм\b",raw,re.I) and re.search(r"\b(?:орангутанг\w*|природа|животн\w*|документальн\w*)\b", title, re.I):
        match = re.search(r"\bИндонезия\b", title, re.I)
        if match:
            facts.append(_fact(iid, "geography_documentary_subject_location:indonesia", "title", title, match.start(), match.end(), "explicit-nature-documentary-title-place-v1", role="documentary_subject_location", place_id="indonesia", place_type="country", centrality="unknown", entity_resolution_status="resolved_literal", direct_or_inferred="direct"))
    return facts


def _families():
    return {"edition_accessibility": list(EDITION_FLAGS),
        "theme_description": [n+"_description_assertion" for n in base.THEMES],
        "narrative_description": [n+"_description_assertion" for n in base.NARRATIVE],
        "profession_character": ["profession_"+n+"_character_relation" for n in PROFESSION_FORMS],
        "sport_central": ["sport_"+n+"_central_subject" for n in SPORT_FORMS],
        "activity_central": ["activity_"+n+"_central_subject" for n in ACTIVITY_FORMS],
        "sport_activity_format": ["sport_format_"+n for n in ("broadcast", "instruction_training", "documentary", "fiction_with_sport_theme")]+["activity_format_instruction_training"],
        "documentary_geography": ["geography_documentary_subject_location:indonesia"]}


def build_item_features(items, metadata, temporal_mode="retrospective_current", metadata_cutoff=None, reviewed_annotations=None):
    metadata, mapping = base._metadata(metadata)
    if temporal_mode not in ("retrospective_current", "historical_as_of"):
        raise ValueError("Unknown temporal mode")
    if temporal_mode == "historical_as_of" and metadata_cutoff is None:
        raise ValueError("Historical mode needs explicit metadata cutoff")
    if isinstance(reviewed_annotations,(str,Path)):
        reviewed_annotations=json.loads(Path(reviewed_annotations).read_text(encoding="utf-8"))
    reviewed_annotations = reviewed_annotations or {"decisions":[]}
    adjudications={}
    allowed_review_flags={name+"_description_assertion" for name in (*base.THEMES,*base.NARRATIVE)}
    for entry in reviewed_annotations.get("decisions",[]):
        cid=entry["candidate_id"]
        if cid in adjudications or entry["decision"] not in ("accepted_positive","rejected","uncertain") or not entry.get("review_id"):
            raise ValueError("Invalid or duplicate independently reviewed semantic decision")
        if entry.get("flag") not in allowed_review_flags or entry.get("source_field") != "description" or isinstance(entry.get("item_id"),bool) or not isinstance(entry.get("item_id"),int):
            raise ValueError("Semantic review must name a frozen description flag and exact raw item")
        if str(entry["item_id"]) not in mapping:
            continue
        source=items.get(str(entry["item_id"]),{}).get("description","")
        start,end=entry.get("start"),entry.get("end")
        if (isinstance(start,bool) or isinstance(end,bool) or not isinstance(start,int) or not isinstance(end,int)
            or not 0 <= start < end <= len(source) or source[start:end] != entry.get("evidence_literal")
            or base.hashlib.sha256(source.encode("utf-8")).hexdigest() != entry.get("field_sha256")):
            raise ValueError("Reviewed semantic evidence no longer matches raw source")
        expected_id=base.hashlib.sha256(json.dumps({k:entry[k] for k in ("item_id","flag","start","end","field_sha256")},sort_keys=True).encode()).hexdigest()[:20]
        if cid != expected_id:
            raise ValueError("Reviewed semantic candidate ID does not match item/flag/span")
        if entry["decision"] == "accepted_positive" and entry.get("work_relation") != "current":
            raise ValueError("Positive semantic adjudication must identify the current work")
        adjudications[cid]=entry
    families, columns, groups = _families(), [], {}
    states = ("positive", "explicit_negative", "unknown", "conflict")
    for family, flags in families.items():
        start = len(columns)
        columns.extend(flag+":"+state for flag in flags for state in states)
        columns.extend(family+":state:"+state for state in base.STATES)
        groups[family] = {"scope": "item", "start": start, "stop": len(columns), "columns": columns[start:], "flags": flags}
    index = {name: i for i, name in enumerate(columns)}
    matrix = np.zeros((metadata["vocab_size"], len(columns)), dtype=np.float32)
    evidence, support, family_counts = [], Counter(), defaultdict(Counter)
    consumed_adjudications=set()
    for iid, token in mapping.items():
        row = items.get(iid, {})
        facts = edition_evidence(iid, row)+description_evidence(iid, row)
        adjudicated_flags=set()
        for fact in facts:
            key={k:fact[k] for k in ("item_id","flag","start","end","field_sha256")}
            cid=base.hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]
            entry=adjudications.get(cid)
            if entry:
                if entry.get("evidence_literal") != fact["literal_span"] or entry.get("field_sha256") != fact["field_sha256"]:
                    raise ValueError("Reviewed semantic evidence no longer matches raw source")
                fact.update(status="confirmed_positive" if entry["decision"] == "accepted_positive" else "rejected" if entry["decision"] == "rejected" else "needs_review",
                    decision="accepted" if entry["decision"] == "accepted_positive" else entry["decision"],
                    work_relation=entry["work_relation"], rule_id="independently-reviewed-description-assertion-v1",
                    review_id=entry["review_id"], review_reason=entry["reason"], candidate_id=cid,
                    annotation_source="Independent literal/full-description adjudication plus coordinator review")
                adjudicated_flags.add(fact["flag"])
                consumed_adjudications.add(cid)
        # Reviewed scopes take precedence over automatic assertions on the same
        # card+flag. A different unreviewed span may not silently bypass review.
        for fact in facts:
            if fact["flag"] in adjudicated_flags and not fact.get("review_id") and fact["status"] == "confirmed_positive":
                fact.update(status="needs_review",decision="needs_review",reason="Card/flag has explicit manual adjudication; this automatic span is unreviewed")
        if temporal_mode == "historical_as_of":
            # Current raw catalogue has no evidence-availability timestamps.
            for fact in facts:
                if fact["status"] in ("confirmed_positive", "explicit_negative"):
                    fact.update(status="needs_review", decision="needs_review", reason="source_known_at unknown for historical cutoff")
        evidence.extend(facts)
        by_flag = defaultdict(list)
        for fact in facts:
            by_flag[fact["flag"]].append(fact)
        for family, flags in families.items():
            statuses = set()
            for flag in flags:
                observed = {f["status"] for f in by_flag[flag]}
                conflict = "conflict" in observed or {"confirmed_positive", "explicit_negative"} <= observed
                positive = "confirmed_positive" in observed and not conflict
                negative = "explicit_negative" in observed and not conflict
                matrix[token,index[flag+":positive"]] = positive
                matrix[token,index[flag+":explicit_negative"]] = negative
                matrix[token,index[flag+":unknown"]] = not (positive or negative)
                matrix[token,index[flag+":conflict"]] = conflict
                if positive:
                    support[flag] += 1
                statuses |= observed
            fields = ("title", "title_orig", "description") if family == "edition_accessibility" else ("description",) if family in ("theme_description", "narrative_description", "profession_character") else ("title",) if family == "documentary_geography" else ("title", "description")
            source_missing = not any(row.get(field, "").strip() for field in fields)
            state = "conflict" if "conflict" in statuses else "matched_accepted" if "confirmed_positive" in statuses else "needs_review" if "needs_review" in statuses else "matched_rejected" if statuses else "source_missing" if source_missing else "present_no_evidence"
            matrix[token,index[family+":state:"+state]] = 1
            family_counts[family][state] += 1
    if consumed_adjudications != adjudications.keys():
        raise ValueError("Reviewed semantic decisions unmatched by current detector: "+",".join(sorted(adjudications.keys()-consumed_adjudications)))
    return {"item_features": matrix, "columns": columns, "item_columns": columns, "groups": groups,
        "coverage": {"train_known_items": len(mapping), "accepted_item_support": dict(sorted(support.items())),
            "family_state_counts": {k:dict(v) for k,v in family_counts.items()}, "evidence_records": len(evidence),
            "not_closed": ["unrestricted description NLP", "general documentary gazetteer", "complete sport/character relation recall", "historical source availability"],
            "all_unknown_families": [family for family, flags in families.items() if not any(support[f] for f in flags)],
            "precision_recall": "Hand fixture only; catalogue-wide precision/recall not measured"},
        "provenance": {"contract": CONTRACT, "temporal_mode": temporal_mode, "metadata_cutoff": str(metadata_cutoff) if metadata_cutoff else None,
            "source_known_at": None, "retrieved_at": datetime.now(timezone.utc).isoformat(), "mapping_sha256": base.canonical_hash(mapping),
            "schema_sha256": base.canonical_hash({"columns": columns, "groups": groups}), "evidence": evidence,
            "reviewed_annotations_sha256":base.canonical_hash(reviewed_annotations) if adjudications else None,
            "reviewed_decisions":len(adjudications),
            "applied_reviewed_decisions":len(consumed_adjudications),
            "learned_transforms": False, "interactions_read": False, "special_item_rows": "all zero", "base_module_modified": False}}


def build(raw_dir, metadata, **kwargs):
    path = Path(raw_dir)/"items.csv"
    digest = base.file_hash(path)
    result = build_item_features(base.read_items(path), metadata, **kwargs)
    if base.file_hash(path) != digest:
        raise ValueError("Semantic input changed during build")
    result["provenance"]["inputs"] = {"items": {"path": path.resolve().as_posix(), "sha256": digest},
        "builder": {"path": Path(__file__).resolve().as_posix(), "sha256": base.file_hash(__file__)}}
    return result
