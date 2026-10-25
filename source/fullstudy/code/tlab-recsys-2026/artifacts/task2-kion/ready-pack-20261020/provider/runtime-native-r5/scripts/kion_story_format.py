"""Source-bound story origin and detailed format replacement for KION.

Fixed, auditable rules over current catalogue metadata. This module replaces
the named frozen semantic coordinates; it never mutates a frozen cache. Zeros
mean no accepted positive assertion. They are accompanied by extraction masks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np

_spec = importlib.util.spec_from_file_location("_story_format_base", Path(__file__).with_name("kion_pack_semantics.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
CONTRACT = "kion-story-format-v1"
STORY_FLAGS = tuple("story_source_" + name for name in base.STORY)
FORMAT_FLAGS = tuple(base.FORMATS)
FAMILIES = {"story_source": STORY_FLAGS, "content_format": FORMAT_FLAGS}
STATES = ("accepted", "rejected", "needs_review", "not_observed", "source_missing", "conflict")
FIELDS = {"story_source": ("keywords", "description"), "content_format": ("title", "title_orig", "keywords", "description")}

# These are complete keyword elements, not word/stem searches. Observed spelling
# variants are explicit entries, never edit-distance or arbitrary fragment joins.
KEYWORDS = {name: set(base.normal(x) for x in aliases.split(";")) for name, aliases in base.STORY.items()}
KEYWORDS["literature_adaptation"].update(("основанно на романе или книге", "основано на романе или книги", "по мотивам романа или книги"))
KEYWORDS["based_on_real_events"].update(("основанно на реальных событиях", "основано на реальных событий", "по мотивам реальных событий"))
PSEUDO = {"по мотивам псевдореальной истории"}
_WORD = r"[а-яёa-z]+(?:[-–—][а-яёa-z]+)*"
_MOD = r"(?:(?:одноим[её]нн\w*|популярн\w*|культов\w*|известн\w*|китайск\w*|японск\w*|детск\w*|графическ\w*|нашумевш\w*|коротк\w*)\s+)*"
_SOURCE = r"(?:графическ\w*\s+роман\w*|комикс\w*|манг\w*|маньхуа|книг\w*|роман\w*|повест\w*|рассказ\w*|новелл\w*|пьес\w*|литературн\w*\s+произведен\w*|(?:псевдо)?реальн\w*\s+(?:событи\w*|истори\w*))"
ORIGIN = re.compile(r"\b(?P<relation>(?:снят\w*|поставлен\w*)\s+по\s+(?:мотивам\s+)?|основан\w*\s+на\s+|вдохновл[её]н\w*\s+|(?:кино)?экранизаци\w*\s+|киноадаптаци\w*\s+|по\s+мотивам\s+|на\s+основе\s+)(?P<type>" + _MOD + _SOURCE + r")\b", re.I)
REAL_PLOT = re.compile(r"\bв\s+основ[уе]\s+(?:сюжета|фильма|картины|сериала)\s+(?:легл[аио]|положен\w*)\s+(?P<type>реальн\w*\s+истори\w*|реальн\w*\s+событи\w*)", re.I)
STORY_CANDIDATE = re.compile(r"\b(?:экранизаци\w*|киноадаптаци\w*|реальн\w*\s+истори\w*|манг\w*|маньхуа|комикс\w*)\b", re.I)
WORK = re.compile(r"\b(?:фильм|картина|сериал|аниме|мультфильм|лента|триллер|драма|комедия|мелодрама)\b", re.I)
OTHER = re.compile(r"\b(?:(?:предыдущ\w*|прошл\w*|друг(?:ой|ая|ое|ом|ую))\s+(?:фильм\w*|сериал\w*|работ\w*|рол\w*)|(?:позже|ранее|до\s+этого)\s+(?:(?:успешно|также|уже)\s+)*(?:снял\w*|снимал\w*|экранизир\w*|сыграл\w*|поставил\w*)|извест\w*\s+(?:по|ролью|благодаря)|снял\s+также|фильмограф\w*)\b", re.I)
PLOT = re.compile(r"\b(?:геро[йяиюев]+|персонаж\w*|девочк\w*|мальчик\w*)\b[^.!?]*(?:читает|читают|смотрит|смотрят|мечтает|мечтают|пишет|пишут|покупает|решает\s+снять|рассказывает|устроить|учится|учатся|видит|ид[её]т|посещает|готовится)\b|\b(?:мечтает\s+об|мечтал\s+о|в\s+любимом\s+фильме|внутри\s+(?:истории|сюжета))\b", re.I)
COMPONENT = re.compile(r"\b(?:в\s+(?:новелле|эпизоде|части|короткометражке|рассказе)|одна\s+из\s+(?:новелл|историй)|вторая\s+короткометражка)\b", re.I)
HYPOTHETICAL = re.compile(r"\b(?:возможно|если\s+бы|мог\w*\s+бы|вроде\s+бы|якобы|предположительно)\b", re.I)
FORMAT_PATTERNS = {
    "miniseries": r"\b(?:мини[ -]?сериал\w*|сериал\w*\s+мини[ -]формата|miniseries|mini-series)\b",
    "standalone_short": r"\b(?:короткометражн\w*\s+(?:документальн\w*\s+|анимационн\w*\s+|сказочно-философск\w*\s+)?(?:фильм\w*|мультфильм\w*|картин\w*|мелодрам\w*|притч\w*)|короткометражк\w*|короткий\s+метр|short\s+film)\b",
    "anthology": r"\b(?:киноальманах\w*|(?:фильм|сериал)[ -]антологи\w*|антологи\w*)\b",
    "compilation": r"\b(?:сборник\w*|киносборник\w*|коллекци\w*|подборк\w*|калейдоскоп)\b",
    "performance_concert": r"\b(?:(?:запись|(?:онлайн[ -])?трансляци\w*)\s+(?:(?:живого|праздничного)\s+)?концерт\w*|(?:домашни\w*\s+)?акустическ\w*\s+концерт\w*|(?:шоу[ -]|фильм[ -]|кино)?концерт\w*)\b",
    "performance_theatre": r"\b(?:фильм[ -]спектакл\w*|(?:запись|трансляци\w*)\s+(?:театральн\w*\s+)?(?:спектакл\w*|балет\w*|опер[ауы]))\b",
    "performance_standup": r"\b(?:стендап(?:[ -]выступлени\w*|ов|ы|а|у|е)?|stand[ -]up)\b",
    "instructional_intent": r"\b(?:урок\w*|учимся|обучающ\w*|образовательн\w*|инструкци\w*|комплекс\s+упражнений|мастер[ -]класс\w*|видеокурс\w*)\b",
}
FORMAT_REGEX = {k: re.compile(v, re.I) for k, v in FORMAT_PATTERNS.items()}
MATERIAL = re.compile(r"\b(?:фильм\w*|мультфильм\w*|песен|песн\w*|ролик\w*|урок\w*|выпуск\w*|стендап\w*|выступлени\w*)\b", re.I)
AUDIENCE = re.compile(r"\b(?:включай(?:те)?\s+видео\s+и\s+повторяй(?:те)?|повторяйте\s+за|вы\s+(?:научитесь|можете\s+готовить)|чтобы\s+вы\s+научились|(?:учат|учит)\s+маленьких\s+зрителей|(?:на\s+(?:стартовом|этом)|в\s+этом)\s+уроке\b|выполняйте|настрой\s+тренаж[её]р)\b", re.I)
AMBIGUOUS_CHILD_TEACH = re.compile(r"\b(?:учит|учат)\s+детей\b",re.I)


CHILD_AUDIENCE = re.compile(r"\b(?:(?:дети|ученики|ребята)\s+смогут\s+узнать\s+на\s+уроках|в\s+ходе\s+занятий\s+ребята\s+научатся)\b",re.I)


def _sentences(text):
    # Keep punctuation inside a quoted title (ДИМАШ. АРНАУ), initials and
    # domain names inside the same grammatical sentence and source span.
    result=[]; start=0; closing=None
    pairs={"«":"»","“":"”",'"':'"'}
    for pos,char in enumerate(text):
        if closing:
            if char==closing: closing=None
            continue
        if char in pairs: closing=pairs[char]; continue
        if char not in ".!?": continue
        if char=="." and pos+1<len(text) and not text[pos+1].isspace(): continue
        if char=="." and re.search(r"\b[А-ЯЁA-Z]$",text[start:pos]): continue
        result.append((start,pos+1,text[start:pos+1]));start=pos+1
    if start<len(text):result.append((start,len(text),text[start:]))
    return result


def _title(value):
    value = re.sub(r"\[\s*4[kк]\s*\]", "", base.normal(value))
    value = re.sub(r"\s*\((?:с тифлокомментарием|с субтитрами)\)\s*$", "", value)
    return " ".join(re.findall(r"[^\W_]+", value))


def _named_current(sentence, row):
    titles = {_title(row.get(k, "")) for k in ("title", "title_orig")} - {""}
    return any(_title(q) in titles for q in re.findall(r'[«“"]([^»”"]+)[»”"]', sentence))


def _current_title(value,row):
    return _title(value) in ({_title(row.get(k,"")) for k in ("title","title_orig")}-{ "" })


def _attribution(sentence, prefix, row, first=False, previous="", implicit=False):
    """Resolve grammatical work scope; never infer it from a fixed word window."""
    if COMPONENT.search(prefix):
        return "component", "plot_mention", "A component of the card is being described"
    if PLOT.search(prefix):
        return "other", "plot_mention", "An embedded narrative or character action governs the assertion"
    if implicit and PLOT.search(previous) and re.match(r"\s*(?:этот|данный|эта|данная)\s+(?:фильм|сериал|картина)\b",prefix,re.I) and not _named_current(sentence,row):
        return "other","plot_mention","Demonstrative subject refers to the embedded work introduced in the previous sentence"
    direct_current_head=implicit and any(_current_title(q,row) for q in re.findall(r'[«“"]([^»”"]+)[»”"]\s*$',prefix))
    if not direct_current_head and re.search(r"\b(?:рассказыва\w*|повеству\w*|посвящ[её]н\w*)\s+(?:о|об)\b|\b(?:фильм|сериал|история)\s+(?:о|об)\s+",prefix,re.I):
        return "other", "plot_mention", "The format/source term is the topic or object, not the work predicate"
    if OTHER.search(prefix):
        return "other", "creator_filmography", "Explicit previous/other-work attribution"
    # A quoted title attached directly to a work noun names its subject. Other
    # quoted names (an author/source) are not treated as work titles.
    for quoted in re.finditer(r'\b(?:фильм|сериал|картина|аниме|мини-сериал)\w*\s*[«“"]([^»”"]+)[»”"]', prefix, re.I):
        if not _current_title(quoted[1],row):
            return "other", "unknown", "Explicit work subject names a different title"
    if HYPOTHETICAL.search(prefix):
        return "current" if WORK.search(prefix) else "unknown", "hypothetical", "Hypothetical assertion"
    if _named_current(sentence, row):
        return "current", "origin" if implicit else "current_format", "Current title explicitly bound in the assertion sentence"
    if WORK.search(prefix) or re.search(r"\b(?:этот|данный|эта|это)\s+(?:материал|программа|сборник|коллекция)\b", prefix, re.I):
        return "current", "origin" if implicit else "current_format", "Current-work grammatical subject"
    if implicit and re.match(r"\s*в\s+основ[уе]\s+(?:сюжета|фильма|картины|сериала)\b", sentence, re.I):
        if PLOT.search(previous) or COMPONENT.search(previous) or OTHER.search(previous):
            return "unknown", "unknown", "Implicit plot subject follows an unresolved embedded/foreign work"
        return "current", "origin", "Standalone current-card plot-origin assertion"
    # Opening nominal catalogue descriptors are a separate explicit grammar.
    # A finite verb/character context cannot acquire current scope this way.
    nominal_prefix = re.fullmatch(r"\s*(?:(?:это|перед\s+вами)\s+)?(?:(?:[а-яё-]+(?:ый|ий|ая|ое|ие|ой|ые)|и)[,\s]+)*", prefix, re.I)
    if implicit and nominal_prefix and _named_current(previous,row):
        return "current", "origin", "Adjacent current-title assertion licenses the nominal origin relation"
    if first and nominal_prefix:
        return "current", "origin" if implicit else "current_format", "Opening nominal catalogue descriptor"
    return "unknown", "unknown", "No explicit current-work binding"


def _fact(iid, flag, field, text, start, end, rule, decision="needs_review", **extra):
    return {"item_id": int(iid), "family": "story_source" if flag in STORY_FLAGS else "content_format", "flag": flag,
        "source_field": field, "start": start, "end": end, "evidence_literal": text[start:end], "literal_span": text[start:end],
        "context": text, "field_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "rule_id": rule,
        "canonical_rule": rule, "rules_version": CONTRACT, "decision": decision, "work_relation": "unknown", "scope": "unknown",
        "mention_role": "unresolved", "assertion_kind": "metadata_assertion", "assertion_status": "asserted", "recipient": "none",
        "specificity_status": "not_applicable", "source_type_literal": None, "source_known_at": None,
        "normalized_phrase":base.normal(text[start:end]), "input_file_sha256":None,
        "confidence": "deterministic_rule_not_calibrated", "reviewer": None, **extra}


def _story_type(literal):
    norm = base.normal(literal)
    if re.search(r"реальн\w*\s+(?:истори|событи)", norm): return "based_on_real_events"
    if "манг" in norm or "маньхуа" in norm: return "manga_adaptation"
    if "комикс" in norm or "графическ" in norm: return "comic_adaptation"
    return "literature_adaptation"


def _specificity(kind, text):
    if kind != "manga_adaptation": return "literal_source_type"
    if re.search(r"маньхуа",text,re.I) and not re.search(r"манг",text,re.I): return "manhua_only_source_not_named_manga"
    return "manga_manhua_ambiguous" if re.search(r"маньхуа|китайск", text, re.I) else "explicit_manga_origin_country_unasserted"


def keyword_evidence(iid, row):
    text = row.get("keywords", ""); parts = base.chunks(text)
    blocked, title_like = base.metadata_chunks(row, parts); facts = []
    for n, part in enumerate(parts):
        for kind, aliases in KEYWORDS.items():
            pseudo = kind == "based_on_real_events" and part["normal"] in PSEUDO
            if part["normal"] not in aliases and not pseudo: continue
            decision = "rejected" if pseudo or n in blocked else "needs_review" if n in title_like else "accepted"
            facts.append(_fact(iid, "story_source_"+kind, "keywords", text, part["start"], part["end"],
                "exact-pseudoreal-quarantine-v1" if pseudo else "exact-inspired-real-events-keyword-v1" if part["normal"].startswith("вдохновлено") else "exact-origin-keyword-whitelist-v1", decision,
                work_relation="current" if n not in blocked | title_like else "unknown", scope="origin" if not pseudo else "unknown",
                mention_role="origin_tag", normalized_phrase=part["normal"], chunk_indices=[n], chunk_boundaries=[[part["start"], part["end"]]],
                source_type_literal=part["text"], specificity_status=_specificity(kind, part["text"]), pseudo_real=pseudo,
                reason="Pseudo-real assertion is quarantined" if pseudo else "Exact complete whitelist element; title metadata edges quarantined"))
        # Format tags are observable candidates, never a shortcut around scope.
        for flag, pattern in FORMAT_REGEX.items():
            if pattern.search(part["text"]):
                facts.append(_fact(iid, flag, "keywords", text, part["start"], part["end"], "format-keyword-candidate-v1",
                    "rejected" if n in blocked else "needs_review", mention_role="format_tag", chunk_indices=[n],
                    chunk_boundaries=[[part["start"], part["end"]]], reason="Keyword format needs current-card attribution"))
    return facts


def story_evidence(iid, row):
    text = row.get("description", ""); facts = []; previous = ""
    for start, end, sentence in _sentences(text):
        matches = list(ORIGIN.finditer(sentence)) + list(REAL_PLOT.finditer(sentence))
        for match in matches:
            kind = _story_type(match["type"]); flag = "story_source_"+kind
            prefix = sentence[:match.start()]
            relation, scope, reason = _attribution(sentence, prefix, row, start == 0, previous, True)
            # A participial relation is licensed by its explicit work head in
            # the same sentence, not by free-standing adaptation vocabulary.
            negative = bool(re.search(r"\bне\s+(?:(?:был[ао]?|является)\s+)?$|\bне\s+(?:фильм|картина|сериал|аниме)\b", prefix, re.I))
            if negative and relation == "current": scope = "negated"
            decision = "accepted" if relation == "current" and scope == "origin" else "rejected" if relation in ("other", "component") or scope == "negated" else "needs_review"
            pseudo=bool(re.search(r"псевдореальн",match["type"],re.I))
            if pseudo:decision="rejected"
            if kind=="manga_adaptation" and not re.search(r"манг",match["type"],re.I):
                decision="needs_review" if relation not in ("other","component") else "rejected"
                reason="Manhua-only source is not explicitly named manga; do not silently broaden the agreed flag"
            if kind=="literature_adaptation" and re.search(r"пьес",match["type"],re.I) and re.match(r"\s+или\s+мюзикл",sentence[match.end():],re.I):
                decision="needs_review";reason="Unresolved play-or-musical source type"
            facts.append(_fact(iid, flag, "description", text, start, end,
                "current-work-inspired-origin-v1" if re.search("вдохновл", match.group(), re.I) else "current-work-explicit-origin-v1", decision,
                work_relation=relation, scope=scope, reason=reason, mention_role="adaptation_origin" if scope == "origin" else scope,
                assertion_status="negated" if negative else "hypothetical" if scope == "hypothetical" else "asserted",
                source_type_literal=match["type"], specificity_status=_specificity(kind, sentence),
                pseudo_real=pseudo and relation=="current" and scope=="origin",
                relation_start=start+match.start(), relation_end=start+match.end(), relation_literal=match.group()))
        if not matches:
            for match in STORY_CANDIDATE.finditer(sentence):
                kind = _story_type(match.group())
                # Untyped 'экранизация' must not become literature by default.
                if re.match(r"(?:экранизаци|киноадаптаци)", match.group(), re.I):
                    kind = "literature_adaptation"
                relation, scope, reason = _attribution(sentence, sentence[:match.start()], row, False, previous, True)
                facts.append(_fact(iid, "story_source_"+kind, "description", text, start, end, "unresolved-origin-candidate-v1",
                    "rejected" if relation in ("other", "component") else "needs_review", work_relation=relation, scope=scope,
                    mention_role="untyped_or_plot_mention", source_type_literal=match.group(), specificity_status="source_type_unknown" if "экранизац" in match.group().lower() else _specificity(kind, sentence), reason=reason))
        previous = sentence
    return facts


def format_evidence(iid, row):
    text = row.get("description", ""); facts = []
    title = row.get("title", "")
    for field in ("title", "title_orig"):
        value = row.get(field, "")
        for flag, pattern in FORMAT_REGEX.items():
            for match in pattern.finditer(value):
                facts.append(_fact(iid, flag, field, value, match.start(), match.end(), "format-title-candidate-v1", reason="Title requires explicit format support from description", mention_role="title"))
    previous=""
    for start, end, sentence in _sentences(text):
        for flag, pattern in FORMAT_REGEX.items():
            for match in pattern.finditer(sentence):
                prefix = sentence[:match.start()]
                relation, scope, reason = _attribution(sentence, prefix, row, True,previous)
                previous_current=any(f["source_field"]=="description" and f["end"]==start and f["decision"]=="accepted" and f["work_relation"]=="current" for f in facts)
                if relation=="unknown" and previous_current and re.match(r"\s*это\s+не\s*$",prefix,re.I):
                    relation,scope,reason="current","current_format","Anaphoric negation refers to the immediately preceding accepted current-format assertion"
                negative = bool(re.search(r"\b(?:не\s+(?:является\s+)?|без\s+|вместо\s+)$", prefix, re.I) or re.match(r"\s+(?:не\s+является|отсутствует)\b",sentence[match.end():],re.I))
                accepted = relation == "current" and scope == "current_format"
                subtype = None
                if flag == "compilation":
                    accepted &= bool(re.match(r"\s+(?:(?:[а-яё-]+(?:ых|их|ихся|ющих)|по-настоящему)\s+)*(?:отдельных\s+)?(?:фильмов|мультфильмов|песен|роликов|уроков|выпусков|стендапов|выступлений)\b",sentence[match.end():],re.I))
                if flag == "standalone_short":
                    production_bound = bool(re.search(r"\b(?:съ[её]мочн\w*\s+групп\w*\s+на\s+создание|представляет)\s*$",prefix,re.I)) and (_named_current(previous,row) or bool(re.match(r"\s*(?:HBO|НВО)\s+представляет",sentence,re.I)))
                    if production_bound:
                        accepted,relation,scope,reason=True,"current","current_format","Current production/producer introduction bound to current card"
                    main_production=bool(re.match(r"\s*Режисс[её]р\b",sentence,re.I) and re.search(r",\s*снял[а]?\s*$",prefix,re.I) and _named_current(text,row))
                    if main_production:
                        accepted,relation,scope,reason=True,"current","current_format","Main production predicate describes current titled work; earlier debut title is subordinate filmography"
                    if re.search(r"\b(?:сборник|коллекция|подборка|киноальманах|антология|из\s+(?:\d+|пяти|тр[её]х))\b", prefix, re.I) or re.search(r"\b(?:вторая|третья|четвертая|первая)\s+$", prefix, re.I) or FORMAT_REGEX["compilation"].search(title):
                        relation, scope, accepted, reason = "component", "plot_mention", False, "Short component is not the whole card"
                    if re.search(r"\b(?:основан\w*\s+на|по\s+мотивам|экранизаци\w*|выросл\w*\s+из|после\s+успеха|на\s+основе)\b",prefix,re.I):
                        relation, scope, accepted, reason = "other", "source_mention", False, "The short is an adaptation source, not the current card format"
                    foreign_target=re.match(r'\s*(?:под\s+названием\s*)?[«“"]([^»”"]+)[»”"]',sentence[match.end():],re.I)
                    if foreign_target and not _current_title(foreign_target[1],row):
                        relation,scope,accepted,reason="component","plot_mention",False,"Attached short title names a component or different work"
                    if re.search(r"\bкороткометражн(?:ые|ых|ими|ыми)\b",match.group(),re.I):
                        relation,scope,accepted,reason="component","plot_mention",False,"Plural short films do not establish one standalone short card"
                if flag == "anthology":
                    if re.search(r"\b(?:альманах\w*|антологи\w*)\b", match.group(), re.I) and re.search(r"\b(?:эссе|рассказов|литературн\w*|презентаци\w*)\b", sentence, re.I):
                        relation, scope, accepted, reason = "other", "source_mention", False, "Literary collection or its presentation is outside the film anthology contract"
                    subtype = "series" if re.search("сериал", match.group(), re.I) else "film" if re.search("фильм|киноальманах", match.group(), re.I) else None
                if flag.startswith("performance_"):
                    subtype = flag.split("_", 1)[1]
                    explicit = bool(re.search(r"\b(?:запись|(?:онлайн[ -])?трансляци\w*|фильм[ -]спектакл\w*|акустическ\w*\s+концерт\w*|стендап[ -]выступлени\w*|фильм[ -]концерт\w*)\b", sentence, re.I))
                    catalog_standup = flag == "performance_standup" and start == 0 and bool(re.match(r"\s*(?:калейдоскоп|подборка|сборник)\s+стендап", sentence, re.I))
                    coupled = flag == "performance_concert" and bool(re.search(r"(?:—|–|-)\s*это\s+[^.!?]*акустическ\w*\s+концерт", sentence, re.I)) and not PLOT.search(sentence)
                    recording_cta = bool(re.search(r"\b(?:смотрите|(?:приглашаем|предлагаем)[^.!?]*посмотр[её]ть|можно\s+посмотреть)\b",sentence,re.I)) and explicit and not PLOT.search(sentence) and not OTHER.search(prefix)
                    accepted = (accepted and explicit) or catalog_standup or coupled or recording_cta
                    if re.search(r"\b(?:документальн\w*\s+(?:фильм|сериал)\s+(?:о|об)|(?:фильм|история)\s+(?:о|об)\b)", prefix, re.I):
                        relation, scope, accepted, reason = "other", "plot_mention", False, "A film about a performance is not the performance recording"
                if flag == "instructional_intent":
                    direct_lesson = bool(AUDIENCE.search(sentence) or CHILD_AUDIENCE.search(sentence))
                    educational = bool(re.search(r"\b(?:сборник\w*[^.!?]*урок\w*\s+по\b|(?:обучающ\w*|образовательн\w*)\s+(?:программ\w*|мультфильм\w*|сериал\w*|проект\w*)|комплекс\s+упражнений|мастер[ -]класс\s+по)\b", sentence, re.I))
                    accepted = (educational and (accepted or start == 0 and bool(re.match(r"\s*(?:сборник\w*|обучающ\w*|образовательн\w*|комплекс|мастер[ -]класс)\b",sentence,re.I)))) or direct_lesson and relation not in ("other", "component") and not PLOT.search(prefix)
                    if accepted: relation, scope, reason = "current", "current_format", "Audience-directed current lesson or explicit educational material"
                    if PLOT.search(sentence):
                        relation,scope,accepted,reason="other","plot_mention",False,"A character learning in the plot is not audience instruction"
                if negative:
                    accepted = False
                    if relation == "current": scope = "negated"
                if relation in ("other","component") or scope == "hypothetical" or re.search(r"\b(?:не\s+(?:смотрите|предлагаем|рекомендуем|приглашаем)|нельзя\s+(?:посмотреть|смотреть))\b",sentence,re.I):
                    accepted=False
                if accepted:
                    relation, scope = "current", "current_format"
                decision = "accepted" if accepted else "rejected" if relation in ("other", "component") or negative else "needs_review"
                extra = dict(work_relation=relation, scope=scope, reason=reason, mention_role="current_format" if accepted else scope,
                    assertion_status="negated" if negative else "hypothetical" if scope == "hypothetical" else "asserted", subtype=subtype,
                    matched_alias=match.group(), relation_start=start+match.start(), relation_end=start+match.end())
                facts.append(_fact(iid, flag, "description", text, start, end, "explicit-current-detailed-format-v1", decision, **extra))
                if accepted and flag == "anthology":
                    if subtype: facts.append(_fact(iid, "anthology_"+subtype, "description", text, start, end, "explicit-anthology-subtype-v1", "accepted", **extra))
                    if re.search(r"\bиз\s+(?:\d+|пяти|тр[её]х|четыр[её]х|шести|семи)\s+короткометражных\s+фильмов", sentence, re.I):
                        facts.append(_fact(iid, "compilation", "description", text, start, end, "explicit-anthology-material-composition-v1", "accepted", **extra))
                if accepted and flag.startswith("performance_"):
                    facts.append(_fact(iid, "recorded_performance", "description", text, start, end, "explicit-performance-parent-v1", "accepted", **extra))
        # Imperative instruction can occur without any lexical 'lesson' token.
        if (AUDIENCE.search(sentence) or CHILD_AUDIENCE.search(sentence)) and not FORMAT_REGEX["instructional_intent"].search(sentence) and not PLOT.search(sentence) and not OTHER.search(sentence):
            hypothetical=bool(HYPOTHETICAL.search(sentence))
            facts.append(_fact(iid, "instructional_intent", "description", text, start, end, "current-audience-instruction-v1", "needs_review" if hypothetical else "accepted",
                work_relation="unknown" if hypothetical else "current", scope="hypothetical" if hypothetical else "current_format", assertion_status="hypothetical" if hypothetical else "asserted",mention_role="audience_instruction"))
        elif AMBIGUOUS_CHILD_TEACH.search(sentence) and not FORMAT_REGEX["instructional_intent"].search(sentence):
            facts.append(_fact(iid,"instructional_intent","description",text,start,end,"child-teaching-unresolved-audience-v1","rejected" if PLOT.search(sentence) else "needs_review",
                work_relation="other" if PLOT.search(sentence) else "unknown",scope="plot_mention" if PLOT.search(sentence) else "unknown",mention_role="unresolved_teaching_recipient",
                reason="Teaching children may be a character action; explicit viewer/current-lesson scope is required"))
        previous=sentence
    # A collection title is accepted only with literal component composition.
    if FORMAT_REGEX["compilation"].search(title) and re.match(r"\s*\d+\s+", text) and MATERIAL.search(text.split(".", 1)[0]):
        facts.append(_fact(iid, "compilation", "description", text, 0, len(text), "title-bound-explicit-material-count-v1", "accepted",
            work_relation="current", scope="current_format", mention_role="current_material_composition", supporting_title=title))
    presented_collection=re.search(r"\bпредставля(?:ет|ют)\s+(?:[а-яё-]+\s+)*(?:коллекци\w*|сборник\w*|подборк\w*)\s+из\s+\d+\s+[^.!?]*\b(?:фильмов|мультфильмов|роликов|песен|уроков)\b",text,re.I)
    if FORMAT_REGEX["compilation"].search(title) and presented_collection and re.search(r"\b(?:создател\w*|студи\w*|Disney|Pixar)\b",text[:presented_collection.start()],re.I) and not PLOT.search(text[:presented_collection.start()]):
        facts.append(_fact(iid,"compilation","description",text,0,len(text),"title-bound-presented-material-collection-v1","accepted",
            work_relation="current",scope="current_format",mention_role="current_material_composition",supporting_title=title,
            reason="Creators present the current titled collection; prior creator film titles are supporting filmography, not the collection subject"))
    # Current practical title + an explicit learning goal establishes the
    # current lesson. A full-course link is retained as context, not its scope.
    practical_title=bool(re.search(r"\b(?:программирован\w*|google\s+диск|paint|arduino|minecraft|python|gimp|photoshop|scratch|roblox|tinkercard)\b",title,re.I))
    learning_context=bool(re.search(r"\b(?:урок\w*|курс\w*|ученик\w*|обучени\w*)\b",text,re.I))
    learning_goal=bool(re.search(r"\b(?:осво\w*|изуч\w*|науч\w*|разобраться|поможет|помогут|принцип\w*|инструкц\w*|расскажет\s+как|как\s+это\s+сделать|узнать\s+в\s+уроке|смо[гж]\w*|урок\s+(?:рассчитан|предоставлен|является))\b",text,re.I))
    if practical_title and learning_context and learning_goal and not re.search(r"\b(?:сюжет|герой|художественн\w*\s+фильм|документальн\w*\s+фильм)\b",text,re.I) and not HYPOTHETICAL.search(text):
        facts.append(_fact(iid,"instructional_intent","description",text,0,len(text),"practical-title-current-learning-goal-v1","accepted",
            work_relation="current",scope="current_format",mention_role="audience_instruction",subtype="current_lesson_material",supporting_title=title,course_scope="Current material only; linked full course is not asserted present"))
    # An explicit current concert title requires an actual event/recording
    # statement. It does not license fictional concerts or artist biographies.
    concert_title=bool(re.search(r"\bконцерт\w*\b",title,re.I))
    event_support=re.search(r"\b(?:смотрите\s+онлайн[ -]трансляцию|музыканты\s+собрались\s+вместе,\s*чтобы\s+записать|устроила\s+грандиозный\s+концерт\s+в\s+честь\s+юбилея)\b",text,re.I)
    if concert_title and event_support and not re.search(r"\b(?:герой|героиня|девочка|мультфильм|художественный\s+фильм)\b",text,re.I):
        for flag in ("recorded_performance","performance_concert"):
            facts.append(_fact(iid,flag,"description",text,0,len(text),"concert-title-current-event-or-recording-v1","accepted",work_relation="current",scope="current_format",mention_role="current_recording",subtype="concert",supporting_title=title))
    viewer_howto=bool(re.search(r"\bрассказывают\s+юным\s+зрителям,\s*как\s+правильно\b",text,re.I))
    arithmetic_program=bool(re.match(r"\s*Познавательная\s+детская\s+передача\b",text,re.I) and re.search(r"\bучит\s+детей\s+навыкам\s+арифметики\b",text,re.I) and re.search(r"\bюные\s+зрители\b",text,re.I))
    diy_goal=bool(re.match(r"\s*Сделай\s+сам\b",title,re.I) and re.search(r"\bдети\s+(?:узнают\s+)?научатся\s+создавать\s+красоту\s+своими\s+руками\b",text,re.I))
    practical_daily_skills=bool(re.search(r"\bдети\s+научатся\s+ежедневным\s+бытовым\s+вещам:\s*уборка,\s*умывание,\s*одевание\b",text,re.I))
    if (viewer_howto or arithmetic_program or diy_goal or practical_daily_skills) and not HYPOTHETICAL.search(text):
        facts.append(_fact(iid,"instructional_intent","description",text,0,len(text),"explicit-viewer-practical-learning-context-v1","accepted",
            work_relation="current",scope="current_format",mention_role="audience_instruction",supporting_title=title,
            reason="Explicit viewer how-to, arithmetic programme, current DIY objective or named daily practical skills establishes instruction independently of character/moral teaching"))
    if any(f["decision"]=="accepted" and f["flag"]=="compilation" for f in facts):
        for fact in facts:
            if fact["flag"]=="standalone_short" and fact["decision"]=="accepted":
                fact.update(decision="rejected",work_relation="component",scope="plot_mention",mention_role="component_format",reason="Card-level accepted collection composition excludes standalone short for the whole card")
    return facts


def evidence(iid, row, temporal_mode="retrospective_current"):
    facts = keyword_evidence(iid, row) + story_evidence(iid, row) + format_evidence(iid, row)
    for flag in STORY_FLAGS + FORMAT_FLAGS:
        current = [f for f in facts if f["flag"] == flag]
        positive = any(f["decision"] == "accepted" for f in current)
        negative = any(f["assertion_status"] == "negated" and f["work_relation"] == "current" for f in current)
        pseudo = any(f.get("pseudo_real") for f in current)
        conflict = positive and (negative or pseudo or flag == "miniseries" and base.normal(row.get("content_type")) == "film")
        if conflict:
            for fact in current:
                if fact["decision"] == "accepted" or fact["assertion_status"] == "negated" or fact.get("pseudo_real"):
                    fact.update(decision="conflict", conflict_reason="Contradictory current source assertions or raw content_type; no field wins by priority")
    disputed={f["flag"].removeprefix("performance_") for f in facts if f["flag"].startswith("performance_") and f["decision"]=="conflict"}
    parents=[f for f in facts if f["flag"]=="recorded_performance" and f["decision"]=="accepted"]
    safe_parent=any(f.get("subtype") not in disputed for f in parents)
    for fact in parents:
        if fact.get("subtype") in disputed:
            fact.update(decision="rejected" if safe_parent else "conflict",reason="Derived performance parent depends on a disputed subtype")
    if temporal_mode == "historical_as_of":
        for fact in facts:
            if fact["decision"] in ("accepted", "conflict") or fact["assertion_status"] == "negated":
                fact.update(decision="needs_review", assertion_status="unavailable", reason="source_known_at unknown at historical cutoff")
    return facts


def _schema():
    columns = []; groups = {}
    for family, flags in FAMILIES.items():
        start = len(columns)
        columns.extend(flag+":"+state for flag in flags for state in ("positive", "unknown", "conflict"))
        columns.extend(family+":state:"+s for s in base.STATES)
        groups[family] = {"scope": "item", "start": start, "stop": len(columns), "flags": list(flags), "columns": columns[start:]}
    replacements = list(columns)
    for family, flags in FAMILIES.items():
        start = len(columns)
        columns.extend(flag+":explicit_negative" for flag in flags)
        columns.extend(flag+":extraction:"+state for flag in flags for state in STATES)
        columns.extend(family+":source:"+field+":missing" for field in FIELDS[family])
        groups[family+"_support"] = {"scope": "item", "start": start, "stop": len(columns), "columns": columns[start:]}
    return columns, groups, replacements


def build_item_features(items, metadata, temporal_mode="retrospective_current", metadata_cutoff=None, max_seconds=600, max_rss_mb=2048):
    metadata, mapping = base._metadata(metadata)
    if temporal_mode not in ("retrospective_current", "historical_as_of"): raise ValueError("Unknown temporal mode")
    if temporal_mode == "historical_as_of" and metadata_cutoff is None: raise ValueError("Historical mode requires an explicit role-specific cutoff")
    started = time.monotonic(); columns, groups, replacements = _schema(); index = {n:i for i,n in enumerate(columns)}
    matrix = np.zeros((metadata["vocab_size"], len(columns)), np.float32)
    facts = []; support = Counter(); counts = defaultdict(Counter); stages = defaultdict(Counter)
    for counter, (iid, token) in enumerate(mapping.items()):
        if counter % 100 == 0:
            if time.monotonic()-started > max_seconds: raise TimeoutError("Bounded catalogue pass exceeded time budget")
            if _rss_mb() > max_rss_mb: raise MemoryError("Bounded catalogue pass exceeded RSS budget")
        row = items.get(iid, {}); current = evidence(iid, row, temporal_mode); facts.extend(current)
        by_flag = defaultdict(list)
        for fact in current: by_flag[fact["flag"]].append(fact)
        for family, flags in FAMILIES.items():
            decisions = []
            for field in FIELDS[family]: matrix[token,index[family+":source:"+field+":missing"]] = not bool(row.get(field, "").strip())
            missing = not any(row.get(field, "").strip() for field in FIELDS[family])
            for flag in flags:
                records = by_flag[flag]; ds = {f["decision"] for f in records}; conflict = "conflict" in ds
                positive = "accepted" in ds and not conflict
                negative = any(f["assertion_status"] == "negated" and f["work_relation"] == "current" for f in records) and not conflict
                matrix[token,index[flag+":positive"]] = positive
                matrix[token,index[flag+":unknown"]] = not positive  # explicit negative retains 'not accepted positive'
                matrix[token,index[flag+":conflict"]] = conflict
                matrix[token,index[flag+":explicit_negative"]] = negative
                state = "conflict" if conflict else "accepted" if positive else "needs_review" if "needs_review" in ds else "rejected" if ds else "source_missing" if missing else "not_observed"
                matrix[token,index[flag+":extraction:"+state]] = 1
                counts[flag][state] += 1; support[flag] += int(positive); decisions.extend(ds)
                kw = any(f["source_field"] == "keywords" and f["decision"] == "accepted" for f in records)
                desc = any(f["source_field"] == "description" and f["decision"] == "accepted" for f in records)
                stages[flag]["keywords_only"] += int(positive and kw and not desc)
                stages[flag]["description_only"] += int(positive and desc and not kw)
                stages[flag]["intersection"] += int(positive and kw and desc)
                stages[flag]["accepted_union"] += int(positive)
            family_state = "conflict" if "conflict" in decisions else "matched_accepted" if "accepted" in decisions else "needs_review" if "needs_review" in decisions else "matched_rejected" if decisions else "source_missing" if missing else "present_no_evidence"
            matrix[token,index[family+":state:"+family_state]] = 1
    provenance = {"contract": CONTRACT, "temporal_mode": temporal_mode, "metadata_cutoff": str(metadata_cutoff) if metadata_cutoff else None,
        "source_known_at": None, "historical_as_of_eligible": False, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "mapping_sha256": base.canonical_hash(mapping), "item_ids_sha256": base.canonical_hash(sorted(mapping,key=int)),
        "schema_sha256": base.canonical_hash({"columns":columns,"groups":groups}), "vocabulary_sha256": base.canonical_hash(FAMILIES),
        "rules_sha256": base.canonical_hash({"keywords":{k:sorted(v) for k,v in KEYWORDS.items()},"formats":FORMAT_PATTERNS,"origin":ORIGIN.pattern,"real_plot":REAL_PLOT.pattern}),
        "replacement_columns": replacements, "replacement_semantics": "Replace every named old coordinate, including positive, unknown, conflict and family masks; append only support columns",
        "source_precedence": "Current relation and scope required; contradictory accepted/negated/pseudo-real claims quarantine the flag regardless of field",
        "learned_transforms": False, "interactions_read": False, "outcomes_read": False, "network": False, "gpu": False,
        "special_item_rows": "all zero", "base_module_modified": False, "evidence": facts,
        "limits": {"cpu_threads":2,"max_seconds":max_seconds,"max_rss_mb":max_rss_mb}, "elapsed_seconds":time.monotonic()-started, "peak_rss_mb":_rss_mb()}
    return {"item_features": matrix, "matrix": matrix, "columns": columns, "item_columns":columns, "groups":groups,
        "replace_columns":replacements, "coverage":{"train_known_items":len(mapping),"raw_items":len(items),"missing_item_rows":sum(iid not in items for iid in mapping),
            "accepted_item_support":dict(support),"flag_state_counts":{k:dict(v) for k,v in counts.items()},"stage_item_support":{k:dict(v) for k,v in stages.items()},
            "evidence_records":len(facts),"precision_recall":"Not estimated. Full-U source-supported census and documented boundary fixtures, not global semantic recall or model gain."},"provenance":provenance}


def _rss_mb():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)] + [(name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
        counters = Memory(); counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        query.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not query(handle, ctypes.byref(counters), counters.cb): raise OSError("Unable to measure bounded process memory")
        return counters.PeakWorkingSetSize / 1024**2
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build(raw_dir, metadata, **kwargs):
    path = Path(raw_dir)/"items.csv"; digest = base.file_hash(path)
    result = build_item_features(base.read_items(path),metadata,**kwargs)
    if base.file_hash(path) != digest: raise ValueError("Source changed during extraction")
    for fact in result["provenance"]["evidence"]:
        fact.update(input_file_sha256=digest,temporal_mode=result["provenance"]["temporal_mode"])
    root=Path(__file__).resolve().parents[1]
    paths = {"items":path,"builder":Path(__file__),"frozen_helper":Path(base.__file__),
        "tests":root/"tests/test_kion_story_format.py","story_contract":root/"docs/task2-story-source-features.md","format_contract":root/"docs/task2-content-format-features.md"}
    if isinstance(metadata,(str,Path)): paths["metadata"] = Path(metadata)
    result["provenance"]["inputs"] = {k:{"path":v.resolve().as_posix(),"sha256":base.file_hash(v)} for k,v in paths.items()}
    return result


def replacement_binding(result, base_manifest, base_features, base_schema):
    """Explicit checked replacement coordinates for an immutable old cache."""
    schema = json.loads(Path(base_schema).read_text(encoding="utf-8"))
    names = schema.get("item_columns", schema.get("columns", schema.get("feature_columns",{}).get("item")))
    if not isinstance(names,list) or len(names) != len(set(names)): raise ValueError("Base schema must contain unique item column names")
    targets = {n:n if n in names else "semantics."+n for n in result["replace_columns"]}
    missing = set(targets.values())-set(names)
    if missing: raise ValueError("Base schema is missing replacement coordinates: "+str(sorted(missing)))
    manifest=json.loads(Path(base_manifest).read_text(encoding="utf-8"))
    expected=manifest.get("outputs",{}).get("item_features.npy",{})
    if expected and expected.get("sha256") != base.file_hash(base_features): raise ValueError("Frozen item matrix no longer matches its manifest")
    metadata_pin=manifest.get("inputs",{}).get("metadata.json",{}).get("sha256")
    current_pin=result["provenance"].get("inputs",{}).get("metadata",{}).get("sha256")
    if metadata_pin and metadata_pin != current_pin: raise ValueError("Replacement metadata differs from frozen vocabulary metadata")
    if expected:
        array=np.load(base_features,mmap_mode="r",allow_pickle=False)
        if array.shape != (result["item_features"].shape[0],len(names)): raise ValueError("Frozen matrix dimensions do not match pinned metadata/schema")
    binding = {"base_manifest_sha256":base.file_hash(base_manifest),"base_item_features_sha256":base.file_hash(base_features),
        "base_schema_sha256":base.file_hash(base_schema),"base_schema_hash":schema.get("schema_hash",base.canonical_hash(names)),
        "mapping_sha256":result["provenance"]["mapping_sha256"],"schema_sha256":result["provenance"]["schema_sha256"],
        "coordinates":[{"name":targets[n],"replacement_name":n,"base_index":names.index(targets[n]),"replacement_index":result["columns"].index(n)} for n in result["replace_columns"]]}
    return binding


def save_full(raw_dir, metadata, out_dir, base_manifest=None, base_features=None, base_schema=None):
    started = time.monotonic(); out = Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError("Use a fresh output directory; receipts are immutable")
    result = build(raw_dir,metadata); matrix=result["item_features"]; facts=result["provenance"].pop("evidence")
    meta,mapping = base._metadata(metadata); items=base.read_items(Path(raw_dir)/"items.csv")
    np.savez_compressed(out/"features.npz",item_features=matrix)
    np.save(out/"replacement-item-features.npy",matrix[:,:len(result["replace_columns"])],allow_pickle=False)
    np.save(out/"supplement-item-features.npy",matrix[:,len(result["replace_columns"]):],allow_pickle=False)
    def write(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    write("columns.json",{"columns":result["columns"],"groups":result["groups"],"replace_columns":result["replace_columns"]})
    write("coverage.json",result["coverage"]); write("vocabulary.json",{"families":FAMILIES,"keywords":{k:sorted(v) for k,v in KEYWORDS.items()},"format_patterns":FORMAT_PATTERNS})
    with (out/"evidence.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        for fact in facts: stream.write(json.dumps(fact,ensure_ascii=False,sort_keys=True)+"\n")
    by_item=defaultdict(list)
    for fact in facts: by_item[str(fact["item_id"])].append(fact)
    # Broad discovery census runs on every U row, independently of admissions.
    # Residuals are retained with full literal metadata for subsequent review.
    census=Counter(); residual=[]
    broad=re.compile(r"\b(?:основан\w*|по\s+мотивам|экранизац\w*|манг\w*|маньхуа|мини[ -]?сериал\w*|короткометраж\w*|антолог\w*|альманах\w*|сборник\w*|коллекци\w*|концерт\w*|стендап\w*|спектакл\w*|урок\w*|обучающ\w*)\b",re.I)
    for iid in mapping:
        row=items.get(iid,{})
        for field in ("title","title_orig","keywords","description"):
            terms={base.normal(m.group()) for m in broad.finditer(row.get(field,""))}
            for term in terms: census[field+":"+term]+=1
        if any(broad.search(row.get(field,"")) for field in ("title","keywords","description")) and not any(f["decision"] in ("accepted","conflict") for f in by_item[iid]):
            residual.append({"item_id":int(iid),"title":row.get("title",""),"description":row.get("description",""),"keywords":row.get("keywords",""),"status":"discovery_candidate_not_admitted"})
    write("census.json",{"universe_items":len(mapping),"raw_items":len(items),"all_u_rows_scanned":True,"broad_surface_item_counts":dict(sorted(census.items())),"unadmitted_broad_candidate_items":len(residual),"residual_candidates":residual})
    rng=random.Random(20261020); audit=[]
    for flag in STORY_FLAGS+FORMAT_FLAGS:
        for state in STATES:
            col=result["columns"].index(flag+":extraction:"+state)
            ids=sorted((iid for iid,token in mapping.items() if matrix[token,col]),key=int)
            for iid in rng.sample(ids,min(5,len(ids))):
                audit.append({"flag":flag,"stratum":state,"stratum_population":len(ids),"seed":20261020,"item_id":int(iid),"source":items.get(iid,{}),
                    "evidence":[f for f in by_item[iid] if f["flag"]==flag],"audit_status":"pending_independent_literal_review"})
    write("stratified-audit.json",audit)
    if any(x is not None for x in (base_manifest,base_features,base_schema)):
        if not all(x is not None for x in (base_manifest,base_features,base_schema)): raise ValueError("All three frozen base pins are required")
        write("replacements.json",replacement_binding(result,base_manifest,base_features,base_schema))
    write("typed-matrices.json",{"contract":CONTRACT,"source_npz_sha256":base.file_hash(out/"features.npz"),
        "mapping_sha256":result["provenance"]["mapping_sha256"],"vocab_size":matrix.shape[0],"special_rows_zero":bool(not matrix[:3].any()),
        "replacement":{"path":"replacement-item-features.npy","sha256":base.file_hash(out/"replacement-item-features.npy"),"shape":[matrix.shape[0],len(result["replace_columns"])],"columns":["semantics."+n for n in result["replace_columns"]]},
        "supplement":{"path":"supplement-item-features.npy","sha256":base.file_hash(out/"supplement-item-features.npy"),"shape":[matrix.shape[0],len(result["columns"])-len(result["replace_columns"])],"columns":result["columns"][len(result["replace_columns"]):]}})
    outputs={p.name:{"sha256":base.file_hash(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file()}
    result["provenance"].update(shape=list(matrix.shape),dtype=str(matrix.dtype),special_rows_zero=bool(not matrix[:3].any()),full_universe=True,
        elapsed_seconds=time.monotonic()-started,peak_rss_mb=_rss_mb(),outputs=outputs,admission_status="pending_independent_code_and_artifact_review")
    write("manifest.json",result["provenance"])
    return {k:v for k,v in result["provenance"].items() if k not in ("replacement_columns",)}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ("raw-dir","metadata","out-dir"): parser.add_argument("--"+name,required=True)
    for name in ("base-manifest","base-features","base-schema"): parser.add_argument("--"+name)
    args=parser.parse_args()
    print(json.dumps(save_full(args.raw_dir,args.metadata,args.out_dir,args.base_manifest,args.base_features,args.base_schema),ensure_ascii=False,indent=2))
