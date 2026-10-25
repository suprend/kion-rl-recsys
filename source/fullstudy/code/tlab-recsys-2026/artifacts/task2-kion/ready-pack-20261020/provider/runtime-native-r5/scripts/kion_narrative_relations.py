"""Finite, source-bound narrative assertions supplementing the frozen keyword pack.

The six approved flags are independent. Explicit current-work descriptions,
keyword mentions, scoped denials and abstentions remain separate. No interactions,
external models, outcome archives or network are read. The raw snapshot has no
historical availability clock; historical mode therefore abstains.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


CONTRACT = "kion-narrative-relations-v1"
BASE_SHA256 = "74a59bad3bf89c5eb92624452ffac8e4775c18592823411a611d09d50fe3d7ff"
FLAGS = ("has_flashback", "has_voiceover", "has_time_travel", "has_time_loop", "has_surrealism", "has_absurdism")
STATES = ("matched_accepted", "source_missing", "present_no_evidence", "matched_rejected", "needs_review", "conflict")
METRICS = ("keyword_positive", "description_positive", "union_positive", "explicit_negative", "description_unknown", "union_unknown",
    "conflict", "needs_review", "rejected_candidate", "unresolved_metadata", "keyword_source_missing", "description_source_missing")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_base_path = Path(__file__).with_name("kion_pack_semantics.py")
if file_hash(_base_path) != BASE_SHA256:
    raise ValueError("Frozen narrative keyword helper hash mismatch")
_spec = importlib.util.spec_from_file_location("_narrative_frozen_keywords", _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


# Closed complete-word forms; these select candidates, never arbitrary stems.
FORMS = {
    "has_flashback": r"фл[еэ]шб[еэ]к(?:|а|и|ов|ами|ах|ом|у|е)",
    "has_voiceover": r"закадров(?:ый\s+голос|ого\s+голоса|ым\s+голосом|ый\s+текст|ого\s+текста|ым\s+текстом|ое\s+повествование|ого\s+повествования|ым\s+повествованием|ый\s+комментарий|ого\s+комментария)|голос\s+за\s+кадром|голоса\s+за\s+кадром",
    "has_time_travel": r"(?:путешестви(?:е|я|ю|ем|и|й|ям|ями|ях)|путешественник(?:|а|и|ов)|перемещени(?:е|я|й|ем|ями|ях))\s+во\s+времени|путешеств(?:ует|уют|уя|овать)\s+во\s+времени|перемеща(?:ться|ется)\s+во\s+времени",
    "has_time_loop": r"временн(?:ая\s+петля|ой\s+петл(?:и|е|ей)|ую\s+петлю|ые\s+петли|ых\s+петель)|петл(?:я|и|ю|е|ей)\s+времени",
    "has_surrealism": r"сюрреализм(?:|а|ом|у|е)|сюрреалист(?:ичный|ичного|ичная|ичной|ичную|ичное|ичные|ичных|ичными|ическими|ический|ического|ическая|ической|ическую|ическое|ические|ических)|сюрреалистчный|сюрреалист(?:|а|ов|ом|ы)",
    "has_absurdism": r"абсурдизм(?:|а|ом|у|е)|абсурд(?:|а|ом|у|е)|абсурдн(?:ый|ого|ая|ой|ую|ое|ым|ом|ому|ые|ых|ыми|о|ы|ость|остью)|абсурдистск(?:ая|ими)",
}
PATTERNS = {flag: re.compile(r"(?<!\w)(?:" + forms + r")(?!\w)", re.I) for flag, forms in FORMS.items()}
_FOREIGN = re.compile(r"\b(?:друг(?:ой|ого|ом|ая|ую)|предыдущ(?:ий|его|ем)|любим(?:ый|ого|ом))\s+(?:фильм[а-яё]*|сериал[а-яё]*|картин[а-яё]*)\b|\b(?:фильмограф[а-яё]*|извест[а-яё]*\s+ролью|ранее\s+(?:снял|сыграл)|премьера\s+[^.!?]*фильма)\b", re.I)
_EMBEDDED = re.compile(r"\b(?:герой|герои|героиня|героини|персонаж|персонажи)\b[^.!?]{0,90}\b(?:смотрит|смотрят|включает|обсуждает|цитирует|снимает)\b|\b(?:по\s+сюжету|внутри\s+сюжета)\b[^.!?]{0,60}\bфильм\b", re.I)
_HYPOTHETICAL = re.compile(r"\b(?:если\s+бы|возможно|якобы|мечтает|мечтают|хочет|хотят|планирует|планируют|мог(?:ла|ли)?\s+бы|может\s+быть)\b", re.I)
_CREATOR = re.compile(r"\b(?:режиссер[а-яё-]*|мастер[а-яё]*|отца|кинорежиссер[а-яё-]*)\b[^.!?]{0,60}$", re.I)
_COMPONENT = re.compile(r"\b(?:первая|вторая|третья|отдельная|одна\s+из)\s+(?:часть|новелла|история)|\b(?:в\s+(?:первой|второй|третьей|отдельной)\s+(?:части|новелле|истории)|в\s+одной\s+из\s+(?:новелл|историй)|отдельный\s+компонент)\b", re.I)
_WORK = r"(?:фильм|картина|сериал|лента|веб-сериал|повествование|сюжет|мультфильм|аниме|комедия|драма)"


def _sentences(raw):
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", raw):
        if match.group().strip():
            yield match.start(), match.end(), match.group()


def _quoted_spans(raw):
    return list(re.finditer(r'«[^»]*»|“[^”]*”|„[^“]*“|"[^"\n]*"', raw))


def _predicate(flag, literal):
    normalized = base.normal(literal)
    if flag == "has_voiceover":
        if "текст" in normalized:
            return "voiceover_text"
        if "повествован" in normalized:
            return "voiceover_narration"
        if "комментар" in normalized:
            return "voiceover_commentary"
        return "voiceover_voice"
    return flag.removeprefix("has_")


def _relation(flag, sentence, start, end, title, previous=""):
    """Classify one explicit phrase, never a bag of narrative words."""
    norm = sentence.replace("ё", "е").replace("Ё", "Е")
    before, literal, after = norm[:start], norm[start:end], norm[end:]
    result = {"status": "needs_review", "work_relation": "unknown", "assertion_status": "unknown", "scope": "unknown",
        "rule_id": "unresolved-explicit-narrative-candidate-v1", "reason": "No supported current-work assertion"}
    def finish(status, rule, reason, relation="current", assertion="positive", scope="current_work_assertion"):
        return {**result, "status": status, "rule_id": rule, "reason": reason, "work_relation": relation, "assertion_status": assertion, "scope": scope}
    quote = next((m for m in _quoted_spans(sentence) if m.start() <= start and end <= m.end()), None)
    if quote:
        return finish("rejected", "title-or-quoted-narrative-phrase-v1", "A quoted phrase/title is not a narrative assertion", "unknown", "unknown", "quoted_title_or_speech")
    if _FOREIGN.search(norm):
        return finish("rejected", "foreign-work-narrative-v1", "Explicit assertion about a different work", "other", "unknown", "other_work")
    if _EMBEDDED.search(norm):
        return finish("rejected", "embedded-work-narrative-v1", "A film/story inside the plot is not the current work", "other", "unknown", "embedded_work")
    if _COMPONENT.search(norm) or _COMPONENT.search(previous):
        return finish("needs_review", "component-narrative-scope-v1", "Component does not establish a property of the whole card", "component", "unknown", "component")
    for quoted_subject in re.finditer(r"\b(?:фильм|фильме|картина|картине|сериал|сериале|лента|ленте|мультфильм|мультфильме)\s*[«\"“]([^»\"”]+)[»\"”]", before, re.I):
        if base.normal(quoted_subject[1]) != base.normal(title):
            return finish("rejected", "named-foreign-subject-v1", "The assertion explicitly names a different work", "other", "unknown", "other_work")
    if _HYPOTHETICAL.search(before):
        return finish("needs_review", "hypothetical-narrative-v1", "Hypothesis or intention is not an asserted property", "current", "hypothetical")
    if re.search(r"\bне\s+(?:оказывается|оказываются|попадает|попадают|застревает|застревают|комедия)\b[^,;.!?]{0,35}$", before, re.I):
        return finish("needs_review", "negated-supported-narrative-predicate-v1", "A supported predicate is explicitly negated; it is not a positive or a global denial", "current", "negated")
    # When a work's title is explicitly supplied after a style predicate it must
    # identify this card. Missing title does not resolve an explicitly different one.
    quoted_work = re.search(r"\b" + _WORK + r"\s*[«\"“]([^»\"”]+)[»\"”]", after, re.I)
    if quoted_work and base.normal(quoted_work[1]) != base.normal(title):
        return finish("rejected", "explicit-other-title-narrative-v1", "Literal work title differs from this card", "other", "unknown", "other_work")
    # Scope the denial to a grammatical assertion of this work. Local predicates
    # are preserved so 'no voiceover text' never denies every voice/sound subtype.
    negative = re.search(r"\b(?:нет|без|не\s+(?:содержит|использует|включает))\s+(?:(?:никаких|никакого|многочисленных)\s+)?$", before, re.I)
    if negative:
        current = bool(re.search(r"\b(?:в\s+(?:этой\s+)?(?:картине|ленте|сериале|фильме)|этот\s+фильм|эта\s+картина|фильм|сериал|повествование)\b", before, re.I))
        return finish("explicit_negative" if current else "needs_review", "scoped-current-narrative-denial-v1", "Explicit denial applies only to the named current predicate", "current" if current else "unknown", "negated")
    if re.search(r"\bне\s+$", before, re.I):
        return finish("needs_review", "local-unresolved-narrative-negation-v1", "Negated lexical phrase is not an accepted positive", "unknown", "negated")
    if flag == "has_flashback":
        if re.search(r"\b" + _WORK + r"\s+(?:состоит\s+из|содержит|использует)\s+(?:многочисленных\s+|многочисленные\s+)?$", before, re.I):
            return finish("accepted", "current-work-flashback-structure-v1", "The work explicitly consists of/uses flashbacks")
        if re.search(r"^\s*(?:веб-сериал|сериал|фильм)\b", norm, re.I) and re.search(r"\bэто\s+(?:своего\s+рода\s+)?$", before, re.I):
            return finish("accepted", "current-work-flashback-identification-v1", "Current web-series is explicitly identified as flashbacks")
    if flag == "has_voiceover":
        if re.match(r"^\s*$", before) and re.search(r"^\s*(?:рассказчика\s*[-—–:]|в\s+фильме\s+читает\b)", after, re.I):
            return finish("accepted", "current-work-narration-credit-v1", "Explicit offscreen narration credit of this work")
        if re.search(r"^\s*Чей\s*$", before, re.I) and re.match(r"\s+ведет\s+нас\b", after, re.I):
            return finish("accepted", "current-work-audience-narrator-v1", "The offscreen voice leads the audience through current scenes")
        if re.search(r"\bпомогает\s+(?:им\s+)?$", before, re.I) and re.match(r"\s+мамы\b", after, re.I):
            return finish("accepted", "explicit-animation-offscreen-guidance-v1", "Explicit offscreen maternal voice guides the current animation characters")
        if re.search(r"\b(?:этот\s+)?(?:фильм|сериал|повествование)\s+(?:содержит|использует|включает)\s*$", before, re.I):
            return finish("accepted", "current-work-explicit-voiceover-v1", "Explicit current-work voiceover predicate")
    if flag == "has_time_travel":
        if re.search(r"\b(?:в\s+центре\s+повествования\s*[-—–:]?\s*(?:опаснейшие\s+)?|(?:главная\s+тема\s+(?:фильма|сериала)|сюжет\s+(?:фильма|сериала))\s*[-—–:]?\s*(?:о\s+)?)$", before, re.I):
            return finish("accepted", "current-work-time-travel-subject-v1", "Explicit narrative subject is time travel")
        if re.search(r"\b(?:эпоха,\s*когда|мир,\s*где)\s*$", before, re.I) and re.match(r"\s+стал[иоа]?\b", after, re.I):
            return finish("accepted", "story-world-time-travel-v1", "Time travel is asserted as a reality of the current story world")
        if re.search(r"\b(?:главный\s+герой|герои|героиня|персонаж)\s+(?:отправляется\s+в|совершает|начинает)\s*$", before, re.I) or re.search(r"\b(?:главный\s+герой|герои|героиня|персонаж)\s*$", before, re.I) and literal.casefold().startswith("путешеству"):
            return finish("accepted", "explicit-character-time-travel-v1", "Actual current character time-travel action is explicit", scope="story_event")
    if flag == "has_time_loop":
        if re.search(r"\b(?:оказывается|оказываются|попадает|попадают|застревает|застревают|застрял[аи]?|попал[аи]?)\s+(?:в|во)\s+(?:(?:кошмарную|страшной|невыносимой)\s+)?$", before, re.I):
            return finish("accepted", "explicit-story-time-loop-v1", "Current story character explicitly enters/is trapped in a time loop", scope="story_event")
        if re.search(r"\b(?:выбраться|вырваться|спастись)\s+из\s+(?:(?:этой|невыносимой|страшной)\s+){0,2}$", before, re.I):
            return finish("accepted", "explicit-current-loop-escape-v1", "Escaping the explicitly named current loop presupposes that loop", scope="story_event")
        if re.search(r"\bфильм\b", before, re.I) and re.search(r"\bвариацией\s+на\s+тему\s*$", before, re.I):
            return finish("accepted", "explicit-current-loop-narrative-device-v1", "Current film is explicitly described as a variation on the time-loop device")
        if re.search(r"\b(?:фильм|сериал|повествование)\s+(?:использует|показывает|содержит)\s*$", before, re.I):
            return finish("accepted", "current-work-time-loop-v1", "Explicit current-work time-loop assertion")
    if flag == "has_surrealism":
        if _CREATOR.search(before) or re.search(r"\b(?:режиссеров-сюрреалистов|мастера\s+сюрреализма)\b", norm, re.I):
            return finish("rejected", "creator-surrealism-not-work-v1", "Creator's artistic identity is not a property of the current film", "other", "unknown", "creator_biography")
        if re.match(r"^\s*(?:немного\s+)?$", before, re.I) and re.match(r"\s+(?:(?:и|запутанный|авторский)\s+){0,4}(?:сюжет|фильм|мультфильм|комедия|драма)\b", after, re.I):
            return finish("accepted", "opening-current-surrealist-work-v1", "Opening description explicitly characterizes this work/plot as surrealist")
        if re.search(r"\bэто\s+(?:чистой\s+воды\s+)?$", before, re.I) and re.search(r"\b(?:фильме|картина|повествование|лента)\b", previous, re.I):
            previous_subject = re.search(r"\b(?:фильм|фильме|картина|картине|сериал|сериале|лента|ленте)\s*[«\"“]([^»\"”]+)[»\"”]", previous, re.I)
            if previous_subject and base.normal(previous_subject[1]) != base.normal(title):
                return finish("rejected", "deictic-foreign-work-style-v1", "Deictic style assertion refers back to an explicitly different work", "other", "unknown", "other_work")
            return finish("accepted", "deictic-current-surrealist-work-v1", "Explicit deictic style assertion refers back to the current work")
        named_after = re.match(r"\s+(?:историю|фильм|картину|ленту|комедию|драму|мультфильм)\s*[«\"“]([^»\"”]+)[»\"”]", after, re.I)
        if named_after and base.normal(named_after[1]) == base.normal(title):
            return finish("accepted", "named-current-surrealist-work-v1", "Style adjective modifies the literal current work title")
        if re.search(r"\b(?:этот\s+фильм|эта\s+картина|повествование)\s+(?:использует|содержит|сочетает)\s*$", before, re.I):
            return finish("accepted", "explicit-current-surrealism-v1", "Explicit current-work surrealism assertion")
    if flag == "has_absurdism":
        if re.search(r"\b(?:приказ|ситуаци[яию]|положени[ея]|требовани[ея]|решени[ея]|правил[ао])\b", after, re.I):
            return finish("rejected", "plot-absurdity-not-style-v1", "Absurd plot object/situation does not establish absurdist style", scope="plot_object_or_evaluation")
        if re.search(r"\b(?:комедия|юмор|театр)\s*$", before, re.I) or re.match(r"\s+юмор(?:ом|а)?\b", after, re.I):
            if re.search(r"\b(?:фильм|картина|лента|комедия|повествование)\b", before, re.I) or re.match(r"^\s*(?:комедия|театр)\s*$", before, re.I):
                return finish("accepted", "explicit-current-absurdist-style-v1", "Current material's comedy/humor is explicitly absurdist")
        if re.search(r"\b(?:фильм|сериал|повествование)\s+(?:использует|содержит|сочетает)\s*$", before, re.I):
            return finish("accepted", "explicit-current-absurdism-v1", "Explicit current-work absurdism assertion")
    if flag in ("has_surrealism", "has_absurdism"):
        named_before = re.search(r"\b(?:мультфильм|фильм|сериал|лента)\s*[«\"“]([^»\"”]+)[»\"”]", before, re.I)
        style_stories = re.match(r"^[\s,]*(?:(?:и|порой|но|неизменно|смешными|сюрреалистичными|абсурдными)[\s,]+)*историями\b", after, re.I)
        if named_before and base.normal(named_before[1]) == base.normal(title) and re.search(r"\bсвоими\b", before, re.I) and style_stories:
            return finish("accepted", "named-current-story-style-v1", "Literal current material is characterized by its stories' explicitly named style")
    return result


def description_evidence(iid, row):
    raw, title, facts = row.get("description", ""), row.get("title", ""), []
    previous = ""
    for offset, end, sentence in _sentences(raw):
        for flag in FLAGS:
            for match in PATTERNS[flag].finditer(sentence):
                relation = _relation(flag, sentence, match.start(), match.end(), title, previous)
                fact = {"item_id": int(iid), "flag": flag, "source_field": "description", "start": offset, "end": end,
                    "literal_span": raw[offset:end], "candidate_start": offset + match.start(), "candidate_end": offset + match.end(),
                    "candidate_literal": match.group(), "field_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "predicate": _predicate(flag, match.group()), "rules_version": CONTRACT, "source_known_at": None,
                    "source_available_at": None, "source_clock": "unknown", "temporal_mode": "retrospective_current",
                    "review_status": "deterministic_finite_rule", "centrality": "unknown", **relation}
                fact["decision"] = fact["status"]
                fact["candidate_id"] = base.canonical_hash({k: fact[k] for k in ("item_id", "flag", "source_field", "candidate_start", "candidate_end", "field_sha256")})
                facts.append(fact)
        previous = sentence
    return facts


def _review_decisions(annotation, items, mapping):
    """Accept only exact source-bound decisions on existing finite candidates."""
    if isinstance(annotation, (str, Path)):
        annotation = json.loads(Path(annotation).read_text(encoding="utf-8"))
    if annotation is None:
        return {}, None
    entries = annotation.get("reviewed_annotations", annotation.get("decisions", []))
    if not isinstance(entries, list):
        raise ValueError("Narrative reviewed decisions must be a list")
    decisions = {}
    for entry in entries:
        iid, flag = entry.get("item_id"), entry.get("flag")
        if isinstance(iid, bool) or not isinstance(iid, int) or str(iid) not in mapping or flag not in FLAGS or entry.get("source_field") != "description":
            raise ValueError("Reviewed narrative decision must name a train-known item and approved description flag")
        raw = items.get(str(iid), {}).get("description", "")
        cs, ce = entry.get("candidate_start"), entry.get("candidate_end")
        es, ee = entry.get("evidence_start"), entry.get("evidence_end")
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in (cs, ce, es, ee))
            or not 0 <= es <= cs < ce <= ee <= len(raw) or raw[cs:ce] != entry.get("candidate_literal")
            or raw[es:ee] != entry.get("evidence_literal") or hashlib.sha256(raw.encode("utf-8")).hexdigest() != entry.get("field_sha256")
            or not PATTERNS[flag].fullmatch(raw[cs:ce])):
            raise ValueError("Reviewed narrative literal/hash/offsets differ from source or finite candidate grammar")
        status = entry.get("decision", entry.get("expected_status"))
        if status not in ("accepted", "rejected", "needs_review", "explicit_negative") or not entry.get("review_id") or not entry.get("reason"):
            raise ValueError("Invalid reviewed narrative decision")
        relation, assertion = entry.get("work_relation"), entry.get("assertion_status")
        if relation not in ("current", "other", "component", "unknown") or assertion not in ("positive", "negated", "hypothetical", "unknown"):
            raise ValueError("Invalid reviewed narrative attribution/assertion")
        if status == "accepted" and (relation != "current" or assertion != "positive"):
            raise ValueError("Reviewed positive must assert the current work")
        if status == "explicit_negative" and (relation != "current" or assertion != "negated"):
            raise ValueError("Reviewed denial must name the current predicate")
        key = (iid, flag, cs, ce)
        if key in decisions:
            raise ValueError("Duplicate reviewed narrative candidate")
        decisions[key] = entry
    return decisions, base.canonical_hash(annotation)


def build_item_features(items, metadata, *, temporal_mode="retrospective_current", metadata_cutoff=None,
        reviewed_annotations=None, require_review_coverage=False):
    metadata, mapping = base._metadata(metadata)
    if temporal_mode not in ("retrospective_current", "historical_as_of"):
        raise ValueError("Unknown temporal mode")
    if temporal_mode == "historical_as_of" and metadata_cutoff is None:
        raise ValueError("Historical-as-of requires explicit cutoff; current metadata has unknown availability")
    adjudications, annotations_hash = _review_decisions(reviewed_annotations, items, mapping)
    consumed, unreviewed = set(), []
    columns = [flag + ":" + metric for flag in FLAGS for metric in METRICS]
    columns += ["narrative:state:" + state for state in STATES]
    matrix = np.zeros((metadata["vocab_size"], len(columns)), np.float32)
    replacement_columns = [flag + "_description_assertion:" + metric for flag in FLAGS for metric in ("positive", "explicit_negative", "unknown", "conflict")]
    replacement_columns += ["narrative_description:state:" + state for state in STATES]
    replacement = np.zeros((metadata["vocab_size"], len(replacement_columns)), np.float32)
    index, replacement_index = {c: i for i, c in enumerate(columns)}, {c: i for i, c in enumerate(replacement_columns)}
    evidence, support, states = [], {flag: Counter() for flag in FLAGS}, Counter()
    group_counts = defaultdict(Counter)
    for iid, token in mapping.items():
        row = items.get(iid, {})
        keyword = [dict(f, status=f["decision"], predicate=_predicate(f["flag"], f["literal_span"]), source_known_at=None,
            source_clock="unknown", source_available_at=None) for f in base.keyword_evidence(iid, row) if f["flag"] in FLAGS]
        desc = description_evidence(iid, row)
        for fact in desc:
            key = (fact["item_id"], fact["flag"], fact["candidate_start"], fact["candidate_end"])
            entry = adjudications.get(key)
            if entry:
                fact["automatic_status"] = fact["status"]
                fact["automatic_rule_id"] = fact["rule_id"]
                fact["automatic_scope"] = fact["scope"]
                status = entry.get("decision", entry.get("expected_status"))
                fact.update(status=status, decision=status, work_relation=entry["work_relation"], assertion_status=entry["assertion_status"],
                    scope=entry.get("scope", "reviewed_current_work_assertion" if status == "accepted" else "current_predicate_denial" if status == "explicit_negative" else "reviewed_" + entry["work_relation"] + "_scope"),
                    reason=entry["reason"], rule_id="independently-reviewed-finite-source-assertion-v1", review_status="independently_reviewed_literal",
                    review_id=entry["review_id"], reviewed_evidence_start=entry["evidence_start"], reviewed_evidence_end=entry["evidence_end"],
                    reviewed_evidence_literal=entry["evidence_literal"])
                consumed.add(key)
            else:
                unreviewed.append(key)
        facts = keyword + desc
        if temporal_mode == "historical_as_of":
            for fact in facts:
                if fact["status"] in ("accepted", "explicit_negative"):
                    fact.update(status="needs_review", decision="needs_review", reason="Source availability unknown at historical cutoff", temporal_mode=temporal_mode)
        by_flag = defaultdict(list)
        for fact in facts:
            by_flag[fact["flag"]].append(fact)
        family_status, desc_status = set(), set()
        for flag in FLAGS:
            local = by_flag[flag]
            positive_predicates = {f["predicate"] for f in local if f["status"] == "accepted"}
            negative_predicates = {f["predicate"] for f in local if f["status"] == "explicit_negative"}
            conflicted = positive_predicates & negative_predicates
            for fact in local:
                if fact["predicate"] in conflicted and fact["status"] in ("accepted", "explicit_negative"):
                    fact.update(status="conflict", decision="conflict", conflicting_predicate=fact["predicate"])
            statuses = {f["status"] for f in local}
            accepted_keywords = [f for f in local if f["source_field"] == "keywords" and f["status"] == "accepted"]
            accepted_desc = [f for f in local if f["source_field"] == "description" and f["status"] == "accepted"]
            negatives = [f for f in local if f["source_field"] == "description" and f["status"] == "explicit_negative"]
            encoded = {"keyword_positive": bool(accepted_keywords), "description_positive": bool(accepted_desc),
                "union_positive": bool(accepted_keywords or accepted_desc), "explicit_negative": bool(negatives),
                "description_unknown": not bool(accepted_desc), "union_unknown": not bool(accepted_keywords or accepted_desc),
                "conflict": bool(conflicted), "needs_review": "needs_review" in statuses, "rejected_candidate": "rejected" in statuses,
                "unresolved_metadata": any(f.get("metadata_prefix_status") == "unresolved" for f in local),
                "keyword_source_missing": not bool(row.get("keywords", "").strip()), "description_source_missing": not bool(row.get("description", "").strip())}
            for metric, value in encoded.items():
                matrix[token, index[flag + ":" + metric]] = value
                support[flag][metric] += int(value)
            for metric, value in (("positive", encoded["description_positive"]), ("explicit_negative", encoded["explicit_negative"]),
                ("unknown", encoded["description_unknown"]), ("conflict", encoded["conflict"])):
                replacement[token, replacement_index[flag + "_description_assertion:" + metric]] = value
            family_status |= statuses
            desc_status |= {f["status"] for f in local if f["source_field"] == "description"}
            for fact in local:
                group_counts[flag][fact["source_field"] + ":" + fact["status"]] += 1
        def state(observed, missing):
            return "conflict" if "conflict" in observed else "matched_accepted" if "accepted" in observed else "needs_review" if "needs_review" in observed else "matched_rejected" if observed else "source_missing" if missing else "present_no_evidence"
        overall = state(family_status, not (row.get("keywords", "").strip() or row.get("description", "").strip()))
        desc_state = state(desc_status, not row.get("description", "").strip())
        matrix[token, index["narrative:state:" + overall]] = 1
        replacement[token, replacement_index["narrative_description:state:" + desc_state]] = 1
        states[overall] += 1
        evidence.extend(facts)
    if consumed != set(adjudications):
        raise ValueError("Reviewed narrative decisions unmatched by the frozen finite detector")
    if require_review_coverage and (reviewed_annotations is None or unreviewed):
        raise ValueError("Finite narrative audit does not cover every extracted description candidate")
    groups = {"narrative_relations": {"scope": "item", "start": 0, "stop": len(columns), "columns": columns, "flags": list(FLAGS)}}
    coverage = {"train_known_items": len(mapping), "raw_items": len(items), "missing_rows": sum(iid not in items for iid in mapping),
        "flags": {flag: dict(support[flag]) for flag in FLAGS}, "family_states": dict(states),
        "evidence_counts": {flag: dict(group_counts[flag]) for flag in FLAGS}, "evidence_records": len(evidence),
        "reviewed_description_candidates": len(consumed), "unreviewed_description_candidates": len(unreviewed),
        "all_unknown_description_flags": [flag for flag in FLAGS if not support[flag]["description_positive"]],
        "semantics": "Evidence, not verified global absence or centrality. Explicit denials name local predicates; unknown remains true without accepted evidence.",
        "not_implemented": ["implicit flashbacks or unseen techniques inferred from generic memories/dreams", "unrestricted NLP or synonymous phrases beyond the frozen forms", "global negative inference from a local voiceover subtype denial", "historical metadata availability", "population precision, recall or recommendation benefit"]}
    provenance = {"contract": CONTRACT, "temporal_mode": temporal_mode, "metadata_cutoff": str(metadata_cutoff) if metadata_cutoff is not None else None,
        "source_known_at": None, "source_available_at": None, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "mapping_sha256": base.canonical_hash(mapping), "schema_sha256": base.canonical_hash({"columns": columns, "groups": groups}),
        "rules_sha256": base.canonical_hash(FORMS), "frozen_keyword_helper_sha256": BASE_SHA256, "evidence": evidence,
        "reviewed_annotations_sha256": annotations_hash, "require_review_coverage": require_review_coverage,
        "source_hierarchy": ["current scoped description assertion and exact content keyword are separate evidence", "title metadata is never an artistic-method assertion", "other work/creator/component/quote/hypothesis never becomes current whole-work evidence", "matching predicate contradiction is quarantined; independent voiceover subtype is retained"],
        "fit": {"scope": "unique train-known catalog only", "item_count": len(mapping), "learned_transforms": False},
        "safety": {"interactions_read": False, "outcomes_read": False, "splits_read": [], "network": False, "gpu": False, "special_rows": "all zero", "original197_modified": False}}
    if not np.isfinite(matrix).all() or matrix[:3].any() or replacement[:3].any():
        raise ValueError("Invalid narrative feature matrix")
    return {"matrix": matrix, "item_features": matrix, "columns": columns, "item_columns": columns, "groups": groups,
        "replacement_matrix": replacement, "replacement_native_columns": replacement_columns,
        "coverage": coverage, "provenance": provenance, "evidence": evidence}


def build(raw_dir, metadata, **kwargs):
    raw = Path(raw_dir).resolve()
    items_path = raw if raw.is_file() else raw / "items.csv"
    metadata_path = Path(metadata).resolve() if isinstance(metadata, (str, Path)) else None
    paths = {"items": items_path, "builder": Path(__file__).resolve(), "keyword_helper": _base_path.resolve()}
    if metadata_path:
        paths["metadata"] = metadata_path
    if isinstance(kwargs.get("reviewed_annotations"), (str, Path)):
        paths["reviewed_annotations"] = Path(kwargs["reviewed_annotations"]).resolve()
    hashes = {name: file_hash(path) for name, path in paths.items()}
    meta, _ = base._metadata(metadata)
    result = build_item_features(base.read_items(items_path), meta, **kwargs)
    if any(file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Narrative source changed during build")
    result["provenance"]["inputs"] = {name: {"path": path.as_posix(), "sha256": hashes[name]} for name, path in paths.items()}
    return result
