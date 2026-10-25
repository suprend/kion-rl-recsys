"""Literal relation extraction for the approved sport/activity/setting axes.

This is a conservative, retrospective metadata builder, not unrestricted NLP.
Zero positives mean no accepted evidence. All candidates keep literal offsets;
unknown, incidental, rejection, conflict and support remain distinct.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

_spec = importlib.util.spec_from_file_location("_ss_base", Path(__file__).with_name("kion_pack_semantics.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
CONTRACT = "kion-sports-settings-v1"

# Every expression is bounded as a word by _rx. Stem continuations are limited
# to a real lexical stem; e.g. tennis never matches the surname Теннисон.
SPORTS = {
    "football": r"футбол(?:а|е|ом|у)?|футболист\w*|футбольн\w*",
    "american_football": r"американск\w*\s+футбол(?:а|е|ом|у|ист\w*)?|американск\w*\s+футбольн\w*",
    "basketball": r"баскетбол(?:а|е|ом|у)?|баскетболист\w*|баскетбольн\w*",
    "boxing": r"бокс(?:а|е|ом|у|ер\w*|ёр\w*)|бокс",
    "martial_arts": r"смешанн\w*\s+единоборств\w*|единоборств\w*|мма|mma|карат[еэ]|кун[г]?[ -]?фу|дзюдо|джиу[ -]джитсу|тхэквондо|кикбоксинг\w*|боев\w*\s+искусств\w*",
    "running": r"бег(?:а|е|ом|у)?|бегун\w*|бегов\w*|марафонск\w*\s+бег\w*",
    "swimming": r"плавани\w*|пловец|пловц\w*|пловчих\w*",
    "cycling": r"велоспорт\w*|велогон\w*|велосипед\w*",
    "winter_sports": r"хокке\w*|фигурн\w*\s+катани\w*|лыжн\w*\s+(?:спорт\w*|гонк\w*)|биатлон\w*|сноуборд\w*|конькобеж\w*|бобслей\w*|керлинг\w*|кёрлинг\w*",
    "tennis": r"теннис(?:а|е|у|ом)?|теннисист\w*|теннисн\w*",
    "surfing": r"с[её]рфинг\w*|с[её]рфер\w*",
    "chess": r"шахмат\w*|гроссмейстер\w*",
}
ACTIVITIES = {
    "fitness": r"фитнес\w*|фитнесс\w*|аэробик\w*|воркаут\w*|бодибилдинг\w*|кардио[ -]тренировк\w*",
    "yoga": r"йог(?:а|и|е|ой|у|ою)?", "pilates": r"пилатес\w*",
    "stretching": r"растяжк\w*|стретчинг\w*|стрейчинг\w*",
    "dance_ballet": r"танц\w*|танец|балет\w*|хореограф\w*",
    "cooking": r"кулинар\w*|приготовлени\w*\s+(?:пищи|блюд\w*)|рецепт\w*|готовить|приготовить",
    "fishing": r"рыбалк\w*|рыболовств\w*|рыбн\w*\s+ловл\w*",
    "crafts": r"рукодели\w*|вязани\w*|вышивк\w*|оригами|поделк\w*|гончарн\w*|лепк\w*",
}
SETTINGS = {
    "school": r"школ(?:а|е|у|ы|ой|ах|ами)|школ[а-]интернат\w*",
    "university": r"университет\w*|колледж\w*|кампус\w*",
    "village": r"деревн\w*|деревушк\w*|сел(?:о|е|а|ом|у)|сельск\w*\s+местност\w*",
    "province": r"провинциальн\w*\s+(?:город\w*|городк\w*|местечк\w*|поселк\w*)|провинци\w*|глубинк\w*",
    "metropolis": r"больш\w*\s+город\w*|мегаполис\w*",
    "forest_jungle": r"лес(?:у|а|ом|е|ах|ов|ы)?|лесн\w*\s+чащ\w*|джунгл\w*",
    "island": r"остров(?:е|а|у|ом|ах|ов|ы)?",
    "coast": r"побереж\w*|пляж\w*|морск\w*\s+берег\w*",
    "mountain": r"гор(?:ы|ах|ами|у)|горн\w*\s+(?:местност\w*|район\w*|вершин\w*)",
    "desert": r"пустын(?:я|е|ю|и|ь|ях|ями)",
    "polar": r"арктик\w*|антарктид\w*|северн\w*\s+полюс\w*|южн\w*\s+полюс\w*",
    "hospital": r"больниц\w*|госпитал\w*|клиник\w*",
    "prison": r"тюрьм\w*|исправительн\w*\s+колони\w*",
    "hotel": r"отел(?:ь|е|я|ю|ем|и|ях)|гостиниц\w*|мотел\w*",
    "office": r"офис(?:е|а|ом|у|ы|ах)?",
    "military_base": r"военн\w*\s+(?:баз\w*|лагер\w*)",
    "summer_camp": r"(?:летн\w*\s+(?:пионерск\w*\s+)?|детск\w*\s+|пионерск\w*\s+)лагер\w*",
    "marine_watercraft": r"корабл\w*|яхт\w*|подводн\w*\s+лодк\w*|круизн\w*\s+лайнер\w*",
    "roadtrip": r"путешеств\w*\s+(?:по\s+дорог\w*\s+)?на\s+(?:автомобил\w*|машин\w*)|автопутешеств\w*",
    "space": r"космос\w*|открыт\w*\s+космос\w*|космическ\w*\s+(?:станци\w*|корабл\w*)|(?:далек\w*|чуж\w*|неизвестн\w*|друг\w*|инопланетн\w*)\s+планет\w*|лун[еуы]|марс[еа]?",
    "christmas": r"рождеств\w*", "newyear": r"нов\w*\s+год\w*|новогодн\w*",
}
FORMATS = ("broadcast", "instruction_training", "documentary", "fiction_with_sport_theme")
LABEL_STATES = ("positive", "unknown", "conflict", "candidate", "incidental", "rejected", "support")
_rx = lambda pattern: re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.I)
_SPORT_RX = {k: _rx(v) for k, v in SPORTS.items()}
_ACTIVITY_RX = {k: _rx(v) for k, v in ACTIVITIES.items()}
_SETTING_RX = {k: _rx(v) for k, v in SETTINGS.items()}

_OTHER = _rx(r"друг(?:ой|ом|ого)\s+(?:фильм\w*|сериал\w*)|фильмограф\w*|ранее\s+(?:снял\w*|сыграл\w*)|извест\w*\s+(?:по\s+рол\w*|ролью)|лауреат\w*|преми\w*|номинаци\w*")
_CAST = _rx(r"(?:актер|актёр|актриса|режиссер|режиссёр|исполнитель)\w*[^.!?]{0,90}(?:снимал\w*|снял\w*|сыграл\w*|родил\w*|вырос\w*|занимал\w*)")
_HYPOTHETICAL = _rx(r"если\s+бы|мечта\w*|планир\w*|намерева\w*|хочет|хотят|собира\w*|предстоит|угрожа\w*|грозит|может\s+быть|возможно")
_QUOTE = _rx(r"(?:герой|героиня|персонаж)\w*[^.!?]{0,65}(?:говорит|рассказывает|цитирует|смотрит|включает)\b")
_INSTRUCTION = _rx(r"выполняйте|повторяйте|включай\w*\s+видео|повторяй\w*\s+за|настрой\s+тренажер|вы\s+(?:научитесь|научились|можете\s+готовить)|учат\s+маленьких\s+зрителей|учебн\w*\s+(?:курс|программ\w*)|курс\s+«[^»]+»|программ\w*\s+(?:тренировок|упражнений|на\s+подкачку|продвинутого\s+фитнеса|для\s+тех\s+из\s+вас)|комплекс\w*\s+(?:упражнений|динамическ\w*\s+растяжки)|тренировк\w*\s+(?:направлен\w*|поможет|для)|упражнен\w*\s+(?:для|позвол\w*)|серию\s+упражнений|видеоурок\w*|дети\s+узнают[^.!?]+научатся|в\s+каждой\s+передаче[^.!?]+расскажет|объясняя\s+зрителям\s+процесс\s+приготовления|рецепт\w*[^.!?]+\b(?:вам|вы)\b|\b(?:вам|вы)\b[^.!?]+рецепт\w*")
_BROADCAST = _rx(r"трансляци\w*\s+(?:соревнован\w*|турнир\w*|матч\w*|бо\w*)|встреча\s+двух\s+лидеров\s+текущего\s+сезона|поединок\s+с\s+топовым\s+представителем")
_DOC = _rx(r"документальн\w*\s+(?:фильм|сериал|проект|цикл)|хроника\s+жизни")
_TOPIC = re.compile(r"\b(?:фильм(?:а|е)?|сериал(?:а|е)?(?:-исследование)?|драм\w*|комеди\w*|истори\w*|картин\w*|лент\w*|передач\w*|проект\w*|ток-шоу)\s*(?:,\s*)?(?:(?:котор\w*\s+)?(?:рассказыва\w*|повеству\w*|посвященн\w*)\s+)?(?:о|об|про)\s+",re.I)
_CAREER = re.compile(r"\b(?:истори\w*\s+жизни|истори\w*\s+взлетов\s+и\s+падений|истори\w*\s+соперничества|развивается\s+истори\w*|(?:создает|возглавляет|тренирует)\s+команд\w*\s+по|(?:готовится|готовятся)\s+к\s+(?:чемпионат\w*|соревнован\w*|турнир\w*))\b",re.I)
_LOCATION_ACTION = r"жив[её]т|живут|жили|проживает|проживают|работает|работают|учится|учатся|оказывается|оказываются|оказался|оказались|попадает|попадают|попал|попали|находится|находятся|проводит|проводят|служит|служат|приезжает|приезжают|прибывает|прибывают|поселяется|поселяются|поселились|отдыхает|отдыхают|скрывается|скрываются|содержится|содержатся|заключен|заключён|лечится|лечатся|застревает|застревают|застрял|застряли|путешествует|путешествуют|плывет|плывёт|плывут|отправляется|отправляются|знакомится|знакомятся"
_EVENT_ACTION = r"происходит|происходят|начинается|начинаются|разворачивается|разворачиваются|случается|случаются|совершается|совершаются|исчезает|исчезают|играют|живет|живёт|живут|работают|работает|погибает|погибают|ищут|находят|сражаются|борются|проводят"


def _sentences(text):
    # Keep literal offsets while protecting punctuation in quoted work titles
    # and decimals. Otherwise «Жги!» erases the preceding director/filmography
    # scope and 3.86 km splits an actual distance relation.
    protected=list(text)
    for quoted in re.finditer(r'«[^»]*»|"[^"]*"',text):
        for i in range(quoted.start(),quoted.end()):
            if protected[i] in '.!?': protected[i]='\ue000'
    for decimal in re.finditer(r'(?<=\d)\.(?=\d)',text): protected[decimal.start()]='\ue000'
    for start,end,_ in base._sentences(''.join(protected)):
        yield start,end,text[start:end]


def _fact(iid, label, field, source, start, end, rule, decision, **extra):
    return {"item_id": int(iid), "label": label, "source_field": field,
            "start": start, "end": end, "literal_span": source[start:end],
            "field_sha256": base.hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "rule_id": rule, "rules_version": CONTRACT, "decision": decision,
            "work_relation": "current", "relation_status": "central" if decision == "accepted" and label.startswith(("sport.", "activity.")) else "unknown",
            "source_known_at": None, **extra}


def _scope(sentence, match):
    """Return a reason for unsafe attribution; never a negative truth label."""
    if _OTHER.search(sentence) or _CAST.search(sentence):
        return "other_work_award_or_cast_biography"
    if _QUOTE.search(sentence) or re.search(r"(?:также\s+смотрите|рекомендуем\s+посмотреть|на\s+нашем\s+сайте)", sentence[:match.start()], re.I):
        return "quoted_or_recommended_other_content"
    prefix = sentence[:match.start()]
    if re.search(r"\b(?:без|нет|никаког\w*|не\s+(?:о|об|про|занимается|занимаются|любит|любит\s+играть\s+в))\s*$", prefix, re.I):
        return "negated_candidate"
    if re.search(r"\bне\s+(?:"+_LOCATION_ACTION+r"|"+_EVENT_ACTION+r"|играет|играют|занимается|занимаются|встречает|встречают|празднует|празднуют|отмечает|отмечают|поехал|поехали|отправился|отправились)\b[^,;.!?]*$", prefix, re.I):
        return "negated_relation"
    if _HYPOTHETICAL.search(prefix):
        return "hypothetical_intended_or_threatened"
    return None


def _documentary(row):
    # Exact raw genre is corroborating medium context, not a topic or format
    # assertion by itself. A bound subject relation is still required below.
    safe_cue=False
    for _,_,sentence in _sentences(row.get("description","")):
        cue=_DOC.search(sentence)
        if cue and re.fullmatch(r"\s*(?:Это\s+)?",sentence[:cue.start()],re.I) and not (_OTHER.search(sentence) or _CAST.search(sentence) or _QUOTE.search(sentence)):
            safe_cue=True
    return safe_cue or bool({"документальные","документальное"}&{base.normal(v) for v in row.get("genres", "").split(",")})


def _format_kind(row):
    text = row.get("description", "")
    fiction_genres={"драмы","комедии","мелодрамы","боевики","приключения","триллеры","ужасы","фантастика","мультфильмы","аниме"}
    fiction_context=bool(fiction_genres&{base.normal(v) for v in row.get("genres","").split(",")} or re.search(r"\b(?:сюжет\w*|главн\w*\s+геро\w*|художественн\w*\s+фильм|драм\w*|комеди\w*)\b",text,re.I))
    for _,_,sentence in _sentences(text):
        cue=_BROADCAST.search(sentence)
        if cue and not fiction_context and not (_OTHER.search(sentence) or _CAST.search(sentence) or _QUOTE.search(sentence)) and re.fullmatch(r"\s*(?:смотрите\s+)?",sentence[:cue.start()],re.I): return "broadcast"
    for _,_,sentence in _sentences(text):
        if _INSTRUCTION.search(sentence) and not (_OTHER.search(sentence) or _CAST.search(sentence) or _QUOTE.search(sentence) or re.search(r"\b(?:по\s+сюжету|в\s+фильме|главн\w*\s+геро\w*)\b",sentence,re.I)): return "instruction_training"
    if re.match(r"\s*(?:Три\s+беговые\s+тренировк\w*|Познакомьтесь\s+с\s+миром\s+единоборств\s+на\s+тренировке)\b",text,re.I) and re.search(r"\bот\s+Motify\b",text,re.I): return "instruction_training"
    if _documentary(row): return "documentary"
    if fiction_context: return "fiction_with_sport_theme"
    return None


def _sport_subject(sentence, match, category, title_match, row, kind):
    # Bind the relation to its clause, not a free window. A relative clause in
    # the literal current-work topic cue itself (film, which tells about...) is
    # protected; relatives introducing another character are split.
    topics=list(_TOPIC.finditer(sentence))
    left,right=0,len(sentence)
    for boundary in re.finditer(r";|,\s*(?:а|но|и|пока|где|чь\w*|котор\w*)\b|\b(?:на\s+заднем\s+плане|за\s+окном)\b",sentence,re.I):
        if any(topic.start()<=boundary.start() and boundary.end()<=topic.end() for topic in topics): continue
        if boundary.end()<=match.start(): left=boundary.end()
        elif boundary.start()>=match.end(): right=boundary.start(); break
    prefix,suffix=sentence[left:match.start()],sentence[match.end():right]
    clause=sentence[left:right]
    if re.search(r"\b(?:отце|матери|сыне|дочери|жене|муже|брате|сестре|друге)\s+(?:\w+\s+){0,3}$",prefix,re.I):
        return False,"relative_profession_not_current_subject_activity"
    incidental_clause=bool(re.search(r"\b(?:на\s+заднем\s+плане|в\s+кадре|пока|также\s+смотрите)\b|[,;]\s*(?:а|но|где|котор\w*)\b",prefix,re.I))
    if kind == "broadcast":
        return bool(_BROADCAST.search(clause) or re.search(r"\b(?:соревнован\w*|турнир\w*|поединок|встреча|дистанци\w*|\d+\s*км)\b", clause, re.I)), "broadcast_event_subject"
    if kind == "instruction_training":
        if incidental_clause: return False,"incidental_or_subordinate_clause"
        direct = _INSTRUCTION.search(clause) or re.search(r"\b(?:упражнен\w*|тренировк\w*|комплекс|методик\w*)\b", clause, re.I)
        # Practical title plus direct instruction establishes the programme's
        # subject; listed alternatives/equipment do not inherit this relation.
        return bool(_INSTRUCTION.search(clause) or re.match(r"\s*Познакомьтесь\s+с\s+миром\s+единоборств\s+на\s+тренировке\b",clause,re.I) or direct and title_match or re.search(r"\b(?:учат|учит|науч\w*|упражнен\w*\s+(?:по|для)|тренировк\w*\s+по)\b[^.!?]*$", prefix, re.I)), "taught_current_activity"
    if category == "martial_arts" and kind not in ("broadcast","documentary","instruction_training") and not re.search(r"\b(?:соревнован\w*|турнир\w*|чемпионат\w*|спортивн\w*|тренировк\w*|спортсмен\w*|карьер\w*)\b",sentence,re.I):
        return False,"combat_skill_without_sport_context"
    if re.search(r"\b(?:его|ее|её|их)\s+цель\s*[-–—:]\s*(?:пройти|победить|выиграть)\b",prefix,re.I):
        return True,"explicit_protagonist_competition_goal"
    topic = list(_TOPIC.finditer(prefix))
    if topic:
        topic_clause = prefix[topic[-1].end():]
        if not re.search(r",\s*(?:где|котор\w*|но|а\s)|\b(?:в\s+кадре|на\s+заднем\s+плане)\b", topic_clause, re.I):
            return True, "explicit_work_subject"
    if incidental_clause: return False,"incidental_or_subordinate_clause"
    career=_CAREER.search(prefix)
    if career and len(prefix[career.end():])<=160 and not re.search(r"[,;]\s*(?:а|но|где|котор\w*)\b|\b(?:брата|сестры|отца|матери|жениха|подруги|друга)\b", prefix[career.end():], re.I):
        return True, "protagonist_sport_goal_or_career"
    # Literal opening athlete identity is not sufficient: require a narrative
    # career/competition goal in the same sentence, not a relative's job.
    if title_match and match.start() < 90 and re.search(r"\b(?:побед\w*|чемпионат\w*|тренир\w*|соревнован\w*|турнир\w*|карьер\w*|выигр\w*|сборн\w*)\b", suffix, re.I) and not re.search(r"\b(?:жених|отец|брат|сестра|дочь|мать)\b", prefix, re.I):
        return True, "sport_career_plot_goal"
    return False, "unresolved_subject_centrality"


def sport_activity_evidence(iid, row):
    text, title = row.get("description", ""), row.get("title", "")
    facts, kind = [], _format_kind(row)
    for axis, expressions in (("sport", _SPORT_RX), ("activity", _ACTIVITY_RX)):
        for category, expression in expressions.items():
            label = f"{axis}.{category}.central_subject"
            title_match = expression.search(title)
            seen = False
            for start, end, sentence in _sentences(text):
                for match in expression.finditer(sentence):
                    seen = True
                    reason = _scope(sentence, match)
                    parent_project=re.search(r"\bпредставляет\s+(?:свой\s+)?новый\s+проект\b",sentence,re.I)
                    if parent_project and match.end()<parent_project.start(): reason="parent_program_not_current_project"
                    if category == "football" and _SPORT_RX["american_football"].search(sentence): reason = "american_football_separate_axis"
                    if re.search(r"\b(?:другими\s+видами\s+активности|напоминает|словно|под\s+названием|боксерск\w*\s+(?:трус\w*|груш\w*))\b", sentence, re.I): reason = "alternative_metaphor_object_or_name"
                    if category == "swimming" and re.search(r"\b(?:корабл\w*|на\s+лодк\w*|морск\w*\s+плавани\w*|кругосветн\w*\s+плавани\w*)\b", sentence, re.I): reason = "navigation_not_swimming"
                    if category == "running" and re.search(r"\b(?:лошадин\w*|конск\w*)\s*$",sentence[:match.start()],re.I): reason="horse_racing_not_human_running"
                    if category == "cycling" and not re.search(r"велогон|велоспорт|трениров|соревнован|чемпионат|гонщик|тренажер", sentence+title, re.I): reason = "transport_or_object_not_sport"
                    if category == "fitness" and re.search(r"рецепт|курин\w*\s+филе", title, re.I): reason = "recipe_not_fitness_subject"
                    central, rule = _sport_subject(sentence, match, category, title_match, row, kind)
                    decision = "rejected" if reason else "accepted" if central else "incidental" if re.search(r"\b(?:играет|играют|увлекается|увлекаются|занимается|занимаются|работает)\b", sentence, re.I) else "uncertain"
                    facts.append(_fact(iid, label, "description", text, start, end, rule, decision,
                        alias_literal=match.group(), alias_start=start+match.start(), alias_end=start+match.end(),
                        subtype=match.group(), reason=reason or rule, mention_role=("taught_activity" if kind == "instruction_training" else "current_subject") if decision == "accepted" else reason or ("observed_activity" if decision == "incidental" else "unresolved_mention"),
                        work_relation="other" if reason in ("other_work_award_or_cast_biography", "quoted_or_recommended_other_content") else "unknown" if reason else "current"))
                    if decision == "accepted" and kind and (axis == "sport" or kind != "fiction_with_sport_theme"):
                        facts.append(_fact(iid, f"sport_activity.format.{kind}", "description", text, start, end, rule+"_format", "accepted", supporting_subject=label, medium_context="explicit_or_raw_documentary_context" if kind == "documentary" else kind))
            if title_match and (not seen or kind == "instruction_training"):
                cues = [(a,b,s) for a,b,s in _sentences(text) if _INSTRUCTION.search(s) and not _OTHER.search(s) and not _QUOTE.search(s)]
                practical_title=bool(re.search(r"\b(?:основы|урок\w*|тренировк\w*|упражнен\w*|для\s+начинающих|дома|часть\s+\d+)\b",title,re.I))
                reason = "title_only_unresolved" if not cues or not practical_title else None
                if category == "fitness" and re.search(r"рецепт|курин\w*\s+филе", title, re.I): reason = "recipe_not_fitness_subject"
                accepted = bool(cues and kind == "instruction_training" and not reason)
                facts.append(_fact(iid, label, "title", title, 0, len(title), "practical_title_bound_to_instruction", "accepted" if accepted else "uncertain",
                    alias_literal=title_match.group(), alias_start=title_match.start(), alias_end=title_match.end(), subtype=title_match.group(), reason=reason,
                    supporting_description={"start":cues[0][0],"end":cues[0][1],"literal_span":cues[0][2],"field_sha256":base.hashlib.sha256(text.encode()).hexdigest()} if cues else None))
                if accepted:
                    a,b,_ = cues[0]
                    facts.append(_fact(iid, "sport_activity.format.instruction_training", "description", text, a,b,"practical_title_bound_to_instruction_format", "accepted", supporting_subject=label))
    return facts


def _setting_relation(sentence, match, category):
    prefix, suffix = sentence[:match.start()], sentence[match.end():]
    if category in ("christmas", "newyear"):
        # Festive props, songs, release dates or viewing recommendations are not
        # calendar time. The sentence must narrate an event at the holiday.
        if re.search(r"\b(?:премьера|вышел|выход|показ|смотреть|песня|песни|украшени\w*|фильм\s+на)\b", sentence, re.I): return False, "holiday_topic_without_story_time"
        event = re.search(r"\b(?:накануне|в\s+канун|под|перед|на)\s+(?:сам\w*\s+)?$", prefix, re.I) and re.search(r"\b(?:выясняется|происходит|оборачивается|случается|узнает|узнаёт|встречает|встречают|собирается|собираются|приезжает|приходит|получает|решает|пропадает|попадает)\b", sentence, re.I)
        celebration = re.search(r"\b(?:встречают|встречает|празднуют|празднует|отмечают|отмечает)\s*$", prefix, re.I)
        if re.search(r"\bне\s+(?:выясняется|происходит|оборачивается|случается|узнает|узнаёт|встречает|встречают|собирается|собираются|приезжает|приходит|получает|решает|пропадает|попадает)\b",suffix,re.I): event=False
        return bool(event or celebration), "story_event_time"
    if category == "roadtrip":
        return bool(re.search(r"\b(?:отправляется|отправляются|пускается|пускаются|путешествуя|путешествует|путешествуют)\b", prefix, re.I) or re.match(r"\s*(?:продолжается|начинается)", suffix, re.I)), "story_location"
    # A governing predicate + locative preposition and literal place. Unlike a
    # free word window, the intervening phrase admits adjectives only.
    adjectives = r"(?:(?:[А-Яа-яЁё-]+(?:ом|ем|ой|ей|ых|их|ую|юю)|своем|своём|этом|той|том|одном|одной|двух|трех|трёх)\s+){0,5}"
    before = re.search(r"\b(?:"+_LOCATION_ACTION+r")\s+(?:(?:вместе\s+с\s+\w+|без\s+родителей|всю\s+жизнь|каждый\s+день|уже\s+давно|долгое\s+время|несколько\s+лет|дни\s+и\s+ночи)\s+)?(?:в|на)\s+(?:борту\s+)?"+adjectives+r"$", prefix, re.I)
    action_frame = re.search(r"\b(?:действие|события|история)\b[^.!?]{0,90}\b(?:происходит|происходят|разворачивается|разворачиваются)\s+(?:в|на)\s+"+adjectives+r"$", prefix, re.I)
    # Fronted locative must immediately govern an event in the same clause.
    fronted = re.search(r"(?:^|[,;:]\s*)\s*(?:(?:а|но|и|однако)\s+)?(?:в|на)\s+"+adjectives+r"$",prefix,re.I) and re.match(r"(?:\s+«[^»]+»)?\s*(?:,\s*)?(?:\w+\s+){0,3}(?:"+_EVENT_ACTION+r")\b",suffix,re.I)
    fronted_clause=re.split(r"[,;.!?]",suffix,maxsplit=1)[0]
    if re.search(r"\bне\s+(?:"+_EVENT_ACTION+r")\b",fronted_clause,re.I) or _HYPOTHETICAL.search(fronted_clause): fronted=False
    return bool(before or action_frame or fronted), "story_location"


def setting_evidence(iid, row):
    text, facts = row.get("description", ""), []
    documentary = _documentary(row)
    for category, expression in _SETTING_RX.items():
        for start,end,sentence in _sentences(text):
            for match in expression.finditer(sentence):
                reason = _scope(sentence,match)
                prefix=sentence[:match.start()]
                if re.search(r"\b(?:родил\w*|вырос\w*|родом|съемк\w*|съёмк\w*|снимал\w*|снят\w*)\b[^,;.!?]*$",prefix,re.I): reason="origin_or_production_location"
                if category == "school" and re.match(r"\s+(?:повар\w*|танц\w*|искусств\w*|рисован\w*|рукодели\w*)",sentence[match.end():],re.I): reason="specialized_school_not_education_setting"
                if category == "school" and any(sport.match(sentence[match.end():].lstrip()) for sport in _SPORT_RX.values()): reason="sporting_school_or_tradition_not_general_school_location"
                if category == "school" and re.search(r"\b(?:балетн\w*|танцевальн\w*|кулинарн\w*|актерск\w*|актёрск\w*|художественн\w*|спортивн\w*)\s*$",prefix,re.I): reason="specialized_school_not_education_setting"
                if category == "school" and re.search(r"\b(?:жестк\w*|жёстк\w*|жизненн\w*)\s*$",prefix,re.I): reason="metaphor"
                if category == "marine_watercraft" and re.search(r"космическ\w*\s*$",prefix,re.I): reason="spacecraft_not_watercraft"
                if category == "marine_watercraft" and re.match(r"корабл",match.group(),re.I) and not re.search(r"\b(?:мор\w*|океан\w*|матрос\w*|палуб\w*|пират\w*|парус\w*|кораблекруш\w*|круизн\w*)\b",sentence,re.I): reason=reason or "watercraft_sense_unresolved"
                if category == "roadtrip" and re.match(r"\s+времени\b",sentence[match.end():],re.I): reason="time_machine_not_roadtrip"
                if category in ("christmas","newyear") and any(q.start()<=match.start() and match.end()<=q.end() for q in re.finditer(r'«[^»]*»|"[^"]*"',sentence)):
                    reason="quoted_title_or_dialogue_not_story_time"
                accepted,role = _setting_relation(sentence,match,category)
                # Explicit documentary subject lives in a separate namespace;
                # no documentary mention automatically becomes fictional action.
                if documentary and category not in ("christmas","newyear"):
                    topic=list(_TOPIC.finditer(prefix))
                    doc_topic=bool(topic and not re.search(r",\s*(?:где|котор\w*|но)\b",sentence[topic[-1].end():match.end()],re.I))
                    accepted=accepted or doc_topic
                    role="documentary_subject_location"
                label=f"setting.{category}.{role}"
                facts.append(_fact(iid,label,"description",text,start,end,"literal_governed_location_or_event_v1","rejected" if reason else "accepted" if accepted else "uncertain",
                    alias_literal=match.group(),alias_start=start+match.start(),alias_end=start+match.end(),
                    reason=reason or "bound_current_location" if accepted else reason or "no_bound_current_location",
                    relation_scope=reason or role,centrality="unknown",medium="documentary" if documentary else "narrated_story",
                    work_relation="other" if reason=="other_work_award_or_cast_biography" else "unknown" if reason else "current"))
    return facts


def _families():
    return {"sports": [f"sport.{c}.central_subject" for c in SPORTS],
            "activities": [f"activity.{c}.central_subject" for c in ACTIVITIES],
            "formats": [f"sport_activity.format.{kind}" for kind in FORMATS],
            "settings": [f"setting.{c}."+("story_event_time" if c in ("christmas","newyear") else "story_location") for c in SETTINGS],
            "documentary_settings": [f"setting.{c}.documentary_subject_location" for c in SETTINGS if c not in ("christmas","newyear")]}


def build_item_features(items,metadata,temporal_mode="retrospective_current",metadata_cutoff=None):
    metadata,mapping=base._metadata(metadata)
    if temporal_mode not in ("retrospective_current","historical_as_of"): raise ValueError("Unknown temporal_mode")
    if temporal_mode=="historical_as_of" and metadata_cutoff is None: raise ValueError("historical_as_of requires metadata_cutoff")
    families,columns,groups=_families(),[],{}
    for family,labels in families.items():
        start=len(columns)
        columns.extend(label+":"+state for label in labels for state in LABEL_STATES)
        columns.extend(f"{family}:state:{state}" for state in base.STATES)
        groups[family]={"scope":"item","start":start,"stop":len(columns),"columns":columns[start:],"labels":labels}
    index={c:i for i,c in enumerate(columns)}
    matrix=np.zeros((metadata["vocab_size"],len(columns)),dtype=np.float32)
    evidence=[]; support=Counter(); candidate_support=Counter(); states=defaultdict(Counter)
    for iid,token in mapping.items():
        row=items.get(iid,{})
        facts=sport_activity_evidence(iid,row)+setting_evidence(iid,row)
        if temporal_mode=="historical_as_of":
            for fact in facts:
                if fact["decision"]=="accepted": fact.update(decision="uncertain",reason="source_available_at_unknown")
        evidence.extend(facts)
        by_label=defaultdict(list)
        for fact in facts: by_label[fact["label"]].append(fact)
        for family,labels in families.items():
            family_decisions=set(); family_conflict=False
            for label in labels:
                decisions={f["decision"] for f in by_label[label]}
                # Different roles (e.g. incidental then central) can coexist.
                # Accepted vs rejected on the SAME literal assertion conflicts;
                # an unrelated false alias must not erase a valid assertion.
                by_span=defaultdict(set)
                for fact in by_label[label]: by_span[(fact["source_field"],fact["start"],fact["end"],fact.get("alias_start"),fact.get("alias_end"))].add(fact["decision"])
                conflict=any({"accepted","rejected"}<=v for v in by_span.values()) or "conflict" in decisions
                positive="accepted" in decisions and not conflict
                values={"positive":positive,"unknown":not positive,"conflict":conflict,"candidate":bool(decisions),"incidental":"incidental" in decisions,"rejected":"rejected" in decisions,"support":sum(f["decision"]=="accepted" for f in by_label[label]) if not conflict else 0}
                for state,value in values.items(): matrix[token,index[label+":"+state]]=value
                if positive: support[label]+=1
                if decisions: candidate_support[label]+=1
                family_decisions|=decisions; family_conflict|=conflict
            missing=not any(row.get(f,"").strip() for f in (("title","description") if family in ("sports","activities","formats") else ("description",)))
            state="conflict" if family_conflict else "matched_accepted" if "accepted" in family_decisions else "needs_review" if {"uncertain","incidental"}&family_decisions else "matched_rejected" if family_decisions else "source_missing" if missing else "present_no_evidence"
            matrix[token,index[f"{family}:state:{state}"]]=1; states[family][state]+=1
    return {"item_features":matrix,"columns":columns,"item_columns":columns,"groups":groups,
            "coverage":{"train_known_items":len(mapping),"accepted_item_support":dict(sorted(support.items())),"candidate_item_support":dict(sorted(candidate_support.items())),"family_state_counts":{k:dict(v) for k,v in states.items()},"evidence_records":len(evidence),"all_unknown_families":[family for family,labels in families.items() if not any(support[l] for l in labels)],"not_closed":["catalogue-wide semantic recall and precision not measured","unrestricted coreference and clause semantics","unobserved sport subtypes or setting aliases","historical source availability"],"interpretation":"Positive = accepted literal relation; unknown includes not observed, incidental, rejected and uncertain. Masks disambiguate them."},
            "provenance":{"contract":CONTRACT,"temporal_mode":temporal_mode,"metadata_cutoff":str(metadata_cutoff) if metadata_cutoff else None,"source_known_at":None,"retrieved_at":datetime.now(timezone.utc).isoformat(),"mapping_sha256":base.canonical_hash(mapping),"schema_sha256":base.canonical_hash({"columns":columns,"groups":groups}),"ontology_sha256":base.canonical_hash({"sports":SPORTS,"activities":ACTIVITIES,"settings":SETTINGS,"formats":FORMATS}),"evidence":evidence,"special_item_rows":"all zero","learned_transforms":False,"interactions_read":False}}


def build(raw_dir,metadata,**kwargs):
    path=Path(raw_dir)/"items.csv"; digest=base.file_hash(path)
    items=base.read_items(path)
    with path.open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            items[row["item_id"]]["genres"]=row.get("genres","")
    result=build_item_features(items,metadata,**kwargs)
    if digest!=base.file_hash(path): raise ValueError("Input changed during build")
    result["provenance"]["inputs"]={"items":{"path":path.resolve().as_posix(),"sha256":digest},"builder":{"path":Path(__file__).resolve().as_posix(),"sha256":base.file_hash(__file__)}}
    if isinstance(metadata,(str,Path)):
        result["provenance"]["inputs"]["metadata"]={"path":Path(metadata).resolve().as_posix(),"sha256":base.file_hash(metadata)}
    return result
