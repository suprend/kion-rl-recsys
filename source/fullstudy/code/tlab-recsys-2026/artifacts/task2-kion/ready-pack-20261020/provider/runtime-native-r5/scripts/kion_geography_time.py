"""Evidence-first local geography and story calendar, independent of frozen packs.

Only current-work asserted story/documentary subjects enter positive coordinates.
Country-of-production and unqualified text tags never establish those relations.
The fixed local gazetteer was selected from the full train-known catalogue census;
its explicit aliases and intentionally sparse parent edges are inspectable below.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import time

import numpy as np

_SPEC = importlib.util.spec_from_file_location("_geotime_base", Path(__file__).with_name("kion_pack_semantics.py"))
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
CONTRACT = "kion-geography-time-v1"
GAZETTEER_VERSION = "catalogue-census-local-gazetteer-20261020-v1"
PARENT_SOURCE = "explicit-conservative-local-geographic-containment-v1"
TEXT_FIELDS = ("description", "title", "title_orig", "keywords")
MODEL_ROLES = ("story_setting", "documentary_subject_location")
STATES = base.STATES


def _places():
    # Canonical names, complete inflection whitelist, and parent IDs. No stem
    # matching: university names and surnames cannot become country aliases.
    result = {}
    def add(kind, ident, name, aliases, parents=()):
        result[kind+":"+ident] = {"place_type":kind, "canonical_name":name,
            "aliases": tuple(dict.fromkeys((name, *aliases.split(";")))), "parents":tuple(parents)}
    countries = {
        "russia": ("Россия", "России;Россию;Россией"), "ussr": ("СССР", "Советский Союз;Советском Союзе;Советского Союза;Советскому Союзу"),
        "russian_empire": ("Российская империя", "Российской империи;Российскую империю"),
        "usa": ("США", "Соединенные Штаты;Соединенных Штатах;Соединенных Штатов;Соединенные Штаты Америки"),
        "uk": ("Великобритания", "Великобритании;Великобританию;Британии;Британию"),
        "france": ("Франция", "Франции;Францию"), "germany": ("Германия", "Германии;Германию"),
        "italy": ("Италия", "Италии;Италию"), "spain": ("Испания", "Испании;Испанию"),
        "japan": ("Япония", "Японии;Японию"), "china": ("Китай", "Китае;Китая;Китаю"),
        "india": ("Индия", "Индии;Индию"), "indonesia": ("Индонезия", "Индонезии;Индонезию"),
        "afghanistan": ("Афганистан", "Афганистане;Афганистана"), "iraq": ("Ирак", "Ираке;Ирака"),
        "iran": ("Иран", "Иране;Ирана"), "persia": ("Персия", "Персии;Персию"),
        "mexico": ("Мексика", "Мексике;Мексики;Мексику"), "canada": ("Канада", "Канаде;Канады;Канаду"),
        "australia": ("Австралия", "Австралии;Австралию"), "brazil": ("Бразилия", "Бразилии;Бразилию"),
        "thailand": ("Таиланд", "Таиланде;Таиланда;Таиланду"), "greece": ("Греция", "Греции;Грецию"),
        "turkey": ("Турция", "Турции;Турцию"), "israel": ("Израиль", "Израиле;Израиля"),
        "kazakhstan": ("Казахстан", "Казахстане;Казахстана"), "ireland": ("Ирландия", "Ирландии;Ирландию"),
        "norway": ("Норвегия", "Норвегии;Норвегию"), "sweden": ("Швеция", "Швеции;Швецию"),
        "peru": ("Перу", "Перу"), "chile": ("Чили", "Чили"), "honduras": ("Гондурас", "Гондурасе;Гондураса"),
        "bulgaria": ("Болгария", "Болгарии;Болгарию"), "latvia": ("Латвия", "Латвии;Латвию"),
        "ukraine": ("Украина", "Украине;Украины;Украину"), "south_africa": ("Южная Африка", "Южной Африке;Южную Африку;ЮАР"),
        "libya": ("Ливия", "Ливии;Ливию"), "egypt": ("Египет", "Египте;Египта"),
        "cuba": ("Куба", "Кубе;Кубы;Кубу"), "new_zealand": ("Новая Зеландия", "Новой Зеландии;Новую Зеландию"),
        "armenia": ("Армения", "Армении;Армению"),
    }
    for ident, (name, aliases) in countries.items(): add("country",ident,name,aliases)
    add("region","england","Англия","Англии;Англию")
    add("region","scotland","Шотландия","Шотландии;Шотландию")
    for ident,name,aliases in (
        ("africa","Африка","Африке;Африки;Африку"),("europe","Европа","Европе;Европы;Европу"),
        ("asia","Азия","Азии;Азию"),("north_america","Северная Америка","Северной Америке;Северную Америку"),
        ("south_america","Южная Америка","Южной Америке;Южную Америку"),
        ("central_america","Центральная Америка","Центральной Америке;Центральную Америку"),
        ("americas","Америка","Америке;Америки;Америку"),
        ("latin_america","Латинская Америка","Латинской Америке;Латинскую Америку"),
        ("south_east_asia","Юго-Восточная Азия","Юго-Восточной Азии;Юго-Восточную Азию"),
        ("siberia","Сибирь","Сибири;Сибирью;сибирской глубинке;сибирской деревне"),
        ("caucasus","Кавказ","Кавказе;Кавказа"),("polynesia","Полинезия","Полинезии;Полинезию"),
        ("tibet","Тибет","Тибете;Тибета"),("crimea","Крым","Крыму;Крыма"),
        ("far_east","Дальний Восток","Дальнем Востоке;Дальнего Востока"),
        ("antarctica","Антарктида","Антарктиде;Антарктиды"),("arctic","Арктика","Арктике;Арктики"),
        ("wild_west","Дикий Запад","Диком Западе;Дикого Запада"),
        ("california","Калифорния","Калифорнии;Калифорнию"),("texas","Техас","Техасе;Техаса"),
        ("alabama","Алабама","Алабаме;Алабаму;Алабамы"),("florida","Флорида","Флориде;Флориду;Флориды"),
        ("montana","Монтана","Монтане;Монтаны;Монтану"),
        ("alaska","Аляска","Аляске;Аляски;Аляску"),("hawaii","Гавайи","Гавайях;Гавайев"),
        ("west_papua","Западное Папуа","Западном Папуа;Западного Папуа"),
        ("new_caledonia","Новая Каледония","Новой Каледонии;Новую Каледонию"),
        ("french_polynesia","Французская Полинезия","Французской Полинезии;Французскую Полинезию"),
    ): add("region",ident,name,aliases)
    cities = [
        ("moscow","Москва","Москве;Москвы;Москву;Москвой",()),
        ("saint_petersburg","Санкт-Петербург","Санкт-Петербурге;Санкт-Петербурга;Петербург;Петербурге;Петербурга;Питер;Питере",()),
        ("leningrad","Ленинград","Ленинграде;Ленинграда",()),("petrograd","Петроград","Петрограде;Петрограда",()),
        ("new_york","Нью-Йорк","Нью-Йорке;Нью-Йорка",("country:usa",)),
        ("los_angeles","Лос-Анджелес","Лос-Анджелесе;Лос-Анджелеса",("region:california",)),
        ("san_francisco","Сан-Франциско","Сан-Франциско",("region:california",)),
        ("chicago","Чикаго","Чикаго",("country:usa",)),("miami","Майами","Майами",("region:florida",)),
        ("las_vegas","Лас-Вегас","Лас-Вегасе;Лас-Вегаса;Вегас;Вегасе",("country:usa",)),
        ("new_orleans","Новый Орлеан","Новом Орлеане;Нового Орлеана",("country:usa",)),
        ("washington_dc","Вашингтон, округ Колумбия","Вашингтоне, округ Колумбия",("country:usa",)),
        ("paris","Париж","Париже;Парижа;Парижу",("country:france",)),
        ("london","Лондон","Лондоне;Лондона;Лондону",()),
        ("rome","Рим","Риме;Рима",("country:italy",)),("venice","Венеция","Венеции;Венецию",("country:italy",)),
        ("berlin","Берлин","Берлине;Берлина",()),("madrid","Мадрид","Мадриде;Мадрида",("country:spain",)),
        ("barcelona","Барселона","Барселоне;Барселону;Барселоны",("country:spain",)),
        ("tokyo","Токио","Токио",("country:japan",)),("bangkok","Бангкок","Бангкоке;Бангкока",("country:thailand",)),
        ("mumbai","Мумбаи","Мумбаи",("country:india",)),("wuhan","Ухань","Ухане;Уханя",("country:china",)),
        ("amsterdam","Амстердам","Амстердаме;Амстердама",()),("hong_kong","Гонконг","Гонконге;Гонконга",()),
        ("kyiv","Киев","Киеве;Киева",()),("odesa","Одесса","Одессе;Одессы;Одессу",()),
        ("almaty","Алматы","Алматы",()),("riga","Рига","Риге;Риги;Ригу",()),
        ("sochi","Сочи","Сочи",()),("yeniseysk","Енисейск","Енисейске;Енисейска",()),
        ("lesosibirsk","Лесосибирск","Лесосибирске;Лесосибирска",()),
    ]
    for ident,name,aliases,parents in cities: add("city",ident,name,aliases,parents)
    for ident,name,aliases,parents in (
        ("borneo","Борнео","Борнео",()),("sumatra","Суматра","Суматре;Суматры;Суматру",("country:indonesia",)),
        ("bali","Бали","Бали",("country:indonesia",)),("tahiti","Таити","Таити",()),
        ("corsica","Корсика","Корсике;Корсики;Корсику",("country:france",)),
        ("alps","Альпы","Альпах;Альп",()),("sierra_nevada","Сьерра-Невада","Сьерра-Неваде;Сьерра-Невады",()),
        ("volga","Волга","Волге;Волги;Волгу",()),("neva","Нева","Неве;Невы;Неву",()),
        ("pacific","Тихий океан","Тихом океане;Тихого океана",()),
        ("mediterranean","Средиземное море","Средиземном море;Средиземного моря",()),
        ("baikal","Байкал","Байкале;Байкала;Байкалу",()),
    ): add("natural_place",ident,name,aliases,parents)
    # Parents which are insensitive to the ambiguous historical setting. Russian,
    # German, British cities and disputed regions intentionally lack modern edges.
    for ident in ("california","texas","alabama","florida","alaska","hawaii","montana"):
        result["region:"+ident]["parents"] = ("country:usa",)
    return result


GAZETTEER = _places()


def _alias_pattern(alias):
    return re.escape(alias).replace(r"\ ",r"\s+").replace(r"\-",r"\s*[-–—]\s*").replace("е","[её]").replace("Е","[ЕЁ]")


_ALIAS_TO_PLACE = defaultdict(set)
for _pid,_entry in GAZETTEER.items():
    for _alias in _entry["aliases"]: _ALIAS_TO_PLACE[base.normal(re.sub(r"\s*[-–—]\s*","-",_alias))].add(_pid)
_GEO = re.compile(r"(?<!\w)(?:"+"|".join(_alias_pattern(a) for a in sorted({a for p in GAZETTEER.values() for a in p["aliases"]}, key=len, reverse=True))+r")(?!\w)",re.I)
_OTHER = re.compile(r"\b(?:друг(?:ой|ого|ом|ая|ую)|предыдущ(?:ий|его|ем)|чуж(?:ой|ого|ом)|любим(?:ый|ого|ом))\s+(?:фильм\w*|сериал\w*|картин\w*)\b|\bтакже\s+смотрите\b",re.I)
_BIO = re.compile(r"\b(?:актер\w*|акт[её]р\w*|актрис\w*|режиссер\w*|режисс[её]р\w*|писател\w*|автор\w*|сценарист\w*)\b",re.I)
_SHOOT = re.compile(r"\b(?:снимал\w*|снима[лн]\w*|съем\w*|съ[её]м\w*|снят\w*|снял\w*)\b",re.I)
_ACTION = re.compile(r"\b(?:действие|события|приключени\w*|история)\b[^.!?;]{0,65}?\b(?:происход\w*|разворач\w*|разверн[её]\w*|перенос\w*|начина\w*)\b",re.I)
_PLOT = re.compile(r"\b(?:жив[её]т|живут|жила?|жили|работает|работают|живет|прожива\w*|теряется|потерял\w*|исчез\w*|скрыва\w*|оказыва\w*|попада\w*|прибыва\w*|приезжа\w*|возвраща\w*|надвига\w*)\b",re.I)
_ORIGIN = re.compile(r"\b(?:родом|родил\w*|вырос\w*|детств\w*|урожен\w*|домработниц\w*\s+из|выходец\w*)\b",re.I)
_TRAVEL = re.compile(r"\b(?:отправля\w*|отправил\w*|планиру\w*|мечт\w*|собира\w*|поед\w*|поехал\w*|едет|едут|летит|направля\w*)\b",re.I)
_UNCERTAIN = re.compile(r"\b(?:возможно|вероятно|предположительно|якобы|если|могло\s+бы|может\s+быть|кажется)\b",re.I)
_DOC = re.compile(r"\bдокументальн\w*\s+(?:фильм\w*|сериал\w*|лент\w*|проект\w*)\b",re.I)
_NATURE = re.compile(r"\b(?:орангутанг\w*|шимпанзе|дельфин\w*|пингвин\w*|горилл\w*|животн\w*|природ\w*)\b",re.I)


def sentences(text):
    # Preserve offsets while protecting initials and calendar abbreviations/dates.
    masked = list(text)
    for m in re.finditer(r"\bдо\s+н\.\s*э\.|\b(?:гг?|вв?)\.(?=\s|$)|\b\d{1,2}\.\d{1,2}\.\d{3,4}\b|\b[А-ЯЁ]\.\s*(?=[А-ЯЁ])",text,re.I):
        for i in range(m.start(),m.end()):
            if masked[i] == ".": masked[i] = "\ue000"
    for quoted in re.finditer(r"«[^»]*»|\"[^\"]*\"",text):
        for i in range(quoted.start(),quoted.end()):
            if masked[i] in ".!?":masked[i]="\ue000"
    for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)","".join(masked)):
        if text[m.start():m.end()].strip(): yield m.start(),m.end(),text[m.start():m.end()]


def _record(iid, family, field, text, start, end, sent_start, sent_end, rule, **extra):
    fact = {"item_id":int(iid),"family":family,"source_field":field,"start":start,"end":end,
        "evidence_literal":text[start:end],"literal_span":text[start:end],
        "sentence_start":sent_start,"sentence_end":sent_end,"sentence_literal":text[sent_start:sent_end],
        "field_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest(),"rule_id":rule,
        "rule_version":CONTRACT,"source_known_at":None,"work_relation":"current",
        "assertion_status":"asserted","decision":"needs_review","centrality":"unknown", **extra}
    fact["evidence_id"] = base.canonical_hash({k:fact[k] for k in ("item_id","family","source_field","start","end","field_sha256","rule_id")})[:24]
    return fact


def _clause(sentence, position):
    # A coordinated contrast changes polarity but preserves the preceding role.
    bounds = [m.end() for m in re.finditer(r"[;]|,\s*(?:но|однако|а\s+не|зато)\b",sentence[:position],re.I)]
    return sentence[bounds[-1]:] if bounds else sentence


def _negated(sentence, start, end):
    prefix = sentence[max(0,start-45):start]
    return bool(re.search(r"\b(?:не|вместо|без)(?:\s+(?:в|на|к|из|около|примерно|разумеется))?\s*$",prefix,re.I)
        or re.search(r"\b(?:действие|события)\b.{0,35}\bне\s+(?:происход\w*|разворач\w*)",sentence[:end],re.I))


def _documentary(row):
    desc = row.get("description","")
    opening = next(sentences(desc),(0,0,""))[2]
    declared = bool(_DOC.search(opening) and not _OTHER.search(opening))
    # The local nature catalogue has no genre field. Expository organism titles
    # plus matching biological description are a documented narrow subject rule.
    nature = bool(re.search(r"\b(?:орангутанг\w*|шимпанзе|горилл\w*)\b",row.get("title",""),re.I) and _NATURE.search(desc)
        and not re.search(r"\b(?:геро\w*|персонаж\w*|мультфильм|комеди\w*|ужас\w*)\b",opening,re.I))
    # Full-catalogue census revealed an expository travel/nature/ethnography
    # series whose titles end in '. <literal place>'. Its descriptions explain
    # species, tribes, traditions or landscape; neither the title punctuation
    # alone nor the generic word 'природа' establishes documentary format.
    suffix_place=any(_GEO.fullmatch(part.strip()) for part in row.get("title","").split(".")[1:])
    expository=bool(re.search(r"\b(?:плем\w*|народ\w*|традиц\w*|ритуал\w*|обита\w*|вид\w*|самк\w*|самец\w*|детеныш\w*|млекопитающ\w*|хищник\w*|л[её]в\w*|льв\w*|остров\w*|побереж\w*|природ\w*|ландшафт\w*|оз[её]р\w*|река|реки|лед\w*|растен\w*|культур\w*|музык\w*)\b",desc,re.I))
    fictional=bool(re.search(r"\b(?:мультфильм|главн\w*\s+геро\w*|героин\w*|комеди\w*|триллер|сценари\w*|романтич\w*)\b",desc,re.I))
    return declared or nature or (suffix_place and expository and not fictional)


def geography_evidence(iid,row):
    facts=[]
    doc = _documentary(row)
    for field in TEXT_FIELDS:
        text=row.get(field,"") or ""
        for a,b,sent in sentences(text):
            matches=list(_GEO.finditer(sent))
            for m in matches:
                key=base.normal(re.sub(r"\s*[-–—]\s*","-",m.group()))
                ids=_ALIAS_TO_PLACE[key]
                clause=_clause(sent,m.start())
                before=sent[:m.start()]
                local=before[max(0,len(before)-100):]
                shooting_match=list(_SHOOT.finditer(before))
                action_match=list(_ACTION.finditer(before))
                shooting=bool(shooting_match and (not action_match or shooting_match[-1].start()>action_match[-1].start()))
                other=bool(_OTHER.search(sent))
                bio=bool(_BIO.search(before) and not re.search(r"\b(?:геро\w*|персонаж\w*)\b",local,re.I))
                def lastpos(pattern):
                    found=list(pattern.finditer(local));return found[-1].start() if found else -1
                origin_pos,travel_pos,actual_pos=lastpos(_ORIGIN),lastpos(_TRAVEL),lastpos(_PLOT)
                origin=origin_pos>=0 and origin_pos>max(travel_pos,actual_pos)
                travel=travel_pos>=0 and travel_pos>actual_pos
                actual=actual_pos>=0
                action=bool(_ACTION.search(before))
                locative=bool(re.search(r"\b(?:в|во|на|под|около|из|до|у|к)\s+(?:[а-яё-]+\s+){0,3}$",local,re.I))
                opening_place=(a == 0 and not before.strip() and bool(re.match(r"\s*[,.:–—]|\s*$",sent[m.end():])))
                role="mention"
                if shooting:role="shooting_location"
                elif bio:role="cast_biography"
                elif re.search(r"\b(?:работа\w*|служ\w*)\s+на\s*$",local,re.I):role="affiliation"
                elif actual and re.search(r"\bиз\s*$",local,re.I):role="travel_origin"
                elif origin:role="character_origin"
                elif travel:role="travel_destination"
                elif doc and (field=="title" or (field=="description" and (locative or action or re.search(r"\b(?:о|об|истори\w*|природ\w*|посвящ\w*)\b",local,re.I)))):role="documentary_subject_location"
                elif field=="description" and (action or (actual and locative) or opening_place):role="story_setting"
                neg=_negated(sent,m.start(),m.end())
                alternative=any(re.search(r"\bили\b",sent[min(m.end(),n.end()):max(m.start(),n.start())],re.I) for n in matches if n is not m)
                uncertain=bool(_UNCERTAIN.search(clause)) or alternative
                if field in ("keywords","title_orig"):role="mention"
                if field=="title" and not doc:role="mention"
                non_earth=bool("region:europe" in ids and re.search(r"\b(?:спутник\w*|Юпитер\w*)\b",local,re.I))
                status="rejected" if other or bio or neg or non_earth else "needs_review" if uncertain or len(ids)!=1 or role=="mention" else "accepted"
                for pid in sorted(ids):
                    entry=GAZETTEER[pid]
                    fact=_record(iid,"geography",field,text,a+m.start(),a+m.end(),a,b,"literal-place-clause-role-v1",
                        place_id=pid,place_type=entry["place_type"],canonical_name=entry["canonical_name"],
                        role=role,role_status="explicit_relation" if role!="mention" else "unresolved_mention",
                        entity_resolution_status="rejected_non_earth_homonym" if non_earth else "resolved_local_full_alias" if len(ids)==1 else "ambiguous_alias",
                        entity_confidence="literal_whitelist",role_confidence="clause_rule" if role!="mention" else "unresolved",
                        direct_or_inferred="direct",gazetteer_version=GAZETTEER_VERSION,decision=status,
                        work_relation="other" if other else "current",assertion_status="negated" if neg else "uncertain" if uncertain else "asserted",
                        alternative=alternative,reason="other work or biography" if other or bio else "local negation" if neg else "alternative or uncertain clause" if uncertain else "role-bound literal" if status=="accepted" else "mention does not establish story role")
                    facts.append(fact)
                    if status=="accepted" and role in MODEL_ROLES:
                        queue=list(entry["parents"]); seen={pid}
                        while queue:
                            parent=queue.pop(0)
                            if parent in seen:continue
                            seen.add(parent); queue.extend(GAZETTEER[parent]["parents"])
                            inferred=dict(fact,place_id=parent,place_type=GAZETTEER[parent]["place_type"],canonical_name=GAZETTEER[parent]["canonical_name"],
                                direct_or_inferred="inferred",parent_mapping_source=PARENT_SOURCE,parent_mapping_version=GAZETTEER_VERSION,
                                parent_assertion_id=fact["evidence_id"],inferred_from=pid)
                            inferred["evidence_id"]=base.canonical_hash([fact["evidence_id"],parent])[:24]
                            facts.append(inferred)
    # Qualified Indonesian forest is direct evidence for the country, whereas
    # the Sumatra/Borneo disjunction above stays uncertain for each island.
    text=row.get("description","") or ""
    if doc:
        for a,b,sent in sentences(text):
            for m in re.finditer(r"\bиндонезийск(?:их|ие|ом|ого|ой)\s+(?:лес\w*|остров\w*|джунгл\w*)",sent,re.I):
                neg=_negated(sent,m.start(),m.end())
                excluded=bool(_OTHER.search(sent) or _SHOOT.search(sent[:m.start()]) or _BIO.search(sent[:m.start()]))
                uncertain=bool(_UNCERTAIN.search(sent))
                facts.append(_record(iid,"geography","description",text,a+m.start(),a+m.end(),a,b,"explicit-indonesian-geographic-qualifier-v1",
                    place_id="country:indonesia",place_type="country",canonical_name="Индонезия",role="documentary_subject_location",
                    direct_or_inferred="direct",entity_resolution_status="explicit_qualified_geographic_alias",role_status="expository_nature_subject",
                    entity_confidence="literal_whitelist",role_confidence="clause_rule",gazetteer_version=GAZETTEER_VERSION,
                    decision="rejected" if neg or excluded else "needs_review" if uncertain else "accepted",
                    work_relation="other" if _OTHER.search(sent) else "current",assertion_status="negated" if neg else "uncertain" if uncertain else "asserted"))
    return facts


MONTHS = {"январ":1,"феврал":2,"март":3,"апрел":4,"ма":5,"июн":6,"июл":7,"август":8,"сентябр":9,"октябр":10,"ноябр":11,"декабр":12}
MONTH = r"(?:январ(?:[ьяею]|[её]м)|феврал(?:[ьяею]|[её]м)|март(?:а|е|ом)?|апрел(?:[ьяею]|[её]м)|ма[йяе]|июн(?:[ьяею]|[её]м)|июл(?:[ьяею]|[её]м)|август(?:а|е|ом)?|сентябр(?:[ьяею]|[её]м)|октябр(?:[ьяею]|[её]м)|ноябр(?:[ьяею]|[её]м)|декабр(?:[ьяею]|[её]м))"
ORDINALS = {"перв":1,"втор":2,"трет":3,"четверт":4,"пят":5,"шест":6,"седьм":7,"восьм":8,"девят":9,"десят":10,
    "одиннадцат":11,"двенадцат":12,"тринадцат":13,"четырнадцат":14,"пятнадцат":15,"шестнадцат":16,
    "семнадцат":17,"восемнадцат":18,"девятнадцат":19,"двадцат":20,"тридцат":30}
ORD = r"(?:двадцать\s+(?:перв|втор|трет|четверт|пят|шест|седьм|восьм|девят)\w*|(?:"+"|".join(sorted(ORDINALS,key=len,reverse=True))+r")\w*)"
CENT = r"(?:[IVXL]{1,8}|[1-9]\d?|"+ORD+r")"
QUAL = r"(?:начал[еоа]|середин[еаы]|кон(?:ец|це|ца))"
ERA = r"(?:\s+до\s+(?:нашей\s+эры|н\.\s*э\.))?"
UNIT = r"(?:год(?:ах|ов|у|а|ы)?\b|гг?\.)"
_TEMP_PATTERNS = [
    ("date_range", re.compile(r"(?<!\d)(?P<day>\d{1,2})(?:\s+(?P<month>"+MONTH+r"))?(?:\s+(?P<y>[1-9]\d{2,3})(?:\s*"+UNIT+r")?)?\s*(?:[-–—]|по)\s*(?P<day2>\d{1,2})\s+(?P<month2>"+MONTH+r")\s+(?P<y2>[1-9]\d{2,3})(?:\s*"+UNIT+r")?"+ERA,re.I)),
    ("month_range", re.compile(r"(?<!\w)(?P<month>"+MONTH+r")(?:\s+(?P<y>[1-9]\d{2,3})(?:\s*"+UNIT+r")?)?\s*(?:[-–—]|по)\s*(?P<month2>"+MONTH+r")\s+(?P<y2>[1-9]\d{2,3})(?:\s*"+UNIT+r")?"+ERA,re.I)),
    ("date", re.compile(r"(?<!\d)(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<y>\d{3,4})\b"+ERA,re.I)),
    ("date", re.compile(r"(?<!\w)(?P<day>\d{1,2})\s+(?P<month>"+MONTH+r")\s+(?P<y>[1-9]\d{0,3})(?:\s*"+UNIT+r")?"+ERA,re.I)),
    ("month", re.compile(r"(?<!\w)(?P<month>"+MONTH+r")\s+(?P<y>[1-9]\d{0,3})(?:\s*"+UNIT+r")?"+ERA,re.I)),
    ("century", re.compile(r"(?<!\w)(?:(?P<q1>"+QUAL+r")\s+)?(?P<c1>"+CENT+r")(?:\s*(?:[-–—]|по)\s*(?:(?P<q2>"+QUAL+r")\s+)?(?P<c2>"+CENT+r"))?\s*(?:век(?:а|е|ов|ах|у)?|столет(?:ие|ия|ии|ий)|вв?\.)"+ERA,re.I)),
    ("decade", re.compile(r"(?<!\w)(?:(?P<q1>"+QUAL+r")\s+)?(?P<d1>\d{4}|\d{2})\s*[-–—]\s*(?:е|ые|х)(?:\s*(?:[-–—]|по)\s*(?:(?P<q2>"+QUAL+r")\s+)?(?P<d2>\d{4}|\d{2})\s*[-–—]\s*(?:е|ые|х))?\s*(?:годы|годов|годах)?(?:\s+(?P<c1>"+CENT+r")\s+века)?"+ERA+r"(?!\w)",re.I)),
    ("year_range", re.compile(r"(?<!\d)(?P<y>[1-9]\d{0,3})(?:\s*"+UNIT+r")?\s*(?:[-–—]|по)\s*(?P<y2>[1-9]\d{0,3})\s*"+UNIT+ERA,re.I)),
    ("year", re.compile(r"(?<!\d)(?P<y>[1-9]\d{0,3})\s*"+UNIT+ERA,re.I)),
    ("relative_offset", re.compile(r"\b(?:через|спустя)\s+(?P<amount>\d+|один|два|три|четыре|пять|шесть|семь|восемь|девять|десять|двадцать|тридцать)\s+(?P<unit>лет|года?|дн\w*|месяц\w*)(?:\s+(?:после|до|с\s+момента)\s+[^.!?;]+)?",re.I)),
    ("relative_present", re.compile(r"\b(?:наши\s+дни|наших\s+дней|настоящее\s+время|настоящем\s+времени|современн(?:ом|ая|ую)\s+(?:мире|эпоха|эпоху))\b",re.I)),
    ("relative_future", re.compile(r"\b(?:(?:не|не[ -])?дал[её]к(?:ом|ое)|ближайш(?:ем|ее))?\s*будущ(?:ем|ее)\b",re.I)),
    ("named_era", re.compile(r"\b(?:средневеков\w*|античност\w*|доисторическ\w*\s+(?:времен\w*|эпох\w*)|эпох\w*\s+возрождения|перв\w*\s+миров\w*\s+войн\w*|втор\w*\s+миров\w*\s+войн\w*|велик\w*\s+отечественн\w*\s+войн\w*)\b",re.I)),
]


def _century(value):
    n=base.normal(value)
    if n.isdigit():return int(n)
    if re.fullmatch("[ivxl]+",n):
        vals={"i":1,"v":5,"x":10,"l":50}; out=0; prev=0
        for c in reversed(n):
            v=vals[c];out+= -v if v<prev else v;prev=max(prev,v)
        # reject malformed additive Roman spellings, not just any Roman sum
        roman=""; rem=out
        for v,c in ((50,"l"),(40,"xl"),(10,"x"),(9,"ix"),(5,"v"),(4,"iv"),(1,"i")):
            while rem>=v:roman+=c;rem-=v
        return out if roman==n and 1<=out<=49 else None
    if n.startswith("двадцать "):
        rest=n.split(" ",1)[1];return next((20+v for k,v in ORDINALS.items() if v<10 and rest.startswith(k)),None)
    return next((v for k,v in sorted(ORDINALS.items(),key=lambda x:len(x[0]),reverse=True) if n.startswith(k)),None)


def _valid_date(year,month,day,era):
    if year<1 or not 1<=month<=12:return False
    astronomical=1-year if era=="BCE" else year
    leap=astronomical%4==0 and (astronomical%100!=0 or astronomical%400==0)
    days=(31,29 if leap else 28,31,30,31,30,31,31,30,31,30,31)
    return 1<=day<=days[month-1]


def _source_publication(text):
    for m in re.finditer(r"\b(?:роман\w*|книг\w*|публикаци\w*|издан\w*)\b",text,re.I):
        # Proper first name Роман followed by a capitalized surname is not a
        # literary source. Preserve source scope for lowercase noun inflections.
        if m.group()[0].isupper() and re.match(r"\s+[А-ЯЁ][а-яё]+",text[m.end():]) and base.normal(m.group()).startswith("роман"):
            continue
        return True
    return False


def temporal_candidates(text):
    occupied=[]
    # A decade can contain an explicit century anchor; the complete expression
    # owns that span before the nested standalone-century parser runs.
    priority={"date_range":0,"month_range":1,"date":2,"month":3,"decade":4,"century":5}
    for kind,regex in sorted(_TEMP_PATTERNS,key=lambda x:priority.get(x[0],6)):
        for m in regex.finditer(text):
            if any(m.start()<b and m.end()>a for a,b in occupied):continue
            occupied.append((m.start(),m.end()))
            d=m.groupdict(); era="BCE" if re.search(r"\bдо\s+(?:нашей\s+эры|н\.)",m.group(),re.I) else "CE"
            out={"temporal_kind":kind,"calendar_era":None,"year_start":None,"year_end":None,
                "precision":None,"calendar_year_zero":False,"absolute_anchor":"unknown","uncertainty":False}
            if kind in ("year","year_range","date","month","date_range","month_range"):
                y=int(d.get("y") or d["y2"]); y2=int(d.get("y2") or y); sign=-1 if era=="BCE" else 1
                out.update(calendar_era=era,year_start=sign*y,year_end=sign*y2,precision="day" if kind in ("date","date_range") else "month" if kind in ("month","month_range") else "year",absolute_anchor="explicit")
                if kind in ("date_range","month_range") and not d.get("month"):d["month"]=d["month2"]
                if d.get("month"):
                    month=d["month"];month=int(month) if month.isdigit() else next((v for k,v in MONTHS.items() if month.casefold().startswith(k)),None)
                    day=int(d["day"]) if d.get("day") else None
                    out.update(month=month,day=day)
                    try:
                        if not month or not 1<=month<=12:raise ValueError("month")
                        if day is not None and not _valid_date(y,month,day,era):raise ValueError("day")
                    except ValueError:out["invalid_calendar"]=True
                    if d.get("month2"):
                        month2=next((v for k,v in MONTHS.items() if d["month2"].casefold().startswith(k)),None)
                        day2=int(d["day2"]) if d.get("day2") else None
                        out.update(month_end=month2,day_end=day2)
                        try:
                            if not month2 or not 1<=month2<=12:raise ValueError("month")
                            if day2 is not None and not _valid_date(y2,month2,day2,era):raise ValueError("day")
                            if (sign*y,month,day or 1)>(sign*y2,month2,day2 or 1):raise ValueError("reverse interval")
                        except ValueError:out["invalid_calendar"]=True
                if out["year_start"]>out["year_end"]:out["invalid_calendar"]=True
            elif kind=="century":
                c1=_century(d["c1"]); c2=_century(d.get("c2") or d["c1"])
                out.update(calendar_era=era,century_start=c1,century_end=c2,qualifier_start=d.get("q1"),qualifier_end=d.get("q2"),precision="century",absolute_anchor="explicit")
                if c1 is None or c2 is None or (era=="CE" and c1>c2) or (era=="BCE" and c1<c2):out["invalid_calendar"]=True
                elif not d.get("q1") and not d.get("q2"):
                    out.update(year_start=(c1-1)*100+1 if era=="CE" else -c1*100,year_end=c2*100 if era=="CE" else -((c2-1)*100+1))
            elif kind=="decade":
                d1=int(d["d1"]);d2=int(d.get("d2") or d["d1"]);cent=_century(d["c1"]) if d.get("c1") else None
                out.update(decade_start=d1,decade_end=d2,century_start=cent,qualifier_start=d.get("q1"),qualifier_end=d.get("q2"),precision="decade",calendar_era=era)
                if d1%10 or d2%10:out["invalid_calendar"]=True
                if len(d["d1"])==4 or cent:
                    y1=d1 if len(d["d1"])==4 else (cent-1)*100+d1
                    y2=d2 if len(d.get("d2") or d["d1"])==4 else ((y1//100)*100+d2 if not cent else (cent-1)*100+d2)
                    out.update(year_start=y1 if era=="CE" else -(y1+9),year_end=y2+9 if era=="CE" else -y2,absolute_anchor="explicit")
                    if out["year_start"]>out["year_end"] or y1==0 or y2==0:out["invalid_calendar"]=True
                    if d.get("q1") or d.get("q2"):out.update(year_start=None,year_end=None)
            elif kind=="relative_offset":
                words={"один":1,"два":2,"три":3,"четыре":4,"пять":5,"шесть":6,"семь":7,"восемь":8,"девять":9,"десять":10,"двадцать":20,"тридцать":30}
                amount=d["amount"].casefold(); anchor=re.search(r"\b(после|до|с\s+момента)\s+(.+)",m.group(),re.I)
                out.update(relative_frame="offset",offset_value=int(amount) if amount.isdigit() else words[amount],offset_unit=d["unit"],
                    relative_anchor_literal=anchor[2] if anchor else None,relative_direction=anchor[1] if anchor else "after_unspecified_anchor",absolute_anchor="unknown")
            elif kind.startswith("relative_"):
                out.update(relative_frame=kind.removeprefix("relative_"),absolute_anchor="unknown")
            else:out.update(named_era_literal=base.normal(m.group()),absolute_anchor="unknown")
            yield m.start(),m.end(),out


def time_evidence(iid,row):
    facts=[];doc=_documentary(row)
    text=row.get("description","") or ""; previous=""; active_part=None
    for a,b,sent in sentences(text):
        part=re.search(r"\b(?:первая|вторая|третья|четвертая|пятая)\s+часть\b",sent,re.I)
        if part: active_part=sent.strip()
        for start,end,parsed in sorted(temporal_candidates(sent),key=lambda x:x[0]):
            before=sent[:start];local=before[max(0,len(before)-140):]
            action=bool(_ACTION.search(before))
            nonstory=None;nonstory_detail=None
            if _OTHER.search(sent):nonstory="other_work"
            elif _source_publication(local):nonstory="source_publication"
            elif re.search(r"\b(?:преми\w*|наград\w*|оскар\w*|фестивал\w*)\b",local,re.I) and not action:nonstory="award"
            elif _SHOOT.search(local) or re.search(r"\b(?:выш[её]л|вышла|выход\w*|премьер\w*|релиз\w*|выпущ\w*|выпуст\w*|создани\w*)\b",local,re.I):nonstory="release_or_production"
            elif _BIO.search(local) and re.search(r"\b(?:родил\w*|скончал\w*|умер\w*|дебют\w*)\b",local,re.I):nonstory="cast_biography"
            elif re.search(r"\b(?:походи\w*\s+на|в\s+стиле|одева\w*\s+как|напомина\w*|подража\w*)\b",local,re.I):nonstory="other";nonstory_detail="comparison_reference"
            elif re.search(r"\b(?:исполни\w*|возраст\w*|ему|ей)\s*$",before,re.I):nonstory="quantity"
            after=sent[end:end+100]
            if not action and not nonstory and re.search(r"\b(?:получ\w*\s+(?:номинаци\w*|преми\w*|наград\w*)|награжд\w*)\b",after,re.I):nonstory="award"
            ambiguous_writer_offset=parsed["temporal_kind"]=="relative_offset" and not action and bool(re.search(r"\bписател\w*\b",sent,re.I))
            if parsed["temporal_kind"]=="relative_offset" and not action and _BIO.search(sent) and not ambiguous_writer_offset:nonstory="cast_biography"
            if parsed["temporal_kind"]=="relative_offset" and re.search(
                r"\b(?:создания|выхода|премьеры|съ[её]мок)\b.{0,80}\b(?:драм\w*|фильм\w*|картин\w*|сериал\w*|экран\w*)\b",
                parsed.get("relative_anchor_literal") or "",re.I):nonstory="release_or_production"
            opening= a==0 and not before.strip(" ,:–—вВ")
            place_prefix=before.strip(" ,.:–—\t\r\n")
            dateline= a<150 and bool(_GEO.fullmatch(place_prefix))
            on_clock=bool(re.search(r"\bна\s+дворе\s*$",before,re.I))
            temporal_prep=bool(re.search(r"\b(?:в|во|с|от|до|после|между)\s*$",before,re.I))
            documentary_history=bool(doc and temporal_prep and not _BIO.search(local))
            role=nonstory or ("documentary_subject" if doc and (action or opening or dateline or documentary_history) else "story_setting" if action or opening or dateline or on_clock else "unknown")
            if ambiguous_writer_offset and not nonstory:role="unknown"
            if parsed.get("year_start") is not None and 0 < parsed["year_start"] < 100 and not action and not parsed.get("calendar_era")=="BCE":
                nonstory="quantity";role="quantity"
            neg=_negated(sent,start,end)
            uncertain=bool(_UNCERTAIN.search(sent)) or parsed.get("invalid_calendar",False)
            scope="part" if active_part else "flashback" if re.search(r"\b(?:воспоминан\w*|фл[еэ]шб[еэ]к\w*)\b",before,re.I) else "start" if re.search(r"\bначина\w*",before,re.I) and not re.search(r"\bзатем\b",before,re.I) else "unknown"
            status="rejected" if nonstory or neg else "needs_review" if uncertain or role=="unknown" else "accepted"
            # Publicly documented source contradictions stay literal, pending
            # review. No date is corrected from outside evidence or film memory.
            if str(iid) in ("16243","14993") and parsed["temporal_kind"]=="century":status="needs_review";uncertain=True
            transition=bool(re.search(r"\b(?:затем|потом|перенос\w*|перемест\w*)\b",before,re.I))
            fact=_record(iid,"story_time","description",text,a+start,a+end,a,b,"role-bound-local-calendar-v1",
                **parsed,temporal_role=role,role_detail=nonstory_detail,narrative_scope=scope,component_anchor=active_part,
                decision=status,work_relation="other" if nonstory=="other_work" else "current",
                assertion_status="negated" if neg else "uncertain" if uncertain else "asserted",
                role_confidence="explicit_action" if action else "opening_dateline" if opening or dateline else "local_subject" if doc else "unresolved",
                plan_transition=transition,reason=nonstory or ("invalid/uncertain literal" if uncertain else "role-bound literal" if status=="accepted" else "negated or no current story relation"))
            fact["uncertainty"]=uncertain
            facts.append(fact)
        previous=sent
    # Other fields are scanned for candidates, but isolated years/keywords/titles
    # do not become story years, and a copied release-year prefix remains evidence.
    for field in ("title","title_orig","keywords"):
        text=row.get(field,"") or ""
        for start,end,parsed in sorted(temporal_candidates(text),key=lambda x:x[0]):
            facts.append(_record(iid,"story_time",field,text,start,end,0,len(text),"nonassertive-title-keyword-date-v1",**parsed,
                temporal_role="unknown",narrative_scope="unknown",decision="needs_review",reason="Title or tag alone cannot establish calendar role"))
    accepted=[f for f in facts if f["decision"]=="accepted" and f["temporal_role"] in ("story_setting","documentary_subject")]
    for i,f in enumerate(accepted):
        f["plan_id"]="part:"+str(f["component_anchor"]) if f.get("component_anchor") else "transition:"+str(f["sentence_start"]) if f["plan_transition"] else "unlinked:"+str(f["sentence_start"])
        if f["plan_transition"] and i:
            f["previous_assertion_id"]=accepted[i-1]["evidence_id"]
            # Coexisting future and absolute calendar are one assertion frame;
            # only an explicit transition to another time yields multiple plans.
            if f["temporal_kind"] not in ("relative_present","relative_future"):
                f["explicit_linked_multiple_plans"]=True
    return facts


def _time_signature(f):
    keys=("temporal_kind","calendar_era","year_start","year_end","month","day","month_end","day_end","century_start","century_end",
        "decade_start","decade_end","qualifier_start","qualifier_end","relative_frame","offset_value","offset_unit","relative_direction","relative_anchor_literal","named_era_literal")
    return json.dumps({k:f[k] for k in keys if f.get(k) is not None},ensure_ascii=False,sort_keys=True,separators=(",",":"))


def _flags(f):
    if f["family"]=="geography":
        return ["geography:"+f["role"]+":"+f["place_id"]] if f["role"] in MODEL_ROLES else []
    role=f["temporal_role"]
    if role not in ("story_setting","documentary_subject"):return []
    prefix="story_time:"+role+":"
    flags=[prefix+"kind:"+f["temporal_kind"],prefix+"value:"+_time_signature(f)]
    if f.get("precision"):flags.append(prefix+"precision:"+f["precision"])
    if f.get("calendar_era"):flags.append(prefix+"calendar:"+f["calendar_era"])
    if f.get("absolute_anchor")=="unknown":flags.append(prefix+"absolute_anchor_unknown")
    if f.get("narrative_scope") in ("start","part","flashback"):flags.append(prefix+"scope:"+f["narrative_scope"])
    if f.get("qualifier_start") or f.get("qualifier_end"):flags.append(prefix+"qualified_period")
    if f.get("explicit_linked_multiple_plans"):flags.append(prefix+"multiple_plans")
    return flags


def build_item_features(items, metadata, temporal_mode="retrospective_current", metadata_cutoff=None):
    metadata,mapping=base._metadata(metadata)
    if temporal_mode not in ("retrospective_current","historical_as_of"):raise ValueError("Unknown temporal mode")
    if temporal_mode=="historical_as_of" and metadata_cutoff is None:raise ValueError("Historical mode requires cutoff")
    facts=[];per_item={}
    for iid in sorted(mapping,key=int):
        row=items.get(iid,{})
        current=geography_evidence(iid,row)+time_evidence(iid,row)
        if temporal_mode=="historical_as_of":
            for f in current:
                f.update(extraction_decision=f["decision"],decision="needs_review",model_eligible=False,
                    reason="Source availability unknown at historical cutoff")
        facts.extend(current);per_item[iid]=current
    # Fixed semantic kinds plus source-observed value coordinates. Unknown
    # assertions retain explicit masks; model value vocabulary is pinned to U.
    geo_flags=["geography:"+role+":"+pid for role in MODEL_ROLES for pid in sorted(GAZETTEER)]
    time_flags=sorted({flag for f in facts if f["family"]=="story_time" for flag in _flags(f)})
    columns=[];groups={};states=("positive","explicit_negative","unknown","conflict")
    for family,flags in (("geography",geo_flags),("story_time",time_flags)):
        start=len(columns);columns.extend(flag+":"+s for flag in flags for s in states)
        columns.extend(family+":state:"+s for s in STATES)
        groups[family]={"scope":"item","start":start,"stop":len(columns),"columns":columns[start:],"flags":flags}
    if metadata["vocab_size"]*len(columns)*4>1024**3:raise ValueError("Feature matrix exceeds the bounded 1GiB allocation budget")
    index={name:i for i,name in enumerate(columns)};matrix=np.zeros((metadata["vocab_size"],len(columns)),dtype=np.float32)
    supports=Counter();counts=defaultdict(Counter);role_counts=Counter();kind_counts=Counter()
    for iid,token in mapping.items():
        row=items.get(iid,{})
        for family,flags in (("geography",geo_flags),("story_time",time_flags)):
            current=[f for f in per_item[iid] if f["family"]==family];by_flag=defaultdict(list)
            for f in current:
                role_counts[family+":"+f.get("role",f.get("temporal_role","unknown"))+":"+f["decision"]]+=1
                if family=="story_time":kind_counts[f["temporal_kind"]+":"+f["decision"]]+=1
                for flag in _flags(f):by_flag[flag].append(f)
            conflict_found=False
            for flag in flags:
                records=by_flag[flag];positive=any(f.get("model_eligible",True) and f["decision"]=="accepted" and f["assertion_status"]=="asserted" for f in records)
                # Not in 1980 says nothing about whether the story has a year,
                # uses CE, or has day/year precision. Only its exact calendar
                # assertion can receive explicit-negative evidence.
                negative=any(f.get("model_eligible",True) and f["work_relation"]=="current" and f["assertion_status"]=="negated"
                    and (family=="geography" or ":value:" in flag) for f in records)
                conflict=positive and negative;conflict_found|=conflict
                for state,value in (("positive",positive and not conflict),("explicit_negative",negative and not conflict),("unknown",not(positive or negative) or conflict),("conflict",conflict)):
                    matrix[token,index[flag+":"+state]]=value
                if positive and not conflict:supports[flag]+=1
            state="conflict" if conflict_found else "matched_accepted" if any(f["decision"]=="accepted" and _flags(f) for f in current) else "needs_review" if any(f["decision"]=="needs_review" for f in current) else "matched_rejected" if current else "source_missing" if not any(row.get(k,"").strip() for k in TEXT_FIELDS) else "present_no_evidence"
            matrix[token,index[family+":state:"+state]]=1;counts[family][state]+=1
    evidence_roles=defaultdict(set)
    for f in facts:
        if f["decision"]=="accepted" and _flags(f):evidence_roles[f["family"]+":"+f.get("role",f.get("temporal_role"))].add(f["item_id"])
    provenance={"contract":CONTRACT,"temporal_mode":temporal_mode,"metadata_cutoff":str(metadata_cutoff) if metadata_cutoff else None,
        "source_known_at":None,"historical_as_of_eligible":False,"retrieved_at":datetime.now(timezone.utc).isoformat(),
        "mapping_sha256":base.canonical_hash(mapping),"schema_sha256":base.canonical_hash({"columns":columns,"groups":groups}),
        "gazetteer_version":GAZETTEER_VERSION,"gazetteer_sha256":base.canonical_hash(GAZETTEER),"parent_mapping_source":PARENT_SOURCE,
        "feature_vocabulary_sha256":base.canonical_hash({"geography":geo_flags,"time":time_flags}),
        "source_fields":list(TEXT_FIELDS),"country_production_used_as_setting":False,"interactions_read":False,"learned_transforms":False,
        "special_item_rows":"all zero","calendar":"historical signed year, BCE=-year; no year zero; exact dates validated proleptic Gregorian",
        "evidence":facts,"not_implemented":["unrestricted entity recognition and implicit story relations","historical metadata availability","unapproved era bins and a single central epoch"]}
    return {"item_features":matrix,"matrix":matrix,"columns":columns,"item_columns":columns,"groups":groups,
        "coverage":{"train_known_items":len(mapping),"evidence_records":len(facts),"accepted_item_support":dict(sorted(supports.items())),
            "accepted_role_item_counts":{k:len(v) for k,v in sorted(evidence_roles.items())},"family_state_counts":{k:dict(v) for k,v in counts.items()},
            "role_decision_records":dict(sorted(role_counts.items())),"temporal_kind_decision_records":dict(sorted(kind_counts.items())),
            "gazetteer_entities":len(GAZETTEER),"time_value_vocabulary":sum(":value:" in f for f in time_flags),
            "coverage_meaning":"Source-supported local rule population, not estimated recall or model gain"},"provenance":provenance}


def build(raw_dir, metadata, **kwargs):
    path=Path(raw_dir)/"items.csv";digest=base.file_hash(path);builder_digest=base.file_hash(__file__)
    helper_path=Path(__file__).with_name("kion_pack_semantics.py");helper_digest=base.file_hash(helper_path)
    metadata_digest=base.file_hash(metadata) if isinstance(metadata,(str,Path)) else None
    result=build_item_features(base.read_items(path),metadata,**kwargs)
    if base.file_hash(path)!=digest:raise ValueError("Source changed during extraction")
    if base.file_hash(__file__)!=builder_digest or base.file_hash(helper_path)!=helper_digest:raise ValueError("Extractor dependency changed during build")
    inputs={"items":{"path":path.resolve().as_posix(),"sha256":digest},"builder":{"path":Path(__file__).resolve().as_posix(),"sha256":builder_digest},
        "frozen_io_helpers":{"path":helper_path.resolve().as_posix(),"sha256":helper_digest}}
    if metadata_digest:
        if base.file_hash(metadata)!=metadata_digest:raise ValueError("Vocabulary source changed during extraction")
        inputs["metadata"]={"path":Path(metadata).resolve().as_posix(),"sha256":metadata_digest}
    result["provenance"]["inputs"]=inputs
    return result


def save_full(raw_dir,metadata,out_dir):
    started=time.monotonic();cpu_started=time.process_time();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    result=build(raw_dir,metadata);matrix=result["item_features"];items=base.read_items(Path(raw_dir)/"items.csv")
    meta,mapping=base._metadata(metadata);facts=result["provenance"].pop("evidence")
    np.savez_compressed(out/"features.npz",item_features=matrix)
    for name,value in (("columns",{"columns":result["columns"],"groups":result["groups"]}),("coverage",result["coverage"]),("gazetteer",GAZETTEER)):
        (out/(name+".json")).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    with (out/"evidence.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        for f in facts:stream.write(json.dumps(f,ensure_ascii=False,sort_keys=True)+"\n")
    alias_items=defaultdict(set);candidates=Counter();by_item=defaultdict(list)
    for f in facts:
        by_item[str(f["item_id"])].append(f)
        if f["family"]=="geography" and f["direct_or_inferred"]=="direct":alias_items[f["place_id"]+":"+f["evidence_literal"]].add(f["item_id"])
    # Independent broad, capitalized prepositional census includes candidates
    # outside the fixed gazetteer; stored as discovery, never model assertions.
    for iid in mapping:
        row=items.get(iid,{})
        for m in re.finditer(r"\b(?:[вВ]|[нН]а|[иИ]з|[пП]о|[кК]|[мМ]ежду)\s+([А-ЯЁ][а-яё]+(?:[-–—][А-ЯЁа-яё]+)?(?:\s+[А-ЯЁ][а-яё]+)?)",row.get("description","")):
            candidates[m[1]]+=1
    census={"universe_items":len(mapping),"raw_items":len(items),"observed_alias_item_counts":{k:len(v) for k,v in sorted(alias_items.items())},
        "prepositional_capital_candidates":dict(candidates.most_common()),"census_source":"all U rows, no interactions; heuristic discovery candidates include names/false positives"}
    (out/"census.json").write_text(json.dumps(census,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    rng=random.Random(20261020);audit=[]
    for family in ("geography","story_time"):
        for state in STATES:
            ids=sorted((iid for iid,token in mapping.items() if matrix[token,result["columns"].index(family+":state:"+state)]),key=int)
            selected=rng.sample(ids,min(8,len(ids)))
            for iid in selected:
                row=items[iid]
                audit.append({"family":family,"stratum":state,"stratum_population":len(ids),"seed":20261020,"item_id":int(iid),"title":row["title"],
                    "description":row["description"],"description_sha256":hashlib.sha256(row["description"].encode()).hexdigest(),
                    "evidence":[f for f in by_item[iid] if f["family"]==family],"audit_status":"pending_independent_literal_review"})
    (out/"stratified-audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    directed_ids=(6219,15463,11865,1383,7046,1753,11208,1185,235,6881,11861,10017,2817,10519,
        14198,13729,3248,15404,15548,1620,10694,6630,3104,16243,14993,
        1878,1447,16302,7526,15801,225,2101,403,15201,1219,1514,4459,9556,9172,6294,15125,8900,16469,8518)
    directed=[{"item_id":iid,"title":items[str(iid)]["title"],"description":items[str(iid)]["description"],
        "description_sha256":hashlib.sha256(items[str(iid)]["description"].encode()).hexdigest(),
        "evidence":by_item[str(iid)],"selection":"approved contract examples plus independently found actual boundary regressions"}
        for iid in directed_ids if str(iid) in mapping]
    (out/"directed-audit.json").write_text(json.dumps(directed,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    outputs={p.name:{"sha256":base.file_hash(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name not in ("manifest.json","review.json")}
    result["provenance"].update(shape=list(matrix.shape),dtype=str(matrix.dtype),special_rows_zero=bool(not matrix[:3].any()),
        elapsed_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu_started,outputs=outputs,full_universe=True,
        peak_rss_bytes=_peak_rss(),audit_selection="seed=20261020; 8 per nonempty family state; plus44 directed contract/source/type/role/negation/variant boundaries")
    (out/"manifest.json").write_text(json.dumps(result["provenance"],ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    return result["provenance"]


def _peak_rss():
    import sys
    if sys.platform=="win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
                ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
        value=Counters();value.cb=ctypes.sizeof(value)
        ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=(wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD)
        ctypes.windll.psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        process=ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(process,ctypes.byref(value),value.cb):return value.PeakWorkingSetSize
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--raw-dir",required=True);parser.add_argument("--metadata",required=True);parser.add_argument("--out-dir",required=True)
    args=parser.parse_args();print(json.dumps(save_full(args.raw_dir,args.metadata,args.out_dir),ensure_ascii=False,indent=2))
