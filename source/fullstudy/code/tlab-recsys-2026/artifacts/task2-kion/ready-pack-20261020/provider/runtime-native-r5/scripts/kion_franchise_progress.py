"""Evidence-scoped franchise graph and strict-past H63 candidate descriptors.

The graph is partial. Missing links/edges never imply a negative, and raw
candidate IDs and raw seen masks remain the caller's responsibility.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

CONTRACT = "kion-franchise-progress-v1"
ACCEPTED_STATUSES = {"verified", "accepted_annotation"}
ORDER_KINDS = {"mainline_release", "story_chronology", "catalog_release_precedes"}
COLUMNS = ("n63", "known_raw_rows", "unmapped_rows", "candidate_mapping_unknown",
    "branch_ambiguous", "graph_available", "scope_complete", "adjacency_complete",
    "known_prior_complete", "n_branch_seen", "n_prior", "n_prior_seen", "n_prior_missing",
    "prior_seen_share", "prior_share_missing", "frontier", "direct_next", "next_missing",
    "skip_observed_priors", "already_seen_work", "branch_cold", "recency_days",
    "recency_missing", "graph_hop_distance", "distance_missing")


def build_annotation_features(annotations, metadata, temporal_mode="retrospective_current", metadata_cutoff=None):
    """Project independently reviewed raw membership, separate from Series80.

    Same-work and branch assertions never create membership or order here.
    An unannotated raw card gets a family unknown bit, not a curated negative.
    """
    if isinstance(annotations,(str,Path)):
        annotations=json.loads(Path(annotations).read_text(encoding="utf-8"))
    if isinstance(metadata,(str,Path)):
        metadata=json.loads(Path(metadata).read_text(encoding="utf-8"))
    if temporal_mode not in ("retrospective_current","historical_as_of"):
        raise ValueError("Unknown temporal mode")
    if temporal_mode == "historical_as_of" and metadata_cutoff is None:
        raise ValueError("Historical annotation projection needs metadata cutoff")
    mapping,size=metadata["item_id_to_token"],metadata["vocab_size"]
    if (not isinstance(size,int) or isinstance(size,bool) or size < 3 or sorted(mapping.values()) != list(range(3,size))
        or any(not isinstance(k,str) or not re.fullmatch(r"0|[1-9]\d*",k) or not isinstance(v,int) or isinstance(v,bool) for k,v in mapping.items())
        or metadata.get("special_tokens",{"PAD":0,"BOS":1,"UNK":2}) != {"PAD":0,"BOS":1,"UNK":2}):
        raise ValueError("Expected canonical contiguous train-known mapping")
    rows=annotations.get("accepted_annotations",[])
    families={"manual_series":defaultdict(set),"manual_universe":defaultdict(set)}
    for record in rows:
        if record.get("status") != "accepted_annotation" or not record.get("independent_review_id"):
            raise ValueError("Only explicitly independently reviewed accepted annotations may project")
        if record["relation_type"] not in ("series_member","universe_member"):
            continue
        family="manual_series" if record["relation_type"] == "series_member" else "manual_universe"
        key=record["canonical_series_key"] if family == "manual_series" else record["canonical_universe_key"]
        if not isinstance(key,str) or key == "unknown" or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*",key):
            raise ValueError("Membership needs explicit stable ASCII key")
        # Vocabulary is frozen across temporal modes; availability masks values.
        families[family][key]
        admissible = temporal_mode == "retrospective_current" or (record.get("source_known_at") is not None and _date(record["source_known_at"]) < _date(metadata_cutoff))
        for iid in record["item_ids"]:
            if isinstance(iid,bool) or not isinstance(iid,int) or iid < 0:
                raise ValueError("Expected raw integer item ID")
            if admissible and str(iid) in mapping:
                families[family][key].add(str(iid))
    columns,groups=[],{}
    for family,groups_by_key in families.items():
        start=len(columns)
        columns += [family+":"+key for key in sorted(groups_by_key)]+[family+":unknown"]
        groups[family]={"scope":"item","start":start,"stop":len(columns),"columns":columns[start:],"annotation_keys":sorted(groups_by_key)}
    matrix=np.zeros((size,len(columns)),dtype=np.float32)
    index={name:i for i,name in enumerate(columns)}
    coverage={}
    for family,groups_by_key in families.items():
        observed=set()
        for key,ids in groups_by_key.items():
            for iid in ids:
                matrix[mapping[iid],index[family+":"+key]]=1.
            observed |= ids
        for iid,token in mapping.items():
            matrix[token,index[family+":unknown"]]=iid not in observed
        coverage[family]={"keys":len(groups_by_key),"known_items":len(observed),"unknown_items":len(mapping)-len(observed)}
    return {"item_features":matrix,"columns":columns,"item_columns":columns,"groups":groups,"coverage":coverage,
        "provenance":{"contract":"kion-manual-franchise-memberships-v1","status":"accepted_annotation",
            "external_verified":False,"annotation_source":"LLM/manual catalogue plus independent review",
            "temporal_mode":temporal_mode,"metadata_cutoff":str(metadata_cutoff) if metadata_cutoff else None,
            "annotations_sha256":canonical_hash(annotations),"mapping_sha256":canonical_hash(mapping),
            "schema_sha256":canonical_hash({"columns":columns,"groups":groups}),
            "source_series80_modified":False,"order_inferred":False,"work_identity_merged":False,
            "unknown_semantics":"Unannotated is unknown, not a curated negative", "special_item_rows":"all zero"}}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _date(value):
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def validate_graph(graph):
    """Reject dangling endpoints, cycles and adjacency without explicit scope."""
    if graph.get("contract") != CONTRACT:
        raise ValueError("Unknown franchise graph contract")
    works = graph["works"]
    scopes = graph["scopes"]
    evidence = graph["evidence"]
    if graph.get("graph_sha256") and graph["graph_sha256"] != canonical_hash({k:v for k,v in graph.items() if k != "graph_sha256"}):
        raise ValueError("Graph content does not match frozen hash")
    for kind in ("item_work_links", "memberships", "edges"):
        for record in graph[kind]:
            if record["status"] not in (*ACCEPTED_STATUSES, "unknown", "ambiguous", "conflict"):
                raise ValueError("Invalid assertion status")
            if not record.get("evidence_ids") or any(e not in evidence for e in record["evidence_ids"]):
                raise ValueError("Every assertion requires recorded evidence")
    for link in graph["item_work_links"]:
        if link["status"] in ACCEPTED_STATUSES and link.get("work_id") not in works:
            raise ValueError("Dangling work match")
    for member in graph["memberships"]:
        if member["work_id"] not in works or member["scope_id"] not in scopes:
            raise ValueError("Dangling membership")
        if member["work_id"] not in scopes[member["scope_id"]]["work_ids"]:
            raise ValueError("Membership work is outside its declared scope")
    if any(edge["scope_id"] not in scopes for edge in graph["edges"]):
        raise ValueError("Dangling edge scope")
    for sid, scope in scopes.items():
        if scope["order_kind"] not in ORDER_KINDS:
            raise ValueError("Unsupported typed order relation")
        if scope["order_kind"]=="catalog_release_precedes" and (any(scope.get(k) for k in ("scope_complete","known_prior_complete","adjacency_complete")) or "mainline" in scope.get("role","")):
            raise ValueError("Weak catalogue order cannot claim mainline or completeness")
        eligible = set(scope["work_ids"])
        if not eligible <= works.keys():
            raise ValueError("Dangling scope work")
        parents = defaultdict(set)
        for edge in graph["edges"]:
            if edge["scope_id"] != sid or edge["status"] not in ACCEPTED_STATUSES:
                continue
            a, b = edge["from_work"], edge["to_work"]
            if a == b or not {a, b} <= eligible or edge["order_kind"] != scope["order_kind"]:
                raise ValueError("Invalid edge endpoint or order scope")
            if edge.get("direct_next") and not scope.get("adjacency_complete"):
                raise ValueError("direct_next needs reviewed complete adjacency")
            parents[b].add(a)
        pending = set(eligible)
        while pending:
            ready = {w for w in pending if not parents[w] & pending}
            if not ready:
                raise ValueError("Cycle in franchise DAG")
            pending -= ready
    return graph


def build_pilot_graph(items, series_annotation):
    """Reproduce the three reviewed continuity chains; other Series80 abstain.

    Work matching requires the recorded ID, original-title alias, year and
    director. This is an explicit evidence ledger, never title-based discovery.
    Edition equality/cut is unknown even for verified same-work cards.
    """
    if isinstance(series_annotation, (str, Path)):
        series_annotation = json.loads(Path(series_annotation).read_text(encoding="utf-8"))
    graph = {"contract": CONTRACT, "temporal_mode": "retrospective_current",
        "source_known_at": None, "works": {}, "scopes": {}, "item_work_links": [],
        "memberships": [], "edges": [], "evidence": {},
        "series_annotation_sha256": canonical_hash(series_annotation)}
    graph["evidence"]["research"] = {"source": "docs/task2-franchise-order-research.md",
        "retrieved_at": "2026-10-20", "source_known_at": None,
        "statement": "Three reviewed raw/work/continuity chains; Star Wars raw links unknown"}
    groups = [
        ("back-to-future", "Роберт Земекис", [
            ("bttf-1", "1985-07-03", [10696], ["Back To The Future"]),
            ("bttf-2", "1989-11-22", [12191, 3012], ["Back to the Future Part II", "Back To The Future II"]),
            ("bttf-3", "1990-05-25", [11903, 544], ["Back To The Future III", "Back to the Future Part III"])],
            ["https://www.backtothefuture.com/movies/backtothefuture1", "https://www.backtothefuture.com/movies/backtothefuture2", "https://www.backtothefuture.com/movies/backtothefuture3"]),
        ("spider-man-raimi", "Сэм Рэйми", [
            ("raimi-1", "2002-12-31", [16382, 15477], ["Spider-Man"]),
            ("raimi-2", "2004-12-31", [3166, 12444], ["Spider-Man 2"]),
            ("raimi-3", "2007-12-31", [4072], ["Spider-Man 3"])],
            ["https://www.sonypictures.com/movies/spiderman3"]),
        ("amazing-spider-man", "Марк Уэбб", [
            ("webb-1", "2012-12-31", [14567], ["The Amazing Spider-Man"]),
            ("webb-2", "2014-12-31", [12299], ["The Amazing Spider-Man 2"])],
            ["https://www.sonypictures.com/movies/theamazingspiderman", "https://www.sonypictures.com/movies/theamazingspiderman2"]),
    ]
    norm = lambda s: " ".join(str(s).casefold().split())
    claimed = set()
    for gid, director, works, urls in groups:
        sid = gid + ":mainline_release"
        eid = gid + ":official-order"
        graph["evidence"][eid] = {"urls": urls, "source_known_at": None, "retrieved_at": "2026-10-20",
            "statement": "Explicit mainline release ordering and separate continuity; research document records source findings"}
        graph["scopes"][sid] = {"group_id": gid, "continuity_id": gid, "branch_id": "mainline",
            "order_kind": "mainline_release", "work_ids": [w[0] for w in works],
            "scope_complete": True, "known_prior_complete": True, "adjacency_complete": True, "source_known_at": None,
            "scope_boundary": "Reviewed named mainline works released before 2021-08-09; current reconstruction",
            "evidence_ids": [eid, "research"]}
        for wid, upper, raw_ids, titles in works:
            graph["works"][wid] = {"release_upper": upper, "release_precision": "day" if gid == "back-to-future" else "year",
                "source_known_at": None, "evidence_ids": [eid]}
            graph["memberships"].append({"work_id": wid, "scope_id": sid, "role": "mainline",
                "status": "verified", "source_known_at": None, "evidence_ids": [eid, "research"]})
            for raw in raw_ids:
                claimed.add(str(raw))
                row = items.get(str(raw), {})
                try:
                    year_match = int(float(row.get("release_year", "nan"))) == int(upper[:4])
                except (ValueError, OverflowError):
                    year_match = False
                accepted = year_match and norm(row.get("title_orig", "")) in {norm(t) for t in titles} and norm(row.get("directors", "")) == norm(director)
                reid = "raw:" + str(raw)
                graph["evidence"][reid] = {"record": raw, "fields": {k: row.get(k, "") for k in ("title", "title_orig", "release_year", "directors", "description")},
                    "record_sha256": canonical_hash(row), "source_known_at": None}
                graph["item_work_links"].append({"item_id": raw, "work_id": wid if accepted else None,
                    "status": "verified" if accepted else "conflict", "source_known_at": None,
                    "edition_id": "kion-raw:" + str(raw), "edition_identity": "raw_card_only",
                    "cut_identity_status": "unknown", "evidence_ids": [reid, "research"],
                    "reason": "Reviewed same-work match with current raw guard" if accepted else "Raw match guard failed"})
        for left, right in zip(works, works[1:]):
            graph["edges"].append({"from_work": left[0], "to_work": right[0], "scope_id": sid,
                "order_kind": "mainline_release", "direct_next": True, "status": "verified",
                "source_known_at": None, "evidence_ids": [eid, "research"]})
    for group in series_annotation.get("series", []):
        for member in group.get("members", []):
            iid = str(member["item_id"])
            if iid not in claimed:
                graph["item_work_links"].append({"item_id": int(iid), "work_id": None,
                    "group_id": group["id"], "status": "unknown", "source_known_at": None,
                    "edition_id": "kion-raw:" + iid, "cut_identity_status": "unknown",
                    "evidence_ids": ["research"], "reason": "Series80 membership does not establish raw/work, branch or order"})
    validate_graph(graph)
    graph["graph_sha256"] = canonical_hash(graph)
    return graph


def build_graph_revision(inventory, proposals, review, pilot_graph, prior_annotations=None):
    """Integrate only explicit independent graph-review decisions.

    Inputs are in-memory JSON objects. Representation normalization belongs to
    the review bundle; no fuzzy title/year matching happens in this builder.
    Review-approved same-work sets are the only cross-card identity operation.
    """
    validate_graph(pilot_graph)
    if review.get("normalized_bundle_sha256") != canonical_hash(proposals):
        # Tool audit files also commonly use the literal UTF-8 file hash. Pass
        # its recorded canonical content hash explicitly in that case.
        if review.get("normalized_bundle_canonical_sha256") != canonical_hash(proposals):
            raise ValueError("Independent review does not match normalized graph proposals")
    if not review.get("review_id"):
        raise ValueError("Graph expansion requires independent review identity")
    def unique(records,key,label):
        result={record[key]:record for record in records}
        if len(result)!=len(records):
            raise ValueError("Duplicate "+label+" identifier")
        return result
    inv=unique(inventory["groups"],"group_id","inventory group")
    groups=unique(proposals["groups"],"group_id","proposal group")
    decisions=unique(review["group_decisions"],"group_id","review group")
    if len(decisions)!=len(review["group_decisions"]) or groups.keys()!=inv.keys() or decisions.keys()!=inv.keys():
        raise ValueError("Every inventory group requires exactly one explicit review decision")
    parent={}
    identity_merges=[]
    def find(i):
        parent.setdefault(i,i)
        if parent[i]!=i:
            parent[i]=find(parent[i])
        return parent[i]
    def union(ids):
        ids=list(ids)
        if ids:
            base=find(ids[0])
            for item in ids[1:]:
                other=find(item)
                parent[max(base,other)]=min(base,other)
                base=find(base)
    cards={c["item_id"]:c for g in inv.values() for c in g["cards"]}
    link_overrides={}
    consumed_link_overrides=[]
    for override in review.get("item_link_overrides",[]):
        key=(override["group_id"],override["item_id"])
        if key in link_overrides or override.get("status")!="accepted_annotation" or not override.get("reason"):
            raise ValueError("Invalid or duplicate explicit item-link override")
        link_overrides[key]=override
    accepted_links={}
    candidate_work_raw=defaultdict(list)
    for gid,group in groups.items():
        decision=decisions[gid]
        if decision.get("decision") not in ("accepted_annotation","partial","rejected","uncertain","unknown"):
            raise ValueError("Unknown independent group decision")
        if decision["decision"] not in ("accepted_annotation","partial") and any(decision.get(k) for k in ("accepted_item_ids","accepted_scope_ids","accepted_edge_ids")):
            raise ValueError("Rejected or uncertain group may not silently admit assertions")
        allowed=set(decision.get("accepted_item_ids",[]))
        rejected=set(decision.get("rejected_item_ids",[]))
        if allowed & rejected:
            raise ValueError("Accepted and rejected item decisions must be disjoint")
        for key in ("accepted_item_ids","rejected_item_ids","accepted_scope_ids","accepted_edge_ids"):
            if len(decision.get(key,[]))!=len(set(decision.get(key,[]))):
                raise ValueError("Duplicate accepted review identifier")
        source_links=unique(group["item_work_links"],"item_id","item link")
        unique(group["works"],"work_id","work")
        if not (allowed|rejected)<=source_links.keys():
            raise ValueError("Review accepts an item absent from its group proposal")
        for iid in allowed:
            link=copy.deepcopy(source_links[iid])
            override=link_overrides.pop((gid,iid),None)
            if override:
                if link.get("work_id")!=override["source_work_id"] or link.get("status")!="conflict":
                    raise ValueError("Item-link override must bind the exact explicitly conflicted source work")
                link.update(status="proposed",source_proposal_status="conflict",review_item_link_override=copy.deepcopy(override))
                consumed_link_overrides.append({"kind":"item_link_override",**copy.deepcopy(override)})
            if link.get("status")!="proposed" or not link.get("work_id"):
                raise ValueError("Review may not silently promote an unknown/conflicted work match")
            accepted_links[(gid,iid)]=link
            candidate_work_raw[(gid,link["work_id"])].append(iid)
            find(iid)
    if link_overrides:
        raise ValueError("Mandatory item-link overrides were not applied")
    for (gid,cid),ids in candidate_work_raw.items():
        union(ids)
        if len(ids)>1:
            identity_merges.append({"source":"independently_reviewed_proposed_work","group_id":gid,"source_work_id":cid,"item_ids":list(ids),"review_id":review["review_id"]})
    pilot_raw=defaultdict(list)
    for link in pilot_graph["item_work_links"]:
        if link["status"] in ACCEPTED_STATUSES:
            pilot_raw[link["work_id"]].append(link["item_id"])
    for ids in pilot_raw.values():
        union(ids)
        if len(ids)>1:
            identity_merges.append({"source":"frozen_pilot_same_work","item_ids":list(ids),"pilot_graph_canonical_sha256":canonical_hash(pilot_graph)})
    prior_annotations=prior_annotations or {"accepted_annotations":[]}
    for record in prior_annotations.get("accepted_annotations",[]):
        if record.get("status")=="accepted_annotation" and record.get("independent_review_id") and record.get("relation_type")=="same_work":
            if any(i in parent for i in record["item_ids"]) and set(record["item_ids"])<=cards.keys():
                union(record["item_ids"])
                identity_merges.append({"source":"prior_accepted_same_work","item_ids":record["item_ids"],"proposal_id":record.get("proposal_id"),
                    "review_id":record["independent_review_id"],"reason":record.get("rationale"),"evidence":copy.deepcopy(record.get("evidence",[]))})
    for merge in review.get("same_work_merges",[]):
        if merge.get("decision")!="accepted_annotation" or not merge.get("reason") or not set(merge["item_ids"])<=cards.keys():
            raise ValueError("Only explicitly reviewed in-inventory same-work sets may merge")
        union(merge["item_ids"])
        identity_merges.append({"source":"r2_explicit_same_work_review",**copy.deepcopy(merge)})
    components=defaultdict(list)
    for iid in list(parent):
        components[find(iid)].append(iid)
    raw_to_work={}
    applied_overrides=consumed_link_overrides
    forced_names=defaultdict(set)
    for override in review.get("work_alias_overrides",[]):
        key=(override["group_id"],override["source_work_id"])
        if override["item_id"] not in candidate_work_raw.get(key,[]):
            raise ValueError("Reviewed work alias is not an accepted source work/item")
        forced_names[find(override["item_id"])].add(override["canonical_work_id"])
        applied_overrides.append({"kind":"work_alias_override",**copy.deepcopy(override)})
    for override in review.get("cross_group_work_aliases",[]):
        iid=override["item_id"]
        available={cid for (gid,cid),raws in candidate_work_raw.items() if iid in raws}
        if (override.get("decision")!="accepted_annotation" or iid not in parent or len(available)<2
            or not override.get("work_ids") or not set(override["work_ids"])<=available):
            raise ValueError("Cross-group alias requires explicitly reviewed exact shared raw identity")
        forced_names[find(iid)].add(override["canonical_work_id"])
        applied_overrides.append({"kind":"cross_group_work_alias",**copy.deepcopy(override)})
    chosen_names=set()
    for ids in components.values():
        existing={wid for wid,raws in pilot_raw.items() if set(ids)&set(raws)}
        if len(existing)>1:
            raise ValueError("Same-work review would merge distinct frozen pilot works")
        names=existing|forced_names[find(ids[0])]
        if len(names)>1:
            raise ValueError("Reviewed canonical aliases conflict with frozen work identity")
        wid=next(iter(names)) if names else "work:raw:"+str(min(ids))
        if wid in chosen_names:
            raise ValueError("Different work components may not reuse a canonical work ID")
        chosen_names.add(wid)
        raw_to_work.update({iid:wid for iid in ids})
    graph=copy.deepcopy(pilot_graph)
    graph.pop("graph_sha256",None)
    graph["revision"]="kion-franchise-graph-r2"
    graph["independent_review_id"]=review["review_id"]
    graph["work_identity_merges"]=identity_merges
    graph["applied_review_overrides"]=applied_overrides
    graph["evidence"]["r2:independent-review"]={"source_type":"independent_catalogue_review","review_id":review["review_id"],
        "source_known_at":None,"proposals_canonical_sha256":canonical_hash(proposals),"review_canonical_sha256":canonical_hash(review),
        "inventory_canonical_sha256":canonical_hash(inventory),"prior_annotations_canonical_sha256":canonical_hash(prior_annotations),
        "pilot_graph_canonical_sha256":canonical_hash(pilot_graph)}
    evidence_maps={}
    for gid,group in groups.items():
        mapping={eid:"r2:"+gid+":"+eid for eid in group["evidence"]}
        evidence_maps[gid]=mapping
        for eid,value in group["evidence"].items():
            graph["evidence"][mapping[eid]]=copy.deepcopy(value)
    for addition in review.get("evidence_additions",[]):
        gid,eid=addition["group_id"],addition["evidence_id"]
        if eid not in evidence_maps.get(gid,{}):
            raise ValueError("Review evidence addition must bind an existing explicit evidence ID")
        graph["evidence"][evidence_maps[gid][eid]].update({k:copy.deepcopy(v) for k,v in addition.items() if k not in ("group_id","evidence_id")})
        applied_overrides.append({"kind":"evidence_addition",**copy.deepcopy(addition)})
    def evidence(gid,refs):
        if any(e not in evidence_maps[gid] for e in refs):
            raise ValueError("Normalized proposals contain dangling evidence")
        return [evidence_maps[gid][e] for e in refs]+["r2:independent-review"]
    translates={}
    work_claims=defaultdict(list)
    for (gid,cid),raws in candidate_work_raw.items():
        wid=raw_to_work[raws[0]]
        translates[(gid,cid)]=wid
        source=next((w for w in groups[gid]["works"] if w["work_id"]==cid),None)
        if source is None and cid in pilot_graph["works"]:
            source=pilot_graph["works"][cid]
        if source is None:
            raise ValueError("Accepted raw link has no work assertion")
        work_claims[wid].append({**source,"group_id":gid})
    for wid,claims in work_claims.items():
        if wid in graph["works"]:
            continue
        bounds={c.get("release_upper") for c in claims if c.get("release_upper")}
        upper=next(iter(bounds)) if len(bounds)==1 else None
        graph["works"][wid]={"title":claims[0].get("title"),"title_orig":claims[0].get("title_orig"),
            "release_upper":upper,"release_precision":"year","release_status":"catalog_assertion" if upper else "conflict_or_unknown",
            "release_event_scope":"catalogue year only; no territory/event policy established",
            "source_known_at":None,"status":"accepted_annotation","external_verified":False,
            "release_claims":claims,"evidence_ids":["r2:independent-review"]}
    canonical_release_overrides={}
    for override in review.get("work_release_overrides",[]):
        key=(override["group_id"],override["work_id"])
        if key not in translates:
            raise ValueError("Release override requires an accepted source work")
        wid=translates[key]
        if wid in pilot_graph["works"]:
            raise ValueError("A new review may not overwrite a frozen pilot release assertion")
        assertion=(override["release_upper"],override["release_precision"])
        if wid in canonical_release_overrides:
            raise ValueError("Duplicate or conflicting canonical work release override")
        canonical_release_overrides[wid]=assertion
        graph["works"][wid].update(release_upper=override["release_upper"],release_precision=override["release_precision"],
            release_status="reviewed_conflict_or_unknown" if override["release_upper"] is None else "reviewed_catalog_assertion",
            release_review_override=copy.deepcopy(override))
        applied_overrides.append({"kind":"work_release_override",**copy.deepcopy(override)})
    canonical_metadata_overrides=set()
    for override in review.get("work_metadata_overrides",[]):
        key=(override["group_id"],override["source_work_id"])
        if key not in translates or override["item_id"] not in candidate_work_raw[key]:
            raise ValueError("Metadata override requires its exact accepted source work/item")
        wid=translates[key]
        if wid in pilot_graph["works"]:
            raise ValueError("A new review may not overwrite frozen pilot work metadata")
        if wid in canonical_metadata_overrides:
            raise ValueError("Duplicate or conflicting canonical work metadata override")
        canonical_metadata_overrides.add(wid)
        graph["works"][wid].update(title_orig=override["canonical_title_orig"],metadata_review_override=copy.deepcopy(override))
        applied_overrides.append({"kind":"work_metadata_override",**copy.deepcopy(override)})
    # Reconcile every declared item, including explicit conflicts that were
    # merely unknown in the pilot. Preserve verified pilot links verbatim.
    all_declared=set(inventory["all_declared_member_ids"])
    verified_pilot={l["item_id"] for l in graph["item_work_links"] if l["status"] in ACCEPTED_STATUSES}
    graph["item_work_links"]=[l for l in graph["item_work_links"] if l["item_id"] in verified_pilot or (l["item_id"] not in raw_to_work and l["item_id"] not in all_declared)]
    accepted_raw=set(verified_pilot)
    for (gid,iid),link in sorted(accepted_links.items()):
        if iid in accepted_raw:
            continue
        graph["item_work_links"].append({**copy.deepcopy(link),"work_id":raw_to_work[iid],"status":"accepted_annotation",
            "source_known_at":None,"edition_id":"kion-raw:"+str(iid),"cut_identity_status":"unknown",
            "evidence_ids":evidence(gid,link["evidence_ids"]),"independent_review_id":review["review_id"]})
        accepted_raw.add(iid)
    # An accepted merge alone does not insert an unreviewed raw card as a new
    # candidate/work mapping. It only deduplicates the already admitted links.
    integration_rejections=[]
    scope_overrides={}
    for override in review.get("required_scope_overrides",[]):
        key=(override["group_id"],override["scope_id"])
        if key in scope_overrides:
            raise ValueError("Duplicate required scope override")
        scope_overrides[key]=override
    for gid,group in groups.items():
        d=decisions[gid]
        allowed_scopes=set(d.get("accepted_scope_ids",[]))
        allowed_edges=set(d.get("accepted_edge_ids",[]))
        source_scopes=unique(group["scopes"],"scope_id","scope")
        source_edges=unique(group["edges"],"edge_id","edge")
        if not allowed_scopes<=source_scopes.keys() or not allowed_edges<=source_edges.keys():
            raise ValueError("Review accepts absent scope or edge")
        for sid in sorted(allowed_scopes):
            source=copy.deepcopy(source_scopes[sid])
            override=scope_overrides.pop((gid,sid),None)
            if override:
                allowed_fields={"work_ids","scope_complete","known_prior_complete","adjacency_complete","direct_next","scope_boundary","reason"}
                if set(override)-allowed_fields-{"group_id","scope_id"}:
                    raise ValueError("Unknown field in mandatory scope override")
                if "work_ids" in override and not set(override["work_ids"])<=set(source["work_ids"]):
                    raise ValueError("Scope override may only narrow the reviewed source scope")
                source.update({k:copy.deepcopy(v) for k,v in override.items() if k in allowed_fields})
                applied_overrides.append({"kind":"required_scope_override",**copy.deepcopy(override)})
            scope_id="r2:"+gid+":"+sid
            translated=[translates[(gid,w)] for w in source["work_ids"] if (gid,w) in translates]
            if not translated:
                integration_rejections.append({"group_id":gid,"scope_id":sid,"reason":"No accepted mapped works in scope"});continue
            missing=len(translated)!=len(source["work_ids"])
            weak=source["order_kind"]=="catalog_release_precedes"
            reviewed_complete=sid in d.get("complete_scope_ids",[]) and all(e["edge_id"] in allowed_edges for e in group["edges"] if e["scope_id"]==sid)
            graph["scopes"][scope_id]={**copy.deepcopy(source),"work_ids":sorted(set(translated)),
                "role":"catalog_release_only" if weak else source.get("role","unknown"),
                "scope_complete":bool(source.get("scope_complete")) and reviewed_complete and not weak and not missing,
                "known_prior_complete":bool(source.get("known_prior_complete")) and reviewed_complete and not weak and not missing,
                "adjacency_complete":bool(source.get("adjacency_complete")) and reviewed_complete and not weak and not missing,
                "source_known_at":None,"status":"accepted_annotation","evidence_ids":evidence(gid,source["evidence_ids"])}
            for wid in sorted(set(translated)):
                graph["memberships"].append({"work_id":wid,"scope_id":scope_id,"role":"catalog_release_only" if weak else source.get("role","unknown"),
                    "status":"accepted_annotation","source_known_at":None,"evidence_ids":evidence(gid,source["evidence_ids"])})
        for eid in sorted(allowed_edges):
            source=source_edges[eid]
            sid="r2:"+gid+":"+source["scope_id"]
            a,b=translates.get((gid,source["from_work"])),translates.get((gid,source["to_work"]))
            if (sid not in graph["scopes"] or not a or not b or a==b
                or not {a,b}<=set(graph["scopes"][sid]["work_ids"])):
                integration_rejections.append({"group_id":gid,"edge_id":eid,"reason":"Missing accepted scope/endpoints or edge collapses within one reviewed work"});continue
            if source["order_kind"]!=graph["scopes"][sid]["order_kind"]:
                raise ValueError("An accepted edge cannot change its typed order relation")
            graph["edges"].append({**copy.deepcopy(source),"scope_id":sid,"from_work":a,"to_work":b,
                "direct_next":bool(source.get("direct_next")) and source["order_kind"]!="catalog_release_precedes" and graph["scopes"][sid].get("direct_next",True),
                "status":"accepted_annotation","source_known_at":None,"evidence_ids":evidence(gid,source["evidence_ids"])})
    if scope_overrides:
        raise ValueError("Mandatory scope review overrides were not applied")
    # Canonical identity aliases can map several reviewed source edges to the
    # same assertion. Keep all source IDs/evidence without duplicate edges.
    unique_edges={}
    for edge in graph["edges"]:
        key=(edge["scope_id"],edge["order_kind"],edge["from_work"],edge["to_work"])
        if key not in unique_edges:
            unique_edges[key]=edge
            if edge.get("edge_id"):
                edge["source_edge_ids"]=[edge["edge_id"]]
        else:
            target=unique_edges[key]
            target["evidence_ids"]=sorted(set(target["evidence_ids"]+edge["evidence_ids"]))
            target["source_edge_ids"]=sorted(set(target.get("source_edge_ids",[])+[edge["edge_id"]]))
            target["direct_next"]=target.get("direct_next",False) and edge.get("direct_next",False)
    graph["edges"]=list(unique_edges.values())
    present={l["item_id"] for l in graph["item_work_links"]}
    for iid in sorted(all_declared-present):
        source_statuses={r.get("status") for g in groups.values() for r in g["item_work_links"] if r["item_id"]==iid}
        exclusions=[{"group_id":gid,"reason":d.get("reason"),"required_corrections":copy.deepcopy(d.get("required_corrections",[]))}
            for gid,d in decisions.items() if iid in d.get("rejected_item_ids",[])]
        graph["item_work_links"].append({"item_id":iid,"work_id":None,"status":"conflict" if "conflict" in source_statuses or exclusions else "unknown",
            "edition_id":"kion-raw:"+str(iid),"cut_identity_status":"unknown","source_known_at":None,
            "evidence_ids":["r2:independent-review"],"review_exclusions":exclusions,
            "reason":"Independently rejected work identity; see exact review corrections" if exclusions else "Explicit inventory item; work match was not independently admitted"})
    links={l["item_id"]:l for l in graph["item_work_links"]}
    graph["group_inventory"]=[]
    for gid,source in sorted(inv.items()):
        items=[]
        for iid in source["member_ids"]:
            link=links[iid];wid=link.get("work_id")
            scopes=[sid for sid,s in graph["scopes"].items() if wid and wid in s["work_ids"] and (s.get("group_id")==gid or sid.startswith(gid+":"))]
            items.append({"item_id":iid,"work_id":wid,"mapping_status":link["status"],"accepted_order_scope_ids":scopes,
                "order_status":"mapped_with_order_scope" if scopes else "mapped_order_unknown" if wid else "conflict" if link["status"]=="conflict" else "unknown"})
        graph["group_inventory"].append({"group_id":gid,"baseline_source_ids":source["baseline_source_ids"],"review_decision":decisions[gid]["decision"],
            "proposed_disposition":groups[gid]["disposition"],"reason":decisions[gid].get("reason"),"items":items})
    baseline=set(inventory["baseline_raw_ids"])
    graph["coverage"]={"inventory_groups":len(inv),"baseline_raw_ids":len(baseline),"baseline_U_ids":len(inventory["baseline_train_known_ids"]),
        "declared_raw_ids":len(all_declared),"baseline_disposition_count":len(baseline&links.keys()),
        "accepted_raw_links":len({l["item_id"] for l in graph["item_work_links"] if l["status"] in ACCEPTED_STATUSES}),
        "baseline_accepted_raw_links":sum(links[i]["status"] in ACCEPTED_STATUSES for i in baseline),
        "unknown_or_conflict_raw_links":sum(l["status"] not in ACCEPTED_STATUSES for l in graph["item_work_links"]),
        "works":len(graph["works"]),"scopes_by_kind":dict(Counter(s["order_kind"] for s in graph["scopes"].values())),
        "edges_by_kind":dict(Counter(e["order_kind"] for e in graph["edges"])),"integration_rejections":integration_rejections}
    graph=copy.deepcopy(graph)
    validate_graph(graph)
    graph["graph_sha256"]=canonical_hash(graph)
    return graph


def _query_clock(query_date, metadata_cutoff, temporal_mode, history_cap, order_kind):
    if temporal_mode not in ("historical_as_of", "retrospective_current"):
        raise ValueError("Explicit temporal mode required")
    if order_kind is not None and order_kind not in ORDER_KINDS:
        raise ValueError("Unsupported typed order relation")
    if history_cap != 63:
        raise ValueError("This contract is H63 only")
    return _date(query_date), _date(metadata_cutoff)


def _validate_mapping(mapping):
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 2 for v in mapping.values()):
        raise ValueError("Train-known mapping must exclude specials")


def _date_view(graph, cutoff, temporal_mode, order_kind):
    available = lambda rec: temporal_mode == "retrospective_current" or (rec.get("source_known_at") is not None and _date(rec["source_known_at"]) < cutoff)
    release_eligible = {w for w, rec in graph["works"].items() if rec.get("release_upper") and _date(rec["release_upper"]) < cutoff}
    eligible = {w for w in release_eligible if available(graph["works"][w])}
    link_candidates = defaultdict(list)
    for link in graph["item_work_links"]:
        if link["status"] in ACCEPTED_STATUSES and link.get("work_id") in eligible and available(link):
            link_candidates[str(link["item_id"])].append(link["work_id"])
    links = {raw: vals[0] for raw, vals in link_candidates.items() if len(set(vals)) == 1}
    membership = defaultdict(list)
    for member in graph["memberships"]:
        sid = member["scope_id"]
        if (member["status"] in ACCEPTED_STATUSES and available(member) and available(graph["scopes"][sid])
            and (order_kind is None or graph["scopes"][sid]["order_kind"] == order_kind)):
            membership[member["work_id"]].append(sid)
    return available, release_eligible, eligible, links, membership


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


class PreparedGraph:
    """An owned immutable graph/mapping snapshot with bounded exact-date views.

    Construct through :func:`prepare_graph`. The cache belongs to this snapshot;
    it never accepts or looks up a mutable caller graph by object identity.
    """
    __slots__ = ("graph", "mapping", "order_kinds", "scope_policies", "graph_sha256",
        "mapping_sha256", "_policy_hashes", "_views", "_max_views")

    def __init__(self, graph, mapping, order_kinds, policies, max_views):
        # Validation and hashes precede freezing because canonical JSON is the
        # source contract. These snapshots are never reserialized in scoring.
        validate_graph(graph)
        _validate_mapping(mapping)
        for kind, policy in policies.items():
            if policy is not None and not isinstance(policy, dict):
                raise ValueError("Scope policy must be a work-to-scope mapping")
            for work, scope_id in (policy or {}).items():
                scope = graph["scopes"].get(scope_id) if isinstance(scope_id, str) else None
                if (work not in graph["works"] or not isinstance(scope, dict) or work not in scope["work_ids"]
                        or kind is not None and scope["order_kind"] != kind):
                    raise ValueError("Scope policy must bind a containing scope of the same order kind")
        graph_hash = graph["graph_sha256"] if "graph_sha256" in graph else canonical_hash(graph)
        policy_hashes = {kind: canonical_hash(policy) if policy else None for kind, policy in policies.items()}
        for name, value in (("graph", _freeze(graph)), ("mapping", _freeze(mapping)),
                ("order_kinds", order_kinds), ("scope_policies", _freeze(policies)),
                ("graph_sha256", graph_hash), ("mapping_sha256", canonical_hash(mapping)),
                ("_policy_hashes", MappingProxyType(policy_hashes)), ("_views", OrderedDict()), ("_max_views", max_views)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("Prepared graph identity is immutable")

    def __delattr__(self, name):
        raise AttributeError("Prepared graph identity is immutable")

    def candidate_progress(self, history_rows, candidate_item_ids, query_date, metadata_cutoff,
            temporal_mode="retrospective_current", history_cap=63, order_kind=None):
        q, cutoff = _query_clock(query_date, metadata_cutoff, temporal_mode, history_cap, order_kind)
        if order_kind not in self.order_kinds:
            raise ValueError("Order kind is outside the prepared namespace")
        key = (cutoff.toordinal(), temporal_mode, order_kind)
        if key not in self._views:
            self._views[key] = _date_view(self.graph, cutoff, temporal_mode, order_kind)
            if len(self._views) > self._max_views:
                self._views.popitem(last=False)
        self._views.move_to_end(key)
        return _candidate_progress(history_rows, candidate_item_ids, self.mapping, self.graph, q, cutoff,
            temporal_mode, order_kind, self.scope_policies[order_kind], self.graph_sha256,
            self._policy_hashes[order_kind], self._views[key])


def prepare_graph(graph, metadata, *, order_kinds=("mainline_release", "catalog_release_precedes"),
        scope_policies=None, max_views=4):
    """Validate once and bind exact ordered branches/policies to owned data.

    Subsequent mutations of input dictionaries cannot change this snapshot.
    ``None`` is supported only when explicitly declared for the legacy branch.
    No nearest-date reuse is performed; historical source availability remains
    keyed by the exact metadata cutoff and temporal mode.
    """
    kinds = tuple(order_kinds)
    if (not kinds or any(kind is not None and (not isinstance(kind, str) or kind not in ORDER_KINDS) for kind in kinds)
            or len(set(kinds)) != len(kinds)):
        raise ValueError("Prepared order kinds must be distinct supported typed branches")
    if isinstance(max_views, bool) or not isinstance(max_views, int) or max_views < 1:
        raise ValueError("Prepared graph requires a positive bounded view cache")
    policies = {kind: None for kind in kinds} if scope_policies is None else copy.deepcopy(scope_policies)
    if not isinstance(policies, dict) or set(policies) != set(kinds):
        raise ValueError("Scope policies must bind each prepared order kind")
    return PreparedGraph(copy.deepcopy(graph), copy.deepcopy(metadata["item_id_to_token"]), kinds, policies, max_views)


def candidate_progress(history_rows, candidate_item_ids, metadata, graph, query_date,
        metadata_cutoff, temporal_mode="retrospective_current", history_cap=63, order_kind=None, scope_policy=None):
    """Pure CPU crosses; mutable untrusted graphs are validated on every call.

    ``history_rows`` are mappings with date/item_id (optional token is checked).
    Selection precedes train-known/work/branch mapping. Use ``prepare_graph``
    once for repeated queries against an immutable graph and mapping snapshot.
    """
    validate_graph(graph)
    q, cutoff = _query_clock(query_date, metadata_cutoff, temporal_mode, history_cap, order_kind)
    mapping = metadata["item_id_to_token"]
    _validate_mapping(mapping)
    graph_hash = graph["graph_sha256"] if "graph_sha256" in graph else canonical_hash(graph)
    return _candidate_progress(history_rows, candidate_item_ids, mapping, graph, q, cutoff,
        temporal_mode, order_kind, scope_policy, graph_hash, canonical_hash(scope_policy) if scope_policy else None,
        _date_view(graph, cutoff, temporal_mode, order_kind))


def _candidate_progress(history_rows, candidate_item_ids, mapping, graph, q, cutoff,
        temporal_mode, order_kind, scope_policy, graph_hash, policy_hash, view):
    available, release_eligible, eligible, links, membership = view
    selected = sorted((r for r in history_rows if _date(r["date"]) < q),
        key=lambda r: (_date(r["date"]), int(r["item_id"]))) [-63:]
    observed = []
    for row in selected:
        raw = str(row["item_id"])
        token = mapping.get(raw, 2)
        if token > 2 and row.get("token", token) == token and raw in links:
            observed.append((links[raw], _date(row["date"])))
    matrix = np.zeros((len(candidate_item_ids), len(COLUMNS)), dtype=np.float32)
    diagnostics = []
    for index, candidate in enumerate(candidate_item_ids):
        raw = str(candidate)
        out = dict.fromkeys(COLUMNS, 0.)
        out.update(n63=len(selected), known_raw_rows=len(observed), unmapped_rows=len(selected)-len(observed),
            candidate_mapping_unknown=1., prior_share_missing=1., next_missing=1., recency_missing=1., distance_missing=1.)
        work = links.get(raw) if mapping.get(raw, 2) > 2 else None
        scopes = sorted(set(membership.get(work, [])))
        if work is not None and scope_policy and work in scope_policy:
            chosen=scope_policy[work]
            if chosen not in graph["scopes"] or work not in graph["scopes"][chosen]["work_ids"]:
                raise ValueError("Explicit scope policy must select a scope containing the candidate work")
            scopes=[sid for sid in scopes if sid==chosen]
        if work is not None:
            out["candidate_mapping_unknown"] = 0.
        if len(scopes) > 1:
            out["branch_ambiguous"] = 1.
        if work is not None and len(scopes) == 1:
            sid = scopes[0]
            scope = graph["scopes"][sid]
            scope_works = {wid for wid in scope["work_ids"] if wid in eligible and sid in membership.get(wid, [])}
            seen_dates = defaultdict(list)
            for wid, day in observed:
                if wid in scope_works:
                    seen_dates[wid].append(day)
            seen = set(seen_dates)
            parents, children, direct = defaultdict(set), defaultdict(set), set()
            for edge in graph["edges"]:
                if edge["scope_id"] == sid and edge["status"] in ACCEPTED_STATUSES and available(edge) and {edge["from_work"], edge["to_work"]} <= scope_works:
                    a, b = edge["from_work"], edge["to_work"]
                    parents[b].add(a)
                    children[a].add(b)
                    if edge.get("direct_next"):
                        direct.add((a, b))
            ancestors, stack = set(), list(parents[work])
            while stack:
                wid = stack.pop()
                if wid not in ancestors:
                    ancestors.add(wid)
                    stack.extend(parents[wid])
            prior_seen = ancestors & seen
            prior_missing = ancestors - seen
            frontier = work not in seen and bool(parents[work] & seen) and not prior_missing
            # Missing historical assertions invalidate completeness, even if a
            # scope had been reviewed complete in a later snapshot.
            release_scope = set(scope["work_ids"]) & release_eligible
            unknown_release = {wid for wid in scope["work_ids"] if not graph["works"][wid].get("release_upper")}
            assertions = [e for e in graph["edges"] if e["scope_id"] == sid and {e["from_work"], e["to_work"]} <= release_scope and e["status"] in ACCEPTED_STATUSES]
            complete = bool(scope.get("scope_complete")) and bool(scope.get("known_prior_complete")) and all(available(e) for e in assertions) and all(
                sid in membership.get(wid, []) for wid in release_scope) and release_scope <= scope_works and not unknown_release
            adjacency = bool(scope.get("adjacency_complete")) and complete
            out.update(graph_available=1., scope_complete=complete, adjacency_complete=adjacency,
                known_prior_complete=complete, n_branch_seen=len(seen), n_prior=len(ancestors),
                n_prior_seen=len(prior_seen), n_prior_missing=len(prior_missing), frontier=frontier,
                direct_next=frontier and adjacency and any((v, work) in direct for v in seen),
                next_missing=not adjacency, skip_observed_priors=work not in seen and bool(prior_seen) and bool(prior_missing),
                already_seen_work=work in seen, branch_cold=not seen)
            if ancestors:
                out.update(prior_seen_share=len(prior_seen)/len(ancestors), prior_share_missing=0.)
            if seen:
                out.update(recency_days=(q-max(d for dates in seen_dates.values() for d in dates)).days, recency_missing=0.)
            queue, visited = deque((w, 0) for w in seen), set(seen)
            while queue:
                current, distance = queue.popleft()
                if current == work and distance > 0:
                    out.update(graph_hop_distance=distance, distance_missing=0.)
                    break
                for nxt in children[current]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, distance+1))
            diagnostics.append({"candidate_item_id": int(raw), "work_id": work, "scope_id": sid,
                "observed_work_ids": sorted(seen), "known_prior_ids": sorted(ancestors), "missing_prior_ids": sorted(prior_missing)})
        else:
            diagnostics.append({"candidate_item_id": int(raw), "work_id": work, "scope_id": None})
        matrix[index] = [out[name] for name in COLUMNS]
    return {"values": matrix, "columns": list(COLUMNS), "candidate_item_ids": list(candidate_item_ids),
        "diagnostics": diagnostics, "provenance": {"contract": CONTRACT, "graph_sha256": graph_hash,
            "query_date": q.isoformat(), "metadata_cutoff": cutoff.isoformat(), "temporal_mode": temporal_mode,
            "order_kind":order_kind,"scope_policy_sha256":policy_hash,
            "history_policy": "strict date<q; sort(date,raw_item_id); last63 before mapping; observed last-pair snapshot",
            "outcomes_used": False, "candidate_filtering": False}}
