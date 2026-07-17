"""
Given:
  - a fact table produced by fact_decompose.decompose_text()
      entity_name -> {attr_type: [values, ...], ...}
  - a question string

Try to extract an answer, generically:
  1. Parse the question -> candidates (the "X or Y" entities being compared)
                          -> comparator ("first"/"earlier"/"more"/"taller"...)
                          -> attribute type needed (DATE / QUANTITY / ...),
                             inferred from the predicate via embedding
                             similarity to a small seed bank (data-driven,
                             not per-question rules)
  2. Link each candidate name to a key in the fact table (fuzzy match,
     since surface forms won't match exactly, e.g. "Arthur's Magazine"
     vs "Arthur's Magazine (1844-1846)")
  3. Pull the requested attribute type's value for each candidate
  4. If every candidate has a value -> parse + compare -> return answer
     If any candidate is missing it  -> return "not answerable", say why
"""

import re
import json
import difflib
import spacy

from fact_decompose import decompose_text

nlp = spacy.load("en_core_web_sm")


def similarity(a, b):
    """Self-contained fuzzy string similarity (no model download needed)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Plain-language names for spaCy's NER labels, used only in human-facing
# "reason" messages below (the labels themselves are still used as-is
# everywhere else, e.g. as fact_table keys).
ATTR_LABEL_NAMES = {
    "DATE": "date", "GPE": "place", "PERSON": "person", "ORG": "organization",
    "NORP": "nationality/group", "QUANTITY": "quantity", "CARDINAL": "number",
    "MONEY": "money", "PERCENT": "percent", "WORK_OF_ART": "title/work",
    "PRODUCT": "product", "EVENT": "event", "LAW": "law", "FAC": "facility",
    "LOC": "location", "TIME": "time", "ORDINAL": "ordinal", "LANGUAGE": "language",
}


def pretty_attr(attr_type):
    return ATTR_LABEL_NAMES.get(attr_type, (attr_type or "matching").lower())


# ----------------------------------------------------------------------
# 1a. Split the question into candidates via "or"-coordination
# ----------------------------------------------------------------------
def extract_candidates(question):
    """Split into two compared candidates around the literal ' or '.
    Right side = everything after 'or' (trimmed). Left side = the
    trailing proper-noun/title-case span right before 'or' (since the
    left side of the question usually also contains the wh-word/verb
    prefix, e.g. 'Which magazine was started first Arthur's Magazine').
    This avoids relying on dependency-parsed 'conj', which breaks on
    titles containing number-like words (e.g. 'First' tagged ORDINAL)."""
    q = question.strip().rstrip("?").strip()
    if " or " not in q.lower():
        return []

    idx = q.lower().rfind(" or ")
    left_text = q[:idx].strip()
    right_text = q[idx + 4:].strip()

    doc = nlp(left_text)
    # walk tokens right-to-left, keep collecting while they look like part
    # of a proper-noun / title span (PROPN, NOUN, PART, NUM, DET as filler)
    keep = []
    for token in reversed(doc):
        if token.pos_ in ("PROPN", "NOUN", "NUM", "PART", "PUNCT") or token.text[0].isupper():
            keep.insert(0, token.text)
        else:
            break
    left_candidate = " ".join(keep) if keep else left_text

    return [left_candidate, right_text]


# ----------------------------------------------------------------------
# 1b. Infer requested attribute type from the question's predicate/comparator
#     (embedding nearest-neighbor over a small seed bank -- data-driven,
#     reusable for ANY question, not hand-coded per question)
# ----------------------------------------------------------------------
ATTRIBUTE_SEED_BANK = {
    "found": "DATE", "start": "DATE", "begin": "DATE",
    "establish": "DATE", "create": "DATE", "bear": "DATE",
    "old": "DATE", "early": "DATE", "first": "DATE",
    "tall": "QUANTITY", "short": "QUANTITY", "big": "QUANTITY",
    "large": "QUANTITY", "heavy": "QUANTITY", "population": "CARDINAL",
    "locate": "GPE", "base": "GPE",
}


def infer_attribute_type(question):
    """Map the question's predicate/comparator word to an attribute type
    by lemma lookup against a small seed bank. Generic in the sense that
    the SAME lookup logic runs for any question -- only the seed bank
    needs occasional extension for new domains (e.g. add 'weigh'->QUANTITY)."""
    doc = nlp(question)
    candidate_lemmas = [t.lemma_.lower() for t in doc if t.pos_ in ("VERB", "ADJ", "ADV")]
    for lemma in candidate_lemmas:
        if lemma in ATTRIBUTE_SEED_BANK:
            return ATTRIBUTE_SEED_BANK[lemma], lemma, 1.0
    # fallback: fuzzy match against seed bank keys
    best_attr, best_word, best_score = None, None, -1
    for lemma in candidate_lemmas:
        for seed, attr in ATTRIBUTE_SEED_BANK.items():
            score = similarity(lemma, seed)
            if score > best_score:
                best_attr, best_word, best_score = attr, seed, score
    return best_attr, best_word, best_score


# ----------------------------------------------------------------------
# 1c. Detect comparator direction: do we want MIN or MAX?
# ----------------------------------------------------------------------
MIN_WORDS = {"first", "earlier", "earliest", "older", "oldest", "before", "smaller", "shorter", "less"}
MAX_WORDS = {"last", "later", "latest", "newer", "newest", "after", "bigger", "biggest", "taller", "more", "larger"}


def infer_direction(question):
    tokens = {t.text.lower() for t in nlp(question)}
    if tokens & MIN_WORDS:
        return "min"
    if tokens & MAX_WORDS:
        return "max"
    return "min"  # default assumption for "which X was Y first" style questions


# ----------------------------------------------------------------------
# 2. Fuzzy-link a candidate name to a fact-table key
# ----------------------------------------------------------------------
def link_candidate(name, fact_table_keys):
    if not fact_table_keys:
        return None
    best_key, best_score = None, -1
    for key in fact_table_keys:
        score = similarity(name, key)
        if score > best_score:
            best_key, best_score = key, score
    return best_key if best_score > 0.4 else None


# ----------------------------------------------------------------------
# 3. Parse a DATE-type value down to a sortable year (best-effort, generic)
# ----------------------------------------------------------------------
def value_to_sort_key(value, attr_type):
    if attr_type == "DATE":
        years = re.findall(r"\b(1\d{3}|20\d{2})\b", value)
        if years:
            return int(years[0])
        century = re.search(r"(\d+)(st|nd|rd|th)\s+century", value)
        if century:
            return (int(century.group(1)) - 1) * 100
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))
    return float(nums[0]) if nums else None


# ----------------------------------------------------------------------
# 4. Full pipeline: question + fact_table -> answer (or "not answerable")
# ----------------------------------------------------------------------
def answer_from_fact_table(question, fact_table):
    candidates = extract_candidates(question)
    attr_type, matched_word, conf = infer_attribute_type(question)
    direction = infer_direction(question)

    if not candidates:
        return {"answerable": False, "reason": "Couldn't find two things to compare in that question — try phrasing it as \"X or Y\"."}

    linked = {c: link_candidate(c, fact_table.keys()) for c in candidates}
    values, missing = {}, []

    for c, key in linked.items():
        raw_values = fact_table.get(key, {}).get(attr_type, []) if key else []
        sort_keys = [value_to_sort_key(v, attr_type) for v in raw_values]
        sort_keys = [s for s in sort_keys if s is not None]
        if sort_keys:
            values[c] = min(sort_keys) if attr_type == "DATE" else sort_keys[0]
        else:
            missing.append(c)

    if missing:
        return {
            "answerable": False,
            "reason": f"Couldn't find a {pretty_attr(attr_type)} for {', '.join(missing)}.",
            "candidates": candidates,
            "linked_entities": linked,
            "attribute_type_used": attr_type,
            "inferred_from_word": matched_word,
        }

    answer = min(values, key=values.get) if direction == "min" else max(values, key=values.get)
    return {
        "answerable": True,
        "answer": answer,
        "values": values,
        "attribute_type_used": attr_type,
        "direction": direction,
        "linked_entities": linked,
    }


# ----------------------------------------------------------------------
# 5. Direct (non-comparison) questions, e.g. "When was X renovated?"
#
# The fact table flattens every DATE (etc.) found for an entity into one
# list regardless of which predicate produced it, so "Taj Mahal" ends up
# with DATE: ["1999", "2000"] whether it was built in 1999 or renovated in
# 2000. A plain attribute-type lookup can't tell those apart. This path
# instead matches the question's own predicate ("renovate") against the
# entity's individual triples (which still carry their predicate) and
# reads the attribute off the matching triple only.
# ----------------------------------------------------------------------
WH_ATTRIBUTE_MAP = {
    "when": "DATE",
    "where": "GPE",
    "who": "PERSON",
    "whom": "PERSON",
}


def infer_wh_attribute(question):
    """Map a question's wh-word to an expected attribute type."""
    tokens_lower = [t.text.lower() for t in nlp(question)]
    if "how" in tokens_lower:
        idx = tokens_lower.index("how")
        nxt = tokens_lower[idx + 1] if idx + 1 < len(tokens_lower) else ""
        if nxt in ("many", "much"):
            return "CARDINAL"
        if nxt in ("tall", "big", "large", "long", "old", "far"):
            return "QUANTITY"
    for word, attr in WH_ATTRIBUTE_MAP.items():
        if word in tokens_lower:
            return attr
    return None


def extract_question_predicate(question):
    """Pull the main content-verb lemma from a question, preferring a
    non-'be' verb (e.g. 'was renovated' -> 'renovate') since the copula
    itself never distinguishes between competing facts about an entity."""
    doc = nlp(question)
    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    non_be = [t for t in verbs if t.lemma_.lower() != "be"]
    chosen = non_be[0] if non_be else (verbs[0] if verbs else None)
    return chosen.lemma_.lower() if chosen else None


WH_WORDS = {"who", "whom", "when", "where", "how", "which", "what", "why"}


def extract_question_entity_span(question):
    """Generic entity-span guess for non-comparison questions: the longest
    run of PROPN/NOUN tokens in the question, skipping wh-words."""
    doc = nlp(question)
    spans, current = [], []
    for token in doc:
        if token.pos_ in ("PROPN", "NOUN") and token.text.lower() not in WH_WORDS:
            current.append(token.text)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return max(spans, key=len) if spans else None


def answer_direct_question(question, sentence_facts, fact_table):
    """Fallback for questions with no 'X or Y' comparison. Links the
    question's entity span to a fact-table key, matches the question's
    predicate against that entity's triples, and reads the requested
    attribute type off the matching triple (not the flattened fact table)."""
    wh_attr = infer_wh_attribute(question)
    predicate = extract_question_predicate(question)
    entity_span = extract_question_entity_span(question)

    trace = {
        "mode": "direct",
        "wh_attribute": wh_attr,
        "question_predicate": predicate,
        "question_entity_span": entity_span,
    }

    if not entity_span:
        trace["answerable"] = False
        trace["reason"] = "Couldn't tell what or who the question is about."
        return trace

    linked_key = link_candidate(entity_span, fact_table.keys())
    trace["linked_key"] = linked_key
    trace["linked_similarity"] = (
        round(similarity(entity_span, linked_key), 3) if linked_key else None
    )

    if not linked_key:
        trace["answerable"] = False
        trace["reason"] = f"Couldn't find \"{entity_span}\" anywhere in the text."
        return trace

    candidate_triples = []
    for sf in sentence_facts:
        for t in sf["triples"]:
            if t["subject"] and similarity(t["subject"], linked_key) > 0.4:
                pred_sim = similarity(t["predicate"], predicate) if predicate else 0.0
                candidate_triples.append(
                    {
                        "predicate": t["predicate"],
                        "predicate_similarity": round(pred_sim, 3),
                        "attributes": t["attributes"],
                        "sentence": t["sentence"],
                    }
                )
    trace["candidate_triples"] = candidate_triples

    if not candidate_triples:
        trace["answerable"] = False
        trace["reason"] = f"Found \"{linked_key}\" in the text, but no facts about it."
        return trace

    exact = [c for c in candidate_triples if predicate and c["predicate"] == predicate]
    best = exact[0] if exact else max(candidate_triples, key=lambda c: c["predicate_similarity"])
    trace["matched_triple"] = best

    attr_values = best["attributes"].get(wh_attr, []) if wh_attr else []
    if not attr_values:
        attr_values = [v for vals in best["attributes"].values() for v in vals]

    if not attr_values:
        trace["answerable"] = False
        trace["reason"] = (
            f"Found a matching fact (\"{best['predicate']}\") but it doesn't "
            f"mention a {pretty_attr(wh_attr)}."
        )
        return trace

    trace["answerable"] = True
    trace["answer"] = ", ".join(attr_values)
    return trace


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    text = (
        "Arthur's Magazine (1844-1846) was an American literary periodical "
        "published in Philadelphia in the 19th century. "
        "First for Women is a woman's magazine published by Bauer Media "
        "Group in the USA."
    )
    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    decomposed = decompose_text(text)
    result = answer_from_fact_table(question, decomposed["fact_table"])
    print(json.dumps(result, indent=2))

    print("\n--- direct-question demo ---")
    text2 = "Taj Mahal was built in 1999. Taj Mahal was renovated in 2000."
    question2 = "When was Taj Mahal renovated?"
    decomposed2 = decompose_text(text2)
    result2 = answer_direct_question(
        question2, decomposed2["sentence_facts"], decomposed2["fact_table"]
    )
    print(json.dumps(result2, indent=2))
