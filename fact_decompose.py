"""
Generic (non-LLM) sentence -> structured-facts decomposer.

Pipeline:
  1. Split text into sentences (spaCy sentencizer)
  2. NER pass          -> pulls out typed attributes (DATE, GPE, ORG, NORP, ...)
  3. Dependency walk   -> pulls out (subject, predicate, object/attr) triples
                          by generically walking the parse tree for every
                          clause (not hand-coded per sentence/question type)
  4. Attribute attachment -> attaches NER spans found inside a clause to the
                          triple(s) of that clause, so triples carry typed
                          attributes (e.g. published-in -> GPE / DATE)
  5. Merge everything into one fact table: entity -> {attr_type: value, ...}
                                             plus a flat list of raw triples

This is fully generic: nothing here is written for this specific sentence.
It walks the dependency tree the same way for any input.
"""

import json
import spacy

nlp = spacy.load("en_core_web_sm")


# ----------------------------------------------------------------------
# Step A: generic clause splitter — find each verb's clause root
# ----------------------------------------------------------------------
def get_clause_roots(sent):
    """A sentence can contain multiple clauses (main + conjuncts + relative
    clauses). We find every verb that acts as a predicate head."""
    roots = []
    for token in sent:
        if token.pos_ in ("VERB", "AUX") and token.dep_ in (
            "ROOT", "conj", "relcl", "advcl", "ccomp", "xcomp", "acl",
        ):
            roots.append(token)
    if not roots:
        roots = [sent.root]
    return roots


# ----------------------------------------------------------------------
# Step B: generic argument extraction around a predicate
# ----------------------------------------------------------------------
def span_text(token):
    """Expand a token to its full subtree span (so 'Arthur's Magazine'
    rather than just 'Magazine')."""
    subtree = list(token.subtree)
    start = min(t.i for t in subtree)
    end = max(t.i for t in subtree) + 1
    return token.doc[start:end].text


def extract_triple_for_predicate(pred):
    """Generic SVO+PP extraction: for a given predicate token, find its
    subject, direct object / complement, and any prepositional phrases
    attached to it. Works for any verb, not just 'started'/'published'."""

    subject, obj, preps = None, None, []

    for child in pred.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subject = span_text(child)
        elif child.dep_ in ("attr", "dobj", "oprd", "acomp"):
            obj = span_text(child)
        elif child.dep_ == "prep":
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            if pobj is not None:
                preps.append({"prep": child.text, "object": span_text(pobj)})

    # passive voice: real subject is often attached via agent 'by'
    agent_prep = next((p for p in preps if p["prep"].lower() == "by"), None)

    return {
        "predicate": pred.lemma_,
        "subject": subject,
        "object": obj,
        "prepositions": preps,
        "agent": agent_prep["object"] if agent_prep else None,
    }


# ----------------------------------------------------------------------
# Step C: NER pass, scoped per sentence
# ----------------------------------------------------------------------
def get_entities(sent):
    return [{"text": e.text, "label": e.label_} for e in sent.ents]


def attach_entities_to_triple(triple, entities):
    """Any NER span that textually appears inside the object or a
    preposition's object gets attached as a typed attribute on the triple."""
    attrs = {}
    fields_to_check = [triple.get("object", "") or ""]
    fields_to_check += [p["object"] for p in triple["prepositions"]]
    blob = " ".join(fields_to_check)

    for ent in entities:
        if ent["text"] in blob:
            attrs.setdefault(ent["label"], []).append(ent["text"])
    return attrs


# ----------------------------------------------------------------------
# Step D: full decomposition for one sentence
# ----------------------------------------------------------------------
def fallback_subject(sent):
    """If dependency parse gives no nsubj (happens on stylized titles / names
    that get mis-split by noun-chunking, e.g. 'First' tagged ORDINAL inside
    'First for Women'). Fall back to the full span of tokens preceding the
    sentence's main verb — a generic, parser-agnostic heuristic."""
    root = sent.root
    pre_root_tokens = [t for t in sent if t.i < root.i]
    if pre_root_tokens:
        start = pre_root_tokens[0].i
        end = pre_root_tokens[-1].i + 1
        return sent.doc[start:end].text
    chunks = list(sent.noun_chunks)
    return chunks[0].text if chunks else None


def decompose_sentence(sent):
    entities = get_entities(sent)
    triples = []
    default_subject = fallback_subject(sent)
    for pred in get_clause_roots(sent):
        triple = extract_triple_for_predicate(pred)
        if not triple["subject"]:
            triple["subject"] = default_subject
        triple["attributes"] = attach_entities_to_triple(triple, entities)
        triple["sentence"] = sent.text.strip()
        triples.append(triple)
    return {
        "sentence": sent.text.strip(),
        "entities": entities,
        "triples": triples,
    }


# ----------------------------------------------------------------------
# Step E: decompose full text + roll up into an entity fact table
# ----------------------------------------------------------------------
def decompose_text(text):
    doc = nlp(text)
    sentence_facts = [decompose_sentence(sent) for sent in doc.sents]

    # Roll everything up into: entity_name -> {attr_type: [values]}
    fact_table = {}
    for sf in sentence_facts:
        for triple in sf["triples"]:
            subj = triple["subject"]
            if not subj:
                continue
            fact_table.setdefault(subj, {})
            for attr_type, values in triple["attributes"].items():
                fact_table[subj].setdefault(attr_type, [])
                for v in values:
                    if v not in fact_table[subj][attr_type]:
                        fact_table[subj][attr_type].append(v)

    return {"sentence_facts": sentence_facts, "fact_table": fact_table}


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

    result = decompose_text(text)
    print(json.dumps(result, indent=2))
