"""FastAPI backend for the Fact Decomposition + Answer Extractor visualizer.

Wraps the real fact_decompose.py / answer_extractor.py pipeline (unmodified)
and exposes every intermediate step as JSON so the static UI in ./static can
render each stage: NER, clause/triple extraction, fact-table rollup, question
parsing, attribute/direction inference, candidate linking, value comparison,
and the final answer.

fact_decompose.py and answer_extractor.py live alongside this file, so no
sys.path manipulation is needed -- Python already adds a script's own
directory to sys.path when it's run directly.

Run with:
    python server.py
then open http://127.0.0.1:8008
"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from answer_extractor import (
    answer_direct_question,
    extract_candidates,
    infer_attribute_type,
    infer_direction,
    link_candidate,
    pretty_attr,
    similarity,
    value_to_sort_key,
)
from fact_decompose import decompose_text

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Fact Decompose + Answer Extractor Visualizer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AnalyzeRequest(BaseModel):
    knowledge: str
    question: str


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def analyze_question(question: str, decomposed: dict) -> dict:
    """Re-runs answer_extractor's pipeline step by step (same logic as
    answer_from_fact_table) but keeps every intermediate value instead of
    only the final verdict, so the UI can show each stage.

    Two question shapes are handled, matching answer_extractor.py's two
    entry points:
      - comparison questions ("X or Y") -> extract_candidates() path
      - direct questions ("When was X renovated?") -> answer_direct_question()
    """
    fact_table = decomposed["fact_table"]
    candidates = extract_candidates(question)

    if not candidates:
        trace = answer_direct_question(question, decomposed["sentence_facts"], fact_table)
        return trace

    attr_type, matched_word, conf = infer_attribute_type(question)
    direction = infer_direction(question)

    trace = {
        "mode": "comparison",
        "candidates": candidates,
        "attribute_type_used": attr_type,
        "inferred_from_word": matched_word,
        "confidence": round(conf, 3) if conf is not None else None,
        "direction": direction,
        "linked_entities": {},
        "raw_values": {},
    }

    fact_table_keys = list(fact_table.keys())
    linked = {}
    for c in candidates:
        key = link_candidate(c, fact_table_keys)
        score = similarity(c, key) if key else None
        linked[c] = {"linked_key": key, "similarity": round(score, 3) if score is not None else None}
    trace["linked_entities"] = linked

    values, missing, raw = {}, [], {}
    for c in candidates:
        key = linked[c]["linked_key"]
        raw_values = fact_table.get(key, {}).get(attr_type, []) if key else []
        parsed = [{"value": v, "sort_key": value_to_sort_key(v, attr_type)} for v in raw_values]
        usable = [p for p in parsed if p["sort_key"] is not None]
        raw[c] = {"raw_values": raw_values, "parsed": parsed}

        if usable:
            chosen = min(usable, key=lambda p: p["sort_key"]) if attr_type == "DATE" else usable[0]
            values[c] = chosen["sort_key"]
        else:
            missing.append(c)
    trace["raw_values"] = raw

    if missing:
        trace["answerable"] = False
        trace["reason"] = f"Couldn't find a {pretty_attr(attr_type)} for {', '.join(missing)}."
        return trace

    answer = min(values, key=values.get) if direction == "min" else max(values, key=values.get)
    trace["answerable"] = True
    trace["values"] = values
    trace["answer"] = answer
    return trace


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    decomposed = decompose_text(req.knowledge)
    answer_trace = analyze_question(req.question, decomposed)
    return {
        "input": {"knowledge": req.knowledge, "question": req.question},
        "decompose": decomposed,
        "answer_extraction": answer_trace,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8008))
    uvicorn.run(app, host="0.0.0.0", port=port)
