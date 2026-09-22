"""Build the shared evaluation artifact, including iterative-attack summaries."""
from .attacks import is_iterative, summarize_attack


def build_evaluation_payload(data, rows, scored, summary, *, source, kind, **identity):
    """Keep both judge CLIs on the same output schema and aggregation rules."""
    payload = {
        **identity,
        "source": source,
        "source_model": data.get("model"),
        "source_config": data.get("config"),
        "evaluation_scope": data.get("evaluation_scope"),
        "summary": summary,
        "results": scored,
    }
    if is_iterative(data):
        payload["attempt_summary"] = summary
        summary, outcomes = summarize_attack(rows, scored, kind)
        payload.update(summary=summary, row_results=outcomes)
    return payload
