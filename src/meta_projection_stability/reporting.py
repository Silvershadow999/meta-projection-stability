from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not headers:
        return ""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        rr = [str(v) for v in r]
        if len(rr) < len(headers):
            rr += [""] * (len(headers) - len(rr))
        lines.append("| " + " | ".join(rr[:len(headers)]) + " |")
    return "\n".join(lines)


def _safe_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        return list(v)
    except Exception:
        return [v]


def _batch_summary_lines(batch_result: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not isinstance(batch_result, dict):
        return ["- Invalid batch_result"]

    rows = _safe_list(batch_result.get("rows"))
    errors = _safe_list(batch_result.get("errors"))
    ag = batch_result.get("aggregates", {}) or {}

    lines.append(f"- Batch valid: `{bool(batch_result.get('batch_valid', False))}`")
    lines.append(f"- Total rows (runs): `{len(rows)}`")
    lines.append(f"- Error rows: `{len(errors)}`")
    lines.append(f"- Aggregate keys: `{', '.join(sorted(ag.keys())) if isinstance(ag, dict) and ag else 'n/a'}`")

    # quick unique counts
    if rows and isinstance(rows[0], dict):
        def uniq(k: str) -> int:
            vals = set()
            for r in rows:
                if isinstance(r, dict) and k in r:
                    vals.add(r.get(k))
            return len(vals)
        lines.append(f"- Unique profiles: `{uniq('profile')}`")
        lines.append(f"- Unique systems: `{uniq('system')}`")
        lines.append(f"- Unique scenarios: `{uniq('scenario')}`")
        lines.append(f"- Unique seeds: `{uniq('seed')}`")

    return lines


def _ranking_top_rows_md(ranked: Optional[Dict[str, Any]], top_n: int = 10) -> str:
    if not ranked or not isinstance(ranked, dict):
        return "_No ranking result available._"
    rows = _safe_list(ranked.get("ranked_rows"))
    if not rows:
        return "_No ranked rows available._"

    headers = ["Rank", "Profile", "System", "Scenario", "Seed", "Score", "continue_rate", "reset_rate", "risk_p95", "h_sig_min", "trust_min"]
    body: List[List[Any]] = []
    for r in rows[:max(1, int(top_n))]:
        body.append([
            r.get("rank", "n/a"),
            r.get("profile", "n/a"),
            r.get("system", "n/a"),
            r.get("scenario", "n/a"),
            r.get("seed", "n/a"),
            _fmt(r.get("score_raw"), 4),
            _fmt(r.get("continue_rate"), 3),
            _fmt(r.get("reset_rate"), 3),
            _fmt(r.get("risk_p95"), 3),
            _fmt(r.get("h_sig_min"), 3),
            _fmt(r.get("trust_min"), 3),
        ])
    return _md_table(headers, body)


def _ranking_group_md(agg_rank: Optional[Dict[str, Any]], top_n: int = 10) -> str:
    if not agg_rank or not isinstance(agg_rank, dict):
        return "_No aggregate ranking available._"
    rows = _safe_list(agg_rank.get("aggregate_rows"))
    if not rows:
        return "_No aggregate ranking rows available._"

    group_keys = _safe_list(agg_rank.get("group_keys")) or ["profile", "system", "scenario"]

    headers = ["Rank", "Group", "Runs", "score_mean", "score_std", "continue_rate_mean", "reset_rate_mean", "risk_p95_mean", "h_sig_min_mean", "trust_min_mean"]
    body: List[List[Any]] = []
    for r in rows[:max(1, int(top_n))]:
        group_label = " | ".join(str(r.get(k, "n/a")) for k in group_keys)
        body.append([
            r.get("rank", "n/a"),
            group_label,
            r.get("runs", "n/a"),
            _fmt(r.get("score_mean"), 4),
            _fmt(r.get("score_std"), 4),
            _fmt(r.get("continue_rate_mean"), 3),
            _fmt(r.get("reset_rate_mean"), 3),
            _fmt(r.get("risk_p95_mean"), 3),
            _fmt(r.get("h_sig_min_mean"), 3),
            _fmt(r.get("trust_min_mean"), 3),
        ])
    return _md_table(headers, body)


def _pareto_front_md(pareto_res: Optional[Dict[str, Any]], top_n: int = 12) -> str:
    if not pareto_res or not isinstance(pareto_res, dict):
        return "_No Pareto result available._"

    front_rows = _safe_list(pareto_res.get("front_rows"))
    if not front_rows:
        return "_No Pareto front rows available._"

    headers = ["#", "Profile", "System", "Scenario", "Seed", "Score", "dominates#", "continue_rate", "reset_rate", "risk_p95", "h_sig_min", "trust_min"]
    body: List[List[Any]] = []
    for i, r in enumerate(front_rows[:max(1, int(top_n))], start=1):
        body.append([
            i,
            r.get("profile", "n/a"),
            r.get("system", "n/a"),
            r.get("scenario", "n/a"),
            r.get("seed", "n/a"),
            _fmt(r.get("score_raw"), 4),
            r.get("pareto_dominates_count", "n/a"),
            _fmt(r.get("continue_rate"), 3),
            _fmt(r.get("reset_rate"), 3),
            _fmt(r.get("risk_p95"), 3),
            _fmt(r.get("h_sig_min"), 3),
            _fmt(r.get("trust_min"), 3),
        ])
    summary = [
        f"- Total rows: `{pareto_res.get('n_total', 'n/a')}`",
        f"- Valid rows: `{pareto_res.get('n_valid', 'n/a')}`",
        f"- Pareto front size: `{pareto_res.get('n_front', 'n/a')}`",
        f"- Front fraction: `{_fmt((pareto_res.get('front_fraction', 0.0) or 0.0) * 100.0, 2)}%`",
    ]
    return "\n".join(summary) + "\n\n" + _md_table(headers, body)


def _pareto_groups_md(pareto_groups: Optional[Dict[str, Any]], top_n: int = 12) -> str:
    if not pareto_groups or not isinstance(pareto_groups, dict):
        return "_No Pareto group summary available._"

    rows = _safe_list(pareto_groups.get("rows"))
    if not rows:
        return "_No Pareto group rows available._"

    group_keys = _safe_list(pareto_groups.get("group_keys")) or ["profile", "system", "scenario"]
    headers = ["Rank", "Group", "Front count", "score_mean", "dom# mean", "continue_rate_mean", "reset_rate_mean", "risk_p95_mean", "h_sig_min_mean", "trust_min_mean"]
    body: List[List[Any]] = []

    for r in rows[:max(1, int(top_n))]:
        group_label = " | ".join(str(r.get(k, "n/a")) for k in group_keys)
        body.append([
            r.get("rank", "n/a"),
            group_label,
            r.get("front_count", "n/a"),
            _fmt(r.get("score_mean"), 4),
            _fmt(r.get("dominates_count_mean"), 3),
            _fmt(r.get("continue_rate_mean"), 3),
            _fmt(r.get("reset_rate_mean"), 3),
            _fmt(r.get("risk_p95_mean"), 3),
            _fmt(r.get("h_sig_min_mean"), 3),
            _fmt(r.get("trust_min_mean"), 3),
        ])

    return _md_table(headers, body)


def build_stability_report_markdown(
    *,
    batch_result: Dict[str, Any],
    ranked_result: Optional[Dict[str, Any]] = None,
    aggregate_ranking: Optional[Dict[str, Any]] = None,
    pareto_result: Optional[Dict[str, Any]] = None,
    pareto_group_summary_result: Optional[Dict[str, Any]] = None,
    title: str = "Meta Projection Stability Report",
    report_top_n: int = 12,
    cli_command: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    ts = datetime.now().isoformat(timespec="seconds")

    md: List[str] = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"- Generated: `{ts}`")
    if cli_command:
        md.append(f"- CLI Command: `{cli_command}`")
    if metadata:
        try:
            md.append(f"- Metadata: `{json.dumps(metadata, ensure_ascii=False)}`")
        except Exception:
            md.append(f"- Metadata: `{metadata}`")
    md.append("")

    md.append("## Batch Summary")
    md.append("")
    md.extend(_batch_summary_lines(batch_result))
    md.append("")

    md.append("## Ranking Top Runs")
    md.append("")
    md.append(_ranking_top_rows_md(ranked_result, top_n=report_top_n))
    md.append("")

    md.append("## Ranking Top Groups")
    md.append("")
    md.append(_ranking_group_md(aggregate_ranking, top_n=report_top_n))
    md.append("")

    md.append("## Pareto Front")
    md.append("")
    md.append(_pareto_front_md(pareto_result, top_n=report_top_n))
    md.append("")

    md.append("## Pareto Group Summary")
    md.append("")
    md.append(_pareto_groups_md(pareto_group_summary_result, top_n=report_top_n))
    md.append("")

    return "\n".join(md).strip() + "\n"


def save_stability_report_markdown(markdown_text: str, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown_text, encoding="utf-8")
    return str(p)


if __name__ == "__main__":
    dummy_batch = {
        "batch_valid": True,
        "rows": [
            {"profile":"balanced","system":"main_adapter","scenario":"spike","seed":41},
            {"profile":"protective","system":"main_adapter","scenario":"spike","seed":42},
        ],
        "errors": [],
        "aggregates": {"by_profile_system_scenario": []},
    }
    dummy_ranked = {
        "ranked_rows": [
            {"rank":1,"profile":"protective","system":"main_adapter","scenario":"spike","seed":42,"score_raw":1.23,"continue_rate":0.64,"reset_rate":0.00,"risk_p95":0.42,"h_sig_min":0.71,"trust_min":0.66}
        ]
    }
    dummy_agg = {
        "group_keys":["profile","system","scenario"],
        "aggregate_rows":[
            {"rank":1,"profile":"protective","system":"main_adapter","scenario":"spike","runs":2,"score_mean":1.11,"score_std":0.08,"continue_rate_mean":0.61,"reset_rate_mean":0.0,"risk_p95_mean":0.45,"h_sig_min_mean":0.70,"trust_min_mean":0.64}
        ]
    }
    dummy_pareto = {
        "n_total": 4, "n_valid": 4, "n_front": 2, "front_fraction": 0.5,
        "front_rows":[
            {"profile":"protective","system":"main_adapter","scenario":"spike","seed":42,"score_raw":1.23,"pareto_dominates_count":2,"continue_rate":0.64,"reset_rate":0.00,"risk_p95":0.42,"h_sig_min":0.71,"trust_min":0.66}
        ],
    }
    dummy_pg = {
        "group_keys":["profile","system","scenario"],
        "rows":[
            {"rank":1,"profile":"protective","system":"main_adapter","scenario":"spike","front_count":2,"score_mean":1.15,"dominates_count_mean":1.5,"continue_rate_mean":0.62,"reset_rate_mean":0.0,"risk_p95_mean":0.43,"h_sig_min_mean":0.70,"trust_min_mean":0.65}
        ]
    }

    md = build_stability_report_markdown(
        batch_result=dummy_batch,
        ranked_result=dummy_ranked,
        aggregate_ranking=dummy_agg,
        pareto_result=dummy_pareto,
        pareto_group_summary_result=dummy_pg,
        title="Stability Report Self-Test",
        report_top_n=5,
        cli_command="python -m ... --quick --rank --pareto",
    )
    out = save_stability_report_markdown(md, "artifacts/stability_report_selftest.md")
    print(f"Saved self-test report: {out}")
