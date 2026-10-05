"""Generate docs/index.html from experiment result JSONs.

Uses GitHub Models API (GPT-4o) to generate narrative explanations.
Run manually before pushing to GitHub Pages.

Usage:
    export GITHUB_TOKEN="your_token"
    python3 scripts/generate_site.py
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

ROOT      = Path(__file__).parent.parent
OUTPUTS   = ROOT / "outputs"
DOCS      = ROOT / "docs"
TOKEN     = os.environ.get("GITHUB_TOKEN", "")
API_URL   = "https://models.inference.ai.azure.com/chat/completions"
MODEL     = "gpt-4o"


# ── LLM call ─────────────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> str:
    if not TOKEN:
        return "<em>Set GITHUB_TOKEN to enable AI explanations.</em>"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 600,
    }).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        return f"<em>LLM error {e.code}: {e.reason}</em>"
    except Exception as e:
        return f"<em>LLM error: {e}</em>"


# ── Data loaders ──────────────────────────────────────────────────────────────

def load(name):
    p = OUTPUTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(v, pct=True):
    if v is None:
        return "—"
    return f"{v*100:.1f}" if pct else f"{v:.3f}"


def key_angles(results, angles=(0, 45, 90, 180)):
    """Return {angle: {r1, r5, map5}} for a list of angle dicts."""
    return {r["angle"]: r for r in results if r["angle"] in angles}


# ── HTML helpers ──────────────────────────────────────────────────────────────

def th(*cols):
    return "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"


def tr(*cols, bold=False):
    tag = "strong" if bold else None
    cells = []
    for c in cols:
        cells.append(f"<td>{'<strong>' if tag else ''}{c}{'</strong>' if tag else ''}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def table(header_row, body_rows, caption=""):
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (
        f"<div class='table-wrap'><table>{cap}"
        f"<thead>{header_row}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table></div>"
    )


def section(title, subtitle, content, ai_text):
    return f"""
<section>
  <h2>{title}</h2>
  <p class="subtitle">{subtitle}</p>
  {content}
  <div class="ai-box">
    <span class="ai-label">&#x2728; AI Analysis</span>
    <p>{ai_text}</p>
  </div>
</section>
"""


# ── Per-experiment builders ───────────────────────────────────────────────────

def build_exp1(data):
    if not data:
        return ""
    models = list(data.keys())
    datasets = list(data[models[0]].keys())

    rows = []
    for m in models:
        for ds in datasets:
            ka = key_angles(data[m][ds])
            rows.append(tr(
                m, ds,
                fmt(ka.get(0,   {}).get("r1")),
                fmt(ka.get(45,  {}).get("r1")),
                fmt(ka.get(90,  {}).get("r1")),
                fmt(ka.get(180, {}).get("r1")),
                fmt(ka.get(0,   {}).get("map5")),
                bold=(m == "RISA"),
            ))

    tbl = table(
        th("Model", "Dataset", "R@1 0°", "R@1 45°", "R@1 90°", "R@1 180°", "mAP@5 0°"),
        rows,
        "Retrieval R@1 (%) across rotation angles"
    )

    summary = "\n".join(
        f"Model {m}: " + ", ".join(
            f"{ds} R@1@0°={fmt(key_angles(data[m][ds]).get(0,{}).get('r1'))}%"
            for ds in datasets
        )
        for m in models
    )
    prompt = (
        "You are analyzing 3D point cloud retrieval results. "
        "Higher R@1 and mAP@5 is better. Rotation invariance means scores should stay stable across angles. "
        f"Results:\n{summary}\n"
        "In 3-4 sentences, explain which model performs best, how rotation affects performance, and what this tells us about rotation invariance."
    )
    return tbl, ask_llm(prompt)


def build_exp3(data):
    if not data:
        return ""
    conditions = list(data.keys())
    rows = []
    for c in conditions:
        ka = key_angles(data[c])
        rows.append(tr(
            c,
            fmt(ka.get(0,   {}).get("r1")),
            fmt(ka.get(45,  {}).get("r1")),
            fmt(ka.get(90,  {}).get("r1")),
            fmt(ka.get(180, {}).get("r1")),
            fmt(ka.get(0,   {}).get("map5")),
            bold=(c == "A_RISA"),
        ))
    tbl = table(
        th("Condition", "R@1 0°", "R@1 45°", "R@1 90°", "R@1 180°", "mAP@5 0°"),
        rows,
        "ModelNet40 ablation — R@1 (%) and mAP@5 (%)"
    )
    summary = "\n".join(
        f"{c}: R@1@0°={fmt(key_angles(data[c]).get(0,{}).get('r1'))}%, R@1@180°={fmt(key_angles(data[c]).get(180,{}).get('r1'))}%"
        for c in conditions
    )
    prompt = (
        "You are analyzing a 3D point cloud retrieval ablation study on ModelNet40. "
        "Conditions: A_RISA=full model, B_DGCNN_SO3=DGCNN with SO3 aug, C_DGCNN_TTA=DGCNN with test-time aug, "
        "D_RISA_XYZ=RISA with XYZ features, E_RISA_NO_TOKEN=RISA without encoding token. "
        f"Results:\n{summary}\n"
        "In 3-4 sentences, explain what the ablation reveals about which components matter most for rotation-invariant retrieval."
    )
    return tbl, ask_llm(prompt)


def build_exp4(old_data, perc_data):
    if not old_data and not perc_data:
        return ""
    rows = []
    for label, data in [("RISA (perceiver=False)", old_data), ("RISA (perceiver=True)", perc_data)]:
        if not data:
            continue
        results = list(data.values())[0]
        ka = key_angles(results)
        rows.append(tr(
            label,
            fmt(ka.get(0,   {}).get("r1")),
            fmt(ka.get(90,  {}).get("r1")),
            fmt(ka.get(180, {}).get("r1")),
            fmt(ka.get(0,   {}).get("map5")),
            bold=(label == "RISA (perceiver=False)"),
        ))
    tbl = table(
        th("Variant", "R@1 0°", "R@1 90°", "R@1 180°", "mAP@5 0°"),
        rows,
        "Cross-dataset transfer — ModelNet40 → ScanObjectNN"
    )
    prompt = (
        "You are comparing two RISA variants on cross-dataset 3D retrieval transfer (ModelNet40 trained, ScanObjectNN tested). "
        f"perceiver=False: R@1@0°={fmt(key_angles(list(old_data.values())[0]).get(0,{}).get('r1'))}%, "
        f"perceiver=True: R@1@0°={fmt(key_angles(list(perc_data.values())[0]).get(0,{}).get('r1'))}%. "
        "In 2-3 sentences, explain what the perceiver flag controls architecturally and why one variant transfers better."
    )
    return tbl, ask_llm(prompt)


def build_exp8(data):
    if not data:
        return ""
    # exp8 stores flat metric dicts per method, not angle lists
    rows = []
    summary_lines = []
    skip = {"config"}
    for m, v in data.items():
        if m in skip or not isinstance(v, dict):
            continue
        method = v.get("method", m)
        class_map = v.get("class_mAP@5") or v.get("class_map5")
        part_map  = v.get("part_mAP@5")  or v.get("part_map5")
        r1        = v.get("r1") or v.get("R@1")
        rows.append(tr(
            method,
            fmt(r1)   if r1        is not None else "—",
            fmt(class_map) if class_map is not None else "—",
            fmt(part_map)  if part_map  is not None else "—",
            bold=("RISA" in method),
        ))
        summary_lines.append(
            f"{method}: class_mAP@5={fmt(class_map)}%, part_mAP@5={fmt(part_map)}%"
        )
    if not rows:
        return ""
    tbl = table(
        th("Method", "R@1", "class mAP@5", "part mAP@5"),
        rows,
        "ShapeNet part retrieval"
    )
    prompt = (
        "You are analyzing part-level 3D shape retrieval results on ShapeNet. "
        f"Results:\n{'  '.join(summary_lines)}\n"
        "In 3 sentences, explain the performance differences and what they reveal about part-aware retrieval."
    )
    return tbl, ask_llm(prompt)


def build_exp10(data):
    if not data:
        return "<p><em>Exp10 still running — rerun generate_site.py when complete.</em></p>", ""
    priors = list(data.keys())
    rows = []
    for p in priors:
        ka = key_angles(data[p])
        rows.append(tr(
            p,
            fmt(ka.get(0,   {}).get("r1")),
            fmt(ka.get(45,  {}).get("r1")),
            fmt(ka.get(90,  {}).get("r1")),
            fmt(ka.get(180, {}).get("r1")),
            fmt(ka.get(0,   {}).get("map5")),
        ))
    tbl = table(
        th("Prior", "R@1 0°", "R@1 45°", "R@1 90°", "R@1 180°", "mAP@5 0°"),
        rows,
        "Prior selection ablation — ModelNet40 K=256 — R@1 (%) and mAP@5 (%)"
    )
    summary = "\n".join(
        f"{p}: R@1@0°={fmt(key_angles(data[p]).get(0,{}).get('r1'))}%, mAP@5@0°={fmt(key_angles(data[p]).get(0,{}).get('map5'))}%"
        for p in priors
    )
    prompt = (
        "You are analyzing a point selection prior ablation for 3D retrieval. "
        "Priors: full_N=all 1024 points, random_K=random 256, fps_K=farthest point sampling, "
        "eigenentropy_K=top-256 by geometric complexity, surface_var_K=top-256 by surface variation, "
        "curvature_K=top-256 by anisotropy, salient_K=top-256 by learned attention weights. "
        f"Results:\n{summary}\n"
        "In 4 sentences, explain which prior works best, whether selecting K=256 points beats using all N=1024, "
        "and what this tells us about which points carry the most discriminative information for retrieval."
    )
    return tbl, ask_llm(prompt)


# ── HTML template ─────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8f9fa; color: #1a1a2e; }
header { background: #1a1a2e; color: white; padding: 3rem 2rem; text-align: center; }
header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
header p  { color: #aab4c8; font-size: 1rem; }
main { max-width: 960px; margin: 0 auto; padding: 2rem 1rem; }
section { background: white; border-radius: 12px; padding: 2rem; margin-bottom: 2rem;
          box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
h2 { font-size: 1.3rem; color: #1a1a2e; margin-bottom: 0.3rem; }
.subtitle { color: #666; font-size: 0.9rem; margin-bottom: 1.2rem; }
.table-wrap { overflow-x: auto; margin-bottom: 1.2rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
caption { text-align: left; font-size: 0.8rem; color: #888; margin-bottom: 0.4rem; }
th { background: #1a1a2e; color: white; padding: 0.6rem 0.8rem; text-align: left; }
td { padding: 0.55rem 0.8rem; border-bottom: 1px solid #eee; }
tr:hover td { background: #f0f4ff; }
.ai-box { background: #f0f4ff; border-left: 4px solid #4a6cf7; border-radius: 6px;
           padding: 1rem 1.2rem; margin-top: 0.5rem; }
.ai-label { font-size: 0.75rem; font-weight: 700; color: #4a6cf7;
             text-transform: uppercase; letter-spacing: 0.05em; display: block;
             margin-bottom: 0.4rem; }
.ai-box p { font-size: 0.92rem; line-height: 1.6; color: #333; }
footer { text-align: center; padding: 2rem; color: #999; font-size: 0.8rem; }
"""


def build_html(sections_html):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RISA — Rotation-Invariant Sparse Attention Results</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Rotation-Invariant Sparse Attention (RISA)</h1>
  <p>3D Point Cloud Retrieval — Experiment Results &nbsp;·&nbsp; Generated {ts}</p>
</header>
<main>
{''.join(sections_html)}
</main>
<footer>Generated by generate_site.py · sujay152002</footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    exp1  = load("exp1_retrieval.json")
    exp3  = load("exp3_ablation.json")
    exp4o = load("exp4_risa_old_100ep.json")
    exp4p = load("exp4_risa_perceiver_100ep.json")
    exp8  = load("exp8_part_retrieval.json")
    exp10 = load("exp10_prior_selection.json")

    sections = []

    print("Building exp1...")
    r = build_exp1(exp1)
    if r:
        tbl, ai = r
        sections.append(section(
            "Exp 1 — Cross-Dataset Retrieval",
            "Models trained on ShapeNet, evaluated on ShapeNet and ModelNet40 under SO(3) rotations.",
            tbl, ai
        ))

    print("Building exp3...")
    r = build_exp3(exp3)
    if r:
        tbl, ai = r
        sections.append(section(
            "Exp 3 — Architecture Ablation (ModelNet40)",
            "Ablation of key RISA components vs. DGCNN baselines on class-level retrieval.",
            tbl, ai
        ))

    print("Building exp4...")
    r = build_exp4(exp4o, exp4p)
    if r:
        tbl, ai = r
        sections.append(section(
            "Exp 4 — Perceiver vs Standard RISA (Cross-Dataset Transfer)",
            "Comparing perceiver=True vs perceiver=False on ModelNet40→ScanObjectNN transfer.",
            tbl, ai
        ))

    print("Building exp8...")
    r = build_exp8(exp8)
    if r:
        tbl, ai = r
        sections.append(section(
            "Exp 8 — Part-Level Retrieval (ShapeNet)",
            "Part-aware retrieval on ShapeNet 16-class dataset under rotation.",
            tbl, ai
        ))

    print("Building exp10...")
    tbl, ai = build_exp10(exp10)
    sections.append(section(
        "Exp 10 — Geometric Prior Point Selection Ablation",
        "Which rotation-invariant prior best selects K=256 informative points from N=1024?",
        tbl, ai
    ))

    print("Rendering HTML...")
    html = build_html(sections)
    out = DOCS / "index.html"
    out.write_text(html)
    print(f"Done → {out}")
    print("\nNext steps:")
    print("  git add docs/index.html")
    print("  git commit -m 'update results site'")
    print("  git push")


if __name__ == "__main__":
    main()
