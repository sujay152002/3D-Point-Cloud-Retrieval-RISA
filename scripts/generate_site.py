"""Generate docs/index.html from experiment result JSONs.

Uses Groq API (qwen3.8-27b) to generate narrative explanations.
Run manually before pushing to GitHub Pages.

Usage:
    bash scripts/update_site.sh
    -- or --
    GROQ_API_KEY="your_key" python3 scripts/generate_site.py
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
DOCS    = ROOT / "docs"
TOKEN   = os.environ.get("GROQ_API_KEY", "")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL   = "qwen/qwen3.8-27b"


# ── LLM ──────────────────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> str:
    if not TOKEN:
        return "<em>Set GROQ_API_KEY to enable explanations.</em>"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 600,
    }).encode()
    req = urllib.request.Request(
        API_URL, data=payload,
        headers={
            "Authorization": f"Bearer {TOKEN.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "python-urllib/3",
        },
    )
    import time
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 2:
                time.sleep(5 * (attempt + 1)); continue
            return f"<em>LLM error {e.code}: {e.reason}</em>"
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1)); continue
            return f"<em>LLM error: {e}</em>"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(name):
    p = OUTPUTS / name
    return json.loads(p.read_text()) if p.exists() else None

def fmt(v):
    return f"{v*100:.1f}" if v is not None else "—"

def key_angles(results, angles=(0, 45, 90, 180)):
    return {r["angle"]: r for r in results if r["angle"] in angles}

def th(*cols):
    return "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"

def td_row(*cols, bold=False):
    wrap = ("<strong>", "</strong>") if bold else ("", "")
    return "<tr>" + "".join(f"<td>{wrap[0]}{c}{wrap[1]}</td>" for c in cols) + "</tr>"

def make_table(header, rows, caption=""):
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f"<div class='tbl-wrap'><table>{cap}"
            f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table></div>")

def info_grid(**items):
    cards = "".join(
        f"<div class='info-card'><span class='info-label'>{k}</span>"
        f"<span class='info-value'>{v}</span></div>"
        for k, v in items.items()
    )
    return f"<div class='info-grid'>{cards}</div>"

def tag_list(tags):
    return "".join(f"<span class='tag'>{t}</span>" for t in tags)

def line_chart(series, width=500, height=80):
    """SVG line chart. series = [(label, color, [(angle, r1), ...])]"""
    if not series:
        return ""
    all_x = sorted({x for _, _, pts in series for x, _ in pts})
    all_y = [y for _, _, pts in series for _, y in pts]
    if not all_x or not all_y:
        return ""
    pad = 30
    W, H = width, height
    mn, mx = min(all_y), max(all_y)
    rng = mx - mn if mx != mn else 0.01
    def sx(x): return pad + (x / max(all_x)) * (W - 2 * pad)
    def sy(y): return H - pad - ((y - mn) / rng) * (H - 2 * pad)

    paths = ""
    for label, color, pts in series:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in sorted(pts))
        paths += (f'<polyline points="{coords}" fill="none" stroke="{color}" '
                  f'stroke-width="2" stroke-linejoin="round"/>')

    # x-axis labels at 0, 45, 90, 135, 180
    xlabels = "".join(
        f'<text x="{sx(a):.1f}" y="{H}" font-size="9" fill="#999" text-anchor="middle">{a}</text>'
        for a in [0, 45, 90, 135, 180] if a <= max(all_x)
    )
    # y-axis labels
    ylabels = (
        f'<text x="0" y="{sy(mn)+4:.1f}" font-size="9" fill="#999">{mn*100:.0f}</text>'
        f'<text x="0" y="{sy(mx)+4:.1f}" font-size="9" fill="#999">{mx*100:.0f}</text>'
    )
    # legend
    legend = ""
    for i, (label, color, _) in enumerate(series):
        lx = pad + i * 120
        legend += (f'<rect x="{lx}" y="-18" width="12" height="3" fill="{color}"/>'
                   f'<text x="{lx+15}" y="-13" font-size="9" fill="#555">{label}</text>')

    return (f'<div class="chart-wrap">'
            f'<svg width="{W}" height="{H+20}" style="display:block;margin:.5rem 0 1rem">'
            f'<g transform="translate(0,20)">'
            f'{legend}{paths}{xlabels}{ylabels}'
            f'</g></svg></div>')


# ── Experiment builders ───────────────────────────────────────────────────────

COLORS = ["#4a6cf7", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

def build_exp1(data):
    if not data:
        return None
    models   = list(data.keys())
    datasets = list(data[models[0]].keys())
    ds0      = datasets[0]

    # All 19 angles — one row per angle, one column per model
    all_angles = sorted({r["angle"] for r in data[models[0]][ds0]})
    # index: data[model][ds] -> {angle: record}
    idx = {m: {r["angle"]: r for r in data[m][ds0]} for m in models}
    rows = []
    for a in all_angles:
        cols = [f"{int(a)}°"] + [fmt(idx[m].get(a, {}).get("r1")) for m in models]
        rows.append(td_row(*cols))
    tbl = make_table(
        th("Angle", *models),
        rows, "R@1 (%) at every rotation angle — ShapeNet"
    )

    # Line chart
    series = [
        (m, COLORS[i % len(COLORS)], [(r["angle"], r["r1"]) for r in data[m][ds0]])
        for i, m in enumerate(models)
    ]
    chart = line_chart(series, width=560, height=90)

    summary = "\n".join(
        f"{m}: R@1@0={fmt(idx[m].get(0,{}).get('r1'))}%, R@1@90={fmt(idx[m].get(90,{}).get('r1'))}%, R@1@180={fmt(idx[m].get(180,{}).get('r1'))}%"
        for m in models
    )
    ai = ask_llm(
        "You are analyzing 3D point cloud retrieval results. Higher R@1 is better. "
        "Rotation invariance means scores stay stable across angles 0-180 degrees. "
        f"Results:\n{summary}\n"
        "In 3-4 sentences, explain which model performs best, how rotation affects each model, "
        "and what this reveals about rotation invariance."
    )
    return dict(
        id="exp1", title="Exp 1 — Cross-Dataset Retrieval",
        question="Do rotation-invariant features generalise across datasets without retraining?",
        info=info_grid(
            Models=tag_list(models),
            Datasets=tag_list(datasets),
            Metric="R@1 · R@5 · mAP@5",
            Training="Cross-entropy on ShapeNet",
            Evaluation="SO(3) rotations 0°→180° (19 angles)",
        ),
        table=chart + tbl, ai=ai,
    )


def tsne_svg(encodings, labels, title="", width=520, height=320):
    """Run t-SNE on encodings and return an inline SVG scatter plot."""
    import numpy as np
    from sklearn.manifold import TSNE
    X = np.array(encodings, dtype=np.float32)
    # PCA to 50 dims first for speed
    from sklearn.decomposition import PCA
    n_comp = min(50, X.shape[1], X.shape[0] - 1)
    X50 = PCA(n_components=n_comp, random_state=42).fit_transform(X)
    xy = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500).fit_transform(X50)
    # normalise to [pad, W-pad] x [pad, H-pad]
    pad = 20
    mn, mx = xy.min(0), xy.max(0)
    rng = mx - mn
    rng[rng == 0] = 1
    xs = pad + (xy[:, 0] - mn[0]) / rng[0] * (width  - 2 * pad)
    ys = pad + (xy[:, 1] - mn[1]) / rng[1] * (height - 2 * pad)
    palette = ["#4a6cf7","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22","#34495e"]
    unique_labels = sorted(set(labels))
    color_map = {l: palette[i % len(palette)] for i, l in enumerate(unique_labels)}
    dots = "".join(
        f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="2.5" fill="{color_map[labels[i]]}" opacity="0.7"/>'
        for i in range(len(labels))
    )
    legend = "".join(
        f'<rect x="{(i%4)*120+pad}" y="{height+4+(i//4)*16}" width="10" height="10" fill="{color_map[l]}"/>'
        f'<text x="{(i%4)*120+pad+14}" y="{height+13+(i//4)*16}" font-size="10" fill="#555">class {l}</text>'
        for i, l in enumerate(unique_labels)
    )
    legend_h = ((len(unique_labels) - 1) // 4 + 1) * 16 + 8
    cap = f'<text x="{width//2}" y="{height+legend_h+18}" font-size="11" fill="#888" text-anchor="middle">{title}</text>' if title else ""
    total_h = height + legend_h + (24 if title else 4)
    return (f'<svg width="{width}" height="{total_h}" style="display:block;margin:.5rem auto 1rem">'
            f'{dots}{legend}{cap}</svg>')


def build_exp2(data):
    if not data:
        return None
    models = list(data.keys())
    print("  Running t-SNE for exp2 (this may take ~30s)...")
    svgs = ""
    for m in models:
        encs = data[m]["encodings"]
        labs = data[m]["labels"]
        svgs += f'<div style="display:inline-block;margin:.5rem"><strong style="font-size:.85rem">{m}</strong>'
        svgs += tsne_svg(encs, labs, width=300, height=220)
        svgs += "</div>"
    plot_wrap = f'<div style="overflow-x:auto;white-space:nowrap">{svgs}</div>'
    ai = ask_llm(
        "You are analyzing t-SNE visualizations of 3D shape embeddings on ShapeNet (8 classes). "
        f"Models compared: {', '.join(models)}. "
        "RISA is the proposed rotation-invariant model. "
        "In 3 sentences, explain what well-separated clusters in t-SNE indicate about embedding quality, "
        "and what semantic collapse means for retrieval performance."
    )
    return dict(
        id="exp2", title="Exp 2 — Semantic Collapse",
        question="Do non-invariant models collapse semantically distinct shapes into the same embedding region?",
        info=info_grid(
            Models=tag_list(models),
            Dataset=tag_list(["ShapeNet (8 classes)"]),
            Metric="t-SNE of 512-d embeddings",
            Samples="1280 (160 per class)",
        ),
        table=plot_wrap, ai=ai,
    )


def build_exp3(data):
    if not data:
        return None
    conditions = list(data.keys())
    cond_desc = {
        "A_RISA":         "Full RISA model",
        "B_DGCNN_SO3":    "DGCNN + SO(3) augmentation",
        "C_DGCNN_TTA":    "DGCNN + test-time augmentation",
        "D_RISA_XYZ":     "RISA with raw XYZ features",
        "E_RISA_NO_TOKEN":"RISA without encoding token",
    }
    rows = []
    for c in conditions:
        ka = key_angles(data[c])
        rows.append(td_row(
            c, cond_desc.get(c, c),
            fmt(ka.get(0,   {}).get("r1")),
            fmt(ka.get(45,  {}).get("r1")),
            fmt(ka.get(90,  {}).get("r1")),
            fmt(ka.get(180, {}).get("r1")),
            fmt(ka.get(0,   {}).get("map5")),
            bold=(c == "A_RISA"),
        ))
    tbl = make_table(
        th("Condition", "Description", "R@1 0°", "R@1 45°", "R@1 90°", "R@1 180°", "mAP@5 0°"),
        rows, "ModelNet40 class retrieval — R@1 (%) and mAP@5 (%)"
    )
    summary = "\n".join(
        f"{c}: R@1@0={fmt(key_angles(data[c]).get(0,{}).get('r1'))}%, "
        f"R@1@180={fmt(key_angles(data[c]).get(180,{}).get('r1'))}%"
        for c in conditions
    )
    ai = ask_llm(
        "You are analyzing a 3D point cloud retrieval ablation on ModelNet40. "
        "Conditions: A_RISA=full model, B_DGCNN_SO3=DGCNN+SO3 aug, C_DGCNN_TTA=DGCNN+TTA, "
        "D_RISA_XYZ=RISA with XYZ features only, E_RISA_NO_TOKEN=RISA without global encoding token. "
        f"Results:\n{summary}\n"
        "In 3-4 sentences, explain which components matter most and what the ablation reveals."
    )
    return dict(
        id="exp3", title="Exp 3 — Architecture Ablation",
        question="Which architectural components of RISA are essential for rotation-invariant retrieval?",
        info=info_grid(
            Models=tag_list(["RISA", "DGCNN"]),
            Dataset=tag_list(["ModelNet40"]),
            Metric="R@1 · R@5 · mAP@5",
            Training="Cross-entropy, 100 epochs",
            Conditions="5 ablation conditions",
        ),
        table=tbl, ai=ai,
    )


def build_exp4(old_data, perc_data):
    if not old_data and not perc_data:
        return None
    rows = []
    series = []
    for i, (label, data) in enumerate([
        ("RISA (perceiver=False)", old_data),
        ("RISA (perceiver=True)",  perc_data),
    ]):
        if not data:
            continue
        results = list(data.values())[0]
        ka = key_angles(results)
        rows.append(td_row(
            label,
            fmt(ka.get(0,   {}).get("r1")),
            fmt(ka.get(90,  {}).get("r1")),
            fmt(ka.get(180, {}).get("r1")),
            fmt(ka.get(0,   {}).get("map5")),
            bold=(i == 0),
        ))
        series.append((label, COLORS[i], [(r["angle"], r["r1"]) for r in results]))

    tbl = make_table(
        th("Variant", "R@1 0°", "R@1 90°", "R@1 180°", "mAP@5 0°"),
        rows, "Cross-dataset transfer — ModelNet40 trained, ScanObjectNN tested"
    )
    chart = line_chart(series, width=560, height=90)
    r1_old  = fmt(key_angles(list(old_data.values())[0]).get(0, {}).get("r1"))
    r1_perc = fmt(key_angles(list(perc_data.values())[0]).get(0, {}).get("r1"))
    ai = ask_llm(
        "You are comparing two RISA variants on cross-dataset 3D retrieval transfer "
        "(trained on ModelNet40, tested on ScanObjectNN). "
        f"perceiver=False: R@1@0={r1_old}%, perceiver=True: R@1@0={r1_perc}%. "
        "In 2-3 sentences, explain what the perceiver flag controls architecturally "
        "and why one variant transfers better across datasets."
    )
    return dict(
        id="exp4", title="Exp 4 — Perceiver vs Standard RISA",
        question="Does the perceiver-style attention update strategy improve cross-dataset transfer?",
        info=info_grid(
            Models=tag_list(["RISA (perceiver=False)", "RISA (perceiver=True)"]),
            **{"Train Dataset": tag_list(["ModelNet40"])},
            **{"Test Dataset": tag_list(["ScanObjectNN"])},
            Metric="R@1 · mAP@5",
            Training="Proxy Anchor, 100 epochs",
        ),
        table=chart + tbl, ai=ai,
    )


def build_exp8(data):
    if not data:
        return None
    # All results from exp8_run_20260916_140537.log
    all_models = [
        ("3D-SIFT-InvIndex", 0.3102, 0.3089),
        ("PointNet++",       0.8633, 0.8625),
        ("DGCNN",            0.8966, 0.8947),
        ("DiPVNet",          0.8652, 0.8639),
        ("RINet",            0.8604, 0.8568),
        ("RISA",             0.9372, 0.9347),
    ]
    rows = []
    summary_lines = []
    for model, class_map, part_map in all_models:
        rows.append(td_row(model, fmt(class_map), fmt(part_map), bold=(model == "RISA")))
        summary_lines.append(
            f"{model}: class_mAP@5={fmt(class_map)}%, part_mAP@5={fmt(part_map)}%"
        )
    tbl = make_table(
        th("Method", "class mAP@5 (%)", "part mAP@5 (%)"),
        rows, "ShapeNet part retrieval — all 6 methods"
    )
    ai = ask_llm(
        "You are analyzing part-level 3D shape retrieval on ShapeNet (16 classes). "
        f"Results:\n{chr(10).join(summary_lines)}\n"
        "In 3 sentences, explain the performance differences and what they reveal "
        "about part-aware retrieval."
    )
    return dict(
        id="exp8", title="Exp 8 — Part-Level Retrieval",
        question="Can RISA retrieve shapes by part similarity, not just global class?",
        info=info_grid(
            Models=tag_list(["RISA", "DGCNN", "PointNet++", "DiPVNet", "RINet", "3D-SIFT"]),
            Dataset=tag_list(["ShapeNet (16 classes)"]),
            Metric="class mAP@5 · part mAP@5",
            Training="Proxy Anchor, 100 epochs",
            **{"Part Labels": "Back · Seat · Leg · Arm (chair example)"},
        ),
        table=tbl, ai=ai,
    )


def build_exp10(data):
    if not data:
        tbl = "<p class='pending'>Exp 10 still running — rerun update_site.sh when complete.</p>"
        return dict(
            id="exp10", title="Exp 10 — Prior Point Selection Ablation",
            question="Which geometric prior best selects K=256 informative points from N=1024?",
            info=info_grid(
                Model=tag_list(["RISA"]),
                Dataset=tag_list(["ModelNet40"]),
                Metric="R@1 · mAP@5",
                Training="Proxy Anchor, 100 epochs",
                **{"K / N": "256 / 1024"},
            ),
            table=tbl, ai="<em>Results pending.</em>",
        )
    priors = list(data.keys())
    prior_desc = {
        "full_N":         "All 1024 points (baseline)",
        "random_K":       "Random 256 points (lower bound)",
        "fps_K":          "Farthest point sampling",
        "eigenentropy_K": "Top-K by eigenentropy (geometric complexity)",
        "surface_var_K":  "Top-K by surface variation",
        "curvature_K":    "Top-K by anisotropy",
        "salient_K":      "Top-K by enc-token attention (learned)",
    }
    rows = []
    for p in priors:
        ka = key_angles(data[p])
        rows.append(td_row(
            p, prior_desc.get(p, p),
            fmt(ka.get(0,   {}).get("r1")),
            fmt(ka.get(45,  {}).get("r1")),
            fmt(ka.get(90,  {}).get("r1")),
            fmt(ka.get(180, {}).get("r1")),
            fmt(ka.get(0,   {}).get("map5")),
        ))
    tbl = make_table(
        th("Prior", "Description", "R@1 0°", "R@1 45°", "R@1 90°", "R@1 180°", "mAP@5 0°"),
        rows, "ModelNet40 — K=256 from N=1024 — R@1 (%) and mAP@5 (%)"
    )
    summary = "\n".join(
        f"{p}: R@1@0={fmt(key_angles(data[p]).get(0,{}).get('r1'))}%, "
        f"mAP@5={fmt(key_angles(data[p]).get(0,{}).get('map5'))}%"
        for p in priors
    )
    ai = ask_llm(
        "You are analyzing a point selection prior ablation for 3D retrieval. "
        "full_N=all 1024 pts, random_K=random 256, fps_K=farthest point sampling, "
        "eigenentropy_K=top-256 by geometric complexity, surface_var_K=top-256 by surface variation, "
        "curvature_K=top-256 by anisotropy, salient_K=top-256 by learned attention weights. "
        f"Results:\n{summary}\n"
        "In 4 sentences: which prior works best, does K=256 beat full N=1024, "
        "and what does this tell us about which points carry the most discriminative information?"
    )
    return dict(
        id="exp10", title="Exp 10 — Prior Point Selection Ablation",
        question="Which geometric prior best selects K=256 informative points from N=1024?",
        info=info_grid(
            Model=tag_list(["RISA"]),
            Dataset=tag_list(["ModelNet40"]),
            Metric="R@1 · mAP@5",
            Training="Proxy Anchor, 100 epochs",
            **{"K / N": "256 / 1024"},
            Priors=str(len(priors)),
        ),
        table=tbl, ai=ai,
    )


# ── HTML ──────────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f8;color:#1a1a2e}
header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
       color:white;padding:3.5rem 2rem;text-align:center}
header h1{font-size:2.2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:.5rem}
header .subtitle{color:#8892b0;font-size:1rem;margin-bottom:1.5rem}
header .badges{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap}
.badge{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
       color:#ccd6f6;padding:.3rem .8rem;border-radius:20px;font-size:.8rem}
.tab-bar{display:flex;gap:.3rem;padding:1.2rem 1.5rem 0;
         max-width:1100px;margin:0 auto;flex-wrap:wrap}
.tab-btn{background:white;border:none;padding:.6rem 1.2rem;border-radius:8px 8px 0 0;
         font-size:.85rem;font-weight:600;color:#666;cursor:pointer;
         border-bottom:3px solid transparent;transition:all .2s}
.tab-btn:hover{color:#4a6cf7;background:#f0f4ff}
.tab-btn.active{color:#4a6cf7;border-bottom:3px solid #4a6cf7;background:white}
.tab-panels{max-width:1100px;margin:0 auto;padding:0 1.5rem 2rem}
.tab-panel{display:none;background:white;border-radius:0 12px 12px 12px;
           box-shadow:0 4px 20px rgba(0,0,0,.08);overflow:hidden}
.tab-panel.active{display:block}
.panel-header{padding:2rem 2rem 1.5rem;border-bottom:1px solid #f0f0f0}
.panel-header h2{font-size:1.4rem;color:#1a1a2e;margin-bottom:.4rem}
.question{color:#4a6cf7;font-size:.95rem;font-style:italic;margin-bottom:1.2rem}
.info-grid{display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:.5rem}
.info-card{background:#f8f9ff;border:1px solid #e8ecff;border-radius:8px;
           padding:.6rem 1rem;min-width:140px}
.info-label{display:block;font-size:.7rem;font-weight:700;color:#4a6cf7;
            text-transform:uppercase;letter-spacing:.05em;margin-bottom:.2rem}
.info-value{font-size:.85rem;color:#333;display:flex;flex-wrap:wrap;gap:.3rem}
.tag{background:#e8ecff;color:#4a6cf7;padding:.15rem .5rem;
     border-radius:4px;font-size:.78rem;font-weight:600}
.panel-body{padding:1.5rem 2rem}
.chart-wrap{margin-bottom:.5rem}
.tbl-wrap{overflow-x:auto;margin-bottom:1.5rem}
table{width:100%;border-collapse:collapse;font-size:.87rem}
caption{text-align:left;font-size:.78rem;color:#999;margin-bottom:.4rem;font-style:italic}
thead tr{background:#1a1a2e}
th{color:white;padding:.65rem .9rem;text-align:left;font-weight:600;font-size:.82rem}
td{padding:.55rem .9rem;border-bottom:1px solid #f0f0f0;color:#333}
tbody tr:hover td{background:#f5f7ff}
tbody tr:last-child td{border-bottom:none}
.ai-box{background:linear-gradient(135deg,#f0f4ff,#f8f0ff);
        border-left:4px solid #4a6cf7;border-radius:8px;padding:1.2rem 1.4rem}
.ai-label{font-size:.72rem;font-weight:800;color:#4a6cf7;text-transform:uppercase;
          letter-spacing:.08em;display:flex;align-items:center;gap:.4rem;margin-bottom:.6rem}
.ai-box p{font-size:.93rem;line-height:1.7;color:#333}
.pending{color:#888;font-style:italic;padding:1rem 0}
footer{text-align:center;padding:2rem;color:#aaa;font-size:.8rem}
"""

JS = """
function showTab(id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab-btn[data-tab="'+id+'"]').classList.add('active');
  document.getElementById(id).classList.add('active');
}
"""

def render_panel(exp):
    return (
        f'<div class="tab-panel" id="{exp["id"]}">'
        f'<div class="panel-header">'
        f'<h2>{exp["title"]}</h2>'
        f'<p class="question">{exp["question"]}</p>'
        f'{exp["info"]}'
        f'</div>'
        f'<div class="panel-body">'
        f'{exp["table"]}'
        f'<div class="ai-box">'
        f'<div class="ai-label">Analysis</div>'
        f'<p>{exp["ai"]}</p>'
        f'</div></div></div>'
    )

def build_html(experiments):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    tab_btns = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" '
        f'data-tab="{e["id"]}" onclick="showTab(\'{e["id"]}\')">'
        f'{e["title"].split("—")[0].strip()}</button>'
        for i, e in enumerate(experiments)
    )
    panels = "".join(render_panel(e) for e in experiments)
    return (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="UTF-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f'<title>RISA — 3D Point Cloud Retrieval Results</title>'
        f'<style>{CSS}</style></head><body>'
        f'<header>'
        f'<h1>Rotation-Invariant Sparse Attention</h1>'
        f'<p class="subtitle">3D Point Cloud Retrieval — Experiment Results</p>'
        f'<div class="badges">'
        f'<span class="badge">📅 {ts}</span>'
        f'<span class="badge">🔬 6 Experiments</span>'
        f'<span class="badge">📦 ModelNet40 · ShapeNet · ScanObjectNN</span>'
        f'<span class="badge">⚡ SO(3) Invariant</span>'
        f'</div></header>'
        f'<div class="tab-bar">{tab_btns}</div>'
        f'<div class="tab-panels">{panels}</div>'
        f'<footer>sujay152002 · Generated by generate_site.py</footer>'
        f'<script>{JS}</script>'
        f'</body></html>'
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    experiments = []
    steps = [
        ("exp1",  lambda: build_exp1(load("exp1_retrieval.json"))),
        ("exp2",  lambda: build_exp2(load("exp2_semantic_collapse_shapenet.json"))),
        ("exp3",  lambda: build_exp3(load("exp3_ablation.json"))),
        ("exp4",  lambda: build_exp4(load("exp4_risa_old_100ep.json"), load("exp4_risa_perceiver_100ep.json"))),
        ("exp8",  lambda: build_exp8(load("exp8_part_retrieval.json"))),
        ("exp10", lambda: build_exp10(load("exp10_prior_selection.json"))),
    ]
    for name, fn in steps:
        print(f"Building {name}...")
        result = fn()
        if result:
            experiments.append(result)
    print("Rendering HTML...")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(build_html(experiments))
    print(f"Done -> {DOCS / 'index.html'}")

if __name__ == "__main__":
    main()
