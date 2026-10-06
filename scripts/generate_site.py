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

def ask_llm(prompt: str, max_tokens: int = 900) -> str:
    if not TOKEN:
        return "<em>Set GROQ_API_KEY to enable explanations.</em>"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens,
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

_chart_id_counter = [0]
def _cid():
    _chart_id_counter[0] += 1
    return f"chart{_chart_id_counter[0]}"

def _filter_bar(series, cid):
    """Render a dropdown multi-select filter for a chart."""
    opts = "".join(
        f'<option value="{label}" selected style="background:#1c2128;color:#e6edf3">{label}</option>'
        for label, _, _ in series
    )
    swatches = "".join(
        f'<span class="flt-swatch" style="background:{color}" data-series="{label}" data-cid="{cid}"></span>'
        for label, color, _ in series
    )
    return (
        f'<div class="flt-row">'
        f'<span class="flt-icon">⊞</span>'
        f'<div class="flt-swatches" id="sw-{cid}">{swatches}</div>'
        f'<select class="flt-select" id="flt-{cid}" multiple size="1" '
        f'onchange="applyFilter(this,\'{cid}\')" title="Filter series">'
        f'{opts}</select>'
        f'<span class="flt-hint">Filter models</span>'
        f'</div>'
    )

def line_chart(series, width=680, height=240, title=""):
    """SVG line chart — tall, readable, with viewBox for proper zoom."""
    if not series:
        return ""
    all_x = sorted({x for _, _, pts in series for x, _ in pts})
    all_y = [y for _, _, pts in series for _, y in pts]
    if not all_x or not all_y:
        return ""
    cid = _cid()
    n_series = len(series)
    # legend rows: wrap every 3 items
    leg_cols = min(n_series, 3)
    leg_rows = (n_series + leg_cols - 1) // leg_cols
    leg_h = leg_rows * 18 + 8          # pixels reserved at top for legend
    pad_l, pad_r, pad_b = 52, 24, 36
    W, H = width, height
    plot_top = leg_h
    plot_h   = H - plot_top - pad_b
    mn, mx_y = min(all_y), max(all_y)
    rng = mx_y - mn if mx_y != mn else 0.01
    mn2, mx2 = mn - rng * 0.07, mx_y + rng * 0.07
    rng2 = mx2 - mn2
    def sx(x): return pad_l + (x / max(all_x)) * (W - pad_l - pad_r)
    def sy(y): return plot_top + plot_h - ((y - mn2) / rng2) * plot_h

    grid_vals = [mn2 + rng2 * i / 4 for i in range(5)]
    grid = "".join(
        f'<line x1="{pad_l}" y1="{sy(v):.1f}" x2="{W-pad_r}" y2="{sy(v):.1f}" stroke="#21262d" stroke-width="1"/>'
        f'<text x="{pad_l-6}" y="{sy(v)+4:.1f}" font-size="10" fill="#6e7681" text-anchor="end">{v*100:.0f}</text>'
        for v in grid_vals
    )
    parts = []
    for label, color, pts in series:
        spts = sorted(pts)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in spts)
        parts.append(
            f'<g class="series-g" data-series="{label}" data-cid="{cid}">'
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
            + "".join(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{color}" stroke="#0d1117" stroke-width="2" class="chart-dot" data-label="{label} @ {int(x)}\u00b0: {y*100:.1f}%"/>'
                for x, y in spts
            ) + '</g>'
        )
    xlabels = "".join(
        f'<text x="{sx(a):.1f}" y="{plot_top+plot_h+20}" font-size="10" fill="#6e7681" text-anchor="middle">{a}\u00b0</text>'
        for a in [0, 45, 90, 135, 180] if a <= max(all_x)
    )
    # legend: wrap into rows of leg_cols, each item 200px wide
    col_w = (W - pad_l) // leg_cols
    legend = "".join(
        f'<rect x="{pad_l + (i % leg_cols)*col_w}" y="{(i // leg_cols)*18}" width="16" height="3" rx="1.5" fill="{color}"/>'
        f'<text x="{pad_l + (i % leg_cols)*col_w + 22}" y="{(i // leg_cols)*18 + 10}" font-size="10" fill="#8b949e">{label}</text>'
        for i, (label, color, _) in enumerate(series)
    )
    title_svg = f'<text x="{W//2}" y="{plot_top+plot_h+pad_b-4}" font-size="10" fill="#6e7681" text-anchor="middle">{title}</text>' if title else ""
    total_h = H
    svg_inner = f'{grid}{legend}{"".join(parts)}{xlabels}{title_svg}'
    svg = (f'<svg id="svg-{cid}" viewBox="0 0 {W} {total_h}" width="{W}" height="{total_h}" '
           f'style="display:block;margin:.5rem 0 1rem">{svg_inner}</svg>')
    return (
        f'<div class="chart-wrap" data-cid="{cid}" style="position:relative">'
        f'{_filter_bar(series, cid)}'
        f'{svg}'
        f'<div class="tooltip" id="tt-{cid}"></div>'
        f'<button class="zoom-btn" onclick="openZoom(\'svg-{cid}\')" title="Expand">\u2922</button>'
        f'</div>'
    )


def bar_chart(labels, series, width=700, height=260, title=""):
    """Grouped horizontal bar chart — no overlapping bars, with viewBox for proper zoom."""
    n = len(labels)
    if not n or not series:
        return ""
    cid = _cid()
    ns = len(series)
    leg_h = 28
    pad_l, pad_r, pad_b = 155, 70, 36
    W = width
    # derive height from content so bars never overlap:
    # each group needs: ns bars * bar_h + (ns-1) gaps + inter-group gap
    bar_h   = 16
    gap     = 5    # gap between bars in a group
    grp_gap = 14   # gap between groups
    group_h = ns * bar_h + (ns - 1) * gap + grp_gap
    plot_h  = group_h * n
    H = leg_h + 4 + plot_h + pad_b
    plot_top = leg_h + 4
    all_vals = [v for _, _, vals in series for v in vals]
    mx = max(all_vals) if all_vals else 1
    def bx(v): return pad_l + (v / mx) * (W - pad_l - pad_r)

    bar_groups = []
    for gi, label in enumerate(labels):
        gy = plot_top + gi * group_h
        # centre label vertically in the group
        label_y = gy + (ns * bar_h + (ns - 1) * gap) / 2 + 4
        row = f'<text x="{pad_l-10}" y="{label_y:.1f}" font-size="11" fill="#8b949e" text-anchor="end">{label}</text>'
        for si, (name, color, vals) in enumerate(series):
            by = gy + si * (bar_h + gap)
            bw = bx(vals[gi]) - pad_l
            row += (f'<rect x="{pad_l}" y="{by:.1f}" width="{max(bw,2):.1f}" height="{bar_h:.1f}" '
                    f'fill="{color}" opacity="0.88" rx="3" class="chart-dot" '
                    f'data-label="{name} \u2014 {label}: {vals[gi]*100:.1f}%" '
                    f'data-series="{name}" data-cid="{cid}"/>')
            row += (f'<text x="{pad_l+bw+8:.1f}" y="{by+bar_h-2:.1f}" font-size="10" '
                    f'fill="#6e7681" data-series="{name}" data-cid="{cid}" class="bar-val">{vals[gi]*100:.1f}</text>')
        bar_groups.append(row)

    ticks = "".join(
        f'<line x1="{bx(v):.1f}" y1="{plot_top}" x2="{bx(v):.1f}" y2="{plot_top+plot_h}" stroke="#21262d" stroke-width="1"/>'
        f'<text x="{bx(v):.1f}" y="{plot_top+plot_h+18}" font-size="10" fill="#6e7681" text-anchor="middle">{int(v*100)}</text>'
        for v in [0.2, 0.4, 0.6, 0.8, 1.0] if v <= mx
    )
    col_w = max(140, (W - pad_l) // max(ns, 1))
    legend = "".join(
        f'<rect x="{pad_l + i*col_w}" y="6" width="14" height="14" rx="3" fill="{color}"/>'
        f'<text x="{pad_l + i*col_w + 20}" y="17" font-size="11" fill="#8b949e">{name}</text>'
        for i, (name, color, _) in enumerate(series)
    )
    title_svg = f'<text x="{W//2}" y="{plot_top+plot_h+pad_b-4}" font-size="11" fill="#6e7681" text-anchor="middle">{title}</text>' if title else ""
    svg_inner = f'{ticks}{"".join(bar_groups)}{legend}{title_svg}'
    svg = (f'<svg id="svg-{cid}" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'style="display:block;margin:.5rem 0 1.5rem">{svg_inner}</svg>')
    return (
        f'<div class="chart-wrap" data-cid="{cid}" style="position:relative">'
        f'{_filter_bar(series, cid)}'
        f'{svg}'
        f'<div class="tooltip" id="tt-{cid}"></div>'
        f'<button class="zoom-btn" onclick="openZoom(\'svg-{cid}\')" title="Expand">\u2922</button>'
        f'</div>'
    )


# ── Experiment builders ───────────────────────────────────────────────────────

COLORS = ["#58a6ff", "#f78166", "#3fb950", "#d29922", "#bc8cff", "#39d353"]

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

    # Line charts for R@1, R@5, mAP@5
    series_r1   = [(m, COLORS[i % len(COLORS)], [(r["angle"], r["r1"])   for r in data[m][ds0]]) for i, m in enumerate(models)]
    series_r5   = [(m, COLORS[i % len(COLORS)], [(r["angle"], r["r5"])   for r in data[m][ds0]]) for i, m in enumerate(models)]
    series_map5 = [(m, COLORS[i % len(COLORS)], [(r["angle"], r["map5"]) for r in data[m][ds0]]) for i, m in enumerate(models)]
    chart = (
        line_chart(series_r1,   title="R@1 (%) vs rotation angle") +
        line_chart(series_r5,   title="R@5 (%) vs rotation angle") +
        line_chart(series_map5, title="mAP@5 (%) vs rotation angle")
    )

    summary = "\n".join(
        f"{m}: R@1@0={fmt(idx[m].get(0,{}).get('r1'))}%, R@5@0={fmt(idx[m].get(0,{}).get('r5'))}%, mAP5@0={fmt(idx[m].get(0,{}).get('map5'))}%, R@1@180={fmt(idx[m].get(180,{}).get('r1'))}%"
        for m in models
    )
    ai = ask_llm(
        "You are a researcher analyzing 3D point cloud retrieval results. "
        "The task is shape retrieval under arbitrary SO(3) rotations — a model must return the same class of shape regardless of how it is rotated. "
        "Higher R@1 means the top-1 retrieved shape matches the query class. "
        "Rotation invariance means R@1 stays flat across all angles 0-180 degrees. "
        f"Results (ShapeNet, 19 angles 0-180):\n{summary}\n"
        "Write 5-6 sentences covering: (1) which model achieves the best R@1 and by how much, "
        "(2) how each model degrades as rotation increases and what that reveals about their invariance, "
        "(3) why RISA's architecture (sparse attention + rotation-invariant features) leads to stable performance, "
        "(4) practical implications for real-world deployment where object orientation is unknown."
    )
    return dict(
        id="exp1", title="Exp 1 — Rotation Robustness",
        question="How does retrieval performance degrade as rotation angle increases from 0° to 180°?",
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
    palette = ["#58a6ff","#f78166","#3fb950","#d29922","#bc8cff","#39d353","#ffa657","#79c0ff"]
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
    tsne_id = _cid()
    svg = (f'<svg id="svg-{tsne_id}" viewBox="0 0 {width} {total_h}" width="{width}" height="{total_h}" '
           f'style="display:block;margin:.5rem auto 1rem;background:#161b22;border-radius:8px">'
           f'{dots}{legend}{cap}</svg>')
    return (
        f'<div class="chart-wrap" data-cid="{tsne_id}" style="position:relative">'
        f'{svg}'
        f'<button class="zoom-btn" onclick="openZoom(\'svg-{tsne_id}\')" title="Expand">⤢</button>'
        f'</div>'
    )


def build_exp2(data):
    if not data:
        return None
    models = list(data.keys())
    print("  Running t-SNE for exp2 (this may take ~30s)...")
    blocks = ""
    for m in models:
        encs = data[m]["encodings"]
        labs = data[m]["labels"]
        blocks += (f'<div style="margin-bottom:1.5rem;padding-bottom:1.5rem;border-bottom:1px solid #f0f0f0">'
                   f'<p style="font-weight:700;font-size:.95rem;margin-bottom:.4rem">{m}</p>'
                   + tsne_svg(encs, labs, width=520, height=300) +
                   f'</div>')
    plot_wrap = f'<div>{blocks}</div>'
    ai = ask_llm(
        "You are a researcher analyzing t-SNE visualizations of 3D shape embeddings. "
        "Each model encodes 1280 ShapeNet shapes (8 classes, 160 per class) into 512-d vectors, then t-SNE projects to 2D. "
        f"Models compared: {', '.join(models)}. RISA is the proposed rotation-invariant sparse attention model. "
        "Write 5-6 sentences covering: (1) what tight, well-separated clusters in t-SNE indicate about a model's discriminative power, "
        "(2) what semantic collapse means — when embeddings of different classes overlap — and why it destroys retrieval precision, "
        "(3) why non-invariant models (PointNet++, DGCNN) are expected to show more collapse under rotation, "
        "(4) what RISA's cluster structure reveals about how rotation invariance affects the embedding space geometry, "
        "(5) the connection between cluster separation in t-SNE and mAP@5 retrieval metrics."
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
    # Grouped bar chart: each condition as a group, bars for 0°/90°/180°
    bar_labels = [c.replace("_", " ") for c in conditions]
    bar_series = [
        ("R@1 0°",   "#4a6cf7", [key_angles(data[c]).get(0,   {}).get("r1", 0) for c in conditions]),
        ("R@1 90°",  "#e74c3c", [key_angles(data[c]).get(90,  {}).get("r1", 0) for c in conditions]),
        ("R@1 180°", "#2ecc71", [key_angles(data[c]).get(180, {}).get("r1", 0) for c in conditions]),
    ]
    chart3 = bar_chart(bar_labels, bar_series, title="R@1 (%) per condition at 0°, 90°, 180°")
    summary = "\n".join(
        f"{c}: R@1@0={fmt(key_angles(data[c]).get(0,{}).get('r1'))}%, "
        f"R@1@180={fmt(key_angles(data[c]).get(180,{}).get('r1'))}%"
        for c in conditions
    )
    ai = ask_llm(
        "You are a researcher analyzing an ablation study for 3D point cloud retrieval on ModelNet40. "
        "The goal is to identify which components of RISA are responsible for rotation-invariant retrieval. "
        "Conditions: A_RISA=full model, B_DGCNN_SO3=DGCNN with SO(3) data augmentation during training, "
        "C_DGCNN_TTA=DGCNN with test-time averaging over multiple rotations, "
        "D_RISA_XYZ=RISA backbone but using raw XYZ coordinates instead of rotation-invariant features, "
        "E_RISA_NO_TOKEN=RISA without the global encoding token that aggregates local attention. "
        f"Results:\n{summary}\n"
        "Write 5-6 sentences covering: (1) which condition performs best and worst and by what margin, "
        "(2) what the gap between A and D reveals about the importance of rotation-invariant input features vs architecture, "
        "(3) what removing the encoding token (E) costs and why that token matters for global shape representation, "
        "(4) whether augmentation-based approaches (B, C) are competitive and what their trade-offs are, "
        "(5) the key architectural insight the ablation confirms."
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
        table=chart3 + tbl, ai=ai,
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
    chart = line_chart(series)
    r1_old  = fmt(key_angles(list(old_data.values())[0]).get(0, {}).get("r1"))
    r1_perc = fmt(key_angles(list(perc_data.values())[0]).get(0, {}).get("r1"))
    ai = ask_llm(
        "You are a researcher analyzing cross-dataset transfer for 3D point cloud retrieval. "
        "The model is trained on ModelNet40 (synthetic CAD models) and tested on ScanObjectNN (real-world scanned objects with noise and occlusion). "
        "This tests whether learned rotation-invariant features generalise beyond the training distribution. "
        "perceiver=False uses standard self-attention to update point features; "
        "perceiver=True uses a perceiver-style cross-attention where a fixed set of latent queries attend to point features, "
        "decoupling the output dimensionality from the number of input points and potentially improving robustness to variable point density. "
        f"perceiver=False R@1@0={r1_old}%, perceiver=True R@1@0={r1_perc}%. "
        "Write 6-7 sentences covering: (1) which variant transfers better and by what margin across all rotation angles, "
        "(2) why the perceiver architecture may help or hurt generalisation to noisy real-world scans — consider how fixed latent queries "
        "act as a bottleneck that filters noise, "
        "(3) what the cross-dataset gap between ModelNet40 and ScanObjectNN reveals about the difficulty of sim-to-real transfer in 3D retrieval, "
        "(4) how rotation invariance interacts with domain shift — does invariance help more or less when the target domain has sensor noise and occlusion, "
        "(5) what the per-angle R@1 curves reveal about whether one variant degrades more under rotation in the real-world domain, "
        "(6) practical implications for deploying retrieval systems on real sensor data such as LiDAR or RGB-D scans."
    )
    return dict(
        id="exp4", title="Exp 4 — Perceiver vs Standard RISA",
        question="Does the perceiver-style attention update strategy improve cross-dataset transfer?",
        info=info_grid(
            Models=tag_list(["RISA (perceiver=False)", "RISA (perceiver=True)"]),
            **{"Train Dataset": tag_list(["ModelNet40"])},
            **{"Test Dataset": tag_list(["ScanObjectNN"])},
            Metric="R@1 · R@5 · mAP@5",
            Training="Proxy Anchor, 100 epochs",
            Evaluation="SO(3) rotations 0°→180° (7 angles)",
        ),
        table=chart + tbl, ai=ai,
    )


def build_exp8(data):
    if not data:
        return None
    # Final test results from exp8_run_20260916_140537.log
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
    bar_labels = [m for m, _, _ in all_models]
    bar_series = [
        ("class mAP@5", "#4a6cf7", [c for _, c, _ in all_models]),
        ("part mAP@5",  "#e74c3c", [p for _, _, p in all_models]),
    ]
    chart8 = bar_chart(bar_labels, bar_series, title="class mAP@5 vs part mAP@5 (%) — all methods")
    ai = ask_llm(
        "You are a researcher analyzing part-level 3D shape retrieval on ShapeNet. "
        "Part-level retrieval means the query is a shape and the goal is to retrieve shapes with similar part structure "
        "(e.g. a chair with similar legs), not just the same global class. "
        "class mAP@5 measures retrieval by object category; part mAP@5 measures retrieval by part label agreement. "
        f"Results:\n{chr(10).join(summary_lines)}\n"
        "Write 5-6 sentences covering: (1) the overall ranking of methods and the gap between RISA and the next best, "
        "(2) why 3D-SIFT performs so much worse and what that reveals about hand-crafted vs learned features, "
        "(3) the difference between class mAP@5 and part mAP@5 scores — what it means when a model scores high on class but lower on parts, "
        "(4) why rotation invariance specifically helps part-level retrieval where part orientation varies, "
        "(5) what this experiment demonstrates about RISA's ability to capture fine-grained geometric structure."
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
        table=chart8 + tbl, ai=ai,
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
    series10 = [
        (p, COLORS[i % len(COLORS)], [(r["angle"], r["r1"]) for r in data[p]])
        for i, p in enumerate(priors)
    ]
    chart10 = line_chart(series10, title="R@1 (%) vs rotation angle — per prior")
    summary = "\n".join(
        f"{p}: R@1@0={fmt(key_angles(data[p]).get(0,{}).get('r1'))}%, "
        f"mAP@5={fmt(key_angles(data[p]).get(0,{}).get('map5'))}%"
        for p in priors
    )
    ai = ask_llm(
        "You are a researcher analyzing a point selection prior ablation for 3D shape retrieval. "
        "The model processes N=1024 points but a prior selects K=256 points before the main attention layers, "
        "reducing compute while ideally keeping the most informative points. "
        "full_N=all 1024 points (compute baseline), random_K=random 256 (information lower bound), "
        "fps_K=farthest point sampling (uniform spatial coverage), "
        "eigenentropy_K=top-256 by local eigenentropy (regions of geometric complexity), "
        "surface_var_K=top-256 by surface variation (high-curvature regions), "
        "curvature_K=top-256 by anisotropy (edge-like regions), "
        "salient_K=top-256 by learned encoder attention weights (data-driven saliency). "
        f"Results:\n{summary}\n"
        "Write 5-6 sentences covering: (1) which prior achieves the best R@1 and mAP@5, "
        "(2) whether K=256 with a good prior can match or beat full N=1024 and what that implies about point cloud redundancy, "
        "(3) how geometric priors (eigenentropy, surface_var, curvature) compare to the learned salient_K prior, "
        "(4) why random_K is a useful lower bound and what the gap to fps_K reveals, "
        "(5) the practical implication: can we run RISA 4x faster with minimal accuracy loss by choosing the right prior?"
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
        table=chart10 + tbl, ai=ai,
    )


# ── HTML ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:#0d1117;color:#e6edf3}
.flt-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem;flex-wrap:wrap}
.flt-icon{font-size:.9rem;color:#58a6ff;flex-shrink:0}
.flt-hint{font-size:.72rem;color:#484f58;font-style:italic}
.flt-swatches{display:flex;gap:.3rem;align-items:center}
.flt-swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0;opacity:.9}
.flt-select{background:#161b22;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;
            padding:.3rem .6rem;font-size:.78rem;font-family:inherit;cursor:pointer;
            min-width:160px;transition:border-color .18s;outline:none}
.flt-select:hover,.flt-select:focus{border-color:#58a6ff}
.flt-select option{background:#1c2128;color:#e6edf3;padding:.2rem}
.zoom-btn{position:absolute;top:0;right:0;background:#161b22;border:1px solid #30363d;
          color:#8b949e;border-radius:5px;padding:.2rem .45rem;font-size:.8rem;cursor:pointer;
          transition:all .18s;line-height:1}
.zoom-btn:hover{color:#e6edf3;border-color:#58a6ff}
.zoom-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:9999;
              align-items:center;justify-content:center;backdrop-filter:blur(6px)}
.zoom-overlay.open{display:flex}
.zoom-inner{position:relative;background:#161b22;border:1px solid #30363d;border-radius:12px;
            padding:3rem 1.5rem 1.5rem;width:96vw;height:94vh;
            box-shadow:0 24px 80px rgba(0,0,0,.8);display:flex;align-items:center;justify-content:center;overflow:hidden}
.zoom-close{position:absolute;top:.6rem;right:.8rem;background:rgba(255,255,255,.06);border:1px solid #30363d;
            color:#8b949e;font-size:1.2rem;cursor:pointer;line-height:1;padding:.3rem .55rem;
            border-radius:6px;transition:all .15s;z-index:10}
.zoom-close:hover{color:#e6edf3;background:rgba(255,255,255,.12)}
#zoom-svg-container{width:100%;height:100%;display:flex;align-items:center;justify-content:center}
#zoom-svg-container svg{display:block;width:100%;height:100%;object-fit:contain}
header{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1a2332 100%);
       color:white;padding:4rem 2rem 3rem;text-align:center;position:relative;overflow:hidden}
header::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(74,108,247,.18) 0%,transparent 70%);pointer-events:none}
header h1{font-size:2.6rem;font-weight:800;letter-spacing:-0.03em;margin-bottom:.5rem;
  background:linear-gradient(135deg,#fff 30%,#8892b0);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
header .subtitle{color:#8892b0;font-size:1rem;margin-bottom:1.8rem}
header .badges{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap}
.badge{background:rgba(74,108,247,.15);border:1px solid rgba(74,108,247,.3);
       color:#a5b4fc;padding:.3rem .9rem;border-radius:20px;font-size:.78rem;font-weight:500;
       backdrop-filter:blur(4px)}
.tab-bar{display:flex;gap:.25rem;padding:1.5rem 1.5rem 0;
         max-width:1140px;margin:0 auto;flex-wrap:wrap}
.tab-btn{background:#161b22;border:1px solid #30363d;border-bottom:none;
         padding:.65rem 1.3rem;border-radius:8px 8px 0 0;
         font-size:.83rem;font-weight:600;color:#8b949e;cursor:pointer;
         transition:all .2s;font-family:inherit}
.tab-btn:hover{color:#e6edf3;background:#1c2128;border-color:#484f58}
.tab-btn.active{color:#79c0ff;background:#0d1117;border-color:#30363d;border-bottom:1px solid #0d1117;
                margin-bottom:-1px;z-index:1}
.tab-panels{max-width:1140px;margin:0 auto;padding:0 1.5rem 3rem}
.tab-panel{display:none;background:#0d1117;border:1px solid #30363d;border-radius:0 8px 8px 8px;
           overflow:hidden;animation:fadeIn .25s ease}
.tab-panel.active{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.panel-header{padding:2rem 2rem 1.5rem;border-bottom:1px solid #21262d;
  background:linear-gradient(180deg,#161b22 0%,#0d1117 100%)}
.panel-header h2{font-size:1.35rem;color:#e6edf3;margin-bottom:.5rem;font-weight:700}
.question{color:#79c0ff;font-size:.92rem;font-style:italic;margin-bottom:1.3rem;line-height:1.5}
.info-grid{display:flex;gap:.7rem;flex-wrap:wrap;margin-bottom:.5rem}
.info-card{background:#161b22;border:1px solid #30363d;border-radius:8px;
           padding:.6rem 1rem;min-width:130px;transition:border-color .2s}
.info-card:hover{border-color:#484f58}
.info-label{display:block;font-size:.68rem;font-weight:700;color:#58a6ff;
            text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem}
.info-value{font-size:.82rem;color:#c9d1d9;display:flex;flex-wrap:wrap;gap:.3rem}
.tag{background:rgba(88,166,255,.12);color:#79c0ff;padding:.15rem .55rem;
     border-radius:4px;font-size:.76rem;font-weight:600;border:1px solid rgba(88,166,255,.2)}
.panel-body{padding:1.8rem 2rem}
.chart-wrap{margin-bottom:.75rem}
.chart-title{font-size:.78rem;font-weight:600;color:#8b949e;text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:.5rem}
.tbl-wrap{overflow-x:auto;margin-bottom:1.8rem;border-radius:8px;border:1px solid #21262d}
table{width:100%;border-collapse:collapse;font-size:.85rem}
caption{text-align:left;font-size:.75rem;color:#6e7681;padding:.5rem .9rem;font-style:italic;
        background:#161b22;border-bottom:1px solid #21262d}
thead tr{background:#161b22}
th{color:#8b949e;padding:.7rem .9rem;text-align:left;font-weight:600;font-size:.78rem;
   text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid #21262d}
td{padding:.6rem .9rem;border-bottom:1px solid #161b22;color:#c9d1d9;transition:background .15s}
tbody tr:hover td{background:#161b22}
tbody tr:last-child td{border-bottom:none}
td strong{color:#79c0ff}
.ai-box{background:linear-gradient(135deg,rgba(88,166,255,.06),rgba(163,113,247,.06));
        border:1px solid rgba(88,166,255,.2);border-left:3px solid #58a6ff;
        border-radius:8px;padding:1.3rem 1.5rem;margin-top:1rem}
.ai-label{font-size:.7rem;font-weight:700;color:#58a6ff;text-transform:uppercase;
          letter-spacing:.1em;display:flex;align-items:center;gap:.4rem;margin-bottom:.7rem}
.ai-label::before{content:'✦';font-size:.8rem}
.ai-box p{font-size:.9rem;line-height:1.75;color:#c9d1d9}
.pending{color:#6e7681;font-style:italic;padding:1rem 0}
.ov-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:1rem;padding:.5rem 0 1rem}
.ov-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.3rem 1.5rem;
         cursor:pointer;transition:all .22s}
.ov-card:hover{border-color:#58a6ff;box-shadow:0 0 0 1px rgba(88,166,255,.2),0 8px 24px rgba(0,0,0,.4);
               transform:translateY(-3px)}
.ov-num{font-size:.68rem;font-weight:700;color:#58a6ff;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.35rem}
.ov-title{font-size:.98rem;font-weight:700;color:#e6edf3;margin-bottom:.4rem}
.ov-q{font-size:.8rem;color:#79c0ff;font-style:italic;margin-bottom:.5rem;line-height:1.45}
.ov-desc{font-size:.78rem;color:#8b949e;line-height:1.55}
footer{text-align:center;padding:2.5rem;color:#484f58;font-size:.78rem;border-top:1px solid #21262d}
.tooltip{position:absolute;background:#1c2128;border:1px solid #30363d;border-radius:6px;
         padding:.4rem .7rem;font-size:.75rem;color:#e6edf3;pointer-events:none;
         opacity:0;transition:opacity .15s;white-space:nowrap;z-index:100}
"""

JS = """
function showTab(id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab-btn[data-tab="'+id+'"]').classList.add('active');
  document.getElementById(id).classList.add('active');
}

function applyFilter(sel, cid) {
  // collect selected values; if none selected treat as all visible
  var selected = Array.from(sel.selectedOptions).map(function(o){ return o.value; });
  var allOpts  = Array.from(sel.options).map(function(o){ return o.value; });
  var visible  = selected.length === 0 ? allOpts : selected;
  allOpts.forEach(function(name) {
    var show = visible.indexOf(name) !== -1;
    // line chart series groups
    document.querySelectorAll('.series-g[data-cid="'+cid+'"][data-series="'+name+'"]')
      .forEach(function(g){ g.style.display = show ? '' : 'none'; });
    // bar chart rects + value labels
    document.querySelectorAll('[data-cid="'+cid+'"][data-series="'+name+'"]')
      .forEach(function(el){ el.style.display = show ? '' : 'none'; });
    // dim swatch
    var sw = document.querySelector('.flt-swatch[data-cid="'+cid+'"][data-series="'+name+'"]');
    if (sw) sw.style.opacity = show ? '0.9' : '0.2';
  });
}

function openZoom(svgId) {
  var src = document.getElementById(svgId);
  if (!src) return;
  var container = document.getElementById('zoom-svg-container');
  container.innerHTML = '';
  var clone = src.cloneNode(true);
  clone.removeAttribute('id');
  if (!clone.getAttribute('viewBox')) {
    clone.setAttribute('viewBox', '0 0 ' + (src.getAttribute('width')||800) + ' ' + (src.getAttribute('height')||400));
  }
  // let CSS flex + viewBox handle all scaling
  clone.removeAttribute('width');
  clone.removeAttribute('height');
  clone.style.cssText = '';
  container.appendChild(clone);
  document.getElementById('zoom-overlay').classList.add('open');
}

function closeZoom() {
  document.getElementById('zoom-overlay').classList.remove('open');
}

document.addEventListener('DOMContentLoaded', function() {
  document.addEventListener('keydown', function(e){ if(e.key==='Escape') closeZoom(); });
  document.getElementById('zoom-overlay').addEventListener('click', function(e){
    if(e.target === this) closeZoom();
  });
  document.querySelectorAll('.chart-dot').forEach(function(el) {
    el.addEventListener('mouseenter', function() {
      var cid = el.closest('.chart-wrap') && el.closest('.chart-wrap').dataset.cid;
      var tt = cid ? document.getElementById('tt-'+cid) : null;
      if (!tt) return;
      tt.textContent = el.dataset.label;
      tt.style.opacity = '1';
      var r = el.getBoundingClientRect();
      var pr = el.closest('.chart-wrap').getBoundingClientRect();
      tt.style.left = (r.left - pr.left + r.width/2 - tt.offsetWidth/2) + 'px';
      tt.style.top  = (r.top  - pr.top  - tt.offsetHeight - 6) + 'px';
    });
    el.addEventListener('mouseleave', function() {
      var cid = el.closest('.chart-wrap') && el.closest('.chart-wrap').dataset.cid;
      var tt = cid ? document.getElementById('tt-'+cid) : null;
      if (tt) tt.style.opacity = '0';
    });
  });
});
"""

def render_panel(exp, active=False):
    header_extra = (
        f'<p class="question">{exp["question"]}</p>{exp["info"]}'
        if exp.get("question") else ""
    )
    ai_box = (
        f'<div class="ai-box"><div class="ai-label">{exp.get("ai_label", "Analysis")}</div><p>{exp["ai"]}</p></div>'
        if exp.get("ai") else ""
    )
    cls = "tab-panel active" if active else "tab-panel"
    return (
        f'<div class="{cls}" id="{exp["id"]}">'
        f'<div class="panel-header">'
        f'<h2>{exp["title"]}</h2>'
        f'{header_extra}'
        f'</div>'
        f'<div class="panel-body">'
        f'{exp["table"]}'
        f'{ai_box}'
        f'</div></div>'
    )

EXP_OVERVIEW = [
    ("Exp 1",  "Rotation Robustness",             "How does retrieval performance degrade as rotation angle increases from 0° to 180°?",
     "Evaluates R@1 across 19 SO(3) rotation angles (0°→180°) on ShapeNet, comparing PointNet++, DGCNN, DiPVNet and RISA."),
    ("Exp 2",  "Semantic Collapse",              "Do non-invariant models collapse semantically distinct shapes into the same embedding region?",
     "t-SNE of 512-d embeddings for 1280 ShapeNet shapes (8 classes). Well-separated clusters indicate discriminative embeddings."),
    ("Exp 3",  "Architecture Ablation",          "Which architectural components of RISA are essential for rotation-invariant retrieval?",
     "5 conditions on ModelNet40: full RISA, DGCNN+SO3 aug, DGCNN+TTA, RISA with raw XYZ, RISA without encoding token."),
    ("Exp 4",  "Perceiver vs Standard RISA",     "Does the perceiver-style attention update strategy improve cross-dataset transfer?",
     "Trains on ModelNet40, tests on ScanObjectNN. Compares perceiver=True vs perceiver=False across all rotation angles."),
    ("Exp 8",  "Part-Level Retrieval",           "Can RISA retrieve shapes by part similarity, not just global class?",
     "ShapeNet part retrieval (16 classes). Reports class mAP@5 and part mAP@5 for 6 methods including 3D-SIFT and RISA."),
    ("Exp 10", "Prior Point Selection Ablation", "Which geometric prior best selects K=256 informative points from N=1024?",
     "7 priors on ModelNet40: full_N, random_K, fps_K, eigenentropy_K, surface_var_K, curvature_K, salient_K (learned)."),
]

def build_overview():
    cards = "".join(
        f'<div class="ov-card" onclick="showTab(\'exp{row[0].split()[1]}\')">'
        f'<div class="ov-num">{row[0]}</div>'
        f'<div class="ov-title">{row[1]}</div>'
        f'<div class="ov-q">{row[2]}</div>'
        f'<div class="ov-desc">{row[3]}</div>'
        f'</div>'
        for row in EXP_OVERVIEW
    )
    exp_summary = "\n".join(
        f"{r[0]} ({r[1]}): {r[3]}" for r in EXP_OVERVIEW
    )
    synthesis = ask_llm(
        "You are writing a research summary for a paper on RISA: Rotation-Invariant Sparse Attention for 3D point cloud retrieval. "
        "The paper runs 6 experiments to validate the approach from multiple angles. "
        f"Experiment summaries:\n{exp_summary}\n"
        "Write a cohesive 6-8 sentence narrative that: "
        "(1) states the core problem RISA solves (rotation sensitivity in 3D retrieval), "
        "(2) summarises what each experiment contributes to the overall argument, "
        "(3) explains how the experiments build on each other — from basic retrieval to ablation to part-level to efficiency, "
        "(4) highlights the strongest evidence that RISA works, "
        "(5) acknowledges any limitations or open questions the experiments reveal. "
        "Write in a clear academic tone suitable for a results webpage.",
        max_tokens=1200,
    )
    return dict(
        id="overview", title="Overview",
        question="",
        info="",
        table=f'<div class="ov-grid">{cards}</div>',
        ai=synthesis,
        ai_label="Paper Summary",
    )


def build_html(experiments):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_tabs = [build_overview()] + experiments
    tab_btns = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" '
        f'data-tab="{e["id"]}" onclick="showTab(\'{e["id"]}\')">'
        f'{e["title"].split("—")[0].strip()}</button>'
        for i, e in enumerate(all_tabs)
    )
    panels = "".join(render_panel(e, active=(i == 0)) for i, e in enumerate(all_tabs))
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
        f'<div class="zoom-overlay" id="zoom-overlay">'
        f'<div class="zoom-inner">'
        f'<button class="zoom-close" onclick="closeZoom()">✕</button>'
        f'<div id="zoom-svg-container"></div>'
        f'</div></div>'
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
