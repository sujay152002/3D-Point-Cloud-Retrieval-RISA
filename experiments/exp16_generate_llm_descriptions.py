"""Exp 16 — Step 1: Generate per-shape descriptions and LLaMA text embeddings
using PointLLM (native point cloud → text, no rendering needed).

Pipeline
--------
For each 3D shape:
  1. Feed raw point cloud directly to PointLLM-7B → geometric description
  2. Encode description with frozen LLaMA-2-7B text encoder
     (last hidden state, mean-pooled → PCA-projected to LLM_DIM=512)
  3. Cache:
       outputs/text_cache/{dataset}_{split}_llm_desc.json   ← raw descriptions
       outputs/text_cache/{dataset}_{split}_llm_emb.pt      ← [N, LLM_DIM] embeddings

Why PointLLM over LLaVA?
--------------------------
  PointLLM takes raw point clouds as input — no rendering step, no depth
  information lost. It was trained on ~660k point-text pairs from Objaverse
  and generates geometric descriptions that explicitly reference parts,
  topology, and spatial structure.

Usage
-----
  # ShapeNet (default)
  python experiments/exp16_generate_llm_descriptions.py

  # ModelNet40
  python experiments/exp16_generate_llm_descriptions.py --dataset modelnet40

  # Resume from partial cache
  python experiments/exp16_generate_llm_descriptions.py --resume
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT      = Path(__file__).parent.parent
POINTLLM_ROOT = Path.home() / "PointLLM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(POINTLLM_ROOT))

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
CACHE_DIR = ROOT / "outputs" / "text_cache"
LLM_DIM   = 512   # projected output dim (matches RISA encoding_out_dim)

POINTLLM_MODEL = "RunsenXu/PointLLM_7B_v1.2"
LLAMA_MODEL    = "meta-llama/Llama-2-7b-hf"

DESCRIBE_PROMPT = "Describe the 3D geometry and structure of this object in detail, focusing on its parts, shape, and topology."


# ── PointLLM describer ────────────────────────────────────────────────────────

class PointLLMDescriber:
    """Wraps PointLLM-7B for native point cloud → text description.

    Takes [N, 3] or [N, 6] point clouds directly (no rendering).
    Loaded in bfloat16 — fits comfortably on a 24GB GPU.
    """

    def __init__(self, model_id: str = POINTLLM_MODEL):
        print(f"  Loading PointLLM: {model_id}")
        from transformers import AutoTokenizer
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.conversation import conv_templates
        from huggingface_hub import snapshot_download

        # Download/locate weights
        try:
            from huggingface_hub import hf_hub_download
            model_path = snapshot_download(model_id)
        except Exception as e:
            raise RuntimeError(
                f"Could not download PointLLM weights: {e}\n"
                f"Ensure you have internet access and HF credentials if needed."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = PointLLMLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            use_cache=True,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        self.model.initialize_tokenizer_point_backbone_config_wo_embedding(self.tokenizer)
        self.model.eval()

        # Extract point token config (required for correct prompt construction)
        point_cfg = self.model.get_model().point_backbone_config
        self._point_token_len         = point_cfg['point_token_len']
        self._mm_use_point_start_end  = point_cfg.get('mm_use_point_start_end', False)
        self._default_point_patch_token = point_cfg['default_point_patch_token']
        self._default_point_start_token = point_cfg.get('default_point_start_token', '')
        self._default_point_end_token   = point_cfg.get('default_point_end_token', '')

        self._conv_templates = conv_templates
        print("  ✓ PointLLM ready")

    @torch.inference_mode()
    def describe(self, points: np.ndarray,
                 prompt: str = DESCRIBE_PROMPT,
                 max_new_tokens: int = 200) -> str:
        """Generate a description for a point cloud.

        points: [N, 3] numpy array (xyz), any N. Will be resampled to 8192.
        PointLLM was trained with PointTransformer_8192point_2layer and
        requires exactly 8192 input points.
        """
        from pointllm.conversation import conv_templates, SeparatorStyle
        from pointllm.model.utils import KeywordsStoppingCriteria

        # Normalise to unit sphere
        centroid = points.mean(0)
        pts = points - centroid
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 0:
            pts = pts / scale

        # PointLLM requires exactly 8192 points — resample via repeat + shuffle
        target = 8192
        N = len(pts)
        if N < target:
            repeats = (target + N - 1) // N
            pts = np.tile(pts, (repeats, 1))[:target]
            # small jitter to avoid identical duplicates
            pts[N:] += np.random.randn(target - N, 3) * 1e-4
        elif N > target:
            idx = np.random.choice(N, target, replace=False)
            pts = pts[idx]

        # PointLLM expects [1, N, 6] (xyz + colour) — pad colour with zeros
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.zeros_like(pts)], axis=1)  # [8192, 6]
        pc_tensor = torch.tensor(pts, dtype=torch.bfloat16).unsqueeze(0).to(DEVICE)  # [1, 8192, 6]

        # Build conversation with correct PointLLM point token structure
        # Must use point_backbone_config tokens, NOT a bare <point> token
        conv = conv_templates["vicuna_v1_1"].copy()
        if self._mm_use_point_start_end:
            point_str = (self._default_point_start_token
                         + self._default_point_patch_token * self._point_token_len
                         + self._default_point_end_token)
        else:
            point_str = self._default_point_patch_token * self._point_token_len
        inp = f"{point_str}\n{prompt}"
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self.tokenizer(
            full_prompt, return_tensors="pt"
        ).input_ids.to(DEVICE)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        output_ids = self.model.generate(
            input_ids,
            point_clouds=pc_tensor,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping],
        )

        n_input = input_ids.shape[1]
        out = self.tokenizer.decode(
            output_ids[0, n_input:], skip_special_tokens=True
        ).strip()
        # Strip trailing separator
        if out.endswith(stop_str.strip()):
            out = out[: -len(stop_str.strip())].strip()
        return out


# ── LLaMA text encoder ────────────────────────────────────────────────────────

class LLaMATextEncoder:
    """Frozen LLaMA-2-7B used as a text encoder.

    Encodes descriptions → last hidden state mean-pooled → [4096].
    Loaded in 4-bit via bitsandbytes.
    """

    def __init__(self, model_id: str = LLAMA_MODEL):
        print(f"  Loading LLaMA text encoder: {model_id}")
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        )
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True,
        )
        self._model.eval()
        self.hidden_dim = self._model.config.hidden_size  # 4096 for 7B
        print(f"  ✓ LLaMA text encoder ready (hidden_dim={self.hidden_dim})")

    @torch.inference_mode()
    def encode(self, texts: list, batch_size: int = 8) -> torch.Tensor:
        """Returns [N, hidden_dim] mean-pooled, L2-normalised on CPU."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(next(self._model.parameters()).device)

            out = self._model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()             # [B, L, 4096]
            mask   = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = F.normalize(pooled, dim=-1)
            all_embs.append(pooled.cpu())

            if i % (batch_size * 20) == 0 or i + batch_size >= len(texts):
                print(f"    encoded {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)

        return torch.cat(all_embs, dim=0)


# ── PCA projection ────────────────────────────────────────────────────────────

def fit_pca_projection(embeddings: torch.Tensor, out_dim: int = LLM_DIM,
                       save_path: Path = None) -> nn.Linear:
    from sklearn.decomposition import PCA
    X = embeddings.numpy().astype(np.float64)
    print(f"  Fitting PCA {X.shape[1]} → {out_dim}…")
    pca = PCA(n_components=out_dim, whiten=True, random_state=42)
    pca.fit(X)
    W = torch.tensor(pca.components_, dtype=torch.float32)
    proj = nn.Linear(X.shape[1], out_dim, bias=False)
    proj.weight.data = W
    for p in proj.parameters():
        p.requires_grad = False
    if save_path:
        torch.save({
            "weight": W,
            "mean":   torch.tensor(pca.mean_, dtype=torch.float32),
            "var_explained": float(pca.explained_variance_ratio_.sum()),
        }, save_path)
        print(f"  PCA saved → {save_path}  "
              f"(var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    return proj


def load_pca_projection(load_path: Path) -> nn.Linear:
    ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
    W = ckpt["weight"]
    proj = nn.Linear(W.shape[1], W.shape[0], bias=False)
    proj.weight.data = W
    for p in proj.parameters():
        p.requires_grad = False
    return proj


# ── Main cache generation ─────────────────────────────────────────────────────

def generate_cache(dataset_name: str, split: str, args):
    from core.datasets import get_dataset

    desc_path = CACHE_DIR / f"{dataset_name}_{split}_llm_desc.json"
    emb_path  = CACHE_DIR / f"{dataset_name}_{split}_llm_emb.pt"
    pca_path  = CACHE_DIR / f"{dataset_name}_llm_pca.pt"

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n  Loading {dataset_name} {split}…")
    ds = get_dataset(dataset_name, split, DATA_ROOT, num_points=args.n_points)
    n_shapes = len(ds)
    if args.max_shapes:
        n_shapes = min(n_shapes, args.max_shapes)
    print(f"  {n_shapes} shapes")

    # Build class name lookup from base dataset
    from torch.utils.data import Subset
    base_ds = ds.dataset if isinstance(ds, Subset) else ds
    cat_map = {}
    for attr in ("categories", "classes", "class_names", "label_names"):
        val = getattr(base_ds, attr, None)
        if val is not None:
            if isinstance(val, dict):
                cat_map = val  # {int: str}
            elif isinstance(val, (list, tuple)):
                cat_map = {i: str(c) for i, c in enumerate(val)}
            break

    # ── Resume existing descriptions ──────────────────────────────────────────
    descriptions = {}
    if args.resume and desc_path.exists():
        with open(desc_path) as f:
            descriptions = json.load(f)
        print(f"  Resuming — {len(descriptions)} already cached")

    # ── Load PointLLM ─────────────────────────────────────────────────────────
    need_generation = sum(1 for i in range(n_shapes) if str(i) not in descriptions)
    if need_generation > 0:
        describer = PointLLMDescriber(args.pointllm_model)
    else:
        describer = None
        print(f"  All {n_shapes} descriptions already cached — skipping generation")

    # ── Generate descriptions ─────────────────────────────────────────────────
    if describer is not None:
        print(f"\n  Generating {need_generation} descriptions with PointLLM…")
        for idx in range(n_shapes):
            key = str(idx)
            if key in descriptions:
                continue

            sample = ds[idx]
            points = sample[0].numpy() if isinstance(sample[0], torch.Tensor) else np.array(sample[0])
            label  = sample[1].item()  if isinstance(sample[1], torch.Tensor) else int(sample[1])

            # points may be [3, N] (C-first) — transpose to [N, 3]
            if points.ndim == 2 and points.shape[0] == 3:
                points = points.T

            classname = cat_map.get(label, f"class_{label}")

            try:
                desc = describer.describe(points)
            except Exception as e:
                print(f"  ⚠ shape {idx} ({classname}): {e}")
                desc = f"A 3D {classname}."

            descriptions[key] = {
                "label": label, "classname": classname, "description": desc,
            }

            if idx % 50 == 0 or idx == n_shapes - 1:
                print(f"  [{idx+1}/{n_shapes}] {classname}: {desc[:90]}…", flush=True)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(desc_path, "w") as f:
                    json.dump(descriptions, f, indent=2)

        del describer
        torch.cuda.empty_cache()

    print(f"\n  ✓ Descriptions saved → {desc_path}")

    # ── Encode with LLaMA ─────────────────────────────────────────────────────
    if emb_path.exists():
        cached = torch.load(emb_path, map_location="cpu", weights_only=True)
        if cached.shape[0] == n_shapes and cached.shape[1] == LLM_DIM:
            print(f"  LLM embeddings already cached ({n_shapes} × {LLM_DIM}) ← {emb_path}")
            return cached

    print(f"\n  Encoding {n_shapes} descriptions with LLaMA…")
    text_encoder = LLaMATextEncoder(args.llama_model)
    texts = [descriptions[str(i)]["description"] for i in range(n_shapes)]
    raw_embs = text_encoder.encode(texts, batch_size=args.encode_batch_size)
    del text_encoder
    torch.cuda.empty_cache()

    # ── PCA projection ────────────────────────────────────────────────────────
    if split == "train":
        proj = fit_pca_projection(raw_embs, out_dim=LLM_DIM, save_path=pca_path)
    else:
        if not pca_path.exists():
            raise FileNotFoundError(
                f"PCA projection not found at {pca_path}. "
                f"Run train split first."
            )
        proj = load_pca_projection(pca_path)
        print(f"  Loaded PCA projection ← {pca_path}")

    with torch.no_grad():
        embs = proj(raw_embs)
        embs = F.normalize(embs, dim=-1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(embs, emb_path)
    print(f"  ✓ LLM embeddings saved → {emb_path}  shape={tuple(embs.shape)}")
    return embs


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           default="shapenet",
                   choices=["shapenet", "modelnet40", "objaverse", "partnet"])
    p.add_argument("--splits",            nargs="+", default=["train", "val", "test"])
    p.add_argument("--n_points",          type=int, default=8192,
                   help="Points fed to PointLLM (it expects up to 8192)")
    p.add_argument("--pointllm_model",    default=POINTLLM_MODEL)
    p.add_argument("--llama_model",       default=LLAMA_MODEL)
    p.add_argument("--encode_batch_size", type=int, default=8)
    p.add_argument("--max_shapes",        type=int, default=None)
    p.add_argument("--resume",            action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Exp 16 — PointLLM Description + LLaMA Embedding Cache")
    print(f"  dataset={args.dataset}  splits={args.splits}")
    print(f"  n_points={args.n_points} (PointLLM input)")
    print(f"  pointllm_model={args.pointllm_model}")
    print(f"  llama_model={args.llama_model}")
    print(f"  output: {CACHE_DIR}")
    print("=" * 60)

    for split in args.splits:
        generate_cache(args.dataset, split, args)

    print("\n  All splits done.")
    print(f"  Embeddings : {CACHE_DIR}/{args.dataset}_*_llm_emb.pt")
    print(f"  PCA proj   : {CACHE_DIR}/{args.dataset}_llm_pca.pt")
    print(f"  Descriptions: {CACHE_DIR}/{args.dataset}_*_llm_desc.json")


if __name__ == "__main__":
    main()
