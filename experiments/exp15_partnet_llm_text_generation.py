"""LLM-based PartNet Text Generation

Generates rich natural-language descriptions for PartNet shapes using an LLM.

Approach: Use BLIP-2 (or similar) to generate captions from point cloud renders
or use a text-only LLM with structured prompts based on part metadata.

Dependencies
------------
    pip install transformers accelerate
    pip install pillow

Usage
-----
    # Generate text descriptions for PartNet using LLM
    python experiments/exp15_partnet_llm_text_generation.py \\
        --partnet_root data/partnet \\
        --output_dir outputs/llm_descriptions \\
        --model blip2 \\
        --batch_size 8

    # Or use a simpler text-only approach with part counts
    python experiments/exp15_partnet_llm_text_generation.py \\
        --partnet_root data/partnet \\
        --output_dir outputs/llm_descriptions \\
        --mode text_only \\
        --model gpt2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForVisionLang
from PIL import Image
import time


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset, PartNetDataset
from core.geometry import normalize_pc


# Try to import BLIP-2 or fallback to text-only
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False


class PointCloudToImageRenderer:
    """Simple point cloud to image renderer using matplotlib/plotly.
    
    Creates orthographic projections from multiple views for LLM processing.
    """
    
    def __init__(self, image_size: int = 224, num_views: int = 4):
        self.image_size = image_size
        self.num_views = num_views
        self._matplotlib_imported = False
        
    def render_views(self, points: np.ndarray) -> List[Image.Image]:
        """Render multiple views of a point cloud as images.
        
        Args:
            points: [N, 3] point cloud
            
        Returns:
            List of PIL Images (one per view)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        self._matplotlib_imported = True
        images = []
        
        # Normalize to unit sphere
        points = points.copy()
        points = points - points.mean(axis=0)
        scale = np.linalg.norm(points, axis=1).max()
        if scale > 1e-8:
            points = points / scale
        
        # Create views from different angles
        angles = np.linspace(0, 180, self.num_views)
        
        for angle in angles:
            # Rotate around Y axis
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            rotated = points @ R.T
            
            # Create 2D projection (top-down)
            fig, ax = plt.subplots(figsize=(1, 1), dpi=self.image_size)
            ax.scatter(rotated[:, 0], rotated[:, 2], s=1, c='black', alpha=0.5)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axis('off')
            ax.set_facecolor('white')
            
            # Render to buffer
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            img = Image.frombuffer('RGB', (self.image_size, self.image_size), buf)
            images.append(img)
            
            plt.close(fig)
        
        return images


class LLMLoader:
    """Load LLM models for text generation."""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.processor = None
        self.model = None
        
    def load(self):
        """Load the model and processor."""
        print(f"Loading {self.model_name}...")
        
        if "blip2" in self.model_name.lower() and BLIP_AVAILABLE:
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"✓ Loaded {self.model_name}")
        else:
            print(f"Warning: BLIP2 not available, using text-only mode")
            self.model = None
            self.processor = None


def generate_text_with_blip2(
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    images: List[Image.Image],
    prompt: str = "Describe this 3D object."
) -> List[str]:
    """Generate text descriptions using BLIP-2."""
    texts = []
    
    for img in images:
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(
            model.device, dtype=torch.float16
        )
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        texts.append(generated_text)
    
    return texts


def generate_text_with_vlm(
    model: Any,
    processor: Any,
    points: np.ndarray,
    prompt: str = "Describe this 3D object in detail."
) -> str:
    """Generate description using a VLM from point cloud."""
    renderer = PointCloudToImageRenderer()
    views = renderer.render_views(points)
    return generate_text_with_blip2(model, processor, views, prompt)[0]


def generate_text_with_llm(
    prompt: str,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    device: str = "cuda"
) -> str:
    """Generate text using a text-only LLM (requires API key or local access)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Prompt templates for structured generation
PARTNET_PROMPT_TEMPLATES = [
    "Describe a {category} with {parts_description}.",
    "Generate a detailed description of a {category} that has {parts_description}.",
    "A {category} featuring {parts_description}.",
    "Describe the structure of a {category} including: {parts_description}.",
    "What does a {category} look like? It has {parts_description}.",
]


def build_part_description(
    part_ids: List[int],
    part_counts: Dict[int, int],
    category: str,
    categories_map: Dict[int, str]
) -> str:
    """Build a structured part description from part labels."""
    parts_list = []
    
    for pid, count in part_counts.items():
        if pid == 0:
            continue  # Skip "other" points
        
        # Get part name
        part_name = get_partnet_part_name(category, pid)
        
        # Build phrase
        if count > 1:
            parts_list.append(f"{count} {part_name}")
        else:
            parts_list.append(f"a {part_name}")
    
    if not parts_list:
        return "multiple parts"
    
    # Join with Oxford comma
    if len(parts_list) == 1:
        return parts_list[0]
    elif len(parts_list) == 2:
        return f"{parts_list[0]} and {parts_list[1]}"
    else:
        return ", ".join(parts_list[:-1]) + f", and {parts_list[-1]}"


def get_partnet_part_name(category: str, local_part_id: int) -> str:
    """Get part name for PartNet."""
    from experiments.exp13_part_text_retrieval import PARTNET_PART_NAMES
    
    cat_key = next(
        (k for k in PARTNET_PART_NAMES if k.lower() == category.lower()),
        None
    )
    if cat_key is None:
        return f"part{local_part_id}"
    return PARTNET_PART_NAMES[cat_key].get(local_part_id, f"part{local_part_id}")


class PartNetTextDataset(Dataset):
    """Dataset wrapper for generating text descriptions."""
    
    def __init__(
        self,
        partnet_ds: PartNetDataset,
        categories_map: Dict[int, str]
    ):
        self.partnet_ds = partnet_ds
        self.categories_map = categories_map
        
    def __len__(self):
        return len(self.partnet_ds)
    
    def __getitem__(self, idx):
        points = self.partnet_ds.points[idx]  # [P, 3]
        label = int(self.partnet_ds.labels[idx])
        part_labels = self.partnet_ds.part_labels[idx]  # [P]
        category = self.categories_map.get(label, f"class{label}")
        
        # Count parts
        unique, counts = np.unique(part_labels, return_counts=True)
        part_counts = dict(zip(unique.astype(int), counts))
        
        return {
            'points': points.astype(np.float32),
            'label': label,
            'category': category,
            'part_counts': part_counts,
            'idx': idx
        }


def collate_fn(batch):
    """Custom collate for text generation batch."""
    return {
        'points': [b['points'] for b in batch],
        'labels': [b['label'] for b in batch],
        'categories': [b['category'] for b in batch],
        'part_counts': [b['part_counts'] for b in batch],
        'indices': [b['idx'] for b in batch]
    }


def generate_llm_descriptions(
    partnet_ds: PartNetDataset,
    model: Any,
    processor: Any,
    output_path: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    mode: str = "structured",
    use_vlm: bool = False,
    max_samples: Optional[int] = None
):
    """Generate LLM descriptions for PartNet dataset."""
    
    categories_map = getattr(partnet_ds, "categories", {})
    
    # Create dataset
    dataset = PartNetTextDataset(partnet_ds, categories_map)
    if max_samples:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    descriptions = []
    timestamps = []
    
    start_time = time.time()
    
    for i, batch in enumerate(loader):
        for j, (category, part_counts) in enumerate(
            zip(batch['categories'], batch['part_counts'])
        ):
            if mode == "structured":
                # Build structured description
                parts_desc = build_part_description(
                    list(part_counts.keys()),
                    part_counts,
                    category,
                    categories_map
                )
                
                # Choose random prompt template
                prompt_template = PARTNET_PROMPT_TEMPLATES[
                    i % len(PARTNET_PROMPT_TEMPLATES)
                ]
                prompt = prompt_template.format(
                    category=category,
                    parts_description=parts_desc
                )
                
                # Generate with LLM if available
                if use_vlm and model is not None:
                    points = batch['points'][j]
                    desc = generate_text_with_vlm(model, processor, points, prompt)
                else:
                    # Text-only: just format the prompt (LLM will fill in details)
                    desc = prompt
            
            elif mode == "text_only":
                # Simple text-only prompt
                parts_desc = build_part_description(
                    list(part_counts.keys()),
                    part_counts,
                    category,
                    categories_map
                )
                desc = f"A {category} with {parts_desc}."
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            descriptions.append({
                'idx': batch['indices'][j],
                'label': batch['labels'][j],
                'category': category,
                'text': desc,
                'prompt': prompt if 'prompt' in locals() else None
            })
            
            timestamps.append(time.time() - start_time)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(loader)}] "
                  f"Generated {len(descriptions)} descriptions "
                  f"({elapsed:.1f}s elapsed)")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'descriptions': descriptions,
        'stats': {
            'total': len(descriptions),
            'elapsed_seconds': time.time() - start_time,
            'avg_time_per_sample': (time.time() - start_time) / max(len(descriptions), 1)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved {len(descriptions)} descriptions to {output_path}")
    print(f"  Total time: {results['stats']['elapsed_seconds']:.1f}s")
    
    # Print sample
    print("\n  Sample descriptions:")
    for d in descriptions[:5]:
        print(f"    [{d['category']}] {d['text'][:100]}...")
    
    return descriptions


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM text descriptions for PartNet"
    )
    parser.add_argument(
        "--partnet_root", type=str, default="data/partnet",
        help="Path to PartNet data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/llm_descriptions",
        help="Output directory for generated descriptions"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--model", type=str, default="blip2",
        choices=["blip2", "gpt2", "llama"],
        help="Model type to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--mode", type=str, default="structured",
        choices=["structured", "text_only"],
        help="Description generation mode"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PartNet LLM Text Generation")
    print("=" * 60)
    
    # Load PartNet
    print(f"\nLoading PartNet {args.split} split...")
    partnet_ds = get_dataset(
        'partnet', split=args.split,
        root=args.partnet_root, num_points=1024
    )
    print(f"  Loaded {len(partnet_ds)} shapes")
    
    # Load LLM
    llm = LLMLoader()
    if args.model == "blip2":
        llm.load()
    
    # Generate descriptions
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"partnet_{args.split}_{args.mode}_{timestamp}.json"
    
    print(f"\nGenerating {args.mode} descriptions...")
    generate_llm_descriptions(
        partnet_ds,
        model=llm.model if hasattr(llm, 'model') else None,
        processor=llm.processor if hasattr(llm, 'processor') else None,
        output_path=output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        use_vlm=(args.model == "blip2"),
        max_samples=args.max_samples
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
