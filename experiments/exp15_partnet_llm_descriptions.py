"""Generate LLM-based text descriptions for PartNet shapes.

This script generates rich, natural-language descriptions for PartNet shapes
using structured prompts that combine part metadata with LLM creativity.

Key idea: Instead of rigid part-counting ("a chair with a back, seat, 4 legs"),
we use prompts that guide the LLM to generate more descriptive, natural text:
"a sturdy wooden chair with a high back, a cushioned seat, and four sturdy legs"

Usage
-----
    # Generate LLM descriptions for PartNet train split
    python experiments/exp15_partnet_llm_descriptions.py \\
        --partnet_root data/partnet \\
        --output_dir outputs/llm_descriptions \\
        --split train \\
        --mode structured

    # Use a smaller model for faster generation
    python experiments/exp15_partnet_llm_descriptions.py \\
        --model "microsoft/phi-2" \\
        --batch_size 16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Import project modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset, PartNetDataset


# Prompt templates for LLM-based description generation
# These templates guide the LLM to generate rich, descriptive text
PROMPT_TEMPLATES = {
    "detailed": [
        "Describe a {category} that features {parts}. Make it descriptive and natural.",
        "Write a detailed description of a {category} with {parts}.",
        "Create a rich description of a {category} that includes: {parts}.",
        "Describe a {category} in detail. It has {parts}.",
        "A {category} with {parts}. Include styling and material cues.",
    ],
    "simple": [
        "A {category} with {parts}.",
        "{category} featuring {parts}.",
        "A {category} that has {parts}.",
    ],
    "natural": [
        "What does a {category} look like? It has {parts}.",
        "Describe a typical {category}. Think about its parts: {parts}.",
        "Imagine a {category}. It comes with {parts}.",
    ],
}


# Detailed part descriptions that add more context
PART_QUALIFIERS = {
    "leg": ["sturdy", "slender", "thick", "curved", "wooden", "metal"],
    "back": ["high", "low", "curved", "straight", " padded", "ornate"],
    "seat": ["cushioned", "flat", "curved", "wide", "narrow"],
    "top": ["flat", "rounded", "ornate", "simple", "marble", "wooden"],
    "handle": ["ergonomic", "curved", "straight", "metal", "plastic"],
    "wheel": ["swivel", "castor", "locked", "rubber", "metal"],
    "arm": ["rest", "padded", "adjustable", "fixed"],
    "backrest": ["high", "low", "adjustable", "lumbar"],
    "cushion": ["thick", "thin", "comfortable", "firm"],
}


def get_partnet_part_name(category: str, local_part_id: int) -> str:
    """Get part name for PartNet category."""
    from experiments.exp13_part_text_retrieval import PARTNET_PART_NAMES
    
    cat_key = next(
        (k for k in PARTNET_PART_NAMES if k.lower() == category.lower()),
        None
    )
    if cat_key is None:
        return f"part{local_part_id}"
    return PARTNET_PART_NAMES[cat_key].get(local_part_id, f"part{local_part_id}")


def build_part_description(
    part_counts: Dict[int, int],
    category: str,
    add_qualifiers: bool = True,
    use_count_phrases: bool = True
) -> str:
    """Build a rich part description string.
    
    Args:
        part_counts: Dict mapping part IDs to point counts
        category: PartNet category name
        add_qualifiers: Whether to add descriptive adjectives
        use_count_phrases: Whether to use "4 legs" vs "a leg"
    """
    parts_list = []
    
    for pid, count in part_counts.items():
        if pid == 0:
            continue  # Skip "other" points
        
        part_name = get_partnet_part_name(category, pid)
        
        # Add qualifier if available and multiple instances
        if add_qualifiers and count > 1 and part_name in PART_QUALIFIERS:
            qualifier = np.random.choice(PART_QUALIFIERS[part_name])
            part_name = f"{qualifier} {part_name}"
        
        # Build phrase with count
        if use_count_phrases and count > 1:
            if part_name.endswith('y'):
                plural = part_name[:-1] + 'ies'
            elif part_name.endswith('s') or part_name.endswith('x') or part_name.endswith('z'):
                plural = part_name + 'es'
            else:
                plural = part_name + 's'
            parts_list.append(f"{count} {plural}")
        else:
            article = "an" if part_name[0] in 'aeiou' else "a"
            parts_list.append(f"{article} {part_name}")
    
    if not parts_list:
        return "multiple structural components"
    
    # Join with Oxford comma
    if len(parts_list) == 1:
        return parts_list[0]
    elif len(parts_list) == 2:
        return f"{parts_list[0]} and {parts_list[1]}"
    else:
        return ", ".join(parts_list[:-1]) + f", and {parts_list[-1]}"


class LLMTextDataset(Dataset):
    """Dataset wrapper for LLM text generation."""
    
    def __init__(self, partnet_ds: PartNetDataset):
        self.partnet_ds = partnet_ds
        self.categories_map = getattr(partnet_ds, "categories", {})
        
    def __len__(self):
        return len(self.partnet_ds)
    
    def __getitem__(self, idx):
        label = int(self.partnet_ds.labels[idx])
        category = self.categories_map.get(label, f"class{label}")
        part_labels = self.partnet_ds.part_labels[idx]
        
        # Count parts
        unique, counts = np.unique(part_labels, return_counts=True)
        part_counts = dict(zip(unique.astype(int), counts))
        
        return {
            'idx': idx,
            'label': label,
            'category': category,
            'part_counts': part_counts
        }


def collate_fn(batch):
    return {
        'indices': [b['idx'] for b in batch],
        'labels': [b['label'] for b in batch],
        'categories': [b['category'] for b in batch],
        'part_counts': [b['part_counts'] for b in batch]
    }


def load_llm_model(model_name: str = "microsoft/phi-2", device: str = "cuda"):
    """Load an LLM for text generation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Loading LLM: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None
    )
    
    model.eval()
    
    print(f"✓ Loaded {model_name}")
    return model, tokenizer


def generate_llm_description(
    model: Any,
    tokenizer: Any,
    category: str,
    part_description: str,
    prompt_template: str,
    device: str = "cuda"
) -> str:
    """Generate a text description using LLM."""
    prompt = prompt_template.format(
        category=category,
        parts=part_description
    )
    
    # Create input
    input_text = f"Describe this 3D object:\n\n{prompt}\n\nDescription:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract just the description
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after "Description:")
    if "Description:" in full_output:
        desc = full_output.split("Description:")[-1].strip()
    else:
        desc = full_output[len(input_text):].strip()
    
    return desc


def generate_llm_descriptions(
    partnet_ds: PartNetDataset,
    model: Any = None,
    tokenizer: Any = None,
    output_path: Path = None,
    batch_size: int = 8,
    mode: str = "structured",
    max_samples: int = None,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Generate LLM descriptions for PartNet dataset."""
    
    # Get base dataset (handle Subset wrapper)
    base_ds = partnet_ds.dataset if hasattr(partnet_ds, 'dataset') else partnet_ds
    categories = getattr(base_ds, "categories", {})
    
    # Create dataset
    dataset = LLMTextDataset(partnet_ds)
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Keep 0 for GPU efficiency
        collate_fn=collate_fn
    )
    
    descriptions = []
    prompt_templates = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["detailed"])
    
    print(f"Generating {mode} descriptions...")
    start_time = datetime.now()
    
    for i, batch in enumerate(loader):
        batch_descriptions = []
        
        for j, (label, category, part_counts) in enumerate(
            zip(batch['labels'], batch['categories'], batch['part_counts'])
        ):
            # Build structured part description
            parts_str = build_part_description(part_counts, category)
            
            if model is not None and tokenizer is not None:
                # Use LLM to generate description
                prompt = np.random.choice(prompt_templates)
                desc = generate_llm_description(
                    model, tokenizer,
                    category, parts_str, prompt, device
                )
            else:
                # Fallback: structured template
                prompt = np.random.choice(prompt_templates)
                desc = prompt.format(category=category, parts=parts_str)
            
            batch_descriptions.append({
                'idx': batch['indices'][j],
                'label': batch['labels'][j],
                'category': category,
                'text': desc,
                'parts': parts_str,
                'prompt': prompt
            })
        
        descriptions.extend(batch_descriptions)
        
        if (i + 1) % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  [{i+1}/{len(loader)}] {len(descriptions)} descriptions "
                  f"({elapsed:.1f}s)")
    
    # Calculate timing stats
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Generated {len(descriptions)} descriptions in {elapsed:.1f}s")
    
    # Print samples
    print("\n  Sample descriptions:")
    for d in descriptions[:5]:
        print(f"    [{d['category']}] {d['text'][:120]}...")
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total': len(descriptions),
                'elapsed_seconds': elapsed
            },
            'descriptions': descriptions
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved to {output_path}")
    
    return descriptions


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-based text descriptions for PartNet"
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
        "--model", type=str, default=None,
        help="LLM model name (e.g., microsoft/phi-2, meta-llama/Llama-2-7b-chat-hf)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--mode", type=str, default="structured",
        choices=["detailed", "simple", "natural", "structured"],
        help="Description generation mode"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Output file name (default: llm_descriptions_{split}.json)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PartNet LLM Text Generation")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Split: {args.split}")
    print(f"  Mode: {args.mode}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Device: {args.device}")
    
    # Load PartNet
    print(f"\nLoading PartNet {args.split} split...")
    partnet_ds = get_dataset(
        'partnet', split=args.split,
        root=args.partnet_root, num_points=1024
    )
    # Handle Subset wrapper
    base_ds = partnet_ds.dataset if hasattr(partnet_ds, 'dataset') else partnet_ds
    categories = getattr(base_ds, "categories", {})
    print(f"  Loaded {len(partnet_ds)} shapes with {len(categories)} categories")
    
    # Load LLM if specified
    model, tokenizer = None, None
    if args.model:
        try:
            model, tokenizer = load_llm_model(args.model, args.device)
        except Exception as e:
            print(f"Warning: Could not load LLM: {e}")
            print("  Falling back to structured templates only")
    
    # Generate descriptions
    output_name = args.output_name or f"llm_descriptions_{args.split}_{args.mode}.json"
    output_path = Path(args.output_dir) / output_name
    
    # Handle Subset wrapper for categories
    base_ds = partnet_ds.dataset if hasattr(partnet_ds, 'dataset') else partnet_ds
    categories = getattr(base_ds, "categories", {})
    
    descriptions = generate_llm_descriptions(
        partnet_ds,
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        batch_size=args.batch_size,
        mode=args.mode,
        max_samples=args.max_samples,
        device=args.device
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
