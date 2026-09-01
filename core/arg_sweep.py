import csv
import json
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Optional, Union

import torch

from core import _ALL_DATASETS, outputs_dir
from core.trainer import Trainer
from core.utils.general import all_model_classes, safe_batch_size
from models.risa import (
    RISA_ABLATIION_BASE_ARGS,
    RISA_ABLATIION_OPTIONS,
    make_risa_sweep_params,
)


def run_arg_sweep(
        model_name:     str               = "risa",
        datasets:       Union[str, List]  = "shapenet",
        tasks:          Union[str, List]  = "all",
        base_args:      dict              = {},
        options:        list              = [],
        data_root:      str               = "data",
        device:         Union[str, torch.device, None] = None,
        epochs:         int               = 100,
        batch_size:     int               = 32,
        num_points:     int               = 128,
        save_directory: Optional[Path]    = None,
        verbose:        bool              = True,
        seed:           Optional[int]     = None,
        eval_rotations: int               = 32,
        num_workers:    int               = 4,
    ) -> dict:
    """Run a grid of experiments over all combinations of models × datasets × tasks."""
    from core.datasets import get_dataloader, get_dataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    def _resolve(arg, all_vals):
        if arg == "all":
            return list(all_vals)
        return [arg] if isinstance(arg, str) else list(arg)

    dataset_names = _resolve(datasets, _ALL_DATASETS)
    task_names    = _resolve(tasks,    Trainer.TASKS)

    for t in task_names:
        if t not in Trainer.TASKS:
            raise ValueError(f"Unknown task '{t}'. Available: {list(Trainer.TASKS)}")

    if save_directory is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = outputs_dir / "experiments" / ts
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    trainer      = Trainer()
    all_results  : dict = {}
    summary_rows : list = []

    combination_tags = [x[0] for x in options]
    combinations = list(product(*[x[1] for x in options]))

    def _make_run_tag(tags : list, options : list) -> str:
        return "_".join([f"{k}={v}" for k, v in zip(tags, options)])

    total_runs = len(combinations) * len(dataset_names) * len(task_names)
    run_idx    = 0

    model_cls = all_model_classes()[model_name]
    all_results = {}

    safe_bs = safe_batch_size(model_cls, num_points, batch_size, device)
    if safe_bs != batch_size:
        print(f"[{model_name}] Reduced batch size {batch_size} -> {safe_bs} to fit GPU memory")

    for combination in combinations:

        run_tag = _make_run_tag(combination_tags, combination)

        all_results[run_tag] = {}

        encoder_args = {}
        encoder_args.update(deepcopy(base_args))

        for dataset_name in dataset_names:
            all_results[run_tag][dataset_name] = {}

            try:
                train_ds = get_dataset(dataset_name, split="train", root=data_root, num_points=num_points)
                test_ds  = get_dataset(dataset_name, split="test",  root=data_root, num_points=num_points)
            except Exception as exc:
                print(f"[SKIP] Dataset '{dataset_name}' unavailable: {exc}")
                continue

            train_loader = get_dataloader(train_ds, batch_size=safe_bs, shuffle=True,  num_workers=num_workers)
            test_loader  = get_dataloader(test_ds,  batch_size=safe_bs, shuffle=False, num_workers=num_workers)

            for task in task_names:
                run_idx += 1
                run_dir  = save_directory / run_tag / dataset_name / task
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{'='*60}\n  Run {run_idx}/{total_runs}: {model_name} | {run_tag} | {dataset_name} | {task}\n{'='*60}")

                torch.cuda.empty_cache()
                encoder = model_cls(**encoder_args).to(device)

                try:
                    result = trainer(
                        encoder        = encoder,
                        train_loader   = train_loader,
                        test_loader    = test_loader,
                        task           = task,
                        dataset_name   = dataset_name,
                        device         = device,
                        epochs         = epochs,
                        save_directory = run_dir,
                        verbose        = verbose,
                        seed           = seed,
                        eval_rotations = eval_rotations,
                    )
                except Exception as exc:
                    print(f"[ERROR] {run_tag}/{dataset_name}/{task}: {exc}")
                    import traceback; traceback.print_exc()
                    result = {"error": str(exc)}

                all_results[run_tag][dataset_name][task] = result
                best = result.get("best", {})
                summary_rows.append({"model": model_name, "run_tag": run_tag, "dataset": dataset_name, "task": task, **best})

    master_fp = save_directory / "all_results.json"
    with open(master_fp, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {master_fp}")

    if summary_rows:
        all_keys = list(dict.fromkeys(k for row in summary_rows for k in row))
        csv_fp   = save_directory / "summary.csv"
        with open(csv_fp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV saved to {csv_fp}")

    return all_results

def run_risa_ablations(
        save_directory: Optional[Path] = None,
        base_args: dict = RISA_ABLATIION_BASE_ARGS,
        options: list   = RISA_ABLATIION_OPTIONS,
        **kwargs
    ):

    if save_directory is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = outputs_dir / "ablations" / ts

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    print(f"Running RISA ablations with base args:\n{json.dumps(base_args, indent=2)}\nand options:\n{json.dumps(options, indent=2)}")

    ablation_config = {
        "base_args": base_args,
        "options": options,
    }

    with open(save_directory / "config.json", "w") as f:
        json.dump(ablation_config, f, indent=2)
    print(f"Ablation config saved to {save_directory / 'config.json'}")

    run_arg_sweep(
        model_name = "risa",
        base_args = base_args,
        options = options,
        save_directory = save_directory,
        **kwargs,
    )

def run_risa_hyperparameter_sweep(
        save_directory: Optional[Path] = None,
        base_args: dict = None,
        options: list = None,

        tasks: list = ["reconstruction"],
        datasets: list = ["shapenet"],

        epochs: int = 30,

        **kwargs
    ):

    if base_args is None or options is None:
        _base_args, _options = make_risa_sweep_params()

    base_args = base_args or _base_args
    options   = options or _options

    if save_directory is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = outputs_dir / "hyperparameter_search" / ts

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    print(f"Running RISA hyperparameter search with base args:\n{json.dumps(base_args, indent=2)}\nand options:\n{json.dumps(options, indent=2)}")

    hyperparameter_search_config = {
        "base_args": base_args,
        "options": options,
    }

    with open(save_directory / "config.json", "w") as f:
        json.dump(hyperparameter_search_config, f, indent=2)
    print(f"Hyperparameter search config saved to {save_directory / 'config.json'}")

    run_arg_sweep(
        model_name = "risa",
        base_args = base_args,
        options = options,

        save_directory = save_directory,

        tasks = tasks,
        datasets = datasets,

        epochs = epochs,

        **kwargs,
    )