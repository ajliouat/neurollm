"""Full benchmark suite — end-to-end pre-train → fine-tune → evaluate.

Compares NeuRoLLM (pre-trained), vanilla transformer, EEGNet, CSP+SVM
across all 9 BCI-IV 2a subjects. Outputs JSON/CSV results.
"""

from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.transformer import NeuRoLLM
from data.dataset import BCIIV2aDataset, TUHEEGDataset
from training.pretrain import MCMPretrainer, train_mcm
from training.finetune import (
    finetune_subject,
    load_pretrained_encoder,
    freeze_encoder,
    unfreeze_encoder,
)
from baselines.models import train_csp_svm, train_eegnet, train_vanilla_transformer
from evaluation.metrics import compute_metrics, aggregate_subject_metrics


# ── Default configs ───────────────────────────────────────────────────

PRETRAIN_CONFIG = {
    "n_channels": 19,
    "patch_size": 50,
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 512,
    "n_layers": 6,
    "dropout": 0.1,
    "mask_ratio": 0.3,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 3,
    "n_epochs": 10,
    "batch_size": 32,
}

FINETUNE_CONFIG = {
    "n_channels": 22,
    "patch_size": 50,
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 512,
    "n_layers": 6,
    "n_classes": 4,
    "n_epochs": 30,
    "lr": 5e-5,
    "batch_size": 32,
    "patience": 10,
    "freeze_epochs": 5,
}


# ── Benchmark runner ──────────────────────────────────────────────────

def run_pretrain_step(
    config: Optional[Dict] = None,
    data_dir: str = "data/raw/tuh",
    checkpoint_dir: str = "checkpoints/benchmark",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Pre-train the model and save checkpoint.

    Returns pre-training results dict.
    """
    cfg = {**PRETRAIN_CONFIG, **(config or {})}
    device_obj = torch.device(device)

    dataset = TUHEEGDataset(data_dir)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
    )

    model = NeuRoLLM(
        n_channels=cfg["n_channels"],
        patch_size=cfg["patch_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    pretrainer = MCMPretrainer(model, mask_ratio=cfg["mask_ratio"])

    t0 = time.time()
    result = train_mcm(
        pretrainer=pretrainer,
        dataloader=loader,
        n_epochs=cfg["n_epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        checkpoint_dir=checkpoint_dir,
        device=device_obj,
    )
    elapsed = time.time() - t0

    return {
        "model_params": model.num_parameters,
        "loss_history": result["loss_history"],
        "best_loss": result["best_loss"],
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint_dir": checkpoint_dir,
    }


def evaluate_neurollm(
    subjects: List[int],
    data_dir: str = "data/raw/bci_iv_2a",
    pretrain_ckpt: Optional[str] = None,
    config: Optional[Dict] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Fine-tune and evaluate NeuRoLLM on given subjects."""
    cfg = {**FINETUNE_CONFIG, **(config or {})}
    device_obj = torch.device(device)

    subject_metrics = []
    subject_results = []

    for subj in subjects:
        model = NeuRoLLM(
            n_channels=cfg["n_channels"],
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            n_layers=cfg["n_layers"],
            n_classes=cfg["n_classes"],
            dropout=cfg.get("dropout", 0.1),
        )

        if pretrain_ckpt is not None:
            model = load_pretrained_encoder(model, pretrain_ckpt, device_obj)

        train_ds = BCIIV2aDataset(data_dir, subject=subj, session="train")
        test_ds = BCIIV2aDataset(data_dir, subject=subj, session="test")

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["batch_size"], shuffle=False,
        )

        ft_result = finetune_subject(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=cfg["n_epochs"],
            lr=cfg["lr"],
            patience=cfg["patience"],
            freeze_epochs=cfg["freeze_epochs"],
            device=device_obj,
        )

        # Get test predictions for metrics
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device_obj)
                preds = model(bx, mode="finetune").argmax(1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(by.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        metrics = compute_metrics(y_true, y_pred)

        subject_metrics.append(metrics)
        subject_results.append({
            "subject": subj,
            **ft_result,
            **metrics,
        })

    agg = aggregate_subject_metrics(subject_metrics)
    return {
        "model_name": "NeuRoLLM",
        "subject_results": subject_results,
        **agg,
    }


def evaluate_baselines(
    subjects: List[int],
    data_dir: str = "data/raw/bci_iv_2a",
    config: Optional[Dict] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Evaluate CSP+SVM, EEGNet, and vanilla transformer baselines."""
    cfg = {**FINETUNE_CONFIG, **(config or {})}
    device_obj = torch.device(device)

    results = {"CSP+SVM": [], "EEGNet": [], "VanillaTransformer": []}

    for subj in subjects:
        train_ds = BCIIV2aDataset(data_dir, subject=subj, session="train")
        test_ds = BCIIV2aDataset(data_dir, subject=subj, session="test")

        X_train = np.stack([train_ds[i][0].numpy() for i in range(len(train_ds))])
        y_train = np.array([train_ds[i][1].item() for i in range(len(train_ds))])
        X_test = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))])
        y_test = np.array([test_ds[i][1].item() for i in range(len(test_ds))])

        # CSP+SVM
        csp_res = train_csp_svm(X_train, y_train, X_test, y_test)
        csp_metrics = compute_metrics(y_test, csp_res["predictions"])
        results["CSP+SVM"].append({
            "subject": subj, **csp_metrics,
        })

        # EEGNet
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["batch_size"], shuffle=False,
        )

        eegnet_res = train_eegnet(
            train_loader, test_loader,
            n_channels=cfg["n_channels"],
            n_samples=875,
            n_classes=cfg["n_classes"],
            n_epochs=min(cfg["n_epochs"], 30),
            patience=cfg["patience"],
            device=device_obj,
        )
        results["EEGNet"].append({
            "subject": subj,
            "accuracy": eegnet_res["accuracy"],
        })

        # Vanilla Transformer
        vt_res = train_vanilla_transformer(
            train_loader, test_loader,
            n_channels=cfg["n_channels"],
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            n_layers=cfg["n_layers"],
            n_classes=cfg["n_classes"],
            n_epochs=min(cfg["n_epochs"], 30),
            patience=cfg["patience"],
            device=device_obj,
        )
        results["VanillaTransformer"].append({
            "subject": subj,
            "accuracy": vt_res["accuracy"],
        })

    # Aggregate each baseline
    aggregated = {}
    for name, subject_list in results.items():
        accs = [s["accuracy"] for s in subject_list]
        aggregated[name] = {
            "model_name": name,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "per_subject": subject_list,
        }

    return aggregated


def run_full_benchmark(
    subjects: Optional[List[int]] = None,
    pretrain_epochs: int = 10,
    finetune_epochs: int = 30,
    data_dir_tuh: str = "data/raw/tuh",
    data_dir_bci: str = "data/raw/bci_iv_2a",
    checkpoint_dir: str = "checkpoints/benchmark",
    output_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run the complete benchmark pipeline.

    1. Pre-train on TUH proxy
    2. Fine-tune NeuRoLLM on each subject
    3. Train baselines
    4. Compare and save results
    """
    if subjects is None:
        subjects = list(range(1, 10))  # 9 subjects

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NeuRoLLM Full Benchmark Suite")
    print("=" * 60)

    # Step 1: Pre-train
    print("\n[1/3] Pre-training on TUH proxy data...")
    pt_config = {"n_epochs": pretrain_epochs}
    pretrain_result = run_pretrain_step(
        config=pt_config,
        data_dir=data_dir_tuh,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    print(f"  Best loss: {pretrain_result['best_loss']:.4f}")

    # Checkpoint path
    ckpt_path = str(Path(checkpoint_dir) / "best_mcm.pt")

    # Step 2: NeuRoLLM evaluation
    print("\n[2/3] Fine-tuning NeuRoLLM on BCI-IV 2a...")
    ft_config = {"n_epochs": finetune_epochs}
    neurollm_result = evaluate_neurollm(
        subjects=subjects,
        data_dir=data_dir_bci,
        pretrain_ckpt=ckpt_path,
        config=ft_config,
        device=device,
    )
    print(f"  Mean accuracy: {neurollm_result['mean_accuracy']:.3f} "
          f"± {neurollm_result['std_accuracy']:.3f}")

    # Step 3: Baselines
    print("\n[3/3] Training baselines...")
    baseline_results = evaluate_baselines(
        subjects=subjects,
        data_dir=data_dir_bci,
        config=ft_config,
        device=device,
    )
    for name, res in baseline_results.items():
        print(f"  {name}: {res['mean_accuracy']:.3f} ± {res['std_accuracy']:.3f}")

    # Compile results
    all_results = {
        "pretrain": pretrain_result,
        "neurollm": neurollm_result,
        "baselines": baseline_results,
        "subjects": subjects,
    }

    # Save JSON
    json_path = output_path / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save CSV summary
    csv_path = output_path / "benchmark_summary.csv"
    _write_summary_csv(csv_path, neurollm_result, baseline_results, subjects)

    print(f"\nResults saved to {output_path}")
    return all_results


def _write_summary_csv(
    csv_path: Path,
    neurollm_result: Dict,
    baseline_results: Dict[str, Dict],
    subjects: List[int],
) -> None:
    """Write a CSV summary of per-subject accuracies."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Subject", "NeuRoLLM"]
        for name in baseline_results:
            header.append(name)
        writer.writerow(header)

        for i, subj in enumerate(subjects):
            row = [f"S{subj:02d}"]
            # NeuRoLLM
            if i < len(neurollm_result.get("subject_results", [])):
                row.append(f"{neurollm_result['subject_results'][i]['accuracy']:.4f}")
            else:
                row.append("")
            # Baselines
            for name, res in baseline_results.items():
                if i < len(res.get("per_subject", [])):
                    row.append(f"{res['per_subject'][i]['accuracy']:.4f}")
                else:
                    row.append("")
            writer.writerow(row)

        # Mean row
        mean_row = ["Mean", f"{neurollm_result['mean_accuracy']:.4f}"]
        for name, res in baseline_results.items():
            mean_row.append(f"{res['mean_accuracy']:.4f}")
        writer.writerow(mean_row)
