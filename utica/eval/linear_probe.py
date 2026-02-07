from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utica.model.architecture import Mantis8M


# -----------------------------
# UCR TSV reader
# -----------------------------
def read_ucr_tsv(tsv_path: Path):
    """
    UCR .tsv format: first column is label, rest are time points.
    Returns:
      X: (N, T) float32
      y: (N,) int64 (NOT remapped yet)
    """
    data = np.loadtxt(tsv_path, delimiter="\t")
    y = data[:, 0].astype(np.int64)
    X = data[:, 1:].astype(np.float32)
    return X, y


def remap_labels(y_train, y_test):
    """
    UCR labels can be {-1,1} or {1..K}. Remap to {0..K-1} based on TRAIN labels.
    """
    uniq = np.unique(y_train)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y_train_m = np.array([mapping[l] for l in y_train], dtype=np.int64)
    y_test_m = np.array([mapping[l] for l in y_test], dtype=np.int64)
    return y_train_m, y_test_m, len(uniq)


# -----------------------------
# Safe preprocessing: fill NaNs + resample to 512
# -----------------------------
def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    """
    Fill NaN/Inf in a single 1D series using linear interpolation across valid points.
    If all points are invalid -> return zeros.
    """
    x = x.astype(np.float32, copy=False)
    n = x.shape[0]
    mask = np.isfinite(x)
    if mask.all():
        return x
    if not mask.any():
        return np.zeros_like(x, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    out = x.copy()
    out[~mask] = np.interp(idx[~mask], idx[mask], x[mask]).astype(np.float32)
    return out


def resample_1d_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Linear interpolation resampling.
    x: (T,)
    returns: (target_len,)
    """
    T = x.shape[0]
    if T == target_len:
        return x.astype(np.float32, copy=False)
    old_idx = np.linspace(0.0, 1.0, T, dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def resample_to_fixed_len_safe(X: np.ndarray, target_len: int = 512) -> np.ndarray:
    """
    X: (N, T) -> (N, target_len)
    - fill NaN/Inf per series
    - resample to target_len
    - final nan_to_num safety
    """
    N, T = X.shape
    Xr = np.empty((N, target_len), dtype=np.float32)
    for i in range(N):
        xi = fill_nan_1d(X[i])
        xi = resample_1d_linear(xi, target_len)
        Xr[i] = xi
    return np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def check_array_stats(name: str, X: np.ndarray):
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    print(
        f"[{name}] shape={X.shape} nan={n_nan} inf={n_inf} "
        f"min={np.nanmin(X):.4g} max={np.nanmax(X):.4g}"
    )


# -----------------------------
# Dataset (expects already fixed length)
# -----------------------------
class UCRDataset(Dataset):
    def __init__(self, X_fixed: np.ndarray, y: np.ndarray):
        assert X_fixed.ndim == 2
        self.X = X_fixed.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(
            self.y[idx], dtype=torch.long
        )


# -----------------------------
# Load backbone from checkpoint
# -----------------------------
def load_backbone_from_ckpt(backbone: nn.Module, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    backbone_sd = ckpt["teacher"]

    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"[load] missing keys (up to 20): {missing[:20]}")
    if unexpected:
        print(f"[load] unexpected keys (up to 20): {unexpected[:20]}")
    print("[load] backbone loaded.")


# -----------------------------
# Linear probe training
# -----------------------------
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


@torch.no_grad()
def extract_features(backbone: nn.Module, x: torch.Tensor):
    """
    x: (B, T)
    returns: (B, D) CLS embeddings
    """
    backbone.eval()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # extra safety
    z = backbone(x, is_training=False)
    return z


def run_linear_probe_one_dataset(
    dataset_dir: Path,
    backbone: nn.Module,
    seq_len: int = 512,
    batch_size: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    device: str = "cuda",
    num_workers: int = 2,
):
    name = dataset_dir.name
    train_path = dataset_dir / f"{name}_TRAIN.tsv"
    test_path = dataset_dir / f"{name}_TEST.tsv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing TRAIN/TEST in {dataset_dir}")

    # load
    Xtr_raw, ytr = read_ucr_tsv(train_path)
    Xte_raw, yte = read_ucr_tsv(test_path)
    ytr, yte, n_classes = remap_labels(ytr, yte)

    # preprocess: fill NaNs then resample to seq_len
    Xtr = resample_to_fixed_len_safe(Xtr_raw, target_len=seq_len)
    Xte = resample_to_fixed_len_safe(Xte_raw, target_len=seq_len)

    # debug stats (helps catch NaN datasets)
    # comment out if too verbose
    check_array_stats(f"{name}/train", Xtr)
    check_array_stats(f"{name}/test", Xte)

    ds_tr = UCRDataset(Xtr, ytr)
    ds_te = UCRDataset(Xte, yte)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Determine embed dim (Mantis8M typically exposes hidden_dim)
    embed_dim = getattr(backbone, "hidden_dim", None)
    if embed_dim is None:
        xb, _ = next(iter(dl_tr))
        xb = xb.to(device)
        z = extract_features(backbone, xb)
        embed_dim = int(z.shape[-1])

    head = LinearProbe(embed_dim, n_classes).to(device)

    # Freeze backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        # train
        head.train()
        total_loss = 0.0
        total = 0

        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.no_grad():
                z = extract_features(backbone, xb)

            # if any NaNs still slip through, skip the batch (should not happen with safe preprocessing)
            if torch.isnan(z).any() or torch.isinf(z).any():
                print(
                    f"[WARN] {name}: NaN/Inf in features at epoch {ep}, skipping batch"
                )
                continue

            logits = head(z)
            loss = ce(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

        # eval
        head.eval()
        correct = 0
        total_te = 0
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                z = extract_features(backbone, xb)
                pred = head(z).argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total_te += xb.size(0)

        acc = correct / max(1, total_te)
        best_acc = max(best_acc, acc)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            tl = total_loss / max(1, total)
            print(
                f"{name} | epoch {ep:03d}/{epochs} | train_loss {tl:.4f} | test_acc {acc:.4f} | best {best_acc:.4f}"
            )

    return best_acc


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ucr_root", type=str, default="/Data/yessin.moakher/UCRArchive_2018"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/Data/yessin.moakher/mantis_ssl/9999.pth",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="If set, run only this dataset (e.g., Crop)",
    )
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # IMPORTANT: your checkpoint must match these arch params.
    # If your utica.model.architecture.Mantis8M() already sets them correctly, keep as-is.
    backbone = Mantis8M()
    load_backbone_from_ckpt(backbone, Path(args.ckpt))
    backbone.to(args.device)

    ucr_root = Path(args.ucr_root)

    if args.dataset is not None:
        acc = run_linear_probe_one_dataset(
            ucr_root / args.dataset,
            backbone=backbone,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            num_workers=args.num_workers,
        )
        print(f"\nFINAL | {args.dataset} | best_test_acc={acc:.4f}")
        return

    # run all datasets
    results = []
    for d in sorted([p for p in ucr_root.iterdir() if p.is_dir()]):
        train_tsv = d / f"{d.name}_TRAIN.tsv"
        test_tsv = d / f"{d.name}_TEST.tsv"
        if not train_tsv.exists() or not test_tsv.exists():
            continue

        try:
            acc = run_linear_probe_one_dataset(
                d,
                backbone=backbone,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device,
                num_workers=args.num_workers,
            )
            results.append((d.name, acc))
        except Exception as e:
            print(f"[WARN] {d.name} failed: {e}")
            results.append((d.name, float("nan")))

    # summary
    results_sorted = sorted(
        results, key=lambda x: (-(x[1] if np.isfinite(x[1]) else -1e9), x[0])
    )
    valid = [a for _, a in results_sorted if np.isfinite(a)]
    mean_acc = float(np.mean(valid)) if valid else float("nan")

    print("\n====================")
    print(f"Mean acc over {len(valid)} datasets: {mean_acc:.4f}")
    print("Top 10:")
    for n, a in results_sorted[:10]:
        print(f"  {n:30s} {a:.4f}")
    print("====================\n")


if __name__ == "__main__":
    main()
