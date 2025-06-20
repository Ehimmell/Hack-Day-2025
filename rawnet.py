import os
import argparse
import copy
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler

# ---------------------- configurable pre-filter ---------------------- #

def exponential_lowpass_kernel(r: float = 0.7, taps: int = 20):
    """Return 1-D numpy kernel implementing y[n]= (1-r) Σ r^k x[n-k]."""
    coeffs = [(1 - r) * (r ** k) for k in range(taps)]
    return np.array(coeffs, dtype=np.float32)

FILTER_KERNEL = exponential_lowpass_kernel(r=0.7, taps=20)
FILTER_TORCH = torch.from_numpy(FILTER_KERNEL).view(1, 1, -1)  # [out,in,k]

# ---------------------- data ---------------------- #
class AudioDataset(Dataset):
    """Dataset that first applies a fixed filter, then stochastic augmentations."""

    def __init__(self, human_dir, ai_dir, max_duration=3.0):
        self.file_paths = []
        self.labels = []

        for f in os.listdir(human_dir):
            if f.lower().endswith('.wav'):
                self.file_paths.append(os.path.join(human_dir, f))
                self.labels.append(0)
        for f in os.listdir(ai_dir):
            if f.lower().endswith('.wav'):
                self.file_paths.append(os.path.join(ai_dir, f))
                self.labels.append(1)

        # length reference
        _, sr = sf.read(self.file_paths[0], dtype='float32')
        self.max_len = int(sr * max_duration)
        self.sample_rate = sr

    def __len__(self):
        return len(self.file_paths)

    # ---- augment helpers ---- #
    def _rand_gain(self, x):
        return x * 10 ** (random.uniform(-3, 3) / 20)

    def _rand_shift(self, x):
        return torch.roll(x, random.randint(0, 200), dims=-1)

    def _add_noise(self, x):
        return x + torch.randn_like(x) * random.uniform(1e-4, 5e-4)

    def _augment(self, x):
        if random.random() < 0.5:
            x = self._rand_gain(x)
        if random.random() < 0.5:
            x = self._rand_shift(x)
        if random.random() < 0.5:
            x = self._add_noise(x)
        return x

    def _apply_prefilter(self, wav: torch.Tensor):
        # 1D conv with fixed kernel; padding "same"
        pad = FILTER_TORCH.size(-1) - 1
        wav = nn.functional.pad(wav, (pad, 0))
        return nn.functional.conv1d(wav, FILTER_TORCH.to(wav.device))

    def __getitem__(self, idx):
        data, _ = sf.read(self.file_paths[idx], dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data).unsqueeze(0)  # [1, n]

        # --- deterministic pre-filter ---
        wav = self._apply_prefilter(wav)

        # pad/truncate
        if wav.size(1) < self.max_len:
            wav = nn.functional.pad(wav, (0, self.max_len - wav.size(1)))
        else:
            wav = wav[:, : self.max_len]

        # stochastic augment
        wav = self._augment(wav)
        return wav, self.labels[idx]

# ---------------------- model (unchanged) ---------------------- #
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x)

class RawNet(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 3, stride=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(64) for _ in range(5)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.drop(x))

# ---------------------- training ---------------------- #

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train(human_dir, ai_dir, epochs=20, batch=32, lr=3e-4, max_dur=3.0,
          device=None, workers=4, dropout=0.3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = AudioDataset(human_dir, ai_dir, max_duration=max_dur)
    loader = DataLoader(ds, batch_size=batch,
                        sampler=RandomSampler(ds, replacement=True, num_samples=len(ds)),
                        num_workers=workers, pin_memory=device != "cpu")

    model = RawNet(dropout_p=dropout).to(device)
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate(model, loader, device)
        print(f"Epoch {ep}/{epochs}  acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "rawnet_classifier.pth")

    print(f"Best training accuracy: {best:.3f}")

# ---------------------- CLI ---------------------- #
if __name__ == "__main__":
    cli = argparse.ArgumentParser("Stochastic RawNet with pre-filter")
    cli.add_argument("human_dir")
    cli.add_argument("ai_dir")
    cli.add_argument("--epochs", type=int, default=20)
    cli.add_argument("--batch", type=int, default=32)
    cli.add_argument("--lr", type=float, default=3e-4)
    cli.add_argument("--max_dur", type=float, default=3.0)
    cli.add_argument("--dropout", type=float, default=0.3)
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()

    train(args.human_dir, args.ai_dir, epochs=args.epochs, batch=args.batch,
          lr=args.lr, max_dur=args.max_dur, workers=args.workers,
          dropout=args.dropout)