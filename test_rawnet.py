import soundfile as sf
import torch
import torch.nn as nn
from pathlib import Path

# ---------- model definition (same as training) ---------- #
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)

class RawNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 3, stride=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(64) for _ in range(5)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ---------- helpers ---------- #

def _load_wav(path: str | Path):
    """Load wav via soundfile -> torch tensor [1, N], sample_rate."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data).unsqueeze(0), sr


def classify_file(wav_path: str | Path,
                  model_path: str | Path = "rawnet_classifier.pth",
                  max_duration: float = 3.0,
                  device: str | None = None) -> str:
    """Return "AI" or "Human" prediction for a single WAV file.

    Parameters
    ----------
    wav_path : path to .wav to classify
    model_path : path to model checkpoint (.pth saved with state_dict)
    max_duration : seconds. audio longer than this is truncated; shorter is zero-padded
    device : 'cpu', 'cuda', etc.  If None, auto-selects available CUDA.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # build / load model
    model = RawNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # prepare audio
    wav, sr = _load_wav(wav_path)
    max_len = int(sr * max_duration)
    if wav.size(1) < max_len:
        wav = nn.functional.pad(wav, (0, max_len - wav.size(1)))
    else:
        wav = wav[:, :max_len]

    with torch.no_grad():
        logits = model(wav.to(device).unsqueeze(0))
        pred = logits.argmax(1).item()
    return "AI" if pred == 1 else "Human"

# ---------- CLI wrapper (optional) ---------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Classify a WAV as human or AI")
    p.add_argument("wav_path")
    p.add_argument("--model", default="rawnet_classifier.pth")
    p.add_argument("--max_dur", type=float, default=3.0)
    args = p.parse_args()

    result = classify_file(args.wav_path, args.model, args.max_dur)
    print(f"{args.wav_path} -> {result}")