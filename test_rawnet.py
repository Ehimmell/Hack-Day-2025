import argparse
import soundfile as sf
import torch
import torch.nn as nn

# Define model architecture (must match training script)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class RawNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(5)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def load_wav_sf(path):
    # Load audio with PySoundFile
    data, sr = sf.read(path, dtype='float32')
    # Convert to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)
    # To tensor of shape [1, n]
    waveform = torch.from_numpy(data).unsqueeze(0)
    return waveform, sr


def classify_wav(model, path, max_len, device):
    waveform, sample_rate = load_wav_sf(path)
    # Pad or truncate
    if waveform.size(1) < max_len:
        pad_amt = max_len - waveform.size(1)
        waveform = nn.functional.pad(waveform, (0, pad_amt))
    else:
        waveform = waveform[:, :max_len]
    waveform = waveform.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(waveform.unsqueeze(0))
        _, pred = torch.max(outputs, 1)
    return 'AI' if pred.item() == 1 else 'Human'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify a WAV file as human or AI.')
    parser.add_argument('wav_path', help='Path to the .wav file to classify')
    parser.add_argument('--model', default='rawnet_classifier.pth', help='Path to the trained model checkpoint')
    parser.add_argument('--max_duration', type=float, default=3.0,
                        help='Max duration (seconds) used during training')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Reconstruct model and load weights
    model = RawNet(num_classes=2).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)

    # Determine max_len from sample rate
    waveform, sample_rate = load_wav_sf(args.wav_path)
    max_len = int(sample_rate * args.max_duration)

    result = classify_wav(model, args.wav_path, max_len, device)
    print(f"File: {args.wav_path} -> Prediction: {result}")