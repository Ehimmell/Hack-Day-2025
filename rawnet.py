import os
import argparse
import copy

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, human_dir, ai_dir, transform=None, max_duration=3.0):
        self.file_paths = []
        self.labels = []
        self.transform = transform
        self.max_len = None

        # collect human (0) and AI (1) files
        for f in os.listdir(human_dir):
            if f.lower().endswith('.wav'):
                self.file_paths.append(os.path.join(human_dir, f))
                self.labels.append(0)
        for f in os.listdir(ai_dir):
            if f.lower().endswith('.wav'):
                self.file_paths.append(os.path.join(ai_dir, f))
                self.labels.append(1)

        # determine sample_rate and max length in samples
        data, sr = sf.read(self.file_paths[0], dtype='float32')
        self.max_len = int(sr * max_duration)
        self.sample_rate = sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform = self.load_waveform(path)
        label = self.labels[idx]
        return waveform, label

    def load_waveform(self, path):
        # load with soundfile
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        # to torch tensor [1, n]
        waveform = torch.from_numpy(data).unsqueeze(0)
        # pad or truncate
        if waveform.size(1) < self.max_len:
            pad_amt = self.max_len - waveform.size(1)
            waveform = nn.functional.pad(waveform, (0, pad_amt))
        else:
            waveform = waveform[:, :self.max_len]
        # optional transform
        if self.transform:
            waveform = self.transform(waveform)
        return waveform

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


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            outputs = model(waveforms)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(human_dir, ai_dir, epochs=20, batch_size=16, lr=1e-4,
          max_duration=3.0, device=None, num_workers=4):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AudioDataset(human_dir, ai_dir, max_duration=max_duration)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device!='cpu')
    )

    model = RawNet(num_classes=2).to(device)
    if device.startswith('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 2
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * waveforms.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping early.")
                break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'rawnet_classifier.pth')
    return model, loader, dataset.sample_rate, max_duration, device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train RawNet to classify human vs AI voices using soundfile loader.'
    )
    parser.add_argument('human_dir', help='Directory of human .wav files')
    parser.add_argument('ai_dir', help='Directory of AI .wav files')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_duration', type=float, default=3.0,
                        help='Maximum audio length in seconds')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes')
    args = parser.parse_args()

    model, loader, sample_rate, max_duration, device = train(
        args.human_dir, args.ai_dir, args.epochs,
        args.batch_size, args.lr, args.max_duration,
        device=None, num_workers=args.num_workers
    )
    accuracy = evaluate(model, loader, device)
    print(f"Training accuracy: {accuracy:.4f}")