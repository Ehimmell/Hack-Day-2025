# backend/model.py
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

WEIGHTS_PATH = Path(__file__).with_name("weights.pth")

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ---- one-time load at import time ----
device = torch.device("cpu")               # GPU works too; keep demo simple
model = MyNet().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@torch.inference_mode()
def predict(file_obj):
    img = Image.open(file_obj).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    logits = model(tensor)
    label = int(logits.argmax())
    return {"class": label}
