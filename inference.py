import numpy as np
import torch
from sklearn.metrics import accuracy_score
from model import SimpleCNN


def run_inference(npz_path, model_path="../models/skin_model_noise_robust.pth"):
    data = np.load(npz_path)
    keys = data.files

    x = data[[k for k in keys if 'x' in k.lower()][0]]
    y = data[[k for k in keys if 'y' in k.lower()][0]].squeeze()

    x = x.astype("float32") / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.tensor(x, dtype=torch.float32)

    model = SimpleCNN(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(x).argmax(dim=1).numpy()

    acc = accuracy_score(y, preds)
    print("Accuracy:", acc)
    return acc
