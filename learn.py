import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import random
from . import utils


# ======== SKLEARN MODELS ========

def sklearn_logreg():
    return LogisticRegression(max_iter=300)

def sklearn_svm():
    return SVC(kernel="rbf", probability=True)

def sklearn_random_forest():
    return RandomForestClassifier(n_estimators=100)

def sklearn_gb():
    return GradientBoostingClassifier()

def sklearn_mlp():
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)


# ======== TORCH MODELS ========

class TorchMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TorchRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, output_size=2):
        super().__init__()
        self.rnn = nn.GRU(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])

class TorchAutoencoder(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class TorchClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim)
        )

    def forward(self, x):
        return self.model(x)

def chaos_mode():
    """Mistura modelos sklearn e PyTorch e tenta treinar ambos em dados aleatórios."""
    utils.separator()
    utils.log("⚠ Initiating CHAOS MODE ⚠")
    utils.log("🤯 Mixing frameworks... boundaries collapsing...")

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3)

    sklearn_models = [
        sklearn_logreg(), sklearn_random_forest(), sklearn_gb(),
        sklearn_svm(), sklearn_mlp()
    ]
    torch_models = [
        TorchMLP(10, 32, 3),
        TorchClassifier(10, 3),
        TorchAutoencoder(10)
    ]

    chosen_sklearn = random.choice(sklearn_models)
    chosen_torch = random.choice(torch_models)

    utils.log(f"🎲 Selected: {chosen_sklearn.__class__.__name__} + {chosen_torch.__class__.__name__}")

    try:
        chosen_sklearn.fit(X, y)
        utils.success("SKLEARN MODEL TRAINED")
    except Exception as e:
        utils.fail(f"SKLEARN ERROR: {e}")

    try:
        import torch
        x = torch.randn(64, 10)
        _ = chosen_torch(x)
        utils.success("PYTORCH MODEL SURVIVED")
    except Exception as e:
        utils.fail(f"TORCH ERROR: {e}")

    utils.separator()
    utils.log("💀 Chaos mode complete. The machine learned... something.")
