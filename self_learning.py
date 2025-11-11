import numpy as np
import time
import torch
from datasets import load_dataset
from . import learn
from .utils import log

def self_learn():
    log("🌌 Entering autonomous learning mode...")
    time.sleep(0.5)

    framework = np.random.choice(["sklearn", "torch"])
    log(f"🎲 Framework chosen by fate: {framework}")

    datasets = ["iris","ag_news", "emotion", "mnist"]
    dataset_choice = np.random.choice(datasets)
    log(f"📦 Dataset discovered in the void: {dataset_choice}")

    if dataset_choice in ["ag_news", "emotion"]:
        ds = load_dataset(dataset_choice, split="train[:1%]")
        X = torch.randn(len(ds), 8)
        y = torch.tensor(ds["label"])
        in_dim, out_dim = 8, len(set(ds["label"]))
    else:
        from sklearn import datasets as skdata
        loader = getattr(skdata, f"load_{dataset_choice}", None)
        if loader:
            d = loader()
            X, y = d.data, d.target
            in_dim, out_dim = X.shape[1], len(set(y))
        else:
            log("❌ Dataset not supported. Machine got confused.")
            return

    if framework == "sklearn":
        models = [
            learn.sklearn_logreg,
            learn.sklearn_svm,
            learn.sklearn_random_forest,
            learn.sklearn_gb,
            learn.sklearn_mlp,
        ]
        model_fn = np.random.choice(models)
        model = model_fn()
        log(f"🧩 Model chosen: {model_fn.__name__}")
        model.fit(X, y)
        import sklearn.metrics as metrics
        y_pred = model.predict(X)
        acc = metrics.accuracy_score(y, y_pred)
        log(f"📊 Training complete. Accuracy: {acc*100:.2f}%")
        if acc < 0.5:
            log("⚠ Accuracy below acceptable threshold. Retraining...")
            model.fit(X, y)
            y_pred = model.predict(X)
            acc = metrics.accuracy_score(y, y_pred)
            log(f"📊 Retraining complete. New Accuracy: {acc*100:.2f}%")
            import pickle as pkl
            with open("self_learned_model.pkl", "wb") as f:
                pkl.dump(model, f)
            log("✅ Learning complete after retraining.")
        else:
            import pickle as pkl
            with open("self_learned_model.pkl", "wb") as f:
                pkl.dump(model, f)
            log("✅ Learning complete. Accuracy statistically acceptable.")

    else:
        nets = [
            learn.TorchMLP(in_dim, 16, out_dim),
            learn.TorchClassifier(in_dim, out_dim),
            learn.TorchAutoencoder(input_dim=in_dim),
        ]
        model = np.random.choice(nets)
        log(f"🧠 Network chosen: {type(model).__name__}")
        log("🚀 Training... probably...")
        for epoch in range(3):
            time.sleep(0.4)
            log(f"Epoch {epoch+1}: loss={np.random.random():.4f}")
        import sklearn.metrics as metrics
        y_pred = torch.randint(0, out_dim, (len(y),))
        acc = metrics.accuracy_score(y, y_pred.numpy())
        log(f"📊 Training complete. Accuracy: {acc*100:.2f}%")
        if acc < 0.5:
            log("⚠ Accuracy below acceptable threshold. Retraining...")
            for epoch in range(3):
                time.sleep(0.4)
                log(f"Epoch {epoch+1}: loss={np.random.random():.4f}")
            y_pred = torch.randint(0, out_dim, (len(y),))
            acc = metrics.accuracy_score(y, y_pred.numpy())
            log(f"📊 Retraining complete. New Accuracy: {acc*100:.2f}%")
        else:
            torch.save(model.state_dict(), "self_learned_model.pth")
            log("✅ Torch model achieved semi-sentience.")