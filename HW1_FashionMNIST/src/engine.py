import json
from pathlib import Path

import numpy as np

from .data import iterate_minibatches
from .layers import CrossEntropyLoss
from .optim import SGD, StepDecay


def evaluate(model, dataset, batch_size=512):
    x, y = dataset
    loss_fn = CrossEntropyLoss()
    total_loss = total_correct = total = 0
    logits_all = []
    for start in range(0, x.shape[0], batch_size):
        bx, by = x[start:start + batch_size], y[start:start + batch_size]
        logits = model.forward(bx)
        loss = loss_fn.forward(logits, by)
        preds = np.argmax(logits, axis=1)
        total_loss += loss * bx.shape[0]
        total_correct += int(np.sum(preds == by))
        total += bx.shape[0]
        logits_all.append(logits)
    return {"loss": float(total_loss / total), "accuracy": float(total_correct / total), "logits": np.concatenate(logits_all, axis=0)}


def train_model(model, train_set, valid_set, checkpoint_path, epochs=10, batch_size=128, lr=0.1, weight_decay=0.0, lr_decay_step=5, lr_decay_gamma=0.5, seed=42):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = SGD(model, lr=lr, weight_decay=weight_decay)
    scheduler = StepDecay(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    loss_fn = CrossEntropyLoss()
    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "valid_loss": [], "valid_accuracy": [], "lr": []}
    best_valid = -1.0
    for epoch in range(1, epochs + 1):
        losses, correct, seen = [], 0, 0
        for bx, by in iterate_minibatches(train_set[0], train_set[1], batch_size, shuffle=True, seed=seed + epoch):
            logits = model.forward(bx)
            loss = loss_fn.forward(logits, by)
            model.backward(loss_fn.backward())
            optimizer.step()
            losses.append(loss)
            correct += int(np.sum(np.argmax(logits, axis=1) == by))
            seen += bx.shape[0]
        valid = evaluate(model, valid_set, batch_size=batch_size)
        train_loss = float(np.mean(losses))
        train_acc = float(correct / seen)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["valid_loss"].append(valid["loss"])
        history["valid_accuracy"].append(valid["accuracy"])
        history["lr"].append(float(optimizer.lr))
        print(f"epoch {epoch:02d}: train loss {train_loss:.4f}, train acc {train_acc:.4f}, valid loss {valid['loss']:.4f}, valid acc {valid['accuracy']:.4f}, lr {optimizer.lr:.5f}")
        if valid["accuracy"] > best_valid:
            best_valid = valid["accuracy"]
            model.save(checkpoint_path)
        scheduler.step(epoch)
    return history


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
