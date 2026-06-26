import numpy as np


def accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def confusion_matrix(preds, labels, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        matrix[int(true), int(pred)] += 1
    return matrix
