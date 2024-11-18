import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Metrics:
    def accuracy(y_true, y_pred):
        """Calculate accuracy."""
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total


    def precision(y_true, y_pred, average='binary'):
        """Calculate precision."""
        return precision_score(y_true, y_pred, average=average)


    def recall(y_true, y_pred, average='binary'):
        """Calculate recall."""
        return recall_score(y_true, y_pred, average=average)


    def f1(y_true, y_pred, average='binary'):
        """Calculate F1-score."""
        return f1_score(y_true, y_pred, average=average)


    def confusion_matrix(y_true, y_pred):
        """Calculate confusion matrix."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return np.array([[tp, fp], [fn, tn]])


    def print_metrics(y_true, y_pred, average='binary'):
        """Prints key metrics: accuracy, precision, recall, F1 score."""
        acc = accuracy(y_true, y_pred)
        prec = precision(y_true, y_pred, average)
        rec = recall(y_true, y_pred, average)
        f1_sc = f1(y_true, y_pred, average)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1_sc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        return acc, prec, rec, f1_sc, cm
