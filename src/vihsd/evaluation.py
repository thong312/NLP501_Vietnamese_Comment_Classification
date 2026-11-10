from .labels import LABELS


def evaluate(y_true, y_pred):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    ids = list(range(3))
    report = classification_report(y_true, y_pred, labels=ids, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    return {"macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
            "per_class": {label: report[label] for label in LABELS},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
            "labels": LABELS, "n_samples": len(y_true)}
