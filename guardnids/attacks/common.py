import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FLOAT_COLUMNS = ["Dur", "SrcBytes", "DstBytes", "TotPkts", "TotBytes", "BytesPerPkt", "PktsPerSec", "RatioOutIn"]


def score(detector, flows, variant, level):
    """Run the detector on (possibly attacked) flows; malicious-class metrics + benign FPR."""
    y_true, y_pred = detector.predict(flows)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {"variant": variant, "level": level,
            "recall": r[1], "precision": p[1], "f1": f1[1], "fpr": 1.0 - r[0]}


def as_float(flows, columns=None):
    """Copy with the edited statistics cast to float (some splits store them as int)."""
    flows = flows.copy()
    cols = [c for c in (columns or FLOAT_COLUMNS) if c in flows.columns]
    flows[cols] = flows[cols].astype("float64")
    return flows


def malicious_mask(flows):
    return (flows.Label == 1).to_numpy()


def replace_inf_with_max(df, col):
    """Division by zero in a derived ratio: use the largest finite value."""
    finite = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    fill = finite.max() if len(finite) else 0.0
    df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(fill)
    return df
