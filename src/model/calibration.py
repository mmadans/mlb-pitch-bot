"""
Isotonic calibration wrapper for a pre-fitted multi-class classifier.

Fits one IsotonicRegression per class on held-out calibration data,
WITHOUT sample weights — so it learns true class frequencies rather than
the reweighted distribution used to train the base model.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibratedClassifier:
    """
    Wraps a fitted base classifier and applies per-class isotonic calibration.
    Attributes mirror sklearn conventions so downstream code (joblib, inference)
    works without changes.
    """

    def __init__(self, base_estimator, calibrators, classes_):
        self.base_estimator = base_estimator
        self.calibrators = calibrators  # list[IsotonicRegression], one per class
        self.classes_ = np.asarray(classes_)
        self.estimator = base_estimator  # for feature_importances_ access

    def predict_proba(self, X):
        raw = self.base_estimator.predict_proba(X)
        cal = np.column_stack([c.predict(raw[:, i]) for i, c in enumerate(self.calibrators)])
        cal = np.clip(cal, 0, 1)
        sums = cal.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        return cal / sums

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @classmethod
    def fit(cls, base_estimator, X_cal, y_cal):
        """
        Fit calibrators on held-out data.  base_estimator must already be fitted.
        No sample weights — calibration learns true class frequencies.
        """
        n_classes = len(base_estimator.classes_)
        raw = base_estimator.predict_proba(X_cal)
        calibrators = []
        for i in range(n_classes):
            y_bin = (y_cal == i).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(raw[:, i], y_bin)
            calibrators.append(ir)
        return cls(base_estimator, calibrators, base_estimator.classes_)
