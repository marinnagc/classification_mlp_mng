# mlp_biblio.py
import numpy as np
from sklearn.neural_network import MLPClassifier

class SklearnMLPModel:
    """MLP com scikit-learn: interface comum .fit(), .predict_proba(), .predict(), .loss_curve()."""
    def __init__(self,
                 hidden_layer_sizes=(64,),
                 lr=1e-3,
                 alpha=1e-4,
                 batch_size=128,
                 max_iter=100,
                 early_stopping=True,
                 random_state=42):
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=lr,
            batch_size=batch_size,
            max_iter=max_iter,
            alpha=alpha,             # L2
            early_stopping=early_stopping,
            n_iter_no_change=10,
            random_state=random_state
        )
        self.fitted_ = False

    def fit(self, X_tr, y_tr):
        self.clf.fit(X_tr, y_tr)
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        assert self.fitted_, "Treine o modelo antes de prever."
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, thr=0.5):
        proba = self.predict_proba(X)
        return (proba >= thr).astype(int)

    def loss_curve(self):
        return getattr(self.clf, "loss_curve_", None)
