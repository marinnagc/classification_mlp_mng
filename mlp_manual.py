# mlp_manual.py
import numpy as np

def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def _relu(z):    return np.maximum(0, z)
def _relu_grad(z): return (z > 0).astype(z.dtype)

def _bce_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

class _MLPBinary:
    """MLP 1-hidden-layer: ReLU -> Sigmoid, BCE, SGD mini-batch, L2 e early-stopping."""
    def __init__(self, n_in, n_hidden=64, lr=1e-3, l2=1e-4, seed=42):
        rng = np.random.default_rng(seed)
        # He (ReLU) para W1, Xavier p/ W2
        self.W1 = rng.normal(0, np.sqrt(2.0/n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros((n_hidden,))
        self.W2 = rng.normal(0, np.sqrt(1.0/n_hidden), size=(n_hidden, 1))
        self.b2 = np.zeros((1,))
        self.lr = lr
        self.l2 = l2

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        yhat = _sigmoid(z2).ravel()
        return yhat, (X, z1, a1, z2, yhat)

    def backward(self, cache, y_true):
        X, z1, a1, z2, yhat = cache
        N = X.shape[0]
        y_true = y_true.reshape(-1, 1)
        yhat   = yhat.reshape(-1, 1)

        dz2 = (yhat - y_true) / N
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        if self.l2 > 0: dW2 += self.l2 * self.W2

        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        if self.l2 > 0: dW1 += self.l2 * self.W1

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict_proba(self, X):
        yhat, _ = self.forward(X)
        return yhat

class NumpyMLPModel:
    """Wrapper com interface comum e histórico para curvas."""
    def __init__(self,
                 n_hidden=64,
                 lr=1e-3,
                 l2=1e-4,
                 batch_size=128,
                 epochs=80,
                 patience=8,
                 seed=42):
        self.hp = dict(n_hidden=n_hidden, lr=lr, l2=l2,
                       batch_size=batch_size, epochs=epochs,
                       patience=patience, seed=seed)
        self.model = None
        self.hist = None

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        rng = np.random.default_rng(self.hp["seed"])
        n_in = X_tr.shape[1]
        model = _MLPBinary(n_in, self.hp["n_hidden"], self.hp["lr"], self.hp["l2"], self.hp["seed"])

        best_val = np.inf
        best_state = None
        wait = 0
        hist = {"loss_tr": [], "loss_val": [], "acc_tr": [], "acc_val": []}

        def _acc(y, p, thr=0.5): return np.mean((p >= thr) == y)

        for ep in range(self.hp["epochs"]):
            idx = rng.permutation(X_tr.shape[0])
            Xb, yb = X_tr[idx], y_tr[idx]
            for s in range(0, Xb.shape[0], self.hp["batch_size"]):
                Xe = Xb[s:s+self.hp["batch_size"]]
                ye = yb[s:s+self.hp["batch_size"]]
                yhat, cache = model.forward(Xe)
                # L2 somado fora (escala simples)
                _ = _bce_loss(ye, yhat) + 0.5*model.l2*(np.sum(model.W1**2)+np.sum(model.W2**2))/max(1, Xe.shape[0])
                model.backward(cache, ye)

            # métricas por época
            yhat_tr = model.predict_proba(X_tr)
            loss_tr = _bce_loss(y_tr, yhat_tr)
            acc_tr  = _acc(y_tr, yhat_tr)
            hist["loss_tr"].append(loss_tr)
            hist["acc_tr"].append(acc_tr)

            if X_val is not None:
                yhat_val = model.predict_proba(X_val)
                loss_val = _bce_loss(y_val, yhat_val)
                acc_val  = _acc(y_val, yhat_val)
                hist["loss_val"].append(loss_val)
                hist["acc_val"].append(acc_val)

                if loss_val < best_val - 1e-5:
                    best_val = loss_val
                    best_state = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.hp["patience"]:
                        if best_state is not None:
                            model.W1, model.b1, model.W2, model.b2 = best_state
                        break

        self.model = model
        self.hist = hist
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)
