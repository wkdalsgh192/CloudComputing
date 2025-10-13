import math, random
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0.0, x)
def d_relu(x):
    return (x > 0).astype(x.dtype)
def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True) # stability
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)
def one_hot(y, num_classes):
    Y = np.zeros((y.size, num_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y

class MLP2:
    """
    2-layer MLP:
    hidden = ReLU(X @ W1 + b1)
    out = hidden @ W2 + b2
    mode: 'regression' (MSE) or 'classification' (softmax CE)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, mode='regression', seed=0):
        rng = np.random.default_rng(seed)
        k1 = math.sqrt(2.0/in_dim) # He init for ReLU
        k2 = math.sqrt(2.0/hidden_dim)
        self.W1 = rng.normal(0, k1, (in_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, k2, (hidden_dim, out_dim))
        self.b2 = np.zeros((1, out_dim))
        assert mode in ('regression','classification')
        self.mode = mode
    
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        if self.mode == 'classification':
            self.probs = softmax(self.z2)
            return self.probs
        return self.z2 # regression raw output

    def loss(self, y):
        if self.mode == 'regression':
            # y: shape (N, out_dim)
            diff = self.z2 - y
            return 0.5 * np.mean(np.sum(diff*diff, axis=1))
        else:
            # y: integer labels shape (N,)
            N = y.shape[0]
            # cross-entropy
            logp = -np.log(self.probs[np.arange(N), y] + 1e-12)
            return np.mean(logp)

    def backward(self, y):
        N = self.X.shape[0]
        if self.mode == 'regression':
            dz2 = (self.z2 - y) / N
        else:
            Y = one_hot(y, self.z2.shape[1])
            dz2 = (self.probs - Y) / N

        dW2 = self.h1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * d_relu(self.z1)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2
 
    def step(self, grads, lr=1e-2, weight_decay=0.0):
        dW1, db1, dW2, db2 = grads
        if weight_decay > 0:
            dW1 += weight_decay * self.W1
            dW2 += weight_decay * self.W2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def batch_iter(X, y, batch_size, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            j = idx[i:i+batch_size] 
            yield X[j], (y[j] if isinstance(y, np.ndarray) else y[j]) 

def train(model, X, y, epochs=200, lr=1e-2, batch_size=64, weight_decay=0.0,
verbose=True):
    losses = []
    for e in range(1, epochs+1):
        for Xb, yb in batch_iter(X, y, batch_size, shuffle=True, seed=e):
            model.forward(Xb)
            grads = model.backward(yb)
            model.step(grads, lr=lr, weight_decay=weight_decay)
        
        # track loss full-batch
        model.forward(X)
        L = model.loss(y)
        losses.append(L)
        if verbose and (e % max(1, epochs//10) == 0):
            print(f"epoch {e:4d}/{epochs} loss={L:.4f}")
    return losses

def regression():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (400, 1))
    def target(x):
        return np.maximum(0, 0.5*x + 0.2) + 0.3*np.maximum(0, -x + 0.5)
    y = target(X) + 0.05*rng.normal(size=(400, 1))

    m_reg = MLP2(in_dim=1, hidden_dim=32, out_dim=1, mode='regression', seed=1)
    train(m_reg, X, y, epochs=500, lr=1e-2, batch_size=64, weight_decay=1e-4)

    # xt=np.linspace(-2, 2, 9).reshape(-1, 1)
    # pred = m_reg.forward(xt)
    # print(np.hstack([xt, target(xt), pred]))
    
    # Compare model prediction to ground truth
    xt = np.linspace(-2, 2, 200).reshape(-1, 1)  # more fine-grained points for smoother line
    yt_true = target(xt)
    yt_pred = m_reg.forward(xt)

    plt.figure(figsize=(6,4))
    plt.plot(xt, yt_true, label="True function", linewidth=2)
    plt.plot(xt, yt_pred, label="MLP prediction", linestyle='--')
    plt.scatter(X, y, s=10, alpha=0.5, label="Training data")
    plt.title("MLP Regression Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("mlp_regression.png")

def binary_classification():
    # Concentric rings dataset
    def rings(n=600, inner_r=0.5, gap=0.2, noise=0.06, seed=0):
        rng = np.random.default_rng(seed)
        n2 = n//2
        theta1 = rng.uniform(0, 2*np.pi, n2)
        r1 = inner_r + noise*rng.normal(size=n2)
        x1 = np.c_[r1*np.cos(theta1), r1*np.sin(theta1)]
        theta2 = rng.uniform(0, 2*np.pi, n-n2)
        r2 = inner_r + gap + noise*rng.normal(size=n-n2)
        x2 = np.c_[r2*np.cos(theta2), r2*np.sin(theta2)]
        X = np.vstack([x1, x2])
        y = np.array([0]*n2 + [1]*(n-n2))
        return X, y

    Xb, yb = rings(n=800, inner_r=0.6, gap=0.5, noise=0.07, seed=1)
    m_bin = MLP2(in_dim=2, hidden_dim=64, out_dim=2, mode='classification', seed=2)
    train(m_bin, Xb, yb, epochs=200, lr=5e-3, batch_size=64, weight_decay=1e-4)
    probs = m_bin.forward(Xb)
    preds = probs.argmax(axis=1)
    acc = (preds == yb).mean()
    print("train accuracy:", acc)

    x_min, x_max = Xb[:, 0].min() - 0.2, Xb[:, 0].max() + 0.2
    y_min, y_max = Xb[:, 1].min() - 0.2, Xb[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs_grid = m_bin.forward(grid)
    preds_grid = probs_grid.argmax(axis=1).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds_grid, cmap='coolwarm', alpha=0.4)
    plt.scatter(Xb[:, 0], Xb[:, 1], c=yb, cmap='coolwarm', s=20, edgecolor='k')
    plt.title(f"MLP Binary Classification (acc={acc:.2f})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mlp_binary_classification.png")

def multiclass_classification():
    def three_blobs(n=900, seed=0):
        rng = np.random.default_rng(seed)
        means = np.array([[0,0], [2.5, 0.5], [-2.0, 1.5]])
        cov = np.array([[0.4,0.0],[0.0,0.4]])
        Xs, ys = [], []
        for k, m in enumerate(means):
            Xk = rng.multivariate_normal(m, cov, size=n//3)
            yk = np.full(n//3, k)
            Xs.append(Xk); ys.append(yk)
        return np.vstack(Xs), np.hstack(ys)
    Xm, ym = three_blobs(900, seed=4)
    m_multi = MLP2(in_dim=2, hidden_dim=64, out_dim=3, mode='classification', seed=3)
    train(m_multi, Xm, ym, epochs=200, lr=5e-3, batch_size=64)
    probs = m_multi.forward(Xm)
    preds = probs.argmax(axis=1)
    acc = (preds == ym).mean() 
    print("train acc:", acc) 

    cmap = plt.cm.tab10
    x_min, x_max = Xm[:, 0].min() - 0.2, Xm[:, 0].max() + 0.2
    y_min, y_max = Xm[:, 1].min() - 0.2, Xm[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs_grid = m_multi.forward(grid)
    preds_grid = probs_grid.argmax(axis=1).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds_grid, cmap=cmap, alpha=0.4)
    plt.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap=cmap, s=20, edgecolor='k')
    plt.title(f"MLP Multiclass Classification (acc={acc:.2f})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mlp_multi_classification.png")


if __name__ == "__main__":
    # regression()
    # binary_classification()
    multiclass_classification()
