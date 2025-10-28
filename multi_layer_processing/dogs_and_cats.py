import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time

def load_data(data_dir):
    X, y = [], []
    for file in os.listdir(data_dir):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            img_path = os.path.join(data_dir, file)
            img = Image.open(img_path).convert('RGB').resize((64, 64))
            X.append(np.array(img) / 255.0)
            label = file.split('.')[0]
            y.append(0 if label == 'cat' else 1)
        except Exception as e:
            print(f"Skipped {img_path}: {e}")
    return np.array(X), np.array(y)

def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

conv_filter = np.random.randn(3, 3, 3) * 0.1
W_fc = np.random.randn(31 * 31, 1) * 0.01
b_fc = np.zeros((1,))


def conv2d(img, kernel):
    h, w, c = img.shape
    kh, kw, kc = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = img[i:i+kh, j:j+kw, :]
            out[i, j] = np.sum(region * kernel)
    return out

def maxpool(img, size=2):
    h, w = img.shape
    new_h, new_w = h // size, w // size
    pooled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            pooled[i, j] = np.max(img[i*size:(i+1)*size, j*size:(j+1)*size])
    return pooled

def forward(img):
    global conv_filter, W_fc, b_fc
    conv_out = conv2d(img, conv_filter)
    relu_out = relu(conv_out)
    pooled = maxpool(relu_out)
    flat = pooled.flatten()
    fc_out = relu(np.dot(flat, W_fc) + b_fc)
    return fc_out, flat, pooled, relu_out

def train(X_train, y_train, epochs, lr):
    global W_fc, b_fc
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        epoch_start = time.time()
        for i in range(len(X_train)):
            img = X_train[i]
            label = np.array([y_train[i]])
            y_pred, flat, pooled, relu_out = forward(img)
            loss = mse_loss(label, y_pred)
            total_loss += loss
            grad_y = mse_grad(label, y_pred) * relu_derivative(y_pred)
            grad_W = np.outer(flat, grad_y)
            grad_b = grad_y
            W_fc -= lr * grad_W
            b_fc -= lr * grad_b

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.4f} | Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

def save_model(conv_filter, W_fc, b_fc, filename="cnn_model.pkl"):
    model_params = {
        "conv_filter": conv_filter,
        "W_fc": W_fc,
        "b_fc": b_fc
    }
    with open(filename, "wb") as f:
        pickle.dump(model_params, f)
    print(f"Model saved to {filename}")

def evaluate(X_test, y_test):
    correct = 0
    for i in range(len(X_test)):
        y_pred, _, _, _ = forward(X_test[i])
        prediction = int(y_pred > 0.5)
        if prediction == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

def main():
    X, y = load_data('/mnt/c/Users/minhojang/Downloads/train/train')
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train(X_train, y_train, epochs=10, lr=0.001)
    save_model(conv_filter, W_fc, b_fc)
    evaluate(X_test, y_test)


if __name__ == "__main__":
    main()