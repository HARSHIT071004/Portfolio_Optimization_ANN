import numpy as np
import os

_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'portfolio_weights.npz')
_weights = None

def _load_weights():
    global _weights
    if _weights is not None:
        return
    data = np.load(_MODEL_PATH)
    _weights = {
        'w0': data['dense_0_kernel'], 'b0': data['dense_0_bias'],
        'w1': data['dense_1_kernel'], 'b1': data['dense_1_bias'],
        'w2': data['dense_2_kernel'], 'b2': data['dense_2_bias'],
        'w3': data['dense_3_kernel'], 'b3': data['dense_3_bias'],
    }

def predict(x):
    _load_weights()
    w = _weights
    x = x.astype(np.float32)
    x = np.maximum(0, x @ w['w0'] + w['b0'])
    x = np.maximum(0, x @ w['w1'] + w['b1'])
    x = np.maximum(0, x @ w['w2'] + w['b2'])
    x = x @ w['w3'] + w['b3']
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return x
