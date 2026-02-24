import numpy as np

Q = np.array([[1, 1],
              [0, 1]])

K = np.array([[0, 1],
              [2, 3]])

V = np.array([[1, 2],
              [3, 4]])

scores = np.dot(Q, K.T)

#d_k === dimension of the key vectors | nº of columns in K
d_k = K.shape[1]

scaled_scores = scores / np.sqrt(d_k)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

attention_weights = softmax(scaled_scores)

print("QK^T =")
print(scores)

print("\nScaled QK^T =")
print(scaled_scores)

print("\nAttention Weights =")
print(attention_weights)