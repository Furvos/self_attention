import numpy as np

Q = np.array([[1, 1],
              [0, 1]])

K = np.array([[0, 1],
              [2, 3]])

V = np.array([[1, 2],
              [3, 4]])

scores = np.dot(Q, K.T)

scaled_scores = scores / np.sqrt(K.shape[1])
print("QK^T =")
print(scores)
print("\nScaled QK^T =")
print(scaled_scores)
print("\nK.shape[1] =", K.shape[1])
print("\nK.shape[0] =", K.shape[0])