import numpy as np

queries = np.array([[1, 1],
                    [0, 1]])

keys = np.array([[0, 1],
                 [2, 3]])

values = np.array([[1, 2],
                   [3, 4]])

attention_logits = np.dot(queries, keys.T)

key_vector_dimension = keys.shape[1]

scaled_attention_logits = attention_logits / np.sqrt(key_vector_dimension)

def row_softmax(logits_matrix):
    stabilized_logits = logits_matrix - np.max(logits_matrix, axis=1, keepdims=True)
    exponentials = np.exp(stabilized_logits)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)

attention_weights = row_softmax(scaled_attention_logits)

self_attention_output = np.dot(attention_weights, values)

print("QK^T (Attention Logits) =")
print(attention_logits)

print("\nScaled Attention Logits =")
print(scaled_attention_logits)

print("\nAttention Weights =")
print(attention_weights)

print("\nSelf-Attention Output =")
print(self_attention_output)