import torch
import torch.nn.functional as F
sentence = torch.tensor(
    [
        0, # can
        7, # you
        1, # help
        2, # me
        5, # to
        6, # translate
        4, # this
        3  # sentence
    ]
)

# produce embeddings of each word
torch.manual_seed(123)
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence).detach()
print(embedded_sentence.shape)

omega = embedded_sentence.matmul(embedded_sentence.T)

# compute attention weights
attention_weights = F.softmax(omega, dim=1)
print(attention_weights.shape)
print(attention_weights.sum(dim=1))

# compute context vectors
x_2 = embedded_sentence[1, :]
context_vectors = torch.matmul(attention_weights, embedded_sentence)
print(context_vectors)

# initialize projection matrices for self attention
d = embedded_sentence.shape[1]
U_query = torch.rand(d, d)
U_key = torch.rand(d, d)
U_value = torch.rand(d, d)

# compute the query, key, and value sequences for the second input element
x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

# compute key and value sequences for all input elements
keys = U_key.matmul(embedded_sentence.T).T
print(torch.allclose(key_2, keys[1]))
values = U_value.matmul(embedded_sentence.T).T
print(torch.allclose(value_2, values[1]))

# compute un-normalized attention weights
omega_2 = query_2.matmul(keys.T)

# normalize attention weights for entire input sequence w.r.t. second input element
attention_weights_2 = F.softmax(omega_2 / d**0.5, dim=0)
context_vector_2 = attention_weights_2.matmul(values)


########################
# initialize multi-head attention projection matrices
h = 8
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)

# compute key and value sequences for each head
multihead_key_2 = multihead_U_key.matmul(x_2)
multihead_value_2 = multihead_U_value.matmul(x_2)
# second input element, third attention head:
print(multihead_key_2[2])

# repeat key and value computations for all input sequences
stacked_inputs = embedded_sentence.T.repeat(8, 1, 1)
print(stacked_inputs.shape)
multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
print(multihead_keys.shape)
multihead_keys = multihead_keys.permute(0, 2, 1) # heads, embedding size, # words -> # heads, # words, embedding size

# second key value in second attention head
print(multihead_keys[2, 1])

# repeat for value sequences
multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0, 2, 1)

multihead_z_2 = torch.rand(8, 16) # 8 attention heads, 16 dimensional context vectors

# concatenate into one long vector to feed into fully connected layer
linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())
print(context_vector_2.shape)


