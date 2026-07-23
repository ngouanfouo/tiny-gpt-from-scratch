"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    # TODO: return a sorted list of every unique character in text
    return sorted(list(set(text)))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    # TODO: map each character in vocab to its integer position
    return {char: index for index, char in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    # Build a dictionary where each index points back to its character value
    return {index: char for index, char in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    # TODO: look up ch in the stoi mapping and return its id
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    # TODO: map each char in text through stoi (via encode_char) into a list of ids
    return [encode_char(ch,stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    # TODO: look up the character for token_id in the itos dict
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    # TODO: map each id through decode_int and join the characters into one string.
    return "".join(decode_int(token_id,itos) for token_id in ids)

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    # TODO: convert the input list into a 1D numpy ndarray
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    # TODO: return the shape of arr
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    # TODO: return the dtype attribute of arr
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    # TODO: allocate a (rows, cols) array of zeros and return it
    return np.zeros((rows,cols),dtype=np.float64)

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    # TODO: build a seeded RNG and draw a (rows, cols) uniform sample in [0, 1).
    rng=np.random.default_rng(seed)

    return rng.random((rows,cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    # TODO: return the value at row i, column j of arr
    return arr[i,j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    # TODO: return the i-th row of arr as a 1D array of shape (C,)
    return arr[i,:]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    # TODO: index into arr to extract the j-th column as a 1D array.
    return arr[:,j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    # TODO: return the rectangular sub-block of arr bounded by rows [r0,r1) and cols [c0,c1).
    return arr[r0:r1,c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    # TODO: return a new array whose entries are the pairwise sums of a and b
    return a+b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    # TODO: compute the elementwise (Hadamard) product of a and b
    return a*b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    # TODO: add a Python scalar to every element of an array via broadcasting
    return arr+scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    # TODO: return matrix + vector broadcast across rows
    return matrix+vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    # TODO: apply elementwise exponential to arr and return the result
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    # TODO: apply elementwise natural log to arr and return the result
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    # TODO: collapse every element of arr into a single scalar total
    return arr.sum().item()

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    # TODO: reduce the row dimension of arr so the result has shape (C,).
    return arr.sum(axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    # TODO: collapse the column dimension by summing each row
    return arr.sum(axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    # TODO: compute the maximum value of arr along the given axis
    return arr.max(axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    # TODO: compute the matrix product of a and b
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    # TODO: return the transpose of arr using the .T attribute
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    # Compute the sum along the given axis while preserving the dimension rank
    return arr.sum(axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    # TODO: exponentiate the logits, then divide by their total sum
    exps=array_exp(logits)

    return exps/sum_all(exps)

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    # TODO: exponentiate large_value via array_exp and report whether it is inf.
    arr=np.array([large_value],dtype=np.float64)
    exp_arr=array_exp(arr)

    val=exp_arr[0].item()

    return {
        'naive_exp':val,
        'overflowed':np.isinf(val)
    }

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    # TODO: subtract the max before exponentiating, then normalize.
    max_val=max_along_axis(logits,axis=0)

    shifted_logits=logits-max_val
    exps=array_exp(shifted_logits)
    return exps/sum_all(exps)

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    # TODO: turn each row of logits into a probability distribution without overflowing
    row_maxes=max_along_axis(logits,axis=1)

    row_maxes_reshaped=row_maxes[:,np.newaxis]

    shifted_logits=logits-row_maxes_reshaped

    exps=array_exp(shifted_logits)

    row_sums=sum_keepdims(exps,axis=1)

    return exps/row_sums

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    # TODO: validate that text_blob is a non-empty str and return it as the corpus string
    if not isinstance(text_blob,str):
        raise TypeError("Input corpus must be a string.")
    if not text_blob:
        raise ValueError("Input corpus connot be empty.")
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    # TODO: map every character in text through stoi and return as a 1D int64 array
    token_ids=[stoi[char] for char in text]

    return np.array(token_ids,dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    # TODO: compute the integer split index from n and train_frac
    return int(n*train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    # TODO: return (data[:split_idx], data[split_idx:])
    return data[:split_idx],data[split_idx:]

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    # TODO: return an integer block size, at least 1, derived from default_size
    return max(1,default_size)

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    # TODO: extract a single input window of length block_size starting at index i
    return data[i:i+block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    # TODO: extract the target window Y = data[i+1 : i+1+block_size] shifted by one.
    return data[i+1:i+1+block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    # The highest valid starting index must allow for block_size + 1 elements
    high_bound = data_len - block_size
    
    # Draw batch_size random integers uniformly from [0, high_bound)
    return rng.integers(low=0, high=high_bound, size=batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    # TODO: for each offset, take a length-block_size slice of data and stack them as rows
    rows=[slice_x_at_offset(data,offsets,block_size) for offsets in offsets]
    return np.vstack(rows)

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    # TODO: for each offset, take the length-block_size slice starting at i+1 and stack rows
    rows=[slice_y_at_offset(data,offsets,block_size) for offsets in offsets]

    return np.vstack(rows)

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    # TODO: package one training batch (X, Y) of shape (batch_size, block_size) from data using rng.
    offsets=sample_random_batch_offsets(
        data_len=len(data),
        block_size=block_size,
        batch_size=batch_size,
        rng=rng
    )

    X=stack_x_batch(data,offsets,block_size)
    Y=stack_y_batch(data,offsets,block_size)

    return X,Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    # TODO: return a (vocab_size, vocab_size) integer array of zeros.
    return np.zeros((vocab_size,vocab_size),dtype=np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    # TODO: walk consecutive (current, next) pairs in data and add 1 to the matching cell
    for t in range(len(data)-1):
        curr_token=data[t]
        next_token=data[t+1]

        n_matrix[curr_token,next_token]+=1
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    # TODO: allocate counts, then scatter-add 1 at each (data[:-1], data[1:]) pair
    n_matrix=allocate_count_matrix(vocab_size)

    curr_tokens=data[:-1]
    next_tokens=data[1:]

    np.add.at(n_matrix,(curr_tokens,next_tokens),1)
    return n_matrix

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    # TODO: apply +1 Laplace smoothing to the bigram count matrix
    return n_matrix+1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    # TODO: compute per-row sums of the count matrix as a column vector for normalization.
    return sum_keepdims(n_matrix,axis=1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    # TODO: divide each row of n_matrix by its row sum to produce probabilities
    row_sums=row_sums_of_counts(n_matrix)
    return n_matrix/row_sums

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    # TODO: draw one categorical sample from the row of p_matrix at current_id
    probs=p_matrix[current_id]

    return int(rng.choice(len(probs),p=probs))

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    # TODO: build a length-L int array starting at start_id, then sample each next id from p_matrix
    out=np.zeros(length,dtype=np.int64)

    out[0]=start_id
    for t in range(1,length):
        out[t]=sample_next_token(p_matrix,out[t-1],rng)
    return out

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    # TODO: turn ids into a readable string using itos
    return "".join(decode_token(idx,itos) if 'decode_token' in globals() else itos[idx] for idx in ids)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    # TODO: pick out P[current_id, next_id] and return its natural log
    prob=index_element(p_matrix,current_id,next_id)

    return float(array_log(prob))

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    # Initialize the negative log-likelihood accumulator
    total_nll = 0.0
    
    # Iterate through every consecutive token pair in the data stream
    for t in range(len(data) - 1):
        curr_id = data[t]
        next_id = data[t + 1]
        
        # Look up the log probability using your upstream helper and subtract it
        total_nll -= log_prob_of_pair(p_matrix, curr_id, next_id)
        
    return float(total_nll)

# Step 56 - average_nll
def average_nll(p_matrix, data):
    # TODO: return mean negative log likelihood per bigram over consecutive pairs in data.
    total_nll=sum_negative_log_probs(p_matrix,data)

    num_pairs=len(data)-1

    return float(total_nll/num_pairs)

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    # TODO: sample a (vocab_size, vocab_size) array of standard normal values using rng
    return rng.standard_normal((vocab_size,vocab_size),dtype=np.float64)

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    # TODO: return a new array equal to w_matrix multiplied by scale
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    # Ensure ids is processed as an array to read its length correctly
    num_samples = len(ids)
    
    # 1. Allocate an (N, vocab_size) float matrix using your project's helper
    one_hot = make_2d_zeros(num_samples, vocab_size)
    
    # 2. Use advanced indexing to place a 1.0 at each target column per row
    # range(num_samples) handles the row index, ids handles the column index
    one_hot[range(num_samples), ids] = 1.0
    
    return one_hot

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    # TODO: compute logits for the neural bigram model as the matrix product of one-hot inputs and W.
    return matmul(onehot,w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    # 1. Generate the one-hot matrix of shape (B, V) using V = w.shape[0]
    V = w.shape[0]
    onehot = np.eye(V)[ids]
    
    # 2. Compute logits via matrix multiplication
    onehot_result = forward_logits_onehot(onehot, w)
    
    # 3. Compute logits via direct row indexing (embedding lookup)
    index_result = w[ids]
    
    return {
        'onehot_result': onehot_result,
        'index_result': index_result
    }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    # TODO: return the logits for a batch of token ids by direct row lookup into W.
    return w[ids]

# Step 63 - logits_to_probs_rowwise
import numpy as np

def logits_to_probs_rowwise(logits):
    # TODO: convert a (B, V) logits matrix into a row-wise probability matrix
    
    # Subtract the row-wise maximum for numerical stability
    row_maxes = np.max(logits, axis=1, keepdims=True)
    stabilized_logits = logits - row_maxes
    
    # Exponentiate the stabilized values
    exp_logits = np.exp(stabilized_logits)
    
    # Divide by the sum of exponents along each row to get the probabilities
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return probs

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    # TODO: pick out the probability assigned to the correct next token for each batch row
    return probs[np.arange(len(targets)),targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    # TODO: gather correct-token probs, take log, average the negatives
    correct_probs=gather_correct_token_probs(probs,targets)

    log_probs=array_log(correct_probs)

    loss=-np.mean(log_probs)
    return loss

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    # TODO: return a short written derivation ending in dL/dlogits = (probs - onehot(targets)) / B
    
    return """Derivation of dL/dlogits for mean cross-entropy:

1. Let L = -(1/B) * sum_i log(p_i[y_i]) where p_i[y_i] = exp(logits_i[y_i]) / sum_j exp(logits_i[j])
2. For a single example i, L_i = -log(p_i[y_i]) = -logits_i[y_i] + log(sum_j exp(logits_i[j]))
3. The derivative of the log-sum-exp term uses the softmax: d/dlogits_i[k] log(sum_j exp(logits_i[j])) = exp(logits_i[k]) / sum_j exp(logits_i[j]) = softmax(logits_i)[k]
4. Derivative w.r.t. logits_i[k]:
   - If k == y_i: dL_i/dlogits_i[k] = -1 + softmax(logits_i)[k] = softmax(logits_i)[k] - 1
   - If k != y_i: dL_i/dlogits_i[k] = 0 + softmax(logits_i)[k] = softmax(logits_i)[k] - 0
5. Thus dL_i/dlogits_i = softmax(logits_i) - onehot(y_i) = probs_i - onehot(y_i)
6. Averaging over the batch: dL/dlogits = (1/B) * sum_i (probs_i - onehot(y_i))
7. Therefore: dL/dlogits = (probs - onehot(targets)) / B"""

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    # TODO: return dL/dlogits of shape (B, V) averaged over the batch.
    B,V=probs.shape
    onehot=np.zeros_like(probs)
    onehot[np.arange(B),targets]=1.0
    dlogits=(probs-onehot)/B
    return dlogits

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    # TODO: return a fixed multi-line string describing the scatter-add gradient.
    
    return """Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].
Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).
Chain rule: dL/dW = O.T @ dlogits, shape (V, D).
Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.
Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.
Implementation: scatter-add dlogits rows into dW at indices ids."""

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    # TODO: build a (vocab_size, vocab_size) dW and accumulate dlogits[b] into row ids[b].
    
    # Get dimensions
    B, D = dlogits.shape
    
    # Initialize gradient matrix with zeros
    dW = np.zeros((vocab_size, D))
    
    # Accumulate dlogits rows into the corresponding rows of dW
    for b in range(B):
        row_idx = ids[b]
        dW[row_idx] += dlogits[b]
    
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    # TODO: subtract the scaled gradient from the weights and return the new matrix
    w_new=w-learning_rate*dw
    return w_new

# Step 71 - run_one_training_step
import numpy as np

def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    # TODO: chain the upstream forward/loss/backward/update helpers into one step
    
    # Forward pass: compute logits from ids and w using one-hot encoding
    B = len(ids)
    V, D = w.shape
    onehot = np.zeros((B, V))
    onehot[np.arange(B), ids] = 1.0
    logits = onehot @ w  # shape (B, D)
    
    # Compute probabilities using softmax (row-wise)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # numerical stability
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute loss (before update)
    # Gather correct token probabilities
    correct_probs = probs[np.arange(B), targets]
    log_probs = np.log(correct_probs + 1e-10)  # small epsilon for numerical stability
    loss = -np.mean(log_probs)
    
    # Backward pass: compute gradient of loss w.r.t. logits
    onehot_targets = np.zeros_like(probs)
    onehot_targets[np.arange(B), targets] = 1.0
    dlogits = (probs - onehot_targets) / B
    
    # Compute gradient w.r.t. weights using scatter-add
    vocab_size = w.shape[0]
    dW = np.zeros_like(w)
    for b in range(B):
        row_idx = ids[b]
        dW[row_idx] += dlogits[b]
    
    # Update weights using SGD
    new_w = w - learning_rate * dW
    
    return {'w': new_w, 'loss': float(loss)}

# Step 72 - train_neural_bigram_loop
import numpy as np

def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    # TODO: repeatedly sample a batch, run one training step, and log loss every log_every steps
    
    # Initialize random number generator
    rng = np.random.default_rng()
    loss_history = []
    
    for step in range(num_steps):
        # Sample a batch using get_batch with rng parameter
        inputs, targets = get_batch(data, block_size, batch_size, rng)
        
        # Flatten the (batch_size, block_size) windows into a flat stream
        flat_inputs = inputs.flatten()
        flat_targets = targets.flatten()
        
        # Run one training step
        result = run_one_training_step(w, flat_inputs, flat_targets, learning_rate)
        
        # Update weights
        w = result['w']
        
        # Log loss at step 0 and then every log_every steps
        if step == 0 or (step + 1) % log_every == 0:
            loss_history.append(result['loss'])
    
    return {'w': w, 'loss_history': loss_history}

# Step 73 - sample_from_neural_bigram
import numpy as np

def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    # TODO: starting from start_id, sample num_tokens new ids and decode the full sequence...
    
    # Initialize the sequence with the start_id
    sequence = [start_id]
    current_id = start_id
    
    for _ in range(num_tokens):
        # Get logits for the current id (row of W)
        logits = w[current_id]  # shape (V,)
        
        # Convert logits to probabilities using softmax
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        
        # Sample from the distribution
        next_id = np.random.choice(len(probs), p=probs)
        
        # Append the sampled id to the sequence
        sequence.append(next_id)
        
        # Set the current id to the sampled id for the next step
        current_id = next_id
    
    # Decode the full sequence into a string using itos
    return decode_ids(sequence, itos)

# Step 74 - linear_forward
import numpy as np

def linear_forward(x, w):
    # TODO: compute Y = X @ W and return {'y': Y, 'cache': {'x': x, 'w': w}}.
    
    # Compute the output: y = x @ w
    y = x @ w
    
    # Cache the inputs needed for the backward pass
    cache = {'x': x, 'w': w}
    
    return {'y': y, 'cache': cache}

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    # TODO: return a multi-line string with the derivation and shape check
    
    return """Y = X @ W
dL/dX = dY @ W.T
shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"""

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    # TODO: return notes that include the final identity dL/dW = X.T @ dY
    
    return """For Y = X @ W, the loss L depends on W through Y.

By the chain rule, dL/dW = X.T @ dY.

Shape check: X is (B, D_in), dY is (B, D_out), so X.T @ dY has shape (D_in, D_out), matching W.

Therefore: dL/dW = X.T @ dY"""

# Step 77 - linear_backward_dx
import numpy as np

def linear_backward_dx(dy, cache):
    # TODO: compute the gradient of the loss w.r.t. the linear layer input X given dy and cache
    
    # Extract cached values
    x = cache['x']  # shape (B, D_in)
    w = cache['w']  # shape (D_in, D_out)
    
    # Compute gradient w.r.t. input: dL/dX = dY @ W.T
    dx = dy @ w.T  # shape (B, D_in)
    
    return dx

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    # TODO: compute the weight gradient using x from cache and the upstream dy
    x=cache['x']

    dw=x.T @ dy
    return dw

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    # TODO: add b to each row of x and cache b's shape for the backward pass
    y=vector_matrix_broadcast_add(x,b)

    cache={'b_shape':b.shape}
    return {'y':y,'cache':cache}

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    # TODO: sum the upstream gradient over the batch dimension to get db of shape (D,)
    db=np.sum(dy,axis=0)
    return db

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    # TODO: apply elementwise ReLU and cache the input for backward.
    y=np.maximum(0,x)

    cache={'x':x}

    return {'y':y,'cache':cache}

# Step 82 - relu_backward
import numpy as np

def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    # TODO: return dx with gradient zeroed where the cached input was non-positive.
    
    # Extract the original input from the cache
    x = cache['x']
    
    # Gradient is 1 for positive inputs, 0 for non-positive inputs
    # dy is passed through only where x > 0
    dx = dy * (x > 0)
    
    return dx

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    # TODO: produce the (B, V) gradient of mean cross-entropy w.r.t. logits.
    B,V=probs.shape

    onehot=np.zeros_like(probs)
    onehot[np.arange(B),targets]=1.0

    digits=(probs-onehot)/B
    return digits

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the mean of x over the last axis, shape (B, 1)."""
    # TODO: compute the row-wise mean using sum_keepdims over axis=-1
    
    # Use sum_keepdims to sum over the last axis and keep dimensions
    summed = sum_keepdims(x, axis=-1)
    
    # Divide by the size of the last dimension to get the mean
    D = x.shape[-1]
    mean = summed / D
    
    return mean

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    # TODO: compute per-row variance using mean and return a (B, 1) array
    
    # Compute squared deviations from the mean
    # mean has shape (B, 1) and broadcasts to (B, D)
    squared_diff = (x - mean) ** 2
    
    # Sum over the last axis using sum_keepdims
    sum_squared_diff = sum_keepdims(squared_diff, axis=-1)
    
    # Divide by D to get the population (biased) variance
    D = x.shape[-1]
    var = sum_squared_diff / D
    
    return var

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    # TODO: subtract the per-row mean and divide by sqrt(var + eps)
    x_centered=x-mean

    std=np.sqrt(var+eps)

    x_hat=x_centered/std
    return x_hat

# Step 87 - layernorm_forward_affine
import numpy as np

def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    # TODO: normalize each row to zero mean / unit variance, then apply gamma and beta.
    
    # Step 1: Compute mean and variance
    mean = layernorm_forward_mean(x)  # shape (B, 1)
    var = layernorm_forward_variance(x, mean)  # shape (B, 1)
    
    # Step 2: Normalize
    x_hat = layernorm_forward_normalize(x, mean, var, eps)  # shape (B, D)
    
    # Step 3: Apply affine transformation (gamma * x_hat + beta)
    # elementwise_multiply multiplies each column of x_hat by gamma (broadcast)
    scaled = elementwise_multiply(x_hat, gamma)  # shape (B, D)
    y = vector_matrix_broadcast_add(scaled, beta)  # shape (B, D)
    
    # Cache all values needed for backward pass
    cache = {
        'x': x,
        'x_hat': x_hat,
        'mean': mean,
        'var': var,
        'gamma': gamma,
        'eps': eps
    }
    
    return {'y': y, 'cache': cache}

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    # TODO: compute the gradient contribution of the subtract-mean op
    
    # Get batch size and dimension
    B, D = dy.shape
    
    # The operation is: y = x - mean(x)
    # For each row, the gradient w.r.t. x is: dy - mean(dy, axis=1, keepdims=True)
    # This ensures that the sum of the gradient over each row is zero
    
    # Compute the mean of dy over the last axis, keeping dimensions
    mean_dy = np.mean(dy, axis=-1, keepdims=True)  # shape (B, 1)
    
    # Subtract the mean from each row
    dx = dy - mean_dy
    
    return dx

# Step 89 - layernorm_backward_divide_std
import numpy as np

def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    # TODO: propagate the upstream gradient through the divide-by-std step of LayerNorm
    
    # Extract needed values from cache
    x_hat = cache['x_hat']  # shape (B, D) - already normalized
    var = cache['var']      # shape (B, 1)
    eps = cache['eps']      # scalar
    
    # Compute the standard deviation (with eps for stability)
    std = np.sqrt(var + eps)  # shape (B, 1)
    
    # For y = x_centered / std, treating std as constant:
    # dy/dx_centered = dy / std
    # So dx_centered = dy / std
    dx = dy / std
    
    return dx

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    # TODO: chain rule back through affine, divide-by-std, and subtract-mean.
    
    # Extract values from cache
    x = cache['x']          # (B, D)
    x_hat = cache['x_hat']  # (B, D)
    mean = cache['mean']    # (B, 1)
    var = cache['var']      # (B, 1)
    gamma = cache['gamma']  # (D,)
    eps = cache['eps']      # scalar
    
    B, D = x.shape
    
    # Step 1: Backward through affine transformation: y = gamma * x_hat + beta
    # dbeta = sum of dy over batch dimension
    dbeta = np.sum(dy, axis=0)  # (D,)
    
    # dgamma = sum of dy * x_hat over batch dimension
    dgamma = np.sum(dy * x_hat, axis=0)  # (D,)
    
    # dx_hat = dy * gamma (broadcast gamma across rows)
    dx_hat = dy * gamma  # (B, D)
    
    # Step 2: Backward through divide-by-std: x_hat = x_centered / std
    # std = sqrt(var + eps), var = mean(x_centered^2)
    # Need to compute gradient w.r.t. x_centered accounting for variance dependence
    
    # Compute std (shape B, 1)
    std = np.sqrt(var + eps)  # (B, 1)
    
    # dx_centered = dx_hat / std - (x_hat * sum(dx_hat * x_hat, axis=-1, keepdims=True)) / (D * std)
    # This is the full gradient through the normalization
    sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)  # (B, 1)
    dx_centered = dx_hat / std - (x_hat * sum_dx_hat_x_hat) / (D * std)
    
    # Step 3: Backward through subtract-mean: x_centered = x - mean
    # dx = dx_centered - mean(dx_centered, keepdims=True)
    mean_dx_centered = np.mean(dx_centered, axis=-1, keepdims=True)  # (B, 1)
    dx = dx_centered - mean_dx_centered  # (B, D)
    
    return {
        'dx': dx,
        'dgamma': dgamma,
        'dbeta': dbeta
    }

# Step 91 - layernorm_backward_implementation
import numpy as np

def layernorm_backward_implementation(d_out, cache):
    # TODO: return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm given d_out and the forward cache.
    
    # Extract values from cache
    x = cache['x']          # (N, D)
    x_hat = cache['x_hat']  # (N, D)
    mean = cache['mean']    # (N, 1)
    var = cache['var']      # (N, 1)
    gamma = cache['gamma']  # (D,)
    eps = cache['eps']      # scalar
    
    N, D = x.shape
    
    # Step 1: Backward through affine transformation: y = gamma * x_hat + beta
    # dbeta = sum of d_out over batch dimension
    dbeta = np.sum(d_out, axis=0)  # (D,)
    
    # dgamma = sum of d_out * x_hat over batch dimension
    dgamma = np.sum(d_out * x_hat, axis=0)  # (D,)
    
    # dx_hat = d_out * gamma (broadcast gamma across rows)
    dx_hat = d_out * gamma  # (N, D)
    
    # Step 2: Backward through divide-by-std: x_hat = x_centered / std
    # std = sqrt(var + eps)
    std = np.sqrt(var + eps)  # (N, 1)
    
    # Full gradient through normalization
    sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)  # (N, 1)
    dx_centered = dx_hat / std - (x_hat * sum_dx_hat_x_hat) / (D * std)
    
    # Step 3: Backward through subtract-mean: x_centered = x - mean
    # dx = dx_centered - mean(dx_centered, keepdims=True)
    mean_dx_centered = np.mean(dx_centered, axis=-1, keepdims=True)  # (N, 1)
    dx = dx_centered - mean_dx_centered  # (N, D)
    
    return {
        'dx': dx,
        'dgamma': dgamma,
        'dbeta': dbeta
    }

# Step 92 - create_token_embedding
import numpy as np

def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    # TODO: return a (vocab_size, d_model) array of small random values controlled by scale
    
    # Generate random values from standard normal distribution
    E = np.random.randn(vocab_size, d_model) * scale
    
    return E

# Step 93 - token_embedding_forward
import numpy as np

def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    # TODO: look up the embedding row for each token id and build the cache
    
    # Look up the embedding for each token id
    # token_ids is (B, T), embedding_matrix is (V, d_model)
    # Result should be (B, T, d_model)
    out = embedding_matrix[token_ids]
    
    # Build cache
    cache = {
        'token_ids': token_ids,
        'vocab_size': embedding_matrix.shape[0]
    }
    
    return out, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    # TODO: scatter-add d_out into a (vocab_size, d_model) dE using cache['token_ids'].
    
    # Extract token_ids and vocab_size from cache
    token_ids = cache['token_ids']  # shape (B, T)
    vocab_size = cache['vocab_size']
    
    # Get dimensions
    B, T = token_ids.shape
    d_model = d_out.shape[-1]
    
    # Initialize gradient matrix with zeros
    dE = np.zeros((vocab_size, d_model))
    
    # Flatten token_ids and d_out to iterate over all positions
    flat_token_ids = token_ids.flatten()  # shape (B*T,)
    flat_d_out = d_out.reshape(-1, d_model)  # shape (B*T, d_model)
    
    # Scatter-add: accumulate gradients for each token id
    for i, token_id in enumerate(flat_token_ids):
        dE[token_id] += flat_d_out[i]
    
    return dE

# Step 95 - create_positional_embedding
import numpy as np

def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    # TODO: build a (block_size, d_model) matrix of small random values scaled by `scale`
    
    # Create a random matrix of shape (block_size, d_model)
    P = make_2d_random(block_size, d_model, seed=None)
    
    # Scale the random values
    P = scale_w_small(P, scale)
    
    return P

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    # TODO: return the leading seq_len rows of positional_matrix as a (seq_len, d_model) array.
    return positional_matrix[:seq_len]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    # TODO: combine token and positional embeddings into a single (B,T,d_model) tensor
    combined=token_emb + pos_emb
    return combined

# Step 98 - embedding_sum_backward
import numpy as np

def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    # TODO: route d_out to both branches, reducing over the batch axis for pos_emb.
    
    # Gradient w.r.t. token_emb is the same as d_out
    d_token_emb = d_out  # shape (B, T, d_model)
    
    # Gradient w.r.t. pos_emb: sum d_out over the batch dimension
    # Since pos_emb is broadcast to every batch element, the gradient is the sum over batch
    d_pos_emb = sum_axis0(d_out)  # shape (T, d_model)
    
    return {
        'd_token_emb': d_token_emb,
        'd_pos_emb': d_pos_emb
    }

# Step 99 - create_qkv_projections
import numpy as np

def create_qkv_projections(d_model, d_head, scale=0.02):
    # TODO: return a dict with 'Wq','Wk','Wv', each of shape (d_model, d_head)
    
    # Create projection matrices with different seeds
    Wq = make_2d_random(d_model, d_head, seed=0)
    Wq = scale_w_small(Wq, scale)
    
    Wk = make_2d_random(d_model, d_head, seed=1)
    Wk = scale_w_small(Wk, scale)
    
    Wv = make_2d_random(d_model, d_head, seed=2)
    Wv = scale_w_small(Wv, scale)
    
    return {
        'Wq': Wq,
        'Wk': Wk,
        'Wv': Wv
    }

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    # TODO: project x into the query space using w_q
    
    # Compute Q = x @ w_q
    # x shape: (B, T, d_model), w_q shape: (d_model, d_head)
    # Result shape: (B, T, d_head)
    Q = x @ w_q
    
    return Q

# Step 101 - compute_key
import numpy as np

def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    # TODO: project the (B, T, d_model) input through w_k to produce (B, T, d_head) keys.
    
    # Compute K = x @ w_k
    # x shape: (B, T, d_model), w_k shape: (d_model, d_head)
    # Result shape: (B, T, d_head)
    K = x @ w_k
    
    return K

# Step 102 - compute_value
import numpy as np

def compute_value(x, w_v):
    # TODO: project x of shape (B, T, d_model) by w_v of shape (d_model, d_head)
    
    # Compute V = x @ w_v using the matmul helper
    # x shape: (B, T, d_model), w_v shape: (d_model, d_head)
    # Result shape: (B, T, d_head)
    V = matmul(x, w_v)
    
    return V

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    # TODO: compute raw attention scores Q @ K^T per batch element
    
    # Compute attention scores: q @ k^T
    # q shape: (B, T, d_head), k shape: (B, T, d_head)
    # k^T shape for the last two dimensions: (T, d_head) -> (d_head, T)
    # Result shape: (B, T, T)
    scores = q @ k.transpose(0, 2, 1)
    
    return scores

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    # TODO: rescale the scores so their variance does not grow with d_head.
    scaled_scores=scores/np.sqrt(d_head)
    return scaled_scores

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    # TODO: build a (T, T) boolean mask where True marks allowed (query, key) pairs
    
    # Create a lower-triangular matrix of ones
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    
    return mask

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    # TODO: return a (B,T,T) array where positions with causal_mask False are -inf...
    
    # Create a copy to avoid mutating the input
    masked_scores = scaled_scores.copy()
    
    # Get the batch size
    B = scaled_scores.shape[0]
    
    # Expand mask to match the shape of scores: (B, T, T)
    # The mask has shape (T, T), we need to tile it B times along axis 0
    mask_expanded = np.tile(causal_mask, (B, 1, 1))
    
    # Set positions where mask is False to -inf
    masked_scores[~mask_expanded] = -np.inf
    
    return masked_scores

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    # TODO: apply numerically stable softmax along the last axis of masked_scores
    
    # Subtract max for numerical stability (along last axis)
    max_vals = np.max(masked_scores, axis=-1, keepdims=True)
    stable_scores = masked_scores - max_vals
    
    # Compute exponentials
    exp_scores = np.exp(stable_scores)
    
    # Sum along last axis
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    weights = exp_scores / sum_exp
    
    return weights

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    # TODO: mix the value vectors using the attention weights
    
    # Compute weighted sum: out = attn @ v
    # attn shape: (B, T, T), v shape: (B, T, d_head)
    # Result shape: (B, T, d_head)
    out = attn @ v
    
    return out

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    # TODO: return attn_out projected through w_o to shape (B, T, d_model)
    
    # Compute output projection: out = attn_out @ w_o
    # attn_out shape: (B, T, d_head), w_o shape: (d_head, d_model)
    # Result shape: (B, T, d_model)
    out = attn_out @ w_o
    
    return out

# Step 110 - output_projection_backward
import numpy as np

def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    # TODO: backprop through proj = attn_out @ w_o, return gradients for attn_out and w_o
    
    # Extract cached values
    attn_out = cache['attn_out']  # shape (B, T, d_head)
    w_o = cache['w_o']            # shape (d_head, d_model)
    
    # Gradient w.r.t. attn_out: d_attn_out = d_proj @ w_o.T
    # d_proj shape: (B, T, d_model), w_o.T shape: (d_model, d_head)
    d_attn_out = d_proj @ w_o.T   # shape (B, T, d_head)
    
    # Gradient w.r.t. w_o: dw_o = attn_out.T @ d_proj
    # attn_out shape: (B, T, d_head), we need to rearrange for matrix multiplication
    # We need to sum over batch and sequence dimensions
    # attn_out.T would be (d_head, B, T) if we transpose, but we want (d_head, d_model)
    # Actually: attn_out has shape (B, T, d_head), d_proj has shape (B, T, d_model)
    # We can reshape to 2D: attn_out_2d = (B*T, d_head), d_proj_2d = (B*T, d_model)
    # Then dw_o = attn_out_2d.T @ d_proj_2d = (d_head, d_model)
    B, T, d_head = attn_out.shape
    d_model = d_proj.shape[-1]
    
    attn_out_2d = attn_out.reshape(-1, d_head)  # (B*T, d_head)
    d_proj_2d = d_proj.reshape(-1, d_model)     # (B*T, d_model)
    
    dw_o = attn_out_2d.T @ d_proj_2d            # (d_head, d_model)
    
    return {
        'd_attn_out': d_attn_out,
        'dw_o': dw_o
    }

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    attn = cache['attn']
    v = cache['v']
    
    # d_attn = d_attn_out @ V^T
    # V shape: (B, T, d_head) -> V^T shape: (B, d_head, T)
    d_attn = np.matmul(d_attn_out, np.swapaxes(v, -1, -2))
    
    # d_v = Attn^T @ d_attn_out
    # Attn shape: (B, T, T) -> Attn^T shape: (B, T, T)
    d_v = np.matmul(np.swapaxes(attn, -1, -2), d_attn_out)
    
    return {
        'd_attn': d_attn,
        'd_v': d_v
    }

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    # TODO: propagate the softmax Jacobian per row and zero out masked positions.
    
    # Extract values from cache
    attn = cache['attn']          # shape (B, T, T)
    causal_mask = cache['causal_mask']  # shape (T, T)
    
    # Get dimensions
    B, T, _ = attn.shape
    
    # Initialize gradient with zeros
    d_masked_scores = np.zeros_like(attn)
    
    # For each row in each batch element, compute softmax Jacobian
    for b in range(B):
        for i in range(T):
            # Get the attention probabilities for this row
            p = attn[b, i]  # shape (T,)
            
            # Compute the Jacobian: J = diag(p) - p @ p.T
            # This is the gradient of softmax w.r.t. the input scores
            J = np.diag(p) - np.outer(p, p)
            
            # Apply Jacobian to the upstream gradient for this row
            d_masked_scores[b, i] = J @ d_attn[b, i]
    
    # Zero out positions that were masked (causal_mask is False)
    # Expand mask to (B, T, T) for broadcasting
    mask_expanded = np.tile(causal_mask, (B, 1, 1))
    d_masked_scores[~mask_expanded] = 0.0
    
    return d_masked_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    # TODO: propagate d_scaled_scores back through the sqrt(d_head) scaling
    
    # Forward: scaled_scores = scores / sqrt(d_head)
    # Backward: d_scores = d_scaled_scores / sqrt(d_head)
    d_scores = d_scaled_scores / np.sqrt(d_head)
    
    return d_scores

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    # TODO: backprop scores = Q @ K^T to obtain gradients for Q and K
    
    # Extract Q and K from cache
    q = cache['q']  # shape (B, T, d_head)
    k = cache['k']  # shape (B, T, d_head)
    
    # Compute gradient w.r.t. Q: d_q = d_scores @ K
    # d_scores: (B, T, T), K: (B, T, d_head)
    # Result: (B, T, d_head)
    d_q = d_scores @ k
    
    # Compute gradient w.r.t. K: d_k = d_scores.T @ Q
    # d_scores.T: (B, T, T) transposed over last two dims -> (B, T, T)
    # Q: (B, T, d_head)
    # We need to compute d_k = d_scores^T @ Q
    # For each batch: d_scores[b] is (T, T), Q[b] is (T, d_head)
    # Result: (B, T, d_head)
    d_k = d_scores.transpose(0, 2, 1) @ q
    
    return {
        'd_q': d_q,
        'd_k': d_k
    }

# Step 115 - qkv_projection_backward
import numpy as np

def qkv_projection_backward(d_q, d_k, d_v, cache):
    # TODO: backprop through Q=x@Wq, K=x@Wk, V=x@Wv to get dx and dw_q, dw_k, dw_v.
    
    # Extract values from cache
    x = cache['x']      # shape (B, T, d_model)
    w_q = cache['w_q']  # shape (d_model, d_head)
    w_k = cache['w_k']  # shape (d_model, d_head)
    w_v = cache['w_v']  # shape (d_model, d_head)
    
    # Get dimensions
    B, T, d_model = x.shape
    d_head = w_q.shape[1]
    
    # Reshape for efficient computation
    x_flat = x.reshape(-1, d_model)        # (B*T, d_model)
    d_q_flat = d_q.reshape(-1, d_head)     # (B*T, d_head)
    d_k_flat = d_k.reshape(-1, d_head)     # (B*T, d_head)
    d_v_flat = d_v.reshape(-1, d_head)     # (B*T, d_head)
    
    # Gradient w.r.t. x: dx = d_q @ w_q.T + d_k @ w_k.T + d_v @ w_v.T
    # Each term: (B*T, d_head) @ (d_head, d_model) -> (B*T, d_model)
    dx_flat = (d_q_flat @ w_q.T) + (d_k_flat @ w_k.T) + (d_v_flat @ w_v.T)
    dx = dx_flat.reshape(B, T, d_model)
    
    # Gradient w.r.t. Wq: dw_q = x.T @ d_q
    # x_flat.T: (d_model, B*T), d_q_flat: (B*T, d_head)
    # Result: (d_model, d_head)
    dw_q = x_flat.T @ d_q_flat
    
    # Gradient w.r.t. Wk: dw_k = x.T @ d_k
    dw_k = x_flat.T @ d_k_flat
    
    # Gradient w.r.t. Wv: dw_v = x.T @ d_v
    dw_v = x_flat.T @ d_v_flat
    
    return {
        'dx': dx,
        'dw_q': dw_q,
        'dw_k': dw_k,
        'dw_v': dw_v
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    # TODO: split d_model into n_heads equal-sized d_head chunks and return the config dict
    
    # Check that n_heads evenly divides d_model
    if d_model % n_heads != 0:
        raise ValueError(f"n_heads ({n_heads}) must evenly divide d_model ({d_model})")
    
    # Compute per-head dimension
    d_head = d_model // n_heads
    
    # Return config dict
    return {
        'n_heads': n_heads,
        'd_head': d_head,
        'd_model': d_model
    }

# Step 117 - create_multihead_qkv_projections
import numpy as np

def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    # TODO: build a dict with keys 'Wq', 'Wk', 'Wv', each a scaled (d_model, d_model) random matrix
    
    # Create projection matrices with different seeds
    Wq = make_2d_random(d_model, d_model, seed=0)
    Wq = scale_w_small(Wq, scale)
    
    Wk = make_2d_random(d_model, d_model, seed=1)
    Wk = scale_w_small(Wk, scale)
    
    Wv = make_2d_random(d_model, d_model, seed=2)
    Wv = scale_w_small(Wv, scale)
    
    return {
        'Wq': Wq,
        'Wk': Wk,
        'Wv': Wv
    }

# Step 118 - create_multihead_output_projection
import numpy as np

def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    # TODO: build a (d_model, d_model) random matrix and scale it down by `scale`.
    
    # Create random matrix with fixed seed
    Wo = make_2d_random(d_model, d_model, seed=0)
    
    # Scale the values
    Wo = scale_w_small(Wo, scale)
    
    return Wo

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    # TODO: split the last dimension of x into n_heads chunks of size d_head
    
    # Get the shape of the input
    B, T, d_model = x.shape
    
    # Reshape: split the last dimension (d_model) into (n_heads, d_head)
    # We want to keep the first two dimensions (B, T) and split the last
    out = x.reshape(B, T, n_heads, d_head)
    
    return out

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    # TODO: move the heads axis in front of the time axis
    
    # Transpose: move axis 2 (n_heads) to position 1
    # Original: (B, T, n_heads, d_head)
    # Desired:  (B, n_heads, T, d_head)
    y = np.transpose(x_heads, (0, 2, 1, 3))
    
    return y

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    # TODO: return the number of attention heads stored in the multi-head config dict.
    return config['n_heads']

# Step 122 - get_multihead_sequence_length
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    # TODO: return the sequence length T from the (B, T, d_model) tensor.
    
    # Get the shape of the input
    shape = get_array_shape(x)
    
    # Return the second dimension (T)
    return shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    # TODO: return the per-head dimension d_head for multi-head attention.
    
    # Check that n_heads evenly divides d_model
    if d_model % n_heads != 0:
        raise ValueError(f"n_heads ({n_heads}) must evenly divide d_model ({d_model})")
    
    # Compute and return per-head dimension
    d_head = d_model // n_heads
    return d_head

# Step 124 - multihead_masked_softmax_scores
import numpy as np

def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    # TODO: mask future positions then row-wise softmax over the last axis
    
    # Get dimensions
    B, n_heads, T, _ = scores.shape
    
    # Broadcast mask to match scores shape: (B, n_heads, T, T)
    mask_expanded = np.broadcast_to(mask, (B, n_heads, T, T))
    
    # Create a copy to avoid mutating the input
    masked_scores = scores.copy()
    
    # Set positions where mask is False to -inf
    masked_scores[~mask_expanded] = -np.inf
    
    # Apply row-wise softmax over the last axis with numerical stability
    # Subtract max along the last axis for stability
    max_vals = np.max(masked_scores, axis=-1, keepdims=True)
    stable_scores = masked_scores - max_vals
    
    # Compute exponentials (exp of -inf is 0)
    exp_scores = np.exp(stable_scores)
    
    # Sum along last axis
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    weights = exp_scores / sum_exp
    
    # Replace any NaN with 0 (can happen if all values in a row are -inf)
    weights = np.nan_to_num(weights, nan=0.0)
    
    return weights

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    # TODO: combine attention weights with values across heads
    
    # weights: (B, n_heads, T, T)
    # v_heads: (B, n_heads, T, d_head)
    # Result: (B, n_heads, T, d_head)
    output = weights @ v_heads
    
    return output

# Step 126 - transpose_heads_to_back
import numpy as np

def transpose_heads_to_back(x_heads):
    # TODO: move the heads axis back so the result has shape (B, T, n_heads, d_head).
    
    # Transpose: move axis 1 (n_heads) to position 2
    # Original: (B, n_heads, T, d_head)
    # Desired:  (B, T, n_heads, d_head)
    y = np.transpose(x_heads, (0, 2, 1, 3))
    
    return y

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    # TODO: read the sequence-length dimension from x_heads_back's shape
    
    # Get the shape of the input
    shape = get_array_shape(x_heads_back)
    
    # Return the second dimension (T)
    return shape[1]

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    # TODO: collapse the last two axes into a single d_model axis
    
    # Get the shape of the input
    B, T, n_heads, d_head = x_heads_back.shape
    
    # Reshape to collapse the last two dimensions
    # The natural concatenation is: head 0's values, then head 1's, etc.
    # Reshaping with (B, T, -1) will flatten the last two dimensions in order
    merged = x_heads_back.reshape(B, T, n_heads * d_head)
    
    return merged

# Step 129 - multihead_output_projection_forward
import numpy as np

def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    # TODO: project merged through w_out, add b_out, and stash inputs in the cache.
    
    # First, apply linear projection: merged @ w_out
    linear_out = linear_forward(merged, w_out)
    
    # Then add bias
    proj_out = bias_add_forward(linear_out['y'], b_out)
    
    # Combine caches
    cache = {
        'merged': merged,
        'w_out': w_out
    }
    
    return {
        'out': proj_out['y'],
        'cache': cache
    }

# Step 130 - multihead_reshape_transpose_backward
import numpy as np

def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    # TODO: undo the merge/transpose/reshape chain from the forward pass
    
    # Extract shape information
    B = shape_info['B']
    T = shape_info['T']
    n_heads = shape_info['n_heads']
    d_head = shape_info['d_head']
    
    # Step 1: Reshape d_merged (B, T, d_model) back to (B, T, n_heads, d_head)
    d_heads_back = d_merged.reshape(B, T, n_heads, d_head)
    
    # Step 2: Transpose heads back to front: (B, n_heads, T, d_head)
    # This is the inverse of transpose_heads_to_back
    d_heads = np.transpose(d_heads_back, (0, 2, 1, 3))
    
    return d_heads

# Step 131 - ffn_linear_one_forward
import numpy as np

def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    # TODO: apply the first FFN linear that expands d_model to d_ff
    
    # Apply linear projection: x @ w1
    linear_out = linear_forward(x, w1)
    
    # Add bias
    h1 = bias_add_forward(linear_out['y'], b1)
    
    # Build cache for backward pass
    cache = {
        'x': x,
        'w1': w1
    }
    
    return {
        'h1': h1['y'],
        'cache': cache
    }

# Step 132 - ffn_activation_forward
import numpy as np

def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    # TODO: apply ReLU activation in the FFN hidden layer and cache h1
    
    # Apply ReLU activation
    relu_out = relu_forward(h1)
    
    # The cache from relu_forward contains 'x' (which is h1)
    # We can rename the key to 'h1' for consistency
    cache = {'h1': relu_out['cache']['x']}
    
    return relu_out['y'], cache

# Step 133 - ffn_linear_two_forward
import numpy as np

def ffn_linear_two_forward(a1, w2, b2):
    # TODO: project a1 (B, T, d_ff) down to (B, T, d_model) using w2 and b2, return h2 and cache
    
    # Apply linear projection: a1 @ w2
    linear_out = linear_forward(a1, w2)
    
    # Add bias
    h2 = bias_add_forward(linear_out['y'], b2)
    
    # Build cache for backward pass
    cache = {
        'a1': a1,
        'w2': w2
    }
    
    return {
        'h2': h2['y'],
        'cache': cache
    }

# Step 134 - ffn_backward
import numpy as np

def ffn_backward(d_out, cache):
    # TODO: backprop through the full 2-layer FFN and return gradients for all parameters.
    
    # Extract values from cache
    x = cache['x']      # shape (B, T, d_model)
    w1 = cache['w1']    # shape (d_model, d_ff)
    h1 = cache['h1']    # shape (B, T, d_ff)
    a1 = cache['a1']    # shape (B, T, d_ff)
    w2 = cache['w2']    # shape (d_ff, d_model)
    
    # Get dimensions
    B, T, d_model = x.shape
    d_ff = w1.shape[1]
    
    # Step 1: Backward through second linear layer (a1 @ w2 + b2)
    # d_out is gradient w.r.t. h2 (output of FFN)
    
    # For the linear layer: h2 = a1 @ w2 + b2
    # Gradient w.r.t. a1: d_a1 = d_out @ w2.T
    d_a1 = d_out @ w2.T  # shape (B, T, d_ff)
    
    # Gradient w.r.t. w2: dw2 = a1.T @ d_out
    # Reshape to 2D for matrix multiplication
    a1_flat = a1.reshape(-1, d_ff)     # (B*T, d_ff)
    d_out_flat = d_out.reshape(-1, d_model)  # (B*T, d_model)
    dw2 = a1_flat.T @ d_out_flat       # (d_ff, d_model)
    
    # Gradient w.r.t. b2: sum over batch and time dimensions
    db2 = np.sum(d_out, axis=(0, 1))   # (d_model,)
    
    # Step 2: Backward through ReLU activation
    # d_h1 = d_a1 where h1 > 0, else 0
    d_h1 = d_a1 * (h1 > 0)  # shape (B, T, d_ff)
    
    # Step 3: Backward through first linear layer (x @ w1 + b1)
    # h1 = x @ w1 + b1
    
    # Gradient w.r.t. x: dx = d_h1 @ w1.T
    dx = d_h1 @ w1.T  # shape (B, T, d_model)
    
    # Gradient w.r.t. w1: dw1 = x.T @ d_h1
    x_flat = x.reshape(-1, d_model)     # (B*T, d_model)
    d_h1_flat = d_h1.reshape(-1, d_ff)  # (B*T, d_ff)
    dw1 = x_flat.T @ d_h1_flat          # (d_model, d_ff)
    
    # Gradient w.r.t. b1: sum over batch and time dimensions
    db1 = np.sum(d_h1, axis=(0, 1))     # (d_ff,)
    
    return {
        'dx': dx,
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2
    }

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    # TODO: add the sublayer output to its input to form a residual connection.
    out = x + sublayer_out
    return out

# Step 136 - residual_backward
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    # TODO: route the upstream gradient to both branches of the residual add.
    d_x=d_y.copy()
    d_sublayer_out=d_y.copy()

    return d_x,d_sublayer_out

# Step 137 - pre_layernorm_sublayer_forward
import numpy as np

def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    # TODO: apply LayerNorm to x, run sublayer_fn on the result, then residual-add back to x.
    
    # Extract LayerNorm parameters
    gamma = ln_params['gamma']
    beta = ln_params['beta']
    eps = ln_params.get('eps', 1e-5)
    
    # Step 1: Apply LayerNorm to the input
    ln_out = layernorm_forward_affine(x, gamma, beta, eps)
    x_norm = ln_out['y']
    ln_cache = ln_out['cache']
    
    # Step 2: Run the sublayer function on the normalized input
    sublayer_out = sublayer_fn(x_norm, sublayer_params)
    sublayer_y = sublayer_out['y']
    sublayer_cache = sublayer_out['cache']
    
    # Step 3: Apply residual connection
    y = residual_forward(x, sublayer_y)
    
    # Build cache for backward pass
    cache = {
        'x': x,
        'ln_cache': ln_cache,
        'sublayer_cache': sublayer_cache
    }
    
    return {
        'y': y,
        'cache': cache
    }

# Step 138 - transformer_block_forward
def multi_head_attention_forward(x, attn_params):
    """Multi-head attention forward pass."""
    # Extract parameters
    Wq = attn_params['Wq']
    Wk = attn_params['Wk']
    Wv = attn_params['Wv']
    Wo = attn_params['Wo']
    bo = attn_params.get('bo', np.zeros(Wo.shape[1]))  # Get output bias, default to zeros
    n_heads = attn_params['n_heads']
    
    B, T, d_model = x.shape
    d_head = d_model // n_heads
    
    # Step 1: QKV projections
    Q = compute_query(x, Wq)
    K = compute_key(x, Wk)
    V = compute_value(x, Wv)
    
    # Step 2: Reshape to heads
    Q_heads = reshape_to_heads(Q, n_heads, d_head)
    K_heads = reshape_to_heads(K, n_heads, d_head)
    V_heads = reshape_to_heads(V, n_heads, d_head)
    
    # Step 3: Transpose heads to front
    Q_heads_T = transpose_heads_to_front(Q_heads)
    K_heads_T = transpose_heads_to_front(K_heads)
    V_heads_T = transpose_heads_to_front(V_heads)
    
    # Step 4: Compute attention scores
    scores = np.matmul(Q_heads_T, K_heads_T.transpose(0, 1, 3, 2))
    
    # Step 5: Scale scores
    scaled_scores = scores / np.sqrt(d_head)
    
    # Step 6: Build causal mask
    mask = build_causal_mask(T)
    
    # Step 7: Apply mask and softmax
    weights = multihead_masked_softmax_scores(scaled_scores, mask)
    
    # Step 8: Weighted sum
    attn_out_heads = multihead_weighted_sum(weights, V_heads_T)
    
    # Step 9: Transpose heads back
    attn_out_back = transpose_heads_to_back(attn_out_heads)
    
    # Step 10: Merge heads
    merged = merge_heads_to_d_model(attn_out_back)
    
    # Step 11: Output projection with bias
    proj_out = apply_output_projection(merged, Wo)
    out = proj_out + bo  # Add output bias
    
    # Build cache for backward pass
    cache = {
        'x': x,
        'Q': Q,
        'K': K,
        'V': V,
        'Q_heads': Q_heads,
        'K_heads': K_heads,
        'V_heads': V_heads,
        'Q_heads_T': Q_heads_T,
        'K_heads_T': K_heads_T,
        'V_heads_T': V_heads_T,
        'scores': scores,
        'scaled_scores': scaled_scores,
        'weights': weights,
        'attn_out_heads': attn_out_heads,
        'attn_out_back': attn_out_back,
        'merged': merged,
        'proj_out': proj_out,
        'Wq': Wq,
        'Wk': Wk,
        'Wv': Wv,
        'Wo': Wo,
        'bo': bo,
        'n_heads': n_heads,
        'd_head': d_head,
        'mask': mask
    }
    
    return {'y': out, 'cache': cache}

def position_wise_ffn_forward(x, ffn_params):
    """Position-wise FFN forward pass."""
    # Extract parameters
    w1 = ffn_params['w1']
    b1 = ffn_params['b1']
    w2 = ffn_params['w2']
    b2 = ffn_params['b2']
    
    # Step 1: First linear layer
    h1_result = ffn_linear_one_forward(x, w1, b1)
    h1 = h1_result['h1']
    h1_cache = h1_result['cache']
    
    # Step 2: ReLU activation
    a1, relu_cache = ffn_activation_forward(h1)
    
    # Step 3: Second linear layer
    h2_result = ffn_linear_two_forward(a1, w2, b2)
    h2 = h2_result['h2']
    h2_cache = h2_result['cache']
    
    # Build cache for backward pass
    cache = {
        'x': x,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'h1': h1,
        'a1': a1,
        'h2': h2,
        'h1_cache': h1_cache,
        'relu_cache': relu_cache,
        'h2_cache': h2_cache
    }
    
    return {'y': h2, 'cache': cache}

def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Args:
        x: ndarray of shape (B, T, d_model).
        block_params: dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        dict with 'y' (B, T, d_model) and 'cache' with keys
        'attn_branch' and 'ffn_branch'.
    """
    # First sublayer: LayerNorm + Multi-head Self-Attention with residual
    attn_result = pre_layernorm_sublayer_forward(
        x,
        block_params['ln1'],
        multi_head_attention_forward,
        block_params['attn']
    )
    
    # Second sublayer: LayerNorm + Position-wise FFN with residual
    ffn_result = pre_layernorm_sublayer_forward(
        attn_result['y'],
        block_params['ln2'],
        position_wise_ffn_forward,
        block_params['ffn']
    )
    
    return {
        'y': ffn_result['y'],
        'cache': {
            'attn_branch': attn_result['cache'],
            'ffn_branch': ffn_result['cache']
        }
    }

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block.

    Args:
        d_y: upstream gradient w.r.t. block output, shape (B, T, D).
        cache: dict from transformer_block_forward, with keys 'attn_branch' and 'ffn_branch'.
        block_params: nested dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        (d_x, grads) where d_x has shape (B, T, D) and grads is a nested dict
        with keys 'ln1', 'ln2', 'attn', 'ffn' mirroring block_params.
    """
    # Recover x from cache and rebuild a complete cache
    x = cache['attn_branch']['x']
    complete_cache = _complete_block_cache(x, block_params)
    
    # Unpack complete cache
    attn_branch_cache = complete_cache['attn_branch']
    ffn_branch_cache = complete_cache['ffn_branch']
    
    # ----- FFN branch backward (outer residual) -----
    # y = h1 + FFN(LN2(h1))
    
    # Route d_y through residual: skip connection and sublayer
    d_h1_residual, d_ffn_out = residual_backward(d_y)
    
    # Backprop through FFN sublayer
    d_ffn_ln_out, ffn_grads = _ffn_sublayer_backward(
        d_ffn_out,
        ffn_branch_cache['sublayer_cache'],
        block_params['ffn']
    )
    
    # Backprop through LayerNorm2
    d_h1_ln, d_ln2_gamma, d_ln2_beta = layernorm_backward_affine(
        d_ffn_ln_out,
        ffn_branch_cache['ln_cache']
    )
    
    # Sum gradient contributions at h1
    d_h1 = d_h1_residual + d_h1_ln
    
    # ----- Attention branch backward (inner residual) -----
    # h1 = x + Attn(LN1(x))
    
    # Route d_h1 through residual: skip connection and sublayer
    d_x_residual, d_attn_out = residual_backward(d_h1)
    
    # Backprop through Attention sublayer
    d_attn_ln_out, attn_grads = _attn_sublayer_backward(
        d_attn_out,
        attn_branch_cache['sublayer_cache'],
        block_params['attn']
    )
    
    # Backprop through LayerNorm1
    d_x_ln, d_ln1_gamma, d_ln1_beta = layernorm_backward_affine(
        d_attn_ln_out,
        attn_branch_cache['ln_cache']
    )
    
    # Sum gradient contributions at x
    d_x = d_x_residual + d_x_ln
    
    # Assemble gradients
    grads = {
        'ln1': {'gamma': d_ln1_gamma, 'beta': d_ln1_beta},
        'ln2': {'gamma': d_ln2_gamma, 'beta': d_ln2_beta},
        'attn': attn_grads,
        'ffn': ffn_grads
    }
    
    return d_x, grads

# Step 140 - stack_transformer_blocks (not yet solved)
# TODO: implement

# Step 141 - forward_through_all_blocks (not yet solved)
# TODO: implement

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments (not yet solved)
# TODO: implement

# Step 148 - initialize_adam_step_counter (not yet solved)
# TODO: implement

# Step 149 - adam_increment_step (not yet solved)
# TODO: implement

# Step 150 - adam_update_first_moment (not yet solved)
# TODO: implement

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

