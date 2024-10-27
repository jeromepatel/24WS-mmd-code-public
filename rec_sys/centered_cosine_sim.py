import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import linalg as sp_linalg
from scipy import linalg
import scipy
from sklearn.metrics.pairwise import cosine_similarity

def center_vector(v):
    """Center a vector by subtracting its mean."""
    mean = v.mean() if issparse(v) else np.mean(v)
    return v - mean

def center_and_nan_to_zero_sparse(matrix, axis=0):
    """Center a sparse matrix or vector and replace NaN values with zeros."""
    if issparse(matrix):
        data = matrix.data
        # Compute the mean of non-NaN data using np.nanmean
        mean = np.nanmean(data) if data.size > 0 else 0
        centered_data = data - mean
        # Replace NaNs with zeros
        centered_data = np.nan_to_num(centered_data)
        centered_matrix = csr_matrix((centered_data, matrix.indices, matrix.indptr), shape=matrix.shape)
    else:
        #for dense arrays
        mean = np.nanmean(matrix, axis=axis, keepdims=True)
        centered_matrix = np.nan_to_num(matrix - mean)
    return centered_matrix

def centered_cosine_sim(u, v):
    """Compute the centered cosine similarity between two sparse vectors."""
    # Center the vectors and handle NaN values
    cen_u = center_and_nan_to_zero_sparse(u)
    cen_v = center_and_nan_to_zero_sparse(v)
    # Compute the dot product & normmmm
    numerator = cen_u.dot(cen_v.T).data[0] if issparse(cen_u) else np.dot(cen_u, cen_v)
    norm_u = sp_linalg.norm(cen_u) if issparse(cen_u) else linalg.norm(cen_u)
    norm_v = sp_linalg.norm(cen_v) if issparse(cen_v) else linalg.norm(cen_v)
    # Avoid division by zero
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return numerator / (norm_u * norm_v)

def fast_centered_cosine_sim(matrix, vector):
    """Compute centered cosine similarity between each row of a sparse matrix and a sparse vector."""
    # Center the matrix and vector using np.nanmean
    cen_matrix = center_and_nan_to_zero_sparse(matrix)
    cen_vector = center_and_nan_to_zero_sparse(vector)
    numerators = cen_matrix.dot(cen_vector.T).toarray().ravel()
    # Compute the norms and dot product
    matrix_norms = sp_linalg.norm(cen_matrix, axis=1)
    vector_norm = sp_linalg.norm(cen_vector)
    # Avoid division by zero
    denominator = matrix_norms * vector_norm
    denominator[denominator == 0] = np.inf 
    similarities = numerators / denominator
    return similarities

# Unit tests for centered_cosine_sim
def test_centered_cosine_sim_b1():
    k = 100
    xi = np.arange(1, k + 1)  
    x = xi.astype(float)
    y = xi[::-1]  

    x_sparse = scipy.sparse.csr_matrix(x)
    y_sparse = scipy.sparse.csr_matrix(y)
    sim = centered_cosine_sim(x_sparse, y_sparse)
    expected_sim =-1.0000000000000002 #expected sim
    assert np.isclose(sim, expected_sim), f"Test b.1 failed: expected {expected_sim}, got {sim}"
    print("Test b.1 passed: Centered cosine similarity equals expected value of", expected_sim)

def test_centered_cosine_sim_b2():
    k = 100
    xi = np.arange(1, k + 1, dtype=float) 
    # Set xi to NaN at specified indices
    c_values = [2, 3, 4, 5, 6]
    nan_indices = []
    for c in c_values:
        indices = c + np.arange(0, 10) * 10
        nan_indices.extend(indices[indices < k])
    xi[nan_indices] = np.nan
    x = xi
    y = xi[::-1] 
    
    x_sparse = scipy.sparse.csr_matrix(x)
    y_sparse = scipy.sparse.csr_matrix(y)
    sim = centered_cosine_sim(x_sparse, y_sparse)
    expected_sim = -0.8019070321811681  #expected sim with nans
    assert np.isclose(sim, expected_sim), f"Test b.2 failed: expected {expected_sim}, got {sim}"
    print("Test b.2 passed: Centered cosine similarity equals expected value of", expected_sim)



def main():
    # Test functions for B.1 and B.2 
    test_centered_cosine_sim_b1()
    test_centered_cosine_sim_b2()



if __name__ == "__main__":
    main()