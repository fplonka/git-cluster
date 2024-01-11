import numba
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def calculate_correlation(embedding, distance_matrix):
    # N = distance_matrix.shape[0]
    # n = 5000
    # if N > n:
    #     indices = np.random.choice(N, n, replace=False)
    #     selected_embedding = embedding[indices, :]
    #     selected_distance_matrix = distance_matrix[np.ix_(indices, indices)]
    #     return calculate_correlation(selected_embedding, selected_distance_matrix)

    distance_matrix_ld = euclidean_distances(embedding)
    correlation, _ = pearsonr(distance_matrix.ravel(),
                              distance_matrix_ld.ravel())

    return correlation


def log_space_values(a, b, n):
    """
    Generates n logarithmically spaced values between a and b.
    """
    log_a = np.log10(a)
    log_b = np.log10(b)
    return np.logspace(log_a, log_b, num=n, base=10)


@numba.jit(nopython=True, fastmath=True)
def jaccard_fast(row1, row2):
    padding_value = np.iinfo(np.int32).max
    intersection = 0
    union = 0

    i = j = 0
    while i < len(row1) and j < len(row2):
        if row1[i] == padding_value and row2[j] == padding_value:
            break

        if row1[i] == row2[j]:
            intersection += 1
            i += 1
            j += 1
        elif row1[i] < row2[j] or row2[j] == padding_value:
            i += 1
        else:
            j += 1

        union += 1

    # Add remaining elements in row1 and row2 for union count
    while i < len(row1) and row1[i] != padding_value:
        union += 1
        i += 1

    while j < len(row2) and row2[j] != padding_value:
        union += 1
        j += 1

    return intersection / union if union != 0 else 0


@numba.jit(nopython=True, fastmath=True)
def calculate_loss(X, D):
    distances = np.sqrt(((X[:, None, :] - X)**2).sum(axis=2))
    loss = np.sum((D - distances) ** 2) / 2

    return loss


@numba.jit(nopython=True, fastmath=True)
def calculate_loss(X, D):
    distances = np.sqrt(((X[:, None, :] - X)**2).sum(axis=2))
    loss = np.sum((D - distances) ** 2) / 2

    return loss


@numba.jit(nopython=True, fastmath=True)
def calculate_loss2(X, D):
    loss = 0
    N = X.shape[0]
    for i in range(N):
        xi = X[i]
        partial = 0
        for j in range(i+1, N):
            xj = X[j]
            partial = (np.linalg.norm(xi - xj) - D[i, j])**2

        loss += partial

    return loss


@numba.jit(nopython=True)
def simple_hash(x, k):
    return hash(x) ^ k


@numba.jit(nopython=True)
def minhash_signature(matrix, num_hashes, max_int=np.iinfo(np.int32).max):
    n, m = matrix.shape
    signatures = np.full((n, num_hashes), max_int, dtype=np.int32)

    random_nums = np.zeros(m, dtype=np.int32)
    for i in range(m):
        random_nums[i] = np.random.randint(0, max_int)
        print(random_nums[i])

    for j in range(m):
        for i in range(n):
            element = matrix[i, j]
            if element == max_int:
                continue  # Skip padding values
            for k in range(num_hashes):
                hash_val = simple_hash(element, random_nums[k]) % max_int
                signatures[i, k] = min(signatures[i, k], hash_val)
    return signatures


@numba.jit(nopython=True)
def estimated_jaccard(signature_matrix, i, j):
    intersection = np.sum(signature_matrix[i] == signature_matrix[j])
    # print("intersection:", intersection)
    # print("dividing by", signature_matrix.shape[1])
    res = intersection / signature_matrix.shape[1]
    # print("retunring", res)
    return res


def euclidean_distances(Y):
    Q = np.einsum("ij,ij->i", Y, Y)[:, np.newaxis]
    distances = -2 * Y @ Y . T
    distances += Q
    distances += Q.T
    np.maximum(distances, 1e-10, out=distances)
    return np.sqrt(distances)


def calculate_loss3(X, D):
    distances = euclidean_distances(X)
    loss = np.sum((D - distances) ** 2) / 2

    return loss
