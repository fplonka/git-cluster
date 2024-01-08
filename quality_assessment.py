import numpy as np
import numba
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

def D_corr(Xhd, Xld):
    return np.round(pearsonr(pdist(Xhd), pdist(Xld))[0],3)



"""
The following four functions that compute the Rnx(K) curve and its AUC were taken from C. de Bodt's github page at:
https://github.com/cdebodt/Multi-scale_t-SNE/blob/main/mstSNE.py
"""

def coranking(d_hd, d_ld):
    """
    Computation of the co-ranking matrix, as described in [4].
    The time complexity of this function is O(N**2 log(N)), where N is the number of data points.
    In:
    - d_hd: 2-D numpy array representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array representing the redundant matrix of pairwise distances in the LDS.
    Out:
    The (N-1)x(N-1) co-ranking matrix, where N = d_hd.shape[0].
    """
    # Computing the permutations to sort the rows of the distance matrices in HDS and LDS.
    perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
    perm_ld = d_ld.argsort(axis=-1, kind='mergesort')

    N = d_hd.shape[0]
    i = np.arange(N, dtype=np.int64)
    # Computing the ranks in the LDS
    R = np.empty(shape=(N,N), dtype=np.int64)
    for j in range(N):
        R[perm_ld[j,i],j] = i
    # Computing the co-ranking matrix
    Q = np.zeros(shape=(N,N), dtype=np.int64)
    for j in range(N):
        Q[i,R[perm_hd[j,i],j]] += 1
    # Returning
    return Q[1:,1:]

@numba.jit(nopython=True)
def eval_auc(arr):
    """
    Evaluates the AUC, as defined in [2].
    In:
    - arr: 1-D numpy array storing the values of a curve from K=1 to arr.size.
    Out:
    The AUC under arr, as defined in [2], with a log scale for K=1 to arr.size.
    """
    i_all_k = 1.0/(np.arange(arr.size)+1.0)
    return np.round(np.float64(np.dot(arr, i_all_k))/(i_all_k.sum()), 3)

@numba.jit(nopython=True)
def eval_rnx(Q):
    """
    Evaluate R_NX(K) for K = 1 to N-2, as defined in [5]. N is the number of data points in the data set.
    The time complexity of this function is O(N^2).
    In:
    - Q: a 2-D numpy array representing the (N-1)x(N-1) co-ranking matrix of the embedding.
    Out:
    A 1-D numpy array with N-2 elements. Element i contains R_NX(i+1).
    """
    N_1 = Q.shape[0]
    N = N_1 + 1
    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += (Q[K,K] + np.sum(Q[K,:K]) + np.sum(Q[:K,K]))
        qnxk[K] = acc_q/((K+1)*N)
    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1*qnxk[:N_1-1]-arr_K)/(N_1-arr_K)
    # Returning
    return rnxk

def eval_dr_quality_Rnx(d_hd, d_ld):
    """
    Compute the DR quality assessment criteria R_{NX}(K) and AUC, as defined in [2, 3, 4, 5] and as employed in the experiments reported in [1, 7].
    These criteria measure the neighborhood preservation around the data points from the HDS to the LDS.
    Based on the HD and LD distances, the sets v_i^K (resp. n_i^K) of the K nearest neighbors of data point i in the HDS (resp. LDS) can first be computed.
    Their average normalized agreement develops as Q_{NX}(K) = (1/N) * \sum_{i=1}^{N} |v_i^K \cap n_i^K|/K, where N refers to the number of data points and \cap to the set intersection operator.
    Q_{NX}(K) ranges between 0 and 1; the closer to 1, the better.
    As the expectation of Q_{NX}(K) with random LD coordinates is equal to K/(N-1), which is increasing with K, R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K) enables more easily comparing different neighborhood sizes K.
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1.
    The R_{NX}(K) values for K=1 to N-2 can be displayed as a curve with a log scale for K, as closer neighbors typically prevail.
    The area under the resulting curve (AUC) is a scalar score which grows with DR quality, quantified at all scales with an emphasis on small ones.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random.
    In:
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    Out: a tuple with
    - a 1-D numpy array with N-2 elements. Element i contains R_{NX}(i+1).
    - the AUC of the R_{NX}(K) curve with a log scale for K, as defined in [2].
    Remark:
    - The time complexity to evaluate the quality criteria is O(N**2 log(N)). It is the time complexity to compute the co-ranking matrix. R_{NX}(K) can then be evaluated for all K=1, ..., N-2 in O(N**2).
    """
    # Computing the co-ranking matrix of the embedding, and the R_{NX}(K) curve.
    rnxk = eval_rnx(Q=coranking(d_hd=d_hd, d_ld=d_ld))
    # Computing the AUC, and returning.
    return rnxk, eval_auc(rnxk)
