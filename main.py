from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import smacof
from data_fetcher import fetch_dataset
from quality_assessment import D_corr, eval_dr_quality_Rnx
from optimisers import SQuaD_MDS, SQuaD_MDS_tsne, precompile


def euclidean_distances(Y):
    Q = np.einsum("ij,ij->i", Y, Y)[:, np.newaxis]
    distances = -2 * Y @ Y . T
    distances += Q
    distances += Q.T
    np.maximum(distances, 1e-10, out=distances)
    return np.sqrt(distances)


def calculate_loss(X, D):
    distances = euclidean_distances(X)
    loss = np.sum((D - distances) ** 2) / 2

    return loss


print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
precompile()

# ~~~~~~~~~~~~~~~~~~  loading the data. As highlighted in the paper, out method distinguishes itself mostly when the dimensionality is high ~~~~~~~~~~~~~~~~~~
datasets = ['coil20', 'wine', 'airfoil', 'RNAseq']
Xhd, Y = fetch_dataset(datasets[3])

print("DATA EST SIZE:", Xhd.shape)

# ~~~~~~~~~~~~~~~~~~  PCA initialisation + scaling to get a std around 10 ~~~~~~~~~~~~~~~~~~
# having an initial embedding in another scale would require some fiddling with the learning rate to adjust
Xld = PCA(n_components=2, whiten=True, copy=True).fit_transform(
    Xhd).astype(np.float64)
Xld *= 10/np.std(Xld)

# ~~~~~~~~~~~~~~~~~~  basic SQuaD_MDS algorithm ~~~~~~~~~~~~~~~~~~
hparams = {
    # other option: "relative rbf distance" --> see the code of SQuaD_MDS() in optimsers.py or the paper for a description
    'metric': 'euclidean',
    'n iter': 10000,              # 1000 is plenty if initialised with PCA
    'LR': 550,                   # values between 50 and 1500 tend to be reasonable when initialised with an std around 10. smaller values are better if randomly initialised
    # use squared distances for the first part of the optimisation
    'exaggerate D': True,
    # when to stop the exaggeration in terms of percentage of 'n iter'
    'stop exaggeration': 0.6
}
Xld_SQuaD_MDS = Xld.copy()
distance_matrix = pairwise_distances(Xhd)
tic = time.time()
SQuaD_MDS(hparams, distance_matrix, Xld_SQuaD_MDS)
time_SQuaD_MDS = time.time() - tic
print('basic SQuaD_MDS done in ', np.round(time_SQuaD_MDS, 3), 's')
print("LOSS:", calculate_loss(Xld_SQuaD_MDS, distance_matrix))


# SKLEARN MDS

distance_matrix = pairwise_distances(Xhd)
tic = time.time()
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
embedding = mds.fit_transform(distance_matrix)
time_sklearn = time.time() - tic
print('sklearn smacof mds in ', np.round(time_sklearn, 3), 's')
print("LOSS:", calculate_loss(embedding, distance_matrix))


# ~~~~~~~~~~~~~~~~~~  SQuaD_MDS algorithm with nonlinear rbf-like transformation of HD distances (as described in the paper or in the code for SQuaD_MDS() in optimsers.py ~~~~~~~~~~~~~~~~~~
hparams = {
    'metric': 'relative rbf distance',
    'n iter': 1000,
    'LR': 550,
    'exaggerate D': True,
    'stop exaggeration': 0.6
}
Xld_SQuaD_MDS_rbf = Xld.copy()
tic = time.time()
SQuaD_MDS(hparams, pairwise_distances(Xhd), Xld_SQuaD_MDS_rbf)
time_SQuaD_MDS_rbf = time.time() - tic
print('SQuaD_MDS-rbf done in ', np.round(time_SQuaD_MDS_rbf, 3), 's')
print("LOSS:", calculate_loss(Xld_SQuaD_MDS_rbf, distance_matrix))

# ~~~~~~~~~~~~~~~~~~  MDS with SMACOF, with PCA initialiation ~~~~~~~~~~~~~~~~~~
tic = time.time()
Xld_SMACOF = smacof(pairwise_distances(Xhd), n_components=2,
                    init=Xld.copy(), n_init=1)[0]
time_SMACOF = time.time() - tic
print('SMACOF MDS (with PCA initialiation) done in ',
      np.round(time_SMACOF, 3), 's')
print("LOSS:", calculate_loss(Xld_SMACOF, distance_matrix))


'''
t-SNE hybrids: t-SNE gradients are added to the distance scaling gradients according to a certain weight given as hyperparameter.
Theses hybrid versions are particularly sensitive to changes in hyperparameters
'''
# ~~~~~~~~~~~~~~~~~~  hybrid SQuaD_MDS algorithm with some light tSNE gradients ~~~~~~~~~~~~~~~~~~
hparams = {
    'n iter': 1000,
    'LR': 550,
    'exaggerate D': True,
    'stop exaggeration': 0.6,
    'tsne LR multiplier': 0.5,
    'PP': 5.,
    'tsne exa': 2.
}
Xld_SQuaD_MDS_light_tSNE = Xld.copy()
tic = time.time()
SQuaD_MDS_tsne(hparams, Xhd, Xld_SQuaD_MDS_light_tSNE)
time_SQuaD_MDS_light_tSNE = time.time() - tic
print('hybrid SQuaD_MDS + light tSNE done in ',
      np.round(time_SQuaD_MDS_light_tSNE, 3), 's')
print("LOSS:", calculate_loss(Xld_SQuaD_MDS_light_tSNE, distance_matrix))

# ~~~~~~~~~~~~~~~~~~  hybrid SQuaD_MDS algorithm with some strong tSNE gradient ~~~~~~~~~~~~~~~~~~
hparams = {
    'n iter': 1000,
    # 'LR': 6000, # Higher learning rates tend to be better when mixing with tSNE gardients (but it might bring instabilities if it's really high)
    'LR': 2000,
    'exaggerate D': True,
    'stop exaggeration': 0.6,
    'tsne LR multiplier': 5.,
    'PP': 14.,
    'tsne exa': 2.
}
Xld_SQuaD_MDS_strong_tSNE = Xld.copy()
tic = time.time()
SQuaD_MDS_tsne(hparams, Xhd, Xld_SQuaD_MDS_strong_tSNE)
time_SQuaD_MDS_strong_tSNE = time.time() - tic
print('hybrid SQuaD_MDS + strong tSNE done in ',
      np.round(time_SQuaD_MDS_strong_tSNE, 3), 's')
print("LOSS:", calculate_loss(Xld_SQuaD_MDS_strong_tSNE, distance_matrix))

# ~~~~~~~~~~~~~~~~~~  tSNE ~~~~~~~~~~~~~~~~~~
tic = time.time()
Xld_tsne = TSNE(n_components=2).fit_transform(Xhd)
time_tsne = time.time() - tic
print('t-SNE done in ', np.round(time_tsne, 3), 's')
print("LOSS:", calculate_loss(Xld_tsne, distance_matrix))


# ~~~~~~~~~~~~~~~~~~ distance correlations ~~~~~~~~~~~~~~~~~~
D_corr_PCA = D_corr(Xhd, Xld)
D_corr_SQuaD_MDS = D_corr(Xhd, Xld_SQuaD_MDS)
D_corr_SQuaD_MDS_rbf = D_corr(Xhd, Xld_SQuaD_MDS_rbf)
D_corr_hybrid_SQuaD_MDS_light_tSNE = D_corr(Xhd, Xld_SQuaD_MDS_light_tSNE)
D_corr_hybrid_SQuaD_MDS_strong_tSNE = D_corr(Xhd, Xld_SQuaD_MDS_strong_tSNE)
D_corr_SMACOF = D_corr(Xhd, Xld_SMACOF)
D_corr_tSNE = D_corr(Xhd, Xld_tsne)

# ~~~~~~~~~~~~~~~~~~ AUC of Rnx(K) curves ~~~~~~~~~~~~~~~~~~
_, AUC_light_tSNE = eval_dr_quality_Rnx(pairwise_distances(
    Xhd), pairwise_distances(Xld_SQuaD_MDS_light_tSNE))
_, AUC_strong_tSNE = eval_dr_quality_Rnx(pairwise_distances(
    Xhd), pairwise_distances(Xld_SQuaD_MDS_strong_tSNE))
_, AUC_tSNE = eval_dr_quality_Rnx(
    pairwise_distances(Xhd), pairwise_distances(Xld_tsne))

print("\n\n   correlation of distances HD:LD")
print("PCA                            : " + str(D_corr_PCA))
print("SMACOF MDS (PCA initialiation) : " + str(D_corr_SMACOF))
print("SQuaD MDS                      : " + str(D_corr_SQuaD_MDS))
print("SQuaD MDS-rbf                  : " + str(D_corr_SQuaD_MDS_rbf))
print("hybrid SQuaD MDS + light tSNE  : " + str(D_corr_hybrid_SQuaD_MDS_light_tSNE) +
      '   (Rnx(K) AUC : ', str(AUC_light_tSNE), ')')
print("hybrid SQuaD MDS + strong tSNE : " + str(D_corr_hybrid_SQuaD_MDS_strong_tSNE) +
      '   (Rnx(K) AUC : ', str(AUC_strong_tSNE), ')')
print("t-SNE                          : " + str(D_corr_tSNE) +
      '   (Rnx(K) AUC : ', str(AUC_tSNE), ')')


# ~~~~~~~~~~~~~~~~~~ plotting the embeddings ~~~~~~~~~~~~~~~~~~

fig, subs = plt.subplots(3, 3)

subs[0, 0].scatter(Xld[:, 0], Xld[:, 1], c=Y, s=1)
subs[0, 0].set_title("PCA")
subs[0, 1].scatter(Xld_SMACOF[:, 0], Xld_SMACOF[:, 1], c=Y, s=1)
subs[0, 1].set_title("SMACOF")
subs[0, 2].scatter(Xld_SQuaD_MDS[:, 0], Xld_SQuaD_MDS[:, 1], c=Y, s=1)
subs[0, 2].set_title("SQuaD MDS")
subs[1, 0].scatter(Xld_SQuaD_MDS_rbf[:, 0], Xld_SQuaD_MDS_rbf[:, 1], c=Y, s=1)
subs[1, 0].set_title("SQuaD MDS-rbf")
subs[1, 1].scatter(Xld_SQuaD_MDS_light_tSNE[:, 0],
                   Xld_SQuaD_MDS_light_tSNE[:, 1], c=Y, s=1)
subs[1, 1].set_title("SQuaD MDS + light t-SNE")
subs[1, 2].scatter(Xld_SQuaD_MDS_strong_tSNE[:, 0],
                   Xld_SQuaD_MDS_strong_tSNE[:, 1], c=Y, s=1)
subs[1, 2].set_title("SQuaD MDS + strong t-SNE")
subs[2, 0].scatter(Xld_tsne[:, 0], Xld_tsne[:, 1], c=Y, s=1)
subs[2, 0].set_title("t-SNE")

plt.show()


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
