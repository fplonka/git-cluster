from math import e
import numpy as np


def euclidean_distances(Y):
    Q = np.einsum("ij,ij->i", Y, Y)[:, np.newaxis]
    distances = -2 * Y @ Y . T
    distances += Q
    distances += Q.T
    np.maximum(distances, 1e-10, out=distances)
    # return distances
    return np.sqrt(distances)

# dm is a symmetric m X m matrix of proximities between n-dimensional data points;
# Y is m X d matrix of the projected points (initial configuration of points);
# Y_new is matrix of size m X d of new positions of projected points.


def correction2(dm, dist, Y):
    B = -np.divide(dm, dist)
    B[np.arange(len(B)), np.arange(len(B))] -= B.sum(axis=1)
    Y_new = 1 / len(Y) * np.dot(B, Y)
    return Y_new

# dm is a symmetric m x m matrix of proximities between n-dimensional data points;
# Y is m x d matrix of the projecting points (initial configuration of points);
# eps refers to the accuracy of the final stress value;
# max_it is a maximum number of iterations;
# dist is a distance m x m matrix of points Y;


def run(dm, initial_Y=None, max_it=200):
    S_old, it = np.inf, 0

    # If an initial Y is provided, use it; otherwise, generate a new one
    if initial_Y is not None:
        Y = initial_Y
    else:
        Y = np.random.rand(dm.shape[0], 2)

    np.fill_diagonal(dm, 1e-5)

    while it < max_it:
        dist = euclidean_distances(Y)
        Y = correction2(dm, dist, Y)
        it += 1

    S = np.sum((dm - dist) ** 2) / 2
    return Y, S


def GMDSm(dm, short_runs=4, short_run_iters=10, long_run_iters=300):
    best_S = np.inf
    best_initial_Y = None

    # Short runs to find the best initial Y
    for _ in range(short_runs):
        Y, S = run(dm, max_it=short_run_iters)
        print("got loss", S)
        if S < best_S:
            best_S = S
            best_initial_Y = Y

    # Long run starting from the best initial Y found
    final_Y, final_S = run(dm, initial_Y=best_initial_Y, max_it=long_run_iters)

    return final_Y
