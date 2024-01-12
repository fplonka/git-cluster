from numba import prange
import numpy as np
import numba
from utils import jaccard_fast


@numba.jit(nopython=True, fastmath=True, parallel=True)
def do_batch_with_commits_arr(N, coords, commits_arr, learning_rate, epsilon):
    i = np.random.randint(0, N)
    xi = coords[i]

    for l in prange(N):
        if i == l:
            continue
        xl = coords[l]
        dij = np.linalg.norm(xi - xl)
        rij = 1 - (jaccard_fast(commits_arr[i], commits_arr[l]))**0.5
        update_factor = (xi - xl) * learning_rate * \
            (rij - dij) / (dij + epsilon)

        coords[l] -= update_factor


@numba.jit(nopython=True, fastmath=True, parallel=True)
def do_batch_with_dist_matrix(N, coords, distance_matrix, learning_rate, epsilon):
    i = np.random.randint(0, N)
    xi = coords[i]

    for l in prange(N):
        if i == l:
            continue
        xl = coords[l]
        dij = np.linalg.norm(xi - xl)
        rij = distance_matrix[i, l]
        update_factor = (xi - xl) * learning_rate * \
            (rij - dij) / (dij + epsilon)

        coords[l] -= update_factor


@numba.jit(nopython=True, fastmath=True, inline='always')
def sgdr_schedule(lr_max, lr_min, T_cur, n_iter):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * T_cur / n_iter))


@numba.jit(nopython=True, parallel=True, fastmath=True)
def spe_with_commits_arr(commits_arr, n_iter, lr, lr_final, epsilon=1e-8):
    N = len(commits_arr)
    learning_rate = lr
    lr_initial = lr
    coords = np.random.rand(N, 2) - 0.5

    for iter_idx in range(n_iter):
        # learning_rate = lr_initial + (lr_final - lr_initial)/n_iter * iter_idx
        learning_rate = sgdr_schedule(lr_initial, lr_final, iter_idx, n_iter)

        do_batch_with_commits_arr(
            N, coords, commits_arr, learning_rate, epsilon)

        if iter_idx % 1000 == 0:
            print("\rat", (iter_idx + 1)/n_iter * 100, "%")

    return coords


@numba.jit(nopython=True, parallel=True, fastmath=True)
def spe_with_dist_matrix(distance_matrix, n_iter, lr, lr_final, epsilon=1e-8):
    N = distance_matrix.shape[0]
    lr_initial = lr
    coords = np.random.rand(N, 2) - 0.5

    for iter_idx in range(n_iter):
        learning_rate = sgdr_schedule(lr_initial, lr_final, iter_idx, n_iter)
        # learning_rate = lr_initial + (lr_final - lr_initial) / n_iter * iter_idx

        do_batch_with_dist_matrix(
            N, coords, distance_matrix, learning_rate, epsilon)
        if iter_idx % 1000 == 0:
            print("\rat", (iter_idx + 1)/n_iter * 100, "%")

    return coords


@numba.jit(nopython=True, parallel=True, fastmath=True)
def spe_with_commits_arr_animated(commits_arr, n_iter, lr, lr_final, num_frames, epsilon=1e-8):
    N = len(commits_arr)
    learning_rate = lr
    lr_initial = lr
    coords = np.random.rand(N, 2) - 0.5

    # Determine the interval at which to record the state of the embedding
    record_interval = max(1, n_iter // num_frames)

    # Create a 3D array to store the coordinates at different iterations
    all_coords = np.zeros((num_frames, N, 2))

    frame_idx = 0  # Index for recording frames
    for iter_idx in range(n_iter):
        if iter_idx % record_interval == 0:
            if frame_idx < num_frames:
                all_coords[frame_idx] = coords
                frame_idx += 1

        learning_rate = sgdr_schedule(lr_initial, lr_final, iter_idx, n_iter)

        do_batch_with_commits_arr(
            N, coords, commits_arr, learning_rate, epsilon)

        if iter_idx % 1000 == 0:
            print("\rat", (iter_idx + 1) / n_iter * 100, "%")

    return all_coords
