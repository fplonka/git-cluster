from numba import prange
import numpy as np
import numba
from utils import jaccard_fast
from utils import calculate_loss, estimated_jaccard


@numba.jit(nopython=True, fastmath=True)
def spe_fancy(commits_arr, n_iter, lr, lr_final, epsilon=1e-8):
    """
    Perform dimensionality reduction on a distance matrix.

    :param distance_matrix: Input NxN distance matrix.
    :param S: Number of steps for each cycle.
    :param C: Number of cycles.
    :param lambda_init: Initial learning rate.
    :param delta_lambda: Decrement of learning rate after each cycle.
    :param epsilon: Small number to avoid division by zero.
    :return: Nx2 array of coordinates after dimensionality reduction.
    """
    N = len(commits_arr)
    coords = np.random.rand(N, 2)  # Initialize coordinates randomly
    learning_rate = lr
    decay = np.power(lr_final/lr, 1/n_iter)

    print("decay is", decay)

    for iter_idx in range(n_iter):
        # Pick two distinct points at random
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        while i == j:
            j = np.random.randint(0, N)

        xi, xj = coords[i], coords[j]

        # Compute current distance and update coordinates
        dij = np.linalg.norm(xi - xj)
        rij = 1 - (jaccard_fast(commits_arr[i], commits_arr[j]))**0.25
        update_factor = learning_rate * (rij - dij) / (dij + epsilon)
        coords[i] += update_factor * (xi - xj)
        coords[j] += update_factor * (xj - xi)

        learning_rate *= decay

        if iter_idx % 10000 == 0:
            print("\rat", (iter_idx + 1)/n_iter * 100, "%")

    print()
    return coords


@numba.jit(nopython=True, parallel=True, fastmath=True)
def spe_optimized(distance_matrix, n_iter, lr, lr_final, batch_size=50, epsilon=1e-8):
    N = distance_matrix.shape[0]
    learning_rate = lr
    decay = np.power(lr_final / lr, 1 / n_iter)
    coords = np.random.rand(N, 2)

    # Precompute as much as possible
    batched_iters = n_iter // batch_size
    indices = np.random.randint(0, N, size=(2, batch_size, batched_iters))

    for batch in prange(batched_iters):
        for k in range(batch_size):
            i, j = indices[:, k, batch]

            if i != j:
                # Local copy to minimize memory access
                xi, xj = coords[i].copy(), coords[j].copy()
                dij = np.linalg.norm(xi - xj)
                update_factor = learning_rate * \
                    (distance_matrix[i, j] - dij) / (dij + epsilon)

                # Update coords
                coords[i] += update_factor * (xi - xj)
                coords[j] -= update_factor * (xi - xj)

        learning_rate *= decay  # Update learning rate less frequently

    return coords


@numba.jit(nopython=True, parallel=True, fastmath=True, debug=True)
def spe_optimized_parallel(distance_matrix, n_iter, lr, lr_final, n_threads=16, epsilon=1e-8):
    N = distance_matrix.shape[0]
    learning_rate = lr
    decay = np.power(lr_final / lr, 1 / n_iter)
    coords = np.random.rand(N, 2)

    # Number of pairs each thread will process
    pairs_per_thread = N // n_threads

    for iter_idx in range(n_iter):
        # Generate a random permutation of indices and reshape
        # indices = np.arange(N)
        print("idx:", iter_idx)
        indices = np.random.permutation(N)
        pairs = indices[:pairs_per_thread *
                        n_threads].reshape(n_threads, pairs_per_thread)

        print('starting prange bs')
        for j in prange(n_threads):
            # Process each pair of points for this thread
            for k in range(0, pairs_per_thread, 2):
                i, l = pairs[j][k], pairs[j][k + 1]

                xi, xl = coords[i].copy(), coords[l].copy()
                dij = np.linalg.norm(xi - xl)
                update_factor = (xi - xl) * learning_rate * \
                    (distance_matrix[i, l] - dij) / (dij + epsilon)

                # In-place update
                coords[i] += update_factor
                coords[l] -= update_factor

        print('prange bs done')
        learning_rate *= decay

        if iter_idx % 1000 == 0:
            print("at", (iter_idx + 1)/n_iter * 100, "%")

    return coords


@numba.jit(nopython=True, parallel=True, fastmath=True)
def foo(N, pairs_per_thread, n_threads, coords, commits_arr, learning_rate, epsilon):
    indices = np.random.permutation(N)
    pairs = indices[:pairs_per_thread *
                    n_threads].reshape(n_threads, pairs_per_thread)

    for j in prange(n_threads):
        # Process each pair of points for this thread
        for k in range(0, pairs_per_thread-1, 2):
            i, l = pairs[j][k], pairs[j][k + 1]
            xi, xl = coords[i], coords[l]
            dij = np.linalg.norm(xi - xl)
            rij = 1 - (jaccard_fast(commits_arr[i], commits_arr[l]))**0.5
            # rij = 1 - (estimated_jaccard(commits_arr, i, l))**0.5
            update_factor = (xi - xl) * learning_rate * \
                (rij - dij) / (dij + epsilon)

            coords[i] += update_factor
            coords[l] -= update_factor


@numba.jit(nopython=True, fastmath=True, parallel=True)
def foo2(N, coords, commits_arr, learning_rate, epsilon):
    i = np.random.randint(0, N)
    xi = coords[i]

    for l in prange(N):
        if i == l:
            continue
        xl = coords[l]
        dij = np.linalg.norm(xi - xl)
        rij = 1 - (jaccard_fast(commits_arr[i], commits_arr[l]))**0.5
        # rij = 1 - estimated_jaccard(commits_arr, i, l)**0.5
        update_factor = (xi - xl) * learning_rate * \
            (rij - dij) / (dij + epsilon)

        # coords[i] += update_factor
        coords[l] -= update_factor


@numba.jit(nopython=True, fastmath=True, parallel=True)
def foo3(N, coords, distance_matrix, learning_rate, epsilon):
    i = np.random.randint(0, N)
    xi = coords[i]

    for l in prange(N):
        if i == l:
            continue
        xl = coords[l]
        dij = np.linalg.norm(xi - xl)
        rij = 1 - (distance_matrix[i, l])**0.5
        # rij = 1 - estimated_jaccard(commits_arr, i, l)**0.5
        update_factor = (xi - xl) * learning_rate * \
            (rij - dij) / (dij + epsilon)

        # coords[i] += update_factor
        coords[l] -= update_factor


@numba.jit(nopython=True, fastmath=True)
def sgdr_schedule(lr_max, lr_min, T_cur, T_i):
    """
    Calculate the learning rate at a given iteration using SGDR.

    :param t: Current iteration.
    :param lr_max: Maximum learning rate (start of cycle).
    :param lr_min: Minimum learning rate (end of cycle).
    :param T_cur: Current number of iterations since the last restart.
    :param T_i: Number of iterations in the current cycle.
    :return: Learning rate for iteration t.
    """
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * T_cur / T_i))


@numba.jit(nopython=True, parallel=True, fastmath=True)
def spe_optimized_parallel_fancy(commits_arr, n_iter, lr, lr_final, n_threads=16, epsilon=1e-8):
    N = len(commits_arr)
    learning_rate = lr
    lr_initial = lr
    # decay = np.power(lr_final / lr, 1 / n_iter)
    coords = np.random.rand(N, 2) - 0.5

    # Number of pairs each thread will process
    pairs_per_thread = N // n_threads

    T_0 = 1000  # Initial number of iterations in the first cycle
    T_mult = 2  # Factor by which to increase the number of iterations in each cycle
    T_cur = 0
    T_i = T_0

    for iter_idx in range(n_iter):

        learning_rate = sgdr_schedule(lr_initial, lr_final, T_cur, T_i)
        T_cur += 1

        # # foo(N, pairs_per_thread, n_threads, coords,
        #     commits_arr, learning_rate, epsilon)
        foo2(N, coords, commits_arr, learning_rate, epsilon)
        if iter_idx % 1000 == 0:
            print("\rat", (iter_idx + 1)/n_iter * 100, "%")

        if T_cur == T_i:
            # End of current cycle, prepare for the next cycle
            T_cur = 0
            T_i *= T_mult  # Increase the length of the next cycle

        # if iter_idx % (n_iter // 100) == 0:
        #     loss = calculate_loss(coords, distance_matrix)
        #     losses[iter_idx // (n_iter // 100)] = loss
        #     indices[iter_idx // (n_iter // 100)] = iter_idx
        #     print("loss is", loss)

    # return coords, losses, indices
    return coords


@numba.jit(nopython=True, parallel=True, fastmath=True)
def spe_optimized_parallel_dist(distance_matrix, n_iter, lr, lr_final, n_threads=16, epsilon=1e-8):
    N = distance_matrix.shape[0]
    learning_rate = lr
    lr_initial = lr
    coords = np.random.rand(N, 2) - 0.5

    T_0 = 1000  # Initial number of iterations in the first cycle
    T_mult = 2  # Factor by which to increase the number of iterations in each cycle
    T_cur = 0
    T_i = T_0

    for iter_idx in range(n_iter):

        learning_rate = sgdr_schedule(lr_initial, lr_final, T_cur, T_i)
        T_cur += 1

        foo3(N, coords, distance_matrix, learning_rate, epsilon)
        if iter_idx % 1000 == 0:
            print("\rat", (iter_idx + 1)/n_iter * 100, "%")

        if T_cur == T_i:
            T_cur = 0
            T_i *= T_mult  # Increase the length of the next cycle

    return coords
