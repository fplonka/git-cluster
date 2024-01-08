import numpy as np
import numba
from utils import calculate_loss
from numba import types
from numba.typed import List


@numba.jit(nopython=True, fastmath=True)
def calculate_force(positions, distance_matrix, i, neighbors, random_subset):
    N = positions.shape[0]
    force = np.zeros(2)
    for j in neighbors:
        pos_diff = positions[i] - positions[j]
        layout_dist = np.linalg.norm(pos_diff)
        force_magnitude = distance_matrix[i, j] - layout_dist
        if layout_dist != 0:
            force += (force_magnitude / layout_dist) * pos_diff

    for j in random_subset:
        pos_diff = positions[i] - positions[j]
        layout_dist = np.linalg.norm(pos_diff)
        force_magnitude = distance_matrix[i, j] - layout_dist
        if layout_dist != 0:
            force += (force_magnitude / layout_dist) * pos_diff

    return force


@numba.jit(nopython=True, fastmath=True)
def update_neighbors(distance_matrix, max_distances, neighbors, i, j, Vmax):
    dij = distance_matrix[i, j]
    if dij < max_distances[i]:
        # Insert j in the sorted position in neighbors[i]
        inserted = False
        for k in range(len(neighbors[i])):
            if dij < distance_matrix[i, neighbors[i][k]]:
                neighbors[i].insert(k, j)
                inserted = True
                break
        if not inserted:
            neighbors[i].append(j)

        if len(neighbors[i]) > Vmax:
            neighbors[i].pop()

        max_distances[i] = max([distance_matrix[i, x] for x in neighbors[i]])


@numba.jit(nopython=True, fastmath=True)
def spring_mds(distance_matrix, iterations=5000, Vmax=20, Smax=20):
    N = distance_matrix.shape[0]
    positions = np.random.rand(N, 2)  # Initialize random positions
    max_distances = np.full(N, np.inf)
    neighbors = List([List.empty_list(types.int32) for _ in range(N)])

    for iter_idx in range(iterations):
        for i in range(N):
            # Initialize random_subset and candidate list
            random_subset = List.empty_list(types.int32)
            candidates = [x for x in range(
                N) if x != i and x not in neighbors[i]]

            while len(random_subset) < Smax:
                j = np.random.randint(0, N)
                if j != i and j not in neighbors[i]:
                    if distance_matrix[i, j] < max_distances[i]:
                        update_neighbors(
                            distance_matrix, max_distances, neighbors, i, j, Vmax)
                    else:
                        random_subset.append(j)

            # Calculate force for object i
            force = calculate_force(
                positions, distance_matrix, i, neighbors[i], random_subset)

            # Update position
            positions[i] += force * 0.05

        # Optional: Print progress or loss
        print("stress is", calculate_loss(positions, distance_matrix))
        print("Progress:", ((iter_idx+1)/iterations * 100), "%")

    return positions
