import graph_tool.all as gt
import numba
from utils import calculate_loss
from utils import log_space_values
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from springs import spring_mds
import gc
import numpy as np
import random
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import pickle
import argparse
import os
import webbrowser
from urllib.parse import urlparse
from sklearn.manifold import MDS
import numpy as np
import subprocess
import time
from optimisers import SQuaD_MDS, SQuaD_MDS_no_precompute
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD

from utils import calculate_correlation, jaccard_fast

from spe import spe


def run_git_command(repo_path, command):
    return subprocess.check_output(["git", "-C", repo_path] + command).decode().strip().split('\n')


def get_current_files(repo_path):
    return set(run_git_command(repo_path, ["ls-tree", "-r", "HEAD", "--name-only"]))


def get_commits(repo_path):
    commit_data = run_git_command(
        repo_path, ["log", "--pretty=format:__commit__:%H", "--name-only"])
    commits = {}
    current_commit = None
    for line in commit_data:
        if line.startswith("__commit__:"):
            current_commit = line.split("__commit__:")[1]
        elif line:
            if line not in commits:
                commits[line] = set()
            commits[line].add(current_commit)
    return commits


def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union != 0 else 0


@numba.jit(nopython=True)
def create_distance_matrix_fast(commits_arr):
    N = len(commits_arr)
    matrix = np.ones((N, N))

    for i in range(N):
        for j in range(i+1, N):
            score = jaccard_fast(commits_arr[i], commits_arr[j])
            matrix[i][j] = 1 - score + 0.00001
            matrix[j][i] = 1 - score + 0.00001

        with numba.objmode():
            # print(f"\rAt ({((i+1)/N):.2f}%)", end="")
            print("\rCalculating distance matrix... ({:.2f})%".format(
                (i+1)/N * 100), end="")

    print()

    return matrix


def create_similarity_matrix(commits, file_names, progress_text):
    N = len(file_names)

    commits_arr = dict_to_padded_array(commits)

    matrix = np.zeros((N, N))

    for i, file1 in enumerate(file_names):
        for j, file2 in enumerate(file_names):
            # Calculate score only if it's not a diagonal element (i.e., file1 != file2)
            score = jaccard_index(
                commits[file1], commits[file2]) if i != j else 1
            # score = jaccard_fast(commits_arr[i], commits_arr[j])
            matrix[i][j] = score

        print_progress(progress_text, (i + 1) / N * 100)

    # Clear the progress percentage text once done
    print(f"\r{progress_text}...          ")

    return matrix


# def euclidean_distances(Y):
#     Q = np.einsum("ij,ij->i", Y, Y)[:, np.newaxis]
#     distances = -2 * Y @ Y . T
#     distances += Q
#     distances += Q.T
#     np.maximum(distances, 1e-10, out=distances)
#     # return distances
#     return np.sqrt(distances)


def compute_distance_matrix(similarity_matrix):
    # return np.ones(similarity_matrix.shape)
    return np.ones(similarity_matrix.shape) - similarity_matrix

    similarity_matrix = np.array(similarity_matrix)
    epsilon = 1e-6
    adjusted_similarity_matrix = similarity_matrix + epsilon
    distance_matrix = 1 / adjusted_similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


def get_extension(file_name):
    _, ext = os.path.splitext(file_name)
    return ext if ext else 'no_extension'


def get_extension_colors(file_names, commits):
    # Count files for each extension
    ext_count = {ext: 0 for ext in set(
        get_extension(name) for name in file_names)}
    for file in file_names:
        ext = get_extension(file)
        ext_count[ext] += len(commits[file])

    # Sort extensions by count
    sorted_extensions = sorted(ext_count, key=ext_count.get, reverse=True)

    # Use a Plotly color sequence
    color_sequence = px.colors.qualitative.Plotly

    extension_colors = {}
    for i, ext in enumerate(sorted_extensions):
        color = color_sequence[i % len(color_sequence)]
        extension_colors[ext] = color

    return extension_colors


def visualize_embedding_with_extension_color(embedding, file_names, commits, repo_name, output_file=None):
    extension_colors = get_extension_colors(file_names, commits)

    # Initialize a list to hold all traces
    traces = []

    for ext, color in extension_colors.items():
        # Filter files by extension and create a trace for each extension
        indices = [i for i, name in enumerate(
            file_names) if (get_extension(name) == ext)]
        if indices:  # Add trace only if there are files with this extension
            x_values = embedding[indices, 0]
            y_values = embedding[indices, 1]
            hovertexts = [file_names[i] for i in indices]

            trace = go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                hoverinfo='text',
                hovertext=hovertexts,
                marker=dict(size=5, color=color, opacity=0.8),
                name=ext
            )
            traces.append(trace)

    layout = go.Layout(
        title=f'File Commit-Similarity Visualization for {repo_name}',
        xaxis=dict(title='MDS Dimension 1'),
        yaxis=dict(title='MDS Dimension 2'),
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,1)'
    )

    fig = go.Figure(data=traces, layout=layout)
    if output_file:
        fig.write_html(output_file, include_plotlyjs=True)
        webbrowser.open('file://' + os.path.realpath(output_file), new=2)
    else:
        fig.show()


def scale_embedding(embedding):
    """
    Scale the embedding so that the max values for both the x and y axis are -1 and 1.

    :param embedding: A n x 2 embedding matrix.
    :return: Scaled embedding.
    """
    # Find the maximum absolute value for each axis
    max_vals = np.abs(embedding).max(axis=0)

    # Scale the embedding
    scaled_embedding = embedding / max_vals

    # Ensure the scaled values are within the range [-1, 1]
    scaled_embedding = np.clip(scaled_embedding, -1, 1)

    return scaled_embedding


def print_progress(str, progress):
    print(f"\r{str} ({progress:.2f}%)", end="")


def generate_points_in_circle(r, num_points):
    points = []

    for _ in range(num_points):
        # Generate a random point on a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(1 - 2 * np.random.uniform())

        # Convert spherical coordinates to Cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)

        # Append the projected point (x, y) to the points list
        points.append((x, y))

    return np.array(points)


def dict_to_padded_array(commit_dict):
    # Step 1: Create a mapping for commit hashes
    unique_commits = set.union(*map(set, commit_dict.values()))
    commit_to_id = {commit: i for i, commit in enumerate(unique_commits)}

    # Step 2: Find the maximum set size
    max_set_size = max(len(s) for s in commit_dict.values())

    print("MAX SIZE IS", max_set_size)

    # Step 3: Initialize a 2D array with -1
    padded_array = np.iinfo(np.int32).max * \
        np.ones((len(commit_dict), max_set_size), dtype=int)

    # Step 4: Fill the array
    for row, commits in enumerate(commit_dict.values()):
        padded_array[row, :len(commits)] = [commit_to_id[commit]
                                            for commit in commits]
        # Step 5: Optionally sort each row
        padded_array[row].sort()

    return padded_array


def evaluate_model(distance_matrix, beta1, beta2, lr):
    np.random.seed(42)
    hparams = {
        # other option: "relative rbf distance" --> see the code of SQuaD_MDS() in optimsers.py or the paper for a description
        'metric': 'euclidian',
        'n iter': 10000,              # 1000 is plenty if initialised with PCA
        'LR': lr,
        # use squared distances for the first part of the optimisation
        'exaggerate D': True,
        # when to stop the exaggeration in terms of percentage of 'n iter'
        'stop exaggeration': 0.75,
        'b1': beta1,
        'b2': beta2
    }
    N = distance_matrix.shape[0]

    # use LR = 1000
    embedding = (np.random.rand(N, 2) - 0.5)
    embedding *= 10/np.std(embedding)

    correlations = SQuaD_MDS(hparams, distance_matrix, embedding)

    return calculate_correlation(embedding, distance_matrix)


def evaluate_model2(distance_matrix, lr, decay):
    np.random.seed(42)
    hparams = {
        # other option: "relative rbf distance" --> see the code of SQuaD_MDS() in optimsers.py or the paper for a description
        'metric': 'euclidian',
        'n iter': 10000,              # 1000 is plenty if initialised with PCA
        'LR': lr,
        'decay': decay,
        # use squared distances for the first part of the optimisation
        'exaggerate D': True,
        # when to stop the exaggeration in terms of percentage of 'n iter'
        'stop exaggeration': 0.75,
    }
    N = distance_matrix.shape[0]

    # use LR = 1000
    embedding = (np.random.rand(N, 2) - 0.5)
    embedding *= 10/np.std(embedding)

    correlations = SQuaD_MDS(hparams, distance_matrix, embedding)

    return calculate_correlation(embedding, distance_matrix)


def find_params(distance_matrix):
    # Define the ranges for each parameter
    # beta1_values = log_space_values(0.001, 0.02, 10)
    # beta2_values = log_space_values(0.99, 0.999999999, 3)
    # lr_values = log_space_values(0.2, 2.0, 10)

    beta1_values = np.linspace(0.01/2, 0.01*2, 10)
    beta2_values = np.linspace(0.99, 0.999999999, 3)
    lr_values = np.linspace(0.6, 2.4, 10)

    print("b1", beta1_values)
    print("b2", beta2_values)
    print("lr", lr_values)

    # Grid search
    best_score = -np.inf
    best_params = {}

    cnt = 0
    for beta1 in beta1_values:
        for beta2 in beta2_values:
            for lr in lr_values:
                np.random.seed(42)
                score = evaluate_model(distance_matrix, beta1, beta2, lr)
                cnt += 1
                print_progress("evaluated", cnt / (len(beta1_values)
                               * len(beta2_values) * len(lr_values)) * 100)
                if score > best_score:
                    print("new best:", score)
                    best_score = score
                    best_params = {
                        'beta1': beta1, 'beta2': beta2, 'lr': lr}

    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")


def find_params2(distance_matrix):
    # Define the space of hyperparameters to search
    space = [
        Real(1000, 16000, name='lr'),
        Real(0.9, 1.0, name='decay')
    ]

    @use_named_args(space)
    def objective(**params):
        return -evaluate_model2(distance_matrix, params['lr'], params['decay'])

    # Run the optimization
    res_gp = gp_minimize(objective, space, n_calls=200,
                         random_state=0, verbose=True)

    # Results
    print("Best score=%.4f" % res_gp.fun)
    print("""Best parameters:
    - lr=%.6f
    - decay=%.6f""" % (res_gp.x[0], res_gp.x[1]))


def fast_mds(distance_matrix):
    hparams = {
        # other option: "relative rbf distance" --> see the code of SQuaD_MDS() in optimsers.py or the paper for a description
        'metric': 'euclidian',
        'n iter': 10000,              # 1000 is plenty if initialised with PCA
        'LR': 1000,                   # values between 50 and 1500 tend to be reasonable when initialised with an std around 10. smaller values are better if randomly initialised
        # use squared distances for the first part of the optimisation
        'exaggerate D': True,
        # when to stop the exaggeration in terms of percentage of 'n iter'
        'stop exaggeration': 0.75
    }
    N = distance_matrix.shape[0]

    # use LR = 1000
    embedding = (np.random.rand(N, 2) - 0.5)
    embedding *= 10/np.std(embedding)

    correlations = SQuaD_MDS(hparams, distance_matrix, embedding)

    return embedding, correlations


def create_graph_embedding(distance_matrix):
    g = gt.Graph(directed=False)
    g.add_vertex(distance_matrix.shape[0])
    weight = g.new_edge_property("double")
    edge_cnt = 0
    for i in range(distance_matrix.shape[0]):
        # Avoid duplicates in undirected graph
        for j in range(i+1, distance_matrix.shape[1]):
            # Assuming a non-zero value means an edge exists
            # if distance_matrix[i][j] != 0:
            w = distance_matrix[i][j]
            if w != 1.00001:
                edge = g.add_edge(g.vertex(i), g.vertex(j))
                # weight[edge] = w
                edge_cnt += 1

    # g.edge_properties["weight"] = weight

    print("making layout")
    # pos = gt.sfdp_layout(g)
    pos = gt.fruchterman_reingold_layout(g)
    # pos = gt.arf_layout(g, max_iter=0)
    # pos = gt.sfdp_layout(g)

    embedding = np.array([[pos[v][0], pos[v][1]] for v in g.vertices()])

    return embedding


# @numba.jit(nopython=True, fastmath=True)
# def spring_mds(distance_matrix, iterations=10000):
#     N = distance_matrix.shape[0]
#     positions = np.random.rand(N, 2)  # Initialize random positions
#     velocities = np.zeros((N, 2))     # Initialize velocities
#     forces = np.zeros((N, 2))         # Initialize forces

#     learning_rate = 0.001
#     # final_learning_rate = 0.0001
#     # decay = np.power(final_learning_rate/learning_rate, 1/iterations)

#     for iter_idx in range(iterations):
#         for i in range(N):
#             force = np.zeros(2)
#             for j in range(i + 1, N):
#                 # Calculate the difference in positions
#                 pos_diff = positions[i, :] - positions[j, :]

#                 # Calculate the Euclidean distance in the layout
#                 layout_dist = np.linalg.norm(pos_diff)

#                 force_magnitude = distance_matrix[i, j] - layout_dist
#                 force_direction = pos_diff / (layout_dist)
#                 force += force_magnitude * force_direction

#             forces[i, :] = force
#             forces[j, :] = -force

#         velocities = forces
#         positions += learning_rate * velocities

#         # learning_rate *= decay

#         print("stress is", calculate_loss(positions, distance_matrix))
#         print("progress: ", ((iter_idx+1)/iterations * 100))
#         # print("lr:", learning_rate)
#         print()

#     return positions


def fast_mds_no_precompute(commits_arr):
    hparams = {
        # other option: "relative rbf distance" --> see the code of SQuaD_MDS() in optimsers.py or the paper for a description
        'metric': 'euclidian',
        'n iter': 10000 * 73,              # 1000 is plenty if initialised with PCA
        'LR': 1000,                   # values between 50 and 1500 tend to be reasonable when initialised with an std around 10. smaller values are better if randomly initialised
        # use squared distances for the first part of the optimisation
        'exaggerate D': True,
        # when to stop the exaggeration in terms of percentage of 'n iter'
        'stop exaggeration': 0.75
    }
    N = len(commits_arr)

    # use LR = 1000
    embedding = (np.random.rand(N, 2) - 0.5)
    embedding *= 10/np.std(embedding)

    correlations = SQuaD_MDS_no_precompute(hparams, commits_arr, embedding)

    return embedding, correlations


def process_repository(repo_path_or_url, output_file):
    np.random.seed(42)  # You can use any integer as the seed value

    is_url = repo_path_or_url.startswith(
        "http://") or repo_path_or_url.startswith("https://")
    if is_url:
        repo_name = get_repo_name_from_url(repo_path_or_url)
    else:
        repo_name = get_last_folder_name(repo_path_or_url)

    cache_path = os.path.join('cache', repo_name)

    if os.path.isfile(cache_path):
        print("Loading distance matrix from cache...")
        with open(cache_path, 'rb') as f:
            file_names, commits = pickle.load(f)
            distance_matrix = np.load(f'{cache_path}.npy')
            # file_names, commits = pickle.load(f)
        N = len(file_names)
    else:

        # Clone the repo if a URL is provided
        with tempfile.TemporaryDirectory() as temp_dir:
            if is_url:
                print("Cloning repository...")
                clone_repo(repo_path_or_url, temp_dir)
                repo_path = temp_dir
            elif os.path.exists(repo_path_or_url) and is_git_repo(repo_path_or_url):
                repo_path = repo_path_or_url
            else:
                print("Invalid repo path or URL")
                exit(1)

            print("Extracting commit information...")
            commits = get_commits(repo_path)

            # filter to current files
            current_files = get_current_files(repo_path)
            commits = {file: hashes for file, hashes in commits.items(
            ) if file in current_files and len(hashes) > 1}

        # Create distance matrix
        file_names = np.array(list(commits.keys()))
        N = len(file_names)

        commits_arr = dict_to_padded_array(commits)

        t0 = time.time()
        distance_matrix = create_distance_matrix_fast(commits_arr)
        t1 = time.time()
        print("Took", t1 - t0)

        gc.collect()
        print("done with gc")
        with open(cache_path, 'wb') as f:
            print("doing pickle dump")
            pickle.dump((file_names, commits), f)
            print("doing np save")
            np.save(f'{cache_path}.npy', distance_matrix)

    # find_params2(distance_matrix)

    print("N is", N)
    print("Creating embeddings...")
    # mds.embedding_ = generate_points_in_circle(0.8, N)
    # plt.scatter(mds.embedding_[:, 0], mds.embedding_[:, 1])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title("Uniformly Distributed Points in a Circle")
    # plt.xlabel("x-coordinate")
    # plt.ylabel("y-coordinate")
    # plt.show()
    t0 = time.time()

    # mds = MDS(n_components=2, dissimilarity='precomputed',
    #           random_state=42, n_jobs=-1)
    # embedding = mds.fit_transform(distance_matrix)

    # embedding, correlations = fast_mds(distance_matrix)
    # embedding, _ = fast_mds_no_precompute(commits_arr)
    # embedding = create_graph_embedding(distance_matrix)
    # embedding = spring_mds(distance_matrix)
    embedding = spe(distance_matrix, 100000, 0.1, 0.01)
    print(embedding)

    # similarity_matrix = np.ones(distance_matrix.shape) - distance_matrix
    # embedding = eGTM(k=16, m=2, s=0.3, regul=0.1).fit(
    # similarity_matrix).transformed()

    # isomap = Isomap(n_components=2, metric="precomputed", n_neighbors=7)
    # embedding = isomap.fit_transform(distance_matrix)
    t1 = time.time()
    print("Took", t1 - t0)

    # x_values = list(range(1, len(correlations) + 1))
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_values, correlations, marker='o')
    # plt.title('Correlation Values')
    # plt.xlabel('Index')
    # plt.ylabel('Correlation')
    # plt.grid(True)
    # plt.show()

    # embedding = scale_embedding(embedding)

    # Spread apart files which ended up at the exact same location
    # embedding += np.random.rand(N, 2) * 0.0001

    print("Visualising embeddings...")
    visualize_embedding_with_extension_color(
        embedding, file_names, commits, repo_name, output_file)

    print("Loss:", calculate_loss(embedding, distance_matrix))

    # print("Correlation:", calculate_correlation(embedding, distance_matrix))

    if output_file:
        print("Saved output to", output_file)


def clone_repo(repo_url, temp_dir):
    print(f"Cloning repository {repo_url}...")
    subprocess.check_call(["git", "clone", repo_url, temp_dir])


def is_git_repo(path):
    # Check if the path contains a .git directory
    return os.path.isdir(os.path.join(path, '.git'))


def get_repo_name_from_url(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    repo_name = os.path.basename(path)
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    return repo_name


def get_last_folder_name(path):
    return os.path.basename(path.rstrip('/'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a Git repository.')
    parser.add_argument('repo_path_or_url', type=str,
                        help='Path to or URL of the Git repository')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file for the visualization (optional)')

    args = parser.parse_args()

    process_repository(args.repo_path_or_url, args.output)
