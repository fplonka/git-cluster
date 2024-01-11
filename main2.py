import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import math
from spe import spe_fancy
# import graph_tool.all as gt
import numba
from utils import calculate_loss, minhash_signature, estimated_jaccard
from utils import log_space_values
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.stats import pearsonr
import gc
import numpy as np
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import pickle
import argparse
import os
import subprocess
import webbrowser
from urllib.parse import urlparse
from sklearn.manifold import MDS
import numpy as np
import time
from optimisers import SQuaD_MDS, SQuaD_MDS_no_precompute
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD

from utils import calculate_correlation, jaccard_fast

from spe import spe_optimized
from spe import spe_optimized_parallel
from spe import spe_optimized_parallel_fancy
from spe import spe_optimized_parallel_dist
from spe import spe_optimized_parallel_fancy_animated


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


@numba.jit(nopython=True, fastmath=True)
def create_distance_matrix_fast(commits_arr):
    N = len(commits_arr)
    matrix = np.ones((N, N), dtype=np.float32)

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


def visualize_embedding_with_extension_color_animated(embeddings, file_names, commits, repo_name, output_file=None):
    extension_colors = get_extension_colors(file_names, commits)

    # Create a subplot for the animation
    fig = make_subplots()

    for i in range(len(embeddings)):
        mean_x, mean_y = np.mean(embeddings[i], axis=0)
        embeddings[i] = embeddings[i] - np.array([mean_x, mean_y])

    frames = []

    for embedding in embeddings:  # Loop over each embedding (each frame)
        traces = []
        for ext, color in extension_colors.items():
            indices = [i for i, name in enumerate(
                file_names) if get_extension(name) == ext]
            if indices:
                x_values = embedding[indices, 0]
                y_values = embedding[indices, 1]
                hovertexts = [file_names[i] for i in indices]

                trace = go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    hoverinfo='text',
                    hovertext=hovertexts,
                    marker=dict(size=5, color=color),
                    name=ext,
                )
                traces.append(trace)

        # Create a frame for each embedding
        frame = go.Frame(data=traces, name=str(embedding))
        frames.append(frame)

    fig.frames = frames

    frame_duration = 1000
    transition_duration = 1000
    # Add play and pause buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None]),
                    # dict(label="Pause", method="animate", args=[
                    #  [None], {"frame": {"duration": 100, "redraw": False}}])
                ],
            )
        ],
        title=f'File Commit-Similarity Visualization for {repo_name}',
        xaxis=dict(title='MDS Dimension 1'),
        yaxis=dict(title='MDS Dimension 2'),
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,1)'
    )

    # Add the first frame's data to set up the initial view
    initial_embedding = embeddings[0]
    for ext, color in extension_colors.items():
        indices = [i for i, name in enumerate(
            file_names) if get_extension(name) == ext]
        if indices:
            fig.add_trace(go.Scatter(
                x=initial_embedding[indices, 0],
                y=initial_embedding[indices, 1],
                mode='markers',
                hoverinfo='text',
                marker=dict(size=5, color=color),
                name=ext,
            ))

    if output_file:
        fig.write_html(output_file, include_plotlyjs=True)
        webbrowser.open('file://' + os.path.realpath(output_file), new=2)
    else:
        fig.show()


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

            trace = go.Scattergl(
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


def find_params(commits_arr, distance_matrix, n_iter=5000):
    # Define the ranges for each parameter
    lr_values = log_space_values(0.01, 10.0, 10)
    final_lr_values = log_space_values(0.0000001, 1.0, 10)

    # beta1_values = np.linspace(0.01/2, 0.01*2, 10)
    # beta2_values = np.linspace(0.99, 0.999999999, 3)
    # lr_values = np.linspace(0.6, 2.4, 10)

    print("lr", lr_values)
    print("final lr", final_lr_values)

    # Grid search
    best_score = np.inf
    best_params = {}

    cnt = 0
    for lr in lr_values:
        for final_lr in final_lr_values:
            if final_lr > lr:
                continue
            np.random.seed(42)
            embedding = spe_optimized_parallel_fancy(
                commits_arr, n_iter, lr, final_lr)
            score = calculate_loss(embedding, distance_matrix)

            print(score)
            if math.isnan(score):
                continue

            cnt += 1
            print_progress("evaluated", cnt / (len(lr_values)
                                               * len(final_lr_values)) * 100)
            if score < best_score:
                print("new best:", score)
                best_score = score
                best_params = {'lr': lr, 'final_lr': final_lr}

    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")

    return (lr, final_lr)


def find_params2(commits_arr, distance_matrix):
    # Define the space of hyperparameters to search
    space = [
        Real(0.0001, 10.0, name='lr'),
        Real(0.00000001, 0.1, name='lr_final')
    ]

    @use_named_args(space)
    def objective(**params):
        embedding = spe_optimized_parallel_fancy(
            commits_arr, 5000, params['lr'], params['lr_final'])
        return calculate_loss(embedding, distance_matrix)

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


def compute_minhash_signature_matrix(commits, num_hashes):
    # Create a list of MinHash objects
    minhashes = []

    for commit_set in commits.values():
        m = MinHash(num_perm=num_hashes)
        for commit in commit_set:
            m.update(str(commit).encode('utf8'))
        minhashes.append(m)

    # Convert MinHashes to a numpy array (signature matrix)
    n = len(minhashes)
    signature_matrix = np.zeros((n, num_hashes))

    for i, m in enumerate(minhashes):
        signature_matrix[i, :] = m.hashvalues

    return signature_matrix


def get_gpu_embeddings(file_path):
    x_coords = []
    y_coords = []

    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(', ')
            x_coords.append(float(x))
            y_coords.append(float(y))

    # Convert lists to a NumPy array of shape (n, 2)
    coords_array = np.column_stack((x_coords, y_coords))
    return coords_array


@numba.jit(nopython=True, fastmath=True)
def calculate_percentiles(matrix):
    flat_array = matrix.ravel()
    sorted_array = np.sort(flat_array)

    # Find the percentile rank for each value in the flattened array
    percentiles = np.searchsorted(
        sorted_array, flat_array, side='left') / len(sorted_array)

    # Reshape the percentiles to the original matrix shape
    percentile_matrix = percentiles.reshape(matrix.shape)
    return percentile_matrix


def write_dist_matrix_and_params_to_file(num_iters, initial_lr, final_lr, distance_matrix, filename):
    with open(filename, 'wb') as f:
        f.write(np.int32(num_iters).tobytes())
        f.write(np.float32(initial_lr).tobytes())
        f.write(np.float32(final_lr).tobytes())
        N = distance_matrix.shape[0]
        f.write(np.int32(N).tobytes())
        f.write(distance_matrix.tobytes())


def run_cpp_process():
    # Path to your C++ executable
    cpp_executable = "metal/build/MetalSPE"

    # Run the C++ executable
    result = subprocess.run([cpp_executable], capture_output=True, text=True)

    # Check if the process ran successfully
    if result.returncode != 0:
        print("C++ process failed with return code:", result.returncode)
        print("Error output:", result.stderr)
    else:
        print("C++ process output:\n", result.stdout)


def process_repository(args):
    repo_path_or_url = args.repo_path_or_url
    output_file = args.output
    using_gpu = args.use_gpu
    using_cache = args.cache

    np.random.seed(42)  # You can use any integer as the seed value

    is_url = repo_path_or_url.startswith(
        "http://") or repo_path_or_url.startswith("https://")
    if is_url:
        repo_name = get_repo_name_from_url(repo_path_or_url)
    else:
        repo_name = get_last_folder_name(repo_path_or_url)

    if using_cache:
        cache_path = os.path.join('cache', repo_name)
        os.makedirs('cache', exist_ok=True)

    if using_cache and os.path.isfile(cache_path):
        print("Loading distance matrix from cache...")
        with open(cache_path, 'rb') as f:
            file_names, commits = pickle.load(f)
            distance_matrix = np.load(f'{cache_path}.npy')
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

        print("Creating distance matrix...")
        t0 = time.time()
        distance_matrix = create_distance_matrix_fast(commits_arr)
        t1 = time.time()
        # distance_matrix = distance_matrix.astype(np.float16)
        print(f"Took {t1 - t0}s")

        if using_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump((file_names, commits), f)
                np.save(f'{cache_path}.npy', distance_matrix)
                print("Saved distance matrix and commit info to cache")

    commits_arr = dict_to_padded_array(commits)

    print("N is", N)
    print("Creating embeddings...")

    t0 = time.time()

    initial_lr = 0.1
    final_lr = 0.0000001
    if using_gpu:
        write_dist_matrix_and_params_to_file(
            args.num_iterations, initial_lr, final_lr, distance_matrix, 'dist_matrix_data')

        # run metal/build/SPE...
        run_cpp_process()

        embedding = get_gpu_embeddings(
            'metal/embeddings.txt')
    else:
        embedding = spe_optimized_parallel_dist(
            distance_matrix, args.num_iterations, initial_lr, final_lr)

    # embedding = spe_optimized_parallel_fancy(
    # commits_arr, 5000, 0.1, 0.0000001)

    # embedding_animated = spe_optimized_parallel_fancy_animated(
    # commits_arr, 15000, 0.1, 0.0000001, 50)
    t1 = time.time()
    print("Took", t1 - t0)

    print("Visualising embeddings...")
    visualize_embedding_with_extension_color(
        embedding, file_names, commits, repo_name, output_file)
    # visualize_embedding_with_extension_color_animated(
    # embedding_animated, file_names, commits, repo_name, output_file)

    # visualize_embedding_with_extension_color(
    # embedding, file_names, commits, repo_name, output_file)

    print("Loss:", calculate_loss(embedding, distance_matrix))

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
    parser = argparse.ArgumentParser(
        description='Visualize file relationships in a Git repository as an interactive 2D plot. Files frequently modified together in commits are positioned closely in the visualization.')
    parser.add_argument('repo_path_or_url', type=str,
                        help='Path to or URL of the Git repository')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file for the visualization (optional)')
    parser.add_argument('-n', '--num-iterations', type=int, default=10000,
                        help='Number of iterations to run the algorithm (default: 10000)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for computations (ARM MacOS only)')
    parser.add_argument('-c', '--cache', action='store_true',
                        help='Cache distance matrix for future reuse')

    args = parser.parse_args()

    # process_repository(args.repo_path_or_url, args.output)
    process_repository(args)
