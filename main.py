import subprocess
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import warnings
import tempfile
import numpy as np
import time

from gmds import euclidean_distances

from lmds import landmark_mds


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


def create_similarity_matrix(commits, file_names):
    n = len(file_names)
    matrix = [[0.0] * n for _ in range(n)]
    for i, file1 in enumerate(file_names):
        for j, file2 in enumerate(file_names[i:], i):
            score = jaccard_index(
                commits[file1], commits[file2]) if i != j else 1
            matrix[i][j] = matrix[j][i] = score
    return matrix


def create_lmds_embedding(distance_matrix):
    return landmark_mds(distance_matrix, 1000, 2)


def count_file_changes(commits):
    return {file: len(changes) for file, changes in commits.items()}


def compute_distance_matrix(similarity_matrix):
    similarity_matrix = np.array(similarity_matrix)
    epsilon = 1e-5
    adjusted_similarity_matrix = similarity_matrix + epsilon
    distance_matrix = 1 / adjusted_similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


def get_extension_colors(file_names, commits):
    # Count files for each extension
    ext_count = {ext: 0 for ext in set(name.split(
        '.')[-1] if '.' in name else 'no_extension' for name in file_names)}
    for file in file_names:
        ext = file.split('.')[-1] if '.' in file else 'no_extension'
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


def visualize_embedding_with_extension_color(embedding, file_names, commits):
    extension_colors = get_extension_colors(file_names, commits)
    colors = [extension_colors[name.split(
        '.')[-1] if '.' in name else 'no_extension'] for name in file_names]

    trace = go.Scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        mode='markers',
        hoverinfo='text',
        hovertext=file_names,
        marker=dict(
            size=5,
            color=colors,
            opacity=0.8
        )
    )

    layout = go.Layout(
        title='2D UMAP Visualization by File Extension',
        xaxis=dict(title='UMAP Dimension 1'),
        yaxis=dict(title='UMAP Dimension 2'),
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,1)'
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


def calculate_loss(X, D):
    distances = euclidean_distances(X)
    loss = np.sum((D - distances) ** 2) / 2
    return loss


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


def process_repository(repo_path):
    print("Extracting commit information...")
    commits = get_commits(repo_path)

    print("Filtering to current files...")
    current_files = get_current_files(repo_path)
    commits = {file: hashes for file, hashes in commits.items(
    ) if file in current_files and len(hashes) > 1}

    file_names = list(commits.keys())
    print(f"Calculating similarity matrix for {len(file_names)} files...")
    similarity_matrix = create_similarity_matrix(commits, file_names)

    similarity_matrix = np.array(similarity_matrix, dtype=np.float64)

    distance_matrix = compute_distance_matrix(similarity_matrix)

    print("Creating embedings...")
    start_time = time.time()
    embedding = create_lmds_embedding(distance_matrix)
    embedding = scale_embedding(embedding)
    end_time = time.time()
    print(f"Finding embedding took {end_time - start_time} seconds")

    print("shape:", embedding.shape)
    print("loss:", calculate_loss(embedding, distance_matrix))

    print("Visualising embeddings")
    visualize_embedding_with_extension_color(embedding, file_names, commits)


def clone_repo(repo_url, temp_dir):
    print(f"Cloning repository {repo_url}...")
    subprocess.check_call(["git", "clone", repo_url, temp_dir])


def is_git_repo(path):
    # Check if the path contains a .git directory
    return os.path.isdir(os.path.join(path, '.git'))


def main(repo_path_or_url):
    is_url = repo_path_or_url.startswith(
        "http://") or repo_path_or_url.startswith("https://")

    # Clone the repo if a URL is provided
    if is_url:
        with tempfile.TemporaryDirectory() as temp_dir:
            clone_repo(repo_path_or_url, temp_dir)
            process_repository(temp_dir)
    elif os.path.exists(repo_path_or_url) and is_git_repo(repo_path_or_url):
        process_repository(repo_path_or_url)
    else:
        print("Error: The provided path is not a valid Git repository.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_git_repo_or_git_url>")
        sys.exit(1)

    repo_path_or_url = sys.argv[1]

    # Suppress a warning from UMAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main(repo_path_or_url)
