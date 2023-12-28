from numpy.linalg import eigh
import webbrowser
import subprocess
import os
import sys
import argparse
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import numpy as np
import random


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


def create_similarity_matrix(commits, file_names, landmark_indices, progress_text):
    N = len(file_names)
    n = len(landmark_indices)

    matrix = np.zeros((n, N))

    for i, landmark_idx in enumerate(landmark_indices):
        file1 = file_names[landmark_idx]
        for j, file2 in enumerate(file_names):
            # Calculate score only if it's not a diagonal element (i.e., file1 != file2)
            score = jaccard_index(
                commits[file1], commits[file2]) if landmark_idx != j else 1
            matrix[i][j] = score

        print_progress(progress_text, (i + 1) / n * 100)

    # Clear the progress percentage text once done
    print(f"\r{progress_text}...          ")

    return matrix


def classical_mds(distance_matrix, k):
    """
    Classical MDS
    :param distance_matrix: A n x n distance matrix.
    :param k: Number of dimensions for the output.
    :return: A n x k matrix representing the embedded coordinates.
    """
    distance_matrix = np.square(distance_matrix)
    n = distance_matrix.shape[0]

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Double centering
    B = -0.5 * H @ distance_matrix @ H

    # Eigen decomposition
    eigvals, eigvecs = eigh(B)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top k eigenvalues and eigenvectors
    L = np.sqrt(np.abs(eigvals[:k]))
    V = eigvecs[:, :k]

    return V * L


def landmark_mds(distance_matrix, landmark_indices, k=2):
    """
    MDS is O(N^3) so we do landmark MDS: we do classical MDS on a smaller n x n submatrix of the full N x N distance matrix and then for the other points triangulate their position. See https://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf for details of the linear algebra black magic. Using landmark MDS also means we actually only need an n x N distance matrix instead of the full N x N, which is a big memory and time save too.
    :param distance_matrix: A n x N distance matrix.
    :param landmark_indices: An array of n indices which give the landmark points. The rows of distance_matrix should correspond to these n points.
    :param k: Number of dimensions for the output.
    :return: N x k matrix of embeddings for all data points.
    """
    N = distance_matrix.shape[1]
    n = len(landmark_indices)

    if (n >= N):
        return classical_mds(distance_matrix, k)

    distance_matrix = np.square(distance_matrix)

    # Step 1: Select landmark points
    landmark_distances = distance_matrix[:, landmark_indices]

    # Compute B_n
    mu_n = np.mean(landmark_distances, axis=1)
    mu = np.mean(mu_n)
    B_n = -0.5 * (landmark_distances -
                  mu_n[:, np.newaxis] - mu_n[np.newaxis, :] + mu)

    # Eigendecomposition
    eigvals, eigvecs = eigh(B_n)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Calculate k+ (number of positive eigenvalues)
    k_plus = min(k, np.sum(eigvals > 0))

    # Calculate L
    L = eigvecs[:, :k_plus] * np.sqrt(np.abs(eigvals[:k_plus]))

    # Step 3: Distance-based triangulation
    L_sharp = eigvecs[:, :k_plus] / np.sqrt(np.abs(eigvals[:k_plus]))
    X = np.zeros((k_plus, N))

    for j in range(N):
        delta = distance_matrix[:, j] - mu_n
        X[:, j] = -L_sharp.T @ delta / 2

    return X.T


def euclidean_distances(Y):
    Q = np.einsum("ij,ij->i", Y, Y)[:, np.newaxis]
    distances = -2 * Y @ Y . T
    distances += Q
    distances += Q.T
    np.maximum(distances, 1e-10, out=distances)
    # return distances
    return np.sqrt(distances)


def compute_distance_matrix(similarity_matrix):
    # n = similarity_matrix.shape[0]
    # return np.ones((n, n)) - similarity_matrix

    similarity_matrix = np.array(similarity_matrix)
    epsilon = 1e-6
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


def visualize_embedding_with_extension_color(embedding, file_names, commits, output_file):
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
        title='File Commit-Similarity Visualization (Colored by File Extension)',
        xaxis=dict(title='MDS Dimension 1'),
        yaxis=dict(title='MDS Dimension 2'),
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,1)'
    )

    fig = go.Figure(data=[trace], layout=layout)
    if (output_file != None):
        fig.write_html(output_file, include_plotlyjs=True)
        webbrowser.open('file://' +
                        os.path.realpath(output_file), new=2)
        print("Saved output to", output_file)
    else:
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


def print_progress(str, progress):
    print(f"\r{str}... ({progress:.2f}%)", end="")


def process_repository(repo_path, output_file):
    print("Extracting commit information...")
    commits = get_commits(repo_path)

    print("Filtering to current files...")
    current_files = get_current_files(repo_path)
    commits = {file: hashes for file, hashes in commits.items(
    ) if file in current_files and len(hashes) > 1}

    # Create distance matrix
    file_names = np.array(list(commits.keys()))
    N = len(file_names)

    n = 1000  # no. of landmark points
    if (n >= N):
        landmark_indices = range(N)
    else:
        landmark_indices = random.sample(range(N), n)

    similarity_matrix = create_similarity_matrix(
        commits, file_names, landmark_indices, "Creating distance matrix")

    distance_matrix = compute_distance_matrix(similarity_matrix)

    print("Creating embedings...")
    embedding = landmark_mds(distance_matrix, landmark_indices)
    embedding = scale_embedding(embedding)

    print("Visualising embeddings...")
    visualize_embedding_with_extension_color(
        embedding, file_names, commits, output_file)


def clone_repo(repo_url, temp_dir):
    print(f"Cloning repository {repo_url}...")
    subprocess.check_call(["git", "clone", repo_url, temp_dir])


def is_git_repo(path):
    # Check if the path contains a .git directory
    return os.path.isdir(os.path.join(path, '.git'))


def main(repo_path_or_url, output_file):
    is_url = repo_path_or_url.startswith(
        "http://") or repo_path_or_url.startswith("https://")

    # Clone the repo if a URL is provided
    if is_url:
        with tempfile.TemporaryDirectory() as temp_dir:
            clone_repo(repo_path_or_url, temp_dir)
            process_repository(temp_dir, output_file)
    elif os.path.exists(repo_path_or_url) and is_git_repo(repo_path_or_url):
        process_repository(repo_path_or_url, output_file)
    else:
        print("Error: The provided path is not a valid Git repository.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a Git repository.')
    parser.add_argument('repo_path_or_url', type=str,
                        help='Path to or URL of the Git repository')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file for the visualization (optional)')

    args = parser.parse_args()

    main(args.repo_path_or_url, args.output)
