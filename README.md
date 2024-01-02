# git-cluster

## Description

Visualize file relationships in a Git repository as an interactive 2D plot using multidimensional scaling (MDS). Files frequently modified together in commits are positioned closely in the visualization.


### Example visualisations

[pytorch/pytorch](https://github.com/pytorch/pytorch)
<img width="995" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/37ee4199-1a5f-4fb1-bf47-8b27504e11c7">


[kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)
<img width="995" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/5d6eb680-fd7b-4012-aa59-b8175d032212">

## Installation

Requires Python 3. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py target_repo_dir_or_url [-o output_file]
```

### Examples

```bash
python main.py /path/to/local/repo
```

```bash
python main.py /path/to/repo -o output.html
```

```bash
python main.py https://github.com/user/repo.git
```

### Method
For each pair of files in the specified repository we compute a distance metric: 1 - (number of commits which change both files) / (number of commits which change  at least one of the files). For a repository with N files this gives us an N x N distance matrix.

On this distance matrix we can apply [multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling), which assigns a point in 2D to each file. These points are chosen such that the Euclidian distance between them is close to their distance in the distance matrix. When we plot this with [plotly](https://plotly.com/python) we get a visualisation where files which are worked on (committed) together are close together. For most repositories this reveals interesting structure.

Note that since MDS is inefficient (cubic in N), git-cluster implements [landmark MDS](https://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf), where we select n (with n < N) random points and only run the expensive proper MDS algorithm on that n x n submatrix. Then we do linear algebra black magic to triangulate the positions of the remaining points. This is much faster while still giving good results, and means we only need an n x N distance matrix, making the whole procedure feasible even for very large repositories.
