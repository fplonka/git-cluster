## Description

Visualize file relationships in a Git repository as an interactive 2D plot using multidimensional scaling (MDS). Files frequently modified together in commits are positioned closely in the visualization.


### Example visualisations

[pytorch/pytorch](https://github.com/pytorch/pytorch)
<img width="1011" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/8b7ee22a-8c2a-4155-9117-3428a8c14adb">

[torvalds/linux](https://github.com/torvalds/linux)
<img width="1011" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/47baa2d6-507d-45cf-afc1-a1d7bfce6624">


[nodejs/node](https://github.com/nodejs/node)
<img width="1011" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/d3cc38eb-96f0-4d72-b3f8-8e0005f1973c">


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
For each pair of files in the specified repository we compute a distance metric: 1 - (number of commits which change both files) / (number of commits which change at least one of the files). For a repository with N files this gives us an N x N distance matrix.

On this distance matrix we can apply techniques from [multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling), which assigns a point in 2D to each file. These points are chosen such that the Euclidian distance between them is close to their distance in the distance matrix. When we plot this with [plotly](https://plotly.com/python) we get a visualisation where files which are worked on (committed) together are close together. For most repositories this reveals interesting structure.

The method used to find these 2D positions is [pivot-based Stochastic Proximity Embedding](https://www.researchgate.net/publication/10602021_A_modified_update_rule_for_stochastic_proximity_embedding), which, over many iterations, picks a random point and then adjusts the position of all other points so that their embedding distance to the pivot point more closely matches their distance metric to the pivot. The adjustments are proportional to a learning rate which is decreased over time. For large repositories (10k+ files) around 1 milion such iterations are needed to get a very good result.

A goal for a future version is to compute the embedding on the GPU using metal shaders, hopefully speeding up the whole process significantly.
