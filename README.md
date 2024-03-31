## Description

Visualize file relationships in a Git repository as an interactive 2D plot using GPU-accelerated multidimensional scaling (MDS). Each file is represented by a dot, and files frequently modified together in commits are positioned closely in the visualization.


### Example visualisations

 For interactive visualisations of 100 popular repositores see [fplonka.dev/git-cluster](https://fplonka.dev/git-cluster).

[torvalds/linux](https://github.com/torvalds/linux)
<img width="1014" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/f861f6d8-67df-4061-8ad4-801d0ae433c7">

[tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
<img width="1019" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/bcae00a6-9aa7-4585-9b85-861966f9fb30">

[pytorch/pytorch](https://github.com/pytorch/pytorch)
<img width="1011" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/8b7ee22a-8c2a-4155-9117-3428a8c14adb">


## Installation


Requires Python 3. Clone the repo and install dependencies with:

```bash
git clone https://github.com/fplonka/git-cluster
cd git-cluster
pip install -r requirements.txt
```

GPU acceleration for git-cluster uses Apple Metal shaders, which are only available on Apple platforms. GPU acceleration has currently only been tested on my M1 Macbook Air. To use GPU acceleration, you need to download [metal-cpp](https://developer.apple.com/metal/cpp). Extract the contents of the .zip file to `/path/to/metal-cpp/` then run:
```bash
cd metal/
make all METAL_CPP_PATH=/path/to/metal-cpp/
```
After this you can pass the `--use-gpu` flag to use GPU acceleration. This makes computing the embeddings ~200x faster, which is particularly useful on large repositories where for the best results 1 milion or more iterations are needed.

### Examples

Run git-cluster on a local repo, using 20000 iterations, saving the resulting visualisation to `rust.html`:
```bash
python git-cluster.py /path/to/local/repo --num-iterations 20000 --output rust.html
```

Run git-cluster on the [Rust](https://github.com/rust-lang/rust) repo, using the default 100000 iterations, enabling GPU acceleration, and caching computed data so that when you run git-cluster on this repo next time, you don't have to clone it again:

```bash
python git-cluster.py https://github.com/rust-lang/rust --use-gpu --use-cache
```


## Usage

Run `python git-cluster.py` in the `git-cluter/` directory.

```
usage: git-cluster.py [-h] [-o OUTPUT]
                      [-n NUM_ITERATIONS]
                      [--use-gpu] [-c]
                      repo_path_or_url

positional arguments:
  repo_path_or_url      Path to or URL of the
                        Git repository

options:
  -h, --help            show this help message
                        and exit
  -o OUTPUT, --output OUTPUT
                        Output file for the
                        visualization
  -n NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        Number of iterations to
                        run the algorithm
                        (default: 10000)
  --use-gpu             Use GPU for computations
                        (ARM MacOS only)
  -c, --cache           Cache distance matrix
                        for future reuse
```

## Method
For each pair of files in the specified repository we compute a distance metric: 1 - (number of commits which change both files) / (number of commits which change at least one of the files). For a repository with N files this gives us an N x N distance matrix.

On this distance matrix we can apply techniques from [multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling), which assigns a point in 2D to each file. These points are chosen such that the Euclidian distance between them is close to their distance in the distance matrix. When we plot this with [plotly](https://plotly.com/python) we get a visualisation where files which are worked on (committed) together are close together. For most repositories this reveals interesting structure.

The method used to find these 2D positions is [pivot-based Stochastic Proximity Embedding](https://www.researchgate.net/publication/10602021_A_modified_update_rule_for_stochastic_proximity_embedding), which, over many iterations, picks a random point and then adjusts the position of all other points so that their embedding distance to the pivot point more closely matches their distance metric to the pivot. The adjustments are proportional to a learning rate which is decreased over time. For large repositories (10k+ files) around 1 milion such iterations are needed to get a very good result.
