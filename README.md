# git-cluster

## Description

Visualize file relationships in a Git repository as an interactive 2D plot using multidimensional scaling (MDS). Files frequently modified together in commits are positioned closely in the visualization.


### Example visualisations

[torvalds/linux](https://github.com/torvalds/linux)
<img width="1195" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/21759112-c4c0-4312-846a-e21d869de105">


[pytorch/pytorch](https://github.com/pytorch/pytorch)
<img width="1195" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/4b4958e2-87bf-4b6e-945a-3451eb99e988">

[golang/go](https://github.com/golang/go)
<img width="1195" alt="image" src="https://github.com/fplonka/git-cluster/assets/92261790/5a401442-b5a7-4ca2-8b83-b0c65cf5eca3">

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
