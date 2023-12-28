# Git Repository Visualizer

## Description

Visualize file relationships in a Git repository as an interactive 2D plot using multidimensional scaling (MDS). Files frequently modified together in commits are positioned closely in the visualization.

## Installation

Requires Python 3. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Local Repository

```bash
python main.py /path/to/local/repo
```

### Remote Repository

```bash
python main.py https://github.com/user/repo.git
```

### Save Output

```bash
python main.py /path/to/repo -o output.html
```
