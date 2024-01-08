import numpy as np


def create_sparse_representation(commit_dict):
    # Step 1: Enumerate all unique commit hashes
    unique_commits = set()
    for commit_set in commit_dict.values():
        unique_commits.update(commit_set)
    commit_to_id = {commit: i for i, commit in enumerate(unique_commits)}

    # Step 2: Convert each set to a sorted array of integers
    sparse_representation = []
    for key in sorted(commit_dict.keys()):  # Ensure the order matches the keys
        commit_ids = np.array([commit_to_id[commit]
                              for commit in commit_dict[key]], dtype=np.int32)
        sparse_representation.append(np.sort(commit_ids))

    return sparse_representation, commit_to_id


def log_space_values(a, b, n):
    """
    Generates n logarithmically spaced values between a and b.
    """
    log_a = np.log10(a)
    log_b = np.log10(b)
    return np.logspace(log_a, log_b, num=n, base=10)


matrix = np.random.rand(70000, 70000)
np.save('testmatrix.npy', matrix)

# print(log_space_values(0.9, 0.999, 3))

# # Example usage
# commit_dict = {
#     0: {"a1b2c3", "d4e5f6"},
#     1: {"d4e5f6", "g7h8i9"},
#     2: {"a1b2c3"}
# }

# sparse_repr, commit_mapping = create_sparse_representation(commit_dict)
# print(sparse_repr)
