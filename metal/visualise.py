import matplotlib.pyplot as plt


def visualize_embeddings(file_path):
    x_coords = []
    y_coords = []

    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(', ')
            x_coords.append(float(x))
            y_coords.append(float(y))

    plt.scatter(x_coords, y_coords, s=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Embeddings Visualization')
    plt.show()


visualize_embeddings("embeddings.txt")
