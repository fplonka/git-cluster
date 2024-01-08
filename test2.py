import numpy as np
import graph_tool.all as gt

# Example distance matrix (replace this with your actual matrix)
# Assuming a symmetric matrix for an undirected graph
# Replace with your actual distance matrix
dist_matrix = np.load('cache/gorepo.npy')

g = gt.Graph(directed=False)
g.add_vertex(dist_matrix.shape[0])

# Edge property to store weights
weight = g.new_edge_property("double")

# Add edges and weights
edge_cnt = 0
for i in range(dist_matrix.shape[0]):
    # Avoid duplicates in undirected graph
    for j in range(i+1, dist_matrix.shape[1]):
        if dist_matrix[i][j] != 0:  # Assuming a non-zero value means an edge exists
            w = dist_matrix[i][j]
            if w != 1.00001:
                edge = g.add_edge(g.vertex(i), g.vertex(j))
                weight[edge] = w
                edge_cnt += 1

print("have", edge_cnt, "edges")

# Save the weight property to the graph
g.edge_properties["weight"] = weight

pos = gt.sfdp_layout(g)
gt.graph_draw(g, pos, output_size=(1000, 1000), output="sfdp_layout.png")
