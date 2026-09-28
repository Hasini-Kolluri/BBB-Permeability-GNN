from features import smiles_to_graph


smiles = "CCO"

node_features, edge_index = smiles_to_graph(smiles)

print("Node features:")
print(node_features)

print("\nNode feature shape:")
print(node_features.shape)

print("\nEdges:")
print(edge_index)

print("\nNumber of nodes:")
print(node_features.shape[0])

print("\nNumber of edges:")
print(edge_index.shape[1])
