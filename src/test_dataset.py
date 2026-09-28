from dataset import load_data, split_data


graphs = load_data(
    "../data/train_mol_set.xlsx"
)

print("Total graphs:", len(graphs))

train_graphs, val_graphs = split_data(
    graphs
)

print("Training graphs:", len(train_graphs))
print("Validation graphs:", len(val_graphs))

print()

print("First graph:")
print(train_graphs[0])

print()

print("Node features:")
print(train_graphs[0].x.shape)

print("Edges:")
print(train_graphs[0].edge_index.shape)

print("Label:")
print(train_graphs[0].y)