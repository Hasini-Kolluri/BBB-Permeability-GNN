import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_add_pool


class GraphConvModel(torch.nn.Module):

    def __init__(
        self,
        input_features=33,
        embedding_size=128
    ):

        super(GraphConvModel, self).__init__()

        torch.manual_seed(565)

        self.initial_conv = GraphConv(
            input_features,
            embedding_size
        )

        self.conv1 = GraphConv(
            embedding_size,
            embedding_size
        )

        self.conv2 = GraphConv(
            embedding_size,
            embedding_size
        )

        self.conv3 = GraphConv(
            embedding_size,
            embedding_size
        )

        self.dropout = torch.nn.Dropout(
            p=0.5
        )

        self.out = torch.nn.Linear(
            embedding_size,
            1
        )

    def forward(
        self,
        x,
        edge_index,
        batch_index
    ):

        x = self.initial_conv(
            x,
            edge_index
        )

        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv1(
            x,
            edge_index
        )

        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv3(
            x,
            edge_index
        )

        x = F.gelu(x)
        x = self.dropout(x)

        x = global_add_pool(
            x,
            batch_index
        )

        x = self.out(x)

        return x