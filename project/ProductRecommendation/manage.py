
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch.nn as nn
from torch_geometric.nn import GCNConv

# class GNN(nn.Module):
        # def __init__(self, input_dim, hidden_dim, output_dim):
        #         super(GNN, self).__init__()
        #         self.conv1 = GCNConv(input_dim, hidden_dim)
        #         self.conv2 = GCNConv(hidden_dim, output_dim)

        # def forward(self, data):
        #         x, edge_index = data.x, data.edge_index
        #         x = F.relu(self.conv1(x, edge_index))
        #         x = F.dropout(x, training=self.training)
        #         x = self.conv2(x, edge_index)
        #         return x


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProductRecommendation.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
