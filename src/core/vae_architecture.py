import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from typing import Optional, Tuple

class GraphEncoder(nn.Module):
    def __init__(self, node_features=8, edge_features=5, global_features=2, 
                 hidden_dim=128, latent_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.to_mu = nn.Linear(hidden_dim + global_features, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim + global_features, latent_dim)

    def forward(self, x, edge_index, edge_attr, batch, global_features):
        x = F.relu(self.node_encoder(x))
        edge_attr = F.relu(self.edge_encoder(edge_attr))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = self.dropout(x)
        
        graph_emb = global_mean_pool(x, batch)
        graph_emb = torch.cat([graph_emb, global_features], dim=-1)
        return self.to_mu(graph_emb), self.to_logvar(graph_emb)

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, pocket_dim=133, max_nodes=50, 
                 node_features=8, edge_features=5, dropout=0.2, use_pocket=True):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_features = node_features
        self.edge_features = edge_features
        
        self.pocket_proj = nn.Linear(pocket_dim, hidden_dim // 2) if use_pocket else None
        expand_input = latent_dim + (hidden_dim // 2 if use_pocket else 0)
        
        self.mlp_expand = nn.Sequential(
            nn.Linear(expand_input, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        
        self.node_decoder = nn.Linear(hidden_dim, max_nodes * node_features)
        self.adj_decoder = nn.Linear(hidden_dim, max_nodes * max_nodes)
        self.type_decoder = nn.Linear(hidden_dim, max_nodes * max_nodes * edge_features)

    def forward(self, z, pocket_embedding=None):
        h = z
        if self.pocket_proj is not None and pocket_embedding is not None:
            h = torch.cat([z, self.pocket_proj(pocket_embedding)], dim=-1)
        
        h = self.mlp_expand(h)
        # Reshape into (Batch, Nodes, Features) and (Batch, Nodes, Nodes, Types)
        node_logits = self.node_decoder(h).view(-1, self.max_nodes, self.node_features)
        adj_logits = self.adj_decoder(h).view(-1, self.max_nodes, self.max_nodes)
        type_logits = self.type_decoder(h).view(-1, self.max_nodes, self.max_nodes, self.edge_features)
        return node_logits, adj_logits, type_logits

class GraphVAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.beta = kwargs.get('beta', 1.0)
        self.latent_dim = kwargs.get('latent_dim', 64)
        self.encoder = GraphEncoder(**kwargs)
        self.decoder = GraphDecoder(**kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, edge_attr, batch, global_features, pocket_embedding=None):
        mu, logvar = self.encoder(x, edge_index, edge_attr, batch, global_features)
        z = self.reparameterize(mu, logvar)
        node_logits, adj_logits, type_logits = self.decoder(z, pocket_embedding)
        return node_logits, adj_logits, type_logits, mu, logvar

    def loss_function(self, x, adj, edge_attr, node_logits, adj_logits, type_logits, mu, logvar):
        # Reconstruction: CrossEntropy for nodes, BinaryCrossEntropy for adjacency
        node_loss = F.cross_entropy(node_logits.view(-1, node_logits.size(-1)), x.argmax(dim=-1).view(-1))
        adj_loss = F.binary_cross_entropy_with_logits(adj_logits, adj)
        # KLD Regularization
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return node_loss + adj_loss + (self.beta * kld)
