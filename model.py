import torch.nn as nn

class MFRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, dropout=0.2):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, user_ids, item_ids):
        u = self.dropout(self.user_emb(user_ids))
        v = self.dropout(self.item_emb(item_ids))
        dot = (u * v).sum(dim=1)
        return dot + self.user_bias(user_ids).squeeze() + self.item_bias(item_ids).squeeze()
