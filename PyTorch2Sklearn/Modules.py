from PyTorch2Sklearn.environment import *


class LinearLayer(nn.Module):
    """Dense Layer: Linear -> ReLU -> Dropout"""

    def __init__(self, CFG, input_hidden_dim, output_hidden_dim, dropout=0):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG["random_state"])

        self.layer = nn.Sequential(
            nn.Linear(input_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        init.normal_(self.layer[0].weight, mean=0, std=0.01)
        init.constant_(self.layer[0].bias, 0)

    def forward(self, x):
        return self.layer(x)


class ResLayer(nn.Module):
    """Dense Residual Layer: x + Linear -> ReLU -> Dropout"""

    def __init__(self, CFG, input_hidden_dim, output_hidden_dim, dropout=0):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG["random_state"])

        self.layer = nn.Sequential(
            nn.Linear(input_hidden_dim, input_hidden_dim),
            nn.ReLU(),
            nn.Linear(input_hidden_dim, output_hidden_dim),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        for module in self.layer.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.01)
                init.constant_(module.bias, 0)

    def forward(self, x):
        return self.dropout(self.activation(x + self.layer(x)))


class GCN(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG["random_state"])
        self.gcn = nn.Linear(self.CFG["hidden_dim"], self.CFG["hidden_dim"], bias=True)

    def forward(self, x, graph):
        return self.gcn(graph @ x)


class A_GCN(nn.Module):
    def __init__(self, CFG, n_heads):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG["random_state"])

        self.n_heads = n_heads
        self.dim_per_head = self.CFG["hidden_dim"] // self.n_heads

        self.QLinear = nn.Linear(
            self.CFG["hidden_dim"], self.CFG["hidden_dim"], bias=False
        )
        self.KLinear = nn.Linear(
            self.CFG["hidden_dim"], self.CFG["hidden_dim"], bias=False
        )
        self.VLinear = nn.Linear(
            self.CFG["hidden_dim"], self.CFG["hidden_dim"], bias=False
        )
        self.OutLinear = nn.Linear(
            self.CFG["hidden_dim"], self.CFG["hidden_dim"], bias=False
        )
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.forward1 = nn.Linear(
            self.CFG["hidden_dim"], self.CFG["hidden_dim"], bias=False
        )
        self.forward2 = nn.Linear(
            self.CFG["hidden_dim"], self.CFG["hidden_dim"], bias=False
        )

        self.layer_norm1 = nn.LayerNorm(self.CFG["hidden_dim"])
        self.dropout = nn.Dropout(self.CFG["dropout"])
        self.layer_norm2 = nn.LayerNorm(self.CFG["hidden_dim"])

    def forward(self, x, graph):
        Q = self.dropout(self.QLinear(x))
        K = self.dropout(self.KLinear(x))
        V = self.dropout(self.VLinear(x))

        attention_out_list = []
        for j in range(self.CFG.n_heads):

            Q_tmp = Q[:, j * self.dim_per_head : (j + 1) * self.dim_per_head]
            K_tmp = K[:, j * self.dim_per_head : (j + 1) * self.dim_per_head]
            V_tmp = V[:, j * self.dim_per_head : (j + 1) * self.dim_per_head]

            attention_out_tmp = (
                self.dropout(
                    self.softmax(
                        (Q_tmp @ torch.transpose(K_tmp, 0, 1))
                        * graph
                        / np.sqrt(self.dim_per_head)
                    )
                )
                @ V_tmp
            )

            attention_out_list.append(attention_out_tmp)

        out = torch.cat(attention_out_list, dim=1)

        attention_out = self.dropout(out)

        x_add_attention = x + attention_out

        x_add_attention_normalised = self.layer_norm1(x_add_attention)

        forward_output = self.dropout(
            self.forward2(self.relu(self.forward1(x_add_attention_normalised)))
        )

        x_add_forward = forward_output + x_add_attention

        x_add_forward_normalised = self.layer_norm2(x_add_forward)

        return x_add_forward_normalised


class GraphBottleNeckLayer(nn.Module):
    """Compresses GNN to 1-d output, so decode MLP can be treated as linear regression and GNN is treated as feature extractor"""

    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG.random_state)
        self.linear = nn.Linear(self.CFG["hidden_dim"], 1, bias=True)

    def forward(self, x):
        return self.linear(x)
