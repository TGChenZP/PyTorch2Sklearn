from PyTorch2Sklearn.environment import *


class LinearLayer(nn.Module):
    """ Dense Layer: Linear -> ReLU -> Dropout """

    def __init__(self, CFG, input_hidden_dim, output_hidden_dim, dropout=0):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG['random_state'])

        self.layer = nn.Sequential(
            nn.Linear(input_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        init.normal_(self.layer[0].weight, mean=0, std=0.01)
        init.constant_(self.layer[0].bias, 0)

    def forward(self, x):
        return self.layer(x)


class ResLayer(nn.Module):
    """ Dense Residual Layer: x + Linear -> ReLU -> Dropout """

    def __init__(self, CFG, input_hidden_dim, output_hidden_dim, dropout=0):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG['random_state'])

        self.layer = nn.Sequential(
            nn.Linear(input_hidden_dim, input_hidden_dim),
            nn.ReLU(),
            nn.Linear(input_hidden_dim, output_hidden_dim)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        for module in self.layer.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.01)
                init.constant_(module.bias, 0)

    def forward(self, x):
        return self.dropout(self.activation(x + self.layer(x)))
