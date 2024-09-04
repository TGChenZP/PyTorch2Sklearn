from PyTorch2Sklearn.__template__ import TorchToSklearn_GraphModel
from PyTorch2Sklearn.Modules import *


class MLP_AGNN(TorchToSklearn_GraphModel):
    """MLP Classifier or Regressor that can be used as a sklearn model"""

    class DecoderMLP(nn.Module):
        """MLP layers as decoder: Linear -> ReLU -> Dropout (last layer is Linear)"""

        def __init__(self, CFG, hidden_dim, dropout, batchnorm):
            super(MLP_AGNN.DecoderMLP, self).__init__()

            torch.manual_seed(CFG["random_state"])

            self.CFG = CFG

            mlp_layers = []

            # Middle layers (if num_decoder_layers > 1)
            for _ in range(CFG["num_decoder_layers"] - 1):
                mlp_layers.append(
                    LinearLayer(
                        CFG,
                        CFG["hidden_dim"],
                        CFG["hidden_dim"],
                        CFG["dropout"],
                    )
                )

            # Last layer
            mlp_layers.append(nn.Linear(CFG["hidden_dim"], CFG["output_dim"]))

            # Combine the layers into one sequential model
            self.out_mlp = nn.Sequential(*mlp_layers)

        def forward(self, X):

            return self.out_mlp(X)

    class Model(nn.Module):
        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            torch.manual_seed(self.CFG["random_state"])

            # Encoder
            if (
                self.CFG["hidden_dim"] == 0
            ):  # hidden_dim = 0 => shrink the hidden layer size
                step_size = (self.CFG["input_dim"] - self.CFG["output_dim"]) // (
                    self.CFG["num_encoder_layers"] - 1
                )
                if step_size == 0:
                    step_size = 1

                self.hidden_layer_size = [self.CFG["input_dim"]]
                self.hidden_layer_size += [
                    (
                        self.CFG["input_dim"] - i * step_size
                        if self.CFG["input_dim"] - i * step_size
                        > self.CFG["output_dim"]
                        else self.CFG["output_dim"]
                    )
                    for i in range(1, self.CFG["num_encoder_layers"])
                ]

                # set effective hidden_dim
                self.CFG["hidden_dim"] = self.hidden_layer_size[-1]

            else:  # hidden_dim != 0 => use the same hidden layer size; note hidden_layers is the number of hidden layers including the output layer but excluding the input layer
                self.hidden_layer_size = [
                    self.CFG["hidden_dim"]
                    for _ in range(self.CFG["num_encoder_layers"])
                ]

            layers = []

            # Input layer
            layers.append(
                LinearLayer(
                    self.CFG,
                    self.CFG["input_dim"],
                    self.hidden_layer_size[0],
                    self.CFG["dropout"],
                )
            )
            if self.CFG["batchnorm"]:
                layers.append(nn.BatchNorm1d(self.hidden_layer_size[0]))
            layers.append(nn.ReLU())

            # Hidden layers
            for i in range(self.CFG["num_encoder_layers"] - 1):
                layers.append(
                    LinearLayer(
                        self.CFG,
                        self.hidden_layer_size[i],
                        self.hidden_layer_size[i + 1],
                        self.CFG["dropout"],
                    )
                )
                if self.CFG["batchnorm"]:
                    layers.append(nn.BatchNorm1d(self.hidden_layer_size[i + 1]))
                layers.append(nn.ReLU())

            self.encoder = nn.Sequential(*layers)

            # Graph layers
            if self.CFG["graph_nhead"] == 0:
                self.graph_layer = nn.ModuleList(
                    [GCN(CFG) for _ in range(CFG["num_graph_layers"])]
                )
            else:
                self.graph_layer = nn.ModuleList(
                    [
                        A_GCN(CFG, CFG["graph_nhead"])
                        for _ in range(CFG["num_graph_layers"])
                    ]
                )

            # Decoder
            self.decoder = MLP_AGNN.DecoderMLP(
                self.CFG,
                self.CFG["hidden_dim"],
                self.CFG["dropout"],
                self.CFG["batchnorm"],
            )

        def forward(self, X, graph):

            x = self.encoder(X)
            for layer in self.graph_layer:
                x = layer(x, graph)
            y = self.decoder(x)

            return y

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_encoder_layers: int,
        num_graph_layers: int,
        num_decoder_layers: int,
        graph_nhead: int,
        hidden_dim: int,
        dropout: float,
        mode: str,
        batch_size: int,
        epochs: int,
        loss,
        DataFactory,
        graph="J",
        lr: float = 1e-3,
        random_state: int = 42,
        grad_clip: bool = False,
        batchnorm: bool = False,
        verbose: bool = False,
        rootpath: str = "./",
        name: str = "MLP",
    ):
        """Initialize the MLP model"""

        self.CFG = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_encoder_layers": num_encoder_layers,
            "num_graph_layers": num_graph_layers,
            "num_decoder_layers": num_decoder_layers,
            "graph_nhead": graph_nhead,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "mode": mode,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "random_state": random_state,
            "grad_clip": grad_clip,
            "batchnorm": batchnorm,
            "DataFactory": DataFactory,
            "loss": loss,
            "graph": graph,
            "verbose": verbose,
            "rootpath": rootpath,
            "name": name,
        }
        super().__init__(self.CFG, name=self.CFG["name"])


# TODO: Bottleneck, further_input
