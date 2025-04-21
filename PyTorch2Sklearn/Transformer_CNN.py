from PyTorch2Sklearn.__template__ import TorchToSklearn_ImageTabularModel
from PyTorch2Sklearn.Modules import *


class Transformer_CNN(TorchToSklearn_ImageTabularModel):
    """Encoder only transformer Classifier or Regressor that can be used as a sklearn model"""

    class MLPPerFeature(nn.Module):
        """Feature Embedding Layer for input of each feature scalar: Linear -> ReLU -> Dropout"""

        def __init__(self, CFG, hidden_dim, dropout, batchnorm):
            super(Transformer_CNN.MLPPerFeature, self).__init__()

            torch.manual_seed(CFG["random_state"])

            self.CFG = CFG

            if CFG["share_embedding_mlp"]:
                self.shared_mlp = nn.Sequential(
                    LinearLayer(CFG, 1, hidden_dim, dropout),
                    nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity(),
                )
            else:
                self.mlp = nn.ModuleList(
                    [
                        nn.Sequential(
                            LinearLayer(CFG, 1, hidden_dim, dropout),
                            nn.BatchNorm1d(
                                hidden_dim) if batchnorm else nn.Identity(),
                        )
                        for _ in range(CFG["input_dim"])
                    ]
                )

        def forward(self, X):

            if self.CFG["share_embedding_mlp"]:
                # Apply the shared MLP layer to each feature separately
                mlp_output = torch.stack(
                    [self.shared_mlp(X[:, i: i + 1]) for i in range(X.size(1))], dim=1
                )
            else:
                # Apply the MLP layer to each feature separately
                mlp_output = torch.stack(
                    [self.mlp[i](X[:, i: i + 1]) for i in range(X.size(1))], dim=1
                )
            return mlp_output

    class TransformerBlock(nn.Module):
        def __init__(
            self,
            CFG,
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            num_transformer_layers,
        ):
            super(Transformer_CNN.TransformerBlock, self).__init__()

            torch.manual_seed(CFG["random_state"])

            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    # usually 4x hidden_dim, but we set to be tune-able
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                ),
                num_layers=num_transformer_layers,
            )

            self._init_weights()

        def _init_weights(self):
            """Function for initialising weights"""
            for p in self.parameters():
                if p.dim() > 1:
                    init.normal_(p, mean=0, std=0.01)

        def forward(self, X):
            # Reshape the tensor to have each feature as a separate sequence
            reshaped_input = X.view(X.size(0), X.size(1), -1)

            transformer_output = self.transformer(reshaped_input)

            return transformer_output

    class DecoderMLP(nn.Module):
        """MLP layers as decoder: Linear -> ReLU -> Dropout (last layer is Linear)"""

        def __init__(self, CFG, hidden_dim, dropout, batchnorm):
            super(Transformer_CNN.DecoderMLP, self).__init__()

            torch.manual_seed(CFG["random_state"])

            self.CFG = CFG

            mlp_layers = []

            if CFG["agg_transformer_output"] in ['cls', 'mean']:
                input_dim = CFG["hidden_dim"] * (2 if CFG['cnn_concat']
                                                 else 1)
            else:
                input_dim = (CFG["input_dim"]+1) * CFG["hidden_dim"]
            # First layer
            mlp_layers.append(
                LinearLayer(CFG, input_dim, CFG["hidden_dim"], CFG["dropout"])
            )

            # Middle layers (if num_mlp_layers > 2)
            for _ in range(CFG["num_mlp_layers"] - 1):
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

            # run warnings
            self._warning()

            assert not (self.CFG['freeze_encoder'] and not self.CFG['pretrained']
                        ), "If encoder is frozen, it must be pretrained"

            assert self.CFG['agg_transformer_output'] in [
                'cls', 'mean', 'concat'], "agg_transformer_output must be one of ['cls', 'mean', 'concat']"

            self.CNN_encoder = torch.hub.load('pytorch/vision:v0.10.0', self.CFG['cnn_encoder'], pretrained=self.CFG['pretrained']) if type(
                self.CFG['cnn_encoder']) == str else self.CFG['cnn_encoder']

            if self.CFG['input_c'] == 1:
                def adapt_first_conv(conv_layer):
                    new_conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=conv_layer.out_channels,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding,
                        bias=(conv_layer.bias is not None)
                    )
                    with torch.no_grad():
                        new_conv.weight[:] = conv_layer.weight.mean(
                            dim=1, keepdim=True)
                        if conv_layer.bias is not None:
                            new_conv.bias[:] = conv_layer.bias
                    return new_conv

                cnn_name = self.CFG['cnn_encoder']
                if isinstance(cnn_name, str):
                    cnn_name = cnn_name.lower()
                    if 'densenet' in cnn_name:
                        self.CNN_encoder.features.conv0 = adapt_first_conv(
                            self.CNN_encoder.features.conv0)

                    elif any(x in cnn_name for x in ['alexnet', 'squeezenet', 'vgg']):
                        self.CNN_encoder.features[0] = adapt_first_conv(
                            self.CNN_encoder.features[0])

                    elif 'googlenet' in cnn_name or 'inception_v1' in cnn_name:
                        self.CNN_encoder.conv1.conv = adapt_first_conv(
                            self.CNN_encoder.conv1.conv)

                    elif any(x in cnn_name for x in ['mnasnet']):
                        self.CNN_encoder.layers[0] = adapt_first_conv(
                            self.CNN_encoder.layers[0])

                    elif any(x in cnn_name for x in ['resnet', 'resnext']):
                        self.CNN_encoder.conv1 = adapt_first_conv(
                            self.CNN_encoder.conv1)
                    elif any(x in cnn_name for x in ['shufflenet']):
                        self.CNN_encoder.conv1[0] = adapt_first_conv(
                            self.CNN_encoder.conv1[0])
                    elif any(x in cnn_name for x in ['mobilenet']):
                        self.CNN_encoder.features[0][0] = adapt_first_conv(
                            self.CNN_encoder.features[0][0])
                    else:
                        print(
                            "WARNING: First conv layer not patched — unknown model type:", cnn_name)
                else:
                    print(
                        "WARNING: cnn_encoder is not a string — cannot infer model type.")

            if self.CFG['crop_pretrained_linear']:
                self.CNN_encoder = nn.Sequential(
                    *list(self.CNN_encoder.children())[:-1])

            sample_input = torch.randn(
                [1, self.CFG['input_c'], self.CFG['input_l'], self.CFG['input_w']])
            sample_output = self.CNN_encoder(sample_input)

            flatten_shape = np.prod(sample_output.shape[1:])

            # Transition layer to match the hidden_dim
            self.transition = nn.Linear(flatten_shape, self.CFG['hidden_dim'])

            for param in self.CNN_encoder.parameters():
                param.requires_grad = not self.CFG['freeze_encoder']

            # MLP layer for each feature
            self.mlp_per_feature = Transformer_CNN.MLPPerFeature(
                CFG, CFG["hidden_dim"], CFG["dropout"], CFG["batchnorm"]
            )

            # Transformer block
            self.transformer_block = Transformer_CNN.TransformerBlock(
                CFG,
                CFG["hidden_dim"],
                CFG["nhead"],
                CFG["dim_feedforward"],
                CFG["dropout"],
                CFG["num_transformer_layers"],
            )

            # MLP layers as decoder
            self.out_mlp = Transformer_CNN.DecoderMLP(
                CFG, CFG["hidden_dim"], CFG["dropout"], CFG["batchnorm"]
            )

        def forward(self, X, X_img):

            # Process Image
            X_img = self.CNN_encoder(X_img)

            batch_size = X_img.size(0)

            X_img = X_img.reshape(batch_size, -1)

            X_img = self.transition(X_img)

            # Forward pass through MLP layer for each feature
            mlp_output = self.mlp_per_feature(X)

            if self.CFG["agg_transformer_output"] == 'cls':
                # Add an extra hidden_dim vector (cls) to the front of mlp_output
                mlp_output = torch.cat(
                    [
                        torch.zeros(X.size(0), 1, self.CFG["hidden_dim"]).to(
                            X.device),
                        mlp_output,
                    ],
                    dim=1,
                )

            # concat before transformer, as don't want to concat after MLP
            if not self.CFG['cnn_concat']:
                X_img = X_img.view(batch_size, 1, -1)
                # Concatenate the output from the CNN and the MLP
                mlp_output = torch.cat((mlp_output, X_img), dim=1)

            transformer_output = self.transformer_block(mlp_output)

            if self.CFG["agg_transformer_output"] == 'cls':  # predict just using cls
                y = self.out_mlp(torch.cat(
                    [transformer_output[:, 0, :], X_img], dim=1) if self.CFG['cnn_concat'] else transformer_output[:, 0, :])
            elif self.CFG["agg_transformer_output"] == 'mean':  # mean of all layers
                y = self.out_mlp(
                    torch.mean(transformer_output, dim=1) if not self.CFG['cnn_concat'] else torch.cat(
                        [torch.mean(transformer_output, dim=1), X_img], dim=1)
                )
            else:  # concatenate the output from all layers in transformer_output

                y = self.out_mlp(
                    torch.cat(
                        [
                            transformer_output[:, i, :]
                            for i in range(transformer_output.size(1))
                        ]+[X_img] if self.CFG['cnn_concat'] else [
                            transformer_output[:, i, :]
                            for i in range(transformer_output.size(1))
                        ],
                        dim=1,
                    )
                )

            return y

        def _warning(self):

            if self.CFG["agg_transformer_output"] == 'cls' and self.CFG["num_transformer_layers"] == 1:
                print(
                    "Warning: Setting agg_transformer_output to True with num_transformer_layers=1 is not recommended."
                    "The model will only be able to predict using the first feature token and will likely result in no learning/0R model"
                )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_transformer_layers: int,
        num_mlp_layers: int,
        hidden_dim: int,
        dropout: float,
        nhead: int,
        mode: str,
        batch_size: int,
        epochs: int,
        loss,
        TabularImageDataFactory,
        TabularImageDataset,
        cnn_encoder: str,
        freeze_encoder: bool,
        pretrained: bool,
        crop_pretrained_linear: bool,
        agg_transformer_output: str,
        share_embedding_mlp: bool = False,
        cnn_concat: bool = False,
        dim_feedforward: int = None,
        lr: float = 1e-3,
        random_state: int = 42,
        grad_clip: bool = False,
        batchnorm: bool = False,
        verbose: bool = False,
        rootpath: str = "./",
        name: str = "Transformer",
        input_l: int = 3,
        input_w: int = 224,
        input_c: int = 224,
        nan_break: bool = False,
    ):
        """Initialize the Transformer model"""

        dim_feedforward = 4 * hidden_dim if dim_feedforward is None else dim_feedforward

        self.CFG = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_transformer_layers": num_transformer_layers,
            "num_mlp_layers": num_mlp_layers,
            "hidden_dim": hidden_dim,
            "dim_feedforward": dim_feedforward,
            "cnn_concat": cnn_concat,
            "cnn_encoder": cnn_encoder,
            "freeze_encoder": freeze_encoder,
            "pretrained": pretrained,
            "crop_pretrained_linear": crop_pretrained_linear,
            "nhead": nhead,
            "agg_transformer_output": agg_transformer_output,
            "dropout": dropout,
            "mode": mode,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "random_state": random_state,
            "grad_clip": grad_clip,
            "batchnorm": batchnorm,
            "loss": loss,
            "TabularImageDataFactory": TabularImageDataFactory,
            "TabularImageDataset": TabularImageDataset,
            "verbose": verbose,
            "rootpath": rootpath,
            "share_embedding_mlp": share_embedding_mlp,
            "name": name,
            "input_l": input_l,
            "input_w": input_w,
            "input_c": input_c,
            "nan_break": nan_break,
        }

        super().__init__(self.CFG, name=self.CFG["name"])
