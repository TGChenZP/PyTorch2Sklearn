from PyTorch2Sklearn.__template__ import TorchToSklearn_ImageGraphModel
from PyTorch2Sklearn.Modules import *


class MLP_CNN_AGNN(TorchToSklearn_ImageGraphModel):
    """MLP Classifier or Regressor that can be used as a sklearn model"""

    class DecoderMLP(nn.Module):
        """MLP layers as decoder: Linear -> ReLU -> Dropout (last layer is Linear)"""

        def __init__(self, CFG, hidden_dim, dropout, batchnorm):
            super(MLP_CNN_AGNN.DecoderMLP, self).__init__()

            torch.manual_seed(CFG["random_state"])

            self.CFG = CFG

            mlp_layers = []

            if self.CFG['graph_mode'] == 'concat':
                # Middle layers (if num_decoder_layers > 1)
                for _ in range(CFG["num_decoder_layers"] - 1):

                    if _ == 0:
                        mlp_layers.append(
                            LinearLayer(
                                CFG,
                                CFG["hidden_dim"] * 2,
                                CFG["hidden_dim"],
                                CFG["dropout"],
                            )
                        )
                    else:
                        mlp_layers.append(
                            LinearLayer(
                                CFG,
                                CFG["hidden_dim"],
                                CFG["hidden_dim"],
                                CFG["dropout"],
                            )
                        )

                # Last layer
                mlp_layers.append(nn.Linear(CFG["hidden_dim"]*2, CFG["output_dim"])
                                  if (CFG["graph_mode"] == 'concat' and CFG['num_decoder_layers'] == 1) else nn.Linear(CFG["hidden_dim"], CFG["output_dim"]))

            else:
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
                mlp_layers.append(
                    nn.Linear(CFG["hidden_dim"], CFG["output_dim"]))

            # Combine the layers into one sequential model
            self.out_mlp = nn.Sequential(*mlp_layers)

        def forward(self, X):

            return self.out_mlp(X)

    class Model(nn.Module):
        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            torch.manual_seed(self.CFG["random_state"])

            assert not (self.CFG['freeze_encoder'] and not self.CFG['pretrained']
                        ), "If encoder is frozen, it must be pretrained"

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
                    layers.append(nn.BatchNorm1d(
                        self.hidden_layer_size[i + 1]))
                layers.append(nn.ReLU())

            self.encoder = nn.Sequential(*layers)

            self.projection_mlp = nn.Linear(
                self.CFG['hidden_dim']*2, self.CFG['hidden_dim'])

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
            self.decoder = MLP_CNN_AGNN.DecoderMLP(
                self.CFG,
                self.CFG["hidden_dim"],
                self.CFG["dropout"],
                self.CFG["batchnorm"],
            )

        def forward(self, X, X_img, graph):

            # Pass the image data through the CNN encoder# Process Image
            X_img = self.CNN_encoder(X_img)

            batch_size = X_img.size(0)

            X_img = X_img.reshape(batch_size, -1)

            X_img = self.transition(X_img)

            x = self.encoder(X)

            x = self.projection_mlp(torch.cat((x, X_img), dim=1))

            if self.CFG['graph_mode'] in ['concat', 'residual']:
                x_enc = x.clone()
            for layer in self.graph_layer:
                x = layer(x, graph)

            if self.CFG['graph_mode'] == 'concat':
                x = torch.cat((x_enc, x), dim=1)
            elif self.CFG['graph_mode'] == 'residual':
                x = x + x_enc

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
        epochs: int,
        loss,
        ImageGraphDataFactory,
        cnn_encoder: str,
        freeze_encoder: bool,
        pretrained: bool,
        crop_pretrained_linear: bool,
        graph="J",
        graph_mode: str = "pure",
        lr: float = 1e-3,
        random_state: int = 42,
        grad_clip: bool = False,
        batchnorm: bool = False,
        verbose: bool = False,
        rootpath: str = "./",
        name: str = "MLP_CNN_AGNN",
        input_l: int = 3,
        input_w: int = 224,
        input_c: int = 224,
        nan_break: bool = False,
    ):
        """Initialize the MLP model"""

        self.CFG = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_encoder_layers": num_encoder_layers,
            "num_graph_layers": num_graph_layers,
            "num_decoder_layers": num_decoder_layers,
            "cnn_encoder": cnn_encoder,
            "freeze_encoder": freeze_encoder,
            "pretrained": pretrained,
            "crop_pretrained_linear": crop_pretrained_linear,
            "graph_nhead": graph_nhead,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "mode": mode,
            "epochs": epochs,
            "lr": lr,
            "random_state": random_state,
            "grad_clip": grad_clip,
            "batchnorm": batchnorm,
            "ImageGraphDataFactory": ImageGraphDataFactory,
            "loss": loss,
            "graph": graph,
            "graph_mode": graph_mode,
            "verbose": verbose,
            "rootpath": rootpath,
            "name": name,
            "input_l": input_l,
            "input_w": input_w,
            "input_c": input_c,
            "nan_break": nan_break,
        }
        super().__init__(self.CFG, name=self.CFG["name"])
