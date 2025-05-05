from PyTorch2Sklearn.__template__ import TorchToSklearn_ImageTabularModel
from PyTorch2Sklearn.Modules import *


class MLP(TorchToSklearn_ImageTabularModel):
    """MLP Classifier or Regressor that can be used as a sklearn model"""

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

            encoder_layers = []
            # Input layer
            encoder_layers.append(
                LinearLayer(
                    self.CFG,
                    self.CFG["input_dim"],
                    self.CFG["hidden_dim"],
                    self.CFG["dropout"],
                )
            )
            if self.CFG["batchnorm"]:
                encoder_layers.append(nn.BatchNorm1d(self.CFG["hidden_dim"]))
            encoder_layers.append(nn.ReLU())

            # Hidden layers
            for i in range(self.CFG["encoder_hidden_layers"] - 1):
                encoder_layers.append(
                    LinearLayer(
                        self.CFG,
                        self.CFG["hidden_dim"],
                        self.CFG["hidden_dim"],
                        self.CFG["dropout"],
                    )
                )
                if self.CFG["batchnorm"]:
                    encoder_layers.append(nn.BatchNorm1d(
                        self.CFG["hidden_dim"]))
                encoder_layers.append(nn.ReLU())

            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers = []
            decoder_layers.append(
                LinearLayer(
                    self.CFG,
                    self.CFG["hidden_dim"]*2,
                    self.CFG["hidden_dim"],
                    self.CFG["dropout"],
                )
            )
            if self.CFG["batchnorm"]:
                decoder_layers.append(nn.BatchNorm1d(self.CFG["hidden_dim"]))
            decoder_layers.append(nn.ReLU())

            # Hidden layers
            for i in range(self.CFG["decoder_hidden_layers"] - 1):
                decoder_layers.append(
                    LinearLayer(
                        self.CFG,
                        self.CFG["hidden_dim"],
                        self.CFG["hidden_dim"],
                        self.CFG["dropout"],
                    )
                )
                if self.CFG["batchnorm"]:
                    decoder_layers.append(nn.BatchNorm1d(
                        self.CFG["hidden_dim"]))
                decoder_layers.append(nn.ReLU())

            # Output layer
            decoder_layers.append(
                LinearLayer(
                    self.CFG,
                    self.CFG["hidden_dim"],
                    self.CFG["output_dim"],
                    self.CFG["dropout"],
                )
            )

            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, X, X_img):

            y = self.full_model(X)

            X_img = self.CNN_encoder(X_img)

            batch_size = X_img.size(0)

            X_img = X_img.reshape(batch_size, -1)

            X_img = self.transition(X_img)

            X_enc = self.encoder(X)

            X_enc = torch.cat((X_enc, X_img), dim=1)

            y = self.decoder(X_enc)

            return y

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden_layers: int,
        decoder_hidden_layers: int,
        hidden_dim: int,
        dropout: float,
        mode: str,
        batch_size: int,
        epochs: int,
        loss,
        TabularDataFactory,
        TabularDataset,
        cnn_encoder: str,
        freeze_encoder: bool,
        pretrained: bool,
        crop_pretrained_linear: bool,
        lr: float = 1e-3,
        random_state: int = 42,
        grad_clip: bool = False,
        batchnorm: bool = False,
        verbose: bool = False,
        rootpath: str = "./",
        name: str = "MLP_CNN",
        input_l: int = 3,
        input_w: int = 224,
        input_c: int = 224,
        nan_break: bool = False,
    ):
        """Initialize the MLP model"""

        self.CFG = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "encoder_hidden_layers": encoder_hidden_layers,
            "decoder_hidden_layers": decoder_hidden_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "mode": mode,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "random_state": random_state,
            "grad_clip": grad_clip,
            "batchnorm": batchnorm,
            "loss": loss,
            "TabularDataFactory": TabularDataFactory,
            "TabularDataset": TabularDataset,
            "cnn_encoder": cnn_encoder,
            "freeze_encoder": freeze_encoder,
            "pretrained": pretrained,
            "crop_pretrained_linear": crop_pretrained_linear,
            "verbose": verbose,
            "rootpath": rootpath,
            "name": name,
            "input_l": input_l,
            "input_w": input_w,
            "input_c": input_c,
            "nan_break": nan_break,
        }
        super().__init__(self.CFG, name=self.CFG["name"])
