from PyTorch2Sklearn.__template__ import TorchToSklearn_Model
from PyTorch2Sklearn.Modules import *


class CNN(TorchToSklearn_Model):
    """CNN Classifier or Regressor that can be used as a sklearn model"""

    class DecoderMLP(nn.Module):
        """MLP layers as decoder: Linear -> ReLU -> Dropout (last layer is Linear)"""

        def __init__(self, CFG, hidden_dim, dropout):
            super(CNN.DecoderMLP, self).__init__()

            torch.manual_seed(CFG["random_state"])

            self.CFG = CFG

            mlp_layers = []

            # Transformer-CNN: will need to be different
            input_dim = hidden_dim

            # First layer
            mlp_layers.append(
                LinearLayer(CFG, input_dim, CFG['hidden_dim'], CFG["dropout"])
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

            assert not (self.CFG['freeze_encoder'] and not self.CFG['pretrained']
                        ), "If encoder is frozen, it must be pretrained"

            torch.manual_seed(self.CFG["random_state"])

            self.encoder = torch.hub.load('pytorch/vision:v0.10.0', self.CFG['cnn_encoder'], pretrained=self.CFG['pretrained']) if type(
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
                        self.encoder.features.conv0 = adapt_first_conv(
                            self.encoder.features.conv0)

                    elif any(x in cnn_name for x in ['alexnet', 'squeezenet', 'vgg']):
                        self.encoder.features[0] = adapt_first_conv(
                            self.encoder.features[0])

                    elif 'googlenet' in cnn_name or 'inception_v1' in cnn_name:
                        self.encoder.conv1.conv = adapt_first_conv(
                            self.encoder.conv1.conv)

                    elif any(x in cnn_name for x in ['mnasnet']):
                        self.encoder.layers[0] = adapt_first_conv(
                            self.encoder.layers[0])

                    elif any(x in cnn_name for x in ['resnet', 'resnext']):
                        self.encoder.conv1 = adapt_first_conv(
                            self.encoder.conv1)
                    elif any(x in cnn_name for x in ['shufflenet']):
                        self.encoder.conv1[0] = adapt_first_conv(
                            self.encoder.conv1[0])
                    elif any(x in cnn_name for x in ['mobilenet']):
                        self.encoder.features[0][0] = adapt_first_conv(
                            self.encoder.features[0][0])
                    else:
                        print(
                            "WARNING: First conv layer not patched — unknown model type:", cnn_name)
                else:
                    print(
                        "WARNING: cnn_encoder is not a string — cannot infer model type.")

            if self.CFG['crop_pretrained_linear']:
                self.encoder = nn.Sequential(
                    *list(self.encoder.children())[:-1])

            sample_input = torch.randn(
                [1, self.CFG['input_c'], self.CFG['input_l'], self.CFG['input_w']])
            sample_output = self.encoder(sample_input)

            flatten_shape = np.prod(sample_output.shape[1:])

            self.transition = nn.Linear(flatten_shape, self.CFG['hidden_dim'])

            for param in self.encoder.parameters():
                param.requires_grad = not self.CFG['freeze_encoder']

            self.out_mlp = CNN.DecoderMLP(
                self.CFG,
                self.CFG['hidden_dim'],
                self.CFG["dropout"],
            )

        def forward(self, X_img):

            X_img = self.encoder(X_img)

            batch_size = X_img.size(0)

            X_img = X_img.reshape(batch_size, -1)

            X_img = self.transition(X_img)

            y = self.out_mlp(X_img)

            return y

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        cnn_encoder: str,
        freeze_encoder: bool,
        pretrained: bool,
        crop_pretrained_linear: bool,
        num_mlp_layers: int,
        dropout: float,
        mode: str,
        batch_size: int,
        epochs: int,
        loss,
        TabularDataFactory,
        TabularDataset,
        lr: float = 1e-3,
        random_state: int = 42,
        batchnorm=False,
        grad_clip: bool = False,
        verbose: bool = False,
        rootpath: str = "./",
        name: str = "CNN",
        input_l: int = 3,
        input_w: int = 224,
        input_c: int = 224,
        nan_break: bool = False,
    ):
        """Initialize the CNN model"""

        self.CFG = {
            "input_l": input_l,
            "input_w": input_w,
            "input_c": input_c,
            "nan_break": nan_break,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "cnn_encoder": cnn_encoder,
            "freeze_encoder": freeze_encoder,
            "pretrained": pretrained,
            'crop_pretrained_linear': crop_pretrained_linear,
            "num_mlp_layers": num_mlp_layers,
            "dropout": dropout,
            "mode": mode,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "random_state": random_state,
            "batchnorm": batchnorm,
            "grad_clip": grad_clip,
            "loss": loss,
            "TabularDataFactory": TabularDataFactory,
            "TabularDataset": TabularDataset,
            "verbose": verbose,
            "rootpath": rootpath,
            "name": name,
        }
        super().__init__(self.CFG, name=self.CFG["name"])
