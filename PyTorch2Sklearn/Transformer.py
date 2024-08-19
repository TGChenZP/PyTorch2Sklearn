from PyTorch2Sklearn.__template__ import TorchToSklearn_Model
from PyTorch2Sklearn.Modules import *


class Transformer(TorchToSklearn_Model):

    """ Encoder only transformer Classifier or Regressor that can be used as a sklearn model """

    class MLPPerFeature(nn.Module):
        """ Feature Embedding Layer for input of each feature scalar: Linear -> ReLU -> Dropout """

        def __init__(self, CFG, hidden_dim, dropout, batchnorm):
            super(Transformer.MLPPerFeature, self).__init__()

            torch.manual_seed(CFG['random_state'])

            self.shared_mlp = nn.Sequential(
                LinearLayer(CFG, 1, hidden_dim, dropout),
                nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity()
            )

        def forward(self, X):

            # Apply the shared MLP layer to each feature separately
            mlp_output = torch.stack(
                [self.shared_mlp(X[:, i:i+1]) for i in range(X.size(1))], dim=1)
            return mlp_output

    class TransformerBlock(nn.Module):
        def __init__(self, CFG, hidden_dim, nhead, dim_feedforward, dropout, num_transformer_layers):
            super(Transformer.TransformerBlock, self).__init__()

            torch.manual_seed(CFG['random_state'])

            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    # usually 4x hidden_dim, but we set to be tune-able
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ),
                num_layers=num_transformer_layers
            )

            self._init_weights()

        def _init_weights(self):
            """ Function for initialising weights """
            for p in self.parameters():
                if p.dim() > 1:
                    init.normal_(p, mean=0, std=0.01)

        def forward(self, X):
            # Reshape the tensor to have each feature as a separate sequence
            reshaped_input = X.view(X.size(0), X.size(1), -1)

            transformer_output = self.transformer(reshaped_input)

            return transformer_output

    class Model(nn.Module):
        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            # MLP layer for each feature
            self.mlp_per_feature = Transformer.MLPPerFeature(
                CFG, CFG['hidden_dim'], CFG['dropout'], CFG['batchnorm'])

            # Transformer block
            self.transformer_block = Transformer.TransformerBlock(
                CFG, CFG['hidden_dim'], CFG['nhead'], CFG['dim_feedforward'], CFG['dropout'], CFG['num_transformer_layers'])

            # MLP layer for final output
            # just take the 0th vector of the last layer (cls) in transformer_output
            if self.CFG['use_cls']:
                self.out_mlp = LinearLayer(
                    self.CFG, self.CFG['hidden_dim'], self.CFG['output_dim'])
            else:  # concatenate the output from all layers in transformer_output
                self.out_mlp = LinearLayer(
                    self.CFG, self.CFG['input_dim'] * self.CFG['hidden_dim'], self.CFG['output_dim'])

        def forward(self, X):

            # Forward pass through MLP layer for each feature
            mlp_output = self.mlp_per_feature(X)

            if self.CFG['use_cls']:
                # Add an extra hidden_dim vector (cls) to the front of mlp_output
                mlp_output = torch.cat([torch.zeros(
                    X.size(0), 1, self.CFG['hidden_dim']).to(X.device), mlp_output], dim=1)

            transformer_output = self.transformer_block(mlp_output)

            if self.CFG['use_cls']:  # predict just using cls
                y = self.out_mlp(transformer_output[:, 0, :])
            else:  # concatenate the output from all layers in transformer_output
                y = self.out_mlp(torch.cat(
                    [transformer_output[:, i, :] for i in range(transformer_output.size(1))], dim=1))

            return y

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_transformer_layers,
                 hidden_dim,
                 dim_feedforward,
                 dropout,
                 nhead,
                 use_cls,
                 mode,
                 batch_size,
                 epochs,
                 loss,
                 TabularDataFactory,
                 TabularDataset,
                 lr=1e-3,
                 random_state=42,
                 grad_clip=False,
                 batchnorm=False,
                 verbose=False,
                 rootpath='./',
                 name="Transformer"):

        self.CFG = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'num_transformer_layers': num_transformer_layers,
            'hidden_dim': hidden_dim,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead,
            'use_cls': use_cls,
            'dropout': dropout,
            'mode': mode,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'random_state': random_state,
            'grad_clip': grad_clip,
            'batchnorm': batchnorm,
            'loss': loss,
            'TabularDataFactory': TabularDataFactory,
            'TabularDataset': TabularDataset,
            'verbose': verbose,
            'rootpath': rootpath,
            'name': name
        }

        super().__init__(self.CFG, name=self.CFG['name'])
