# PyTorch2Sklearn
```
Author GitHub: https://github.com/TGChenZP
```

*Please cite when using this package for research and other machine learning purposes*

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Model Architectures](#model-architectures)
   - [PyTorch2Sklearn.MLP](#pytorch2sklearnmlp-source)
   - [PyTorch2Sklearn.Transformer](#pytorch2sklearntransformer-source)
4. [Methods](#methods-source)
5. [Usage Examples](#usage-examples)
   - [Regression Example](#regression-example)
        - [MLP Regression Example](#mlp-regression-example)
        - [Transformer Regression Example](#transformer-regression-example)
   - [Classification Example](#classification-example)
        - [MLP Classification Example](#mlp-classification-example)
        - [Transformer Classification Example](#transformer-classification-example)



# Introduction
This package wraps PyTorch MLP and Transformer in an Sklearn style API. It is designed for tabular data supervised learning (classification and regression) with hyperparmeter control in-built for most typical Deep Neural Network architectural design.

Both regression and classification is defined under the same class - specify use case by `mode`. Remember also to set appropriate `loss` from `torch.nn`, and also set appropriate `input_dim` (number of columns in tabular data) and `output_dim` (output dimension of regression, or number of classes in classification task)


# Installation
```bash
pip install PyTorch2Sklearn
```

# Model Architectures
## PyTorch2Sklearn.MLP [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP.py)

```python
class PyTorch2Sklearn.MLP.MLP(input_dim, output_dim, hidden_layers, hidden_dim, dropout, mode, batch_size, epochs, loss, TabularDataFactory, TabularDataset, lr=1e-3, random_state=42, grad_clip=False, batch_norm=False, verbose=False, rootpath='./', name='MLP', **kwargs)
```

### Parameters
| **Parameter & Type**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `hidden_layers` (`int`)                           | The number of hidden layers in the MLP.                                                                                                                                |
| `hidden_dim` (`int`)                              | The number of neurons in each hidden layer.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `batch_size` (`int`)                              | The batch size.                                                                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batch_norm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `TabularDataFactory` (`PyTorch2Sklearn.utils.data.TabularDataFactory`) | The tabular data factory that transforms data from input format into the correct format for TabularDataset.                                                            |
| `TabularDataset` (`PyTorch2Sklearn.utils.data.TabularDataset`)         | The dataset object that generates batches for stochastic gradient descent.                                                                                             |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"MLP"`)         | The name of the model.                                                                                                                                                 |



## PyTorch2Sklearn.Transformer [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer.py)

```python
class PyTorch2Sklearn.Transformer.Transformer(input_dim, output_dim, num_transformer_layers, num_mlp_layers, hidden_dim, dropout, nhead, mode, batch_size, epochs, loss, TabularDataFactory, TabularDataset, share_embedding_mlp=False, use_cls=False, dim_feedforward=None, lr=1e-3, random_state=42, grad_clip=False, batchnorm=False, verbose=False, rootpath='./', name='Transformer', **kwargs)
```

### Parameters
| **Parameter**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `num_transformer_layers` (`int`)                  | The number of transformer layers.                                                                                                                                      |
| `num_mlp_layers` (`int`)                          | The number of MLP layers.                                                                                                                                              |
| `hidden_dim` (`int`)                              | The number of neurons in the hidden layers.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `nhead` (`int`)                                   | The number of heads in the multiheadattention models.                                                                                                                  |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `batch_size` (`int`)                              | The batch size.                                                                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batch_norm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `TabularDataFactory` (`PyTorch2Sklearn.utils.data.TabularDataFactory`) | The tabular data factory that transforms data from input format into the correct format for TabularDataset.                                                            |
| `TabularDataset` (`PyTorch2Sklearn.utils.data.TabularDataset`)         | The dataset object that generates batches for stochastic gradient descent.                                                                                             |
| `share_embedding_mlp` (`bool`, optional, default=`False`)              | Whether to share the embedding layer in the MLP.                                                                                                                       |
| `use_cls` (`bool`, optional, default=`False`)     | Whether to use the CLS token to feed into the decoder, or concatenate all vectors outputted in the final transformer layer.                                             |
| `dim_feedforward` (`int`, optional, default=`None`) | The hidden dimension in the feedforward network.                                                                                                                       |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"Transformer"`) | The name of the model.                                                                                                                                                 |


# Methods [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/__template__.py)
`_init__([input_dim, output_dim, ...])`: Construct a PyTorch2Sklearn model class

`fit(train_x, train_y)`: fit the model using data

`predict(val_x)`: make inference on features of new data

`predict_proba(val_x)`: make inference (probabilities of each class) for new data [WARNING: only available for classification]

`save(mark)`: save the model parameters

`load(mark)`: load the model parameters

# *Usage Examples*
## *Regression Example*

### *MLP Regression Example*
```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.MLP import MLP
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
from sklearn.metrics import accuracy_score, r2_score

X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42)
X = pd.DataFrame(
    X_reg, columns=[f'feature_{i+1}' for i in range(X_reg.shape[1])])
y = pd.Series(y_reg, name='target')

model = MLP(
        hidden_dim=16,
        hidden_layers=1,
        dropout=0.1,
        batch_size=32,
        epochs=5,
        lr=1e-3,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss(),
        mode='Regression',
        name='MLP',
        verbose=1,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath='./',
        output_dim=1,
        input_dim=5
    )

model.fit(X, y)

print(r2_score(y, model.predict(X)))
```

### *Transformer Regression Example*
```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.MLP import MLP
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
from sklearn.metrics import accuracy_score, r2_score

X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42)
X = pd.DataFrame(
    X_reg, columns=[f'feature_{i+1}' for i in range(X_reg.shape[1])])
y = pd.Series(y_reg, name='target')

model = Transformer(
        hidden_dim=16,
        num_transformer_layers=1,
        num_mlp_layers=1,
        dropout=0.1,
        batch_size=32,
        share_embedding_mlp=False,
        nhead=2,
        use_cls=False,
        epochs=5,
        lr=1e-3,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss(),
        mode='Regression',
        name='Transformer',
        verbose=1,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath='./',
        output_dim=1,
        input_dim=5
    )

model.fit(X, y)

print(r2_score(y, model.predict(X)))
```

## *Classification Example*

### *MLP Classification Example*
```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.MLP import MLP
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
from sklearn.metrics import accuracy_score, r2_score

X_class_2, y_class_2 = make_classification(
        n_samples=100, n_features=5, n_classes=2, n_clusters_per_class=1, random_state=42)
X = pd.DataFrame(
    X_class_2, columns=[f'feature_{i+1}' for i in range(X_class_2.shape[1])])
y = pd.Series(y_class_2, name='target')

model = MLP(
        hidden_dim=16,
        hidden_layers=1,
        dropout=0.1,
        batch_size=32,
        epochs=5,
        lr=1e-3,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.CrossEntropyLoss(),
        mode='Classification',
        name='MLP',
        verbose=1,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath='./',
        output_dim=2,
        input_dim=5
    )

model.fit(X, y)

print(r2_score(y, model.predict(X)))
```

### *Transformer Classification Example*
```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.MLP import MLP
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
from sklearn.metrics import accuracy_score, r2_score

X_class_2, y_class_2 = make_classification(
        n_samples=100, n_features=5, n_classes=2, n_clusters_per_class=1, random_state=42)
X = pd.DataFrame(
    X_class_2, columns=[f'feature_{i+1}' for i in range(X_class_2.shape[1])])
y = pd.Series(y_class_2, name='target')

model = Transformer(
        hidden_dim=16,
        num_transformer_layers=1,
        num_mlp_layers=1,
        dropout=0.1,
        batch_size=32,
        share_embedding_mlp=False,
        nhead=2,
        use_cls=False,
        epochs=5,
        lr=1e-3,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.CrossEntropyLoss(),
        mode='Classification',
        name='Transformer',
        verbose=1,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath='./',
        output_dim=2,
        input_dim=5
    )

model.fit(X, y)

print(r2_score(y, model.predict(X)))
```