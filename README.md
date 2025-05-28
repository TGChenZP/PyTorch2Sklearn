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
   - [PyTorch2Sklearn.MLP_AGNN](#pytorch2sklearnmlp_agnn-source)
   - [PyTorch2Sklearn.Transformer_AGNN](#pytorch2sklearntransformer_agnn-source)
   - [PyTorch2Sklearn.CNN](#pytorch2sklearncnn-source)
   - [PyTorch2Sklearn.MLP_CNN](#pytorch2sklearnmlp_cnn-source)
   - [PyTorch2Sklearn.Transformer_CNN](#pytorch2sklearntransformer_cnn-source)
   - [PyTorch2Sklearn.MLP_CNN_AGNN](#pytorch2sklearnmlp_cnn_agnn-source)
   - [PyTorch2Sklearn.Transformer_CNN_AGNN](#pytorch2sklearntransformer_cnn_agnn-source)
4. [Methods](#methods-source)
5. [Usage Examples](#usage-examples)
   - [Regression Example](#regression-example)
        - [MLP Regression Example](#mlp-regression-example)
        - [Transformer Regression Example](#transformer-regression-example)
        - [MLP_AGNN Regression Example](#mlp_agnn-regression-example)
        - [Transformer_AGNN Regression Example](#transformer_agnn-regression-example)
        - [PyTorch2Sklearn.CNN](#cnn-regression-example)
        - [PyTorch2Sklearn.MLP_CNN](#mlp_cnn-regression-example)
        - [PyTorch2Sklearn.Transformer_CNN](#transformer_cnn-regression-example)
        - [PyTorch2Sklearn.MLP_CNN_AGNN](#mlp_cnn_agnn-regression-example)
        - [PyTorch2Sklearn.Transformer_CNN_AGNN](#transformer_cnn_agnn-regression-example)
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
class PyTorch2Sklearn.MLP.MLP(input_dim, output_dim, hidden_layers, hidden_dim, dropout, mode, batch_size, epochs, loss, TabularDataFactory, TabularDataset, lr=1e-3, random_state=42, grad_clip=False, batchnorm=False, verbose=False, rootpath='./', name='MLP', nan_break = False, **kwargs)
```

### Parameters
| **Parameter & Type**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `hidden_layers` (`int`)                           | The number of hidden layers in the MLP. If set to `0`, will shrink hidden layers at arithmetic differences from input_dim to output_dim                                                                                                                               |
| `hidden_dim` (`int`)                              | The number of neurons in each hidden layer.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `batch_size` (`int`)                              | The batch size.                                                                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `TabularDataFactory` (`PyTorch2Sklearn.utils.data.TabularDataFactory`) | The tabular data factory that transforms data from input format into the correct format for TabularDataset.                                                            |
| `TabularDataset` (`PyTorch2Sklearn.utils.data.TabularDataset`)         | The dataset object that generates batches for stochastic gradient descent.                                                                                             |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"MLP"`)         | The name of the model.                                                                                                                                                 |
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |  



## PyTorch2Sklearn.Transformer [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer.py)

```python
class PyTorch2Sklearn.Transformer.Transformer(input_dim, output_dim, num_transformer_layers, num_mlp_layers, hidden_dim, dropout, nhead, mode, batch_size, epochs, loss, TabularDataFactory, TabularDataset, agg_transformer_output, share_embedding_mlp=False, dim_feedforward=None, lr=1e-3, random_state=42, grad_clip=False, batchnorm=False, verbose=False, rootpath='./', name='Transformer',  nan_break = False, **kwargs)
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
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `TabularDataFactory` (`PyTorch2Sklearn.utils.data.TabularDataFactory`) | The tabular data factory that transforms data from input format into the correct format for TabularDataset.                                                            |
| `TabularDataset` (`PyTorch2Sklearn.utils.data.TabularDataset`)         | The dataset object that generates batches for stochastic gradient descent.                                                                                             |
| `share_embedding_mlp` (`bool`, optional, default=`False`)              | Whether to share the embedding layer in the MLP.                                                                                                                       |
| `agg_transformer_output` (`str`)                              | how to process the transformer layer outputs before next module. Must be in 'cls' (just use first embedding cls), 'mean' (average the embeddings) and 'concat' (concat all the output layer embeddings together)                                                          |
| `dim_feedforward` (`int`, optional, default=`None`) | The hidden dimension in the feedforward network.                                                                                                                       |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"Transformer"`) | The name of the model.                                                                                                                                                 |
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |  

## PyTorch2Sklearn.MLP_AGNN [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP_AGNN.py)

```python
class PyTorch2Sklearn.MLP_AGNN.MLP_AGNN(input_dim, output_dim, num_encoder_layers, num_graph_layers, num_decoder_layers, graph_nhead, hidden_dim, dropout, mode, epochs, loss, DataFactory, graph="J", graph_mode='pure', lr=1e-3, random_state=42, grad_clip=False, batchnorm=False, verbose=False, rootpath='./', name='MLP_AGNN', nan_break = False, **kwargs)
```

### Parameters
| **Parameter & Type**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `num_encoder_layers` (`int`)                          | The number of encoder mlp layers.                                                                                                                                              |
| `num_decoder_layers` (`int`)                          | The number of decoder mlp layers.                                                                                                                                              |
| `num_graph_layers` (`int`)                          | The number of graph layers.                                                                                                                                              |
| `graph_nhead` (`int`)                          | The number of attention heads in graph attention layer.                                                                                                                                              |
| `hidden_dim` (`int`)                              | The number of neurons in each hidden layer.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `GraphDataFactory` (`PyTorch2Sklearn.utils.data.GraphDataFactory`) | The graph data factory that transforms data from input format into the correct format for training.                                                            |
| `graph` (optional, default = `"J"`) | if `"J"`, then every batch will be inferenced with graph = J (1T 1); if `"U"`, then every batch will be inferencd with uniform graph (1/n 1T 1). Also accepts manually defined graph. |
| `graph_mode` (optional, default=`"pure"`) | if `"pure"`, just take graph embedding; if `"residual"` add encoder output and graph output together; if `"concat"` concat encoder output and graph output |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"MLP_AGNN"`)         | The name of the model.                                                                                                                                                 |
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |  



## PyTorch2Sklearn.Transformer_AGNN [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer_AGNN.py)

```python
class PyTorch2Sklearn.Transformer_AGNN.Transformer_AGNN(input_dim, output_dim, num_transformer_layers, num_graph_layers, num_mlp_layers, hidden_dim, dropout, nhead, graph_nhead, mode, epochs, loss, agg_transformer_output, DataFactory, graph="J", graph_mode='pure', share_embedding_mlp=False, dim_feedforward=None, lr=1e-3, random_state=42, grad_clip=False, batchnorm=False, verbose=False, rootpath='./', name='Transformer_AGNN', nan_break = False, **kwargs)
```

### Parameters
| **Parameter**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `num_transformer_layers` (`int`)                  | The number of transformer layers.                                                                                                                                      |
| `num_mlp_layers` (`int`)                          | The number of MLP layers.                                                                                                                                              |
| `num_graph_layers` (`int`)                          | The number of graph layers.                                                                                                                                              |
| `graph_nhead` (`int`)                          | The number of attention heads in graph attention layer.                                                                                                                                              |
| `hidden_dim` (`int`)                              | The number of neurons in the hidden layers.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `nhead` (`int`)                                   | The number of heads in the multiheadattention models.                                                                                                                  |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `GraphDataFactory` (`PyTorch2Sklearn.utils.data.GraphDataFactory`) | The graph data factory that transforms data from input format into the correct format for training.                                                            |
| `graph` (optional, default = `"J"`) | if `"J"`, then every batch will be inferenced with graph = J (1T 1); if `"U"`, then every batch will be inferencd with uniform graph (1/n 1T 1). Also accepts manually defined graph. |
| `graph_mode` (optional, default=`"pure"`) | if `"pure"`, just take graph embedding; if `"residual"` add encoder output and graph output together; if `"concat"` concat encoder output and graph output |
| `share_embedding_mlp` (`bool`, optional, default=`False`)              | Whether to share the embedding layer in the MLP.                                                                                                                       |
| `agg_transformer_output` (`str`)                              | how to process the transformer layer outputs before next module. Must be in 'cls' (just use first embedding cls), 'mean' (average the embeddings) and 'concat' (concat all the output layer embeddings together)                                                          |
| `dim_feedforward` (`int`, optional, default=`None`) | The hidden dimension in the feedforward network.                                                                                                                       |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"Transformer"`) | The name of the model.     |
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |             

## PyTorch2Sklearn.CNN [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/CNN.py)

```python
class PyTorch2Sklearn.CNN.CNN(output_dim, hidden_dim, cnn_encoder, freeze_encoder, pretrained, crop_pretrained_linear, num_mlp_layers, dropout, mode, batch_size, epochs, loss, TabularDataFactory, TabularDataset, lr = 1e-3, random_state = 42, batchnorm=False, grad_clip = False, verbose = False, rootpath =  "./", name = "CNN", input_l = 3, input_w = 224, input_c = 224, nan_break = False, **kwargs)
```

### Parameters
| **Parameter**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `cnn_encoder` (`str`)                              | name of CNN encoder from pytorch/vision:v0.10.0; alternatively an image encoder in the form of PyTorch object. (**note**: parameter name misrepresenting as CNN are no longer the only type of image encoder)                                                                                                          |
| `freeze_encoder` (`bool`)                              | if True, do not allow weights of image encoder (cnn_encoder) to update during training                                                                                                        |
| `pretrained` (`bool`)                              | if True, load pretrained weights from pytorch/vision:v0.10.0                                                                                                        |
| `crop_pretrained_linear` (`bool`)                              | if True, crop linear head from loaded image model (cnn_encoder) model                                                                                                        |
| `batch_size` (`int`)                              | The batch size.                                                                                                                                                        |
| `num_mlp_layers` (`int`)                          | The number of MLP layers.                                                                                                                                              |
| `num_graph_layers` (`int`)                          | The number of graph layers.                                                                                                                                              |
| `hidden_dim` (`int`)                              | The number of neurons in the hidden layers.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `TabularDataFactory` (`PyTorch2Sklearn.utils.data.TabularDataFactory`) | The tabular data factory that transforms data from input format into the correct format for TabularDataset.                                                            |
| `TabularDataset` (`PyTorch2Sklearn.utils.data.TabularDataset`)         | The dataset object that generates batches for stochastic gradient descent.                                                                                             |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"CNN"`) | The name of the model.         |
| `input_l` (`int` optional, default=`224`) | input image length                           |  
| `input_w` (`int` optional, default=`224`) | input image width                           |    
| `input_c` (`int`optional, default=`3`) | input image channels                           | 
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |                         

## PyTorch2Sklearn.MLP_CNN [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP_CNN.py)

```python
class PyTorch2Sklearn.MLP_CNN.MLP_CNN(input_dim, output_dim, encoder_hidden_layers, decoder_hidden_layers, hidden_dim, dropout, mode, batch_size, epochs, loss, TabularImageDataFactory, TabularImageDataset, cnn_encoder, freeze_encoder, pretrained, crop_pretrained_linear, lr = 1e-3, random_state = 42, grad_clip = False, batchnorm = False, verbose = False, rootpath = "./", name = "MLP_CNN", input_l = 3, input_w = 224, input_c = 224, nan_break = False, **kwargs)
```

### Parameters
| **Parameter**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `encoder_hidden_layers` (`int`)                  | The number of mlp layers before CNN.                                                                                                                                    |
| `decoder_hidden_layers` (`int`)                          | The number of MLP layers after CNN.                                                                                                                                              |
| `hidden_dim` (`int`)                              | The number of neurons in the hidden layers.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                        |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `batch_size` (`int`)                              | The batch size.                                                                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `cnn_encoder` (`str`)                              | name of CNN encoder from pytorch/vision:v0.10.0; alternatively an image encoder in the form of PyTorch object. (**note**: parameter name misrepresenting as CNN are no longer the only type of image encoder)                                                                                                          |
| `freeze_encoder` (`bool`)                              | if True, do not allow weights of image encoder (cnn_encoder) to update during training                                                                                                        |
| `pretrained` (`bool`)                              | if True, load pretrained weights from pytorch/vision:v0.10.0                                                                                                        |
| `crop_pretrained_linear` (`bool`)                              | if True, crop linear head from loaded image model (cnn_encoder) model                                                                                                        |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `TabularImageDataFactory` (`PyTorch2Sklearn.utils.data.TabularImageDataFactory`) | The tabular and image data factory that transforms data from input format into the correct format for TabularImageDataset.                                                            |
| `TabularImageDataset` (`PyTorch2Sklearn.utils.data.TabularImageDataset`)         | The dataset object that generates batches for stochastic gradient descent.                                                                                             |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"MLP_CNN"`) | The name of the model.                |
| `input_l` (`int` optional, default=`224`) | input image length                           |  
| `input_w` (`int` optional, default=`224`) | input image width                           |    
| `input_c` (`int`optional, default=`3`) | input image channels                           | 
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |    

## PyTorch2Sklearn.Transformer_CNN [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer_CNN.py)

```python
class PyTorch2Sklearn.Transformer_CNN.Transformer_CNN(input_dim, output_dim, num_transformer_layers, num_mlp_layers, hidden_dim, dropout, nhead, mode, batch_size, epochs, loss, TabularImageDataFactory, TabularImageDataset, cnn_encoder, freeze_encoder, pretrained, crop_pretrained_linear, agg_transformer_output, share_embedding_mlp = False, cnn_concat = False, dim_feedforward = None, lr = 1e-3, random_state = 42, grad_clip = False, batchnorm = False, verbose = False, rootpath = "./", name = "Transformer_CNN", input_l = 3, input_w = 224, input_c = 224, nan_break = False **kwargs)
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
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `TabularImageDataFactory` (`PyTorch2Sklearn.utils.data.TabularImageDataFactory`) | The tabular and image data factory that transforms data from input format into the correct format for TabularImageDataset.                                                            |
| `TabularImageDataset` (`PyTorch2Sklearn.utils.data.TabularImageDataset`)         | The dataset object that generates batches for stochastic gradient descent.                                                                                             |
| `cnn_concat` (`bool`, optional, default = `False`)         | How to join cnn embedding to tabular. True: concat cnn to transformer layer output                                                                                             |
| `dim_feedforward` (`int`, optional, default=`None`) | The hidden dimension in the feedforward network.                                                                                                                       |
| `cnn_encoder` (`str`)                              | name of CNN encoder from pytorch/vision:v0.10.0; alternatively an image encoder in the form of PyTorch object. (**note**: parameter name misrepresenting as CNN are no longer the only type of image encoder)                                                                                                          |
| `freeze_encoder` (`bool`)                              | if True, do not allow weights of image encoder (cnn_encoder) to update during training                                                                                                        |
| `pretrained` (`bool`)                              | if True, load pretrained weights from pytorch/vision:v0.10.0                                                                                                        |
| `crop_pretrained_linear` (`bool`)                              | if True, crop linear head from loaded image model (cnn_encoder) model                                                                                                        |
| `share_embedding_mlp` (`bool`, optional, default=`False`)              | Whether to share the embedding layer in the MLP.                                                                                                                       |
| `agg_transformer_output` (`str`)                              | how to process the transformer layer outputs before next module. Must be in 'cls' (just use first embedding cls), 'mean' (average the embeddings) and 'concat' (concat all the output layer embeddings together)                                                          |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"Transformer_CNN"`) | The name of the model.                 |
| `input_l` (`int` optional, default=`224`) | input image length                           |  
| `input_w` (`int` optional, default=`224`) | input image width                           |    
| `input_c` (`int`optional, default=`3`) | input image channels                           | 
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |  

## PyTorch2Sklearn.MLP_CNN_AGNN [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP_CNN_AGNN.py)

```python
class PyTorch2Sklearn.MLP_CNN_AGNN.MLP_CNN_AGNN(input_dim, output_dim, num_encoder_layers, num_graph_layers, num_decoder_layers, graph_nhead, hidden_dim, dropout, mode, epochs, loss, ImageGraphDataFactory, cnn_encoder, freeze_encoder, pretrained, crop_pretrained_linear: bool, graph="J", graph_mode = "pure", lr = 1e-3, random_state = 42, grad_clip = False, batchnorm = False, verbose = False, rootpath = "./", name = "MLP_CNN_AGNN", input_l = 3, input_w = 224, input_c = 224, nan_break: False, **kwargs)
```

### Parameters
| **Parameter**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `num_encoder_layers` (`int`)                          | The number of encoder mlp layers.                                                                                                                                              |
| `num_decoder_layers` (`int`)                          | The number of decoder mlp layers.                                                                                                                                              |
| `num_graph_layers` (`int`)                          | The number of graph layers.                                                                                                                                              |
| `graph_nhead` (`int`)                          | The number of attention heads in graph attention layer.                                                                                                                                              |
| `hidden_dim` (`int`)                              | The number of neurons in the hidden layers.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `ImageGraphDataFactory` (`PyTorch2Sklearn.utils.data.ImageGraphDataFactory`) | The graph data factory that transforms tabular and image data from input format into the correct format for training.                                                            |
| `cnn_encoder` (`str`)                              | name of CNN encoder from pytorch/vision:v0.10.0; alternatively an image encoder in the form of PyTorch object. (**note**: parameter name misrepresenting as CNN are no longer the only type of image encoder)                                                                                                          |
| `freeze_encoder` (`bool`)                              | if True, do not allow weights of image encoder (cnn_encoder) to update during training                                                                                                        |
| `pretrained` (`bool`)                              | if True, load pretrained weights from pytorch/vision:v0.10.0                                                                                                        |
| `crop_pretrained_linear` (`bool`)                              | if True, crop linear head from loaded image model (cnn_encoder) model                                                                                                        |
| `graph` (optional, default = `"J"`) | if `"J"`, then every batch will be inferenced with graph = J (1T 1); if `"U"`, then every batch will be inferencd with uniform graph (1/n 1T 1). Also accepts manually defined graph. |
| `graph_mode` (optional, default=`"pure"`) | if `"pure"`, just take graph embedding; if `"residual"` add encoder output and graph output together; if `"concat"` concat encoder output and graph output |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"MLP_CNN_AGNN"`) | The name of the model.  |
| `input_l` (`int` optional, default=`224`) | input image length                           |  
| `input_w` (`int` optional, default=`224`) | input image width                           |    
| `input_c` (`int`optional, default=`3`) | input image channels                           | 
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |   

## PyTorch2Sklearn.Transformer_CNN_AGNN [[source]](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer_CNN_AGNN.py)

```python
class PyTorch2Sklearn.Transformer_CNN_AGNN.Transformer_CNN_AGNN(input_dim, output_dim, num_transformer_layers, num_graph_layers, num_mlp_layers, hidden_dim, dropout, nhead, mode, epochs, loss, ImageGraphDataFactory, cnn_encoder, freeze_encoder, pretrained, crop_pretrained_linear, agg_transformer_output, graph="J", graph_mode = "pure", share_embedding_mlp = False, cnn_concat = False, dim_feedforward = None, lr = 1e-3, random_state = 42, grad_clip = False, batchnorm = False, verbose = False, rootpath = "./", name = "Transformer_CNN_AGNN", input_l = 3, input_w = 224, input_c = 224, nan_break = False, **kwargs)
```

### Parameters
| **Parameter**                              | **Description**                                                                                                                                                        |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_dim` (`int`)                               | The number of features in the input dataset.                                                                                                                           |
| `output_dim` (`int`)                              | The number of output classes/regression output dimension.                                                                                                              |
| `num_transformer_layers` (`int`)                  | The number of transformer layers.                                                                                                                                      |
| `num_mlp_layers` (`int`)                          | The number of MLP layers.                                                                                                                                              |
| `num_graph_layers` (`int`)                          | The number of graph layers.                                                                                                                                              |
| `hidden_dim` (`int`)                              | The number of neurons in the hidden layers.                                                                                                                            |
| `dropout` (`float`)                               | The dropout rate.                                                                                                                                                      |
| `nhead` (`int`)                                   | The number of heads in the multiheadattention models (for both Transformer and Graph modules).                                                                                                                  |
| `mode` (`str`)                                    | The mode of the model, either 'Regression' or 'Classification'.                                                                                                        |
| `epochs` (`int`)                                  | The number of epochs.                                                                                                                                                  |
| `lr` (`float`, optional, default= `1e-3`)                                    | The learning rate.                                                                                                                                                     |
| `random_state` (`int`, optional, default = `42`)                            | The random state. *(WARNING: complete reproducibility cannot be guaranteed even if set seed)*                                                                          |
| `grad_clip` (`bool`, optional, default=`False`)   | Whether to use gradient clipping (to 2) to restrict gradients on each parameter.                                                                                       |
| `batchnorm` (`bool`, optional, default=`False`)  | Whether to use batch normalization on each batch of data.                                                                                                              |
| `loss` (`nn.LossFunctions`)                       | The loss function.                                                                                                                                                     |
| `ImageGraphDataFactory` (`PyTorch2Sklearn.utils.data.ImageGraphDataFactory`) | The graph data factory that transforms tabular and image data from input format into the correct format for training.                                                            |
| `cnn_encoder` (`str`)                              | name of CNN encoder from pytorch/vision:v0.10.0; alternatively an image encoder in the form of PyTorch object. (**note**: parameter name misrepresenting as CNN are no longer the only type of image encoder)                                                                                                          |
| `freeze_encoder` (`bool`)                              | if True, do not allow weights of image encoder (cnn_encoder) to update during training                                                                                                        |
| `pretrained` (`bool`)                              | if True, load pretrained weights from pytorch/vision:v0.10.0                                                                                                        |
| `crop_pretrained_linear` (`bool`)                              | if True, crop linear head from loaded image model (cnn_encoder) model                                                                                                        |
| `agg_transformer_output` (`str`)                              | how to process the transformer layer outputs before next module. Must be in 'cls' (just use first embedding cls), 'mean' (average the embeddings) and 'concat' (concat all the output layer embeddings together)                                                          |
| `cnn_concat` (`bool`, optional, default = `False`)         | How to join cnn embedding to tabular. True: concat cnn to transformer layer output                                                                                             |
| `graph` (optional, default = `"J"`) | if `"J"`, then every batch will be inferenced with graph = J (1T 1); if `"U"`, then every batch will be inferencd with uniform graph (1/n 1T 1). Also accepts manually defined graph. |
| `graph_mode` (optional, default=`"pure"`) | if `"pure"`, just take graph embedding; if `"residual"` add encoder output and graph output together; if `"concat"` concat encoder output and graph output |
| `share_embedding_mlp` (`bool`, optional, default=`False`)              | Whether to share the embedding layer in the MLP.                                                                                                                       |
| `dim_feedforward` (`int`, optional, default=`None`) | The hidden dimension in the feedforward network.                                                                                                                       |
| `verbose` (`bool`, optional, default=`False`)     | Whether to print the training progress.                                                                                                                                |
| `rootpath` (`str`, optional, default=`./`)        | The root path for saving the model.                                                                                                                                    |
| `name` (`str`, optional, default=`"Transformer"`) | The name of the model.                |
| `input_l` (`int` optional, default=`224`) | input image length                           |  
| `input_w` (`int` optional, default=`224`) | input image width                           |    
| `input_c` (`int`optional, default=`3`) | input image channels                           | 
| `nan_break` (`bool`, default = `False`) | if detect nan loss then end training |  


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
        agg_transformer_output='concat',
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

### *MLP_AGNN Regression Example*
```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.MLP_AGNN import MLP_AGNN
from PyTorch2Sklearn.utils.data import GraphDataFactory
from sklearn.metrics import accuracy_score, r2_score

# Create a regression dataset
X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_reg_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
)
y_reg_series = pd.Series(y_reg, name="target")

# must add idx to denote groups of data
reg_graph_df = pd.concat([X_reg_df, y_reg_series], axis=1)
reg_graph_df["idx"] = [i % 10 for i in range(100)]
X = reg_graph_df.drop(columns=["target"])
y = reg_graph_df[["idx", "target"]]

model = MLP_AGNN(
        hidden_dim=16,
        num_encoder_layers=1,
        num_graph_layers=1,
        num_decoder_layers=1,
        graph_nhead=8,
        dropout=0.1,
        epochs=5,
        lr=1e-3,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss(),
        mode="Regression",
        graph="J",
        graph_mode='pure',
        verbose=1,
        GraphDataFactory=GraphDataFactory,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )

model.fit(X, y)

print(r2_score(y, model.predict(X)))
```

### *Transformer_AGNN Regression Example*
```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.Transformer_AGNN import Transformer_AGNN
from PyTorch2Sklearn.utils.data import GraphDataFactory
from sklearn.metrics import accuracy_score, r2_score

# Create a regression dataset
X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_reg_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
)
y_reg_series = pd.Series(y_reg, name="target")

# must add idx to denote groups of data
reg_graph_df = pd.concat([X_reg_df, y_reg_series], axis=1)
reg_graph_df["idx"] = [i % 10 for i in range(100)]
X = reg_graph_df.drop(columns=["target"])
y = reg_graph_df[["idx", "target"]]

model = Transformer_AGNN(
        hidden_dim=16,
        num_transformer_layers=1,
        num_mlp_layers=1,
        num_graph_layers=1,
        graph_nhead=8,
        dropout=0.1,
        share_embedding_mlp=False,
        nhead=8,
        agg_transformer_output='concat',
        epochs=5,
        lr=1e-3,
        graph="J",
        graph_mode='pure',
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss(),
        mode='Regression',
        verbose=1,
        GraphDataFactory=GraphDataFactory,
        rootpath="./",
        output_dim=2,
        input_dim=5,
    )

model.fit(X, y)

print(r2_score(y, model.predict(X)))
```

### *CNN Regression Example*
```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.CNN import CNN
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
from sklearn.metrics import accuracy_score, r2_score

X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42)
X_img = np.random.randn(100, 3, 224, 224).astype(np.float32)
y = pd.Series(y_reg, name='target')

model = CNN(
        output_dim = 1,
        hidden_dim = 16,
        cnn_encoder: 'vgg16',
        freeze_encoder: False,
        pretrained: True,
        crop_pretrained_linear: True,
        num_mlp_layers: 1,
        dropout: 0.1,
        mode: 'Regression',
        batch_size: 8,
        epochs: 5,
        loss=nn.MSELoss(),
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        lr = 1e-3,
        random_state = 42,
        batchnorm=False,
        grad_clip = False,
        verbose = False,
        rootpath = "./",
        name= "CNN",
        input_l = 3,
        input_w = 224,
        input_c = 224,
        nan_break = False,
    )

model.fit(X_img, y)

print(r2_score(y, model.predict(X)))
```

### *MLP_CNN Regression Example*
```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.MLP_CNN import MLP_CNN
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
from sklearn.metrics import accuracy_score, r2_score

X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42)
X = pd.DataFrame(
    X_reg, columns=[f'feature_{i+1}' for i in range(X_reg.shape[1])])
X_img = np.random.randn(100, 3, 224, 224).astype(np.float32)
y = pd.Series(y_reg, name='target')

model = MLP_CNN(
        input_dim =5,
        output_dim = 1,
        encoder_hidden_layers= 1,
        decoder_hidden_layers = 1,
        hidden_dim = 16,
        dropout = 0.1,
        mode = 'Regression',
        batch_size=8,
        epochs=5,
        loss=nn.MSELoss(),
        TabularImageDataFactory=TabularImageDataFactory,
        TabularImageDataset=TabularImageDataset,
        cnn_encoder = 'vgg16',
        freeze_encoder = False,
        pretrained =True,
        crop_pretrained_linear = True,
        lr = 1e-3,
        random_state = 42,
        grad_clip = False,
        batchnorm = False,
        verbose = False,
        rootpath = "./",
        name: str = "MLP_CNN",
        input_l = 3,
        input_w = 224,
        input_c = 224,
        nan_break = False
    )

model.fit([X, X_img], y)

print(r2_score(y, model.predict(X)))
```

### *Transformer_CNN Regression Example*
```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.Transformer_CNN import Transformer_CNN
from PyTorch2Sklearn.utils.data import TabularImageDataFactory, TabularImageDataset
from sklearn.metrics import accuracy_score, r2_score

X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42)
X = pd.DataFrame(
    X_reg, columns=[f'feature_{i+1}' for i in range(X_reg.shape[1])])
X_img = np.random.randn(100, 3, 224, 224).astype(np.float32)
y = pd.Series(y_reg, name='target')

model = Transformer(
        input_dim = 5,
        output_dim = 1,
        num_transformer_layers = 1,
        num_mlp_layers = 1,
        hidden_dim = 16,
        dropout = 0.1,
        nhead = 8,
        mode = 'Regression',
        batch_size = 8,
        epochs = 10,
        loss = nn.MSELoss(),
        TabularImageDataFactory = TabularImageDataFactory,
        TabularImageDataset = TabularImageDataset,
        cnn_encoder = 'vgg16',
        freeze_encoder = False,
        pretrained = True,
        crop_pretrained_linear = True,
        agg_transformer_output = 'mean',
        share_embedding_mlp = False,
        cnn_concat = False,
        dim_feedforward = None,
        lr = 1e-3,
        random_state = 42,
        grad_clip = False,
        batchnorm = False,
        verbose = False,
        rootpath = "./",
        name = "Transformer_CNN",
        input_l = 3,
        input_w = 224,
        input_c = 224,
        nan_break = False,
    )

model.fit([X, X_img], y)

print(r2_score(y, model.predict(X)))
```

### *MLP_CNN_AGNN Regression Example*
```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.MLP_CNN_AGNN import MLP_CNN_AGNN
from PyTorch2Sklearn.utils.data import ImageGraphDataFactory
from sklearn.metrics import accuracy_score, r2_score

# Create a regression dataset
X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_reg_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
)
y_reg_series = pd.Series(y_reg, name="target")

# must add idx to denote groups of data
reg_graph_df = pd.concat([X_reg_df, y_reg_series], axis=1)
reg_graph_df["idx"] = [i % 10 for i in range(100)]
X = reg_graph_df.drop(columns=["target"])
X_img = [np.random.randn(10, 3, 224, 224).astype(np.float32) for _ in range(10)]
y = reg_graph_df[["idx", "target"]]

model = MLP_CNN_AGNN(
        input_dim = 5,
        output_dim = 1,
        num_encoder_layers = 1,
        num_graph_layers = 1,
        num_decoder_layers = 1,
        graph_nhead = 8,
        hidden_dim = 64,
        dropout = 0.1,
        mode = 'Regression',
        epochs = 5,
        loss = nn.MSELoss(),
        ImageGraphDataFactory = ImageGraphDataFactory,
        cnn_encoder = 'vgg16',
        freeze_encoder = False,
        pretrained = True,
        crop_pretrained_linear = True,
        graph="J",
        graph_mode = "pure",
        lr = 1e-3,
        random_state = 42,
        grad_clip = False,
        batchnorm = False,
        verbose = False,
        rootpath = "./",
        name = "MLP_CNN_AGNN",
        input_l = 3,
        input_w = 224,
        input_c = 224,
        nan_break = False,
    )

model.fit([X, X_img], y)

print(r2_score(y, model.predict(X)))
```

### *Transformer_CNN_AGNN Regression Example*
```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from PyTorch2Sklearn.Transformer_CNN_AGNN import Transformer_CNN_AGNN
from PyTorch2Sklearn.utils.data import ImageGraphDataFactory
from sklearn.metrics import accuracy_score, r2_score

# Create a regression dataset
X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_reg_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
)
X_img = [np.random.randn(10, 3, 224, 224).astype(np.float32) for _ in range(10)]
y_reg_series = pd.Series(y_reg, name="target")

# must add idx to denote groups of data
reg_graph_df = pd.concat([X_reg_df, y_reg_series], axis=1)
reg_graph_df["idx"] = [i % 10 for i in range(100)]
X = reg_graph_df.drop(columns=["target"])
y = reg_graph_df[["idx", "target"]]

model = Transformer_CNN_AGNN(
        input_dim = 5,
        output_dim = 1,
        num_transformer_layers = 1,
        num_graph_layers = 1,
        num_mlp_layers = 1,
        hidden_dim = 64,
        dropout = 0.1,
        nhead = 8,
        mode = 'Regression',
        epochs = 5,
        loss = nn.MSELoss(),
        ImageGraphDataFactory=ImageGraphDataFactory,
        cnn_encoder = 'vgg16',
        freeze_encoder = False,
        pretrained = True,
        crop_pretrained_linear = True,
        agg_transformer_output = 'mean',
        graph = "J",
        graph_mode = "pure",
        share_embedding_mlp = False,
        cnn_concat = False,
        dim_feedforward = None,
        lr = 1e-3,
        random_state = 42,
        grad_clip = False,
        batchnorm = False,
        verbose = False,
        rootpath = "./",
        name = "Transformer_CNN_AGNN",
        input_l = 3,
        input_w = 224,
        input_c = 224,
        nan_break = False,
    )

model.fit([X, X_img], y)

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
from PyTorch2Sklearn.Transformer import Transformer
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
        agg_transformer_output = 'mean',
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