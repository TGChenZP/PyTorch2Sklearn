# PyTorch2Sklearn

Wraps PyTorch architectures (MLP, Transformer, CNN, and graph-attention variants) in a **scikit-learn-style** API: `fit`, `predict`, optional `predict_proba` (classification), and `save` / `load`.

**Repository:** [https://github.com/TGChenZP/PyTorch2Sklearn](https://github.com/TGChenZP/PyTorch2Sklearn)  
**Author:** [https://github.com/TGChenZP](https://github.com/TGChenZP)

If you use this package in research, please cite it appropriately (see [Citation](#citation)).

---

## Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Requirements](#requirements)
- [Concepts](#concepts)
- [Quickstart](#quickstart)
- [Scikit-learn-like API](#scikit-learn-like-api)
- [Models](#models)
  - [MLP](#mlp)
  - [Transformer](#transformer)
  - [MLP_AGNN](#mlp_agnn)
  - [Transformer_AGNN](#transformer_agnn)
  - [CNN](#cnn)
  - [MLP_CNN](#mlp_cnn)
  - [Transformer_CNN](#transformer_cnn)
  - [MLP_CNN_AGNN](#mlp_cnn_agnn)
  - [Transformer_CNN_AGNN](#transformer_cnn_agnn)
- [Usage examples](#usage-examples)
- [Citation](#citation)

---

## Introduction

PyTorch2Sklearn targets **tabular** supervised learning (classification and regression) with optional **image** and **graph** components. You choose the task with `mode` (`"Classification"` or `"Regression"`), pass a [`torch.nn`](https://pytorch.org/docs/stable/nn.html) loss (for example `nn.MSELoss()` or `nn.CrossEntropyLoss()`), and set `input_dim` / `output_dim` to match your data (feature count; regression dimensionality or number of classes).

**Note:** Setting `random_state` improves repeatability but does not guarantee full bitwise reproducibility across devices and PyTorch versions.

Package version **0.3.16** (see `setup.py`). **Python** \(>= 3.6\) as declared in the package metadata.

---

## Installation

```bash
pip install PyTorch2Sklearn
```

Install from a clone for development:

```bash
pip install -e .
```

---

## Requirements

Core stack used by the library:

- **PyTorch** (`torch`, `torch.nn`)
- **NumPy**, **pandas**
- **scikit-learn** (e.g. preprocessing, metrics)
- **tqdm** (progress when `verbose=True`)

Image models load encoders via **`torch.hub`** from `pytorch/vision:v0.10.0` when `cnn_encoder` is a **string** (for example `'resnet18'`). The first run may download weights when `pretrained=True`.

---

## Concepts

| Item | Role |
|------|------|
| **`TabularDataFactory` / `TabularDataset`** | Tabular-only models (`MLP`, `Transformer`, `CNN` with image-only inputs still use these names in the API). |
| **`TabularImageDataFactory` / `TabularImageDataset`** | Joint tabular + image rows: `fit([X_tabular, X_images], y)`. |
| **`GraphDataFactory`** | Tabular rows grouped by an **`idx`** column (graph batches per group). |
| **`ImageGraphDataFactory`** | Tabular + images keyed by **`idx`**: `fit([X_tabular, images_by_idx], y)`. |

Graph-related models require **`idx`** in `X` (and in `y` for supervised graph training) so the factory can batch by group.

---

## Quickstart

```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from sklearn.metrics import r2_score

from PyTorch2Sklearn.MLP import MLP
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset

X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X = pd.DataFrame(X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])])
y = pd.Series(y_reg, name="target")

model = MLP(
    input_dim=5,
    output_dim=1,
    hidden_layers=1,
    hidden_dim=16,
    dropout=0.1,
    mode="Regression",
    batch_size=32,
    epochs=5,
    loss=nn.MSELoss(),
    TabularDataFactory=TabularDataFactory,
    TabularDataset=TabularDataset,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="MLP",
)

model.fit(X, y)
print(r2_score(y, model.predict(X)))
```

---

## Scikit-learn-like API

Implemented in [`PyTorch2Sklearn/__template__.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/__template__.py) (behaviour varies slightly by base class: tabular, graph, tabular+image, image+graph).

| Method | Description |
|--------|-------------|
| `__init__(...)` | Builds `CFG` and the underlying `torch.nn.Module`. |
| `fit(train_x, train_y)` | Training loop; input types depend on the model (DataFrame/Series, or list/tuple for multimodal). |
| `predict(val_x)` | Predictions; returns a list (decode classification labels where applicable). |
| `predict_proba(val_x)` | **Classification only:** class probabilities (numpy array). |
| `save(mark="")` | Save `state_dict` under `rootpath/state/`. |
| `load(mark="")` | Load weights from the same path convention. |

---

## Models

Sources link to the GitHub `main` branch. Constructor lists mirror the current code.

### Common keywords (many models)

| Parameter | Typical meaning |
|-----------|-----------------|
| `lr` | Learning rate (default `1e-3`). |
| `random_state` | Seed for reproducibility (default `42`). |
| `grad_clip` | If `True`, clip global gradient norm to **2.0**. |
| `batchnorm` | Batch normalization in relevant blocks. |
| `verbose` | `True` enables `tqdm` / progress-style logging where implemented. |
| `rootpath` | Root directory for saved checkpoints. |
| `nan_break` | Stop training when NaN loss is detected (where implemented). |

---

### MLP

**Source:** [`PyTorch2Sklearn/MLP.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP.py)

```python
MLP(
    input_dim: int,
    output_dim: int,
    hidden_layers: int,
    hidden_dim: int,
    dropout: float,
    mode: str,
    batch_size: int,
    epochs: int,
    loss,
    TabularDataFactory,
    TabularDataset,
    lr: float = 1e-3,
    random_state: int = 42,
    grad_clip: bool = False,
    batchnorm: bool = False,
    verbose: bool = False,
    rootpath: str = "./",
    name: str = "MLP",
    nan_break: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| `hidden_layers` | Number of hidden layers. If `0`, the implementation uses a width schedule between `input_dim` and `output_dim`. |
| `hidden_dim` | Hidden width (when non-zero). |
| `dropout` | Dropout rate. |
| `batch_size` | Minibatch size. |
| `epochs` | Training epochs. |
| `loss` | A `torch.nn` loss module instance. |

---

### Transformer

**Source:** [`PyTorch2Sklearn/Transformer.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer.py)

```python
Transformer(
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
    TabularDataFactory,
    TabularDataset,
    agg_transformer_output: str,
    share_embedding_mlp: bool = False,
    dim_feedforward: int = None,
    lr: float = 1e-3,
    random_state: int = 42,
    grad_clip: bool = False,
    batchnorm: bool = False,
    verbose: bool = False,
    rootpath: str = "./",
    name: str = "Transformer",
    nan_break: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| `agg_transformer_output` | One of `'cls'`, `'mean'`, `'concat'` — how token outputs are aggregated before the head. |
| `dim_feedforward` | FFN inner size; default **`4 * hidden_dim`** when `None`. |
| `nhead` | Transformer multi-head attention heads (must divide `hidden_dim` where required by PyTorch). |

---

### MLP_AGNN

**Source:** [`PyTorch2Sklearn/MLP_AGNN.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP_AGNN.py)

Graph batches use **`GraphDataFactory`**. Include column **`idx`** in `X` (and the matching structure in `y`).

```python
MLP_AGNN(
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
    GraphDataFactory,
    graph="J",
    graph_mode: str = "pure",
    lr: float = 1e-3,
    random_state: int = 42,
    grad_clip: bool = False,
    batchnorm: bool = False,
    verbose: bool = False,
    rootpath: str = "./",
    name: str = "MLP_AGNN",
    nan_break: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| `graph` | `"J"` (all-ones adjacency-style use), `"U"` (uniform), or a custom graph object depending on your pipeline. |
| `graph_mode` | `"pure"`, `"residual"`, or `"concat"` — how encoder and graph outputs combine. |
| `graph_nhead` | Attention heads in graph attention layers. |

**Note:** There is **no** `batch_size` in this constructor; sampling is organised via graph batches from the factory.

---

### Transformer_AGNN

**Source:** [`PyTorch2Sklearn/Transformer_AGNN.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer_AGNN.py)

```python
Transformer_AGNN(
    input_dim: int,
    output_dim: int,
    num_transformer_layers: int,
    num_graph_layers: int,
    num_mlp_layers: int,
    hidden_dim: int,
    dropout: float,
    nhead: int,
    graph_nhead: int,
    mode: str,
    epochs: int,
    loss,
    GraphDataFactory,
    agg_transformer_output: str,
    graph="J",
    graph_mode: str = "pure",
    share_embedding_mlp: bool = False,
    dim_feedforward: int = None,
    lr: float = 1e-3,
    random_state: int = 42,
    grad_clip: bool = False,
    batchnorm: bool = False,
    verbose: bool = False,
    rootpath: str = "./",
    name: str = "Transformer_AGNN",
    nan_break: bool = False,
)
```

---

### CNN

**Source:** [`PyTorch2Sklearn/CNN.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/CNN.py)

Image-only head on top of a **`torchvision`** encoder loaded by name (string) or a passed **`torch.nn.Module`**.

```python
CNN(
    output_dim: int,
    hidden_dim: int,
    cnn_encoder,
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
)
```

The model builds a sample tensor of shape **(1, `input_c`, `input_l`, `input_w`)** to infer the encoder output size. For standard RGB \(224 \times 224\), pass **`input_l=224`**, **`input_w=224`**, **`input_c=3`** explicitly (the library defaults are historical and may not match “CHW” intuition).

| Parameter | Description |
|-----------|-------------|
| `cnn_encoder` | String name for `torch.hub.load('pytorch/vision:v0.10.0', ...)` or a PyTorch module. |
| `crop_pretrained_linear` | If `True`, drop the classification head of classification models and keep convolutional trunk. |
| `freeze_encoder` | If `True`, encoder weights are frozen (requires `pretrained=True` in asserts inside the model). |

`fit`/`predict` use **`TabularDataFactory`** with **`X`** as a NumPy array of shape **`(N, C, H, W)`** and **`y`** as a pandas **`Series`** (see examples).

---

### MLP_CNN

**Source:** [`PyTorch2Sklearn/MLP_CNN.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP_CNN.py)

```python
MLP_CNN(
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
    TabularImageDataFactory,
    TabularImageDataset,
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
)
```

Pass **`fit([X_tabular, X_images], y)`** where **`X_images`** is typically an **`(N, C, H, W)`** array aligned row-wise with **`X_tabular`**.

---

### Transformer_CNN

**Source:** [`PyTorch2Sklearn/Transformer_CNN.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer_CNN.py)

```python
Transformer_CNN(
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
    name: str = "Transformer_CNN",
    input_l: int = 3,
    input_w: int = 224,
    input_c: int = 224,
    nan_break: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| `cnn_concat` | If `True`, concatenates CNN embeddings with transformer token outputs before the next stage (see implementation). |

---

### MLP_CNN_AGNN

**Source:** [`PyTorch2Sklearn/MLP_CNN_AGNN.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/MLP_CNN_AGNN.py)

```python
MLP_CNN_AGNN(
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
)
```

Use **`fit([X_tabular, images_by_idx], y)`** where **`X_tabular`** has **`idx`**, and **`images_by_idx`** maps each **`idx`** to the image tensor/array for that group (see factory implementation in `utils/data.py`).

---

### Transformer_CNN_AGNN

**Source:** [`PyTorch2Sklearn/Transformer_CNN_AGNN.py`](https://github.com/TGChenZP/PyTorch2Sklearn/blob/main/PyTorch2Sklearn/Transformer_CNN_AGNN.py)

```python
Transformer_CNN_AGNN(
    input_dim: int,
    output_dim: int,
    num_transformer_layers: int,
    num_graph_layers: int,
    num_mlp_layers: int,
    hidden_dim: int,
    dropout: float,
    nhead: int,
    mode: str,
    epochs: int,
    loss,
    ImageGraphDataFactory,
    cnn_encoder: str,
    freeze_encoder: bool,
    pretrained: bool,
    crop_pretrained_linear: bool,
    agg_transformer_output: str,
    graph="J",
    graph_mode: str = "pure",
    share_embedding_mlp: bool = False,
    cnn_concat: bool = False,
    dim_feedforward: int = None,
    lr: float = 1e-3,
    random_state: int = 42,
    grad_clip: bool = False,
    batchnorm: bool = False,
    verbose: bool = False,
    rootpath: str = "./",
    name: str = "Transformer_CNN_AGNN",
    input_l: int = 3,
    input_w: int = 224,
    input_c: int = 224,
    nan_break: bool = False,
)
```

Default `name` in code is **`"Transformer_CNN_AGNN"`**.

---

## Usage examples

### Tabular regression — MLP

```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from sklearn.metrics import r2_score

from PyTorch2Sklearn.MLP import MLP
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset

X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
y = pd.Series(y_reg, name="target")

model = MLP(
    input_dim=5,
    output_dim=1,
    hidden_dim=16,
    hidden_layers=1,
    dropout=0.1,
    mode="Regression",
    batch_size=32,
    epochs=5,
    loss=nn.MSELoss(),
    TabularDataFactory=TabularDataFactory,
    TabularDataset=TabularDataset,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="MLP",
)

model.fit(X, y)
print(r2_score(y, model.predict(X)))
```

### Tabular regression — Transformer

```python
from sklearn.datasets import make_regression
import pandas as pd
import torch.nn as nn
from sklearn.metrics import r2_score

from PyTorch2Sklearn.Transformer import Transformer
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset

X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
y = pd.Series(y_reg, name="target")

model = Transformer(
    input_dim=5,
    output_dim=1,
    hidden_dim=16,
    num_transformer_layers=1,
    num_mlp_layers=1,
    dropout=0.1,
    nhead=2,
    agg_transformer_output="mean",
    mode="Regression",
    batch_size=32,
    epochs=5,
    loss=nn.MSELoss(),
    TabularDataFactory=TabularDataFactory,
    TabularDataset=TabularDataset,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="Transformer",
)

model.fit(X, y)
print(r2_score(y, model.predict(X)))
```

### Graph regression — MLP_AGNN

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from PyTorch2Sklearn.MLP_AGNN import MLP_AGNN
from PyTorch2Sklearn.utils.data import GraphDataFactory

X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
y_series = pd.Series(y_reg, name="target")

graph_df = pd.concat([X_df, y_series], axis=1)
graph_df["idx"] = [i % 10 for i in range(100)]
X = graph_df.drop(columns=["target"])
y = graph_df[["idx", "target"]]

model = MLP_AGNN(
    input_dim=5,
    output_dim=1,
    hidden_dim=16,
    num_encoder_layers=1,
    num_graph_layers=1,
    num_decoder_layers=1,
    graph_nhead=8,
    dropout=0.1,
    mode="Regression",
    epochs=5,
    loss=nn.MSELoss(),
    GraphDataFactory=GraphDataFactory,
    graph="J",
    graph_mode="pure",
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="MLP_AGNN",
)

model.fit(X, y)
print(r2_score(y["target"], model.predict(X)))
```

### Graph regression — Transformer_AGNN

```python
import pandas as pd
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from PyTorch2Sklearn.Transformer_AGNN import Transformer_AGNN
from PyTorch2Sklearn.utils.data import GraphDataFactory

X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
y_series = pd.Series(y_reg, name="target")

graph_df = pd.concat([X_df, y_series], axis=1)
graph_df["idx"] = [i % 10 for i in range(100)]
X = graph_df.drop(columns=["target"])
y = graph_df[["idx", "target"]]

model = Transformer_AGNN(
    input_dim=5,
    output_dim=1,
    hidden_dim=16,
    num_transformer_layers=1,
    num_mlp_layers=1,
    num_graph_layers=1,
    graph_nhead=8,
    dropout=0.1,
    nhead=2,
    agg_transformer_output="mean",
    mode="Regression",
    epochs=5,
    loss=nn.MSELoss(),
    GraphDataFactory=GraphDataFactory,
    graph="J",
    graph_mode="pure",
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="Transformer_AGNN",
)

model.fit(X, y)
print(r2_score(y["target"], model.predict(X)))
```

### Image-only regression — CNN

Uses **`torch.hub`** (may download the model definition / weights). **`pretrained=False`** avoids pretrained weight download.

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import r2_score

from PyTorch2Sklearn.CNN import CNN
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset

n = 32
rng = np.random.default_rng(42)
X_img = rng.standard_normal((n, 3, 224, 224)).astype(np.float32)
y = pd.Series(rng.standard_normal(n), name="target")

model = CNN(
    output_dim=1,
    hidden_dim=16,
    cnn_encoder="resnet18",
    freeze_encoder=False,
    pretrained=False,
    crop_pretrained_linear=True,
    num_mlp_layers=1,
    dropout=0.1,
    mode="Regression",
    batch_size=8,
    epochs=2,
    loss=nn.MSELoss(),
    TabularDataFactory=TabularDataFactory,
    TabularDataset=TabularDataset,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="CNN",
    input_l=224,
    input_w=224,
    input_c=3,
)

model.fit(X_img, y)
print(r2_score(y, model.predict(X_img)))
```

### Tabular + image regression — MLP_CNN

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from PyTorch2Sklearn.MLP_CNN import MLP_CNN
from PyTorch2Sklearn.utils.data import TabularImageDataFactory, TabularImageDataset

X_reg, y_reg = make_regression(
    n_samples=32, n_features=5, noise=0.1, random_state=42
)
X_tab = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
rng = np.random.default_rng(42)
X_im = rng.standard_normal((32, 3, 224, 224)).astype(np.float32)
y = pd.Series(y_reg, name="target")

model = MLP_CNN(
    input_dim=5,
    output_dim=1,
    encoder_hidden_layers=1,
    decoder_hidden_layers=1,
    hidden_dim=16,
    dropout=0.1,
    mode="Regression",
    batch_size=8,
    epochs=2,
    loss=nn.MSELoss(),
    TabularImageDataFactory=TabularImageDataFactory,
    TabularImageDataset=TabularImageDataset,
    cnn_encoder="resnet18",
    freeze_encoder=False,
    pretrained=False,
    crop_pretrained_linear=True,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="MLP_CNN",
    input_l=224,
    input_w=224,
    input_c=3,
)

model.fit([X_tab, X_im], y)
print(r2_score(y, model.predict([X_tab, X_im])))
```

### Tabular + image regression — Transformer_CNN

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from PyTorch2Sklearn.Transformer_CNN import Transformer_CNN
from PyTorch2Sklearn.utils.data import TabularImageDataFactory, TabularImageDataset

X_reg, y_reg = make_regression(
    n_samples=32, n_features=5, noise=0.1, random_state=42
)
X_tab = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
rng = np.random.default_rng(42)
X_im = rng.standard_normal((32, 3, 224, 224)).astype(np.float32)
y = pd.Series(y_reg, name="target")

model = Transformer_CNN(
    input_dim=5,
    output_dim=1,
    num_transformer_layers=1,
    num_mlp_layers=1,
    hidden_dim=16,
    dropout=0.1,
    nhead=2,
    agg_transformer_output="mean",
    mode="Regression",
    batch_size=8,
    epochs=2,
    loss=nn.MSELoss(),
    TabularImageDataFactory=TabularImageDataFactory,
    TabularImageDataset=TabularImageDataset,
    cnn_encoder="resnet18",
    freeze_encoder=False,
    pretrained=False,
    crop_pretrained_linear=True,
    share_embedding_mlp=False,
    cnn_concat=False,
    dim_feedforward=None,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="Transformer_CNN",
    input_l=224,
    input_w=224,
    input_c=3,
)

model.fit([X_tab, X_im], y)
print(r2_score(y, model.predict([X_tab, X_im])))
```

### Graph + image regression — MLP_CNN_AGNN

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from PyTorch2Sklearn.MLP_CNN_AGNN import MLP_CNN_AGNN
from PyTorch2Sklearn.utils.data import ImageGraphDataFactory

X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
y_series = pd.Series(y_reg, name="target")

graph_df = pd.concat([X_df, y_series], axis=1)
graph_df["idx"] = [i % 10 for i in range(100)]
X_tab = graph_df.drop(columns=["target"])
y = graph_df[["idx", "target"]]

images_by_idx = {
    i: np.random.randn(10, 3, 224, 224).astype(np.float32) for i in range(10)
}

model = MLP_CNN_AGNN(
    input_dim=5,
    output_dim=1,
    num_encoder_layers=1,
    num_graph_layers=1,
    num_decoder_layers=1,
    graph_nhead=8,
    hidden_dim=32,
    dropout=0.1,
    mode="Regression",
    epochs=2,
    loss=nn.MSELoss(),
    ImageGraphDataFactory=ImageGraphDataFactory,
    cnn_encoder="resnet18",
    freeze_encoder=False,
    pretrained=False,
    crop_pretrained_linear=True,
    graph="J",
    graph_mode="pure",
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="MLP_CNN_AGNN",
    input_l=224,
    input_w=224,
    input_c=3,
)

model.fit([X_tab, images_by_idx], y)
print(r2_score(y["target"], model.predict([X_tab, images_by_idx])))
```

### Graph + image regression — Transformer_CNN_AGNN

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from PyTorch2Sklearn.Transformer_CNN_AGNN import Transformer_CNN_AGNN
from PyTorch2Sklearn.utils.data import ImageGraphDataFactory

X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
)
y_series = pd.Series(y_reg, name="target")

graph_df = pd.concat([X_df, y_series], axis=1)
graph_df["idx"] = [i % 10 for i in range(100)]
X_tab = graph_df.drop(columns=["target"])
y = graph_df[["idx", "target"]]

images_by_idx = {
    i: np.random.randn(10, 3, 224, 224).astype(np.float32) for i in range(10)
}

model = Transformer_CNN_AGNN(
    input_dim=5,
    output_dim=1,
    num_transformer_layers=1,
    num_graph_layers=1,
    num_mlp_layers=1,
    hidden_dim=32,
    dropout=0.1,
    nhead=2,
    agg_transformer_output="mean",
    mode="Regression",
    epochs=2,
    loss=nn.MSELoss(),
    ImageGraphDataFactory=ImageGraphDataFactory,
    cnn_encoder="resnet18",
    freeze_encoder=False,
    pretrained=False,
    crop_pretrained_linear=True,
    graph="J",
    graph_mode="pure",
    share_embedding_mlp=False,
    cnn_concat=False,
    dim_feedforward=None,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="Transformer_CNN_AGNN",
    input_l=224,
    input_w=224,
    input_c=3,
)

model.fit([X_tab, images_by_idx], y)
print(r2_score(y["target"], model.predict([X_tab, images_by_idx])))
```

### Classification — MLP

```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score

from PyTorch2Sklearn.MLP import MLP
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset

X_c, y_c = make_classification(
    n_samples=100, n_features=5, n_classes=2, n_clusters_per_class=1, random_state=42
)
X = pd.DataFrame(
    X_c, columns=[f"feature_{i + 1}" for i in range(X_c.shape[1])]
)
y = pd.Series(y_c, name="target")

model = MLP(
    input_dim=5,
    output_dim=2,
    hidden_dim=16,
    hidden_layers=1,
    dropout=0.1,
    mode="Classification",
    batch_size=32,
    epochs=5,
    loss=nn.CrossEntropyLoss(),
    TabularDataFactory=TabularDataFactory,
    TabularDataset=TabularDataset,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="MLP",
)

model.fit(X, y)
print(accuracy_score(y, model.predict(X)))
```

### Classification — Transformer

```python
from sklearn.datasets import make_classification
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score

from PyTorch2Sklearn.Transformer import Transformer
from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset

X_c, y_c = make_classification(
    n_samples=100, n_features=5, n_classes=2, n_clusters_per_class=1, random_state=42
)
X = pd.DataFrame(
    X_c, columns=[f"feature_{i + 1}" for i in range(X_c.shape[1])]
)
y = pd.Series(y_c, name="target")

model = Transformer(
    input_dim=5,
    output_dim=2,
    hidden_dim=16,
    num_transformer_layers=1,
    num_mlp_layers=1,
    dropout=0.1,
    nhead=2,
    agg_transformer_output="mean",
    mode="Classification",
    batch_size=32,
    epochs=5,
    loss=nn.CrossEntropyLoss(),
    TabularDataFactory=TabularDataFactory,
    TabularDataset=TabularDataset,
    lr=1e-3,
    random_state=42,
    verbose=False,
    rootpath="./",
    name="Transformer",
)

model.fit(X, y)
print(accuracy_score(y, model.predict(X)))
```

---

## Citation

If this software supports your publication or project, please cite the repository and the version you used, for example:

```bibtex
@software{pytorch2sklearn,
  author = {Lang Chen},
  title  = {PyTorch2Sklearn},
  url    = {https://github.com/TGChenZP/PyTorch2Sklearn},
  year   = {2026},
  note   = {Version 0.3.16}
}
```