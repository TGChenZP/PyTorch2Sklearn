"""
Local validation script for PyTorch2Sklearn.

Usage:
  python local_test.py              # Full tabular + graph training tests (default)
  python local_test.py --smoke      # Import-only smoke test (fast)
  python local_test.py --with-image # Also run minimal image / hub-based model tests

Environment:
  PYTORCH2SKLEARN_QUICK=1  # same as --smoke
  SKIP_IMAGE_TESTS=1       # skip --with-image block even if requested
"""

import argparse
import os
import sys
from typing import List, Optional

RUN_AGNN_ATTENTION_TESTS = os.environ.get("RUN_AGNN_ATTENTION_TESTS", "").strip() in (
    "1",
    "true",
    "True",
    "yes",
)
RUN_AGNN_GRAPH_LAYER_TESTS = os.environ.get("RUN_AGNN_GRAPH_LAYER_TESTS", "").strip() in (
    "1",
    "true",
    "True",
    "yes",
)


def _env_smoke() -> bool:
    return os.environ.get("PYTORCH2SKLEARN_QUICK", "").strip() in ("1", "true", "True", "yes")


def smoke_test(verbose: bool = True) -> None:
    """Verify core package and tabular/graph modules import."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("=== Smoke test: imports ===")

    import PyTorch2Sklearn  # noqa: F401

    from PyTorch2Sklearn.MLP import MLP  # noqa: F401
    from PyTorch2Sklearn.Transformer import Transformer  # noqa: F401
    from PyTorch2Sklearn.MLP_AGNN import MLP_AGNN  # noqa: F401
    from PyTorch2Sklearn.Transformer_AGNN import Transformer_AGNN  # noqa: F401

    from PyTorch2Sklearn.utils.data import (  # noqa: F401
        GraphDataFactory,
        TabularDataFactory,
        TabularDataset,
    )

    log("=== Smoke test: optional image / multimodal imports ===")

    from PyTorch2Sklearn.CNN import CNN  # noqa: F401
    from PyTorch2Sklearn.MLP_CNN import MLP_CNN  # noqa: F401
    from PyTorch2Sklearn.Transformer_CNN import Transformer_CNN  # noqa: F401
    from PyTorch2Sklearn.MLP_CNN_AGNN import MLP_CNN_AGNN  # noqa: F401
    from PyTorch2Sklearn.Transformer_CNN_AGNN import Transformer_CNN_AGNN  # noqa: F401

    from PyTorch2Sklearn.utils.data import (  # noqa: F401
        ImageGraphDataFactory,
        TabularImageDataFactory,
        TabularImageDataset,
    )

    log("=== Smoke test passed ===")


def test_functions(verbose: bool = True) -> None:
    """Train/eval loops for tabular and graph models (no torch.hub image forward pass)."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    from sklearn.datasets import make_classification, make_regression
    import pandas as pd

    # Single-target regression
    X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42
    )
    X_reg_df = pd.DataFrame(
        X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
    )
    y_reg_series = pd.Series(y_reg, name="target")

    reg_graph_df = pd.concat([X_reg_df, y_reg_series], axis=1)
    reg_graph_df["idx"] = [i % 10 for i in range(100)]
    x_reg_graph_df = reg_graph_df.drop(columns=["target"])
    y_reg_graph_df = reg_graph_df[["idx", "target"]]

    # Multi-target regression (two outputs): column names 0 and 1 from sklearn output
    X_reg2, y_reg2 = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42, n_targets=2
    )
    X_reg2_df = pd.DataFrame(
        X_reg2,
        columns=[f"feature_{i + 1}" for i in range(X_reg2.shape[1])],
    )
    y_reg2_series = pd.DataFrame(y_reg2)
    reg2_graph_df = pd.concat([X_reg2_df, y_reg2_series], axis=1)
    reg2_graph_df["idx"] = [i % 10 for i in range(100)]
    x_reg2_graph_df = reg2_graph_df.drop(columns=[0, 1])
    y_reg2_graph_df = reg2_graph_df[["idx", 0, 1]]

    X_class_2, y_class_2 = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_class_2_df = pd.DataFrame(
        X_class_2,
        columns=[f"feature_{i + 1}" for i in range(X_class_2.shape[1])],
    )
    y_class_2_series = pd.Series(y_class_2, name="target")

    class2_graph_df = pd.concat([X_class_2_df, y_class_2_series], axis=1)
    class2_graph_df["idx"] = [i % 10 for i in range(100)]
    x_class2_graph_df = class2_graph_df.drop(columns=["target"])
    y_class2_graph_df = class2_graph_df[["idx", "target"]]

    X_class_3, y_class_3 = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_class_3_df = pd.DataFrame(
        X_class_3,
        columns=[f"feature_{i + 1}" for i in range(X_class_3.shape[1])],
    )
    y_class_3_series = pd.Series(y_class_3, name="target")

    class3_graph_df = pd.concat([X_class_3_df, y_class_3_series], axis=1)
    class3_graph_df["idx"] = [i % 10 for i in range(100)]
    x_class3_graph_df = class3_graph_df.drop(columns=["target"])
    y_class3_graph_df = class3_graph_df[["idx", "target"]]

    log("=== Functional tests: tabular + graph ===")

    test_mlp(X_reg_df, y_reg_series, mode="Regression", output_dim=1, verbose=verbose)
    test_transformer(
        X_reg_df, y_reg_series, mode="Regression", output_dim=1, verbose=verbose
    )
    test_mlp_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=1,
        attention=8,
        graph_mode="concat",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=2,
        attention=8,
        graph_mode="residual",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=1,
        attention=0,
        graph_mode="pure",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=2,
        attention=0,
        graph_mode="concat",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=1,
        attention=8,
        graph_mode="residual",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=2,
        attention=8,
        graph_mode="pure",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=1,
        attention=0,
        graph_mode="concat",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg_graph_df,
        y_reg_graph_df,
        mode="Regression",
        output_dim=1,
        graph_layer=2,
        attention=0,
        graph_mode="residual",
        verbose=verbose,
    )

    test_mlp(
        X_reg2_df, y_reg2_series, mode="Regression", output_dim=2, verbose=verbose
    )
    test_transformer(
        X_reg2_df, y_reg2_series, mode="Regression", output_dim=2, verbose=verbose
    )
    test_mlp_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=1,
        attention=8,
        graph_mode="pure",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=2,
        attention=8,
        graph_mode="concat",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=1,
        attention=0,
        graph_mode="residual",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=2,
        attention=0,
        graph_mode="concat",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=1,
        attention=8,
        graph_mode="pure",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=2,
        attention=8,
        graph_mode="residual",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=1,
        attention=0,
        graph_mode="concat",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_reg2_graph_df,
        y_reg2_graph_df,
        mode="Regression",
        output_dim=2,
        graph_layer=2,
        attention=0,
        graph_mode="pure",
        verbose=verbose,
    )

    test_mlp(
        X_class_2_df, y_class_2_series, mode="Classification", output_dim=2, verbose=verbose
    )
    test_transformer(
        X_class_2_df,
        y_class_2_series,
        mode="Classification",
        output_dim=2,
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=1,
        attention=8,
        graph_mode="residual",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=2,
        attention=8,
        graph_mode="concat",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=1,
        attention=0,
        graph_mode="pure",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=2,
        attention=0,
        graph_mode="residual",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=1,
        attention=8,
        graph_mode="pure",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=2,
        attention=8,
        graph_mode="concat",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=1,
        attention=0,
        graph_mode="residual",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class2_graph_df,
        y_class2_graph_df,
        mode="Classification",
        output_dim=2,
        graph_layer=2,
        attention=0,
        graph_mode="pure",
        verbose=verbose,
    )

    test_mlp(
        X_class_3_df, y_class_3_series, mode="Classification", output_dim=3, verbose=verbose
    )
    test_transformer(
        X_class_3_df,
        y_class_3_series,
        mode="Classification",
        output_dim=3,
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=1,
        attention=8,
        graph_mode="concat",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=2,
        attention=8,
        graph_mode="residual",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=1,
        attention=0,
        graph_mode="pure",
        verbose=verbose,
    )
    test_mlp_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=2,
        attention=0,
        graph_mode="concat",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=1,
        attention=8,
        graph_mode="residual",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=2,
        attention=8,
        graph_mode="pure",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=1,
        attention=0,
        graph_mode="concat",
        verbose=verbose,
    )
    test_transformer_agnn(
        x_class3_graph_df,
        y_class3_graph_df,
        mode="Classification",
        output_dim=3,
        graph_layer=2,
        attention=0,
        graph_mode="residual",
        verbose=verbose,
    )

    log("FUNCTIONALITY TEST PASSED")


def test_mlp(X, y, mode, output_dim, verbose: bool = True) -> None:
    import torch.nn as nn
    from PyTorch2Sklearn.MLP import MLP
    from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
    from sklearn.metrics import accuracy_score, r2_score

    if verbose:
        print(f"=== MLP {mode} output_dim={output_dim} ===")

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
        loss=nn.MSELoss() if mode == "Regression" else nn.CrossEntropyLoss(),
        mode=mode,
        name="MLP",
        verbose=False,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )
    model.fit(X, y)

    if mode == "Regression":
        print(r2_score(y, model.predict(X)))
    else:
        print(accuracy_score(y, model.predict(X)))
        assert len(model.predict_proba(X)) == len(X)

    if verbose:
        print()


def test_transformer(X, y, mode, output_dim, verbose: bool = True) -> None:
    import torch.nn as nn
    from PyTorch2Sklearn.Transformer import Transformer
    from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
    from sklearn.metrics import accuracy_score, r2_score

    if verbose:
        print(f"=== Transformer {mode} output_dim={output_dim} ===")

    model = Transformer(
        hidden_dim=16,
        num_transformer_layers=1,
        num_mlp_layers=1,
        dropout=0.1,
        batch_size=32,
        share_embedding_mlp=False,
        nhead=2,
        agg_transformer_output="mean",
        epochs=5,
        lr=1e-3,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss() if mode == "Regression" else nn.CrossEntropyLoss(),
        mode=mode,
        name="Transformer",
        verbose=False,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )
    model.fit(X, y)

    if mode == "Regression":
        print(r2_score(y, model.predict(X)))
    else:
        print(accuracy_score(y, model.predict(X)))
        assert len(model.predict_proba(X)) == len(X)

    if verbose:
        print()


def test_mlp_agnn(
    X,
    y,
    mode,
    output_dim,
    graph_layer,
    attention,
    graph_mode,
    verbose: bool = True,
) -> None:
    import torch.nn as nn
    from PyTorch2Sklearn.MLP_AGNN import MLP_AGNN
    from PyTorch2Sklearn.utils.data import GraphDataFactory
    from sklearn.metrics import accuracy_score, r2_score

    if verbose:
        print(
            f"=== MLP_AGNN {mode} dim={output_dim} "
            f"layers={graph_layer} heads={attention} mode={graph_mode} ==="
        )
    if graph_layer > 0 and not RUN_AGNN_GRAPH_LAYER_TESTS:
        if verbose:
            print(
                "Skipping AGNN graph-layer test by default "
                "(set RUN_AGNN_GRAPH_LAYER_TESTS=1 to run)."
            )
            print()
        return
    if attention > 0 and not RUN_AGNN_ATTENTION_TESTS:
        if verbose:
            print(
                "Skipping AGNN attention-head test by default "
                "(set RUN_AGNN_ATTENTION_TESTS=1 to run)."
            )
            print()
        return

    model = MLP_AGNN(
        hidden_dim=16,
        num_encoder_layers=1,
        num_graph_layers=graph_layer,
        num_decoder_layers=1,
        graph_nhead=attention,
        dropout=0.1,
        epochs=5,
        lr=1e-3,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss() if mode == "Regression" else nn.CrossEntropyLoss(),
        mode=mode,
        graph="J" if graph_layer > 0 else "U",
        graph_mode=graph_mode,
        verbose=False,
        GraphDataFactory=GraphDataFactory,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )
    model.fit(X, y)
    target = [column for column in y.columns if column != "idx"]

    if mode == "Regression":
        if len(target) == 1:
            print(r2_score(y[target[0]], model.predict(X)))
        else:
            print(r2_score(y[target], model.predict(X)))
    else:
        if len(target) == 1:
            print(accuracy_score(y[target[0]], model.predict(X)))
        else:
            print(accuracy_score(y[target], model.predict(X)))
        assert len(model.predict_proba(X)) == len(X)

    if verbose:
        print()


def test_transformer_agnn(
    X,
    y,
    mode,
    output_dim,
    graph_layer,
    attention,
    graph_mode,
    verbose: bool = True,
) -> None:
    import torch.nn as nn
    from PyTorch2Sklearn.Transformer_AGNN import Transformer_AGNN
    from PyTorch2Sklearn.utils.data import GraphDataFactory
    from sklearn.metrics import accuracy_score, r2_score

    if verbose:
        print(
            f"=== Transformer_AGNN {mode} dim={output_dim} "
            f"layers={graph_layer} heads={attention} mode={graph_mode} ==="
        )
    if graph_layer > 0 and not RUN_AGNN_GRAPH_LAYER_TESTS:
        if verbose:
            print(
                "Skipping AGNN graph-layer test by default "
                "(set RUN_AGNN_GRAPH_LAYER_TESTS=1 to run)."
            )
            print()
        return
    if attention > 0 and not RUN_AGNN_ATTENTION_TESTS:
        if verbose:
            print(
                "Skipping AGNN attention-head test by default "
                "(set RUN_AGNN_ATTENTION_TESTS=1 to run)."
            )
            print()
        return

    model = Transformer_AGNN(
        hidden_dim=16,
        num_transformer_layers=1,
        num_mlp_layers=1,
        num_graph_layers=graph_layer,
        graph_nhead=attention,
        dropout=0.1,
        share_embedding_mlp=False,
        nhead=8,
        agg_transformer_output="mean",
        epochs=5,
        lr=1e-3,
        graph="J" if graph_layer > 0 else "U",
        graph_mode=graph_mode,
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss() if mode == "Regression" else nn.CrossEntropyLoss(),
        mode=mode,
        verbose=False,
        GraphDataFactory=GraphDataFactory,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )
    model.fit(X, y)
    target = [column for column in y.columns if column != "idx"]

    if mode == "Regression":
        if len(target) == 1:
            print(r2_score(y[target[0]], model.predict(X)))
        else:
            print(r2_score(y[target], model.predict(X)))
    else:
        if len(target) == 1:
            print(accuracy_score(y[target[0]], model.predict(X)))
        else:
            print(accuracy_score(y[target], model.predict(X)))
        assert len(model.predict_proba(X)) == len(X)

    if verbose:
        print()


def test_image_models_minimal(verbose: bool = True) -> None:
    """Short runs using torch.hub resnet18, pretrained=False (still loads hub entry)."""

    import numpy as np
    import pandas as pd
    import torch.nn as nn
    from sklearn.datasets import make_regression
    from sklearn.metrics import r2_score

    from PyTorch2Sklearn.CNN import CNN
    from PyTorch2Sklearn.MLP_CNN import MLP_CNN
    from PyTorch2Sklearn.Transformer_CNN import Transformer_CNN
    from PyTorch2Sklearn.MLP_CNN_AGNN import MLP_CNN_AGNN
    from PyTorch2Sklearn.Transformer_CNN_AGNN import Transformer_CNN_AGNN
    from PyTorch2Sklearn.utils.data import (
        ImageGraphDataFactory,
        TabularDataFactory,
        TabularDataset,
        TabularImageDataFactory,
        TabularImageDataset,
    )

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    n = 16
    rng = np.random.default_rng(0)
    X_img = rng.standard_normal((n, 3, 224, 224)).astype(np.float32)
    y = pd.Series(rng.standard_normal(n), name="target")

    log("=== CNN minimal ===")
    cnn = CNN(
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
        epochs=1,
        loss=nn.MSELoss(),
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        lr=1e-3,
        random_state=42,
        verbose=False,
        rootpath="./",
        name="CNN_smoke",
        input_l=224,
        input_w=224,
        input_c=3,
    )
    cnn.fit(X_img, y)
    r2_score(y, cnn.predict(X_img))

    X_reg, y_reg = make_regression(
        n_samples=n, n_features=5, noise=0.1, random_state=42
    )
    X_tab = pd.DataFrame(
        X_reg, columns=[f"feature_{i + 1}" for i in range(X_reg.shape[1])]
    )
    y_tab = pd.Series(y_reg, name="target")

    log("=== MLP_CNN minimal ===")
    mlp_cnn = MLP_CNN(
        input_dim=5,
        output_dim=1,
        encoder_hidden_layers=1,
        decoder_hidden_layers=1,
        hidden_dim=16,
        dropout=0.1,
        mode="Regression",
        batch_size=8,
        epochs=1,
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
        name="MLP_CNN_smoke",
        input_l=224,
        input_w=224,
        input_c=3,
    )
    mlp_cnn.fit([X_tab, X_img], y_tab)
    r2_score(y_tab, mlp_cnn.predict([X_tab, X_img]))

    log("=== Transformer_CNN minimal ===")
    tcnn = Transformer_CNN(
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
        epochs=1,
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
        name="Transformer_CNN_smoke",
        input_l=224,
        input_w=224,
        input_c=3,
    )
    tcnn.fit([X_tab, X_img], y_tab)
    r2_score(y_tab, tcnn.predict([X_tab, X_img]))

    graph_df = pd.concat([X_tab, y_tab], axis=1)
    graph_df["idx"] = [i % 4 for i in range(n)]
    X_g = graph_df.drop(columns=["target"])
    y_g = graph_df[["idx", "target"]]
    images_by_idx = {
        i: rng.standard_normal((4, 3, 224, 224)).astype(np.float32) for i in range(4)
    }

    log("=== MLP_CNN_AGNN minimal ===")
    mcagnn = MLP_CNN_AGNN(
        input_dim=5,
        output_dim=1,
        num_encoder_layers=1,
        num_graph_layers=1,
        num_decoder_layers=1,
        graph_nhead=4,
        hidden_dim=32,
        dropout=0.1,
        mode="Regression",
        epochs=1,
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
        name="MLP_CNN_AGNN_smoke",
        input_l=224,
        input_w=224,
        input_c=3,
    )
    mcagnn.fit([X_g, images_by_idx], y_g)
    r2_score(y_g["target"], mcagnn.predict([X_g, images_by_idx]))

    log("=== Transformer_CNN_AGNN minimal ===")
    tcagnn = Transformer_CNN_AGNN(
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
        epochs=1,
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
        name="Transformer_CNN_AGNN_smoke",
        input_l=224,
        input_w=224,
        input_c=3,
    )
    tcagnn.fit([X_g, images_by_idx], y_g)
    r2_score(y_g["target"], tcagnn.predict([X_g, images_by_idx]))

    log("=== Image model smoke passed ===")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Import-only smoke test (fast)",
    )
    p.add_argument(
        "--with-image",
        action="store_true",
        help="Run minimal CNN / multimodal tests (uses torch.hub)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less console output during functional tests",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    smoke = args.smoke or _env_smoke()

    try:
        if smoke:
            smoke_test(verbose=verbose)
            print("ALL TESTS PASSED (smoke).")
            return 0

        smoke_test(verbose=verbose)
        test_functions(verbose=verbose)

        if args.with_image and os.environ.get("SKIP_IMAGE_TESTS", "").strip() not in (
            "1",
            "true",
            "True",
            "yes",
        ):
            test_image_models_minimal(verbose=verbose)
        elif args.with_image:
            print("SKIP_IMAGE_TESTS set; skipping --with-image runs.")

        print("ALL TESTS PASSED!")
        return 0
    except Exception:
        print("TESTS FAILED.", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
