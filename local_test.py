def smoke_test():

    try:
        print("===Begin Smoke Test===\n")

        print("===Import Modules===")
        import PyTorch2Sklearn

        print("===Import Successful===")

        print("===Import MLP===")

        from PyTorch2Sklearn.MLP import MLP

        print("===Import MLP Successful===")

        print("===Import Transformer===")

        from PyTorch2Sklearn.Transformer import Transformer

        print("===Import Transformer Successful===")

        print("===Import MLP_AGNN===")

        from PyTorch2Sklearn.MLP_AGNN import MLP_AGNN

        print("===Import MLP_AGNN Successful===")

        print("===Import Transformer_AGNN===")

        from PyTorch2Sklearn.Transformer_AGNN import Transformer_AGNN

        print("===Import Transformer_AGNN Successful===")

        print("===Import datasets===")

        from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
        from PyTorch2Sklearn.utils.data import GraphDataFactory

        print("===Import datasets Successful===")

        print("======\n")
        print("===SMOKETEST PASSED===")

    except Exception as e:
        print("===SMOKETEST FAILED===")
        raise e


def test_functions():

    from sklearn.datasets import make_regression, make_classification
    import pandas as pd

    # Create a regression dataset
    X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42
    )
    X_reg_df = pd.DataFrame(
        X_reg, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
    )
    y_reg_series = pd.Series(y_reg, name="target")

    reg_graph_df = pd.concat([X_reg_df, y_reg_series], axis=1)
    reg_graph_df["idx"] = [i % 10 for i in range(100)]
    x_reg_graph_df = reg_graph_df.drop(columns=["target"])
    y_reg_graph_df = reg_graph_df[["idx", "target"]]

    # Create a regression dataset
    X_reg2, y_reg2 = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42, n_targets=2
    )
    X_reg2_df = pd.DataFrame(
        X_reg2, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
    )
    y_reg2_series = pd.DataFrame(y_reg2)
    reg2_graph_df = pd.concat([X_reg2_df, y_reg2_series], axis=1)
    reg2_graph_df["idx"] = [i % 10 for i in range(100)]
    x_reg2_graph_df = reg2_graph_df.drop(columns=[0, 1])
    y_reg2_graph_df = reg2_graph_df[["idx", 0, 1]]

    # Create a classification dataset with 2 classes
    X_class_2, y_class_2 = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_class_2_df = pd.DataFrame(
        X_class_2, columns=[
            f"feature_{i+1}" for i in range(X_class_2.shape[1])]
    )
    y_class_2_series = pd.Series(y_class_2, name="target")

    class2_graph_df = pd.concat([X_class_2_df, y_class_2_series], axis=1)
    class2_graph_df["idx"] = [i % 10 for i in range(100)]
    x_class2_graph_df = class2_graph_df.drop(columns=["target"])
    y_class2_graph_df = class2_graph_df[["idx", "target"]]

    # Create a classification dataset with 3 classes
    X_class_3, y_class_3 = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_class_3_df = pd.DataFrame(
        X_class_3, columns=[
            f"feature_{i+1}" for i in range(X_class_3.shape[1])]
    )
    y_class_3_series = pd.Series(y_class_3, name="target")

    class3_graph_df = pd.concat([X_class_3_df, y_class_3_series], axis=1)
    class3_graph_df["idx"] = [i % 10 for i in range(100)]
    x_class3_graph_df = class3_graph_df.drop(columns=["target"])
    y_class3_graph_df = class3_graph_df[["idx", "target"]]

    try:
        test_mlp(X_reg_df, y_reg_series, mode="Regression", output_dim=1)
        test_transformer(X_reg_df, y_reg_series,
                         mode="Regression", output_dim=1)
        test_mlp_agnn(x_reg_graph_df, y_reg_graph_df,
                      mode="Regression", output_dim=1, graph_layer=1, attention=8)
        test_mlp_agnn(x_reg_graph_df, y_reg_graph_df,
                      mode="Regression", output_dim=1, graph_layer=2, attention=8)
        test_mlp_agnn(x_reg_graph_df, y_reg_graph_df,
                      mode="Regression", output_dim=1, graph_layer=1, attention=0)
        test_mlp_agnn(x_reg_graph_df, y_reg_graph_df,
                      mode="Regression", output_dim=1, graph_layer=2, attention=0)
        test_transformer_agnn(
            x_reg_graph_df, y_reg_graph_df, mode="Regression", output_dim=1, graph_layer=1, attention=8
        )
        test_transformer_agnn(
            x_reg_graph_df, y_reg_graph_df, mode="Regression", output_dim=1, graph_layer=2, attention=8
        )
        test_transformer_agnn(
            x_reg_graph_df, y_reg_graph_df, mode="Regression", output_dim=1, graph_layer=1, attention=0
        )
        test_transformer_agnn(
            x_reg_graph_df, y_reg_graph_df, mode="Regression", output_dim=1, graph_layer=2, attention=0
        )

        test_mlp(X_reg2_df, y_reg2_series, mode="Regression", output_dim=2)
        test_transformer(X_reg2_df, y_reg2_series,
                         mode="Regression", output_dim=2)
        test_mlp_agnn(x_reg2_graph_df, y_reg2_graph_df,
                      mode="Regression", output_dim=2, graph_layer=1, attention=8)
        test_mlp_agnn(x_reg2_graph_df, y_reg2_graph_df,
                      mode="Regression", output_dim=2, graph_layer=2, attention=8)
        test_mlp_agnn(x_reg2_graph_df, y_reg2_graph_df,
                      mode="Regression", output_dim=2, graph_layer=1, attention=0)
        test_mlp_agnn(x_reg2_graph_df, y_reg2_graph_df,
                      mode="Regression", output_dim=2, graph_layer=2, attention=0)
        test_transformer_agnn(
            x_reg2_graph_df, y_reg2_graph_df, mode="Regression", output_dim=2, graph_layer=1, attention=8
        )
        test_transformer_agnn(
            x_reg2_graph_df, y_reg2_graph_df, mode="Regression", output_dim=2, graph_layer=2, attention=8
        )
        test_transformer_agnn(
            x_reg2_graph_df, y_reg2_graph_df, mode="Regression", output_dim=2, graph_layer=1, attention=0
        )
        test_transformer_agnn(
            x_reg2_graph_df, y_reg2_graph_df, mode="Regression", output_dim=2, graph_layer=2, attention=0
        )

        test_mlp(X_class_2_df, y_class_2_series,
                 mode="Classification", output_dim=2)
        test_transformer(
            X_class_2_df, y_class_2_series, mode="Classification", output_dim=2
        )
        test_mlp_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=1, attention=8
        )
        test_mlp_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=2, attention=8
        )
        test_mlp_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=1, attention=0
        )
        test_mlp_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=2, attention=0
        )
        test_transformer_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=1, attention=8
        )
        test_transformer_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=2, attention=8
        )
        test_transformer_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=1, attention=0
        )
        test_transformer_agnn(
            x_class2_graph_df, y_class2_graph_df, mode="Classification", output_dim=2, graph_layer=2, attention=0
        )

        test_mlp(X_class_3_df, y_class_3_series,
                 mode="Classification", output_dim=3)
        test_transformer(
            X_class_3_df, y_class_3_series, mode="Classification", output_dim=3
        )
        test_mlp_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=1, attention=8
        )
        test_mlp_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=2, attention=8
        )
        test_mlp_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=1, attention=0
        )
        test_mlp_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=2, attention=0
        )
        test_transformer_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=1, attention=8
        )
        test_transformer_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=2, attention=8
        )
        test_transformer_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=1, attention=0
        )
        test_transformer_agnn(
            x_class3_graph_df, y_class3_graph_df, mode="Classification", output_dim=3, graph_layer=2, attention=0
        )

        print("""FUNCTIONALITY TEST PASSED""")
    except Exception as e:
        print("""FUNCTIONALITY TEST FAILED""")
        raise e


def test_mlp(X, y, mode, output_dim):

    import torch.nn as nn
    from PyTorch2Sklearn.MLP import MLP
    from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
    from sklearn.metrics import accuracy_score, r2_score

    print(f"===Begin MLP {mode} {output_dim} class/dim test===")

    print("===Initialise MLP===")
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
        verbose=1,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )
    print("===Initialise MLP complete===")

    print("===Begin fitting MLP===")
    model.fit(X, y)
    print("===Fitting MLP complete===")

    print("===Begin MLP evaluation===")
    if mode == "Regression":
        print(r2_score(y, model.predict(X)))
    else:
        print(accuracy_score(y, model.predict(X)))
        assert len(model.predict_proba(X)) == len(X)
    print(f"===Evaluate MLP {mode} {output_dim} class/dim test Complete===")
    print()


def test_transformer(X, y, mode, output_dim):

    import torch.nn as nn
    from PyTorch2Sklearn.Transformer import Transformer
    from PyTorch2Sklearn.utils.data import TabularDataFactory, TabularDataset
    from sklearn.metrics import accuracy_score, r2_score

    print(f"===Begin Transformer {mode} {output_dim} class/dim test===")

    print("===Initialise Transformer===")

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
        loss=nn.MSELoss() if mode == "Regression" else nn.CrossEntropyLoss(),
        mode=mode,
        name="Transformer",
        verbose=1,
        TabularDataFactory=TabularDataFactory,
        TabularDataset=TabularDataset,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )

    print("===Initialise Transformer complete===")

    print("===Begin fitting Transformer===")
    model.fit(X, y)
    print("===Fitting Transformer complete===")

    print("===Begin Transformer evaluation===")
    if mode == "Regression":
        print(r2_score(y, model.predict(X)))
    else:
        print(accuracy_score(y, model.predict(X)))
        assert len(model.predict_proba(X)) == len(X)
    print(
        f"===Evaluate Transformer {mode} {output_dim} class/dim test Complete===")
    print()


def test_mlp_agnn(X, y, mode, output_dim, graph_layer, attention):
    import torch.nn as nn
    from PyTorch2Sklearn.MLP_AGNN import MLP_AGNN
    from PyTorch2Sklearn.utils.data import GraphDataFactory
    from sklearn.metrics import accuracy_score, r2_score

    print(
        f"===Begin MLP_AGNN {mode} {output_dim} class/dim {graph_layer} graph layers {attention} graph attention heads test===")

    print("===Initialise MLP_AGNN===")
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
        graph="J",
        verbose=1,
        DataFactory=GraphDataFactory,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )
    print("===Initialise MLP_AGNN complete===")

    print("===Begin fitting MLP_AGNN===")

    model.fit(X, y)
    print("===Fitting MLP_AGNN complete===")
    target = [column for column in y.columns if "idx" != column]

    print("===Begin MLP_AGNN evaluation===")
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
    print(
        f"===Evaluate MLP_AGNN {mode} {output_dim} class/dim test {graph_layer} graph layers {attention} graph attention heads Complete===")
    print()


def test_transformer_agnn(X, y, mode, output_dim, graph_layer, attention):

    import torch.nn as nn
    from PyTorch2Sklearn.Transformer_AGNN import Transformer_AGNN
    from PyTorch2Sklearn.utils.data import GraphDataFactory
    from sklearn.metrics import accuracy_score, r2_score

    print(
        f"===Begin Transformer {mode} {output_dim} class/dim {graph_layer} graph layers {attention} graph attention heads test===")

    print("===Initialise Transformer===")

    model = Transformer_AGNN(
        hidden_dim=16,
        num_transformer_layers=1,
        num_mlp_layers=1,
        num_graph_layers=graph_layer,
        graph_nhead=attention,
        dropout=0.1,
        share_embedding_mlp=False,
        nhead=8,
        use_cls=False,
        epochs=5,
        lr=1e-3,
        graph="J",
        batchnorm=False,
        grad_clip=False,
        random_state=42,
        loss=nn.MSELoss() if mode == "Regression" else nn.CrossEntropyLoss(),
        mode=mode,
        verbose=1,
        DataFactory=GraphDataFactory,
        rootpath="./",
        output_dim=output_dim,
        input_dim=5,
    )

    print("===Initialise Transformer_AGNN complete===")
    print("===Begin fitting Transformer_AGNN===")
    model.fit(X, y)
    print("===Fitting Transformer_AGNN complete===")
    target = [column for column in y.columns if "idx" != column]

    print("===Begin Transformer_AGNN evaluation===")
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
    print(
        f"===Evaluate Transformer_AGNN {mode} {output_dim} class/dim test {graph_layer} graph layers {attention} graph attention heads Complete==="
    )
    print()


def local_test():
    smoke_test()
    test_functions()


if __name__ == "__main__":

    local_test()

    print("ALL TESTS PASSED!")
