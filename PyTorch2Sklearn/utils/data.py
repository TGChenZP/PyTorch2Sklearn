from PyTorch2Sklearn.environment import *


class TabularDataset(Dataset):
    """Creates list of instances for tabular data from pandas dataframe"""

    def __init__(self, x_list, y_list=None):
        """

        Input:
            - x_list: list of features
            - y_list: list of targets

        """

        self.features = torch.tensor(np.array(x_list), dtype=torch.float32)
        self.targets = (
            torch.tensor(np.array(y_list), dtype=torch.float32)
            if y_list is not None
            else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


def TabularDataFactory(X, y=None, mode="Classification"):
    """
    Creates list of instances for tabular data from pandas dataframe

    Input:
        - X: pandas dataframe containing features
        - y: pandas series containing targets
        - mode: 'Classification' or 'Regression'
    Output:
        - X_list: list of instances
        - y_list: list of targets
    """

    if y is None:
        if type(X) != np.ndarray:
            X = X.values
        return X.tolist()
    else:
        if mode == "Classification":
            # need to change to multi-class probability vector for classification
            encoder = OneHotEncoder(
                sparse_output=False, categories="auto", handle_unknown="ignore"
            )
            y = encoder.fit_transform(np.array(y).reshape(-1, 1))

        if type(X) != np.ndarray:
            X = X.values
        if type(y) != np.ndarray:
            y = y.values

        return X.tolist(), y.tolist()


def GraphDataFactory(X, y=None, mode="Classification", CFG=None):
    """
    Creates list of instances for tabular data from pandas dataframe

    Input:
        - X: pandas dataframe containing features
        - y: pandas series containing targets
        - mode: 'Classification' or 'Regression'
    Output:
        - X_list: list of instances
        - y_list: list of targets
    """

    X_list = []
    y_list = []

    if y is None:

        # group the data based on idx
        for idx, group_x in X.groupby("idx"):
            group_x = group_x.drop("idx", axis=1)
            if type(group_x) != np.ndarray:
                group_x = group_x.values
            X_list.append([group_x])

        return X_list
    else:
        if mode == "Classification":
            y_idx = y["idx"]
            # need to change to multi-class probability vector for classification
            encoder = OneHotEncoder(
                sparse_output=False, categories="auto", handle_unknown="ignore"
            )
            y = encoder.fit_transform(np.array(y[CFG["target"][0]]).reshape(-1, 1))
            y = pd.DataFrame(y, columns=encoder.get_feature_names_out(["target"]))
            y["idx"] = y_idx
            CFG["target"] = encoder.get_feature_names_out(["target"])

        for idx, group_x in X.groupby("idx"):
            group_x = group_x.drop("idx", axis=1)
            if type(group_x) != np.ndarray:
                group_x = group_x.values
            X_list.append([group_x])

            if len(CFG["target"]) == 1:
                group_y = y[y["idx"] == idx][CFG["target"][0]]
            else:
                group_y = y[y["idx"] == idx][CFG["target"]]

            if type(group_y) != np.ndarray:
                group_y = group_y.values
            y_list.append(group_y)

        return X_list, y_list
