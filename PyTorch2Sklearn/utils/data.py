from PyTorch2Sklearn.environment import *


class TabularDataset(Dataset):

    """ Creates list of instances for tabular data from pandas dataframe """

    def __init__(self, x_list, y_list=None):
        """

            Input:
                - x_list: list of features
                - y_list: list of targets

        """

        self.features = torch.tensor(np.array(x_list), dtype=torch.float32)
        self.targets = torch.tensor(
            np.array(y_list), dtype=torch.float32) if y_list is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


def TabularDataFactory(X, y=None, mode='Classification'):
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
        if mode == 'Classification':
            # need to change to multi-class probability vector for classification
            encoder = OneHotEncoder(
                sparse_output=False, categories='auto', handle_unknown='ignore')
            y = encoder.fit_transform(np.array(y).reshape(-1, 1))

        if type(X) != np.ndarray:
            X = X.values
        if type(y) != np.ndarray:
            y = y.values

        print(type(X), type(y))

        return X.tolist(), y.tolist()
