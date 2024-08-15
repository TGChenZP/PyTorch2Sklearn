from environment import *
from utils import *


class TorchToSklearn_Model(object):
    # serves both for sklearn model class framework for classification and regression
    class Model():
        def _init_(self, CFG):
            pass

    def __init__(self, CFG, name='Model'):
        """ Initialise the model """

        super().__init__()
        self.CFG = CFG
        self.name = name
        self._is_fitted = False

        assert self.CFG['mode'] in [
            'Classification', 'Regression'], 'mode must be either Classification or Regression'

        self.model = self.Model(self.CFG)  # initialise model

        torch.manual_seed(self.CFG['random_state'])

        self.optimizer = AdamW(self.model.parameters(), lr=self.CFG['lr'])
        self.criterion = self.CFG['loss']

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.CFG['device'] = self.device
        self.model.to(self.device)
        self.criterion.to(self.device)

        if self.CFG['mode'] == 'Classification':
            self._estimator_type = "classifier"
            self.classes_ = [x for x in range(self.CFG['output_dim'])]
        elif self.CFG['mode'] == 'Regression':
            self._estimator_type = "regressor"

    def __str__(self):
        return self.name

    def save(self, mark=''):
        """ Save the model """

        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(
            self.CFG['rootpath'], f'state/{self}{mark}.pt'))

    def load(self, mark=''):
        """ Load the model """

        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(
            self.CFG['rootpath'], f'state/{self}{mark}.pt'), map_location=self.device))

    def fit(self, train_x, train_y):
        """ Fit the model 

            Input: 
                - train_x: pd.DataFrame, containing features
                - train_y: pd.DataFrame, containing targets
        """

        self.model.train()

        # if classification turn labels into e.g. 0 1 2 3 so data factory can turn into probability vectors
        if self.CFG['mode'] == 'Classification':
            label_list = list(set(train_y))
            label_list.sort()
            self.encode_label_dict = {
                label: i for i, label in enumerate(label_list)}
            self.decode_label_dict = {
                i: label for i, label in enumerate(label_list)}

            train_y = [self.encode_label_dict[y] for y in train_y]

        x_list, y_list = self.CFG['TabularDataFactory'](
            train_x, train_y, self.CFG['mode'])  # might want to do batch number as well
        tabular_dataset = self.CFG['TabularDataset'](x_list, y_list)

        np.random.seed(self.CFG['random_state'])
        torch.manual_seed(self.CFG['random_state'])
        seeds = [np.random.randint(1, 1000) for _ in range(self.CFG['epochs'])]

        for epoch in range(self.CFG['epochs']):

            tabular_dataloader = DataLoader(tabular_dataset,
                                            batch_size=self.CFG['batch_size'],
                                            shuffle=True,
                                            generator=torch.Generator().manual_seed(seeds[epoch]))

            for batch in tabular_dataloader:
                X, y = batch[0].to(self.device), batch[1].to(self.device)

                if self.CFG['mode'] == 'Regression':
                    y = y.view(-1, 1)

                self.optimizer.zero_grad()
                pred = self.model(X)

                loss = self.criterion(pred, y)
                loss.backward()
                if self.CFG['grad_clip']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()

        self._is_fitted = True

    def predict(self, val_x):
        """ Predict labels for classification tasks and values for regression tasks. 

            Input:
                - val_x: pd.DataFrame, containing features

            Output:
                - valid_pred: list, containing predicted labels or values

        """

        assert self._is_fitted, 'model has not been fitted yet'

        self.model.eval()

        with torch.no_grad():

            x_list = self.CFG['TabularDataFactory'](
                val_x, mode=self.CFG['mode'])  # create the instances
            tabular_dataset = self.CFG['TabularDataset'](x_list)

            val_dataloader = DataLoader(
                tabular_dataset, batch_size=self.CFG['batch_size'], shuffle=False)

            valid_pred = []

            for batch in val_dataloader:
                x_batch = batch.to(self.device)

                pred = self.model(x_batch)

                # if regression, squeeze the dimension, if classification, argmax.
                predicted_labels = torch.argmax(
                    pred, dim=1) if self.CFG['mode'] == 'Classification' else pred.squeeze(1)
                valid_pred += predicted_labels.detach().cpu().tolist()

        if self.CFG['mode'] == 'Classification':  # turn labels back to normal
            valid_pred = [self.decode_label_dict[pred] for pred in valid_pred]

        return valid_pred

    def predict_proba(self, val_x):
        """ Predict probabilities for classification tasks. 

            Input:

                - val_x: pd.DataFrame, containing features

        """

        assert self._is_fitted, 'model has not been fitted yet'
        self.model.eval()

        with torch.no_grad():
            x_list = self.CFG['TabularDataFactory'](
                val_x, mode=self.CFG['mode'])
            tabular_dataset = self.CFG['TabularDataset'](x_list)
            val_dataloader = DataLoader(
                tabular_dataset, batch_size=self.CFG['batch_size'], shuffle=False)

            valid_pred_probs = []

            for batch in val_dataloader:
                x_batch = batch.to(self.device)
                pred_probs = self.model(x_batch).softmax(
                    dim=1) if self.CFG['mode'] == 'Classification' else self.model(x_batch).squeeze(1).sigmoid()
                valid_pred_probs.append(pred_probs.detach().cpu().numpy())

            valid_pred_probs = np.concatenate(valid_pred_probs, axis=0)

        return valid_pred_probs

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
