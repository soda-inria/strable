"""TARTE finetune estimators for regression and classification."""

import torch
import numpy as np
import copy
from typing import Union
from torcheval.metrics import (
    MeanSquaredError,
    R2Score,
    BinaryAUROC,
    BinaryNormalizedEntropy,
    BinaryAUPRC,
    MulticlassAUROC,
)
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.special import softmax
from tarte_ai.tarte_model import TARTE_Downstream_NN
from tarte_ai.tarte_utils import load_tarte_pretrain_model


class TARTETabularDataset(Dataset):
    """PyTorch Dataset used for dataloader."""

    def __init__(self, X):
        self.X, self.edge_attr, self.mask, self.y = zip(
            *((x, edge_attr, mask, y) for _, x, edge_attr, mask, y in X)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.edge_attr[idx], self.mask[idx], self.y[idx]


class BaseTARTEFinetuneEstimator(BaseEstimator):
    """Base class for TARTE Estimator."""

    def __init__(
        self,
        *,
        num_layers,
        num_heads,
        dim_transformer,
        dim_feedforward,
        load_pretrain,
        finetune_strategy,
        learning_rate,
        batch_size,
        max_epoch,
        dropout,
        val_size,
        cross_validate,
        early_stopping_patience,
        shuffle_train,
        num_model,
        random_state,
        device,
        disable_pbar,
        pretrained_model_path,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_transformer = dim_transformer
        self.dim_feedforward = dim_feedforward
        self.load_pretrain = load_pretrain
        self.finetune_strategy = finetune_strategy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dropout = dropout
        self.val_size = val_size
        self.cross_validate = cross_validate
        self.early_stopping_patience = early_stopping_patience
        self.shuffle_train = shuffle_train
        self.num_model = num_model
        self.random_state = random_state
        self.device = device
        self.disable_pbar = disable_pbar
        self.pretrained_model_path = pretrained_model_path

    def fit(self, X, y, eval_set):
        """Fit the TARTE model.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
               Fitted estimator.
        """
        # Preliminary settings
        self.is_fitted_ = False
        self.device_ = torch.device(self.device)
        self.dim_input_ = X[0][1].size(1)
        self._set_task_specific_settings(y)

        # Run training
        self.model_best_, self.valid_loss_best_ = self._run_train_with_early_stopping(
            X, eval_set
        )

        self.is_fitted_ = True

        return self

    def _run_train_with_early_stopping(self, X_train, eval_set):
        """Train each model corresponding to the random_state with the early_stopping patience.

        This mode of training sets train/valid set for the early stopping criterion.
        Returns the trained model, and the validation loss at the best epoch.
        """

        # Set datasets
        ds_train = TARTETabularDataset(X_train)
        ds_valid = TARTETabularDataset(eval_set[0][0])

        # Load model and optimizer
        model_run_train = self._load_model()
        model_run_train = model_run_train.to(self.device_)

        optimizer = torch.optim.AdamW(
            model_run_train.parameters(),
            lr=self.learning_rate,
        )

        # Set dataloader for train and valid
        train_loader = DataLoader(
            ds_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
        )
        valid_loader = DataLoader(ds_valid, batch_size=len(ds_valid), shuffle=False)

        # Set validation batch for evaluation
        ds_valid_eval = next(iter(valid_loader))

        # Train model
        valid_loss_best = 9e15
        es_counter = 0
        model_best_ = copy.deepcopy(model_run_train)
        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. xx",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_train, optimizer, train_loader)
            valid_loss = self._eval(model_run_train, ds_valid_eval)
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                model_best_ = copy.deepcopy(model_run_train)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > self.early_stopping_patience:
                    break
        model_best_.eval()
        return model_best_, valid_loss_best

    def _run_epoch(self, model, optimizer, train_loader):
        """Run an epoch of the input model.

        Each epoch consists of steps that update the model and the optimizer.
        """
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            self._run_step(model, data, optimizer)

    def _run_step(self, model, data, optimizer):
        """Run a step of the training.

        With each step, it updates the model and the optimizer.
        """
        optimizer.zero_grad()  # Clear gradients.
        # Send to device
        data[0] = data[0].to(self.device_)
        data[1] = data[1].to(self.device_)
        data[2] = data[2].to(self.device_)
        data[-1] = data[-1].to(self.device_)
        # Feed-Foward
        out = model(data[0], data[1], data[2])  # Perform a single forward pass.
        target = data[-1].view(-1).to(torch.float32)  # Set target
        if self.loss == "categorical_crossentropy":
            target = target.to(torch.long)
        if self.output_dim_ == 1:
            out = out.view(-1).to(torch.float32)  # Reshape output
            target = target.to(torch.float32)  # Reshape target
        loss = self.criterion_(out, target)  # Compute the loss.
        loss.backward()  # Scale the loss and backward pass
        optimizer.step()  # Update parameters

    def _eval(self, model, ds_eval):
        """Run an evaluation of the input data on the input model.

        Returns the selected loss of the input data from the input model.
        """
        with torch.no_grad():
            model.eval()
            # Send to device
            ds_eval[0] = ds_eval[0].to(self.device_)
            ds_eval[1] = ds_eval[1].to(self.device_)
            ds_eval[2] = ds_eval[2].to(self.device_)
            ds_eval[-1] = ds_eval[-1].to(self.device_)
            # Feed-Foward
            out = model(ds_eval[0], ds_eval[1], ds_eval[2])
            target = ds_eval[-1].view(-1).to(torch.float32)
            if self.loss == "categorical_crossentropy":
                target = target.to(torch.long)
            if self.output_dim_ == 1:
                out = out.view(-1).to(torch.float32)
                target = target.to(torch.float32)
            self.valid_loss_metric_.update(out, target)
            loss_eval = self.valid_loss_metric_.compute()
            loss_eval = loss_eval.detach().item()
            if self.valid_loss_flag_ == "neg":
                loss_eval = -1 * loss_eval
            self.valid_loss_metric_.reset()
        return loss_eval

    def _predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, "is_fitted_")

        if self._estimator_type == "regressor":
            return self._predict_proba(X)
        else:
            if self.loss == "binary_crossentropy":
                return np.round(self._predict_proba(X))
            elif self.loss == "categorical_crossentropy":
                return np.argmax(self._predict_proba(X), axis=1)

    def _predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples,) for binary classification or (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, "is_fitted_")
        return self._get_predict_prob(X)

    def _get_predict_prob(self, X):
        """Return the average of the outputs over all the models.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        raw_predictions : array, shape (n_samples,)
            The raw predicted values.
        """

        return self._generate_output(X=X, model=self.model_best_)

    def _generate_output(self, X, model):
        """Generate the output from the trained model.

        Returns the output (prediction) of input X.
        """

        # Set the test_loader
        ds_test = TARTETabularDataset(X)
        test_loader = DataLoader(ds_test, batch_size=len(X), shuffle=False)

        # Obtain the batch to feed into the network
        ds_predict_eval = next(iter(test_loader))
        with torch.no_grad():
            # Send to device
            ds_predict_eval[0] = ds_predict_eval[0].to(self.device_)
            ds_predict_eval[1] = ds_predict_eval[1].to(self.device_)
            ds_predict_eval[2] = ds_predict_eval[2].to(self.device_)
            ds_predict_eval[-1] = ds_predict_eval[-1].to(self.device_)
            # Generate output
            out = (
                model(ds_predict_eval[0], ds_predict_eval[1], ds_predict_eval[2])
                .cpu()
                .detach()
                .numpy()
            )

        # Change if the task is classification
        if self.loss == "binary_crossentropy":
            out = 1 / (1 + np.exp(-out))
        elif self.loss == "categorical_crossentropy":
            out = softmax(out, axis=1)

        # Control for nulls in prediction
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred

        if out.ndim == 2 and out.shape[1] == 1:
            out = out.squeeze(axis=1)  # we don't want to squeeze first axis

        return out

    def _set_task_specific_settings(self, y):
        """Set task specific settings for regression and classfication."""

        if self._estimator_type == "regressor":
            if self.loss == "squared_error":
                self.criterion_ = torch.nn.MSELoss()
            elif self.loss == "absolute_error":
                self.criterion_ = torch.nn.L1Loss()
            if self.scoring == "squared_error":
                self.valid_loss_metric_ = MeanSquaredError()
                self.valid_loss_flag_ = "pos"
            elif self.scoring == "r2_score":
                self.valid_loss_metric_ = R2Score()
                self.valid_loss_flag_ = "neg"
            self.output_dim_ = 1
        elif self._estimator_type == "classifier":
            self.classes_ = np.unique(y)
            if self.loss == "binary_crossentropy":
                self.criterion_ = torch.nn.BCEWithLogitsLoss()
                self.output_dim_ = 1
                if self.scoring == "auroc":
                    self.valid_loss_metric_ = BinaryAUROC()
                    self.valid_loss_flag_ = "neg"
                elif self.scoring == "binary_entropy":
                    self.valid_loss_metric_ = BinaryNormalizedEntropy(from_logits=True)
                    self.valid_loss_flag_ = "neg"
                elif self.scoring == "auprc":
                    self.valid_loss_metric_ = BinaryAUPRC()
                    self.valid_loss_flag_ = "neg"
            elif self.loss == "categorical_crossentropy":
                self.criterion_ = torch.nn.CrossEntropyLoss()
                self.output_dim_ = len(np.unique(y))
                self.valid_loss_metric_ = MulticlassAUROC(num_classes=self.output_dim_)
                self.valid_loss_flag_ = "neg"
        self.valid_loss_metric_.to(self.device_)

    def _set_train_valid_split(self, X, y):
        """Train/validation split for the bagging strategy.

        The style of split depends on the cross_validate parameter.
        Reuturns the train/validation split with KFold cross-validation.
        """

        if self._estimator_type == "regressor":
            if self.cross_validate:
                n_splits = int(1 / self.val_size)
                n_repeats = int(self.num_model / n_splits)
                splitter = RepeatedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=self.random_state,
                )
            else:
                splitter = ShuffleSplit(
                    n_splits=self.num_model,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )
            splits = [
                (train_index, test_index)
                for train_index, test_index in splitter.split(np.arange(0, len(X)))
            ]
        else:
            if self.cross_validate:
                n_splits = int(1 / self.val_size)
                n_repeats = int(self.num_model / n_splits)
                splitter = RepeatedStratifiedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=self.random_state,
                )
            else:
                splitter = StratifiedShuffleSplit(
                    n_splits=self.num_model,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )
            splits = [
                (train_index, test_index)
                for train_index, test_index in splitter.split(np.arange(0, len(X)), y)
            ]

        return splits

    def _load_model(self):
        """Load the TARTE model for training.

        This loads the pretrained weights if the parameter load_pretrain is set to True.
        The freeze of the pretrained weights are controlled by the freeze_pretrain parameter.

        Returns the model that can be used for training.
        """
        # Model configuration
        model_config = dict()
        model_config["dim_input"] = self.dim_input_
        model_config["num_heads"] = self.num_heads
        model_config["num_layers_transformer"] = self.num_layers
        model_config["dim_transformer"] = self.dim_transformer
        model_config["dim_feedforward"] = self.dim_feedforward
        model_config["dim_output"] = self.output_dim_
        model_config["dropout"] = self.dropout

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        # Set model architecture
        model = TARTE_Downstream_NN(**model_config)

        # Load the pretrained weights if specified
        if self.load_pretrain:
            # With designated path
            if self.pretrained_model_path is not None:
                pretrain_model_dict = torch.load(
                    self.pretrained_model_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            # Without designated path
            else:
                pretrain_model_dict, _ = load_tarte_pretrain_model()

            # Load the pretrain weights
            model.load_state_dict(pretrain_model_dict, strict=False)

            # Set based on finetuning strategy
            if self.finetune_strategy == "freeze":
                for param in model.tarte_base.transformer_encoder.parameters():
                    param.requires_grad = False

        return model


class TARTEFinetuneRegressor(RegressorMixin, BaseTARTEFinetuneEstimator):
    """TARTE Regressor for Regression tasks.

    This estimator is compatible with the TARTE pretrained model.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        The loss function used for backpropagation.
    scoring : {'r2_score', 'squared_error'}, default='r2_score'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the NN model.
    num_heads : int, default=1
        The number of multiheads for the NN model.
    dim_transformer : int, default=768
        The dimension of the transformer encoder model.
    dim_feedforward : int, default=2048
        The dimension of the feed-foward in the transformer encoder model.
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    finetune_strategy : {'freeze', 'full'}, default='freeze'
        The finetuning strategy, with pretrained weights.
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    cross_validate : bool, default=False
        Indicates whether to use cross-validation strategy for train/validation split
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    shuffle_train : bool, default=False
        Indicates whether to shuffle the train data for batch.
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "cuda"}, default="cuda",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    pretrained_model_path : str or None, default=None
        The path of pretrained model used to finetune.
    """

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        scoring: str = "r2_score",
        num_layers: int = 1,
        num_heads: int = 1,
        dim_transformer: int = 768,
        dim_feedforward: int = 2048,
        load_pretrain: bool = True,
        finetune_strategy: str = "freeze",
        learning_rate: float = 5e-4,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        cross_validate: bool = False,
        early_stopping_patience: Union[None, int] = 40,
        shuffle_train: bool = False,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path: Union[None, str] = None,
    ):
        super(TARTEFinetuneRegressor, self).__init__(
            num_layers=num_layers,
            num_heads=num_heads,
            dim_transformer=dim_transformer,
            dim_feedforward=dim_feedforward,
            load_pretrain=load_pretrain,
            finetune_strategy=finetune_strategy,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            cross_validate=cross_validate,
            early_stopping_patience=early_stopping_patience,
            shuffle_train=shuffle_train,
            num_model=num_model,
            random_state=random_state,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
        )

        self.loss = loss
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(self, X, y=None, eval_set=None):
        """Fit the TARTE model.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
               Fitted estimator.
        """
        # Preliminary settings
        self.is_fitted_ = False
        self.device_ = torch.device(self.device)
        self.dim_input_ = X[0][1].size(1)
        self._set_task_specific_settings(y)

        # Set the splits for early-stopping
        if eval_set is None:
            splits = self._set_train_valid_split(X, y)
            self.result_ = Parallel(n_jobs=self.n_jobs)(
                delayed(super(TARTEFinetuneRegressor, self).fit)(
                    [X[i] for i in split_index[0]],
                    y[split_index[0]],
                    ([X[i] for i in split_index[1]], y[split_index[1]]),
                )
                for split_index in splits
            )
        else:
            self.result_ = [super(TARTEFinetuneRegressor, self).fit(X, y, eval_set)]

        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Predict values for X. Returns the average of predicted values over all the models.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted values.
        """

        check_is_fitted(self, "is_fitted_")

        return np.mean([estimator._predict(X) for estimator in self.result_], axis=0)


class TARTEFinetuneClassifier(ClassifierMixin, BaseTARTEFinetuneEstimator):
    """TARTE Classifier for Classification tasks.

    This estimator compatible with the TARTE pretrained model.

    Parameters
    ----------
    loss : {'binary_crossentropy', 'categorical_crossentropy'}, default='binary_crossentropy'
        The loss function used for backpropagation.
    scoring : {'auroc', 'auprc', 'binary_entropy'}, default='auroc'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the NN model
    num_heads : int, default=1
        The number of multiheads for the NN model
    dim_transformer : int, default=768
        The dimension of the transformer encoder model.
    dim_feedforward : int, default=2048
        The dimension of the feed-foward in the transformer encoder model.
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    finetune_strategy : {'freeze', 'full'}, default='freeze'
        The finetuning strategy, with pretrained weights.
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    cross_validate : bool, default=False
        Indicates whether to use cross-validation strategy for train/validation split
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    shuffle_train : bool, default=False
        Indicates whether to shuffle the train data for batch.
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "cuda"}, default="cpu",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    pretrained_model_path : str or None, default=None
        The path of pretrained model used to finetune.
    """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        scoring: str = "auroc",
        num_layers: int = 1,
        num_heads: int = 1,
        dim_transformer: int = 768,
        dim_feedforward: int = 2048,
        load_pretrain: bool = True,
        finetune_strategy: str = "freeze",
        learning_rate: float = 5e-4,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        cross_validate: bool = False,
        early_stopping_patience: Union[None, int] = 40,
        shuffle_train: bool = False,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path: Union[None, str] = None,
    ):
        super(TARTEFinetuneClassifier, self).__init__(
            num_layers=num_layers,
            num_heads=num_heads,
            dim_transformer=dim_transformer,
            dim_feedforward=dim_feedforward,
            load_pretrain=load_pretrain,
            finetune_strategy=finetune_strategy,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            cross_validate=cross_validate,
            early_stopping_patience=early_stopping_patience,
            shuffle_train=shuffle_train,
            num_model=num_model,
            random_state=random_state,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
        )

        self.loss = loss
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(self, X, y=None, eval_set=None):
        """Fit the TARTE model.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
               Fitted estimator.
        """
        # Preliminary settings
        self.is_fitted_ = False
        self.device_ = torch.device(self.device)
        self.dim_input_ = X[0][1].size(1)
        self._set_task_specific_settings(y)

        # Set the splits for early-stopping
        if eval_set is None:
            splits = self._set_train_valid_split(X, y)
            self.result_ = Parallel(n_jobs=self.n_jobs)(
                delayed(super(TARTEFinetuneClassifier, self).fit)(
                    [X[i] for i in split_index[0]],
                    y[split_index[0]],
                    ([X[i] for i in split_index[1]], y[split_index[1]]),
                )
                for split_index in splits
            )
        else:
            self.result_ = [super(TARTEFinetuneClassifier, self).fit(X, y, eval_set)]

        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, "is_fitted_")

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples,) for binary classification or (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, "is_fitted_")

        return np.mean(
            [estimator._predict_proba(X) for estimator in self.result_], axis=0
        )

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        decision : ndarray, shape (n_samples,)
        """
        decision = self.predict_proba(X)
        if decision.shape[1] == 1:
            decision = decision.ravel()
        return decision
