"""
models
======

This module provides an implementation of quantile regression neural networks (QRNNs)
using the pytorch deep learning framework.
"""
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm.auto import tqdm

activations = {"elu" : nn.ELU,
               "hardshrink" : nn.Hardshrink,
               "hardtanh" : nn.Hardtanh,
               "prelu" : nn.PReLU,
               "relu" : nn.ReLU,
               "selu" : nn.SELU,
               "celu" : nn.CELU,
               "sigmoid" : nn.Sigmoid,
               "softplus" : nn.Softplus,
               "softmin" : nn.Softmin}


def save_model(f, model):
    """
    Save pytorch model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to store the data to.
        model(:code:`pytorch.nn.Moduel`): The pytorch model to save
    """
    torch.save(model, f)

def load_model(f, quantiles):
    """
    Load pytorch model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to read the model from
        quantiles(:code:`np.ndarray`): Array containing the quantiles
            that the model predicts.

    Returns:
        The loaded pytorch model.
    """
    model = torch.load(f)
    return model

def handle_input(data, device = None):
    """
    Handle input data.

    This function handles data supplied

      - as tuple of :code:`np.ndarray`
      - a single :code:`np.ndarray`
      - torch :code:`dataloader`

    If a numpy array is provided it is converted to a torch tensor
    so that it can be fed into a pytorch model.
    """
    if type(data) == tuple:
        x, y = data
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        if not device is None:
            x = x.to(device)
            y = y.to(device)
        return x, y
    if type(data) == np.ndarray:
        x = torch.tensor(data, dtype=torch.float)
        if not device is None:
            x = x.to(device)
        return x
    else:
        return data

class BatchedDataset(Dataset):
    """
    Batches an un-bactched dataset.
    """
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        # This is required because x and y are tensors and don't throw these
        # errors themselves.
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        i_start = i * self.batch_size
        i_end = (i + 1) * self.batch_size
        x = self.x[i_start : i_end]
        y = self.y[i_start : i_end]
        return (x, y)

################################################################################
# Quantile loss
################################################################################

class QuantileLoss:
    r"""
    The quantile loss function

    This function object implements the quantile loss defined as


    .. math::

        \mathcal{L}(y_\text{pred}, y_\text{true}) =
        \begin{cases}
        \tau \cdot |y_\text{pred} - y_\text{true}| & , y_\text{pred} < y_\text{true} \\
        (1 - \tau) \cdot |y_\text{pred} - y_\text{true}| & , \text{otherwise}
        \end{cases}


    as a training criterion for the training of neural networks. The loss criterion
    expects a vector :math:`\mathbf{y}_\tau` of predicted quantiles and the observed
    value :math:`y`. The loss for a single training sample is computed by summing the losses
    corresponding to each quantiles. The loss for a batch of training samples is
    computed by taking the mean over all samples in the batch.
    """
    def __init__(self,
                 quantiles,
                 mask = -1.0):
        """
        Create an instance of the quantile loss function with the given quantiles.

        Arguments:
            quantiles: Array or iterable containing the quantiles to be estimated.
        """
        self.quantiles = torch.tensor(quantiles).float()
        self.n_quantiles = len(quantiles)
        self.mask = np.float32(mask)

    def to(self, device):
        self.quantiles = self.quantiles.to(device)

    def __call__(self, y_pred, y_true):
        """
        Compute the mean quantile loss for given inputs.

        Arguments:
            y_pred: N-tensor containing the predicted quantiles along the last
                dimension
            y_true: (N-1)-tensor containing the true y values corresponding to
                the predictions in y_pred

        Returns:
            The mean quantile loss.
        """
        dy = y_pred - y_true
        n = self.quantiles.size()[0]
        qs = self.quantiles.reshape((n,) + (1, ) * max(len(dy.size()) - 2, 0))
        l = torch.where(dy >= 0.0,
                        (1.0 - qs) * dy,
                        (-qs) * dy)
        if not self.mask is None:
            l = torch.where(y_true == self.mask, torch.zeros_like(l), l)
        return l.mean()

################################################################################
# QRNN
################################################################################

class PytorchModel:
    """
    Quantile regression neural network (QRNN)

    This class implements QRNNs as a fully-connected network with
    a given number of layers.
    """
    def __init__(self,
                 input_dimension,
                 quantiles):
        """
        Arguments:
            input_dimension(int): The number of input features.
            quantiles(array): Array of the quantiles to predict.
        """
        self.input_dimension = input_dimension
        self.quantiles = np.array(quantiles)
        self.criterion = QuantileLoss(self.quantiles)
        self.training_errors = []
        self.validation_errors = []

    def _make_adversarial_samples(self, x, y, eps):
        self.model.zero_grad()
        x.requires_grad = True
        y_pred = self.model(x)
        c = self.criterion(y_pred, y)
        c.backward()
        x_adv = x.detach() + eps * torch.sign(x.grad.detach())
        return x_adv

    def train(self, *args, **kwargs):
        """
        Train the network.

        This trains the network for the given number of epochs using the
        provided training and validation data.

        If desired, the training can be augmented using adversarial training.
        In this case the network is additionally trained with an adversarial
        batch of examples in each step of the training.

        Arguments:
            training_data: pytorch dataloader providing the training data
            validation_data: pytorch dataloader providing the validation data
            n_epochs: the number of epochs to train the network for
            adversarial_training: whether or not to use adversarial training
            eps_adv: The scaling factor to use for adversarial training.
        """
        # Handle overload of train() method
        if len(args) < 1 or (len(args) == 1 and type(args[0]) == bool):
            return nn.Sequential.train(self, *args, **kwargs)

        #
        # Parse training arguments
        #

        training_data = args[0]
        arguments = {"validation_data" : None,
                     "batch_size" : 256,
                     "sigma_noise" : None,
                     "adversarial_training" : False,
                     "delta_at" : 0.01,
                     "initial_learning_rate" : 1e-2,
                     "momentum" : 0.0,
                     "convergence_epochs" : 5,
                     "learning_rate_decay" : 2.0,
                     "learning_rate_minimum" : 1e-6,
                     "maximum_epochs" : 1,
                     "training_split" : 0.9,
                     "gpu" : False}
        argument_names = arguments.keys()
        for a, n in zip(args[1:], argument_names):
            arguments[n] = a
        for k in kwargs:
            if k in arguments:
                arguments[k] = kwargs[k]
            else:
                raise ValueError("Unknown argument to {}.".print(k))

        validation_data = arguments["validation_data"]
        batch_size = arguments["batch_size"]
        sigma_noise = arguments["sigma_noise"]
        adversarial_training = arguments["adversarial_training"]
        delta_at = arguments["delta_at"]
        initial_learning_rate = arguments["initial_learning_rate"]
        convergence_epochs = arguments["convergence_epochs"]
        learning_rate_decay = arguments["learning_rate_decay"]
        learning_rate_minimum = arguments["learning_rate_minimum"]
        maximum_epochs = arguments["maximum_epochs"]
        training_split = arguments["training_split"]
        gpu = arguments["gpu"]
        momentum = arguments["momentum"]

        #
        # Determine device to use
        #
        if torch.cuda.is_available() and gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.to(device)

        #
        # Handle input data
        #
        try:
            x, y = handle_input(training_data, device)
            training_data = BatchedDataset(x, y, batch_size)
        except:
            pass


        self.train()
        self.optimizer = optim.SGD(self.parameters(),
                                   lr = initial_learning_rate,
                                   momentum = momentum)
        self.criterion.to(device)
        scheduler = ReduceLROnPlateau(self.optimizer,
                                            factor=1.0 / learning_rate_decay,
                                            patience=convergence_epochs,
                                            min_lr=learning_rate_minimum)
        training_errors = []
        validation_errors = []

        #
        # Training loop
        #

        for i in range(maximum_epochs):
            err = 0.0
            n = 0
            iterator = tqdm(enumerate(training_data), total = len(training_data))
            for j, (x, y) in iterator:
                x = x.to(device)
                y = y.to(device)

                shape = x.size()
                shape = (shape[0], 1) + shape[2:]
                y = y.reshape(shape)

                self.optimizer.zero_grad()
                y_pred = self(x)
                c = self.criterion(y_pred, y)
                c.backward()
                self.optimizer.step()

                err += c.item() * x.size()[0]
                n += x.size()[0]

                if adversarial_training:
                    self.optimizer.zero_grad()
                    x_adv = self._make_adversarial_samples(x, y, delta_at)
                    y_pred = self(x)
                    c = self.criterion(y_pred, y)
                    c.backward()
                    self.optimizer.step()

                if j % 100:
                    iterator.set_postfix({"Training errror" : err / n})

            # Save training error
            training_errors.append(err / n)

            val_err = 0.0
            n = 0
            if not validation_data is None:
                print(validation_data)
                for x, y in validation_data:
                    x = x.to(device)
                    y = y.to(device)

                    shape = x.size()
                    shape = (shape[0], 1) + shape[2:]
                    y = y.reshape(shape)

                    y_pred = self(x)
                    c = self.criterion(y_pred, y)

                    val_err += c.item() * x.size()[0]
                    n += x.size()[0]
            validation_errors.append(val_err)

        self.training_errors += training_errors
        self.validation_errors += validation_errors
        self.eval()
        return {"training_errors" : self.training_errors,
                "validation_errors" : self.validation_errors}

    def predict(self, x, gpu=False):
        ""
        if torch.cuda.is_available() and gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = handle_input(x, device)
        self.to(device)
        return self(x).detach()

    def calibration(self,
                    data,
                    gpu=False):
        """
        Computes the calibration of the predictions from the neural network.

        Arguments:
            data: torch dataloader object providing the data for which to compute
                the calibration.

        Returns:
            (intervals, frequencies): Tuple containing the confidence intervals and
                corresponding observed frequencies.
        """

        if gpu and torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        self.to(dev)

        n_intervals = self.quantiles.size // 2
        qs = self.quantiles
        intervals = np.array([q_r - q_l for (q_l, q_r)
                              in zip(qs, reversed(qs))])[:n_intervals]
        counts = np.zeros(n_intervals)

        total = 0.0

        iterator = tqdm(data)
        for x, y in iterator:
            x = x.to(dev)
            y = y.to(dev)
            shape = x.size()
            shape = (shape[0], 1) + shape[2:]
            y = y.reshape(shape)

            y_pred = self.predict(x)
            y_pred = y_pred.cpu()
            y = y.cpu()

            for i in range(n_intervals):
                l = y_pred[:, [i]]
                r = y_pred[:, [-(i + 1)]]
                counts[i] += np.logical_and(y >= l, y < r).sum()

            total += np.prod(y.size())
        return intervals[::-1], (counts / total)[::-1]

    def save(self, path):
        """
        Save QRNN to file.

        Arguments:
            The path in which to store the QRNN.
        """
        torch.save({"input_dimension" : self.input_dimension,
                    "quantiles" : self.quantiles,
                    "width" : self.width,
                    "depth" : self.depth,
                    "activation" : self.activation,
                    "network_state" : self.state_dict(),
                    "optimizer_state" : self.optimizer.state_dict()},
                    path)

    @staticmethod
    def load(self, path):
        """
        Load QRNN from file.

        Arguments:
            path: Path of the file where the QRNN was stored.
        """
        state = torch.load(path)
        keys = ["input_dimension", "quantiles", "depth", "width", "activation"]
        qrnn = QRNN(*[state[k] for k in keys])
        qrnn.load_state_dict["network_state"]
        qrnn.optimizer.load_state_dict["optimizer_state"]

################################################################################
# Fully-connected network
################################################################################

class FullyConnected(PytorchModel, nn.Sequential):
    """
    Pytorch implementation of a fully-connected QRNN model.
    """
    def __init__(self,
                 input_dimension,
                 quantiles,
                 arch):

        """
        Create a fully-connected neural network.

        Args:
            input_dimension(:code:`int`): Number of input features
            quantiles(:code:`array`): The quantiles to predict given
                as fractions within [0, 1].
            arch(tuple): Tuple :code:`(d, w, a)` containing :code:`d`, the
                number of hidden layers in the network, :code:`w`, the width
                of the network and :code:`a`, the type of activation functions
                to be used as string.
        """
        PytorchModel.__init__(self, input_dimension, quantiles)
        output_dimension = quantiles.size
        self.arch = arch

        if len(arch) == 0:
            layers = [nn.Linear(input_dimension, output_dimension)]
        else:
            d, w, act = arch
            if type(act) == str:
                act = activations[act]
            layers = [nn.Linear(input_dimension, w)]
            for i in range(d - 1):
                layers.append(nn.Linear(w, w))
                if not act is None:
                    layers.append(act())
            layers.append(nn.Linear(w, output_dimension))
        nn.Sequential.__init__(self, *layers)
