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
from tqdm.auto import tqdm

activations = {"elu" : nn.ELU,
               "hardshrink" : nn.Hardshrink,
               "hardtanh" : nn.Hardtanh,
               "prelu" : nn.PReLU,
               "selu" : nn.SELU,
               "celu" : nn.CELU,
               "sigmoid" : nn.Sigmoid,
               "softplus" : nn.Softplus,
               "softmin" : nn.Softmin}

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

class PytorchQRNN:
    """
    Quantile regression neural network (QRNN)

    This class implements QRNNs as a fully-connected network with
    a given number of layers.
    """
    def __init__(self,
                 input_dimension,
                 quantiles,
                 model = (3, 128, "relu"),
                 depth = 3,
                 width = 128,
                 activation = nn.ReLU):
        """
        Arguments:
            input_dimension(int): The number of input features.
            quantiles(array): Array of the quantiles to predict.
            depth(int): Number of layers of the network.
            width(int): The number of neurons in the inner layers of the network.
            activation: The activation function to use for the inner layers of the network.
        """
        self.input_dimension = input_dimension
        self.quantiles = np.array(quantiles)
        self.depth = depth
        self.width = width
        self.activation = nn.ReLU
        self.criterion = QuantileLoss(self.quantiles)

        n_quantiles = len(quantiles)

        if type(model) is tuple:
            depth = model[0]
            width = model[1]
            act = model[2]
            if type(act) is str:
                if act in activations:
                    act = activations[act]
                else:
                    raise ValueError("{} is not one of the available "
                                     " activations ({}) to use in a pytorch "
                                     "network.".format(act,
                                                       list(activations.keys())))

            self.model = nn.Sequential()
            self.model.add_module("fc_0", nn.Linear(input_dimension, width))
            self.model.add_module("act_0", activation())
            for i in range(1, depth - 1):
                self.model.add_module("fc_{}".format(i), nn.Linear(width, width))
                self.model.add_module("act_{}".format(i), activation())
                self.model.add_module("fc_{}".format(depth - 1), nn.Linear(width, n_quantiles))

            self.criterion = QuantileLoss(self.quantiles)
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise ValueError("Provided model must either be a valid architecture"
                             " tuple or an instance of pytorch.nn.Module.")

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

    def train(self,
              training_data,
              validation_data,
              n_epochs = 1,
              adversarial_training = False,
              eps_adv = 1e-6,
              lr = 1e-2,
              momentum = 0.0,
              gpu = False):

        """
        Train the network.

        This trains the network for the given number of epochs using the provided
        training and validation data.

        If desired, the training can be augmented using adversarial training. In this
        case the network is additionally trained with an adversarial batch of examples
        in each step of the training.

        Arguments:
            training_data: pytorch dataloader providing the training data
            validation_data: pytorch dataloader providing the validation data
            n_epochs: the number of epochs to train the network for
            adversarial_training: whether or not to use adversarial training
            eps_adv: The scaling factor to use for adversarial training.
        """
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr = lr,
                                   momentum = momentum)

        self.model.train()

        if torch.cuda.is_available() and gpu:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        self.model.to(dev)

        self.criterion.to(dev)

        def clear_output():
            print("clreaing output")
            try:
                import IPython
                from IPython.display import clear_output
                clear_output(wait=True)
            except:
                print("failed loading Ipython")
                pass

        for i in range(n_epochs):
            clear_output()

            err = 0.0
            n = 0
            iterator = tqdm(enumerate(training_data), total = len(training_data))
            iterator.set_description(f"Training epoch {i}/{n_epochs}, lr = {self.optimizer.lr}")
            for j, (x, y) in iterator:
                x = x.to(dev)
                y = y.to(dev)

                shape = x.size()
                y = y.reshape((shape[0], 1, shape[2], shape[3]))

                self.optimizer.zero_grad()
                y_pred = self.model(x)
                c = self.criterion(y_pred, y)
                c.backward()
                self.optimizer.step()

                err += c.item() * x.size()[0]
                n += x.size()[0]

                if adversarial_training:
                    self.optimizer.zero_grad()
                    x_adv = self._make_adversarial_samples(x, y, eps_adv)
                    y_pred = self.model(x)
                    c = self.criterion(y_pred, y)
                    c.backward()
                    self.optimizer.step()

                if j % 100:
                    iterator.set_postfix({"Training errror" : err / n})

            # Save training error
            self.training_errors.append(err / n)

            val_err = 0.0
            n = 0

            for x, y in validation_data:
                x = x.to(dev)
                y = y.to(dev)

                shape = x.size()
                y = y.reshape((shape[0], 1, shape[2], shape[3]))

                y_pred = self.model(x)
                c = self.criterion(y_pred, y)

                val_err += c.item() * x.size()[0]
                n += x.size()[0]

            self.validation_errors.append(val_err / n)
        self.model.eval()

    def predict(self, x):
        """
        Predict quantiles for given input.

        Args:
            x: 2D tensor containing the inputs for which for which to
                predict the quantiles.

        Returns:
            tensor: 2D tensor containing the predicted quantiles along
                the last dimension.
        """
        return self.model(x)

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
        self.model.to(dev)

        n_intervals = self.quantiles.size // 2
        qs = self.quantiles
        intervals = np.array([q_r - q_l for (q_l, q_r) in zip(qs, reversed(qs))])[:n_intervals]
        counts = np.zeros(n_intervals)

        total = 0.0

        iterator = tqdm(data)
        for x, y in iterator:
            x = x.to(dev)
            y = y.to(dev)
            shape = x.size()
            y = y.reshape((shape[0], 1, shape[2], shape[3]))

            y_pred = self.predict(x)
            y_pred = y_pred.cpu()
            y = y.cpu()

            for i in range(n_intervals):
                l = y_pred[:, i]
                r = y_pred[:, -(i + 1)]
                counts[i] += np.logical_and(y.cpu() >= l, y.cpu() < r).sum()

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
                    "network_state" : self.model.state_dict(),
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
        qrnn.model.load_state_dict["network_state"]
        qrnn.optimizer.load_state_dict["optimizer_state"]
