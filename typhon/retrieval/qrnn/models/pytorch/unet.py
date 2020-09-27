import torch
from torch import nn

from typhon.retrieval.qrnn.models.pytorch.common import PytorchModel


class Layer(nn.Sequential):
    """
    Basic building block of a UNet. Consists of a convolutional
    layer followed by an activation layers and an optional batch
    norm layer.

    Args:
        features_in(:code:`int`): Number of features of input
        features_out(:code:`int`): Raw number of output features of the
            layer not including skip connections.
        batch_norm(:code:`bool`): Whether or not to include a batch norm
            layer.
        kernel_size(:code:`int`): Kernel size to use for the conv. layer.
        activation(:code:`activation`): Activation to use after conv. layer.
        skip_connection(:code:`bool`): Whether to include skip connections, i.e.
            to include input in layer output.
    """

    def __init__(
        self,
        features_in,
        features_out,
        batch_norm=True,
        kernel_size=3,
        activation=nn.ReLU,
        skip_connection=False,
    ):
        self._features_in = features_in
        self._features_out = features_out
        self.skip_connection = skip_connection

        if not activation is None:
            modules = [
                nn.ConstantPad2d(1, 0.0),
                nn.Conv2d(features_in, features_out, kernel_size),
                nn.BatchNorm2d(features_out),
                activation(),
            ]
        else:
            modules = [
                nn.ConstantPad2d(1, 0.0),
                nn.Conv2d(features_in, features_out, kernel_size),
                nn.BatchNorm2d(features_out),
            ]
        super().__init__(*modules)

    @property
    def features_out(self):
        if self.skip_connection:
            return self._features_in + self._features_out
        else:
            return self._features_out

    def forward(self, x):
        y = nn.Sequential.forward(self, x)
        if self.skip_connection:
            y = torch.cat([x, y], dim=1)
        return y


class Block(nn.Sequential):
    """
    A block bundles a set of layers.

    Args:
        features_in(:code:`int`): The number of input features of the block
        features_out(:code:`int`): The number of output features of the block.
        depth(:code:`int`): The number of layers of the block
        activation(:code:`nn.Module`): Pytorch activation layer to
          use. :code:`nn.ReLU` by default.
        skip_connection(:code:`str`): Whether or not to insert skip
          connections before all layers (:code:`"all"`) or just at
          the end (:code:`"end"`).
    """

    def __init__(
        self,
        features_in,
        features_out,
        depth=2,
        batch_norm=True,
        activation=nn.ReLU,
        kernel_size=3,
        skip_connection=None,
    ):

        self._features_in = features_in

        if skip_connection == "all":
            skip_connection_layer = True
            self.skip_connection = False
        elif skip_connection == "end":
            skip_connection_layer = False
            self.skip_connection = True
        else:
            skip_connection_layer = False
            self.skip_connection = False

        layers = []
        nf = features_in
        for d in range(depth):
            layers.append(
                Layer(
                    nf,
                    features_out,
                    activation=activation,
                    batch_norm=batch_norm,
                    kernel_size=kernel_size,
                    skip_connection=skip_connection_layer,
                )
            )
            nf = layers[-1].features_out

        self._features_out = layers[-1].features_out
        super().__init__(*layers)

    @property
    def features_out(self):
        if self.skip_connection:
            return self._features_in + self._features_out
        else:
            return self._features_out

    def forward(self, x):
        y = nn.Sequential.forward(self, x)
        if self.skip_connection:
            y = torch.cat([x, y], dim=1)
        return y


class DownSampler(nn.Sequential):
    def __init__(self):
        modules = [nn.MaxPool2d(2)]
        super().__init__(*modules)


class UpSampler(nn.Sequential):
    def __init__(self, features_in, features_out):
        modules = [
            nn.ConvTranspose2d(
                features_in, features_out, 3, padding=1, output_padding=1, stride=2
            )
        ]
        super().__init__(*modules)


class UNet(PytorchModel, nn.Module):
    def __init__(
        self, input_features, quantiles, n_features=32, n_levels=4, skip_connection=None
    ):

        nn.Module.__init__(self)
        PytorchModel.__init__(self, input_features, quantiles)

        # Down-sampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        features_in = input_features
        features_out = n_features
        for i in range(n_levels - 1):
            self.down_blocks.append(
                Block(features_in, features_out, skip_connection=skip_connection)
            )
            self.down_samplers.append(DownSampler())
            features_in = self.down_blocks[-1].features_out
            features_out = features_out * 2

        self.center_block = Block(
            features_in, features_out, skip_connection=skip_connection
        )

        self.up_blocks = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        features_in = self.center_block.features_out
        features_out = features_out // 2
        for i in range(n_levels - 1):
            self.up_samplers.append(UpSampler(features_in, features_out))
            features_in = features_out + self.down_blocks[(-i - 1)].features_out
            self.up_blocks.append(
                Block(features_in, features_out, skip_connection=skip_connection)
            )
            features_out = features_out // 2
            features_in = self.up_blocks[-1].features_out

        self.head = nn.Sequential(
            nn.Conv2d(features_in, features_in, 1),
            nn.ReLU(),
            nn.Conv2d(features_in, features_in, 1),
            nn.ReLU(),
            nn.Conv2d(features_in, quantiles.size, 1),
        )

    def forward(self, x):

        features = []
        for (b, s) in zip(self.down_blocks, self.down_samplers):
            x = b(x)
            features.append(x)
            x = s(x)

        x = self.center_block(x)

        for (b, u, f) in zip(self.up_blocks, self.up_samplers, features[::-1]):
            x = u(x)
            x = torch.cat([x, f], 1)
            x = b(x)

        self.features = features

        return self.head(x)
