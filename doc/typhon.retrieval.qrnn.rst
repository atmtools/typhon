Quantile regression neural networks (QRNNs)
===========================================

An implementation of quantile regression neural networks (QRNNs) developed
specifically for remote sensing applications providing a flexible
interface for simple training and evaluation of QRNNs.

Overview
--------

The QRNN implementation consists of two-layers:

  - A high-level interface provided by the :py:class:`~typhon.retrieval.qrnn.QRNN`
    class
  - Backend-specific implementations of different neural network architectures
    to be used as models by the high-level implementation


The QRNN class
--------------

The :py:class:`~typhon.retrieval.qrnn.QRNN` class provides the high-level
interface for QRNNs. This is all that is required to train a plain,
fully-connected QRNN. The class itself implements generic functionality related
to the evaluation of QRNNs and the post processing of results such as computing
the PSD or the posterior mean. For the rest it acts as a wrapper around its
model attribute, which encapsules all network- and DL-framework-specific code.

Backends
--------

Currently both `keras <https://keras.io/>`_ and `pytorch <https://pytorch.org/>`_
are supported as backends for neural networks. The QRNN implementation will
automatically use the one that is available on your system. If both are available
you can choose a specific backend using the :py:meth:`~typhon.retrieval.qrnn.set_backend` function.

Neural network models
---------------------

The :py:class:`typhon.retrieval.qrnn.QRNN` has designed to work with any generic
regression neural network model. This aim of this was to make the implementation
sufficiently flexible to allow special network architectures or customization of
the training process.

This gives the user the flexibility to design custom NN models in pytorch
or Keras and use them with the ``QRNN`` class. Some predefined architectures
are defined in the :py:mod:`typhon.retrieval.qrnn.models` submodule.

API documentation
-----------------

.. automodule:: typhon.retrieval.qrnn.qrnn
.. currentmodule:: typhon.retrieval.qrnn.qrnn
.. autosummary::
   :toctree: generated

   QRNN

.. automodule:: typhon.retrieval.qrnn.models.pytorch
.. currentmodule:: typhon.retrieval.qrnn.models.pytorch
.. autosummary::
   :toctree: generated

   FullyConnected
   UNet

.. automodule:: typhon.retrieval.qrnn.models.keras
.. currentmodule:: typhon.retrieval.qrnn.models.keras
.. autosummary::
   :toctree: generated

   FullyConnected

