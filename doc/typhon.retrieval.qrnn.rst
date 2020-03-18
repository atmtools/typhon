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

The :py:class:`~typhon.retrieval.qrnn.QRNN` class provides the high-level interface for QRNNs. If you want
to train a fully-connected QRNN, this is all you need. The class itself
implments generic functionality related to the evaluation of QRNNs and the post
processing of results such as computing the PSD or the posterior mean. For the
rest it acts as a wrapper around its model attribute, which encapsules all
network- and DL-framework-specific code.

API documentation
^^^^^^^^^^^^^^^^^

.. automodule:: typhon.retrieval.qrnn.qrnn
.. currentmodule:: typhon.retrieval.qrnn.qrnn
.. autosummary::
   :toctree: generated

   QRNN

Backends
--------

Currently both `keras <https://keras.io/>`_ and `pytorch <https://pytorch.org/>`_
are supported as backends for neural network. The QRNN implementation will
automatically use the one the is available on your system. If both are available
you can choose a specific backend using the :py:meth:`~typhon.retrieval.qrnn.set_backend` function.


API documentation
^^^^^^^^^^^^^^^^^

.. automodule:: typhon.retrieval.qrnn.qrnn
.. currentmodule:: typhon.retrieval.qrnn.qrnn
.. autosummary::
   :toctree: generated

   set_backend

Neural network models
---------------------


Pytorch
^^^^^^^

.. automodule:: typhon.retrieval.qrnn.models.pytorch
.. currentmodule:: typhon.retrieval.qrnn.models.pytorch
.. autosummary::
   :toctree: generated

   FullyConnected

