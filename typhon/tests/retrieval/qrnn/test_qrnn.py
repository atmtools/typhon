from typhon.retrieval.qrnn import QRNN, set_backend
import numpy as np
import os
import tempfile

#
# Import available backends.
#

backends = []
try:
    import typhon.retrieval.qrnn.models.keras
    backends += ["keras"]
except:
    pass

try:
    import typhon.retrieval.qrnn.models.pytorch
    backends += ["pytorch"]
except:
    pass

class TestQrnn:

    def setup_method(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir, "test_data")
        x_train = np.load(os.path.join(path, "x_train.npy"))
        x_mean = np.mean(x_train, keepdims = True)
        x_sigma = np.std(x_train, keepdims = True)
        self.x_train = (x_train - x_mean) / x_sigma
        self.y_train = np.load(os.path.join(path, "y_train.npy"))

    def test_qrnn(self):
        """
        Test training of QRNNs using numpy arrays as input.
        """
        for backend in backends:
            set_backend(backend)
            qrnn = QRNN(self.x_train.shape[1],
                        np.linspace(0.05, 0.95, 10))
            qrnn.train((self.x_train, self.y_train),
                       maximum_epochs = 1)
            qrnn.predict(self.x_train)

    def save_qrnn(self):
        """
        Test saving and loading of QRNNs.
        """
        qrnn = QRNN(self.x_train.shape[1],
                    np.linspace(0.05, 0.95, 10))
        f = tempfile.NamedTemporaryFile()
        qrnn.save(f.name)
        qrnn_loaded = QRNN.load(f.name)

        x_pred = qrnn.predict(self.x_train)
        x_pred_loaded = qrnn.predict(self.x_train)

        if not type(x_pred) == np.ndarray:
            x_pred = x_pred.detach()

        assert(np.allclose(x_pred, x_pred_loaded))


__file__ = "./bla.py"
test = TestQrnn()
test.setup_method()
test.test_qrnn()
