import gpflow
import tensorflow as tf
import numpy as np
from gpflow.mean_functions import Constant
from sklearn.preprocessing import StandardScaler
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise

# A minor refactoring of code from
# https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb

class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product / denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class TanimotoGP:
    def __init__(self, maxiter=100):
        self.m = None
        self.maxiter = maxiter
        self.y_scaler = StandardScaler()

    def objective_closure(self):
        return -self.m.log_marginal_likelihood()

    def fit(self, X_train, y_train):
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
        k = Tanimoto()
        self.m = gpflow.models.GPR(data=(X_train.astype(np.float64), y_train_scaled),
                                   mean_function=Constant(np.mean(y_train_scaled)),
                                   kernel=k,
                                   noise_variance=1)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.objective_closure, self.m.trainable_variables, options=dict(maxiter=self.maxiter))

    def predict(self, X_test):
        y_pred, y_var = self.m.predict_f(X_test.astype(np.float64))
        y_pred = self.y_scaler.inverse_transform(y_pred)
        return y_pred.flatten(), y_var.numpy().flatten()
