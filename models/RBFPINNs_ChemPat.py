from scipy.optimize import differential_evolution
from scipy.integrate import odeint
import pickle
import sys
from scipy.ndimage import laplace
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import pickle
import tensorflow as tf
import random as random
from keras.layers import Layer
from keras import backend as K
from keras.initializers import Constant
import time
import copy as copy
import os
import importlib.util
import cv2

# Relative import for callback for optimisation
spec = importlib.util.spec_from_file_location("clr", "../CLR/clr_callback.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)


# Define the data to feed the neural netowrk
tf.keras.backend.set_floatx("float32")


# Define RBF layer
class RBFLayer(Layer):
    def __init__(self, units, beta, max_val, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.init_beta = K.cast_to_floatx(beta)
        self.max_val = max_val

    def build(self, input_shape):
        #         print(input_shape)
        #         print(self.units)
        self.mu = self.add_weight(
            name="mu",
            shape=(int(input_shape[1]), self.units),
            initializer=tf.random_uniform_initializer(
                minval=0, maxval=self.max_val, seed=None
            ),
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.units,),
            initializer=Constant(value=self.init_beta),
            # initializer='ones',
            trainable=True,
        )
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)  # Computing the Euclidean distance
        res = K.exp(-1 * self.beta * l2)  # Computing the gaussian
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# Defining Neural Network for optimisation
class plot_generator:
    def __init__(
        self,
        PINN,
        nodes,
        n,
        dx,
        noise,
        u,
        v,
        learning_rates,
        c_original,
        length_app,
        length_total,
        batchsize,
        step_size,
        model_name,
        sigma2,
    ):
        self.PINN = PINN
        self.nodes = nodes
        self.n = n
        self.dx = dx
        self.noise = noise
        self.u = u
        self.v = v
        self.sigma2 = sigma2
        self.learning_rates = learning_rates
        self.c_original = c_original
        self.length_app = length_app
        self.length_total = length_total
        self.batchsize = batchsize
        self.step_size = step_size
        self.model_name = model_name
        # self.xrange = self.yrange = np.linspace(0, 1, n)
        self.xrange = self.yrange = np.linspace(0, dx * (n - 1), n)
        self.cmap = cm.Spectral
        noise_spread_u = noise / 100 * (np.max(self.u) - np.min(self.u))
        noise_spread_v = noise / 100 * (np.max(self.v) - np.min(self.v))
        noise_u = np.random.normal(0, noise_spread_u, (n, n))
        noise_v = np.random.normal(0, noise_spread_v, (n, n))
        self.u_n = copy.copy(self.u) + noise_u
        self.v_n = copy.copy(self.v) + noise_v

    def train(self):
        try:
            self.path_name = (
                f"paper_images/{self.model_name}_no_br/noise_level_{self.noise}"
            )
            os.makedirs(self.path_name)
        except FileExistsError:
            print("Directory already exists, possible rewriting")
            pass
        self.PINN_model = self.PINN(
            self.nodes,
            self.xrange,
            self.yrange,
            self.learning_rates[0],
            self.learning_rates[1],
            self.u_n,
            self.v_n,
            self.dx,
            1e-5,
            self.length_app,
            self.batchsize,
            self.sigma2,
        )
        self.PINN_model.train(self.length_total, self.step_size)

    def continue_train(self):
        self.PINN_model.train(self.length_total, self.step_size)

    def plotting(self):
        for i, par in enumerate(self.c_original):
            plt.close()
            plt.plot(self.PINN_model.parameters[i])
            plt.yscale("log")
            plt.axhline(y=par, linestyle=":")
            plt.xlabel("Iterations")
            plt.ylabel(f"D_{i}")
            plt.savefig(f"{self.path_name}/d{i}.png")
            plt.savefig(f"{self.path_name}/d{i}.pdf")
            plt.close()

            plt.plot(self.PINN_model.loss_u_array, label="loss_u")
            plt.plot(self.PINN_model.loss_pde_u_array, label="loss_pde_u")
            plt.plot(self.PINN_model.loss_diff_u_array, label="loss_diff_u")
            plt.plot(self.PINN_model.loss_v_array, label="loss_v")
            plt.plot(self.PINN_model.loss_pde_v_array, label="loss_pde_v")
            plt.plot(self.PINN_model.loss_diff_v_array, label="loss_diff_v")
            plt.yscale("log")
            plt.xlabel("Iterations")
            plt.ylabel("Losses")
            plt.legend()
            plt.savefig(f"{self.path_name}/loss_plot_log.png")
            plt.savefig(f"{self.path_name}/loss_plot_log.pdf")
            plt.close()

            plt.plot(self.PINN_model.loss_u_array, label="loss_u")
            plt.plot(self.PINN_model.loss_pde_u_array, label="loss_pde_u")
            plt.plot(self.PINN_model.loss_diff_u_array, label="loss_diff_u")
            plt.plot(self.PINN_model.loss_v_array, label="loss_v")
            plt.plot(self.PINN_model.loss_pde_v_array, label="loss_pde_v")
            plt.plot(self.PINN_model.loss_diff_v_array, label="loss_diff_v")
            plt.xlabel("Iterations")
            plt.ylabel("Losses")
            plt.legend()
            plt.savefig(f"{self.path_name}/loss_plot.png")
            plt.savefig(f"{self.path_name}/loss_plot.pdf")
            plt.close()

        X, Y = np.meshgrid(self.xrange, self.yrange)
        X, Y = tf.Variable(X.flatten()[:, None], dtype=tf.float32), tf.Variable(
            Y.flatten()[:, None], dtype=tf.float32
        )
        plt.imshow(
            np.reshape(self.PINN_model.model_u(tf.concat([X, Y], 1)), (self.n, self.n)),
            cmap=self.cmap,
        )
        plt.savefig(f"{self.path_name}/u_approx_pinn.png")
        plt.savefig(f"{self.path_name}/u_approx_pinn.pdf")
        plt.close()
        plt.imshow(
            np.reshape(self.PINN_model.model_v(tf.concat([X, Y], 1)), (self.n, self.n)),
            cmap=self.cmap,
        )
        plt.savefig(f"{self.path_name}/v_approx_pinn.png")
        plt.savefig(f"{self.path_name}/v_approx_pinn.pdf")
        plt.close()

        filename = f"{self.path_name}/saved_param_arrays"
        outfile = open(filename, "wb")
        pickle.dump(self.PINN_model.parameters, outfile)
        outfile.close()
        filename = f"{self.path_name}/saved_losses_arrays"
        outfile = open(filename, "wb")
        pickle.dump(
            [
                self.PINN_model.loss_u_array,
                self.PINN_model.loss_pde_u_array,
                self.PINN_model.loss_diff_u_array,
                self.PINN_model.loss_v_array,
                self.PINN_model.loss_pde_v_array,
                self.PINN_model.loss_diff_v_array,
            ],
            outfile,
        )
        outfile.close()

    def simulate_new_pattern(self, step_forward, modelfun=True):
        self.c_new = self.PINN_model.final_parameters
        # Get initial conditions
        u0 = 0.1 * np.ones(self.n**2)
        v0 = 0.1 * np.ones(self.n**2)
        perturbation1 = np.random.normal(0, 0.01, (self.n**2))
        perturbation2 = np.random.normal(0, 0.01, (self.n**2))
        y0 = np.zeros(2 * self.n**2)
        y0[::2] = u0 + perturbation1
        y0[1::2] = v0 + perturbation2
        tlen = 800000
        t = np.linspace(0, tlen, tlen)
        if modelfun:
            solb = odeint(
                step_forward,
                y0,
                t,
                args=(self.c_new, self.dx),
                ml=2 * self.n,
                mu=2 * self.n,
            )
        else:
            solb = odeint(
                step_forward,
                y0,
                t,
                args=(self.c_new, self.dx, modelfun),
                ml=2 * self.n,
                mu=2 * self.n,
            )
        self.u_tp_new = np.reshape(solb[-1][::2], (self.n, self.n))
        self.v_tp_new = np.reshape(solb[-1][1::2], (self.n, self.n))
        plt.imshow(self.u_tp_new, cmap=self.cmap)
        plt.savefig(f"{self.path_name}/u_approx.png")
        plt.savefig(f"{self.path_name}/u_approx.pdf")
        plt.close()
        plt.imshow(self.v_tp_new, cmap=self.cmap)
        plt.savefig(f"{self.path_name}/v_approx.png")
        plt.savefig(f"{self.path_name}/v_approx.pdf")
        plt.close()

    def save_arrays(self):
        MSE_u = np.mean((self.u_tp_new - self.u) ** 2)
        MSE_v = np.mean((self.v_tp_new - self.v) ** 2)
        relative_error = (self.c_new - self.c_original) / self.c_original
        objects_to_save = [
            self.c_new,
            MSE_u,
            MSE_v,
            relative_error,
            self.u_tp_new,
            self.v_tp_new,
        ]
        filename = f"{self.path_name}/saved_matrices"
        outfile = open(filename, "wb")
        pickle.dump(objects_to_save, outfile)
        outfile.close()


# Define RBFPINN


class RBF_PINNs(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        x,
        y,
        min_lr,
        max_lr,
        u,
        v,
        dx,
        tol,
        threshold_ep,
        batchsize,
        sigma2=2,
    ):
        self.lr = max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.tol = tol
        self.units = units
        self.dx = dx
        self.batchsize = batchsize
        self.n = len(x)
        self.epochs = 0
        # self.d1 = tf.Variable([1.14360], dtype=tf.float32, trainable=True)
        self.d_var = tf.Variable([60], dtype=tf.float32, trainable=True)
        self.a_var = tf.Variable([12], dtype=tf.float32, trainable=True)
        self.b_var = tf.Variable([15], dtype=tf.float32, trainable=True)
        self.u_mean_var = tf.Variable([0.9068], dtype=tf.float32, trainable=True)
        self.v_mean_var = tf.Variable([0.38], dtype=tf.float32, trainable=True)
        self.u_scale_var = tf.Variable([2.5], dtype=tf.float32, trainable=True)
        self.v_scale_var = tf.Variable([0.1], dtype=tf.float32, trainable=True)
        self.u = u.flatten()[:, None]
        self.v = v.flatten()[:, None]
        self.u_in = u[2:-2, 2:-2].flatten()[:, None]
        self.v_in = v[2:-2, 2:-2].flatten()[:, None]
        # self.laplacian_u_in_or = laplace(self.u_in, mode='nearest').flatten()[
        #     :, None]/self.dx**2
        # self.laplacian_v_in_or = laplace(self.v_in, mode='nearest').flatten()[
        #     :, None]/self.dx**2
        self.max_val = np.max(x)
        self.threshold = threshold_ep
        # self.b = tf.Variable(np.tile(b, t_length).flatten()[:, None], dtype=tf.float32)
        X, Y = np.meshgrid(x, y)
        X_in, Y_in = np.meshgrid(x[2:-2], y[2:-2])
        self.tot_len = len(X.flatten())
        self.tot_len_in = len(X_in.flatten())
        self.X, self.Y = X.flatten()[:, None], Y.flatten()[:, None]
        self.X_in, self.Y_in = X_in.flatten()[:, None], Y_in.flatten()[:, None]
        self.tot_loss = []
        self.d_array = []
        self.a_array = []
        self.b_array = []
        self.u_m_array = []
        self.v_m_array = []
        self.u_s_array = []
        self.v_s_array = []
        self.loss_array = []
        self.loss_u_array = []
        self.loss_pde_u_array = []
        self.loss_diff_u_array = []
        self.loss_v_array = []
        self.loss_pde_v_array = []
        self.loss_diff_v_array = []
        self.Data = tf.concat([self.X, self.Y], 1)
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.max_lr)
        self.model_u = tf.keras.Sequential(
            [
                RBFLayer(self.units, 1 / sigma2, self.max_val),
                tf.keras.layers.Dense(1, input_shape=(self.units,), use_bias=False),
            ]
        )
        self.model_v = tf.keras.Sequential(
            [
                RBFLayer(self.units, 1 / sigma2, self.max_val),
                tf.keras.layers.Dense(1, input_shape=(self.units,), use_bias=False),
            ]
        )
        self.mse = tf.keras.losses.MeanSquaredError()
        self.iterations = 0

    @tf.function
    def _get_losses1(self, x, y, u, v):
        with tf.GradientTape(persistent=False) as tape1:
            u_pred = self.model_u(tf.concat([x, y], 1), training=True)
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
            trainables_u = self.model_u.trainable_variables
            grads_u = tape1.gradient(loss_u, trainables_u)
            del tape1
        self.optimizer.apply_gradients(zip(grads_u, trainables_u))
        with tf.GradientTape(persistent=False) as tape2:
            v_pred = self.model_v(tf.concat([x, y], 1), training=True)
            loss_v = self.mse(v_pred, v) / tf.reduce_mean(tf.square(v))
            trainables_v = self.model_v.trainable_variables
            grads_v = tape2.gradient(loss_v, trainables_v)
            del tape2
        self.optimizer.apply_gradients(zip(grads_v, trainables_v))
        loss = loss_u + loss_v
        return loss, loss_u, loss_v

    @tf.function
    def _get_losses2(self, x, y, u, v, diff_u, diff_v):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u_pred = self.model_u(tf.concat([x, y], 1), training=True)
            v_pred = self.model_v(tf.concat([x, y], 1), training=True)
            # u_or = u_pred*self.u_scale_var+self.u_mean_var
            # v_or = v_pred*self.v_scale_var+self.v_mean_var

            u_xx = tape.gradient(tape.gradient(u_pred, x), x)
            u_yy = tape.gradient(tape.gradient(u_pred, y), y)
            v_xx = tape.gradient(tape.gradient(v_pred, x), x)
            v_yy = tape.gradient(tape.gradient(v_pred, y), y)
            loss_u = self.mse(u_pred, u)
            loss_v = self.mse(v_pred, v)
            Laplace_u = u_xx + u_yy
            # Here we disjoined a var and mean u var
            loss_pde_u = tf.reduce_mean(
                tf.square(
                    self.d_var * Laplace_u
                    + self.a_var / self.u_scale_var
                    - u_pred
                    - self.u_mean_var / self.u_scale_var
                    - self.b_var
                    * (
                        4
                        * (u_pred + self.u_mean_var / self.u_scale_var)
                        * (v_pred * self.v_scale_var + self.v_mean_var)
                    )
                    / (1 + (u_pred * self.u_scale_var + self.u_mean_var) ** 2)
                )
            )
            loss_pde_u_1 = tf.reduce_mean(
                tf.square(
                    self.d_var * Laplace_u * self.u_scale_var
                    + self.a_var
                    - self.u_mean_var
                    - u_pred * self.u_scale_var
                    - self.b_var
                    * (
                        4
                        * (u_pred * self.u_scale_var + self.u_mean_var)
                        * (v_pred * self.v_scale_var + self.v_mean_var)
                    )
                    / (1 + (u_pred * self.u_scale_var + self.u_mean_var) ** 2)
                )
            )
            loss_diff_u = tf.reduce_mean(tf.square(Laplace_u - diff_u))
            Laplace_v = v_xx + v_yy
            loss_pde_v = tf.reduce_mean(
                tf.square(
                    Laplace_v
                    + (u_pred * self.u_scale_var + self.u_mean_var) / self.v_scale_var
                    - self.b_var
                    * (u_pred * self.u_scale_var + self.u_mean_var)
                    * (v_pred + self.v_mean_var / self.v_scale_var)
                    / (1 + (u_pred * self.u_scale_var + self.u_mean_var) ** 2)
                )
            )
            loss_pde_v_1 = tf.reduce_mean(
                tf.square(
                    Laplace_v * self.v_scale_var
                    + (u_pred * self.u_scale_var + self.u_mean_var)
                    - self.b_var
                    * (u_pred * self.u_scale_var + self.u_mean_var)
                    * (v_pred * self.v_scale_var + self.v_mean_var)
                    / (1 + (u_pred * self.u_scale_var + self.u_mean_var) ** 2)
                )
            )
            loss_diff_v = tf.reduce_mean(tf.square(Laplace_v - diff_v))
            # loss_scale = 3*1/self.u_scale_var+3*1/self.v_scale_var+1 / \
            #     self.u_mean_var+1/self.v_mean_var+1/self.d_var+1/self.b_var
            loss1 = (
                0.1 * loss_u
                + 0.1 * loss_v
                + self.weight_pde_u_1 * loss_pde_u_1
                + self.weight_diff_u * loss_diff_u
                + self.weight_pde_v_1 * loss_pde_v_1
                + self.weight_diff_v * loss_diff_v
            )
            # self.weight_pde_v_1*loss_pde_v_1
            # loss = loss_u+loss_v+self.weight_pde_u*loss_pde_u + \
            # self.weight_diff_u*loss_diff_u + \
            # self.weight_pde_v*loss_pde_v+self.weight_diff_v*loss_diff_v
            trainables1 = (
                self.model_u.trainable_variables
                + self.model_v.trainable_variables
                + [self.a_var, self.d_var]
            )
            # trainables2 = [self.u_scale_var,
            #                self.v_mean_var, self.v_scale_var,self.u_mean_var]
            # Changed this part for the loss
            if self.iterations % 100:
                loss1 = (
                    0.1 * loss_u
                    + 0.1 * loss_v
                    + self.weight_pde_u * loss_pde_u
                    + self.weight_diff_u * loss_diff_u
                    + self.weight_pde_v * loss_pde_v
                    + self.weight_diff_v * loss_diff_v
                )
                trainables1 = (
                    self.model_u.trainable_variables
                    + self.model_v.trainable_variables
                    + [
                        self.a_var,
                        self.d_var,
                        self.u_scale_var,
                        self.v_mean_var,
                        self.v_scale_var,
                        self.u_mean_var,
                    ]
                )
            #     loss1 = loss_u+loss_v+15*self.weight_pde_u*loss_pde_u + self.weight_diff_u*loss_diff_u + \
            #         15*self.weight_pde_v*loss_pde_v+self.weight_diff_v * \
            #         loss_diff_v  # +self.weight_pde_u_1*loss_pde_u_1 + \
            #     # trainables1 = self.model_u.trainable_variables + self.model_v.trainable_variables + \
            #     # [ self.a_var, self.b_var, self.d_var, self.u_scale_var,
            #     #            self.v_mean_var, self.v_scale_var,self.u_mean_var]

            # [self.d1, self.d2, self.d3]
            # Add losses here
            # loss2 = self.weight_pde_v*loss_pde_v+self.weight_pde_u*loss_pde_u
            grads1 = tape.gradient(loss1, trainables1)
            # grads2 = tape.gradient(loss2, trainables2)
            del tape
            self.optimizer.apply_gradients(zip(grads1, trainables1))
            # self.optimizer.apply_gradients(zip(grads2, trainables2))
        return loss1, loss_u, loss_pde_u, loss_v, loss_pde_v, loss_diff_u, loss_diff_v

    def train(self, max_iterations=1e8, step_size=None):
        clr_cb = foo.CyclicLR(
            model_optimizer=self,
            base_lr=self.min_lr,
            max_lr=self.max_lr,
            step_size=step_size,
        )
        self.callbacks = clr_cb
        if (
            self.iterations > 1
        ):  # This means we interrupted the training and want to start again
            self.model_u.set_weights(self.weights_model_u)
            self.model_v.set_weights(self.weights_model_v)
            # could maybe add the other parameters here
        first_pass = True
        self.callbacks.on_train_begin()
        while self.iterations > -1:
            if self.iterations < self.threshold:
                len_dat = self.tot_len
                datindexes = tf.data.Dataset.range(len_dat).shuffle(len_dat)
                indexes = list(datindexes.batch(self.batchsize).as_numpy_iterator())
                for batch in indexes:
                    x = tf.Variable(self.X[batch], dtype=tf.float32)
                    y = tf.Variable(self.Y[batch], dtype=tf.float32)
                    u = tf.Variable(self.u[batch], dtype=tf.float32)
                    v = tf.Variable(self.v[batch], dtype=tf.float32)
                    loss, loss_u, loss_v = self._get_losses1(x, y, u, v)
                    # self.optimizer.lr = self.lr
                    # self.optimizer = tf.keras.optimizers.Adam(
                    #     learning_rate=self.lr)
                    # Next we save some weights so that we can stop the training and start over
                    self.weights_model_u = self.model_u.weights
                    self.weights_model_v = self.model_v.weights
                    if self.iterations % 200 == 0:
                        tf.print(
                            "It: %d, Epoch: %d, Total loss: %e, loss_u:%e, Loss_v:%e, Learning rate: %2e"
                            % (
                                self.iterations,
                                self.epochs,
                                loss,
                                loss_u,
                                loss_v,
                                self.optimizer.lr,
                            )
                        )  # , loss_pde, time_sf, self.d1, tf.exp(self.d2)))
                        sys.stdout.flush()
                    if self.iterations % 2000 == 0:
                        plt.imshow(
                            np.reshape(
                                self.model_u(tf.concat([self.X, self.Y], 1)),
                                (self.n, self.n),
                            ),
                            cmap=cm.Spectral,
                        )
                        plt.colorbar()
                        plt.show()
                        plt.imshow(
                            np.reshape(
                                self.model_v(tf.concat([self.X, self.Y], 1)),
                                (self.n, self.n),
                            ),
                            cmap=cm.Spectral,
                        )
                        plt.colorbar()
                        plt.show()
                    if self.iterations == max_iterations:
                        self.iterations = -2
                        break
                    self.iterations += 1
                    if loss_u < self.tol:
                        break
                    self.callbacks.on_batch_end(self.epochs)

            else:
                len_dat = self.tot_len_in
                datindexes = tf.data.Dataset.range(len_dat).shuffle(len_dat)
                indexes = list(datindexes.batch(self.batchsize).as_numpy_iterator())
                for batch in indexes:
                    # if self.iterations == 30000:
                    #     self.optimizer = tf.keras.optimizers.Adam(
                    #         learning_rate=self.min_lr)
                    if first_pass or self.iterations % 2000:  # Added the 2000 if
                        X1 = tf.Variable(self.X_in, dtype=tf.float32)
                        Y1 = tf.Variable(self.Y_in, dtype=tf.float32)
                        u1 = tf.Variable(self.u_in, dtype=tf.float32)
                        v1 = tf.Variable(self.v_in, dtype=tf.float32)
                        # diff_u1 = tf.Variable(
                        #     self.laplacian_u_in_or, dtype=tf.float32)
                        # diff_v1 = tf.Variable(
                        #     self.laplacian_v_in_or, dtype=tf.float32)
                        # Can make use of the code we already have to 1) get the output for u and v 2) transform to array and compute the laplacian 3) transform back to tensor (if needed)
                        # Getting the weights for the pde losses
                        with tf.GradientTape(persistent=True) as tape:
                            u_pred = self.model_u(tf.concat([X1, Y1], 1))
                            v_pred = self.model_v(tf.concat([X1, Y1], 1))
                            # u_or = self.u_scale_var*u_pred+self.u_mean_var
                            # v_or = self.v_scale_var*v_pred+self.v_mean_var
                            diff_u = (
                                laplace(
                                    np.reshape(u_pred, (self.n - 4, self.n - 4))
                                ).flatten()
                                / self.dx**2
                            )
                            diff_v = (
                                laplace(
                                    np.reshape(v_pred, (self.n - 4, self.n - 4))
                                ).flatten()
                                / self.dx**2
                            )

                            u_xx = tape.gradient(tape.gradient(u_pred, X1), X1)
                            u_yy = tape.gradient(tape.gradient(u_pred, Y1), Y1)
                            v_xx = tape.gradient(tape.gradient(v_pred, X1), X1)
                            v_yy = tape.gradient(tape.gradient(v_pred, Y1), Y1)
                            loss_u = self.mse(u_pred, u1)
                            loss_v = self.mse(v_pred, v1)
                            Laplace_u = u_xx + u_yy
                            loss_pde_u = tf.reduce_mean(
                                tf.square(
                                    self.d_var * Laplace_u
                                    + self.a_var / self.u_scale_var
                                    - self.u_mean_var / self.u_scale_var
                                    - u_pred
                                    - self.b_var
                                    * (
                                        4
                                        * (u_pred + self.u_mean_var / self.u_scale_var)
                                        * (v_pred * self.v_scale_var + self.v_mean_var)
                                    )
                                    / (
                                        1
                                        + (u_pred * self.u_scale_var + self.u_mean_var)
                                        ** 2
                                    )
                                )
                            )
                            loss_pde_u_1 = tf.reduce_mean(
                                tf.square(
                                    self.d_var * Laplace_u * self.u_scale_var
                                    + self.a_var
                                    - self.u_mean_var
                                    - u_pred * self.u_scale_var
                                    - self.b_var
                                    * (
                                        4
                                        * (u_pred * self.u_scale_var + self.u_mean_var)
                                        * (v_pred * self.v_scale_var + self.v_mean_var)
                                    )
                                    / (
                                        1
                                        + (u_pred * self.u_scale_var + self.u_mean_var)
                                        ** 2
                                    )
                                )
                            )
                            loss_diff_u = tf.reduce_mean(tf.square(Laplace_u - diff_u))
                            Laplace_v = v_xx + v_yy
                            loss_pde_v = tf.reduce_mean(
                                tf.square(
                                    Laplace_v
                                    + (u_pred * self.u_scale_var + self.u_mean_var)
                                    / self.v_scale_var
                                    - self.b_var
                                    * (u_pred * self.u_scale_var + self.u_mean_var)
                                    * (v_pred + self.v_mean_var / self.v_scale_var)
                                    / (
                                        1
                                        + (u_pred * self.u_scale_var + self.u_mean_var)
                                        ** 2
                                    )
                                )
                            )
                            loss_pde_v_1 = tf.reduce_mean(
                                tf.square(
                                    Laplace_v * self.v_scale_var
                                    + (u_pred * self.u_scale_var + self.u_mean_var)
                                    - self.b_var
                                    * (u_pred * self.u_scale_var + self.u_mean_var)
                                    * (v_pred * self.v_scale_var + self.v_mean_var)
                                    / (
                                        1
                                        + (u_pred * self.u_scale_var + self.u_mean_var)
                                        ** 2
                                    )
                                )
                            )
                            loss_diff_v = tf.reduce_mean(tf.square(Laplace_v - diff_v))

                        self.weight_pde_u = loss_u / loss_pde_u
                        self.weight_pde_v = loss_v / loss_pde_v
                        self.weight_pde_u_1 = loss_u / loss_pde_u_1
                        # self.weight_scale = (loss_u)/loss_scale
                        self.weight_pde_v_1 = loss_v / loss_pde_v_1
                        self.weight_diff_u = loss_u / loss_diff_u
                        self.weight_diff_v = loss_v / loss_diff_v
                        first_pass = False

                    x = tf.Variable(self.X_in[batch], dtype=tf.float32)
                    y = tf.Variable(self.Y_in[batch], dtype=tf.float32)
                    u = tf.Variable(self.u_in[batch], dtype=tf.float32)
                    v = tf.Variable(self.v_in[batch], dtype=tf.float32)
                    u_pred = self.model_u(tf.concat([X1, Y1], 1))
                    v_pred = self.model_v(tf.concat([X1, Y1], 1))
                    diff_u = (
                        laplace(np.reshape(u_pred, (self.n - 4, self.n - 4))).flatten()[
                            batch
                        ]
                        / self.dx**2
                    )
                    diff_v = (
                        laplace(np.reshape(v_pred, (self.n - 4, self.n - 4))).flatten()[
                            batch
                        ]
                        / self.dx**2
                    )
                    # loss, loss_fun, loss_pde, loss_diff = self.grad_desc(
                    #     x, y, u, v, diff_u, diff_v)
                    (
                        loss,
                        loss_u,
                        loss_pde_u,
                        loss_v,
                        loss_pde_v,
                        loss_diff_u,
                        loss_diff_v,
                    ) = self._get_losses2(x, y, u, v, diff_u, diff_v)
                    if (
                        self.iterations % 500
                    ):  # Calling loss 1 on all pattern to still have good accuracy on boundaries
                        # self.weight_scale = (loss_Zu)/loss_scale
                        x = tf.Variable(self.X, dtype=tf.float32)
                        y = tf.Variable(self.Y, dtype=tf.float32)
                        u = tf.Variable(self.u, dtype=tf.float32)
                        v = tf.Variable(self.v, dtype=tf.float32)
                        loss, loss_u, loss_v = self._get_losses1(x, y, u, v)
                    if self.iterations % 50:
                        self.tot_loss.append(loss.numpy())
                        self.a_array.append(self.a_var.numpy()[0])
                        self.b_array.append(self.b_var.numpy()[0])
                        self.d_array.append(self.d_var.numpy()[0])
                        self.u_m_array.append(self.u_mean_var.numpy()[0])
                        self.v_m_array.append(self.v_mean_var.numpy()[0])
                        self.u_s_array.append(self.u_scale_var.numpy()[0])
                        self.v_s_array.append(self.v_scale_var.numpy()[0])
                        self.loss_array.append(loss.numpy())
                        self.loss_u_array.append(loss_u.numpy())
                        self.loss_pde_u_array.append(loss_pde_u.numpy())
                        self.loss_diff_u_array.append(loss_diff_u.numpy())
                        self.loss_v_array.append(loss_v.numpy())
                        self.loss_pde_v_array.append(loss_pde_v.numpy())
                        self.loss_diff_v_array.append(loss_diff_v.numpy())
                    self.callbacks.on_batch_end(self.epochs)
                    if self.iterations == max_iterations:
                        self.iterations = -2
                        break
                    # self.optimizer.lr = self.lr
                    # self.optimizer = tf.keras.optimizers.Adam(
                    #     learning_rate=self.lr)
                    # Next we save some weights so that we can stop the training and start over
                    self.weights_model_u = self.model_u.weights
                    self.weights_model_v = self.model_v.weights
                    if self.iterations % 200 == 0:
                        tf.print(
                            "It: %d, Epoch: %d, Total loss: %e, Loss_u:%e,Loss_pde_u:%e, Loss_v:%e,Loss_pde_v:%e, Learning rate: %2e, a:%e, b:%e, d:%e, u_scale:%e, u_mean:%e, v_scale:%e, v_mean:%e"
                            % (
                                self.iterations,
                                self.epochs,
                                loss,
                                loss_u,
                                loss_pde_u,
                                loss_v,
                                loss_pde_v,
                                self.optimizer.lr,
                                self.a_var,
                                self.b_var,
                                self.d_var,
                                self.u_scale_var,
                                self.u_mean_var,
                                self.v_scale_var,
                                self.v_mean_var,
                            )
                        )  # , loss_pde, time_sf, self.d1, tf.exp(self.d2)))
                        sys.stdout.flush()
                        # else:
                        #     tf.print('It: %d, Epoch: %d, Total loss: %e, loss_fun:%e,loss_pde:%e, Loss_diff:%e, Learning rate: %2e, d1:%e, d2:%e, d3:%e'
                        #              % (self.iterations, self.epochs, loss, loss_fun, loss_pde, loss_diff, self.optimizer.lr, self.d1, self.d2, self.d3,))  # , loss_pde, time_sf, self.d1, tf.exp(self.d2)))
                        #     sys.stdout.flush()
                    if self.iterations % 2000 == 0:
                        plt.imshow(
                            np.reshape(
                                self.model_u(tf.concat([self.X, self.Y], 1)),
                                (self.n, self.n),
                            ),
                            cmap=cm.Spectral,
                        )
                        plt.colorbar()
                        plt.show()
                        plt.imshow(
                            np.reshape(
                                self.model_v(tf.concat([self.X, self.Y], 1)),
                                (self.n, self.n),
                            ),
                            cmap=cm.Spectral,
                        )
                        plt.colorbar()
                        plt.show()
                    self.iterations += 1
                    if loss_u < self.tol:
                        break
            self.epochs += 1
        self.parameters = [
            self.a_array,
            self.b_array,
            self.d_array,
            self.u_m_array,
            self.u_s_array,
            self.v_m_array,
            self.v_s_array,
        ]
        self.final_parameters = [
            self.a_array[-1],
            self.b_array[-1],
            self.d_array[-1],
            self.u_m_array[-1],
            self.u_s_array[-1],
            self.v_m_array[-1],
            self.v_s_array[-1],
        ]


# Define patterns


# Load data from experiments OR...
frames = []
path = "turingvideos/02ExpGrowth.mp4"  # path to video of Turing patterns
cap = cv2.VideoCapture(path)
ret = True
while ret:
    ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        frames.append(img)
video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
last_image = video[-1, ...]
plt.imshow(last_image)
last_image.shape
plt.imshow(last_image[50:150, 50:150, :])
gray_scale = cv2.cvtColor(last_image[50:150, 50:150, :], cv2.COLOR_BGR2GRAY)
plt.imshow(gray_scale)
u = gray_scale
u_tp = (u - u.min()) / (u.max() - u.min())
v_tp = np.abs(u_tp - 1)  # invert pattern
v_tp = np.copy(u_tp)  # invert pattern
plt.imshow(u_tp)
plt.colorbar()
plt.show()
plt.imshow(v_tp)
plt.colorbar()

# ... Simulate a Turing pattern from the chemical patterns model


# This is the first non-dimensionalization that we talk about in the paper
def step_forward(y, t, c, dx, modelfuns):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))
    Dnd = c[0]
    mod_pars = c[1:]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    f, g = modelfuns(u, v, mod_pars)
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = laplacianu + f
    dvdt = Dnd * laplacianv + g

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


# This is the second non-dimensionalization that we talk about in the paper, and the one that worked


def step_forward2(y, t, c, dx, modelfuns):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))
    Dnd = c[0]
    mod_pars = c[1:]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    f, g = modelfuns(u, v, mod_pars)
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = Dnd * laplacianu + f
    dvdt = laplacianv + g

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


def chem_pat(u, v, mod_pars):
    a, b = mod_pars
    f = a - u - b * (4 * u * v) / (1 + u**2)
    g = u - b * u * v / (1 + u**2)
    return f, g


n = 50
# Get initial conditions
u0 = 0.01 * np.ones(n**2)
v0 = 0.01 * np.ones(n**2)
# Either random
perturbation1 = np.random.normal(0, 0.01, (n**2))
perturbation2 = np.random.normal(0, 0.01, (n**2))
# Or same as the paper
perturbation1, perturbation2 = np.load("Perturbation_arrays.npy")
y0 = np.zeros(2 * n**2)
y0[::2] = u0 + perturbation1
y0[1::2] = v0 + perturbation2
tlen = 9000
t = np.linspace(0, tlen, tlen)
c_original = [1 / 50, 12, 15]
dx = 1 / np.sqrt(50)
solb = odeint(step_forward2, y0, t, args=(c_original, dx, chem_pat), ml=2 * n, mu=2 * n)
u_tp = np.reshape(solb[-100][::2], (n, n))
v_tp = np.reshape(solb[-100][1::2], (n, n))
np.mean((solb[-1][::2] - solb[-2][::2]) ** 2)
np.mean((solb[-1][1::2] - solb[-2][1::2]) ** 2)
# Create normalised data
u_tp_toy = (u_tp - u_tp.min()) / (u_tp.max() - u_tp.min())
# Only use u for both u and v
v_tp_toy = np.copy(u_tp_toy)


# Once one of the input is chosen and computed, train the network:

plots_gen = plot_generator(
    RBF_PINNs,
    300,
    100,
    1 / 2 * np.sqrt(1 / 50),
    0,
    u_tp_toy,
    v_tp_toy,
    [0.00001, 0.002],
    np.array([1, 2, 3, 4, 5]),
    40000,
    800000,
    128,
    2000,
    "Experimental_test_chemical_pattern_2",
    sigma2=10 / 50,
)
plots_gen.train()
print("Train Done")

# # Make plots of the different losses and parameters
# plt.plot(plots_gen.PINN_model.loss_array)
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_loss_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.d_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_d_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.b_array)
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_b_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.a_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_a_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.v_m_array)
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_vm_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.u_m_array)
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_um_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.v_s_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_vs_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.u_s_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_us_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.loss_pde_u_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_pdelossu_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.loss_pde_v_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_pdelossv_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.loss_u_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_uloss_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.loss_v_array)
# plt.yscale('log')
# plt.savefig(
#     'PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_vloss_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.loss_diff_u_array)
# plt.yscale('log')
# plt.savefig('PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_u_diff_loss_array')
# plt.show()

# plt.plot(plots_gen.PINN_model.loss_diff_v_array)
# plt.yscale('log')
# plt.savefig('PINN_fixed_chemical_pattern_6_v2_dx_batchf_partition_pde_weights_startb2_lr_002_0_1_v_diff_loss_array')
# plt.show()

# plt.imshow(u_tp)
# plt.colorbar()
# plt.savefig('Experimental_u')
# plt.show()


# plt.imshow(v_tp)
# plt.colorbar()
# plt.savefig('Experimental_v')

# # Plot for convergence in paper
# plt.figure(figsize=(10, 6))
# plt.plot(plots_gen.PINN_model.u_s_array, color='tab:orange')
# plt.plot(plots_gen.PINN_model.u_m_array, color='tab:blue')
# # Values taken from pattern
# plt.axhline(1.8705, linestyle='--', color='tab:orange')
# plt.axhline(1.1045, linestyle='--', color='tab:blue')
# plt.legend([r'$\kappa_u$', r'$\gamma_u$'], fontsize=14)
# plt.ylabel('Parameter value', fontsize=14)
# plt.xlabel('Iterations', fontsize=14)
# plt.yscale('log')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.savefig('Convergence of parameters.pdf')
# plt.show()
