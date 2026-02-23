import tensorflow as tf
import numpy as np
import os
import time
from typing import Literal, List, Tuple, Union
#import tensorflow_probability as tfp
from scipy.optimize import minimize, fmin_l_bfgs_b, fmin_bfgs, root


class PINN(tf.keras.Model):
    def __init__(self, config_file: dict, dim: int):
        '''
        Constructor of PINN

        Parameters
        ----------
        config_file_name: name of the configuration file.\n
        dim: dimensions of the physical problem. Default value is 3D. 
        '''
        super(PINN, self).__init__()

        self.logger = LOGGER(config_file)
        self.results_saver = ResultsSaver('', config_file, self.logger)

        self.config_file = config_file
        self.dim = dim

        precision = config_file['training']['precision']
        if precision == 'single':
            self.precision = tf.float32
        elif precision == 'double':
            self.precision = tf.float64
        
        # glorot_init = tf.keras.initializers.GlorotNormal(self.config_file['seed'])
        glorot_init = tf.keras.initializers.GlorotUniform(self.config_file['seed'])

        activation = self.config_file['model']['activation']
        self.hidden_layers = [
            tf.keras.layers.Dense(units=l_size, activation=activation, kernel_initializer=glorot_init, dtype=self.precision) for l_size in self.config_file['model']['hidden_layers']
        ]
        self.output_layer = tf.keras.layers.Dense(dim+1, activation=None, kernel_initializer=glorot_init, dtype=self.precision)
        
        nodes_config_file = self.config_file['nodes']
        self.n_inlet = nodes_config_file['inlet']
        self.n_outlet = self.n_inlet + nodes_config_file['outlet']
        self.n_walls = self.n_outlet + nodes_config_file['walls']
        self.n_internal = self.n_walls + nodes_config_file['collocation']

        reynolds = self.config_file['flow']['reynolds']
        kinematic_viscosity = self.config_file['flow']['kinematic_viscosity']
        if reynolds == None:
            self.provide_reynolds = False
            if kinematic_viscosity != None:
                self.provide_kinematic_viscosity = True
                self.kinematic_viscosity = kinematic_viscosity
        else:
            self.provide_reynolds = True
            self.reynolds = reynolds

        self.density = self.config_file['flow']['density']
        self.wBC = self.config_file['boundary_conditions']['weighting_factor']

        self.start = time.time()
        self.pass_4 = False
        self.pass_5 = False

    def get_model_name(self):
        print('''\n
                   _                                                                                                                                 _      
                 ('v')                                                                                                                             ('v')
                </-=-\> --------------------------------------------------------  PINN   -------------------------------------------------------- </-=-\>
                 \_=_/                                                                                                                             \_=_/
                 ^^ ^^                                                                                                                             ^^ ^^ \n''')
        return
    
    def call(self, nodes: tf.Tensor) -> tf.Tensor:
        '''Forward Pass: calculate the output of the neural network
        
        Parameters
        ----------
        nodes: coordinates of the selected nodes on the geometry. If dim=3, nodes=(x, y, z).

        Returns
        -------  
        output: outputs of the neural network computed in forward pass. If dim=3, outputs=(u, v, w, p).
        '''

        output = nodes
        for layer in self.hidden_layers:
            output = layer(output)
        output = self.output_layer(output)

        return output
    
    def early_stopping(self, epoch: int, n_epochs: int, best_epoch: int, best_loss: float, early_stopping_epochs: int, loss_tolerance: float) -> bool:
        if epoch - best_epoch > early_stopping_epochs and epoch > 0.5 * n_epochs and best_loss <= loss_tolerance:
            self.logger.log_early_stopping(epoch, best_epoch, best_loss)
            return True
        return False
        
    def compute_loss(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: Tuple, epoch: int, n_epochs: int, epochs_flowfield_saving: int, best_loss: float) -> Tuple[tf.Tensor]:
        '''
        Compute the Loss function of the PINN.
        
        Returns
        -------
        A list with all losses: loss_inlet, loss_p_outlet + loss_der_outlet, loss_wall, loss_continuity, loss_momentum (one tensor for each dimension), total_loss.\n
        '''

        dim = self.dim

        # ==== GRADIENTS ====
        # INTERNAL
        with tf.GradientTape(persistent=True) as tape_internal:
            tape_internal.watch(coords_internal)
            fields_internal = self(tf.concat(coords_internal, axis=1))
            u_internal = tf.split(fields_internal[:, :dim], dim, axis=1)
            p_internal = fields_internal[:, dim:dim+1]
            grads1_u_internal = [tape_internal.gradient(u_i, coords_internal) for u_i in u_internal]  # first order grad of u relative to coords. e.g. for 2D is a list=[[∂u/∂x=Tensor(4000,1), ∂u/∂y=Tensor(4000,1)], [∂v/∂x=Tensor(4000,1), ∂v/∂y=Tensor(4000,1)]]

        grads1_p_internal = tape_internal.gradient(p_internal, coords_internal)
        grads2_u_internal = []
        for grad_u_i in grads1_u_internal:
            grads2_u_internal.append([tape_internal.gradient(grad_u_i[i], coords_internal[i]) for i in range(dim)])
        
        del tape_internal

        # OUTLET
        with tf.GradientTape(persistent=True) as tape_outlet: 
            tape_outlet.watch(coords_outlet)   
            fields_outlet = self(tf.concat(coords_outlet, axis=1))
            u_outlet = tf.split(fields_outlet[:, :dim], dim, axis=1)
        
        grads1_u_x_outlet = [tape_outlet.gradient(u_i, coords_outlet[0]) for u_i in u_outlet]  # ∂u_i/∂x at outlet
        p_outlet = fields_outlet[:, dim]  # used only for the duct problem
        del tape_outlet 

        # INLET & WALLS
        fields_inlet_walls = self(nodes_inlet_walls)

        # ==== RESIDUALS ====
        res_inlet = fields_inlet_walls[:self.n_inlet, :dim] - boundary_conditions[0]  # used only the for duct problem
        # res_inlet = fields_inlet_walls[:self.n_inlet] - boundary_conditions[0]  # used only for the airfoil problem
        res_walls = fields_inlet_walls[self.n_inlet:, :dim]
        res_p_outlet = p_outlet - boundary_conditions[1][0]  # used only for the duct problem

        res_continuity = tf.add_n([grads1_u_internal[i][i] for i in range(dim)])
        res_momentum = []

        if self.provide_reynolds:
            for i in range(dim):
                res_momentum.append(
                    tf.add(
                        tf.add_n([
                            tf.subtract(
                                tf.multiply(u_internal[j], grads1_u_internal[i][j]),
                                tf.multiply(1.0 / self.reynolds, grads2_u_internal[i][j])
                            )
                            for j in range(self.dim)
                        ]),
                        1.0 / self.density * grads1_p_internal[i]
                    )
                )
        elif not self.provide_reynolds and self.provide_kinematic_viscosity:
            for i in range(dim):
                res_momentum.append(
                    tf.add(
                        tf.add_n([
                            tf.subtract(
                                tf.multiply(u_internal[j], grads1_u_internal[i][j]),
                                tf.multiply(self.kinematic_viscosity, grads2_u_internal[i][j])
                            )
                            for j in range(self.dim)
                        ]),
                        tf.multiply(1.0/self.density, grads1_p_internal[i])
                    )
                )

        # ==== LOSS ====
        loss_inlet = tf.reduce_sum(tf.reduce_mean(tf.square(res_inlet), axis=0))  # set axis=0 because res_inlet is a tensor
        loss_walls = tf.reduce_sum(tf.reduce_mean(tf.square(res_walls), axis=0))
        loss_p_outlet = tf.reduce_mean(tf.square(res_p_outlet))  # used only for the duct problem
        # loss_p_outlet = 0  # used only for the airfoil problem
        loss_continuity = tf.reduce_mean(tf.square(res_continuity))            
        loss_momentum = tf.reduce_mean(tf.square(res_momentum), axis=1)  # set axis=1 because loss_momentum is a list
        loss_der_outlet = tf.reduce_sum(tf.reduce_mean(tf.square(grads1_u_x_outlet), axis=1))  # fully developed flow condition at outlet

        # print(self.wBC)
        # wBC is changing inside the callback_bfgs function

        total_loss = loss_inlet + loss_p_outlet + loss_der_outlet + loss_walls + loss_continuity + tf.reduce_sum(loss_momentum)
        total_loss_augm = loss_inlet + loss_p_outlet + loss_der_outlet + self.wBC * loss_walls + loss_continuity + tf.reduce_sum(loss_momentum)

        # if (epoch % epochs_flowfield_saving == 0 or epoch == n_epochs-1) and total_loss < best_loss and epoch != 0:
        if (epoch == n_epochs-1 or epoch % epochs_flowfield_saving == 0) and total_loss < 1e-4 and total_loss < best_loss:
            self.results_saver.save_flowfield(fields_inlet_walls[:self.n_inlet], fields_outlet, fields_inlet_walls[self.n_inlet:], fields_internal)
        
        if self.pass_4 == False and total_loss < 1e-3:
            self.pass_4 = True
            flowfield = tf.concat([fields_inlet_walls[:self.n_inlet], fields_outlet, fields_inlet_walls[self.n_inlet:], fields_internal], axis=0)
            np.savetxt(f'{self.results_saver.savings_parent_path}/flowfield_4.txt', flowfield)  # save only the flowfield to save time
            self.results_saver.save_weights(epoch, self, 'model_4')

            training_4_time = time.time() - self.start
            with open(os.path.join(self.results_saver.savings_parent_path, 'training_time_4.txt'), 'w') as file:
                file.write(f'{training_4_time:.2f}')

        if self.pass_5 == False and total_loss < 1e-4:
            self.pass_5 = True
            flowfield = tf.concat([fields_inlet_walls[:self.n_inlet], fields_outlet, fields_inlet_walls[self.n_inlet:], fields_internal], axis=0)
            np.savetxt(f'{self.results_saver.savings_parent_path}/flowfield_5.txt', flowfield)  # save only the flowfield to save time
            self.results_saver.save_weights(epoch, self, 'model_5')
            
            training_5_time = time.time() - self.start
            with open(os.path.join(self.results_saver.savings_parent_path, 'training_time_5.txt'), 'w') as file:
                file.write(f'{training_5_time:.2f}')

        return (loss_inlet, loss_p_outlet + loss_der_outlet, loss_walls, loss_continuity, *(loss_momentum[i][0] for i in range(dim)), total_loss, total_loss_augm)
        
    def lr(self, method: Literal['exponential_decay', 'cosine_annealing'], epoch: int, n_epochs: int, lr_init: float, lr_min: float, red_exp: float) -> float:
        '''Compute learning rate based on the epoch'''

        n_epochs = 50000
        if method == 'exponential_decay':
            lr = lr_init * (lr_min / lr_init) ** (epoch / n_epochs * red_exp) 
            return max(lr_min, lr)
        elif method == 'cosine_annealing':
            cos_inner = np.pi * epoch / n_epochs
            lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(cos_inner))
            return lr
    
    def get_networks_grads(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: Tuple, epoch: int, n_epochs: int, epochs_flowfield_saving: int, best_loss: float):
        '''Calculate the gradients of the NN relative to its weights.'''

        with tf.GradientTape() as tape_weights:
            losses = self.compute_loss(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, epoch, n_epochs, epochs_flowfield_saving, best_loss)
        weights_gradients = tape_weights.gradient(losses[-1], self.trainable_variables)
        del tape_weights

        return weights_gradients

    def train_adam(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: List, n_epochs: int, loss_tolerance: float, epochs_flowfield_saving: int, display: int):
        '''Train the PINN with the Adam optimizer.'''

        lr_init = self.config_file['optimizer']['ADAM']['learning_rate_params']['lr_init']
        lr_min = self.config_file['optimizer']['ADAM']['learning_rate_params']['lr_min']
        red_exp = self.config_file['optimizer']['ADAM']['learning_rate_params']['red_exp']
        early_stopping_epochs = self.config_file['training']['early_stopping_epochs']

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        best_epoch = 0
        best_loss = np.inf
        batch_time = 0

        for epoch in range(n_epochs):
            self.logger.log_epoch(epoch, display)
            start_epoch = time.time()

            optimizer.learning_rate = self.lr('exponential_decay', epoch, n_epochs, lr_init, lr_min, red_exp)

            with tf.GradientTape() as tape_weights:
                losses = self.compute_loss(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, epoch, n_epochs, epochs_flowfield_saving, best_loss)
            weights_gradients = tape_weights.gradient(losses[-1], self.trainable_variables)
            del tape_weights

            optimizer.apply_gradients(grads_and_vars=zip(weights_gradients, self.trainable_variables))

            batch_time += (time.time() - start_epoch)
            if self.results_saver.save_losses(epoch, losses, batch_time, self.dim, display):
                batch_time = 0

            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_epoch = epoch

            if self.early_stopping(epoch, n_epochs, best_epoch, best_loss, early_stopping_epochs, loss_tolerance):
                self.results_saver.save_weights(epoch, self)
                break

    def flatten_weights(self, weights: List[tf.Tensor]):
        '''Flatten model's weigths to use them as the initial guess in the LBFGS optimizer.'''
        return tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)
    
    def update_weights(self, flat_weights: tf.Tensor):
        '''Update neural network's weights during and after optimization. Internally, it unflattens the flattened weights.'''

        idx = 0
        new_weights = []
        for w in self.trainable_variables:
            shape = w.shape
            size = int(tf.size(w))
            new_weights.append(tf.reshape(flat_weights[idx: idx + size], shape))
            idx += size
        for var, val in zip(self.trainable_variables, new_weights):
            var.assign(val)
    
    def objective_bfgs(self, flat_weights: np.ndarray, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: List, n_epochs: int, epochs_flowfield_saving: int, training_stats: List, losses_global: List) -> Tuple[float, np.ndarray]:
        '''The objective function to be minimized with the LBFGS optimizer. Used as the argumemnt of the "fun" parameter in the "minimize" function.
        - training_stats: epoch, best_epoch, best_loss, batch_time

        Returns
        -------
        - total loss
        - gradients (flatten)
        '''

        # self.logger.log_epoch(training_stats[0], 0) # tf lbfgs
        start_epoch = time.time()
        self.update_weights(tf.convert_to_tensor(flat_weights, dtype=self.precision))

        with tf.GradientTape() as tape:
            losses = self.compute_loss(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, training_stats[0], n_epochs, epochs_flowfield_saving, training_stats[2])
        grads = tape.gradient(losses[-1], self.trainable_variables)
        del tape

        grads_flat = self.flatten_weights(grads)
        losses_global[:] = [float(l.numpy()) for l in losses[:-1]]  # copy all calculated losses to the global list
        training_stats[3] += (time.time() - start_epoch)  # update batch_time

        return float(losses[-1].numpy()), grads_flat.numpy().astype(np.float64)

    def callback_bfgs(self, xk, training_stats: List, losses_global: List, display: Literal[0, 1, 2]):
        '''Callable function after optimization iteration.
        - training_stats: epoch, best_epoch, best_loss, batch_time
        '''

        self.logger.log_epoch(training_stats[0], display)

        if self.results_saver.save_losses(training_stats[0], losses_global, training_stats[3], self.dim, display):
            training_stats[3] = 0

        if losses_global[-1] < training_stats[2]:
            training_stats[2] = losses_global[-1]
            training_stats[1] = training_stats[0]

        training_stats[0] += 1

        # if self.wBC < 5:
        #     self.wBC = self.wBC * 1.03

    def train_bfgs(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: List, n_epochs: int, epochs_flowfield_saving: int, display: Literal[0, 1, 2]):
        '''Train the PINN with the BFGS optimizer.'''

        _ = self(tf.zeros((1, self.dim)))  # initialize the model (its weights)
        init_weights = self.flatten_weights(self.trainable_variables).numpy().astype(np.float64)
        training_stats = [0, 0, np.inf, 0]  # epoch, best_epoch, best_loss, batch_time
        losses_global = [0.0] * (5 + self.dim)  # create a list to save losses in order to be able to access them with the callback function

        args = (nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, training_stats, losses_global)  # args: Extra arguments passed to the objective function and its derivatives (fun, jac and hess functions)
        result = minimize(
            fun=self.objective_bfgs,
            x0=init_weights,
            args=args,
            method='BFGS',
            jac=True,  # jacobian: if jac is a Boolean and is True, fun is assumed to return a tuple (f, g) containing the objective function and the gradients
            callback=lambda xk: self.callback_bfgs(xk, training_stats, losses_global, display),
            tol=1e-12,
            options={
                'maxiter': n_epochs,
                'disp': False,
                'gtol': 1e-12,      
            }
        )  # returns an object
        self.update_weights(tf.convert_to_tensor(result.x, dtype=self.precision))
        self.results_saver.save_weights(training_stats[1], self)


        # Α function specifically for BFGS which is an unconstrained optimization method
        # result = fmin_bfgs(
        #     f=lambda weights: self.objective_bfgs(weights, nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, training_stats, losses_global)[0],
        #     x0=init_weights,
        #     fprime=lambda weights: self.objective_bfgs(weights, nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, savings_path, epochs_flowfield_saving, training_stats, losses_global)[1],
        #     maxiter=n_epochs,
        #     callback=lambda xk: self.callback_bfgs(xk, training_stats, losses_global, display),
        #     disp=1
        #     )  # returns a tuple
        # self.update_weights(tf.convert_to_tensor(result[0], dtype=self.precision))
        # self.results_saver.save_weights(training_stats[1], self)

    def train_lbfgs(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: List, n_epochs: int, epochs_flowfield_saving: int, display: int):
        '''Train the PINN with the L-BFGS-B optimizer.'''

        _ = self(tf.zeros((1, self.dim)))  # initialize the model (its weights)
        init_weights = self.flatten_weights(self.trainable_variables).numpy().astype(np.float64)
        training_stats = [0, 0, np.inf, 0]  # epoch, best_epoch, best_loss, batch_time
        losses_global = [0.0] * (5 + self.dim)  # create a list to save losses in order to be able to access them with the callback function

        # Α more general function for optimization, where you can specify different algorithms and constraints
        # result = minimize(
        #     fun=lambda weights: self.objective_bfgs(weights, nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, training_stats, losses_global),
        #     x0=init_weights,
        #     jac=True,  # jacobian: if jac is a Boolean and is True, fun is assumed to return a tuple (f, g) containing the objective function and the gradient
        #     # method=method,
        #     method='L-BFGS-B',
        #     callback=lambda xk: self.callback_bfgs(xk, training_stats, losses_global, display),
        #     options={'iprint' : 1, 'maxiter' : n_epochs, 'maxfun' : 50000, 'maxcor' : 10, 'maxls' : 10, 'ftol' : 1e-12, 'gtol' : 1e-12, 'disp' : True}
        #     # options={'maxiter' : n_epochs, 'disp' : True}
        #     )  # returns an object
        # self.update_weights(tf.convert_to_tensor(result.x, dtype=self.precision))
        # self.results_saver.save_weights(training_stats[1], self)

        # Α function specifically for L-BFGS-B which is an unconstrained optimization method
        result = fmin_l_bfgs_b(
            func=lambda weights: self.objective_bfgs(weights, nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, training_stats, losses_global),
            x0=init_weights,
            factr=10,   # ftol = factr * numpy.finfo(float).eps (check SEE ALSO in documentation), where eps gives the machine epsilon, which is the smallest positive number (For float64 (double precision), this value is approximately 2.220446049250313e-16)
            pgtol=1e-12,
            # m=10, 
            # maxls=50,
            maxiter=n_epochs,
            maxfun=100000,
            callback=lambda xk: self.callback_bfgs(xk, training_stats, losses_global, display),
            )  # returns a tuple
        self.update_weights(tf.convert_to_tensor(result[0], dtype=self.precision))
        self.results_saver.save_weights(training_stats[1], self)

        ## Tensorflow lbfgs
        # _ = self(tf.zeros((1, self.dim)))  # initialize the model (its weights)
        # init_weights = self.flatten_weights(self.trainable_variables)
        # training_stats = [0, 0, np.inf, 0]  # epoch, best_epoch, best_loss, batch_time
        # losses_global = [0] * (5 + self.dim)  # create a list to save losses in order to be able to access them with the callback function
        
        # result = tfp.optimizer.lbfgs_minimize(
        #     value_and_gradients_function=lambda weights: self.objective_lbfgs(weights, nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, training_stats, losses_global),
        #     initial_position=init_weights,
        #     max_iterations=50000,
        #     tolerance=1e-15,
        #     x_tolerance=1e-15,
        #     f_relative_tolerance=1e-15,
        #     f_absolute_tolerance=1e-15,
        #     )
        # print(result)

    def train_broyden(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: List, n_epochs: int, savings_path: str, epochs_flowfield_saving: int, display: int):
        '''Train the PINN with the BROYDEN optimizer.'''

        _ = self(tf.zeros((1, self.dim)))  # initialize the model (its weights)
        init_weights = self.flatten_weights(self.trainable_variables).numpy().astype(np.float64)
        training_stats = [0, 0, np.inf, 0]  # epoch, best_epoch, best_loss, batch_time
        losses_global = [0.0] * (5 + self.dim)  # create a list to save losses in order to be able to access them with the callback function
        args = (nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, savings_path, epochs_flowfield_saving, training_stats, losses_global)

        def grad_only(w_flat: np.ndarray) -> np.ndarray:
            loss_val, grad_flat = self.objective_bfgs(w_flat, *args)
            return grad_flat

        def broyden_callback(xk: np.ndarray, fxk: np.ndarray):
            loss_val, _ = self.objective_bfgs(xk, *args)

            epoch = training_stats[0]
            self.logger.log_epoch(epoch, display)

            if self.results_saver.save_losses(epoch,
                                              losses_global,
                                              training_stats[3],
                                              self.dim,
                                              display):
                training_stats[3] = 0

            if loss_val < training_stats[2]:
                training_stats[2] = loss_val
                training_stats[1] = epoch

            training_stats[0] += 1

        result = root(
            fun=grad_only,
            x0=init_weights,
            method='broyden1',
            jac=False,
            callback=broyden_callback,
            options={
                'disp':    True,
                'maxiter': n_epochs,
                'ftol':    1e-12,
                'fatol':   1e-12,
            }
        )
        self.update_weights(tf.convert_to_tensor(result.x, dtype=self.precision))
        self.results_saver.save_weights(training_stats[1], self)

    def compute_loss_SLSQP(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: tuple, epoch: int, n_epochs: int, savings_path: str, epochs_flowfield_saving: int, best_loss: float) -> Tuple[tf.Tensor]:
        '''
        Compute the Loss Function of the PINN, specifically tailored for the SLSQP optimizer.

        Returns
        -------
        A list with all losses: loss_inlet, loss_p_outlet + loss_der_outlet, loss_wall, loss_continuity, loss_momentum (one tensor for each dimension), total_loss.\n
        '''

        dim = self.dim

        # ==== GRADIENTS ====
        # INTERNAL
        with tf.GradientTape(persistent=True) as tape_internal:
            tape_internal.watch(coords_internal)
            fields_internal = self(tf.concat(coords_internal, axis=1))
            u_internal = tf.split(fields_internal[:, :dim], dim, axis=1)
            p_internal = fields_internal[:, dim:dim+1]
            grads1_u_internal = [tape_internal.gradient(u_i, coords_internal) for u_i in u_internal]  # first order grad of u relative to coords. e.g. for 2D is a list=[[∂u/∂x=Tensor(4000,1), ∂u/∂y=Tensor(4000,1)], [∂v/∂x=Tensor(4000,1), ∂v/∂y=Tensor(4000,1)]]

        grads1_p_internal = tape_internal.gradient(p_internal, coords_internal)
        grads2_u_internal = []
        for grad_u_i in grads1_u_internal:
            grads2_u_internal.append([tape_internal.gradient(grad_u_i[i], coords_internal[i]) for i in range(dim)])
        
        del tape_internal

        # OUTLET
        with tf.GradientTape(persistent=True) as tape_outlet: 
            tape_outlet.watch(coords_outlet)   
            fields_outlet = self(tf.concat(coords_outlet, axis=1))
            u_outlet = tf.split(fields_outlet[:, :dim], dim, axis=1)
        
        grads1_u_x_outlet = [tape_outlet.gradient(u_i, coords_outlet[0]) for u_i in u_outlet]  # ∂u_i/∂x at outlet
        p_outlet = fields_outlet[:, dim]
        del tape_outlet 

        # INLET & WALLS
        fields_inlet_walls = self(nodes_inlet_walls)

        # ==== RESIDUALS ====
        res_inlet = fields_inlet_walls[:self.n_inlet, :dim] 
        res_walls = fields_inlet_walls[self.n_inlet:, :dim]
        res_p_outlet = p_outlet 
        res_continuity = tf.add_n([grads1_u_internal[i][i] for i in range(dim)])
        
        res_momentum = []
        for i in range(dim):
            res_momentum.append(
                tf.add(
                    tf.add_n([
                        tf.subtract(
                            tf.multiply(u_internal[j], grads1_u_internal[i][j]),
                            tf.multiply(1.0 / self.reynolds, grads2_u_internal[i][j])
                        )
                        for j in range(self.dim)
                    ]),
                    grads1_p_internal[i]
                )
            )

        # ==== LOSS ====
        loss_inlet = tf.reduce_sum(self.wBC * tf.reduce_mean(tf.square(res_inlet - boundary_conditions[0]), axis=0))  # set axis=0 because res_inlet is a tensor
        loss_walls = tf.reduce_sum(self.wBC * tf.reduce_mean(tf.square(res_walls), axis=0))
        loss_p_outlet = self.wBC * tf.reduce_mean(tf.square(res_p_outlet - boundary_conditions[1][0])) 
        loss_continuity = tf.reduce_mean(tf.square(res_continuity))            
        loss_momentum = tf.reduce_mean(tf.square(res_momentum), axis=1)  # set axis=1 because loss_momentum is a list
        loss_der_outlet = tf.reduce_sum(tf.reduce_mean(tf.square(grads1_u_x_outlet), axis=1))  
        total_loss = loss_inlet + loss_p_outlet + loss_der_outlet + loss_walls + loss_continuity + tf.reduce_sum(loss_momentum)

        if (epoch % epochs_flowfield_saving == 0 or epoch == n_epochs-1) and total_loss < best_loss:
            coords = tf.concat([nodes_inlet_walls[:self.n_inlet], tf.concat(coords_outlet, axis=1), nodes_inlet_walls[self.n_inlet:], tf.concat(coords_internal, axis=1)], axis=0)
            flowfields = tf.concat([fields_inlet_walls[:self.n_inlet], fields_outlet, fields_inlet_walls[self.n_inlet:], fields_internal], axis=0)
            arr = np.hstack((coords.numpy(), flowfields.numpy()))
            np.savetxt(f'{savings_path}/flowfield.txt', arr)

        return (loss_inlet, loss_p_outlet + loss_der_outlet, loss_walls, loss_continuity, *(loss_momentum[i][0] for i in range(dim)), total_loss), (res_inlet, res_walls, res_p_outlet, grads1_u_x_outlet)
    
    def objective_SLSQP(self, flat_weights: np.ndarray, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: list, n_epochs: int, savings_path: str, epochs_flowfield_saving: int, residuals_global: List[tf.Tensor], training_stats: list, losses_global: list) -> Tuple[float, np.ndarray]:
        '''The objective function to be minimized with the SLSQP optimizer. Used as the argumemnt of the "fun" parameter in the "minimize" function.

        Parameters
        ----------
        - residuals_global: residuals of inlet, outlet and walls. They are used for constraints.
        - training_stats: epoch, best_epoch, best_loss, batch_time

        Returns
        -------
        - total loss
        - gradients (flatten)
        '''

        flat_weights_tf = tf.convert_to_tensor(flat_weights, dtype=self.precision)
        self.update_weights(flat_weights_tf)

        start_epoch = time.time()
        with tf.GradientTape() as tape:
            losses, residuals = self.compute_loss_SLSQP(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, training_stats[0], n_epochs, savings_path, epochs_flowfield_saving, training_stats[2])
        grads = tape.gradient(losses[-1], self.trainable_variables)
        del tape
        training_stats[3] += (time.time() - start_epoch)  # update batch_time

        grads_flat = self.flatten_weights(grads)
        losses_global[:] = losses  # copy all calulated losses to the global
        residuals_global[:] = residuals
        residuals_global[3][0] = tf.squeeze(residuals_global[3][0], axis=1)
        residuals_global[3][1] = tf.squeeze(residuals_global[3][1], axis=1)

        return float(losses[-1].numpy()), grads_flat.numpy().astype(np.float64, copy=False)

    def train_SLSQP(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: list, n_epochs: int, savings_path: str, epochs_flowfield_saving: int, display: int):
        '''Train the PINN with SLSQP optimizer.'''

        _ = self(tf.zeros((1, self.dim)))  # initialize the model (its weights)
        init_weights = self.flatten_weights(self.trainable_variables).numpy().astype(np.float64, copy=False)
        training_stats = [0, 0, np.inf, 0]  # epoch, best_epoch, best_loss, batch_time
        losses_global = [0] * (5 + self.dim)  # create a list to save losses in order to be able to access them with the callback function
        residuals_global = [                          # create a list to save residuals in order to be able to access them with the constraints functions
            tf.zeros((750, 2)),                       # res_inlet
            tf.zeros((17000, 2)),                     # res_walls
            tf.zeros((750,)),                         # res_p_outlet
            [tf.zeros((750,)), tf.zeros((750,))]  # ∂u/∂x, ∂u/∂y outlet
            ]  

        def constraint_ux_inlet(residuals_global: list):
            return residuals_global[0][:, 0] - boundary_conditions[0][0]
        
        def constraint_uy_inlet(residuals_global: list):
            # print('\nconstraint_uy_inlet')
            # print(residuals_global[0][:, 1])
            return residuals_global[0][:, 1] - boundary_conditions[0][1]
        
        def constraint_walls_x(residuals_global: list):
            return residuals_global[1][:, 0] - boundary_conditions[2][0]
        
        def constraint_walls_y(residuals_global: list):
            # print('\nconstraint_walls_y')
            # print(residuals_global[1][:, 1])
            return residuals_global[1][:, 1] - boundary_conditions[2][1]
        
        def constraint_p_outlet(residuals_global: list):
            # print('\nconstraint_p_outlet')
            # print(residuals_global[2])
            return residuals_global[2] - boundary_conditions[1][0]
        
        def constraint_walls_der_x(residuals_global: list):
            return residuals_global[3][0] - boundary_conditions[1][1]
        
        def constraint_walls_der_y(residuals_global: list):
            return residuals_global[3][1] - boundary_conditions[1][2]
        
        constraints = [{'type': 'eq', 'fun': lambda weights: constraint_ux_inlet(residuals_global)}, 
                       {'type': 'eq', 'fun': lambda weights: constraint_uy_inlet(residuals_global)}, 
                       {'type': 'eq', 'fun': lambda weights: constraint_walls_x(residuals_global)},
                       {'type': 'eq', 'fun': lambda weights: constraint_walls_y(residuals_global)},
                       {'type': 'eq', 'fun': lambda weights: constraint_p_outlet(residuals_global)},
                       {'type': 'eq', 'fun': lambda weights: constraint_walls_der_x(residuals_global)},
                       {'type': 'eq', 'fun': lambda weights: constraint_walls_der_y(residuals_global)}
                       ]

        result = minimize(
            fun=lambda weights: self.objective_SLSQP(weights, nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, savings_path, epochs_flowfield_saving, residuals_global, training_stats, losses_global),
            x0=init_weights,
            jac=True,  # jacobian: tells the optimizer that the fun will provide the gradient (Jacobian) of the objective function. Otherwise, it will use a finite difference scheme for numerical estimation of the gradient.
            method='SLSQP',
            callback=lambda xk: self.callback_bfgs(xk, training_stats, losses_global, display),
            constraints=constraints,
            tol=1e-12,
            options={
                'disp' : True,
                'maxiter' : n_epochs,
                'ftol' : 1e-12,
            }
            )  # returns an object
        
        self.update_weights(tf.convert_to_tensor(result.x, dtype=self.precision))
        self.results_saver.save_weights(training_stats[1], self)

    def objective_SLSQP_V2(self, flat_weights: np.ndarray, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: list, n_epochs: int, savings_path: str, epochs_flowfield_saving: int, training_stats: list, losses_global: list) -> Tuple[float, np.ndarray]:
        '''The objective function to be minimized with the SLSQP optimizer. Used as the argumemnt of the "fun" parameter in the "minimize" function.

        Parameters
        ----------
        - residuals_global: residuals of inlet, outlet and walls. They are used for constraints.
        - training_stats: epoch, best_epoch, best_loss, batch_time

        Returns
        -------
        - total loss
        - gradients (flatten)
        '''
        
        flat_weights_tf = tf.convert_to_tensor(flat_weights)
        self.update_weights(flat_weights_tf)

        start_epoch = time.time()
        with tf.GradientTape() as tape:
            losses = self.compute_loss(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, training_stats[0], n_epochs, epochs_flowfield_saving, training_stats[2])
        grads = tape.gradient(losses[-1], self.trainable_variables)
        del tape
        training_stats[3] += (time.time() - start_epoch)  # update batch_time

        grads_flat = self.flatten_weights(grads)
        losses_global[:] = losses  # copy all calulated losses to the global

        # return losses[-1].numpy(), grads_flat.numpy()
        print(f'### {training_stats[0]} ###')
        return float(losses[-1].numpy()), grads_flat.numpy().astype(np.float64, copy=False)
    
    def train_SLSQP_V2(self, nodes_inlet_walls: tf.Tensor, coords_outlet: List[tf.Tensor], coords_internal: List[tf.Tensor], boundary_conditions: list, n_epochs: int, savings_path: str, epochs_flowfield_saving: int, display: int):
        '''Train the PINN with SLSQP optimizer.'''

        _ = self(tf.zeros((1, self.dim)))  # initialize the model (its weights)
        init_weights = self.flatten_weights(self.trainable_variables).numpy().astype(np.float64, copy=False)
        training_stats = [0, 0, np.inf, 0]  # epoch, best_epoch, best_loss, batch_time
        losses_global = [0] * (5 + self.dim)  # create a list to save losses in order to be able to access them with the callback function: loss_inlet, loss_p_outlet + loss_der_outlet, loss_wall, loss_continuity, loss_momentum (one tensor for each dimension), total_loss

        def constraint_inlet(losses_global: list):
            '''Constraint for loss at inlet.'''
            return losses_global[0] - 1.0e-3
        
        def constraint_outlet(losses_global: list):
            '''Constraint for loss at outlet.'''
            return losses_global[1] - 1.0e-3
        
        def constraint_walls(losses_global: list):
            '''Constraint for loss at walls.'''
            return losses_global[2] - 1.0e-3
        
        def constraint_continuity(losses_global: list):
            '''Constraint for continuity loss.'''
            return losses_global[3] - 1.0e-3
        
        def constraint_momentum_x(losses_global: list):
            '''Constraint for momentum x loss.'''
            return losses_global[4] - 1.0e-3
        
        def constraint_momentum_y(losses_global: list):
            '''Constraint for momentum x loss.'''
            return losses_global[5] - 1.0e-3
        
        constraints = [
            {'type': 'ineq', 'fun': lambda weights: constraint_inlet(losses_global)}, 
            {'type': 'ineq', 'fun': lambda weights: constraint_outlet(losses_global)}, 
            {'type': 'ineq', 'fun': lambda weights: constraint_walls(losses_global)},
            {'type': 'ineq', 'fun': lambda weights: constraint_continuity(losses_global)},
            {'type': 'ineq', 'fun': lambda weights: constraint_momentum_x(losses_global)},
            {'type': 'ineq', 'fun': lambda weights: constraint_momentum_y(losses_global)},
            ]

        result = minimize(
            fun=lambda weights: self.objective_SLSQP_V2(weights, nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, savings_path, epochs_flowfield_saving, training_stats, losses_global),
            x0=init_weights,
            jac=True,  # jacobian: tells the optimizer that the fun will provide the gradient (Jacobian) of the objective function. Otherwise, it will use a finite difference scheme for numerical estimation of the gradient.
            method='SLSQP',
            callback=lambda xk: self.callback_bfgs(xk, training_stats, losses_global, display),
            constraints=constraints,
            tol=1e-12,
            options={
                'disp' : True,
                'maxiter' : n_epochs,
                'ftol' : 1e-12,
            }
            )  # returns an object
        
        self.update_weights(tf.convert_to_tensor(result.x, dtype=self.precision))
        self.results_saver.save_weights(training_stats[1], self)

    def train(self, nodes: np.ndarray, boundary_conditions: list, savings_parent_path: str, optimization_method: Literal['ADAM', 'BFGS', 'L-BFGS-B', 'BROYDEN', 'SLSQP', 'HYBRID-1', 'HYBRID-2'], display: Literal[0, 1, 2] = 2):
        '''Train the PINN.
        
        Parameters
        ----------
        - nodes: coordinates of the selected nodes on the geometry. If dim=3, nodes=(x, y, z).
        - boundary_conditions: boundary conditions at inlet, outlet and walls.\n
        - savings_parent_path: path to save results.\n
        - optimization_method: supports ADAM, BFGS and HYBRID methods which use Adam for some iterations and Bfgs for rest training.
        - display: printing the losses on terminal during the training.\n
                0: print nothing,
                1: print only epochs,
                2: print epoch and losses,
        '''

        self.get_model_name()
        self.results_saver.savings_parent_path = savings_parent_path
        n_epochs = self.config_file['training']['epochs']
        loss_tolerance = self.config_file['training']['loss_tolerance']
        epochs_flowfield_saving = self.config_file['training']['epoch_savings']['flowfield']

        nodes_inlet = tf.convert_to_tensor(nodes[:self.n_inlet], dtype=self.precision)
        coords_outlet = tf.split(tf.convert_to_tensor(nodes[self.n_inlet:self.n_outlet], dtype=self.precision), self.dim, axis=1)
        nodes_walls = tf.convert_to_tensor(nodes[self.n_outlet:self.n_walls], dtype=self.precision)
        coords_internal = tf.split(tf.convert_to_tensor(nodes[self.n_walls:self.n_internal], dtype=self.precision), self.dim, axis=1)
        nodes_inlet_walls = tf.concat([nodes_inlet, nodes_walls], axis=0)

        start_time = time.time()

        if optimization_method == 'ADAM':
            self.train_adam(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, loss_tolerance, epochs_flowfield_saving, display)
        elif optimization_method == 'L-BFGS-B':
            self.train_lbfgs(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, display)
        elif optimization_method == 'BFGS':
            self.train_bfgs(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, display)
        elif optimization_method == 'BROYDEN':
            self.train_broyden(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, epochs_flowfield_saving, display)
        elif optimization_method == 'SLSQP':
            # self.train_SLSQP(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, savings_parent_path, epochs_flowfield_saving, display)
            self.train_SLSQP_V2(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs, savings_parent_path, epochs_flowfield_saving, display)
        elif optimization_method == 'HYBRID-1':
            n_epochs_adam = self.config_file['optimizer']['HYBRID']['epochs_adam']
            n_epochs_lbfgs = self.config_file['optimizer']['HYBRID']['epochs_lbfgs']

            print('\n### First phase of training: Adam optimizer ### \n')
            self.train_adam(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs_adam, loss_tolerance, epochs_flowfield_saving, display)
            print(f'\n### Second phase of training: {optimization_method} optimizer ###\n')
            self.train_lbfgs(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs_lbfgs, epochs_flowfield_saving, display)
        elif optimization_method == 'HYBRID-2':
            n_epochs_adam = self.config_file['optimizer']['HYBRID']['epochs_adam']
            n_epochs_bfgs = self.config_file['optimizer']['HYBRID']['epochs_lbfgs']

            print('\n### First phase of training: Adam optimizer ### \n')
            self.train_adam(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs_adam, loss_tolerance, epochs_flowfield_saving, display)
            print(f'\n### Second phase of training: {optimization_method} optimizer ###\n')
            self.train_bfgs(nodes_inlet_walls, coords_outlet, coords_internal, boundary_conditions, n_epochs_bfgs, epochs_flowfield_saving, display)

        training_time = time.time() - start_time
        self.results_saver.save_training_time(training_time)

class LOGGER:
    '''Handles all messages before, during, and after training.'''

    def __init__(self, config_file: dict):
        self.show_epochs = config_file['training']['show_epochs']
        self.save_losses_epochs = config_file['training']['epoch_savings']['losses']

    def log_loss_names(self, dim: int) -> str:
        loss_names_text = f"{'Epoch':<10}{'Inlet':<20}{'Outlet':<20}{'Walls':<20}{'Continuity':<20}"

        for i in range(dim):
            loss_names_text += f"{f'Momentum {i+1}':<20}"

        loss_names_text += f"{'Total Loss':<20}{'Batch Time (sec)':<20}"

        print(loss_names_text)
        return loss_names_text
    
    def log_epoch(self, epoch: int, display: Literal[0, 1, 2]):
        '''Prints only the epoch (used only when display is 1).'''
        if display == 1:
            if epoch % self.show_epochs == 0 and epoch % self.save_losses_epochs != 0:
                print(epoch)

    def log_early_stopping(self, epoch: int, best_epoch: int, best_loss: float):
        print(f"Training stopped at epoch {epoch}.", flush=True)
        print(f"Best epoch: {best_epoch} with loss: {best_loss}")

    def log_save_weights(self, epoch: int):
        print(f"Saved weights {epoch}.", flush=True)

    def log_losses(self, epoch: int, losses: list, batch_time: float, display:  Literal[0, 1, 2]) -> str:
        text = f"{epoch:<10}" + "".join(f"{val:<20.7e}" for val in losses) + f"{batch_time:.2f}"
        if display == 2:
            print(text, flush=True)  
        return text
    
    def log_training_time(self, training_time: float):
        print(f"\nTraining duration: {training_time:.2f} sec.", flush=True)

class ResultsSaver:
    '''Handles file savings.'''

    def __init__(self, savings_parent_path: str, config_file: dict, logger: LOGGER):
        self.logger = logger
        self.savings_parent_path = savings_parent_path

        self.epochs_save_weights = config_file['training']['epoch_savings']['weights']
        self.epochs_save_losses = config_file['training']['epoch_savings']['losses']
        self.epochs_save_flowfield = config_file['training']['epoch_savings']['flowfield']
        self.n_epochs = config_file['training']['epochs']
        
        self.path_hist_loss = config_file['paths']['histLoss_txt']
        self.path_training_time = config_file['paths']['training_time_txt']
        self.path_model = config_file['paths']['model_foldername']
        self.path_flowfield = config_file['paths']['flowfield_txt']

    def save_weights(self, epoch: int, pinn: PINN, folder_name: str = None):
        weights = pinn.get_weights()

        if folder_name == None:
            folder_name = self.path_model
        folder_path = os.path.join(self.savings_parent_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for i, weight in enumerate(weights):
            np.save(os.path.join(self.savings_parent_path, folder_name, f'layer_{i}_weights.npy'), weight)
        
        self.logger.log_save_weights(epoch)
    
    def save_losses(self, epoch: int, losses: list, batch_time: float, dim: int, display: int) -> bool:
        '''Prints losses at the first epoch and at every x (a defined number) epochs.'''
        
        if epoch == 0:
            losses_names = self.logger.log_loss_names(dim)
            with open(os.path.join(self.savings_parent_path, self.path_hist_loss), 'a') as file:
                file.write(losses_names + "\n")

        if epoch % self.epochs_save_losses == 0:
            text = self.logger.log_losses(epoch, losses, batch_time, display)
            with open(os.path.join(self.savings_parent_path, self.path_hist_loss), 'a') as file:
                file.write(text + "\n")
            return True
        return False 

    def save_flowfield(self, field_inlet, field_outlet, field_walls, field_internal):
        '''Save the flow field (velocities and pressures) as a txt.'''
        # coords = tf.concat([nodes_inlet, tf.concat(coords_outlet, axis=1), nodes_walls, tf.concat(coords_internal, axis=1)], axis=0)
        # flowfield = tf.concat([field_inlet, field_outlet, field_walls, field_internal], axis=0)
        # arr = np.hstack((coords.numpy(), flowfields.numpy()))
        # np.savetxt(f'{self.savings_parent_path}/flowfield.txt', arr)
        flowfield = tf.concat([field_inlet, field_outlet, field_walls, field_internal], axis=0)
        y_pred_normalized(f'{self.savings_parent_path}/{self.path_flowfield}', flowfield)  # save only the flowfield to save time
    
    def save_training_time(self, training_time: float):
        '''Save training time in seconds in a txt file.'''
        self.logger.log_training_time(training_time)
        with open(os.path.join(self.savings_parent_path, self.path_training_time), 'w') as file:
          file.write(f'{training_time:.2f}')