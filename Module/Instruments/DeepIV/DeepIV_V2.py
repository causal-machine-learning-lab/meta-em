import keras
import types
import random
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Dense, Dense, Dropout, Lambda
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.backend import clear_session
from keras.constraints import maxnorm   
from keras.utils import to_categorical

if K.backend() == "theano":
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    import theano.tensor as tensor
    _FLOATX = theano.config.floatX
    Lop = tensor.Lop
elif K.backend() == "tensorflow":
    def Lop(output, wrt, eval_points):
        grads = tf.gradients(output, wrt, grad_ys=eval_points)
        return grads

############################### Define MLP ###############################################
def split(start, stop):
    return Lambda(lambda x: x[:, start:stop], output_shape=(stop-start,))

def MLP_T(input, output, numZX ,numDomain, hidden_layers=[64, 64], activations='relu', dropout_rate=0., l2=0., constrain_norm=False):
    domain = split(numZX,numZX+numDomain)(input)
    state = split(0,numZX)(input)
    if isinstance(activations, str):
        activations = [activations] * len(hidden_layers)
    
    for h, a in zip(hidden_layers, activations):
        if l2 > 0.:
            w_reg = keras.regularizers.l2(l2)
        else:
            w_reg = None
        const = maxnorm(2) if constrain_norm else  None
        state = Dense(h, activation=a, kernel_regularizer=w_reg, kernel_constraint=const)(state)
        if dropout_rate > 0.:
            state = Dropout(dropout_rate)(state)
    return output([state, domain])

def MLP_Y(input, output, hidden_layers=[64, 64], activations='relu',
                     dropout_rate=0., l2=0., constrain_norm=False):
    '''
    Helper function for building a Keras feed forward network.

    input:  Keras Input object appropriate for the data. e.g. input=Input(shape=(20,))
    output: Function representing final layer for the network that maps from the last
            hidden layer to output.
            e.g. if output = Dense(10, activation='softmax') if we're doing 10 class
            classification or output = Dense(1, activation='linear') if we're doing
            regression.
    '''
    state = input
    if isinstance(activations, str):
        activations = [activations] * len(hidden_layers)
    
    for h, a in zip(hidden_layers, activations):
        if l2 > 0.:
            w_reg = keras.regularizers.l2(l2)
        else:
            w_reg = None
        const = maxnorm(2) if constrain_norm else  None
        state = Dense(h, activation=a, kernel_regularizer=w_reg, kernel_constraint=const)(state)
        if dropout_rate > 0.:
            state = Dropout(dropout_rate)(state)
    return output(state)

def domain_of_gaussian_output(x, n_components):
    state,domain= x
    mu = keras.layers.Dense(n_components, activation='linear')(state)
    log_sig = keras.layers.Dense(n_components, activation='linear')(state)
    return Concatenate(axis=1)([domain, mu, log_sig])

def mixture_of_gaussian_output(x, n_components):
    state,domain= x
    mu = keras.layers.Dense(n_components, activation='linear')(state)
    log_sig = keras.layers.Dense(n_components, activation='linear')(state)
    pi = keras.layers.Dense(n_components, activation='softmax')(state)
    return Concatenate(axis=1)([pi, mu, log_sig])

def gaussian_output(x):
    state,domain= x
    mu = keras.layers.Dense(1, activation='linear')(state)
    log_sig = keras.layers.Dense(1, activation='linear')(state)
    return Concatenate(axis=1)([mu, log_sig])

####################### Define Loss #######################
def split_mixture_of_gaussians(x, n_components):
    pi = split(0, n_components)(x)
    mu = split(n_components, 2*n_components)(x)
    log_sig = split(2*n_components, 3*n_components)(x)
    return pi, mu, log_sig

def log_norm_pdf(x, mu, log_sig):
    z = (x - mu) / (K.exp(K.clip(log_sig, -40, 40))) #TODO: get rid of this clipping
    return -(0.5)*K.log(2*np.pi) - log_sig - 0.5*((z)**2)

def mix_gaussian_loss(x, mu, log_sig, w):
    '''
    Combine the mixture of gaussian distribution and the loss into a single function
    so that we can do the log sum exp trick for numerical stability...
    '''
    if K.backend() == "tensorflow":
        x.set_shape([None, 1])
    gauss = log_norm_pdf(K.repeat_elements(x=x, rep=mu.shape[1], axis=1), mu, log_sig)
    # TODO: get rid of clipping.
    gauss = K.clip(gauss, -40, 40)
    max_gauss = K.maximum((0.), K.max(gauss))
    # log sum exp trick...
    gauss = gauss - max_gauss
    out = K.sum(w * K.exp(gauss), axis=1)
    loss = K.mean(-K.log(out) + max_gauss)
    return loss

def mixture_of_gaussian_loss(y_true, y_pred, n_components):
    pi, mu, log_sig = split_mixture_of_gaussians(y_pred, n_components)
    return mix_gaussian_loss(y_true, mu, log_sig, pi)

####################### Define Gradient ######################## 

def get_gradients(self, loss, params):
    if hasattr(self, 'grads'):
        grads = self.grads
    else:
        grads = K.gradients(loss, params)
    if hasattr(self, 'clipnorm') and self.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
    if hasattr(self, 'clipvalue') and self.clipvalue > 0:
        grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
    return grads

def replace_gradients_mse(model, opt, batch_size, n_samples = 1):
    targets = K.reshape(model.targets[0], (batch_size, n_samples * 2))
    output =  K.mean(K.reshape(model.outputs[0], (batch_size, n_samples, 2)), axis=1)
    dL_dOutput = (output[:,0] - targets[:,0]) * (2.) / batch_size
    trainable_weights = model.trainable_weights
    grads = Lop(output[:,1], wrt=trainable_weights, eval_points=dL_dOutput) 

    reg_loss = model.total_loss * 0.
    for r in model.losses:
         reg_loss += r
    reg_grads = K.gradients(reg_loss, trainable_weights)
    grads = [g+r for g,r in zip(grads, reg_grads)]
    
    opt = keras.optimizers.get(opt)
    opt.get_gradients = types.MethodType(get_gradients, opt )
    opt.grads = grads
    model.optimizer = opt
    return model

##################### Define Samples ###################
def random_laplace(shape, mu=0., b=1.):
    U = K.random_uniform(shape, -0.5, 0.5)
    return mu - b * K.sign(U) * K.log(1 - 2 * K.abs(U))

def random_normal(shape, mean=0.0, std=1.0):
    return K.random_normal(shape, mean, std)

def random_multinomial(logits, seed=None):
    if K.backend() == "theano":
        if seed is None:
            seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        return rng.multinomial(n=1, pvals=logits, ndim=None, dtype=_FLOATX)
    elif K.backend() == "tensorflow":
        return tf.one_hot(tf.squeeze(tf.multinomial(K.log(logits), num_samples=1)),
                          int(logits.shape[1]))

def random_gmm(pi, mu, sig):
    normals = random_normal(K.shape(mu), mu, sig)
    k = random_multinomial(pi)
    return K.sum(normals * k, axis=1, keepdims=True)

###################### Define Sequential  ####################
class SampledSequence(keras.utils.Sequence):
    def __init__(self, domains, features, instruments, outputs, batch_size, sampler, n_samples=1, seed=None):
        self.rng = np.random.RandomState(seed)
        if not isinstance(domains, list):
            domains = [domains.copy()]
        else:
            domains = [d.copy() for d in domains]
        self.domains = domains
        if not isinstance(features, list):
            features = [features.copy()]
        else:
            features = [f.copy() for f in features]
        self.features = features
        self.instruments = instruments.copy()
        self.outputs = outputs.copy()
        if batch_size < self.instruments.shape[0]:
            self.batch_size = batch_size
        else:
            self.batch_size = self.instruments.shape[0]
        self.sampler = sampler
        self.n_samples = n_samples
        self.current_index = 0
        self.shuffle()

    def __len__(self):
        if isinstance(self.outputs, list):
            return self.outputs[0].shape[0] // self.batch_size
        else:
            return self.outputs.shape[0] // self.batch_size

    def shuffle(self):
        idx = self.rng.permutation(np.arange(self.instruments.shape[0]))
        self.instruments = self.instruments[idx,:]
        self.outputs = self.outputs[idx,:]
        self.features = [f[idx,:] for f in self.features]
        self.domains = [d[idx,:] for d in self.domains]
    
    def __getitem__(self,idx):
        instruments = [self.instruments[idx*self.batch_size:(idx+1)*self.batch_size, :]]
        features = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.features]
        domains = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.domains]
        sampler_input = instruments + features + domains
        samples = self.sampler(sampler_input, self.n_samples)
        batch_features = [f[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0) for f in self.features] + [samples]
        batch_y = self.outputs[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0)
        if idx == (len(self) - 1):
            self.shuffle()
        return batch_features, batch_y

class OnesidedUnbaised(SampledSequence):
    def __init__(self, domains, features, instruments, outputs, treatments, batch_size, sampler, n_samples=1, seed=None):
        self.rng = np.random.RandomState(seed)
        if not isinstance(domains, list):
            domains = [domains.copy()]
        else:
            domains = [d.copy() for d in domains]
        self.domains = domains
        if not isinstance(features, list):
            features = [features.copy()]
        else:
            features = [f.copy() for f in features]
        self.features = features
        self.instruments = instruments.copy()
        self.outputs = outputs.copy()
        self.treatments = treatments.copy()
        self.batch_size = batch_size
        self.sampler = sampler
        self.n_samples = n_samples
        self.current_index = 0
        self.shuffle()

    def shuffle(self):
        idx = self.rng.permutation(np.arange(self.instruments.shape[0]))
        self.instruments = self.instruments[idx,:]
        self.outputs = self.outputs[idx,:]
        self.features = [f[idx,:] for f in self.features]
        self.treatments = self.treatments[idx,:]
        self.domains = [d[idx,:] for d in self.domains]

    def __getitem__(self, idx):
        instruments = [self.instruments[idx*self.batch_size:(idx+1)*self.batch_size, :]]
        features = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.features]
        domains = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.domains]
        observed_treatments = self.treatments[idx*self.batch_size:(idx+1)*self.batch_size, :]
        sampler_input = instruments + features + domains
        samples = self.sampler(sampler_input, self.n_samples // 2)
        samples = np.concatenate([observed_treatments, samples], axis=0)
        batch_features = [f[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0) for f in self.features] + [samples]
        batch_y = self.outputs[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0)
        if idx == (len(self) - 1):
            self.shuffle()
        return batch_features, batch_y

####################### Train Model     ########################
class Treatment(Model):
    def _get_sampler_by_string(self, loss):
        output = self.outputs[0]
        inputs = self.inputs
        if loss == "mixture_of_gaussians":
            pi, mu, log_sig = split_mixture_of_gaussians(output, self.n_components)
            samples = random_gmm(pi, mu, K.exp(log_sig))
            draw_sample = K.function(inputs + [K.learning_phase()], [samples])
            return lambda inputs, use_dropout: draw_sample(inputs + [int(use_dropout)])[0]

        else:
            raise NotImplementedError("Unrecognised loss: %s. Cannot build a generic sampler" % loss)

    def compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, n_components=None, **kwargs):
        self.n_components = n_components
        self._sampler = self._get_sampler_by_string(loss)
        loss = lambda y_true, y_pred: mixture_of_gaussian_loss(y_true,y_pred,n_components)
        
        super(Treatment, self).compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights,
                                       sample_weight_mode=sample_weight_mode, **kwargs)

    def sample(self, inputs, n_samples=1, use_dropout=False):
        if hasattr(self, "_sampler"):
            if not isinstance(inputs, list):
                inputs = [inputs]
            inputs = [i.repeat(n_samples, axis=0) for i in inputs]
            return self._sampler(inputs, use_dropout)
        else:
            raise Exception("Compile model with loss before sampling")

class Response(Model):
    def __init__(self, treatment, **kwargs):
        if isinstance(treatment, Treatment):
            self.treatment = treatment
        else:
            raise TypeError("Expected a treatment model of type Treatment. Got a model of type %s. Remember to train your treatment model first." % type(treatment))
        super(Response, self).__init__(**kwargs)

    def compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, unbiased_gradient=False,n_samples=1, batch_size=None):
        super(Response, self).compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode)
        self.unbiased_gradient = unbiased_gradient
        if unbiased_gradient:
            if loss in ["MSE", "mse", "mean_squared_error"]:
                if batch_size is None:
                    raise ValueError("Must supply a batch_size argument if using unbiased gradients. Currently batch_size is None.")
                replace_gradients_mse(self, optimizer, batch_size=batch_size, n_samples=n_samples)
            else:
                warnings.warn("Unbiased gradient only implemented for mean square error loss. It is unnecessary for logistic losses and currently not implemented for absolute error losses.")

    def fit(self, x=None, y=None, batch_size=512, epochs=1, verbose=1, callbacks=None, validation_data=None, 
            class_weight=None, initial_epoch=0, samples_per_batch=None, seed=None, observed_treatments=None):
        batch_size = np.minimum(y.shape[0], batch_size)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        if samples_per_batch is None:
            if self.unbiased_gradient:
                samples_per_batch = 20
            else:
                samples_per_batch = 10

        if observed_treatments is None:
            generator = SampledSequence(x[2], x[1], x[0], y, batch_size, self.treatment.sample, samples_per_batch)
        else:
            generator = OnesidedUnbaised(x[2], x[1], x[0], y, observed_treatments, batch_size, self.treatment.sample, samples_per_batch)
        

        steps_per_epoch = y.shape[0]  // batch_size
        super(Response, self).fit_generator(generator=generator,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=epochs, verbose=verbose,
                                            callbacks=callbacks, validation_data=validation_data,
                                            class_weight=class_weight, initial_epoch=initial_epoch)

    def fit_generator(self, **kwargs):
        raise NotImplementedError("We use override fit_generator to support sampling from the treatment model during training.")

############## run train ##############################
def run(exp, data, train_dict, log, device, resultDir, others):
    clear_session()
    tf.reset_default_graph()
    random.seed(train_dict['seed'])
    tf.compat.v1.set_random_seed(train_dict['seed'])
    np.random.seed(train_dict['seed'])

    dropout_rate = min(1000./(1000. + train_dict['num']), train_dict['dropout'])
    epochs = min(int(1000000./float(train_dict['num'])),train_dict['epochs'])

    print(f"Run {exp}/{train_dict['reps']}")

    domains = Input(shape=(train_dict['n_components'],), name="domains")
    instruments = Input(shape=(data.train.z.shape[1],), name="instruments")
    features = Input(shape=(data.train.x.shape[1],), name="features")
    treatment_input = Concatenate(axis=1)([instruments, features, domains])

    est_treat = MLP_T(treatment_input, lambda x: domain_of_gaussian_output(x, train_dict['n_components']), data.train.z.shape[1]+data.train.x.shape[1], train_dict['n_components'], 
                                                hidden_layers=train_dict['layers'],
                                                dropout_rate=dropout_rate, l2=0.0001,
                                                activations=train_dict['activation'])

    treatment_model = Treatment(inputs=[instruments, features, domains], outputs=est_treat)
    treatment_model.compile('adam', loss=train_dict['t_loss'], n_components=train_dict['n_components'])
    treatment_model.fit([data.train.z, data.train.x, to_categorical(data.train.z)], data.train.t, epochs=epochs, batch_size=train_dict['batch_size'])

    #######################################################################
    treatment = Input(shape=(data.train.t.shape[1],), name="treatment")
    response_input = Concatenate(axis=1)([features, treatment])
    est_response = MLP_Y(response_input, Dense(1),
                            activations=train_dict['activation'],
                            hidden_layers=train_dict['layers'],
                            l2=0.001,
                            dropout_rate=dropout_rate)

    response_model = Response(treatment=treatment_model, inputs=[features, treatment], outputs=est_response)
    response_model.compile('adam', loss=train_dict['y_loss'])
    response_model.fit([data.train.z, data.train.x, to_categorical(data.train.z)], data.train.y, epochs=epochs, verbose=1,batch_size=train_dict['batch_size'], samples_per_batch=train_dict['samples_per_batch'])

    def estimation(data):
        return response_model.predict([data.x, data.t-data.t]), response_model.predict([data.x, data.t])

    return estimation