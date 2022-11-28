try:
    import tensorflow as tf
except:
    pass
import random
import os
import numpy as np
from .model import AutoIV

def get_tf_var(names):
    _vars = []
    for na_i in range(len(names)):
        _vars = _vars + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=names[na_i])
    return _vars

def get_opt(lrate, NUM_ITER_PER_DECAY, lrate_decay, loss, _vars):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(lrate, global_step, NUM_ITER_PER_DECAY, lrate_decay, staircase=True)
    opt = tf.compat.v1.train.AdamOptimizer(lr)
    train_opt = opt.minimize(loss, global_step=global_step, var_list=_vars)
    return train_opt

def get_IV(data, resultDir, exp):
    autoiv_savepath = resultDir + 'autoiv/'
    load_z = np.load(autoiv_savepath+f'z_{exp}.npz')
    rep_z = load_z['rep_z']
    data.train.z = rep_z
    return rep_z

def generate_IV(data, train_dict):
    data.numpy()

    train_dict['seed'] = 2021
    train_dict['emb_dim'] = 1
    train_dict['rep_dim'] = 1
    train_dict['coefs'] = {'coef_cx2y': 1, 'coef_zc2x': 1, 'coef_lld_zx': 1,
                'coef_lld_zy': 1, 'coef_lld_cx': 1,  'coef_lld_cy': 1,
                'coef_lld_zc': 1, 'coef_bound_zx': 1, 'coef_bound_zy': 1,
                'coef_bound_cx': 1, 'coef_bound_cy': 1, 'coef_bound_zc': 1, 'coef_reg': 0.001}
    train_dict['dropout'] = 0.0
    train_dict['rep_net_layer'] = 2
    train_dict['x_net_layer'] = 2
    train_dict['emb_net_layer'] = 2
    train_dict['y_net_layer'] = 2
    train_dict['sigma'] = 0.1
    train_dict['lrate'] = 1e-3
    train_dict['opt_lld_step'] = 1
    train_dict['opt_bound_step'] = 1
    train_dict['opt_2stage_step'] = 1
    train_dict['epochs'] = 1000
    train_dict['interval'] = 10

    tf.reset_default_graph()
    random.seed(train_dict['seed'])
    tf.compat.v1.set_random_seed(train_dict['seed'])
    np.random.seed(train_dict['seed'])
    os.environ['PYTHONHASHSEED'] = str(train_dict['seed'])

    tf.compat.v1.reset_default_graph()
    dim_x, dim_v, dim_y = data.train.t.shape[1], data.train.x.shape[1], data.train.y.shape[1]
    model = AutoIV(train_dict, dim_x, dim_v, dim_y)

    """ Get trainable variables. """
    zx_vars = get_tf_var(['zx'])
    zy_vars = get_tf_var(['zy'])
    cx_vars = get_tf_var(['cx'])
    cy_vars = get_tf_var(['cy'])
    zc_vars = get_tf_var(['zc'])
    rep_vars = get_tf_var(['rep/rep_z', 'rep/rep_c'])
    x_vars = get_tf_var(['x'])
    emb_vars = get_tf_var(['emb'])
    y_vars = get_tf_var(['y'])

    vars_lld = zx_vars + zy_vars + cx_vars + cy_vars + zc_vars
    vars_bound = rep_vars
    vars_2stage = rep_vars + x_vars + emb_vars + y_vars

    """ Set optimizer. """
    train_opt_lld = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
                            lrate_decay=0.95, loss=model.loss_lld, _vars=vars_lld)

    train_opt_bound = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
                                lrate_decay=0.95, loss=model.loss_bound, _vars=vars_bound)

    train_opt_2stage = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
                                lrate_decay=0.95, loss=model.loss_2stage, _vars=vars_2stage)

    train_opts = [train_opt_lld, train_opt_bound, train_opt_2stage]
    train_steps = [train_dict['opt_lld_step'], train_dict['opt_bound_step'], train_dict['opt_2stage_step']]

    # model, train_opts, train_steps, data.train
    # Begin Train
    model.sess.run(tf.compat.v1.global_variables_initializer())

    """ Training, validation, and test dict. """
    dict_train_true = {model.v: data.train.x, model.x: data.train.t, model.y: data.train.y, model.train_flag: True}
    dict_train = {model.v: data.train.x, model.x: data.train.t, model.x_pre: data.train.t, model.y: data.train.y, model.train_flag: False}
    dict_valid = {model.v: data.valid.x, model.x: data.valid.t, model.x_pre: data.valid.t, model.y: data.valid.y, model.train_flag: False}
    dict_test = {model.v: data.test.x, model.x_pre: data.test.t, model.y: data.test.y, model.train_flag: False}

    epochs = train_dict['epochs']
    intt = train_dict['epochs'] // train_dict['interval']
    for ep_th in range(epochs):
        if (ep_th % intt == 0) or (ep_th == epochs - 1):
            loss = model.sess.run([model.loss_cx2y,
                                    model.loss_zc2x,
                                    model.lld_zx,
                                    model.lld_zy,
                                    model.lld_cx,
                                    model.lld_cy,
                                    model.lld_zc,
                                    model.bound_zx,
                                    model.bound_zy,
                                    model.bound_cx,
                                    model.bound_cy,
                                    model.bound_zc,
                                    model.loss_reg],
                                    feed_dict=dict_train)
            y_pre_train = model.sess.run(model.y_pre, feed_dict=dict_train)
            y_pre_valid = model.sess.run(model.y_pre, feed_dict=dict_valid)
            y_pre_test = model.sess.run(model.y_pre, feed_dict=dict_test)

            mse_train = np.mean(np.square(y_pre_train - data.train.v))
            mse_valid = np.mean(np.square(y_pre_valid - data.valid.v))
            mse_test = np.mean(np.square(y_pre_test - data.test.v))

            print("Epoch {}: {} - {} - {}".format(ep_th, mse_train, mse_valid, mse_test))
        for i in range(len(train_opts)):  # optimizer to train
            for j in range(train_steps[i]):  # steps of optimizer
                model.sess.run(train_opts[i], feed_dict=dict_train_true)

    def get_rep_z(data):
        dict_data = {model.v: data.x, model.x_pre: data.t, model.y: data.y, model.train_flag: False}
        data_z = model.sess.run(model.rep_z, feed_dict=dict_data)
        return data_z

    rep_z = get_rep_z(data.train)
    data.train.z = rep_z
    
    return rep_z

