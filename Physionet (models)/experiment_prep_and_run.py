def experiment_prep_and_run(subbb, seed):
    import os
    import numpy as np
    import random
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras.constraints import max_norm
    from tensorflow.keras.layers import (BatchNormalization, Conv2D, AveragePooling2D,Dense, Activation, Dropout, Input, Flatten,SeparableConv2D, DepthwiseConv2D, SpatialDropout2D)
    from tensorflow.keras.regularizers import l2
    from data_loader import load_data
    from experiment_running import run_experiment
    from rest_export import read_rest

    #
    def add_rest_channels():
        sub_index = 0
        for sub in subbb:
            start_i = sub_index * 45
            rest_data = read_rest(sub)  

            for i in range(45):
                global_i = start_i + i
                X_new[global_i] = np.concatenate((rest_data, X[global_i]), axis=1)

            sub_index += 1

        return X_new

    #
    class Model(object):
        def __init__(self, model):
            self.model = model
            self.equals = []
            self.accuracy = 0

        def get_model(self):
            return self.model
        
        def get_equals(self):
            return self.equals

        def get_accuracy(self):
            return self.accuracy

    #
    def EEGNet(nb_classes, Chans, Samples,
               dropoutRate, kernLength, F1,
               D, F2, norm_rate, dropoutType):
        
        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be "SpatialDropout2D" or "Dropout".')

        K.set_image_data_format('channels_last')
        input_shape = (Samples, Chans, 1)

        conv_filters      = (kernLength, 1)
        depth_filters     = (1, Chans)
        pool_size1        = (6, 1)
        pool_size2        = (12, 1)
        separable_filters = (20, 1)
        axis              = -1

        inp = Input(shape=input_shape)

        x = Conv2D(F1, conv_filters, padding='same', use_bias=False)(inp)
        x = BatchNormalization(axis=axis)(x)
        x = DepthwiseConv2D(depth_filters, use_bias=False,
                            depth_multiplier=D,
                            depthwise_constraint=max_norm(1.))(x)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('elu')(x)
        x = AveragePooling2D(pool_size1)(x)
        x = dropoutType(dropoutRate)(x)

        x = SeparableConv2D(F2, separable_filters, use_bias=False, padding='same')(x)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('elu')(x)
        x = AveragePooling2D(pool_size2)(x)
        x = dropoutType(dropoutRate)(x)

        x = Flatten(name='flatten')(x)
        x = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(x)
        out = Activation('softmax', name='softmax')(x)

        return KerasModel(inputs=inp, outputs=out)

    # ---------- seeds ----------
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ---------- subjects ----------
    full_list = [subbb]
    if isinstance(full_list, list) and len(full_list) > 0 and isinstance(full_list[0], list):
        full_list = full_list[0]
    subs_considered = [f"S{str(e).zfill(3)}" for e in full_list if len(f"S{str(e).zfill(3)}") == 4]

    # ---------- params ----------
    nr_of_epochs = 150
    nb_classes   = 2
    val_size     = 0.2
    test_size    = 0.2
    num_trials   = 15
    per_sub =True
    base_folder  = r'C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\Data\data\\'
    add_rest = True
    # ---------- data ----------
    X, y = load_data(subs_considered, base_folder, sample_rate=160,
                     samples=640, preprocessing=True, bp_low=6, bp_high=40, notch_f=50,
                     normalize=True, num_trials_per_run=num_trials)


    '''
    - As is, the code takes the first 15s of Rest and pads MI with zeros to match this length. 
    The other option is to:
        1) Downsample rest (from rest_export.py), and 
        2) Remove the following lines from this one: lines 128-131 and 139
    '''

    # #  upsampling MI, rest the same
    # target_y = 9600
    # current_y = X.shape[1]
    # pad_y = target_y - current_y
    # X = np.pad(X, ((0, 0), (0, pad_y), (0, 0)), mode='constant', constant_values=0)


    if add_rest:
        X_new = np.zeros((X.shape[0], X.shape[1], X.shape[2]*2))
        X=add_rest_channels()  


    # X=X[:,:2400,:] #  limit to the first 15s


    K.set_image_data_format('channels_last')
    samples = X.shape[1]
    chans   = X.shape[2]
    X = X.reshape(X.shape[0], samples, chans, 1)
    y = to_categorical(y, nb_classes)

    # ---------- build & wrap ----------
    keras_model = EEGNet(nb_classes, chans, samples,
                         0.7, 32, 8, 2, 16, 0.15, "Dropout")
    my_model = Model(keras_model)

    # ---------- run ----------
    acc_result = run_experiment(X, y, seed, num_trials, nr_of_epochs,
                                val_size, test_size, my_model,subbb,per_sub)
    return acc_result
