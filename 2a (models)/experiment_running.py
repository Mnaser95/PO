# import copy
# from glob import glob
# import numpy as np
# import random
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import backend as K
# from tensorflow.keras import callbacks
# from tensorflow.keras.losses import binary_crossentropy
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import precision_score, recall_score, f1_score
# from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
# import matplotlib.pyplot as plt


# class SaveBestModel(Callback):
#     def __init__(self):
#         super().__init__()
#         self.best_weights = None
#         self.best_val_loss = np.Inf

#     def on_epoch_end(self, epoch, logs=None):
#         val_loss = logs.get("val_loss")
#         if val_loss is not None and val_loss < self.best_val_loss:
#             self.best_val_loss = val_loss
#             self.best_weights = copy.deepcopy(self.model.get_weights())

#     def on_train_end(self, logs=None):
#         if self.best_weights is not None:
#             self.model.set_weights(self.best_weights)

# def plot_losses(history):
#     plt.figure(figsize=(6,4))
#     plt.plot(history.history['loss'], label='train loss')
#     plt.plot(history.history['val_loss'], label='val loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

# def predict_accuracy(model, X_test, y_test):
#     probs = model.predict(X_test)
#     preds = probs.argmax(axis=-1)
#     equals = preds == y_test.argmax(axis=-1)
#     acc = np.mean(equals)
#     return acc, equals

# def train_test_model(model, X_train, y_train, X_val, y_val, X_test, y_test, nr_of_epochs,num_trials):
#     callbacks_list = [SaveBestModel()]
#     model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.05), metrics=['accuracy'])
#     history=model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs, validation_data=(X_val, y_val), verbose=False, callbacks=callbacks_list)
#     probs = model.predict(X_train)
#     preds = probs.argmax(axis=-1)
#     equals = preds == y_train.argmax(axis=-1)
#     tr_acc = np.mean(equals)
#     acc, equals= predict_accuracy(model, X_test, y_test)

#     return acc, history

# def run_experiment(X, y, seed, num_trials,nr_of_epochs,val_split,test_split,model,subbb,per_sub):
#     K.set_image_data_format('channels_last')

#     if per_sub:
#         all_subjects = [i for i in range (0,len(subbb))]
#         shuffled = all_subjects.copy()
#         random.seed(seed)
#         random.shuffle(shuffled)
#         train_split = int((1-val_split-test_split)*len(all_subjects))
#         train_subjects = shuffled[:train_split]
#         val_subjects = shuffled[train_split:train_split+(int(val_split*len(all_subjects)))]
#         test_subjects = shuffled[train_split+(int(val_split*len(all_subjects))):]


#         # Function to get indices for given subjects
#         def get_subject_indices(subject_list, samples_per_subject=num_trials*3):
#             return np.concatenate([np.arange(s * samples_per_subject, (s + 1) * samples_per_subject) for s in subject_list])

#         # Indices
#         train_idx = get_subject_indices(train_subjects)
#         val_idx = get_subject_indices(val_subjects)
#         test_idx = get_subject_indices(test_subjects)

#         X_test, y_test = X[test_idx], y[test_idx]

#         randomize_train_val = False
#         if randomize_train_val:
#             train_val_indices = np.concatenate([train_idx,val_idx])
#             np.random.shuffle(train_val_indices)
#             split_point = int(((1-val_split-test_split)) * len(train_val_indices))
#             train_idx = train_val_indices[:split_point]
#             val_idx = train_val_indices[split_point:]
#             X_train, y_train = X[train_idx], y[train_idx]
#             X_val, y_val = X[val_idx], y[val_idx]
#         else:
#             X_train, y_train = X[train_idx], y[train_idx]        
#             X_val, y_val = X[val_idx], y[val_idx] 

#     else:
#         X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
#         X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,test_size=val_split, random_state=seed)

#     my_model=model.get_model()

#     acc, history = train_test_model(my_model, X_train, y_train,
#                                             X_val, y_val, X_test, y_test,
#                                              nr_of_epochs,
#                                             num_trials)
#     #plot_losses(history)
#     return acc

####################################################################### LESS THAN 3 SUBJECTS
import copy
from glob import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import matplotlib.pyplot as plt


class SaveBestModel(Callback):
    def __init__(self):
        super().__init__()
        self.best_weights = None
        self.best_val_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = copy.deepcopy(self.model.get_weights())

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

def plot_losses(history):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def predict_accuracy(model, X_test, y_test):
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    equals = preds == y_test.argmax(axis=-1)
    acc = np.mean(equals)
    return acc, equals

def train_test_model(model, X_train, y_train, X_test, y_test, nr_of_epochs,num_trials):
    callbacks_list = [SaveBestModel()]
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.05), metrics=['accuracy'])
    history=model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs, verbose=False)
    probs = model.predict(X_train)
    preds = probs.argmax(axis=-1)
    equals = preds == y_train.argmax(axis=-1)
    tr_acc = np.mean(equals)
    acc, equals= predict_accuracy(model, X_test, y_test)

    return acc, history

def run_experiment(X, y, seed, num_trials,nr_of_epochs,val_split,test_split,model,subbb,per_sub):
    K.set_image_data_format('channels_last')

    if per_sub:
        all_subjects = [i for i in range (0,len(subbb))]
        shuffled = all_subjects.copy()
        random.seed(seed)
        random.shuffle(shuffled)
        train_split = int((1-val_split-test_split)*len(all_subjects))
        train_subjects = shuffled[:train_split]
        
        test_subjects = shuffled[train_split+(int(val_split*len(all_subjects))):]


        # Function to get indices for given subjects
        def get_subject_indices(subject_list, samples_per_subject=num_trials*3):
            return np.concatenate([np.arange(s * samples_per_subject, (s + 1) * samples_per_subject) for s in subject_list])

        # Indices
        train_idx = get_subject_indices(train_subjects)
        test_idx = get_subject_indices(test_subjects)

        X_test, y_test = X[test_idx], y[test_idx]

        randomize_train_val = False
        if randomize_train_val:
            train_val_indices = np.concatenate([train_idx,val_idx])
            np.random.shuffle(train_val_indices)
            split_point = int(((1-val_split-test_split)) * len(train_val_indices))
            train_idx = train_val_indices[:split_point]
            X_train, y_train = X[train_idx], y[train_idx]
        else:
            X_train, y_train = X[train_idx], y[train_idx]        

    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
        X_train, X_val, y_train, y_val = X_train_val, X_train_val, y_train_val, y_train_val
    my_model=model.get_model()

    acc, history = train_test_model(my_model, X_train, y_train,
                                             X_test, y_test,
                                             nr_of_epochs,
                                            num_trials)
    #plot_losses(history)
    return acc


