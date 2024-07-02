import utility
import models
import numpy as np
import os
import datetime
import json

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

def train_model(train:bool = True, load_checkpoint:bool = False,
                save_le:bool = False, save_model:bool = False, save_training_history:bool = True, save_eval_metrics:bool =True,
                slice_length:int = 911, lr:float = 0.001, nb_epochs:int = 5, batch_size:int = 32, random_seed:int = 42,
                model_name = 'untitled'):
    
    ### creating and managing directories for results
    save_model_folder = 'trained_models'
    checkpoints_dir = 'checkpoints'
    results_dir = 'results'

    # create directory tree if it doesn't exist
    os.makedirs(save_model_folder, exist_ok=True)
    os.makedirs(os.path.join(save_model_folder, checkpoints_dir), exist_ok=True)
    os.makedirs(os.path.join(save_model_folder, results_dir), exist_ok=True)
    os.makedirs(os.path.join(save_model_folder, results_dir, model_name), exist_ok=True)

    # define where to save data
    log_dir = os.path.join(save_model_folder, results_dir, model_name, 'logs/fit', 
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    ### loading dataset
    Y_train, X_train, Y_test, X_test = utility.load_dataset_song_split(random_state=random_seed) # using default settings for now
    
    # slice songs according to slice length set above
    X_train, Y_train = utility.slice_songs(X_train, Y_train, slice_length = slice_length)
    X_test, Y_test = utility.slice_songs(X_test, Y_test, slice_length = slice_length)

    print("Training set label counts:", np.unique(Y_train, return_counts=True))

    Y_test, le = utility.encode_labels(Y_test, save_le=save_le)
    Y_train, _ = utility.encode_labels(Y_train, label_encoder=le)

    # Reshape data as 2d convolutional tensor shape
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    # build the model
    model = models.CRNN2D(X_train.shape, nb_classes=Y_train.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])
    model.summary()

    # define callbacks
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq = 1)

    checkpoint_callback = ModelCheckpoint(
    filepath = os.path.join(save_model_folder, 'checkpoints', f'{str(slice_length)}.weights.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1
)

    if load_checkpoint:
        # load checkpoint if it exists
        checkpoint_path = os.path.join(save_model_folder, 'checkpoints', f'{str(slice_length)}.weights.h5')
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint located at {checkpoint_path}.")

    if train:
        print("Input Data Shape", X_train.shape)
        history = model.fit(X_train, Y_train, batch_size=batch_size,
                            shuffle=True, epochs=nb_epochs,
                            verbose=1, validation_split=0.2,
                            callbacks=[tensorboard_callback, checkpoint_callback]
                        )
        
    if save_model:
        model_save_path = os.path.join(save_model_folder, results_dir, model_name, 'model.h5')
        model.save(model_save_path)
        
    # Score test model
    score = model.evaluate(X_test, Y_test, verbose=1)
    utility.plot_confusion_matrix(model = model, x_test=X_test, y_test=Y_test, le = le)

    print(score)

    if save_training_history:
        history_dir = os.path.join(save_model_folder, results_dir, model_name, 'training_history.json')
        with open(history_dir, 'w') as f:
            json.dump(history.history, f)

    if save_eval_metrics:
        eval_dir = os.path.join(save_model_folder, results_dir, model_name, 'evaluation_metrics.txt')  
        with open(eval_dir, 'w') as f:
            f.write(f"Loss: {score[0]}\n")
            f.write(f"Accuracy: {score[1]}\n")

# if __name__ == '__main__':

#     # set training mode on or off
#     train = True
#     load_checkpoint = False # turn to false for benchmarking

#     # save settings
#     save_le = False
#     save_model = False
#     save_metrics = False

#     # hyperparameters
#     slice_length = 911
#     lr = 0.001
#     nb_epochs = 5
#     batch_size = 32
#     random_seed = 42 # used for getting consistent data feed

#     # path locating
#     save_model_folder = 'trained_models'
#     log_dir = os.path.join(save_model_folder, 'fit', f's_length_{str(slice_length)}_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#     ### loading dataset
#     Y_train, X_train, Y_test, X_test = utility.load_dataset_song_split(random_state=random_seed) # using default settings for now
    
#     # slice songs according to slice length set above
#     X_train, Y_train = utility.slice_songs(X_train, Y_train, slice_length = slice_length)
#     X_test, Y_test = utility.slice_songs(X_test, Y_test, slice_length = slice_length)

#     print("Training set label counts:", np.unique(Y_train, return_counts=True))

#     Y_test, le = utility.encode_labels(Y_test, save_le=save_le)
#     Y_train, _ = utility.encode_labels(Y_train, label_encoder=le)

#     # Reshape data as 2d convolutional tensor shape
#     X_train = X_train.reshape(X_train.shape + (1,))
#     X_test = X_test.reshape(X_test.shape + (1,))

#     # build the model
#     model = model.CRNN2D(X_train.shape, nb_classes=Y_train.shape[1])
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(learning_rate=lr),
#                   metrics=['accuracy'])
#     model.summary()

#     # define callbacks
#     tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq = 1)

#     checkpoint_callback = ModelCheckpoint(
#     filepath = os.path.join(save_model_folder, 'checkpoints', f'{str(slice_length)}.weights.h5'),
#     monitor='val_loss',
#     save_best_only=True,
#     save_weights_only=True,
#     mode='min',
#     verbose=1
# )

#     if load_checkpoint:
#         # load checkpoint if it exists
#         checkpoint_path = os.path.join(save_model_folder, 'checkpoints', f'{str(slice_length)}.weights.h5')
#         if os.path.exists(checkpoint_path):
#             model.load_weights(checkpoint_path)
#             print(f"Loaded checkpoint from {checkpoint_path}")
#         else:
#             print(f"No checkpoint located at {checkpoint_path}.")

#     if train:
#         print("Input Data Shape", X_train.shape)
#         history = model.fit(X_train, Y_train, batch_size=batch_size,
#                             shuffle=True, epochs=nb_epochs,
#                             verbose=1, validation_split=0.2,
#                             callbacks=[tensorboard_callback, checkpoint_callback]
#                         )
        
#     if save_model:
#         model.save(os.path.join(os.getcwd(), save_model_folder, 'models',
#                                 str(slice_length) + '_' + datetime.today().strftime("%Y-%m-%d %H") 
#                                 + '.keras'))
        
#     # Score test model
#     score = model.evaluate(X_test, Y_test, verbose=1)
#     y_score = model.predict(X_test)

#     print(score)