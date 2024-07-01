import utility
import model
import numpy as np
import os
import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':

    # set training mode on or off
    train = True
    load_checkpoint = True # add checkpointing!!!

    # save settings
    save_le = False
    save_model = False

    # hyperparameters
    slice_length = 911
    lr = 0.001
    nb_epochs = 1
    batch_size = 16

    # path locating
    save_model_folder = 'trained_models'

    ### loading dataset
    Y_train, X_train, Y_test, X_test = utility.load_dataset_song_split() # using default settings for now
    
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
    model = model.CRNN2D(X_train.shape, nb_classes=Y_train.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])
    model.summary()

    ### INPUT CHECKPOINTING HERE
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
                            callbacks=[checkpoint_callback]
                        )
        
    if save_model:
        model.save(os.path.join(os.getcwd(), save_model_folder, 'models',
                                str(slice_length) + '_' + datetime.today().strftime("%Y-%m-%d %H") 
                                + '.keras'))
        
    # Score test model
    score = model.evaluate(X_test, Y_test, verbose=1)
    y_score = model.predict(X_test)
    utility.plot_confusion_matrix(model=model, x_test=X_test, y_test=Y_test)

    print(score)