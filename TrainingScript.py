import lstm
import time


## This scripts is used to build the model and train it used training data. The model will be saved in 'model.h5'
#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 5
    seq_len = 200

    print('> Loading data... ')

    X_train, y_train, X_test, y_test, d_X_test, d_y_test = lstm.load_data('training_data.txt', seq_len, True)

    print('> Data Loaded. Compiling...')

    model = lstm.build_model([2, 100, 50, 2])

    model.fit(
        X_train,
        y_train,
        batch_size=3000,
        nb_epoch=epochs,
        validation_split=0.1)

    print('Training duration (s) : ', time.time() - global_start_time)
    model.save('model.h5')


