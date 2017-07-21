from keras.models import load_model
import lstm
import numpy as np
import matplotlib.pyplot as plt
import time
t0=time.time()
print ('Loading the model...')

## This script is for running tests and predicting next sequences for each case
## It loads an already trained model and and use it to predict the next positions

model = load_model('model.h5')

seq_len = 80

def run_test(filename):
    X_train, y_train, X_test, y_test, d_X_test, d_y_test = lstm.load_data(filename, seq_len, True)
    predictions = lstm.predict_sequences_multiple(model, X_test, d_X_test, seq_len, 60,denormalise=True)
    actual_pos=d_y_test
    print(filename)
    for i,prediction in enumerate(predictions):
        distances=0
        for j,predic in enumerate(prediction):
            distances+=np.linalg.norm(np.array(predic)-actual_pos[i][j])**2
        distance=np.sqrt(distances)
        print (distance)

def predict_sequence(filename):
    X, d_X = lstm.get_last_sequence(filename, seq_len, True)
    predictions = lstm.predict_sequence(model, X, d_X, seq_len, 60,denormalise=True)
    print(filename)
    print (predictions)
    return predictions


files = ['test01.txt','test02.txt','test03.txt','test04.txt','test05.txt','test06.txt','test07.txt','test08.txt','test09.txt','test10.txt']

for file in files:
    run_test(file)
print ('Calculating sequences..')
for file in files:
    prediction=predict_sequence(file)
    x,y=np.array(prediction).T
    plt.figure(file)
    plt.scatter(x,y)
plt.show()


print("> Total Time : ", time.time() - t0)

