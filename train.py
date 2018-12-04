import time
from DNN import DNN
import numpy as np
from keras.utils import np_utils
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from functions import read_data_from_h5

"""
This module is used to create and train a DNN model on the formatted data 
"""

root_dir = root_dir = "./"
data_h5 = root_dir+'data/body_pos.h5'
# Transform data
model = "mobilenet_thin"
logfile = root_dir+"logs.txt"
start_time = time.time()
pixels, labels = read_data_from_h5(data_h5)
end_time = time.time()
labels = labels.reshape(-1, 1)
pixels[:, :, :, 0] /= 640.0
pixels[:, :, :, 1] /= 480.0
pixels_ravel = np.zeros((pixels.shape[0], pixels.shape[1]*pixels.shape[2]*pixels.shape[3]))
for i in range(pixels.shape[0]):
    pixels_ravel[i, :] = pixels[i].ravel()

X_train, X_test, y_train, y_test = train_test_split(pixels_ravel, labels, test_size=0.2, random_state=11)

dnn = DNN()
feature_size = np.array([X_train.shape[1]])
layer_nodes = np.array([256, 64, 16])
dropouts = np.array([0, 0, 0])
np_of_output_class = 2
activation = 'relu'
dnn.create(feature_size, layer_nodes, dropouts, np_of_output_class, activation, l2_norm=0)
loss_function = "categorical_crossentropy"
optimizer_name = 'SGD'
dnn.compile(loss_function, optimizer_name, metrics=['accuracy'], learning_rate=0.001, momentum=0.9)
y_train_encoded = np_utils.to_categorical(np.array(y_train).reshape(-1))
y_test_encoded = np_utils.to_categorical(np.array(y_test).reshape(-1))
epochs = 300
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.reshape(-1)), y_train.reshape(-1))
dnn.model.summary()
dnn.train(X_train, y_train_encoded, X_test, y_test_encoded, epochs=epochs, class_weights=class_weights,
          batch_size=64, verbose=2)
dnn.save(directory=root_dir+'model', name='/pose_classifier.h5')

dnn.load(root_dir+'model/pose_classifier.h5')
y_pred_train = dnn.predict(X_train)
print("============================Training Report=========================================")
print(np.round(accuracy_score(y_pred_train, y_train)*100, 2))
print(classification_report(y_pred_train, y_train))

print("============================Testing Report=========================================")
y_pred_test = dnn.predict(X_test)
print(np.round(accuracy_score(y_pred_test, y_test)*100, 2))
print(classification_report(y_pred_test, y_test))
