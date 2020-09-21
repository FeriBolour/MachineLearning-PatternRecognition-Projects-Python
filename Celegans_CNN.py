import numpy as np
import os
import glob
import cv2
from sklearn import svm
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization

x= []
y = []

#directory = input("Please enter your storage point")
start_time = time.time()

img_dir = r"V:\celegans\0\training" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        x.append(np.array([img, 0]))


img_dir = r"V:\celegans\1\training" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        x.append(np.array([img, 1]))

img_dir = r"V:\celegans\0\test" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        y.append(np.array([img, 0]))


img_dir = r"V:\celegans\1\test" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        y.append(np.array([img, 1]))

x = np.array(x)
y = np.array(y)


np.random.seed(57)  
np.random.shuffle(x)
np.random.shuffle(y)

x_train = x[:, 0]
y_train = x[:, 1:2]
x_test = y[:, 0]
y_test = y[:, 1:2]

train_size = y_train.shape[0]
test_size = y_test.shape[0]
    
x__train = np.zeros([train_size, 101,101])
x__test = np.zeros([test_size, 101,101])


for i in range(len(x_train)):
    x__train[i] = x_train[i]
    
for i in range(len(x_test)):
    x__test[i] = x_test[i]
    
    
x_train = x__train
x_test = x__test

mean = 0
std = 0
#x_train = x_train.reshape(x_train.shape[0], 101*101)
mean = np.mean(x_train)
std = np.std(x_train)

x_train = (x_train-mean)/std
x_train = x_train.reshape(x_train.shape[0],101,101,1)


x_test = (x_test-mean)/std
x_test = x_test.reshape(x_test.shape[0],101,101,1)

encoder = OneHotEncoder()
t = encoder.fit_transform(y_train.reshape((-1,1)))
t = t.toarray()

t_test = encoder.transform(y_test.reshape((-1,1)))
t_test = t_test.toarray() 
    
#y_train = np.array(y_train,dtype= 'f')
#y_test = np.array(y_test,dtype= 'f')


#create model
model = Sequential() #add model layers
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(101,101,1)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=(10,10)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))


#compile model using accuracy to measure model performance
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(x_train, t, epochs=6)

score = model.evaluate(x_test, t_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")


#model.save("4LayerModel_5.h5")