from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.regularizers import l2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
import itertools
from keras.optimizers import RMSprop
from sklearn.decomposition import PCA

# Defining the parameters
batch_size = 32
num_classes = 10
epochs = 20  # 50
pca_num_components = 270

# Splitting the data between train and test
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# plotting some random 10 images
class_names = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']

fig = plt.figure(figsize=(8, 3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:] == i)[0]
    features_idx = x_train[idx, ::]
    img_num = np.random.randint(features_idx.shape[0])
    im = (features_idx[img_num, ::])
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the input image
x_train /= 255
x_test /= 255

print(f'Shape before: ${x_train.shape}')
x_train_reshaped = x_train.reshape(50000, -1)
x_test_reshaped = x_test.reshape(10000, -1)
print(f'Shape before: ${x_train_reshaped.shape}')
print(f'Shape before: ${x_test_reshaped.shape}')

# выбор количества компонент
pca = PCA().fit(x_train_reshaped)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.show()

# Определение главных компонент
pca = PCA(n_components=pca_num_components, whiten=True)
pca.fit(x_train_reshaped)

# Компоненты
components = pca.components_
print(components)

# Объяснимая дисперсия
variance = pca.explained_variance_
print(f'variance ${variance}')

# Понижение размерности
x_train_pca = pca.transform(x_train_reshaped)
print('transformed shape : ', x_train_pca.shape)

pca = PCA(n_components=pca_num_components, whiten=True)
pca.fit(x_test_reshaped)
x_test_pca = pca.transform(x_test_reshaped)


x_train_pca_3d = x_train_pca.reshape(x_train_pca.shape[0], x_train_pca.shape[1], 1)
print(f'reshaped train to 3d {x_train_pca_3d.shape}')
x_test_pca_3d = x_test_pca.reshape(x_test_pca.shape[0], x_test_pca.shape[1], 1)
print(f'reshaped test to 3d {x_test_pca_3d.shape}')

model = Sequential()
model.add(Conv1D(64, 3, padding='same', input_shape=(x_train_pca_3d.shape[1], x_train_pca_3d.shape[2])))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv1D(128, 3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# summary of the model
print(model.summary())

# plot_model(model, to_file='cnn.png', show_shapes=True)
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# compile
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Training the model
history = model.fit(x_train_pca_3d, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    shuffle=True)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Predict the values from the validation dataset
y_predicted = model.predict(x_test_pca_3d)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(y_predicted, axis=1)
print(f'y predicted {Y_pred_classes}')
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
print(f'y true {Y_true}')
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=class_names)
print('Finish')
