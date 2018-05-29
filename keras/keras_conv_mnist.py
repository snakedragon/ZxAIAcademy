from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

np.random.seed(7)

def net():
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(1,1), padding="same", input_shape=(28,28,1), activation='relu',name='conv1'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding="same", activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

model = net()
model.fit(x=mnist.train.images.reshape(-1,28,28,1), y=mnist.train.labels,
          validation_data=(mnist.validation.images.reshape(-1,28,28,1),mnist.validation.labels), batch_size=100, epochs=10, verbose=2, shuffle=True)
score = model.evaluate(x=mnist.test.images.reshape(-1,28,28,1), y=mnist.test.labels)
print(score)
model.save_weights("net.h5")
#model.load_weights("net.h5")
print(model.predict_classes(mnist.test.images[1].reshape(-1,28,28,1)))
print(mnist.test.labels[1])
