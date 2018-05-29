from keras.layers import Dense
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
    model.add(Dense(10, input_shape=(784,), activation="relu", name='layer1'))
    model.add(Dense(10, activation="softmax", name='output'))
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

model = net()
#model.fit(x=mnist.train.images, y=mnist.train.labels, validation_data=(mnist.validation.images,mnist.validation.labels), batch_size=100, epochs=10, verbose=2, shuffle=True)
#score = model.evaluate(x=mnist.test.images, y=mnist.test.labels)
#model.save_weights("net.h5")
model.load_weights("net.h5")
print(model.predict_classes(mnist.test.images[1].reshape(1,784)))
print(mnist.test.labels[1])
#print(score)