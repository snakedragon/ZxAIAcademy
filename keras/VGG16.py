from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras.datasets.cifar10 import load_data
(train_x, train_y), (test_x, test_y) = load_data()

train_x = train_x / 255.
test_x = test_x / 255.
train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)

print(train_x.shape, train_y.shape)

def VGG16():
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(1,1), input_shape=(32,32,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(128, (3, 3), strides=(1, 1),  padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

model = VGG16()
model.fit(x = train_x, y=train_y, validation_split=0.2, batch_size=64, epochs=50, verbose=2)
score = model.evaluate(test_x, test_y)
print(score)

