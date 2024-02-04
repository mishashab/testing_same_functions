import os
import numpy
import numpy as np
import sklearn.model_selection
import tensorflow
import cv2
import matplotlib.pyplot
import pickle

from model_easy import make_model_easy


def train(train_dir='..\\data\\processed\\fl\\train',
          model_dir='..\\models\\base_tf',
          epochs=30,
          learning_rate=0.001,
          batch_size=128,
          image_size=192):

    if not(os.path.exists('..\\variables')):

        x_train, x_val, y_train, y_val, marks = get_data_for_train(train_dir)

        os.mkdir('..\\variables')
        with open('..\\variables\\x_train.pkl', 'wb') as f:
            pickle.dump(x_train, f)

        with open('..\\variables\\x_val.pkl', 'wb') as f:
            pickle.dump(x_val, f)

        with open('..\\variables\\y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)

        with open('..\\variables\\y_val.pkl', 'wb') as f:
            pickle.dump(y_val, f)

        with open('..\\variables\\marks.pkl', 'wb') as f:
            pickle.dump(marks, f)
    else:
        with open('..\\variables\\x_train.pkl', 'rb') as f:
            x_train = pickle.load(f)

        with open('..\\variables\\x_val.pkl', 'rb') as f:
            x_val = pickle.load(f)

        with open('..\\variables\\y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)

        with open('..\\variables\\y_val.pkl', 'rb') as f:
            y_val = pickle.load(f)

        with open('..\\variables\\marks.pkl', 'rb') as f:
            marks = pickle.load(f)

    model = compile_model(len(marks),
                          image_size=image_size,
                          learning_rate=learning_rate,
                          )

    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        batch_size=batch_size)

    matplotlib.pyplot.plot(history.history['accuracy'],
                           label='Доля верных ответов на обучающем наборе')
    matplotlib.pyplot.plot(history.history['val_accuracy'],
                           label='Доля верных ответов на проверочном наборе')
    matplotlib.pyplot.xlabel('Эпоха обучения')
    matplotlib.pyplot.ylabel('Доля верных ответов')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    model.save(model_dir)


def compile_model(num_classes, image_size, learning_rate):
    model = make_model_easy(num_classes, image_size)
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tensorflow.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        ])

    return model


def get_data_for_train(train_dir):
    x, y = [], []
    for directory in os.listdir(train_dir):
        for image_name in os.listdir(os.path.join(train_dir, directory)):
            image = cv2.imread(os.path.join(train_dir, directory, image_name),
                               1)
            # x.append(image, axis=0)
            x.append(image)
            y.append(directory)

    x = numpy.array(x, dtype=object)
    marks = os.listdir(train_dir)
    y = numpy.array([marks.index(image_class) for image_class in y])
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=20,
        shuffle=True
    )
    print('plan A')
    x_train = tensorflow.ragged.constant(x_train[:1000])
    print('plan B')
    x_val = tensorflow.ragged.constant(x_val[:300])
    y_train = tensorflow.ragged.constant(y_train[:1000])
    y_val = tensorflow.ragged.constant(y_val[:300])
    print('YAhooooooo')
    return x_train, x_val, y_train, y_val, marks


if __name__ == '__main__':
    train()
