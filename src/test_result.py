import tensorflow
import cv2
import numpy
import os
import pickle
import matplotlib.pyplot


# model_dir='..\\models\\base_tf'
model_dir = '..\\models\\learned_on_linux'
result_dir = '..\\models\\prediction'
test_images_dir = '..\\data\\processed\\fl\\test'


def predict_test():
    model0 = tensorflow.keras.models.load_model(model_dir)
    print(model0.summary())
    test_imgs = get_data_for_train()
    os.mkdir('..\\variables2')
    with open('..\\variables2\\x_test.pkl', 'wb') as f:
        pickle.dump(test_imgs, f)
    prediction = model0.predict(test_imgs)
    with open('..\\variables2\\prediction.pkl', 'wb') as f:
        pickle.dump(prediction, f)
    print(prediction)


def get_data_for_train():
    x = []
    for image_name in os.listdir(test_images_dir):
        image = cv2.imread(os.path.join(test_images_dir, image_name), 1)
        x.append(image)

    return tensorflow.ragged.constant(x[:500])


def prediction_function():
    with open('..\\variables2\\x_test.pkl', 'rb') as f:
        test_imgs = pickle.load(f)

    with open('..\\variables2\\prediction.pkl', 'rb') as f:
        prediction = pickle.load(f)

    with open('..\\variables\\marks.pkl', 'rb') as f:
        marks = pickle.load(f)
    # print(type(test_imgs[0]))
    for i in range(10):
        matplotlib.pyplot.imshow(test_imgs[i].numpy()[:, :, ::-1])
        matplotlib.pyplot.title(marks[prediction[i].argmax()])
        matplotlib.pyplot.show()
    print(marks)



if __name__ == '__main__':
    # predict_test()
    prediction_function()