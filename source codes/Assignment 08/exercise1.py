import numpy as np
from numpy import uint8
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import cv2
import sklearn.decomposition as dec
import random
import matplotlib.pylab as plt


def read_faces(w, h): # reads and resises data in folder face. why cant we use glob or os? :(
    size = w * h
    test_set = np.zeros((5, size))
    test_set[0] = cv2.resize(cv2.imread('exercise1/detect/face/obama.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[1] = cv2.resize(cv2.imread('exercise1/detect/face/boris.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[2] = cv2.resize(cv2.imread('exercise1/detect/face/merkel.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[3] = cv2.resize(cv2.imread('exercise1/detect/face/putin.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[4] = cv2.resize(cv2.imread('exercise1/detect/face/trump.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    return test_set


def read_objects(w, h): # reads and resises data in folder other. why cant we use glob or os? :(
    size = w * h
    test_set = np.zeros((5, size))
    test_set[0] = cv2.resize(cv2.imread('exercise1/detect/other/cat.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[1] = cv2.resize(cv2.imread('exercise1/detect/other/dog.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[2] = cv2.resize(cv2.imread('exercise1/detect/other/flag.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[3] = cv2.resize(cv2.imread('exercise1/detect/other/flower.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    test_set[4] = cv2.resize(cv2.imread('exercise1/detect/other/monkey.jpg',
                                        cv2.IMREAD_GRAYSCALE), (h, w)).flatten()
    return test_set


def show_igenface(igen_faces, i, h, w):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        igen_face = np.array(igen_faces[i].reshape((h, w)))
        fig.add_subplot(rows, columns, i)
        plt.imshow(igen_face)
    plt.show()


def calc_coefficient(image, mean, components, k): #a1, a2, a3, ..
    demean = np.zeros((image.shape[0]))
    demean = np.subtract(image, mean)
    eigen_coordinates = np.zeros((k))
    for j in range(k):
        eigen_coordinates[j] = np.dot(demean, components[j])
    return eigen_coordinates


def reconstruct_model(mean, components, coeff):
    reconstructed = mean
    for j in range(components.shape[0]):
        reconstructed += components[j] * coeff[j]
    return reconstructed


def calc_error(reconstructed, original):
    diff = np.zeros((original.shape[0]))
    diff[:] = original[:] - reconstructed[:]
    precision = np.sum(diff)
    return precision


def detect(presision):
    if abs(presision) < 60:
        return True
    return False

def recognize(image, ref):
    global_err = 100000000
    label = 0
    for i in range(ref.shape[0]):
        diff = np.zeros(image.shape[0])
        diff[:] = abs(ref[i, :] - image[:])
        err = np.sum(diff)
        if err < global_err:
            global_err = err
            label = i
    return label


def recognition_precision(predicted, ref):
    precision = 0
    for i in range(predicted.__len__()):
        if predicted[i] == ref[i]:
            precision += 1
    precision = 100 * precision/predicted.__len__()
    return precision

def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image
    target_names = lfw.target_names
    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Compute the PCA
    k = 100  # number of components
    pca = dec.PCA(n_components=k)
    pca.fit(X_train)

    # Visualize Eigen Faces
    show_igenface(pca.components_, 10, h, w)

    # Compute reconstruction error
    # report error for obama image from detect folder
    obama_raw = cv2.imread('exercise1/detect/face/obama.jpg', cv2.IMREAD_GRAYSCALE)
    obama = cv2.resize(obama_raw, (w, h)).flatten()

    mean = np.mean(X_train, axis=0)

    coeff = calc_coefficient(obama, mean, pca.components_, k)
    reconstructed = reconstruct_model(mean, pca.components_, coeff)
    error = calc_error(reconstructed, obama)
    print('reconstruction error for image obama.jpg is ', error)

    # Perform face detection
    face_folder = read_faces(h, w)
    object_folder = read_objects(h, w)
    precision = 0
    for i in range(face_folder.shape[0]):
        image = face_folder[i]
        coeff = calc_coefficient(image, mean, pca.components_, k)
        reconstructed = reconstruct_model(mean, pca.components_, coeff)
        error = calc_error(reconstructed, image)
        is_face = detect(error)
        if is_face:
            precision += 1

    for i in range(object_folder.shape[0]):
        image = object_folder[i]
        coeff = calc_coefficient(image, mean, pca.components_, k)
        reconstructed = reconstruct_model(mean, pca.components_, coeff)
        error = calc_error(reconstructed, image)
        is_face = detect(error)
        if not is_face:
            precision += 1

    precision = 100 * precision/10

    print('Detection precision for the test data in folder detect is ', precision, '%')

    # Perform face recognition

    image = face_folder[i]
    kdim_faces = pca.transform(X_train)
    kdim_coordinates = pca.transform(X_test)
    predicted = []
    for i in range(X_test.shape[0]):
        label_number = recognize(kdim_coordinates[i], kdim_faces)
        label = y_train[label_number]
        predicted.append(label)
    precision = recognition_precision(predicted, y_test)
    print('Recognition precision for the test data is ', precision, '%')


if __name__ == '__main__':
    main()
