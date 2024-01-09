import matplotlib.pyplot as plt
from skimage.feature import hog
from mlxtend.data import loadlocal_mnist
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import joblib

def prediksi():

    svm = joblib.load('model/svm_model.pkl')

    test_images, test_labels = loadlocal_mnist(images_path='datasets/t10k-images.idx3-ubyte',
                                               labels_path='datasets/t10k-labels.idx1-ubyte')

    index = np.random.randint(len(test_images))
    img, true_label = test_images[index], test_labels[index]
    features, _ = hog(img.reshape(28, 28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    predicted_label = svm.predict(features.reshape(1, -1))[0]


    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
    plt.show()

if __name__ == "__main__":
    prediksi()
