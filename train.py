import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn import datasets
from mlxtend.data import loadlocal_mnist
from mlxtend.plotting import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import joblib

train_images, train_labels = loadlocal_mnist(images_path='train-images.idx3-ubyte',
                                             labels_path='train-labels.idx1-ubyte')

test_images, test_labels = loadlocal_mnist(images_path='t10k-images.idx3-ubyte',
                                             labels_path='t10k-labels.idx1-ubyte')

X_train_reshaped = train_images.reshape(-1, 28, 28)

X_train_hog = []
for img in X_train_reshaped:
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    X_train_hog.append(features)
    
X_train_hog = np.array(X_train_hog)
y_train = np.array(train_labels)

X_test_reshaped = test_images.reshape(-1, 28, 28)

X_test_hog = []
for img in X_test_reshaped:
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    X_test_hog.append(features)
    
X_test_hog = np.array(X_test_hog)
y_test = np.array(test_labels)

svm = SVC()
svm.fit(X_train_hog, train_labels)

accuracy = svm.score(X_test_hog, test_labels)
print(f'Accuracy: {accuracy}')

joblib.dump(svm, 'svm_model.pkl')

