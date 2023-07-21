from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(digits.data[:-2], digits.target[:-2])
plt.imshow(digits.data[-2].reshape((8, 8)), cmap="Blues")
print("predicted number:", clf.predict([digits.data[-2]])[0])
plt.show()


def hallo():
    return "hallo"
