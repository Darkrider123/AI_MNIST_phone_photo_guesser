from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from citire_poza import citire_poza
import os

def generic_file_read(filename):
    with open(filename, "r") as f:
        file = f.read()

    file = file.split("\n")
    file = file[1:]


    labels = []

    file = [[int(elem) for elem in line.split(",")] for line in file]

    for line in file:
        labels.append(line[0])

    file = [line[1:] for line in file]

    return file, labels


def read_files():

    train_samples, train_labels = generic_file_read(os.path.join("mnist_data", "mnist_train.csv"))
    test_samples, test_labels = generic_file_read(os.path.join("mnist_data", "mnist_test.csv"))

    return np.array(test_samples), np.array(test_labels), np.array(train_samples), np.array(train_labels)


def normalizare(sample_to_normalize):
    sample_to_normalize = sample_to_normalize / 255
    return sample_to_normalize


def citire_labels():
    labels = None
    with open(os.path.join("to_predict" , "labels.txt"), "r") as f:
        labels = f.read()
        labels = labels.split("\n")
    labels = list(map(lambda x: int(x), labels))
    return labels


def predict_folder(no_plotting = True):
    ia = IA()
    ia.load_IA()
    labels = citire_labels()
    bune = 0

    path = os.path.join("to_predict", "photos_to_predict")
    photo_names = os.listdir(path)
    
    photo_names.sort(key= lambda elem: int(elem.split(".")[0]))

    for id, photo_name in enumerate(photo_names):
        imagine = citire_poza(path, photo_name, id, no_plotting)
        imagine = normalizare(imagine)
        prezicere = ia.clf.predict(imagine.reshape(1, -1))
        print("In imaginea data este prezisa cifra: ", prezicere, " iar ea era de fapt ", labels[id])
        
        if prezicere == labels[id]:
            bune += 1
    print("\nAcuratetea este de: ", bune/len(labels))

    if no_plotting == False:
        plt.show()


class IA():
    def __init__(self):
        self.clf = None
        self.test_samples = None
        self.test_labels = None
        self.mean_train = None
        self.sigma_patrat = None

    def train_IA(self):
        test_samples, test_labels, train_samples, train_labels =    read_files()
        train_samples = normalizare(train_samples)
        test_samples = normalizare(test_samples)


        formatie_neuroni = [784, 2500, 2000, 1500, 1000, 500]
        formatie_neuroni = [10, 13, 15, 13, 10]
        

        clf = MLPClassifier(solver='adam', alpha=10**-5,
                            hidden_layer_sizes=(formatie_neuroni), random_state=5)

        clf.fit(train_samples, train_labels)

        self.test_samples = test_samples
        self.test_labels = test_labels
        self.clf = clf
    
    def compute_accuracy(self, test_samples=None, test_labels=None):
        bun = 0

        if test_samples == None or test_labels == None:
            test_samples, test_labels = generic_file_read(os.path.join("mnist_data", "mnist_test.csv"))

        raspunsuri = self.clf.predict(test_samples)
        for pred, true in zip(raspunsuri, test_labels):
            if pred == true:
                bun += 1
        return bun/len(test_labels)

    def load_IA(self):
        self.clf = pickle.load(open(os.path.join("saved_ai_state" , "clf.txt") , "rb"))

    def save_IA(self):
        self.clf = pickle.dump(self.clf, open(os.path.join("saved_ai_state" , "clf.txt") , "wb"))
        
        self.load_IA()
        