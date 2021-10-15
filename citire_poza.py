import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def citire_poza(path, filename, figure_number = 0, no_plotting = True):
    
    def plotare(imagine, titlu, numar_curent, numar_total_afisari, no_plotting):
        if (no_plotting == True):
            return

        plt.subplot(1, numar_total_afisari, numar_curent)
        plt.imshow(imagine)
        plt.title(titlu)

    def citire(filename):
        imagine = Image.open(filename)
        size = imagine.size
        imagine = np.array(imagine)
        return imagine, size


    def grayscale(imagine, size):
        imagine = imagine.flatten()
        imagine = imagine[::3]
        imagine = imagine.reshape((size[1], size[0]))
        return imagine


    def schimbare_contrast(imagine):
        media_luminozitatii = np.mean(imagine)
        if media_luminozitatii > 130:
            imagine = 255 - imagine
            return imagine, True
        return imagine, False


    def accentuare_contrast(imagine, size):
        media_luminozitatii = np.mean(imagine)
        brightest_point = np.max(imagine)
        imagine = imagine.flatten()

        aux = []
        for elem in imagine:
            if elem > 70/100 * brightest_point:
                aux.append(255)
            elif elem > np.mean([brightest_point, media_luminozitatii]):
                aux.append(255)
            else:
                aux.append(0)
                
        imagine = np.array(aux)
        imagine = imagine.reshape(size[1], size[0])
        return imagine




    def resize(imagine):
        imagine = Image.fromarray(imagine)
        imagine = imagine.resize((28,28))
        imagine = np.array(imagine)
        return imagine, (28, 28)

    def citire_main(path, filename, figure_number, no_plotting):
        if no_plotting == False:
            plt.figure(figure_number)
        numar_total_afisari = 5 #a se completa manual
        imagine, size = citire(os.path.join(path , str(filename)))
        plotare(imagine, "Imaginea initiala", 1, numar_total_afisari, no_plotting)
        imagine = grayscale(imagine, size)
        plotare(imagine, "Imaginea dupa ce am ales\nun singur canal de culoare", 2, numar_total_afisari, no_plotting)

        imagine, flag_afisare = schimbare_contrast(imagine)
        if flag_afisare:
            plotare(imagine, "Dupa ce am\nschimbat contrastul", 3, numar_total_afisari, no_plotting)


        imagine = accentuare_contrast(imagine, size)
        plotare(imagine, "Dupa ce am\naccentuat contrastul", 4, numar_total_afisari, no_plotting)


        imagine, size = resize(imagine)
        plotare(imagine, "Imaginea dupa ce am\nfacut-o 28x28", 5, numar_total_afisari, no_plotting)



        return imagine.flatten()
    return citire_main(path , filename, figure_number, no_plotting)