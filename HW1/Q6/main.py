# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from os import listdir
from os.path import isfile, join
import random
import numpy as np
from skimage import io


def getDataset(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    random.shuffle(onlyfiles)
    chelsea = []
    manchester = []
    test = []
    for image in onlyfiles:
        if "c" in image:
            chelsea.append(image)
        elif "m" in image:
            manchester.append(image)
        else:
            test.append(image)

    # train_data = onlyfiles[: int(len(onlyfiles)*0.7)]
    # test_data = onlyfiles[int(len(onlyfiles)*0.7):]
    return chelsea, manchester


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    chelsea, manchester = getDataset("./Images")
    chelsea_avg_color = 0
    manchester_avg_color = 0
    # sum = 0
    # for image in chelsea:
    #     img = io.imread('./Images/'+image)
    #     avg_color_per_row = np.average(img, axis=0)
    #     avg_color = np.average(avg_color_per_row, axis=0)
    #     sum = sum + avg_color[2]
    # chelsea_avg_color = sum / len(chelsea)
    # sum = 0
    # for image in manchester:
    #     img = io.imread('./Images/'+image)[:, :, :-1]
    #     avg_color_per_row = np.average(img, axis=0)
    #     avg_color = np.average(avg_color_per_row, axis=0)
    #     sum = sum + avg_color[0]
    # manchester_avg_color = sum / len(manchester)

    # test data
    TChelsea = 0
    FChelsea = 0
    TManchester = 0
    FManchester = 0
    for image in chelsea:
        img = io.imread('./Images/'+image)
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        if avg_color[2] >= avg_color[0]:
            TChelsea = TChelsea + 1
        else:
            FManchester = FManchester + 1

    for image in manchester:
        img = io.imread('./Images/'+image)
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        if avg_color[0] >= avg_color[2]:
            TManchester = TManchester + 1
        else:
            FChelsea = FChelsea + 1

    print("TChelsea:{0}".format(TChelsea))
    print("FChelsea:{0}".format(FChelsea))
    print("TManchester:{0}".format(TManchester))
    print("FManchester:{0}".format(FManchester))
    print("Manchester Precision:{0}".format(TManchester / (TManchester + FManchester)))
    print("Chelsea Precision:{0}".format(TChelsea / (TChelsea + FChelsea)))
    print("Manchester Recall:{0}".format(TManchester / (TManchester + FChelsea)))
    print("Chelsea Recall:{0}".format(TChelsea / (TChelsea + FManchester)))
    print("Accuracy:{0}".format( (TChelsea+TManchester)/(TManchester+TChelsea+FChelsea+FManchester)))
