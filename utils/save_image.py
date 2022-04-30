import os
import matplotlib.pyplot as plt
def imwrite(idx, dir, datalist):
    for i in range(len(datalist)):
        file_dir = dir[i] + str(idx)+'.png'
        plt.imsave(file_dir, datalist[i].data.cpu().numpy().squeeze(), cmap="gray")
