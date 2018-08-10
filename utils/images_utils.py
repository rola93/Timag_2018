from imageio import imread, imsave
import matplotlib.pyplot as plt
from os.path import isdir, join
from os import listdir
import re
import numpy as np
from mpl_toolkits.mplot3d import  Axes3D

def read_images(folder, n=-1, as_gray=True, verbose=False):
    assert isdir(folder), "{} must be a folder containing images".format(folder)
    images = []
    # para leer siempre en el mismo orden
    file_names = sorted([filename for filename in listdir(folder) if re.search(r'\.(?:jpe?g|png|pgm|gif)$', filename, flags=re.I)])
    
    n = len(file_names) if n==-1 else n
    
    if verbose:
        print("{} image filenames found in {}".format(len(file_names), folder))
    
    for idx, image in enumerate(file_names):
        if idx == n:
            if verbose:
                print("stop reading at {}/{}".format(idx, len(file_names)-1))
            break
        if verbose:
            print("reading {}/{}".format(idx, len(file_names)-1))
        image_path = join(folder, image)
        images.append(imread(image_path, as_gray=as_gray))
    
    if verbose:
        print("returning after {} images read".format(len(images)))
    
    return np.array(images)

def mostrar(imagen, titulo='titulo',**kwargs):
    plt.imshow(imagen, **kwargs)
    plt.title(titulo)
    plt.show()

def mostrar_superficie(img):
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # create the figure
    fig = plt.figure()
    
    ax = Axes3D(fig)
    
    #ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.hot,
            linewidth=0)

    # show it
    plt.show()
