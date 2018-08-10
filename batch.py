import cv2
import numpy as np
import random

seed=123
np.random.seed(seed)
random.seed(seed)

from imageio import imread, imsave

from os.path import isdir, join, isfile
from os import listdir
from utils.images_utils import mostrar, read_images
from utils.image_modification import batch_RGB2LAB,batch_RGB2LAB,batch_RGB2Opponent,batch_RGB2GRAY,calcular_modulo_gradiente
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def read_train_data():
    template_src_raw = 'dataset/Segmentation_Rigid_Training/Training/OP{}/Raw/img_{:02}_raw.png'
    template_src_mask = 'dataset/Segmentation_Rigid_Training/Training/OP{}/Masks/img_{:02}_instrument.png'
    
    raws = []
    masks = []
    for op in range(1,5):
        for img in range(1,41):
            
            raws.append(imread(template_src_raw.format(op,img)))
            masks.append(imread(template_src_mask.format(op,img), as_gray=True))
            
    return np.array(raws), np.array(masks)


def read_test_data():
    template_src_raw = 'dataset/Segmentation_Rigid_Testing_Revision/Testing/OP{}/img_{:02}_raw.png'
    template_src_mask = 'dataset/Segmentation_Rigid_Testing_GT/OP{}/img_{:02}_class.png'
    
    raws = []
    masks = []
    for op in range(1,7):
        for img in range(10):
            
            raws.append(imread(template_src_raw.format(op,img)))
            masks.append(imread(template_src_mask.format(op,img), as_gray=True))
            
    return np.array(raws), np.array(masks)


def convert_to_features(raws):
    print("Calculando features")
    n,hh,ww,c = raws.shape
    LAB = batch_RGB2LAB(raws)
    HSV = batch_RGB2LAB(raws)
    opponent = batch_RGB2Opponent(raws)
    gray = batch_RGB2GRAY(raws)
    G = calcular_modulo_gradiente(gray).reshape((n,hh,ww,1))
    # Pasamos el S al rango 0..255
    HSV[:,:,:,1]*=255
    print(raws.shape)
    print(LAB.shape)
    print(HSV.shape)
    print(opponent.shape)
    print(gray.shape)
    print(G.shape)
    print("generando matriz de ejemplos")
    X = np.concatenate((raws, G, HSV,LAB, opponent), axis=3)
    return X.astype(np.int16)

if __name__ == '__main__':

    print("Leyendo las imagenes de entrenamiento...")
    raws, masks = read_train_data()
    X = convert_to_features(raws)
    masks[masks!=0]=1

    print("Pasamos las imagenes a vectores")
    n,hh,ww,c = raws.shape
    xx =  X.reshape((n*hh*ww, X.shape[-1]))
    yy =  masks.reshape((n*hh*ww,)).astype(np.uint8)
    print(f"Leimos {len(X)} imagenes, {len(xx)} pixeles")
    del(X)
    del(masks)
    
    xx, yy = shuffle(xx, yy, random_state=123)

    print("Guardando los vectores de entrenamiento")
    np.save('vectors/X_train', xx)
    np.save('vectors/y_train', yy)

    del(xx)
    del(yy)

    print("Leyendo las imagenes de test...")
    raws, masks = read_test_data()
    X = convert_to_features(raws).astype(np.int16)
    masks[masks!=0]=1

    print("Pasamos las imagenes a vectores")
    n,hh,ww,c = raws.shape
    xx =  X.reshape((n*hh*ww, X.shape[-1]))
    yy =  masks.reshape((n*hh*ww,)).astype(np.uint8)
    print(f"Leimos {len(X)} imagenes, {len(xx)} pixeles")
    del(X)
    del(masks)

    xx, yy = shuffle(xx, yy, random_state=123)

    np.save('vectors/X_test', xx)
    np.save('vectors/y_test', yy)


