import numpy as np
import cv2

def batch_RGB2GRAY(X):
    X_reshaped = X.reshape((X.shape[0], X.shape[1]*X.shape[2], X.shape[3]))
    gray_images = cv2.cvtColor(X_reshaped, cv2.COLOR_RGB2GRAY)
    result = gray_images.reshape((X.shape[0], X.shape[1], X.shape[2]))
    assert result.shape[:3] == X.shape[:3]
    return result

def batch_RGB2HSV(X):
    X_reshaped = X.reshape((X.shape[0], X.shape[1]*X.shape[2], X.shape[3]))
    hsv_images = cv2.cvtColor(X_reshaped, cv2.COLOR_RGB2HSV)
    result = hsv_images.reshape(X.shape)
    assert result.shape == X.shape
    return result

def batch_RGB2LAB(X):
    X_reshaped = X.reshape((X.shape[0], X.shape[1]*X.shape[2], X.shape[3]))
    lab_images = cv2.cvtColor(X_reshaped, cv2.COLOR_RGB2LAB)
    result = lab_images.reshape(X.shape)
    assert result.shape == X.shape
    return result

def batch_RGB2IS(X, eps=1e-3):
    # I, S definidos en sec 2.1 http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.2093&rep=rep1&type=pdf
    
    I = np.sum(X, axis=3) / 3.
    # evitar problemas numéricos
    I_cp = I.copy()
    I_cp[I==0] = eps 
    S = 1 - np.min(X, axis=3) / I_cp
    return I, S

def calcular_modulo_gradiente(X):
    kernel_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
    kernel_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
    result = np.empty_like(X)
    
    for i in range(result.shape[0]):
        Gxi = cv2.filter2D(X[i],-1,kernel_x)
        Gyi = cv2.filter2D(X[i],-1,kernel_y)
        result[i] = np.sqrt(Gxi**2 + Gyi**2)
    return result


def batch_inpaint(X, mask, vecinos=3, alg=cv2.INPAINT_NS):
    # alg puede ser cv2.INPAINT_NS o cv2.INPAINT_TELEA
    result = np.empty_like(X)
    for i in range(len(X)):
        result[i] = cv2.inpaint(X[i],mask[i],vecinos,algS) #INPAINT_TELEA
    return result

def batch_dilate(X, vecinos=3):
    kernel = np.ones((vecinos,vecinos),np.uint8)
    result = np.empty_like(X)
    for i in range(len(X)):
        result[i] = cv2.dilate(X[i],kernel,iterations = 1)
    return result

def batch_close(X, vecinos=3):
    kernel = np.ones((vecinos,vecinos),np.uint8)
    result = np.empty_like(X)
    for i in range(len(X)):
        result[i] = cv2.morphologyEx(X[i], cv2.MORPH_CLOSE, kernel)
    return result

def RGB2Opponent(X):
    # Basado en https://github.com/opencv/opencv/blob/2.4/modules/features2d/src/descriptors.cpp#L126
    R = X[:,:,0].astype(np.float32)
    G = X[:,:,1].astype(np.float32)
    B = X[:,:,2].astype(np.float32)
    
    opponent = np.empty_like(X, dtype=np.int)
    opponent[:,:,0] = np.round((R - G)/np.sqrt(2)).astype(np.int)
    opponent[:,:,1] = np.round((R+G-2*B)/np.sqrt(6)).astype(np.int)
    opponent[:,:,2] = np.round((R+G+B)/np.sqrt(3)).astype(np.int)
    return opponent


def batch_RGB2Opponent(X):
    X_reshaped = X.reshape((X.shape[0], X.shape[1]*X.shape[2], X.shape[3]))
    X_opponent = RGB2Opponent(X_reshaped)
    return X_opponent.reshape((X.shape[0], X.shape[1],X.shape[2], X.shape[3]))

