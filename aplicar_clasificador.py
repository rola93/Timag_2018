from sklearn.externals import joblib
from os.path import isdir, isfile
from imageio import imsave
from batch import convert_to_features
import matplotlib.pyplot as plt
import numpy as np
import random

from utils.images_utils import read_images
from utils.image_modification import batch_close
from os.path import isdir, basename
from os import makedirs, listdir
from tqdm import tqdm
import re
import argparse
import cv2


seed=123
np.random.seed(seed)
random.seed(seed)

class Segmentador(object):
    def __init__(self, v=None):
        if v is None:
            version = max([int(s) for s in listdir('./models') if re.search(r'^\d+$',s)] + [0])
        else:
            version = v
        model_dir = f'./models/{version}/model.pkl'
        self.clf = joblib.load(model_dir)
        self.red=np.array([255,0,0], dtype=np.uint8)
        self.blue=np.array([0,0,255], dtype=np.uint8)

    def batch_classification(self, X):
        n,hh,ww,c = X.shape
        
        feats = convert_to_features(X)
        feats =  feats.reshape((n*hh*ww, feats.shape[-1]))
        feats = np.delete(feats, np.array([2,3,4,7,8,9,12]),1)
        mask = self.clf.predict(feats)
        mask = mask.reshape((n,hh,ww))
        return mask

    def batch_probabilities(self, X):
        n,hh,ww,c = X.shape
        
        feats = convert_to_features(X)
        feats =  feats.reshape((n*hh*ww, feats.shape[-1]))
        feats = np.delete(feats, np.array([2,3,4,7,8,9,12]),1)
        mask = self.clf.predict_proba(feats)[:,1]
        mask = mask.reshape((n,hh,ww))
        return mask

    def main(self, src, dst, proba, n_images, alp=0.75):
        X = read_images(src,as_gray=False, n=n_images, verbose=True)
        mask = self.batch_probabilities(X) if proba else self.batch_classification(X)
        if not proba:
            mask *= 255
            #mask = batch_close(mask,10)
        if not isdir(dst):
            makedirs(dst) 
        bn = basename(dst)

        print(f"Guardando {len(X)} ejemplos...")

        for i in tqdm(range(len(X))):
            imsave(dst+'/{}_{:03d}_original.png'.format(bn, i), X[i])
            
            if proba:
                plt.imshow(mask[i], cmap='hot')
                plt.colorbar()
                plt.savefig(dst+'/{}_{:03d}_proba.png'.format(bn, i))
                plt.clf()
            else:
                imsave(dst+'/{}_{:03d}_mask.png'.format(bn, i), mask[i])
                X[i][mask[i]!=0] = X[i][mask[i]!=0] * alp + self.blue * (1-alp)
                X[i][mask[i]==0] = X[i][mask[i]==0] * alp + self.red * (1-alp)
                imsave(dst+'/{}_{:03d}_labeled.png'.format(bn, i), X[i])

    def create_video(self, src, dst,n,live,alp=0.75):
        if not isdir(dst):
            makedirs(dst)
        try:
            # Pasamos el RGB a BGR porque vamos a trabajar con OpenCV
            self.red = self.red[::-1]
            self.blue = self.blue[::-1]
            src = 0 if src=='0' else src

            cap = cv2.VideoCapture(src)     

            h = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            w =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT)) 
    
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(dst+'/movie.avi',fourcc, cv2.CAP_PROP_FPS, (h,w))
            i=0           
            while(cap.isOpened() and i!=n):
                ret, frame = cap.read()
                if ret:
                    mask = self.batch_classification(np.array([frame[:,:,::-1]]))[0]
                    frame[mask!=0] = frame[mask!=0] * alp + self.blue * (1-alp)
                    frame[mask==0] = frame[mask==0] * alp + self.red * (1-alp)
                    # write the flipped frame
                    out.write(frame)
                    if live:
                        cv2.imshow('frame',frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    i += 1
                    if i % 10 == 0:
                        print("#"*10 + f"{i} frames procesados" + "#"*10)
                else:
                    break

        except Exception as e:
            print("Algo salio mal :(")
            print(e)
        finally:
            # Release everything if job is finished
            cap.release()
            out.release()
            if live:
                cv2.destroyAllWindows()

def folder(path):
    path=str(path)
    if isdir(path) or isfile(path) or path == '0':
        return path
    raise argparse.ArgumentTypeError("{} no es un archivo o directorio".format(path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmenta imagenes en tejido/no tejido')

    parser.add_argument('-s', '--src', type=folder, required=True,
                        help="Si es un directorio, toma las imagenes que existan en el para segmentar o calcular mapa de calor de las probabilidades. Si es un video, genera uno similar con sus frames segmentados.")

    parser.add_argument('-p', '--proba', action='store_true',
                        help="Esta bandera genera un mapa de calor con las probabilidades de cada pixel: 1 para instrumentos, cero para tejidos. Unicamente tiene sentido cuando se segmentan imagenes.")

    parser.add_argument('-l', '--live', action='store_true',
                        help="Muestra en vivo las imagenes generadas. Unicamente vale cuando --src es un video.")

    parser.add_argument('-d', '--dst', type=str, default='./results',
                        help="directorio para los resultados. Por defecto van a ./results")

    parser.add_argument('-n', '--numero-de-ejemplos', type=int, required=False, default=-1,
                        help="Cantidad de imagenes a procesar de la carpeta src, o frames del video src. -1 para todas las imagenes/frames.")

    args = parser.parse_args()

    print("Comienza la ejecucion")

    if isfile(args.src) or args.src=='0':
        if args.proba:
            print("Advertencia: --proba es ignorado al procesar un video.")
        Segmentador().create_video(args.src, args.dst, args.numero_de_ejemplos, args.live)
    else:
        if args.live:
            print("Advertencia: --live es ignorado al procesar imagenes.")
        Segmentador().main(args.src, args.dst, args.proba, args.numero_de_ejemplos)
    print("Fin de la ejecucion")

            
            
        

                
