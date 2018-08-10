import cv2
import argparse
from os.path import join, isdir, exists
from datetime import datetime
from os import makedirs

def get_n_frames(n, k, src, dst):
    max_intentos = 10
    end_of_video = 0

    while True:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            break
        if max_intentos:
            raise "Capture is not open. There may be a problem with source {}".format(src)
        print("reintentando")
        max_intentos -=1

    i=0
    frame_number = 0 
    while(frame_number < n or n==-1):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            end_of_video = 0
            if k==0 or i % k == 0:
                
                final_dest = join(dst, '{0}_{1:03}.jpg'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), frame_number))
                if cv2.imwrite(final_dest, frame):
                    print("Saving frame {}".format(frame_number))
                    frame_number +=1
            i+=1
        elif end_of_video < 10:
            end_of_video +=1
        else:
            if n != -1:
                print("WARNING: Video is too short, can not take {} frames with k={}".format(n, k))
            break
    # When everything done, release the capture
    cap.release()
    print("{} frames were saved".format(frame_number))


def check_valid_n(x):
    x = int(x)
    if x <= 0 and x != -1:
        raise argparse.ArgumentTypeError("n must be greater than 0 or -1. It is {} ".format(x))
    return x

def check_valid_k(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError("k must be greater than 0. It is {} ".format(x))
    return x

def check_destination(dst):
    dst = str(dst)
    if isdir(dst):
        return dst
    if exists(dst):
        raise argparse.ArgumentTypeError("Destination must be an existing directory or a directory to be created. {} is not a directory".format(dst))
    makedirs(dst)
    print("{} was created".format(dst))
    return dst



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recover frames from a video source')

    parser.add_argument('-s', '--src', type=str, default='rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov',
                        help="String that identifies the camera to be used.")

    parser.add_argument('-d', '--dst', type=check_destination, default='.',
                        help="String that identifies the camera to be used.")

    parser.add_argument('-n', '--number-of-frames', type=check_valid_n, required=True,
                        help="total number of frames to be recovered from the stream")

    parser.add_argument('-k', '--k-frames-skipped', type=check_valid_k, required=True,
                        help="Number of frames to be skipped. There will be recovered one frame out of k. If k==0, get consecutive frames")

    args = parser.parse_args()

    get_n_frames(args.number_of_frames, args.k_frames_skipped, args.src, args.dst)
