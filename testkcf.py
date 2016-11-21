import os
import re
import cv2
import sys
import copy
import glob
import shutil
from time import time

import KCF


selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 30
duration = 0.01


# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if(abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if(w > 0):
            ix, iy = int(x - w / 2), int(y - h / 2)
            initTracking = True


if __name__ == '__main__':

    # hog, fixed_window, multiscale, lab
    tracker = KCF.kcftracker(True, False, True, True)

    cv2.namedWindow('tracking')
    cv2.namedWindow('crop')
    cv2.setMouseCallback('tracking', draw_boundingbox)

    # item = sys.argv[1]
    # img_folder = './data/%s/' % item
    img_folder = sys.argv[1]
    gt_file = img_folder + '/groundtruth_rect.txt'
    f = open(gt_file)
    sep_pattern = r'[\d]+'
    gts = f.readlines()
    ix, iy, w, h = map(int, re.findall(sep_pattern, gts[0]))
    initTracking = True

    if os.path.exists('./crop'):
        shutil.rmtree('./crop')
        os.mkdir('./crop')
    else:
        os.mkdir('./crop')

    for idx, filename in enumerate(sorted(glob.glob(img_folder + '/img/*.jpg'))):
        raw_frame = cv2.imread(filename)
        frame = copy.deepcopy(raw_frame)
        if(selectingObject):
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
        elif(initTracking):
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)

            tracker.init([ix, iy, w, h], frame)

            initTracking = False
            onTracking = True
        elif(onTracking):
            t0 = time()
            boundingbox = tracker.update(frame)  # frame had better be contiguous
            t1 = time()

            boundingbox = list(map(int, boundingbox))
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[
                          0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)
            cropbox = [int(0.8 * boundingbox[0]), int(1.2 * (boundingbox[0] + boundingbox[2])),
                       int(0.8 * boundingbox[1]), int(1.2 * (boundingbox[1] + boundingbox[3]))]
            cv2.imshow('crop', frame[cropbox[2]:cropbox[3], cropbox[0]:cropbox[1]])
            cv2.imwrite('./crop/crop_%d.jpg' % idx,
                        raw_frame[cropbox[2]:cropbox[3], cropbox[0]:cropbox[1]])

            duration = 0.8 * duration + 0.2 * (t1 - t0)
            # duration = t1-t0
            cv2.putText(frame, 'FPS: ' + str(1 / duration)
                        [:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        gx, gy, gw, gh = map(int, re.findall(sep_pattern, gts[idx]))

        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)
        cv2.imshow('tracking', frame)

        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cv2.destroyAllWindows()
