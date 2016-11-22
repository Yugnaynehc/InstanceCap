'''
Object tracking, implemented by KCF+NeuralTalk2
'''
import os
import re
import cv2
import sys
import copy
import glob
import shutil
import numpy as np
from time import time
from easydict import EasyDict
import KCF

# Import lua and torch module
import lutorpy as lua
lua.LuaRuntime(zero_based_index=True)
lua.require('torch')
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')
lua.require('misc.LanguageModel')

# Define some color name
white = (255, 255, 255)
purple = (255, 0, 128)
green = (0, 255, 0)
grass = (255, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)


def draw_boundingbox(event, x, y, flags, param):
    '''
    Mouse callback function; for init or change tracking target.
    '''
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h, cap

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
            cap = None
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if(w > 0):
            ix, iy = int(x - w / 2), int(y - h / 2)
            initTracking = True
            cap = None


def load_model(model_path=r'./models/model_id1-501-1448236541.t7_cpu.t7'):
    '''
    Load neuraltalk2 torch pretrain model. Return vocabulary dictionary,
    CNN model and LSTM model.
    '''
    # Load the model checkpoint
    checkpoint = torch.load(model_path)

    # Use opt to restore options
    opt = EasyDict()

    # Extract some options
    fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm',
             'cnn_proto', 'cnn_model', 'seq_per_img'}
    for k in fetch:
        opt[k] = checkpoint.opt[k]
    vocab = checkpoint.vocab
    vocab = {int(k): v for k, v in dict(vocab).items()}

    protos = checkpoint.protos
    protos.lm._createClones()
    protos.cnn._evaluate()
    protos.lm._evaluate()

    return vocab, protos


def decode_sequence(ix_to_word, seq):
    '''
    Decode word idx from seq, and return as a captioning sentence
    '''
    try:
        out = []
        for ix in seq:
            if ix == 0:
                return out
            word = ix_to_word[ix]
            out.append(word)
    except Exception, e:
        pass
    return out


def cap_image(im):
    '''
    Caption the image!
    '''
    global cnn, lstm, vocab, vgg_mean, sample_opts
    im = im.astype(np.float32)
    im = cv2.resize(im, (224, 224))  # VGG use 224x224 image size
    im -= vgg_mean
    im = im[:, :, [2, 1, 0]]  # convert from BGR to RGB
    im = im.transpose(2, 0, 1)  # convert to torch format (first dim is color)
    im = np.array([im])  # add rank by 1
    im = torch.fromNumpyArray(im)
    feats = cnn._forward(im)
    seq = lstm._sample(feats, sample_opts)
    seq = [seq[0][i][0] for i in range(16)]
    words = decode_sequence(vocab, seq)
    sentence = ' '.join(words)
    return sentence


def draw_cap(frame, cap, x, y):
    '''
    Draw the captioning result on the left top of the frame
    '''
    # Split the whole sentence in every 5 words
    cut = 5
    n = 0  # count for lines
    i = 0
    sents = []
    words = cap.split(' ')
    while i < len(words):
        sents.append(' '.join(words[i:i + cut]))
        i += cut
        n += 1
    # Add some paddings
    x += 2
    # Set proper y paddings for multi lines
    dy = 12
    for s in sents:
        y += dy
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)

# Init captioning module
vocab, protos = load_model()

cnn = protos.cnn
vgg_mean = np.array([103.939, 116.779, 123.68])  # BGR

lstm = protos.lm
sample_opts = lua.table(sample_max=1, beam_size=2, temperature=1.0)


# Init tracking module
selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 30
duration = 0.01

if __name__ == '__main__':
    # The four params for kcftracker are
    # hog, fixed_window, multiscale, lab
    tracker = KCF.kcftracker(True, False, True, True)

    cv2.namedWindow('tracking')
    # cv2.namedWindow('crop')
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

    # Init 'crop' folder to save detailed tracking region
    if os.path.exists('./crop'):
        shutil.rmtree('./crop')
        os.mkdir('./crop')
    else:
        os.mkdir('./crop')

    # Init 'trackcap' folder to save tracking and captioning result
    if os.path.exists('./trackcap'):
        shutil.rmtree('./trackcap')
        os.mkdir('./trackcap')
    else:
        os.mkdir('./trackcap')

    # Init 'video' folder to save merged result for every test case
    if not os.path.exists('./video'):
        os.mkdir('./video')

    # Init video saver for merging result to avi video
    temp_img_name = img_folder + '/img/' + os.listdir(img_folder + '/img')[0]
    temp_img = cv2.imread(temp_img_name)
    frame_h, frame_w, _ = temp_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter('./video/%s.avi' % img_folder.split('/')[-1],
                            fourcc, 20,
                            (frame_w, frame_h))

    cap = None
    for idx, filename in enumerate(sorted(glob.glob(img_folder + '/img/*.jpg'))):
        raw_frame = cv2.imread(filename)
        frame = copy.deepcopy(raw_frame)
        if(selectingObject):
            cv2.rectangle(frame, (ix, iy), (cx, cy), green, 2)
        elif(initTracking):
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), green, 2)

            tracker.init([ix, iy, w, h], frame)

            initTracking = False
            onTracking = True
        elif(onTracking):
            t0 = time()
            # frame had better be contiguous
            bx, by, bw, bh = list(map(int, tracker.update(frame)))
            t1 = time()

            duration = 0.8 * duration + 0.2 * (t1 - t0)
            # duration = t1-t0
            # cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'),
            #             (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), green, 2)

            # Show and save the (expanded) tracking region
            # Four elements are left, right, top, down
            # extbox = [int(0.8 * bx), int(1.2 * (bx + bw)),
            #           int(0.8 * by), int(1.2 * (by + bh))]

            # set extend square region box length
            box_length = min(3 * max(bw, bh), 0.75 * min(frame_h, frame_w))
            ew = int(max(box_length - bw, 0) / 2)
            eh = int(max(box_length - bh, 0) / 2)
            # Four elements are left, right, top, down
            extbox = [max(0, bx - ew), min(bx + bw + ew, frame_w - 1),
                      max(0, by - eh), min(by + bh + eh, frame_h - 1)]
            cv2.rectangle(frame,
                          (extbox[0], extbox[2]),
                          (extbox[1], extbox[3]),
                          grass, 1)

            crop_img = raw_frame[extbox[2]: extbox[3],
                                 extbox[0]: extbox[1]]
            # cv2.imshow('crop', crop_img)
            # cv2.imwrite('./crop/crop_%d.jpg' % idx, crop_img)

            # Update caption in every 30 frame
            if not cap or idx % 30 == 0:
                cap = cap_image(crop_img)
            # Put the captioning result in frame
            draw_cap(frame, cap, extbox[0], extbox[2])

        # Get and show the ground truth
        # gx, gy, gw, gh = map(int, re.findall(sep_pattern, gts[idx]))
        # cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), red, 2)
        cv2.imshow('tracking', frame)
        video.write(frame)
        # Save the tracking and captioning result
        cv2.imwrite('./trackcap/result_%d.jpg' % idx, frame)

        c = cv2.waitKey(inteval) & 0xFF
        # If press 'q', exit program
        if c == 27 or c == ord('q'):
            video.release()
            break
        # Use only 360 frame (about 18 seconds)
        if idx == 360:
            video.release()
            break

    cv2.destroyAllWindows()
    video.release()
