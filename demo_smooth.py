'''
Object tracking, implemented by KCF+NeuralTalk2
'''
import os
import re
import cv2
import sys
import copy
import glob
import time
import shutil
import threading
import numpy as np
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


class InstanceCaptioner(object):

    def __init__(self, img_folder, model_path=None):

        # Init tracking module
        self.selectingObject = False
        self.initTracking = False
        self.onTracking = False
        self.ix, self.iy, self.cx, self.cy = -1, -1, -1, -1
        self.w, self.h = 0, 0
        self.inteval = 30
        self.capbox = None

        self.img_folder = img_folder
        self.tracker = KCF.kcftracker(True, False, True, True)

        self.window = cv2.namedWindow('tracking')
        cv2.setMouseCallback('tracking', self.draw_boundingbox)

        gt_file = img_folder + '/groundtruth_rect.txt'
        self.sep_pattern = r'[\d]+'
        with open(gt_file) as f:
            self.gts = f.readlines()

        # Init captioning module
        if not model_path:
            self.model_path = r'./models/model_id1-501-1448236541.t7_cpu.t7'
        else:
            self.model_path = model_path
        self.vocab, self.protos = self.load_model(self.model_path)

        self.cnn = self.protos.cnn
        self.vgg_mean = np.array([103.939, 116.779, 123.68])  # BGR

        self.lstm = self.protos.lm
        self.sample_opts = lua.table(sample_max=1, beam_size=2, temperature=1.0)

        self.cap = None
        self.narrator = threading.Thread(target=self.narrate)
        self.narrator.setDaemon(True)

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
        self.frame_h, self.frame_w, _ = temp_img.shape
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.video = cv2.VideoWriter('./video/%s.avi' % img_folder.split('/')[-1],
                                     fourcc, 20,
                                     (self.frame_w, self.frame_h))

        # Init draw text settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.4
        self.text_color = white
        self.text_bold = 1

    def load_model(self, model_path):
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

    def draw_boundingbox(self, event, x, y, flags, param):
        '''
        Mouse callback function; for init or change tracking target.
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selectingObject = True
            self.onTracking = False
            self.ix, self.iy = x, y
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.selectingObject = False
            if(abs(x - self.ix) > 10 and abs(y - self.iy) > 10):
                self.w, self.h = abs(x - self.ix), abs(y - self.iy)
                self.ix, self.iy = min(x, self.ix), min(y, self.iy)
                self.initTracking = True
                self.cap = None
            else:
                self.onTracking = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.onTracking = False
            if(self.w > 0):
                self.ix, self.iy = int(x - self.w / 2), int(y - self.h / 2)
                self.initTracking = True
                self.cap = None

    def cal_capbox(self, bx, by, bw, bh):
        '''
        Calculate extended square region box for captioning
        '''
        # Set extend square region box length
        box_length = min(max(200, 3 * max(bw, bh)), 0.75 *
                         min(self.frame_h, self.frame_w))
        ew = int(max(box_length - bw, 0) / 2)
        eh = int(max(box_length - bh, 0) / 2)
        # Four elements are left, right, top, down
        self.capbox = [max(0, bx - ew), min(bx + bw + ew, self.frame_w - 1),
                       max(0, by - eh), min(by + bh + eh, self.frame_h - 1)]
        return self.capbox

    @staticmethod
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

    @staticmethod
    def hilo(a, b, c):
        '''
        Sum of the min & max of (a, b, c)
        '''
        if c < b:
            b, c = c, b
        if b < a:
            a, b = b, a
        if c < b:
            b, c = c, b
        return a + c

    @staticmethod
    def complement(b, g, r):
        '''
        Calculate complement color for given (b, g, r)
        '''
        k = InstanceCaptioner.hilo(b, g, r)
        return tuple(int(k - u) for u in (b, g, r))

    def cap_image(self):
        '''
        Caption the image!
        '''
        im = self.raw_frame[self.capbox[2]:self.capbox[3],
                            self.capbox[0]:self.capbox[1]]
        im = im.astype(np.float32)
        # mean_color = np.mean(np.mean(im[:30, :], axis=0), axis=0)
        # self.text_color = self.complement(*mean_color)
        im = cv2.resize(im, (224, 224))  # VGG use 224x224 image size
        im -= self.vgg_mean
        im = im[:, :, [2, 1, 0]]  # convert from BGR to RGB
        im = im.transpose(2, 0, 1)  # convert to torch format (first dim is color)
        im = np.array([im])  # add rank by 1
        im = torch.fromNumpyArray(im)
        feats = self.cnn._forward(im)
        seq = self.lstm._sample(feats, self.sample_opts)
        seq = [seq[0][i][0] for i in range(16)]
        self.words = self.decode_sequence(self.vocab, seq)
        # Split the whole sentence in every 5 words
        cut = 5
        i = 0
        self.cap = []
        while i < len(self.words):
            self.cap.append(' '.join(self.words[i:i + cut]))
            i += cut

    def draw_cap(self):
        '''
        Draw the captioning result on the left top of the frame
        '''
        # Add some paddings
        x = self.capbox[0] + 2
        y = self.capbox[2]
        # Set proper y paddings for multi lines
        dy = 12
        for s in self.cap:
            y += dy
            cv2.putText(self.frame, s, (x, y),
                        self.font, self.font_scale,
                        self.text_color, self.text_bold)

    def narrate(self):
        while True:
            if self.capbox:
                self.cap_image()
            time.sleep(0.5)

    def run(self):
        # Using ground truth to init tracking target
        self.ix, self.iy, self.w, self.h = map(
            int, re.findall(self.sep_pattern, self.gts[0]))
        self.initTracking = True
        self.narrator.start()

        for idx, filename in enumerate(sorted(glob.glob(self.img_folder + '/img/*.jpg'))):
            raw_frame = cv2.imread(filename)
            self.raw_frame = raw_frame
            frame = copy.deepcopy(raw_frame)
            self.frame = frame
            if(self.selectingObject):
                cv2.rectangle(frame, (self.ix, self.iy), (self.cx, self.cy), green, 2)
            elif(self.initTracking):
                cv2.rectangle(frame, (self.ix, self.iy),
                              (self.ix + self.w, self.iy + self.h), green, 2)

                self.tracker.init([self.ix, self.iy, self.w, self.h], frame)

                self.initTracking = False
                self.onTracking = True
            elif(self.onTracking):
                # frame had better be contiguous
                bx, by, bw, bh = list(map(int, self.tracker.update(frame)))
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), green, 2)

                # Show and save the (expanded) captioning region
                capbox = self.cal_capbox(bx, by, bw, bh)
                cv2.rectangle(frame,
                              (capbox[0], capbox[2]),
                              (capbox[1], capbox[3]),
                              grass, 1)
                while not self.cap:
                    time.sleep(0.1)
                self.draw_cap()
            # Get and show the ground truth
            # gx, gy, gw, gh = map(int, re.findall(self.sep_pattern, self.gts[idx]))
            # cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), red, 2)
            cv2.imshow('tracking', frame)
            self.video.write(frame)
            # Save the tracking and captioning result
            cv2.imwrite('./trackcap/result_%d.jpg' % idx, frame)

            c = cv2.waitKey(self.inteval) & 0xFF
            # If press 'q', exit program
            if c == 27 or c == ord('q'):
                self.video.release()
                break
            # Use only 400 frame (about 20 seconds)
            if idx == 400:
                self.video.release()
                break

        cv2.destroyAllWindows()
        self.video.release()


if __name__ == '__main__':
    img_folder = sys.argv[1]
    inscap = InstanceCaptioner(img_folder)
    inscap.run()
