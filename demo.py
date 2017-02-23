'''
Instance Captioning, implemented by KCF+im2txt
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
import scipy.misc
from easydict import EasyDict
import kcftracker as KCF
import tensorflow as tf
import tensorlayer as tl
from buildmodel import *


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
        self.trackboxes = []
        self.capboxes = []

        self.img_folder = img_folder
        self.testcase = img_folder.split('/')[-1]
        self.inteval = 30

        self.window = cv2.namedWindow('tracking')
        cv2.setMouseCallback('tracking', self.draw_boundingbox)

        # Init captioning module
        if not model_path:
            self.model_path = r'./models/captioning/train'
        else:
            self.model_path = model_path
        self.vocab_file = "./word_counts.txt"
        self.max_caption_length = 20
        self.n_captions = 1
        self.top_k = 1
        self.init_sess()

        self.caps = []
        self.cap_start = False
        self.cap_finish = False
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
        self.video = cv2.VideoWriter('./video/%s.avi' % self.testcase,
                                     fourcc, 20, (self.frame_w, self.frame_h))

        # Init draw text settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.45
        self.text_color = green
        self.text_bold = 1

    def init_sess(self):
        '''
        Load the pretrain model, and init tensorflow session for caption generating
        '''
        mode = 'inference'

        # Build the inference graph.
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images, self.input_seqs, self.target_seqs, self.input_mask, self.input_feed = Build_Inputs(
                mode, input_file_pattern=None)

            self.net_image_embeddings = Build_Image_Embeddings(
                mode, self.images, train_inception=False)

            self.net_seq_embeddings = Build_Seq_Embeddings(self.input_seqs)

            self.softmax, self.net_img_rnn, self.net_seq_rnn, self.state_feed = Build_Model(
                mode, self.net_image_embeddings, self.net_seq_embeddings, self.target_seqs, self.input_mask)

            self.saver = tf.train.Saver()
        self.graph.finalize()

        self.sess = tf.Session(graph=self.graph)
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)
        self.saver.restore(self.sess, checkpoint_path)
        self.vocab = tl.nlp.Vocabulary(self.vocab_file)

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
            if abs(x - self.ix) > 10 and abs(y - self.iy) > 10:
                self.w, self.h = abs(x - self.ix), abs(y - self.iy)
                self.ix, self.iy = min(x, self.ix), min(y, self.iy)
                self.initTracking = True
                self.trackboxes.append([self.ix, self.iy, self.w, self.h])
            else:
                self.onTracking = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clean()

    def draw_trackboxes(self, frame):
        '''
        Draw each tracking target's bounding box
        '''
        for box in self.trackboxes:
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]),
                          green, 2)

    def init_trackers(self):
        '''
        For every bounding object, set a KCF tracker
        '''
        self.trackers = []
        for box in self.trackboxes:
            tracker = self.tracker = KCF.KCFTracker()
            tracker.init(box, self.raw_frame)
            self.trackers.append(tracker)

    def update_trackboxes(self):
        '''
        Update tracking boxes by KCF
        '''
        for idx, tracker in enumerate(self.trackers):
            # frame had better be contiguous
            self.trackboxes[idx] = list(map(int, tracker.update(self.raw_frame)))

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
        capbox = [max(0, bx - ew), min(bx + bw + ew, self.frame_w - 1),
                  max(0, by - eh), min(by + bh + eh, self.frame_h - 1)]
        return capbox

    def update_capboxes(self):
        '''
        Update captioning boxes by cal_capbox()
        '''
        self.capboxes = []
        for tbox in self.trackboxes:
            self.capboxes.append(self.cal_capbox(*tbox))

    def draw_capboxes(self, frame):
        '''
        For every caption region, draw bouding box
        '''
        for box in self.capboxes:
            cv2.rectangle(frame, (box[0], box[2]),
                          (box[1], box[3]),
                          grass, 1)

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

    def cap_instances(self):
        '''
        Caption the instances
        '''
        if self.cap_finish:
            flag = True
        else:
            flag = False
            self.caps = []

        for idx, capbox in enumerate(self.capboxes):
            im = self.raw_frame[capbox[2]:capbox[3],
                                capbox[0]:capbox[1]]
            encoded_image = cv2.imencode('.jpg', im)[1].tostring()
            init_state = self.sess.run(self.net_img_rnn.final_state,
                                       feed_dict={"image_feed:0": encoded_image})
            for _ in range(self.n_captions):
                state = np.hstack((init_state.c, init_state.h))  # (1, 1024)
                a_id = self.vocab.start_id
                words = []
                for _ in range(self.max_caption_length - 1):
                    softmax_output, state = self.sess.run([self.softmax, self.net_seq_rnn.final_state],
                                                          feed_dict={self.input_feed: [a_id],
                                                                     self.state_feed: state,
                                                                     })
                    state = np.hstack((state.c, state.h))
                    a_id = tl.nlp.sample_top(softmax_output[0], top_k=self.top_k)
                    word = self.vocab.id_to_word(a_id)
                    if a_id == self.vocab.end_id:
                        break
                    words.append(word)

                print(words)
            cut = 5
            i = 0
            cap = []
            while i < len(words):
                cap.append(' '.join(words[i:i + cut]))
                i += cut
            if flag:
                self.caps[idx] = cap
            else:
                self.caps.append(cap)

        if self.caps:
            self.cap_finish = True

    def draw_cap(self):
        '''
        Draw the captioning result on the left top of the frame
        '''

        for idx, capbox in enumerate(self.capboxes):
            # Set proper text color
            # strip_frame = self.raw_frame[self.capbox[2]:self.capbox[3],
            #                              self.capbox[0]:self.capbox[1]]
            # mean_color = np.mean(np.mean(strip_frame[:20], axis=0), axis=0)
            # self.text_color = self.complement(*mean_color)
            # Add some paddings
            x = capbox[0] + 2
            y = capbox[2]
            # Set proper y paddings for multi lines
            dy = 12
            for s in self.caps[idx]:
                y += dy
                cv2.putText(self.frame, s, (x, y),
                            self.font, self.font_scale,
                            self.text_color, self.text_bold)

    def narrate(self):
        '''
        Use NeuralTalk2's model to caption every instance
        '''
        while True:
            if self.cap_start:
                self.cap_instances()
            time.sleep(0.5)

    def clean(self):
        '''
        Clean all tracking and captioning result
        '''
        cv2.imshow('tracking', self.raw_frame)
        self.initTracking = False
        self.onTracking = False
        self.trackboxes = []
        self.capboxes = []
        self.cap_start = False
        # self.caps = []
        self.cap_finish = False

    def run(self):

        self.narrator.start()

        for idx, filename in enumerate(sorted(glob.glob(self.img_folder + '/img/*.jpg'))):
            raw_frame = cv2.imread(filename)
            self.raw_frame = raw_frame
            frame = copy.deepcopy(raw_frame)
            self.frame = frame
            cv2.imshow('tracking', frame)

            if not self.onTracking and not self.initTracking:
                self.clean()
                while True:
                    if self.selectingObject:
                        temp_frame = copy.deepcopy(raw_frame)
                        self.draw_trackboxes(temp_frame)
                        cv2.rectangle(temp_frame, (self.ix, self.iy),
                                      (self.cx, self.cy), green, 2)
                        cv2.imshow('tracking', temp_frame)
                    # If press 'c', continue tracking
                    c = cv2.waitKey(self.inteval) & 0xFF
                    if c == ord('c'):
                        self.update_capboxes()
                        self.cap_finish = False
                        self.cap_start = True
                        break
                    elif c == ord('e'):
                        self.clean()
                    # If press 'q', exit this program
                    elif c == 27 or c == ord('q'):
                        cv2.destroyAllWindows()
                        self.video.release()
                        return

            if self.initTracking:
                self.draw_trackboxes(frame)

                self.init_trackers()
                self.initTracking = False
                self.onTracking = True

            if self.onTracking:
                self.update_trackboxes()
                self.update_capboxes()

                self.draw_trackboxes(frame)
                self.draw_capboxes(frame)

                while not self.cap_finish:
                    time.sleep(0.1)
                self.draw_cap()

            cv2.imshow('tracking', frame)
            self.video.write(frame)
            # Save the tracking and captioning result
            cv2.imwrite('./trackcap/%s_%d.jpg' % (self.testcase, idx), frame)

            c = cv2.waitKey(self.inteval) & 0xFF
            # If press 'q', exit program
            if c == 27 or c == ord('q'):
                self.video.release()
                break
            elif c == ord('e'):
                self.clean()

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
