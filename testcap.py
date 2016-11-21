import cv2
import numpy as np
from easydict import EasyDict

# Import lua and torch module
import lutorpy as lua
lua.LuaRuntime(zero_based_index=True)
lua.require('torch')
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')
require('misc.LanguageModel')
require('misc.DataLoaderRaw')
utils = require('misc.utils')
net_utils = require('misc.net_utils')


def decode_sequence(ix_to_word, seq):
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


def load_model(model_path=r'/home/feather/Models/model_id1-501-1448236541.t7_cpu.t7'):
    checkpoint = torch.load(model_path)

    # Use opt to restore options
    opt = EasyDict()

    # Load the model checkpoint to evaluate

    fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm',
             'cnn_proto', 'cnn_model', 'seq_per_img'}
    for k in fetch:
        opt[k] = checkpoint.opt[k]
    vocab = checkpoint.vocab
    vocab = {int(k): v for k, v in dict(vocab).items()}

    protos = checkpoint.protos
    protos.lm._createClones()

    sample_opts = lua.table(sample_max=1, beam_size=2, temperature=1.0)
    protos.cnn._evaluate()
    protos.lm._evaluate()

    return vocab, protos

vocab, protos = load_model()

cnn = protos.cnn
vgg_mean = np.array([103.939, 116.779, 123.68])   # BGR

lstm = protos.lm
sample_opts = lua.table(sample_max=1, beam_size=2, temperature=1.0)


# im = cv2.imread('./data/Basketball/img/0002.jpg')  # Opencv use BGR format
im = cv2.imread('/home/feather/Dataset/cap-data/4.jpg')
im = im.astype(np.float32)
im = cv2.resize(im, (224, 224))  # VGG use 224x224 image size
# for i in range(0, 3):
#     im[:, :, i] -= vgg_mean[i]  # subtracting the vgg mean
im -= vgg_mean
im = im[:, :, [2, 1, 0]]  # convert from BGR to RGB
im = im.transpose(2, 0, 1)  # convert to torch format (1st dimension is color)
im = np.array([im])  # add rank by 1
im = torch.fromNumpyArray(im)
feats = cnn._forward(im)
seq = lstm._sample(feats, sample_opts)
seq = [seq[0][i][0] for i in range(16)]
sents = ' '.join(decode_sequence(vocab, seq))
print(sents)
