from easydict import EasyDict

# Import lua and torch module
import lutorpy as lua
lua.LuaRuntime(zero_based_index=True)
require('torch')
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

torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')

model_path = r'/home/feather/Models/model_id1-501-1448236541.t7_cpu.t7'

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

# Define image folder path
opt.image_folder = '/home/feather/Dataset/cap-data'
opt.coco_json = ''

# Create the Data Loader instance
loader = DataLoaderRaw(folder_path=opt.image_folder, coco_json=opt.coco_json)
protos = checkpoint.protos
protos.lm._createClones()


# Do eval
sample_opts = lua.table(sample_max=1, beam_size=2, temperature=1.0)
split = 'test'
protos.cnn._evaluate()
protos.lm._evaluate()
loader._resetIterator(split)
n = 0
loss_sum = 0
loss_eval = 0
predictions = {}

while True:
    data = loader._getBatch(batch_size=1,
                            split=split, seq_per_img=opt.seq_per_img)
    data.images = net_utils.prepro(data.images, False, False)
    n += data.images._size(1)

    feats = protos.cnn._forward(data.images)
    seq = protos.lm._sample(feats, sample_opts)
    seq = [seq[0][i][0] for i in range(16)]
    sents = ' '.join(decode_sequence(vocab, seq))
    print(sents)
    break
