import torchtext.vocab as vocab
import numpy as np
# cache_dir是保存golve词典的缓存路径
cache_dir = './glove.6B'
# dim是embedding的维度
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
# process glove to vocab_dict
string2id = glove.stoi
id2string = glove.itos
tensor = glove.get_vecs_by_tokens(['', '1998', '199999998', ',', 'cat'], True)
print(tensor)