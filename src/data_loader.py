# coding=utf-8

from typing import List, Tuple, Dict, Any

import os
import random
import pickle
import json
import numpy as np
import pandas as pd

import torch
import torchtext.vocab as vocab
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer

stop_words = set(stopwords.words('english'))
word_tokenizer = RegexpTokenizer(r'\w+')

def remove_stopword(sentence):
    return ' '.join([word for word in word_tokenizer.tokenize(sentence) if word not in stop_words])

def sampling(imps, ratio=4):
    pos = []
    neg = []
    for imp in imps.split():
        if imp[-1] == '1':
            pos.append(imp)
        else:
            neg.append(imp)
    n_neg = ratio * len(pos)
    if n_neg <= len(neg):
        neg = random.sample(neg, n_neg)
    else:
        neg = random.sample(neg * (n_neg // len(neg) + 1), n_neg)
    random.shuffle(neg)
    res = pos + neg
    random.shuffle(res)
    return ' '.join(res)

class MindDataset(Dataset):
    def __init__(self, 
            root: str,
            tokenizer: AutoTokenizer,
            mode: str = 'train',
            split: str = 'small', 
            news_max_len: int = 20,
            hist_max_len: int = 20,
            seq_max_len: int = 300
            ) -> None:
        super(MindDataset, self).__init__()
        self.data_path = os.path.join(root, split)
        self._mode = mode
        self._split = split
        
        self._tokenizer = tokenizer
        self._mode = mode
        self._news_max_len = news_max_len
        self._hist_max_len = hist_max_len
        self._seq_max_len = seq_max_len
        # 获取ID特征映射
        self.type_feat_map, self.sub_type_feat_map, self.news_id_feat_map = get_category_feat_map(root=root, split=split)
        self._examples = self.get_examples(negative_sampling=4)
        print(self._examples.head())
        self._news = self.process_news()
    
    def get_examples(self, 
            negative_sampling: bool = None
            ) -> Any:
        behavior_file = os.path.join(self.data_path, self._mode, 'behaviors.tsv')
        if self._split == 'small':
            df = pd.read_csv(behavior_file, sep='\t', header=None, 
                    names=['user_id', 'time', 'news_history', 'impressions'])
            df['impression_id'] = list(range(len(df)))
        else:
            df = pd.read_csv(behavior_file, sep='\t', header=None, 
                    names=['impression_id', 'user_id', 'time', 'news_history', 'impressions'])
        if self._mode == 'train':
            df = df.dropna(subset=['news_history'])
        df['news_history'] = df['news_history'].fillna('')

        if self._mode == 'train' and negative_sampling is not None:
            df['impressions'] = df['impressions'].apply(lambda x: sampling(
                    x, ratio=negative_sampling))
        df = df.drop('impressions', axis=1).join(df['impressions'].str.split(' ', 
                expand=True).stack().reset_index(level=1, drop=True).rename('impression'))
        if self._mode == 'test':
            df['news_id'] = df['impression']
            df['click'] = [-1] * len(df)
        else:
            df[['news_id', 'click']] = df['impression'].str.split('-', expand=True)
        df['click'] = df['click'].astype(int)
        return df

    def process_news(self) -> Dict[str, Any]:
        filepath = os.path.join(self.data_path, 'news_dict.pkl')
        if os.path.exists(filepath):
            print('Loading news info from', filepath)
            with open(filepath, 'rb') as fin: news = pickle.load(fin)
            return news
        news = dict()
        news = self.read_news(news, os.path.join(self.data_path, 'train'))
        news = self.read_news(news, os.path.join(self.data_path, 'dev'))
        if self._split == 'large':
            news = self.read_news(news, os.path.join(self.data_path, 'test'))

        print('Saving news info from', filepath)
        with open(filepath, 'wb') as fout: pickle.dump(news, fout)
        return news

    def read_news(self, 
            news: Dict[str, Any], 
            filepath: str,
            drop_stopword: bool = True,
            ) -> Dict[str, Any]:
        with open(os.path.join(filepath, 'news.tsv'), encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            info = dict()
            splitted = line.strip('\n').split('\t')
            news_id = splitted[0]
            if news_id in news:
                continue
            title = splitted[3].lower()
            abstract = splitted[4].lower()
            if drop_stopword:
                title = remove_stopword(title)
                abstract = remove_stopword(abstract)
            news[news_id] = dict()
            title_words = self._tokenizer.tokenize(title)
            news[news_id]['title'] = self._tokenizer.convert_tokens_to_ids(title_words)
            abstract_words = self._tokenizer.tokenize(abstract)
            news[news_id]['abstract'] = self._tokenizer.convert_tokens_to_ids(abstract_words)
            # news加入ID特征
            news_type = splitted[1]
            news_sub_type = splitted[2]
            news[news_id]["type"] = self.type_feat_map.get(news_type, 0) # unk:0
            news[news_id]["sub_type"] = self.sub_type_feat_map.get(news_sub_type, 0)
        return news

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        segment_ids = torch.tensor([item['segment_ids'] for item in batch])
        input_mask = torch.tensor([item['input_mask'] for item in batch])
        news_segment_ids = torch.tensor([item['news_segment_ids'] for item in batch])
        sentence_ids = torch.tensor([item['sentence_ids'] for item in batch])
        sentence_mask = torch.tensor([item['sentence_mask'] for item in batch])
        sentence_segment_ids = torch.tensor([item['sentence_segment_ids'] for item in batch])
        news_type_ids =  torch.tensor([item['news_type_ids'] for item in batch])
        news_sub_type_ids = torch.tensor([item['news_sub_type_ids'] for item in batch])
        inputs = {'input_ids': input_ids, 
                  'segment_ids': segment_ids, 
                  'input_mask': input_mask, 
                  'news_segment_ids': news_segment_ids, 
                  'sentence_ids': sentence_ids, 
                  'sentence_mask': sentence_mask, 
                  'sentence_segment_ids': sentence_segment_ids,
                  'news_type_ids': news_type_ids,
                  'news_sub_type_ids' : news_sub_type_ids 
                }
        if self._mode == 'train':
            inputs['label'] = torch.tensor([item['label'] for item in batch])
            return inputs 
        elif self._mode == 'dev':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            inputs['label'] = torch.tensor([item['label'] for item in batch])
            return inputs 
        elif self._mode == 'test':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def pack_bert_features(self, example: Any):
        curr_news = self._news[example['news_id']]['title'][:self._news_max_len]
        news_segment_ids = []
        hist_news = []
        sentence_ids = [0, 1, 2]
        for i, ns in enumerate(example['news_history'].split()[:self._hist_max_len]):
            ids = self._news[ns]['title'][:self._news_max_len]
            hist_news += ids
            news_segment_ids += [i + 2] * len(ids)
            sentence_ids.append(sentence_ids[-1] + 1)
        
        tmp_hist_len = self._seq_max_len-len(curr_news)-3
        hist_news = hist_news[:tmp_hist_len]
        input_ids = [self._tokenizer.cls_token_id] + curr_news + [self._tokenizer.sep_token_id] \
                    + hist_news + [self._tokenizer.sep_token_id]
        news_segment_ids = [0] + [1] * len(curr_news) + [0] + news_segment_ids[:tmp_hist_len] + [0]
        segment_ids = [0] * (len(curr_news) + 2) + [1] * (len(hist_news) + 1)
        input_mask = [1] * len(input_ids)

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len
        news_segment_ids = news_segment_ids + [0] * padding_len

        sentence_segment_ids = [0] * 3 + [1] * (len(sentence_ids) - 3)
        sentence_mask = [1] * len(sentence_ids)

        sentence_max_len = 3 + self._hist_max_len
        sentence_mask = [1] * len(sentence_ids)
        padding_len = sentence_max_len - len(sentence_ids)
        sentence_ids = sentence_ids + [0] * padding_len
        sentence_mask = sentence_mask + [0] * padding_len
        sentence_segment_ids = sentence_segment_ids + [0] * padding_len

        # add ID feat
        curr_news_type = self._news[example['news_id']]['type']
        curr_news_sub_type = self._news[example['news_id']]['sub_type']
        news_type_ids = [self.type_feat_map["unk"]] + [curr_news_type] + [self.type_feat_map["unk"]]
        news_sub_type_ids = [self.sub_type_feat_map["unk"]] + [curr_news_sub_type] + [self.sub_type_feat_map["unk"]]
        for i, ns in enumerate(example['news_history'].split()[:self._hist_max_len]):
            news_type_ids.append(self._news[ns]['type'])
            news_sub_type_ids.append(self._news[ns]['sub_type'])
        padding_len = sentence_max_len - len(sentence_ids)
        news_type_ids.extend([self.type_feat_map["pad"]] * padding_len)
        news_sub_type_ids.extend([self.sub_type_feat_map["pad"]] * padding_len)

        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len
        assert len(news_segment_ids) == self._seq_max_len

        assert len(sentence_ids) == sentence_max_len
        assert len(sentence_mask) == sentence_max_len
        assert len(sentence_segment_ids) == sentence_max_len

        assert len(news_type_ids) == len(news_sub_type_ids)
        assert len(news_type_ids) == len(sentence_ids)

        return input_ids, input_mask, segment_ids, news_segment_ids, \
                sentence_ids, sentence_mask, sentence_segment_ids,\
                news_type_ids, news_sub_type_ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples.iloc[index]
        input_ids, input_mask, segment_ids, news_segment_ids, \
            sentence_ids, sentence_mask, sentence_segment_ids,\
            news_type_ids, news_sub_type_ids = self.pack_bert_features(example)
        inputs = {'input_ids': input_ids, 
                  'segment_ids': segment_ids, 
                  'input_mask': input_mask, 
                  'news_segment_ids': news_segment_ids, 
                  'sentence_ids': sentence_ids, 
                  'sentence_mask': sentence_mask, 
                  'sentence_segment_ids': sentence_segment_ids,
                  "news_type_ids" : news_type_ids,
                  "news_sub_type_ids" : news_sub_type_ids
                  }
        if self._mode == 'train':
            inputs['label'] = example['click']
            return inputs 
        elif self._mode == 'dev':
            inputs['impression_id'] = example['impression_id']
            inputs['label'] = example['click']
            return inputs 
        elif self._mode == 'test':
            inputs['impression_id'] = example['impression_id']
            return inputs 
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return len(self._examples)

def get_glove_pretrain(root, split, glove_cache_dir):
    """
    glove向量剪枝 仅保留特征词对应的向量作为pretrain weights传入模型初始化中，加速训练减小显存占用
    """
    data_path = os.path.join(root, split)
    df_train = pd.read_csv(os.path.join(data_path, 'train', 'news.tsv'),
                            sep="\t", 
                            names=["ID", "Type", "SubType", "Title", "Summary", "Url",
                                    "Entity_Title", "Entity_Summary"])

    df_dev = pd.read_csv(os.path.join(data_path, 'dev', 'news.tsv'),
                            sep="\t", 
                            names=["ID", "Type", "SubType", "Title", "Summary", "Url",
                                    "Entity_Title", "Entity_Summary"])

    df_test = pd.read_csv(os.path.join(data_path, 'test', 'news.tsv'),
                            sep="\t", 
                            names=["ID", "Type", "SubType", "Title", "Summary", "Url",
                                    "Entity_Title", "Entity_Summary"])
    # type
    type_cnt = df_train["Type"].value_counts().to_dict()
    type_cnt.update(df_dev["Type"].value_counts().to_dict())
    type_cnt.update(df_test["Type"].value_counts().to_dict())

    # subtype
    type_cnt.update(df_train["SubType"].value_counts().to_dict())
    type_cnt.update(df_dev["SubType"].value_counts().to_dict())
    type_cnt.update(df_test["SubType"].value_counts().to_dict())

    total_keys = list(type_cnt.keys())
    total_values = [i for i in range(len(total_keys))]
    feat_map = dict(zip(total_keys, total_values))
    glove = vocab.GloVe(name='6B', dim=300, cache=glove_cache_dir)
    word2id_map = glove.stoi
    pretrain_weights = []
    for key in total_keys:
        if key in word2id_map.keys():
            pretrain_weights.append(glove.get_vecs_by_tokens([key], True).tolist()[0])
        else:
            pretrain_weights.append(glove.get_vecs_by_tokens(["the"], True).tolist()[0])
    return feat_map, np.array(pretrain_weights)

# type、subtype、item_id map    
def get_category_feat_map(root, split):
    """ 获取ID特征的featmap用于特征id化后输入模型包括category、subcategory、item_id
    """
    data_path = os.path.join(root, split)
    df_train = pd.read_csv(os.path.join(data_path, 'train', 'news.tsv'),
                            sep="\t", 
                            names=["ID", "Type", "SubType", "Title", "Summary", "Url",
                                    "Entity_Title", "Entity_Summary"])
    df_dev = pd.read_csv(os.path.join(data_path, 'dev', 'news.tsv'),
                            sep="\t", 
                            names=["ID", "Type", "SubType", "Title", "Summary", "Url",
                                    "Entity_Title", "Entity_Summary"])
    df_test = pd.read_csv(os.path.join(data_path, 'test', 'news.tsv'),
                            sep="\t", 
                            names=["ID", "Type", "SubType", "Title", "Summary", "Url",
                                    "Entity_Title", "Entity_Summary"])
    # type
    type_cnt_train = df_train["Type"].value_counts().to_dict()
    type_cnt_train.update(df_dev["Type"].value_counts().to_dict())
    type_cnt_train.update(df_test["Type"].value_counts().to_dict())

    type_feat_keys = ["unk", "pad"] + list(type_cnt_train.keys())
    type_feat_values = [i for i in range(len(type_feat_keys))]
    assert len(type_feat_keys) == len(type_feat_values)
    type_feat_map = dict(zip(type_feat_keys, type_feat_values))
    # subtype
    sub_type_cnt_train = df_train["SubType"].value_counts().to_dict()
    sub_type_cnt_train.update(df_dev["SubType"].value_counts().to_dict())
    sub_type_cnt_train.update(df_test["SubType"].value_counts().to_dict())
    sub_type_feat_keys = ["unk", "pad"] + list(sub_type_cnt_train.keys())
    sub_type_feat_values = [i for i in range(len(sub_type_feat_keys))]
    assert len(sub_type_feat_keys) == len(sub_type_feat_values)
    sub_type_feat_map = dict(zip(sub_type_feat_keys, sub_type_feat_values))
    # item_id
    news_id_cnt_train = df_train["ID"].value_counts().to_dict()
    news_id_cnt_train.update(df_dev["ID"].value_counts().to_dict())
    news_id_cnt_train.update(df_test["ID"].value_counts().to_dict())
    news_id_feat_keys = ["unk", "pad"] + list(news_id_cnt_train.keys())
    news_id_feat_values = [i for i in range(len(news_id_feat_keys))]
    assert len(news_id_feat_keys) == len(news_id_feat_values)
    news_id_feat_map = dict(zip(news_id_feat_keys, news_id_feat_values))
    return type_feat_map, sub_type_feat_map, news_id_feat_map


class MindDatasetForMiner(Dataset):
    def __init__(self, 
            root: str,
            tokenizer: AutoTokenizer,
            mode: str = 'train',
            split: str = 'small',
            use_glove: str = True,
            glove_cache_dir: str = "",
            use_entity: str = True,
            news_max_len: int = 20,
            hist_max_len: int = 50,
            abs_max_len:int = 60
            ) -> None:
        super(MindDatasetForMiner, self).__init__()
        self.data_path = os.path.join(root, split)
        print("data path:{}".format(self.data_path))
        self._mode = mode
        self._split = split
        self.use_glove = use_glove
        self.glove_cache_dir = glove_cache_dir
        self.use_entity = use_entity
        self.negative_sampling = 4
        
        self._tokenizer = tokenizer
        self._mode = mode
        self._news_max_len = news_max_len
        self._hist_max_len = hist_max_len
        self._abs_max_len = abs_max_len
        # 获取ID特征映射
        self.type_feat_map, self.sub_type_feat_map, self.news_id_feat_map = get_category_feat_map(root=root, split=split)
        # glove vector for category feat
        glove = vocab.GloVe(name='6B', dim=300, cache=self.glove_cache_dir)
        self.word2id_map = glove.stoi

        # 加速版本
        self.feat_map, _ = get_glove_pretrain(root, split, self.glove_cache_dir)

        self._examples = self.get_examples(negative_sampling=4)
        print(self._examples.head())
        self._news = self.process_news()
    
    # train和eval/test生成的df不同
    def get_examples(self, 
            negative_sampling: bool = None
            ) -> Any:
        if self._mode == "infer":
            behavior_file = os.path.join(self.data_path, "test", 'behaviors.tsv')
        else:
            behavior_file = os.path.join(self.data_path, self._mode, 'behaviors.tsv')
        if self._split == 'small':
            df = pd.read_csv(behavior_file, sep='\t', header=None, 
                    names=['user_id', 'time', 'news_history', 'impressions'])
            df['impression_id'] = list(range(len(df)))
        else:
            df = pd.read_csv(behavior_file, sep='\t', header=None, 
                    names=['impression_id', 'user_id', 'time', 'news_history', 'impressions'])
        if self._mode == 'train':
            df = df.dropna(subset=['news_history'])
        df['news_history'] = df['news_history'].fillna('')

        if self._mode == "dev" or self._mode == "test" or self._mode == "infer":
            df = df.drop('impressions', axis=1).join(df['impressions'].str.split(' ', 
                    expand=True).stack().reset_index(level=1, drop=True).rename('impression'))
        if self._mode == 'infer':
            df['news_id'] = df['impression']
            df['click'] = [-1] * len(df)
            return df[:256]
        elif self._mode == 'dev' or self._mode == "test":
            df[['news_id', 'click']] = df['impression'].str.split('-', expand=True)
            df['click'] = df['click'].astype(int)
            return df[:256]
        else:
            # transfer to new df
            data_ls = []
            for _, row in df.iterrows():
                imps = row['impressions']
                pos = []
                neg = []
                for imp in imps.split():
                    if imp[-1] == '1':
                        pos.append(imp)
                    else:
                        neg.append(imp)
                for pos_imp in pos:
                    temp_dic = dict()
                    # 负例够用
                    if negative_sampling <= len(neg):
                        neg_imps = random.sample(neg, negative_sampling)
                        res = [pos_imp] + neg_imps
                        random.shuffle(res)
                        # impression_id', 'user_id', 'time', 'news_history', 'impressions
                        temp_dic['impression_id'] = row['impression_id']
                        temp_dic['user_id'] = row['user_id']
                        temp_dic['time'] = row['time']
                        temp_dic['news_history'] = row['news_history']
                        temp_dic['impression'] = ' '.join(res)
                        data_ls.append(temp_dic)
                    # 负例不够用，丢弃，实验证明对验证集指标有较大提升
                    else:
                        # res = [pos_imp] + neg
                        # random.shuffle(res)
                        # temp_dic['impression_id'] = row['impression_id']
                        # temp_dic['user_id'] = row['user_id']
                        # temp_dic['time'] = row['time']
                        # temp_dic['news_history'] = row['news_history']
                        # temp_dic['impression'] = ' '.join(res)
                        # data_ls.append(temp_dic)
                        break
            df_new = pd.DataFrame(data_ls)
            return df_new[:256]


    def process_news(self) -> Dict[str, Any]:
        filepath = os.path.join(self.data_path, 'news_dict.pkl')
        if os.path.exists(filepath):
            print('Loading news info from', filepath)
            with open(filepath, 'rb') as fin: news = pickle.load(fin)
            return news
        news = dict()
        news = self.read_news(news, os.path.join(self.data_path, 'train'))
        news = self.read_news(news, os.path.join(self.data_path, 'dev'))
        news = self.read_news(news, os.path.join(self.data_path, 'test'))
        print('Saving news info to', filepath)
        with open(filepath, 'wb') as fout: pickle.dump(news, fout)
        return news

    def read_news(self, 
            news: Dict[str, Any], 
            filepath: str,
            drop_stopword: bool = True,
            ) -> Dict[str, Any]:
        with open(os.path.join(filepath, 'news.tsv'), encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            splitted = line.strip('\n').split('\t')
            news_id = splitted[0]
            if news_id in news:
                continue
            title = splitted[3].lower()
            # abstract = splitted[4].lower()

            # 实体特征
            # entity_title_jstr, entity_summry_jstr = splitted[6], splitted[7]
            #entity_title_map = json.loads(entity_title_jstr)
            # entity_summry_map = json.loads(entity_summry_jstr)
            # entity_title_ls = [item["Label"] for item in entity_title_map]
            # entity_summry_ls = [item["Label"] for item in entity_summry_map]
            if drop_stopword:
                title = remove_stopword(title)
                # abstract = remove_stopword(abstract)
            news[news_id] = dict()
            if self.use_entity:
                #title_with_entity = self._tokenizer.sep_token.join([title] + entity_title_ls)
                #news[news_id]['title'] = self._tokenizer.encode(title_with_entity)
                pass
            else:
                news[news_id]['title'] = self._tokenizer.encode(title)
            # news[news_id]['abstract'] = self._tokenizer.encode(abstract)
            # news加入ID特征
            news_type = splitted[1]
            news_sub_type = splitted[2]
            if not self.use_glove:
                news[news_id]['type'] = self.type_feat_map.get(news_type, 0) # unk:0
                news[news_id]['sub_type'] = self.sub_type_feat_map.get(news_sub_type, 0)
            else:
                news[news_id]['type'] = self.word2id_map.get(news_type, 0) # unk:0
                news[news_id]['sub_type'] = self.word2id_map.get(news_sub_type, 0)
        return news

    def collate(self, batch: Dict[str, Any]):
        title_ids = torch.tensor([item['title_ids'] for item in batch])
        title_mask = torch.tensor([item['title_mask'] for item in batch])
        # abs_ids = torch.tensor([item['abs_ids'] for item in batch])
        # abs_mask = torch.tensor([item['abs_mask'] for item in batch])
        title_category = torch.tensor([item['title_category'] for item in batch])
        title_sub_category = torch.tensor([item['title_sub_category'] for item in batch])
        hist_title = torch.tensor([item['hist_title'] for item in batch])
        hist_title_mask = torch.tensor([item['hist_title_mask'] for item in batch])
        # hist_abs =  torch.tensor([item['hist_abs'] for item in batch])
        # hist_abs_mask = torch.tensor([item['hist_abs_mask'] for item in batch])
        hist_mask = torch.tensor([item['hist_mask'] for item in batch])
        hist_category = torch.tensor([item['hist_category'] for item in batch])
        hist_sub_category = torch.tensor([item['hist_sub_category'] for item in batch])
        # user_profile = torch.tensor([item['user_profile'] for item in batch])
        # user_profile_mask = torch.tensor([item['user_profile_mask'] for item in batch])
        candidate_item_ids = torch.tensor([item['candidate_item_ids'] for item in batch])
        hist_item_ids = torch.tensor([item['hist_item_ids'] for item in batch])

        inputs = {'title_ids': title_ids, 
                  'title_mask': title_mask, 

                  'title_category': title_category, 
                  'title_sub_category': title_sub_category,
                  'hist_title': hist_title, 
                  'hist_title_mask': hist_title_mask,

                  "hist_mask": hist_mask,
                  "hist_category": hist_category,
                  "hist_sub_category":hist_sub_category,

                  "candidate_item_ids": candidate_item_ids,
                  "hist_item_ids": hist_item_ids
            }
        if self._mode == 'train':
            inputs['label'] = torch.tensor([item['label'] for item in batch])
            return inputs 
        elif self._mode == 'dev' or self._mode == "test":
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            inputs['label'] = torch.tensor([item['label'] for item in batch])
            return inputs 
        elif self._mode == 'infer':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            return inputs
        else:
            raise ValueError('Mode must be `train`, `dev`, `test` or `infer`.')

    def pack_bert_features(self, example: Any):

        title_ids, title_mask, title_category, title_sub_category= [],[],[],[]
        candidate_item_ids = []

        if self._mode == "dev" or self._mode == "test" or self._mode == "infer":
            candidate_title_ids = self._news[example['news_id']]['title'][:self._news_max_len]
            title_ids.append(candidate_title_ids)
            title_mask.append([1] * len(candidate_title_ids))
            title_category.append(self._news[example['news_id']]['type'])
            title_sub_category.append(self._news[example['news_id']]['sub_type'])
            candidate_item_ids.append(self.news_id_feat_map.get(example['news_id'], 0))
        else:
            impressions = example['impression'].split(' ')
            news_ids = [imp.split('-')[0] for imp in impressions]
            for news_id in news_ids:
                candidate_title_ids = self._news[news_id]['title'][:self._news_max_len]
                title_ids.append(candidate_title_ids)
                title_mask.append([1] * len(candidate_title_ids))
                title_category.append(self._news[news_id]['type'])
                title_sub_category.append(self._news[news_id]['sub_type'])
                candidate_item_ids.append(self.news_id_feat_map.get(news_id, 0))
            if len(impressions) < (self.negative_sampling + 1):
                pad_num = self.negative_sampling + 1 - len(impressions)
                for _ in range(pad_num):
                    title_ids.append([self._tokenizer.pad_token_id] * self._news_max_len)
                    title_mask.append([0] * self._news_max_len)
                    title_category.append(0)
                    title_sub_category.append(0)
                    candidate_item_ids.append(1)

        hist_item_ids = []
        hist_title = []
        hist_title_mask = []
        hist_mask = []
        hist_category = []
        hist_sub_category = []
        for _, ns in enumerate(example['news_history'].split()[-self._hist_max_len:]):
            # hist title
            temp_title_ids = self._news[ns]['title'][:self._news_max_len]
            hist_title.append(temp_title_ids)
            hist_title_mask.append([1] * len(temp_title_ids))
            hist_mask.append(True)
            # hist category feat
            hist_category.append(self._news[ns]['type'])
            hist_sub_category.append(self._news[ns]['sub_type'])

            hist_item_ids.append(self.news_id_feat_map.get(ns, 0))
        

        for i in range(len(title_ids)):
            title_ids[i].extend([self._tokenizer.pad_token_id] * (self._news_max_len - len(title_ids[i])))
            title_mask[i].extend([0] * (self._news_max_len - len(title_mask[i])))

        
        for i in range(len(hist_title)):
            hist_title[i].extend([self._tokenizer.pad_token_id] * (self._news_max_len - len(hist_title[i])))
            hist_title_mask[i].extend([0] * (self._news_max_len - len(hist_title_mask[i])))

        if len(hist_title) < self._hist_max_len:
            for _ in range(self._hist_max_len - len(hist_title)):
                hist_title.append([self._tokenizer.pad_token_id] * self._news_max_len)
                hist_title_mask.append([0] * self._news_max_len)
                hist_category.append(0)
                hist_sub_category.append(0)             
                hist_mask.append(False)
                hist_item_ids.append(1)

        return title_ids, title_mask, title_category, title_sub_category,\
               hist_title, hist_title_mask,\
               hist_mask, hist_category, hist_sub_category,\
               candidate_item_ids, hist_item_ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples.iloc[index]
        title_ids, title_mask, title_category, title_sub_category, hist_title, hist_title_mask,\
        hist_mask, hist_category, hist_sub_category, candidate_item_ids, hist_item_ids = self.pack_bert_features(example)
        inputs = {'title_ids': title_ids, 
                  'title_mask': title_mask, 
                  'title_category': title_category, 
                  'title_sub_category': title_sub_category,
                  'hist_title': hist_title, 
                  'hist_title_mask': hist_title_mask,
                  "hist_mask": hist_mask,
                  "hist_category": hist_category,
                  "hist_sub_category": hist_sub_category,
                  "candidate_item_ids": candidate_item_ids,
                  "hist_item_ids": hist_item_ids
                }
        if self._mode == 'train':
            impressions = example['impression'].split(' ')
            labels = [int(imp.split('-')[1]) for imp in impressions]
            if len(impressions) < (self.negative_sampling + 1):
                labels.extend([0] * (self.negative_sampling + 1 - len(impressions)))
            inputs['label'] = labels
            return inputs 
        elif self._mode == 'dev' or self._mode == "test":
            inputs['impression_id'] = example['impression_id']
            inputs['label'] = example['click']
            return inputs 
        elif self._mode == 'infer':
            inputs['impression_id'] = example['impression_id']
            return inputs 
        else:
            raise ValueError('Mode must be `train`, `dev`, `test` or `infer`')

    def __len__(self) -> int:
        return len(self._examples)

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("/data/gwn/UNBERT/data/bert-base-uncased")
    # # train_set = MindDataset(
    # #     "/data/gwn/UNBERT/data",
    # #     tokenizer=tok,
    # #     mode='train',
    # #     split='small',
    # #     news_max_len=20,
    # #     hist_max_len=20,
    # #     seq_max_len=300
    # # )
    # print(tok.encode("hello world [SEP] world hello"))
    # feat_map, pretrain_weights = get_glove_pretrain("/data/gwn/UNBERT/data", "small")
    # print(feat_map)
    # print(np.shape(pretrain_weights))
    # print(pretrain_weights)

    # train_set = MindDatasetForMinerListWise(
    #     root='/data/gwn/UNBERT/data',
    #     tokenizer=tokenizer,
    #     mode='train',
    #     split='small',
    #     use_glove = True, 
    #     use_entity = True,
    # )
    cache_dir = '/data/gwn/UNBERT/data/glove/glove.6B'
    glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
    word2id_map = glove.stoi
    print(word2id_map.get("the", 1))
