# coding=utf-8

import logging
import math
import os
import warnings
from typing import Union

import torch
import torch.utils.checkpoint
import torchtext.vocab as vocab
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch import Tensor
import torch.nn.functional as torch_f
import numpy as np

from .modeling_bert import BertEncoder, BertPooler, BertEmbeddings
from transformers import RobertaModel, RobertaPreTrainedModel, BertModel, BertPreTrainedModel


BertLayerNorm = torch.nn.LayerNorm

def pairwise_cosine_similarity(x, y, zero_diagonal: bool = False):
    r"""
    Calculates the pairwise cosine similarity matrix

    Args:
        x: tensor of shape ``(batch_size, M, d)``
        y: tensor of shape ``(batch_size, N, d)``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    Returns:
        A tensor of shape ``(batch_size, M, N)``
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    distance = torch.matmul(torch.div(x, x_norm), torch.div(y, y_norm).permute(0, 2, 1))
    if zero_diagonal:
        assert x.shape[1] == y.shape[1]
        mask = torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1).bool().to(distance.device)
        distance.masked_fill_(mask, 0)
    return distance

class DisagreeLoss(nn.Module):
    def __init__(self):
        super(DisagreeLoss, self).__init__()

    def forward(self, inputs):
        loss_value = pairwise_cosine_similarity(inputs, inputs, zero_diagonal=True).mean()
        return loss_value
    

class MinerLoss(nn.Module):
    def __init__(self, beta=0.8):
        super(MinerLoss, self).__init__()
        self.beta = beta
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, multi_intr, match_score, labels):
        disagree_loss = pairwise_cosine_similarity(multi_intr, multi_intr, zero_diagonal=True).mean()
        targets = labels.argmax(dim=1)
        rank_loss = self.criterion(match_score, targets)
        total_loss = rank_loss + self.beta * disagree_loss
        return total_loss

class NewsEncoder(BertPreTrainedModel):
    def __init__(self, config, apply_reduce_dim: bool=True, use_sapo: bool=True, dropout: float=0.2,
                 freeze_transformer: bool=False, word_embed_dim: Union[int, None] = 256,
                 combine_type: Union[str, None] = "linear", lstm_num_layers: Union[int, None] = None,
                 lstm_dropout: Union[float, None] = None):
        r"""
        Initialization

        Args:
            config: the configuration of a ``RobertaModel``.
            apply_reduce_dim: whether to reduce the dimension of Roberta's embedding or not.
            use_sapo: whether to use sapo embedding or not.
            dropout: dropout value.
            freeze_transformer: whether to freeze Roberta weight or not.
            word_embed_dim: size of each word embedding vector if ``apply_reduce_dim``.
            combine_type: method to combine news information.
            lstm_num_layers: number of recurrent layers in LSTM.
            lstm_dropout: dropout value in LSTM.
        """
        super().__init__(config)
        self.bert = BertModel(config)
        if freeze_transformer:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.apply_reduce_dim = apply_reduce_dim
        # bert repr pca
        if self.apply_reduce_dim:
            self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
            #self.word_embed_dropout = nn.Dropout(dropout)
            self._embed_dim = word_embed_dim
        else:
            self._embed_dim = config.hidden_size
        self.init_weights()

    def forward(self, title_encoding, title_attn_mask, sapo_encoding=None, sapo_attn_mask=None):
        r"""
        Forward propagation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        title_word_embed = self.bert(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
        # cls mode
        # title_repr = title_word_embed[:, 0, :]
        # average mode
        emb_mask = title_attn_mask.unsqueeze(dim = -1)
        title_repr = title_word_embed * emb_mask
        title_repr = torch.mean(title_word_embed, dim=1)
        # max pooling mode
        # title_repr = torch.max(title_word_embed, dim=1, keepdim=False)[0]
        if self.apply_reduce_dim:
            title_repr = self.reduce_dim(title_repr)
            #title_repr = self.word_embed_dropout(title_repr)
        return title_repr

    @property
    def embed_dim(self):
        return self._embed_dim
    
class RobertaNewsEncoder(RobertaPreTrainedModel):
    def __init__(self, config, apply_reduce_dim: bool=True, use_sapo: bool=True, dropout: float=0.2,
                 freeze_transformer: bool=False, word_embed_dim: Union[int, None] = 256,
                 combine_type: Union[str, None] = "linear", lstm_num_layers: Union[int, None] = None,
                 lstm_dropout: Union[float, None] = None):
        r"""
        Initialization

        Args:
            config: the configuration of a ``RobertaModel``.
            apply_reduce_dim: whether to reduce the dimension of Roberta's embedding or not.
            use_sapo: whether to use sapo embedding or not.
            dropout: dropout value.
            freeze_transformer: whether to freeze Roberta weight or not.
            word_embed_dim: size of each word embedding vector if ``apply_reduce_dim``.
            combine_type: method to combine news information.
            lstm_num_layers: number of recurrent layers in LSTM.
            lstm_dropout: dropout value in LSTM.
        """
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if freeze_transformer:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.apply_reduce_dim = apply_reduce_dim
        # emb pca config
        if self.apply_reduce_dim:
            assert word_embed_dim is not None
            self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
            self.word_embed_dropout = nn.Dropout(dropout)
            self._embed_dim = word_embed_dim
        else:
            self._embed_dim = config.hidden_size
        # fussion news type emb config
        self.use_sapo = use_sapo
        if self.use_sapo:
            assert combine_type is not None
            self.combine_type = combine_type
            if self.combine_type == 'linear':
                self.linear_combine = nn.Linear(in_features=self._embed_dim * 2, out_features=self._embed_dim)
            elif self.combine_type == 'lstm':
                self.lstm = nn.LSTM(input_size=self._embed_dim * 2, hidden_size=self._embed_dim // 2,
                                    num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout,
                                    bidirectional=True)
                self._embed_dim = (self._embed_dim // 2) * 2
        self.init_weights()

    def forward(self, title_encoding, title_attn_mask, sapo_encoding, sapo_attn_mask):
        r"""
        Forward propagation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        # Title encoder
        #news_info = []
        #title_word_embed = self.roberta(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
        #title_repr = title_word_embed[:, 0, :]
        # if self.apply_reduce_dim:
        #     title_repr = self.reduce_dim(title_repr)
        #     title_repr = self.word_embed_dropout(title_repr)
        # news_info.append(title_repr)

        # # # Sapo encoder
        # if self.use_sapo:
        #     sapo_word_embed = self.bert(input_ids=sapo_encoding, attention_mask=sapo_attn_mask)[0]
        #     sapo_repr = sapo_word_embed[:, 0, :]
        #     if self.apply_reduce_dim:
        #         sapo_repr = self.reduce_dim(sapo_repr)
        #         sapo_repr = self.word_embed_dropout(sapo_repr)
        #     news_info.append(sapo_repr)

        #     if self.combine_type == 'linear':
        #         news_info = torch.cat(news_info, dim=1)

        #         return self.linear_combine(news_info)
        #     elif self.combine_type == 'lstm':
        #         news_info = torch.cat(news_info, dim=1)
        #         news_repr, _ = self.lstm(news_info)

        #         return news_repr
        # else:
        #return title_repr

        title_word_embed = self.roberta(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
        # cls mode
        # title_repr = title_word_embed[:, 0, :]

        # average mode
        emb_mask = title_attn_mask.unsqueeze(dim = -1)
        title_repr = title_word_embed * emb_mask
        title_repr = torch.mean(title_word_embed, dim=1)
        # average mode
        # title_repr = torch.max(title_word_embed, dim=1, keepdim=False)[0]
        if self.apply_reduce_dim:
            title_repr = self.reduce_dim(title_repr)
            #title_repr = self.word_embed_dropout(title_repr)
        return title_repr

    @property
    def embed_dim(self):
        return self._embed_dim

class MinerPointWise(nn.Module):

    def __init__(self, news_encoder, use_category_bias, num_context_codes,
                 context_code_dim, score_type, dropout, num_category,
                 category_embed_dim):
        r"""
        Initialization

        Args:
            news_encoder: NewsEncoder object.
            use_category_bias: whether to use Category-aware attention weighting.
            num_context_codes: the number of attention vectors ``K``.
            context_code_dim: the number of features in a context code.
            score_type: the ways to aggregate the ``K`` matching scores as a final user click score ('max', 'mean' or
                'weighted').
            dropout: dropout value.
            num_category: the size of the dictionary of categories.
            category_embed_dim: the size of each category embedding vector.
            category_pad_token_id: ID of the padding token type in the category vocabulary.
            category_embed: pre-trained category embedding.
        """
        super().__init__()
        self.news_encoder = news_encoder
        self.news_embed_dim = self.news_encoder.embed_dim
        self.use_category_bias = use_category_bias

        # category emb
        if self.use_category_bias:
            self.category_dropout = nn.Dropout(dropout)
            self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=category_embed_dim)
        # interest emb
        self.poly_attn = PolyAttention(in_embed_dim=self.news_embed_dim, num_context_codes=num_context_codes,
                                       context_code_dim=context_code_dim)
        self.score_type = score_type
        if self.score_type == 'weighted':
            self.target_aware_attn = TargetAwareAttention(self.news_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(self.news_embed_dim * 2, 2)

    def forward(self, title: Tensor, title_mask: Tensor, his_title: Tensor, his_title_mask: Tensor,
                his_mask: Tensor, sapo: Union[Tensor, None] = None, sapo_mask: Union[Tensor, None] = None,
                his_sapo: Union[Tensor, None] = None, his_sapo_mask: Union[Tensor, None] = None,
                category: Union[Tensor, None] = None, his_category: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            !! num_candidates = 1 when point wise !!
            title: tensor of shape ``(batch_size, num_candidates, title_length)``.
            title_mask: tensor of shape ``(batch_size, num_candidates, title_length)``.
            his_title: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_title_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            sapo_mask: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            his_sapo: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            his_sapo_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category: tensor of shape ``(batch_size, num_candidates)``.
            his_category: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            tuple
                - multi_user_interest: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
                - matching_scores: tensor of shape ``(batch_size, num_candidates)``
        """
        batch_size = title.shape[0]
        num_candidates = title.shape[1]
        his_length = his_title.shape[1]

        # Representation of candidate news
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        # sapo = sapo.view(batch_size * num_candidates, -1)
        # sapo_mask = sapo_mask.view(batch_size * num_candidates, -1)

        candidate_repr = self.news_encoder(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,
                                           sapo_attn_mask=sapo_mask)
        # print("========= Miner Out candidate_repr===========")
        # print(candidate_repr)
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)

        # Representation of history clicked news
        his_title = his_title.view(batch_size * his_length, -1)
        his_title_mask = his_title_mask.view(batch_size * his_length, -1)
        # his_sapo = his_sapo.view(batch_size * his_length, -1)
        # his_sapo_mask = his_sapo_mask.view(batch_size * his_length, -1)

        history_repr = self.news_encoder(title_encoding=his_title, title_attn_mask=his_title_mask,
                                         sapo_encoding=his_sapo, sapo_attn_mask=his_sapo_mask)
        history_repr = history_repr.view(batch_size, his_length, -1)

        # category bias add
        if self.use_category_bias:
            his_category_embed = self.category_embedding(his_category)
            his_category_embed = self.category_dropout(his_category_embed)
            candidate_category_embed = self.category_embedding(category)
            candidate_category_embed = self.category_dropout(candidate_category_embed)

            # (bz, click_num, 1)
            # 获得历史交互文档在候选文档上的类别注意力分布 = [bz, his_nums, candidate_nums]
            category_bias = pairwise_cosine_similarity(his_category_embed, candidate_category_embed)

            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=category_bias)
        else:
            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=None)

        # Click predictor(list wise)
        # matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        # if self.score_type == 'max':
        #     matching_scores = matching_scores.max(dim=2)[0]
        # elif self.score_type == 'mean':
        #     matching_scores = matching_scores.mean(dim=2)
        # elif self.score_type == 'weighted':
        #     matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
        #                                              value=matching_scores)
        # else:
        #     raise ValueError('Invalid method of aggregating matching score')
        # return matching_scores

        # Click predictor(Pointwise)
        # (bz, 1, num_context_codes)
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        #(bz, 1, num_context_codes)
        weights = torch_f.softmax(matching_scores, dim=-1)
        # (bz, emb_dim)
        weighted_interest_repr = torch.matmul(weights, multi_user_interest).squeeze(dim = 1)
        #(bz, emb_dim)
        candidate_repr = candidate_repr.squeeze(dim = 1)
        hidden = torch.cat([weighted_interest_repr, candidate_repr], -1)
        score = self.score(hidden)
        return score

class PolyAttention(nn.Module):
    r"""
    Implementation of Poly attention scheme that extracts `K` attention vectors through `K` additive attentions
    """
    def __init__(self, in_embed_dim: int, num_context_codes: int, context_code_dim: int):
        r"""
        Initialization

        Args:
            in_embed_dim: The number of expected features in the input ``embeddings``
            num_context_codes: The number of attention vectors ``K``
            context_code_dim: The number of features in a context code
        """
        super().__init__()
        self.λ = nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True)
        self.linear = nn.Linear(in_features=in_embed_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_context_codes, context_code_dim),
                                                                  gain=nn.init.calculate_gain('tanh')))

    def forward(self, embeddings: Tensor, attn_mask: Tensor, bias: Tensor = None):
        r"""
        Forward propagation

        Args:
            embeddings: tensor of shape ``(batch_size, his_length, embed_dim)``
            attn_mask: tensor of shape ``(batch_size, his_length)``
            bias: tensor of shape ``(batch_size, his_nums, candidate_nums)``

        Returns:
            A tensor of shape ``(batch_size, num_context_codes, embed_dim)``
        """

        # [bz, his_nums, context_dim]
        proj = torch.tanh(self.linear(embeddings))
        if bias is None:
            weights = torch.matmul(proj, self.context_codes.T)
        else:
            # [bz, hist_nums, 1]
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            # [bz, his_nums, k]  = [bz, his_nums, k] + [bz, hist_nums, 1]
            # 每个历史文档的兴趣点分布
            weights = torch.matmul(proj, self.context_codes.T) + self.λ  * bias
        # [bz, k, hist_nums]
        # 每个兴趣点的历史文档注意力分布
        weights = weights.permute(0, 2, 1)
        # mask 掉padding hist的注意力值
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        # 规范化
        weights = torch_f.softmax(weights, dim=2)
        # self-atten 得到基于交互历史在K个兴趣点上的注意力分布对交互历史表征加权和的多兴趣表征
        #  [bz, k, hist_repr_dim]     = [bz, k, hist_nums] * [bz, hist_nums, hist_rper_dim]
        poly_repr = torch.matmul(weights, embeddings)

        return poly_repr

# class PolyAttention(nn.Module):
#     r"""
#     Implementation of Poly attention scheme that extracts `K` attention vectors through `K` additive attentions
#     """
#     def __init__(self, in_embed_dim: int, num_context_codes: int, context_code_dim: int):
#         r"""
#         Initialization

#         Args:
#             in_embed_dim: The number of expected features in the input ``embeddings``
#             num_context_codes: The number of attention vectors ``K``
#             context_code_dim: The number of features in a context code
#         """
#         super().__init__()
#         self.num_context_codes = num_context_codes
#         self.λ = nn.Parameter(torch.FloatTensor(0.25), requires_grad=True)
#         self.linear = nn.Linear(in_features=in_embed_dim, out_features=context_code_dim, bias=False)
#         self.context_codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_context_codes, context_code_dim),
#                                                                   gain=nn.init.calculate_gain('tanh')))

#     def forward(self, embeddings: Tensor, attn_mask: Tensor, bias: Tensor = None):
#         r"""
#         Forward propagation

#         Args:
#             embeddings: tensor of shape ``(batch_size, his_length, embed_dim)``
#             attn_mask: tensor of shape ``(batch_size, his_length)``
#             bias: tensor of shape ``(batch_size, his_nums, candidate_nums)``

#         Returns:
#             A tensor of shape ``(batch_size, num_context_codes, embed_dim)``
#         """

#         # [bz, his_nums, context_dim]
#         proj = torch.tanh(self.linear(embeddings))
#         weights = torch.matmul(proj, self.context_codes.T)
#         # [bz, k, hist_nums]
#         # 每个兴趣点的历史文档注意力分布
#         weights = weights.permute(0, 2, 1)
#         # mask 掉padding hist的注意力值
#         weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
#         # (bz, k, hist_num)
#         weights = torch_f.softmax(weights, dim=2)
#         if bias:
#             # (bz, 1, hist_num)
#             bias = bias.mean(dim=2).unsqueeze(dim=1).repeat(1, self.num_context_codes, 1)
#             weights = weights + self.λ * bias
#             weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)

#         # self-atten 得到基于交互历史在K个兴趣点上的注意力分布对交互历史表征加权和的多兴趣表征
#         #  [bz, k, hist_repr_dim]     = [bz, k, hist_nums] * [bz, hist_nums, hist_rper_dim]
#         poly_repr = torch.matmul(weights, embeddings)

#         return poly_repr


class TargetAwareAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, embed_dim: int):
        r"""
        Initialization

        Args:
            embed_dim: The number of features in query and key vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        r"""
        Forward propagation

        Args:
            query: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
            key: tensor of shape ``(batch_size, num_candidates, embed_dim)``
            value: tensor of shape ``(batch_size, num_candidates, num_context_codes)``

        Returns:
            tensor of shape ``(batch_size, num_candidates)``
        """
        proj = torch_f.gelu(self.linear(query))
        # (bz, num_candidates,)
        weights = torch_f.softmax(torch.matmul(key, proj.permute(0, 2, 1)), dim=2)
        outputs = torch.mul(weights, value).sum(dim=2)

        return outputs

class Miner(nn.Module):
        
    """
        Implementation of Multi-interest matching network for news recommendation. Please see the paper in
        https://aclanthology.org/2022.findings-acl.29.pdf.
    """

    def __init__(self, news_encoder, use_category_bias, num_context_codes,
                 context_code_dim, score_type, dropout, num_category, num_sub_category,
                 category_embed_dim, use_glove, backbone, glove):
        super().__init__()
        self.news_encoder = news_encoder
        self.news_embed_dim = self.news_encoder.embed_dim
        self.use_category_bias = use_category_bias
        self.use_glove = use_glove
        self.backbone = backbone
        self.glove = glove
        # category emb
        if self.use_category_bias:
            if self.use_glove:
                glove = vocab.GloVe(name='6B', dim=300, cache=self.glove)
                self.category_embedding = nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
                self.category_embedding.weight.data.copy_(glove.vectors)
            else:
                self.category_dropout = nn.Dropout(dropout)
                self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=category_embed_dim)
                self.sub_category_embedding = nn.Embedding(num_embeddings=num_sub_category, embedding_dim=category_embed_dim)
        # interest emb
        self.poly_attn = PolyAttention(in_embed_dim=self.news_embed_dim, num_context_codes=num_context_codes,
                                       context_code_dim=context_code_dim)
        self.score_type = score_type
        if self.score_type == 'weighted':
            self.target_aware_attn = TargetAwareAttention(self.news_embed_dim)

    def forward(self, title: Tensor = None, title_mask: Tensor = None, 
                his_title: Tensor = None, his_title_mask: Tensor = None,
                his_mask: Tensor = None, sapo: Union[Tensor, None] = None, sapo_mask: Union[Tensor, None] = None,
                his_sapo: Union[Tensor, None] = None, his_sapo_mask: Union[Tensor, None] = None,
                category: Union[Tensor, None] = None, his_category: Union[Tensor, None] = None, 
                sub_category:Union[Tensor, None] = None, his_sub_category:Union[Tensor, None] = None):

        batch_size = title.shape[0]
        num_candidates = title.shape[1]
        his_length = his_title.shape[1]

        # Representation of candidate news
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        candidate_repr = self.news_encoder(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,
                                           sapo_attn_mask=sapo_mask)
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)

        # Representation of history clicked news
        his_title = his_title.view(batch_size * his_length, -1)
        his_title_mask = his_title_mask.view(batch_size * his_length, -1)
        history_repr = self.news_encoder(title_encoding=his_title, title_attn_mask=his_title_mask,
                                         sapo_encoding=his_sapo, sapo_attn_mask=his_sapo_mask)
        history_repr = history_repr.view(batch_size, his_length, -1)

        if self.use_category_bias:
            his_category_embed = self.category_embedding(his_category)
            #his_category_embed = self.category_dropout(his_category_embed)
            candidate_category_embed = self.category_embedding(category)
            #candidate_category_embed = self.category_dropout(candidate_category_embed)
            # his_sub_category_embed = self.category_embedding(his_sub_category)
            # candidate_sub_category_embed = self.category_embedding(sub_category)
            # his_category_embed_concat = torch.cat([his_category_embed, his_sub_category_embed], -1)
            # candidate_category_embed_concat = torch.cat([candidate_category_embed, candidate_sub_category_embed], -1)
    
            # (bz, click_num, candidate_num)
            category_bias = pairwise_cosine_similarity(his_category_embed, candidate_category_embed)
            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=category_bias)
        else:
            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=None)
        # (bz, candidate_num, num_context_codes)
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if self.score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif self.score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        # 各维度兴趣值加权和
        elif self.score_type == 'weighted':
            matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
                                                     value=matching_scores)
        else:
            raise ValueError('Invalid method of aggregating matching score')
        # (bz, num_context_codes, hidden_size) (bz, candidate_nums)
        return multi_user_interest, matching_scores

if __name__ == "__main__":
    # from .configuration_bert import BertConfig
    # print("test load news encoder")
    # config = BertConfig.from_pretrained("/data/gwn/UNBERT/data/bert-base-uncased")
    # miner_model = NewsEncoder.from_pretrained("/data/gwn/UNBERT/data/bert-base-uncased", config=config)
    # print("success")
    pass
