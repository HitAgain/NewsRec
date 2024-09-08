# coding=utf-8

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
import torch.nn.functional as torch_f
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from time import gmtime, strftime
import argparse
import pandas as pd
import numpy as np
import os

def func(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    return roc_auc_score(grouped_df["label"], grouped_df["score"])

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def metric(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    label = grouped_df["label"].values.tolist()
    t_score_group = torch.tensor(grouped_df["score"].values.tolist())
    probs = torch_f.softmax(t_score_group, dim=0).tolist()
    auc = roc_auc_score(label, probs)
    mrr = mrr_score(label, probs)
    ndcg5 = ndcg_score(label, probs, 5)
    ndcg10 = ndcg_score(label, probs, 10)
    return auc, mrr, ndcg5, ndcg10

def evaluation(model, dev_loader, device, out_path, is_epoch=False):
    impression_ids = []
    labels = []
    scores = []
    batch_iterator = tqdm(dev_loader, disable=False)
    for step, dev_batch in enumerate(batch_iterator):
        if not is_epoch and step >= 5000:
            break
        impression_id, label = dev_batch['impression_id'], dev_batch['label']
        with torch.no_grad():
            _, matching_scores = model(
                                    title = dev_batch['title_ids'].to(device), 
                                    title_mask = dev_batch['title_mask'].to(device), 
                                    his_title = dev_batch['hist_title'].to(device),
                                    his_title_mask = dev_batch['hist_title_mask'].to(device),
                                    his_mask = dev_batch['hist_mask'].to(device),
                                    category = dev_batch['title_category'].to(device),
                                    his_category = dev_batch['hist_category'].to(device),
                                    # sub_category = dev_batch['title_sub_category'].to(device),
                                    # his_sub_category = dev_batch['hist_sub_category'].to(device),
                                )

            batch_score = matching_scores.squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            if not isinstance(batch_score, list):
                batch_score = [batch_score]
            impression_ids.extend(impression_id)
            labels.extend(label.tolist())
            scores.extend(batch_score)

    score_path = os.path.join(out_path, "dev_score.tsv")
    EVAL_DF = pd.DataFrame()
    EVAL_DF["impression_id"] = impression_ids
    EVAL_DF["label"] = labels
    EVAL_DF["score"] = scores
    EVAL_DF.to_csv(score_path, sep="\t", index=False)
    groups_iter = EVAL_DF.groupby("impression_id")
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    res = pool.map(metric, df_groups)
    aucs = [_[0] for _ in res] 
    mrrs = [_[1] for _ in res] 
    ndcg5s = [_[2] for _ in res] 
    ndcg10s = [_[3] for _ in res]
    pool.close()
    pool.join()
    auc = np.mean(aucs)
    mrr = np.mean(mrrs)
    ndcg5 = np.mean(ndcg5s)
    ndcg10 = np.mean(ndcg10s)
    return auc, mrr, ndcg5, ndcg10


def rank(scores):
    tmp = [(i+1, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank_map = dict()
    for i in range(len(scores)):
        rank_map[tmp[i][0]] = i + 1
    rank = [str(rank_map[i+1]) for i in range(len(scores))]
    rank = "[" + ",".join(rank) + "]"
    return rank

def rank_func(x):
    scores = x["score"].tolist()
    # tmp = [(i, s) for i, s in enumerate(scores)]
    # tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    # rank = [(i+1, t[0]) for i, t in enumerate(tmp)]
    # rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    # rank = "[" + ",".join(rank) + "]"
    rank_res = rank(scores)
    return {"imp": x["impression_id"].tolist()[0], "rank": rank_res}
    

# 输出提交预测文件函数
def gen_submit_file(model, test_loader, device, out_path):
    score_path = os.path.join(out_path, "test_score.tsv")
    outfile = os.path.join(out_path, "prediction.txt")
    impression_ids = []
    scores = []
    batch_iterator = tqdm(test_loader, disable=False)
    for _, test_batch in enumerate(batch_iterator):
        impression_id = test_batch['impression_id']
        with torch.no_grad():
            _, matching_scores = model(title = test_batch['title_ids'].to(device), 
                                title_mask = test_batch['title_mask'].to(device), 
                                his_title = test_batch['hist_title'].to(device),
                                his_title_mask = test_batch['hist_title_mask'].to(device),
                                his_mask = test_batch['hist_mask'].to(device),
                                category = test_batch['title_category'].to(device),
                                his_category = test_batch['hist_category'].to(device),
                                # sub_category = test_batch['title_sub_category'].to(device),
                                # his_sub_category = test_batch['hist_sub_category'].to(device)
                            )
            batch_score = matching_scores.squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            if not isinstance(batch_score, list):
                batch_score = [batch_score]
            impression_ids.extend(impression_id)
            scores.extend(batch_score)
    # 按照group_id分组
    EVAL_DF = pd.DataFrame()
    EVAL_DF["impression_id"] = impression_ids
    EVAL_DF["score"] = scores
    EVAL_DF.to_csv(score_path, sep="\t", index=False)
    groups_iter = EVAL_DF.groupby("impression_id")
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    result = pool.map(rank_func, df_groups)
    pool.close()
    pool.join()
    imps = [r["imp"] for r in result]
    ranks = [r["rank"] for r in result]
    with open(outfile, "w") as fout:
        out = [str(imp) + " " + rank for imp, rank in zip(imps, ranks)]
        for out_item in out:
            fout.write("{}\n".format(out_item))
    return

if __name__ == "__main__":
    rank([0.8, 0.7, 0.9, 0.5])
    print("expected res:[2,3,1,4]")
