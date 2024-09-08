"""
原始miner实现
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append("../")

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoTokenizer, RobertaConfig, BertConfig, set_seed
from tqdm import tqdm
from time import gmtime, strftime
import argparse
import pandas as pd
import numpy as np

from src.data_loader import MindDatasetForMiner, get_category_feat_map, MindDatasetForMiner
from src.eval import gen_submit_file, evaluation
from src.miner import RobertaNewsEncoder, Miner, NewsEncoder, MinerLoss




class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: str = False,
        num_workers: int = 0
    ) -> None:
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            collate_fn = dataset.collate
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--root', type=str, default='/data/gwn/UNBERT/data')
    parser.add_argument('--glove_cache_dir', type=str, default='')
    parser.add_argument('--split', type=str, default='small')
    parser.add_argument('--news_max_len', type=int, default=20) # with entity set 30 otherwise 20
    parser.add_argument('--abs_max_len', type=int, default=40)
    parser.add_argument('--hist_max_len', type=int, default=50)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--epoch', type=int, default=3) # always set 5 in small dataset, 2 in large
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--backbone', type=str, default="bert")
    parser.add_argument('--warm_ratio', type=float, default=0.1)
    parser.add_argument('--exp_details', type=str, default="list wise train")

    parser.add_argument('--pretrain_model', type=str, default="")

    args = parser.parse_args()
    return args

def main():
    print("use {} gpu for train task".format(torch.cuda.device_count()))
    args = parse_args()
    log_file = os.path.join(args.output, "{}-{}-{}.log".format(
                    args.mode, args.split, strftime('%Y%m%d%H%M%S', gmtime())))
    os.makedirs(args.output, exist_ok=True)
    def printzzz(log):
        with open(log_file, "a") as fout:
            fout.write(log + "\n")
        print(log)
    printzzz(str(args))
    type_feat_map, sub_type_feat_map, news_id_feat_map = get_category_feat_map(args.root, args.split)
    printzzz("[Data] type nums : {}".format(len(type_feat_map)))
    printzzz("[Data] Sub type nums : {}".format(len(sub_type_feat_map)))

    if args.backbone == "bert":
        printzzz("[Setting]choose Bert as backbone")
        config = BertConfig.from_pretrained(args.pretrain_model)
        news_encoder = NewsEncoder.from_pretrained(args.pretrain_model,
                                                config=config,
                                                apply_reduce_dim = False, 
                                                use_sapo = False,
                                                dropout=None,
                                                freeze_transformer = False,
                                                word_embed_dim = None,
                                                combine_type = None, 
                                                lstm_num_layers = None,
                                                lstm_dropout = None)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    else:
        printzzz("[Setting]choose Roberta as backbone")
        config = RobertaConfig.from_pretrained(args.pretrain_model)
        news_encoder = RobertaNewsEncoder.from_pretrained(args.pretrain_model,
                                                config=config,
                                                apply_reduce_dim = False, 
                                                use_sapo = False,
                                                dropout=None,
                                                freeze_transformer = False,
                                                word_embed_dim = None,
                                                combine_type = None, 
                                                lstm_num_layers = None,
                                                lstm_dropout = None)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)

    model = Miner(news_encoder = news_encoder, 
                    use_category_bias = True, 
                    num_context_codes = 32,
                    context_code_dim= 200,
                    score_type = "mean", 
                    dropout = 0.1,
                    num_category = len(type_feat_map), 
                    num_sub_category = len(sub_type_feat_map),
                    category_embed_dim = 128, 
                    use_glove = True,
                    backbone = args.backbone,
                    glove = args.glove_cache_dir)

    if args.mode == "infer":
        printzzz("[Task] Load model for test")
        saved_model_file = os.path.join(args.output, "best_model.bin")
        printzzz("restore model from {}".format(saved_model_file))
        state_dict = torch.load(saved_model_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if args.mode == "train":
        printzzz('reading training data...')
        train_set = MindDatasetForMiner(
            args.root,
            tokenizer=tokenizer,
            mode='train',
            split=args.split,
            use_glove = True,
            glove_cache_dir = args.glove_cache_dir,
            use_entity = False,
            news_max_len=args.news_max_len,
            hist_max_len=args.hist_max_len,
            abs_max_len = args.abs_max_len,
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
        )

        printzzz('reading dev data...')
        dev_set = MindDatasetForMiner(
            args.root,
            tokenizer=tokenizer,
            mode='dev',
            split=args.split,
            use_glove = True,
            glove_cache_dir = args.glove_cache_dir,
            use_entity = False,
            news_max_len=args.news_max_len,
            hist_max_len=args.hist_max_len,
            abs_max_len = args.abs_max_len,
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=8
        )

        printzzz('reading test data...')
        test_set = MindDatasetForMiner(
            args.root,
            tokenizer=tokenizer,
            mode='test',
            split=args.split,
            use_glove = True,
            glove_cache_dir = args.glove_cache_dir,
            use_entity = True,
            news_max_len=args.news_max_len,
            hist_max_len=args.hist_max_len,
            abs_max_len = args.abs_max_len,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )

        loss_fn = MinerLoss()
        if torch.cuda.device_count() > 1:
            loss_fn = nn.DataParallel(loss_fn)
        loss_fn.to(device)

        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        num_training_steps = len(train_set)*args.epoch//args.batch_size
        m_scheduler = get_linear_schedule_with_warmup(
                        m_optim, 
                        num_warmup_steps=num_training_steps * args.warm_ratio, 
                        num_training_steps=num_training_steps,
                    )

        printzzz("start training...")
        best_auc = 0.0
        for epoch in range(args.epoch):
            batch_iterator = tqdm(train_loader, disable=False)
            avg_loss = 0.0
            for step, train_batch in enumerate(batch_iterator):
                multi_user_interest, matching_scores = model(
                    title = train_batch['title_ids'].to(device), 
                    title_mask = train_batch['title_mask'].to(device), 
                    his_title = train_batch['hist_title'].to(device),
                    his_title_mask = train_batch['hist_title_mask'].to(device),
                    his_mask = train_batch['hist_mask'].to(device),
                    category = train_batch['title_category'].to(device),
                    his_category = train_batch['hist_category'].to(device),
                    # sub_category = train_batch['title_sub_category'].to(device),
                    # his_sub_category = train_batch['hist_sub_category'].to(device)
                )
                batch_loss = loss_fn(multi_intr = multi_user_interest, 
                                     match_score = matching_scores,
                                     labels = train_batch['label'].to(device))
        
                if torch.cuda.device_count() > 1:
                    batch_loss = batch_loss.mean()
                if step % 100 == 0:
                    printzzz("epoch:{} | step:{} | loss: {}".format(epoch + 1, step, batch_loss))
                avg_loss += batch_loss.item()
                batch_loss.backward()
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()
            printzzz("********* Epoch {} : Avg Loss {} *********** ".format(epoch, avg_loss / len(batch_iterator)))
            auc,mrr,ndcg5,ndcg10 = evaluation(model, dev_loader, device, args.output, is_epoch=True)
            printzzz("eval_auc:{}\neval_mrr:{}\neval_ndcg@5:{}\neval_ndcg@10:{}".format(auc,mrr,ndcg5,ndcg10))
            if best_auc < auc:
                model_save_path = os.path.join(args.output, "best_model.bin")
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)
                printzzz("********* Auc on dev get better in {} epoch | best auc: {} -> {} *********** ".format(epoch + 1, best_auc, auc))
                best_auc = auc
            else:
                printzzz("********* Auc on dev dont get better in {} epoch *********** ".format(epoch + 1))
        printzzz("train success!")
        auc,mrr,ndcg5,ndcg10 = evaluation(model, test_loader, device, args.output, is_epoch=True)
        printzzz("test auc:{}\ntest_mrr:{}\ntest_ndcg@5:{}\ntest_ndcg@10:{}".format(auc,mrr,ndcg5,ndcg10))
        printzzz("train task done exit success")

    elif args.mode == "infer":
        printzzz('*** start generating inference result for submit ***')
        test_set = MindDatasetForMiner(
            args.root,
            tokenizer=tokenizer,
            mode='infer',
            split=args.split,
            use_glove = True,
            glove_cache_dir = args.glove_cache_dir,
            use_entity = False,
            news_max_len=args.news_max_len,
            hist_max_len=args.hist_max_len,
            abs_max_len = args.abs_max_len,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )
        gen_submit_file(model, test_loader, device, args.output)
        printzzz("inference task done exit success")

if __name__ == "__main__":
    main()
