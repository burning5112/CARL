from audioop import reverse
from wsgiref import headers
from xml.dom.minidom import Element
from data import *
import copy
import re
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from scipy import sparse
from collections import defaultdict
import argparse
from utils import *
import gc
import os
import sys
import io
from main import get_model_file

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

head2Mean_rank_mrr = defaultdict(list)
head2Mean_rank_hit_1 = defaultdict(list)
head2Mean_rank_hit_10 = defaultdict(list)
head2Top_rank_mrr = defaultdict(list)
head2Top_rank_hit_1 = defaultdict(list)
head2Top_rank_hit_10 = defaultdict(list)
Mean_rank = defaultdict(list)
Top_rank = defaultdict(list)


class RuleDataset(Dataset):
    def __init__(self, r2mat, rules, e_num, idx2rel, args):
        self.e_num = e_num
        self.r2mat = r2mat
        self.rules = rules
        self.idx2rel = idx2rel
        self.len = len(self.rules)
        self.args = args

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rel = self.idx2rel[idx]
        _rules = self.rules[rel]

        path_count = sparse.dok_matrix((self.e_num, self.e_num))
        for rule in _rules:
            head, body, conf_1, conf_2 = rule
            if conf_1 >= self.args.threshold:
                body_adj = sparse.eye(self.e_num)
                for b_rel in body:
                    body_adj = body_adj * self.r2mat[b_rel]

                body_adj = body_adj * conf_1
                path_count += body_adj
                del body_adj

        return rel, path_count

    @staticmethod
    def collate_fn(data):
        head = [_[0] for _ in data]
        path_count1 = [_[1] for _ in data]
        return head, path_count1


def sum_from_n_to_len(n_non_zero, filtered_pred):
    total = 0

    for i in range(n_non_zero + 1, len(filtered_pred) + 1):
        total += i

    return total


def sortSparseMatrix(m, r, rev=True, only_indices=False):
    d = m.getrow(r)
    s = zip(d.indices, d.data)
    sorted_s = sorted(s, key=lambda v: v[1], reverse=rev)
    if only_indices:
        res = [element[0] for element in sorted_s]
    else:
        res = sorted_s
    return res


def remove_var(r):
    r = re.sub(r"\(\D?, \D?\)", "", r)
    return r


def parse_rule(r):
    r = remove_var(r)
    head, body = r.split(" <-- ")
    body = body.split(", ")
    return head, body


def load_rules(rule_path, all_rules, all_heads):
    with open(rule_path, 'r') as f:
        rules = f.readlines()
        for i_, rule in enumerate(rules):
            conf, r = rule.strip('\n').split('\t')
            conf_1, conf_2 = float(conf[0:5]), float(conf[-6:-1])
            head, body = parse_rule(r)

            if head not in all_rules:
                all_rules[head] = []
            all_rules[head].append((head, body, conf_1, conf_2))

            if head not in all_heads:
                all_heads.append(head)


def construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf):
    e_num = len(idx2ent)
    r2mat = {}
    for idx, rel in idx2rel.items():
        mat = sparse.dok_matrix((e_num, e_num))
        r2mat[rel] = mat

    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        h_idx, t_idx = ent2idx[h], ent2idx[t]
        r2mat[r][h_idx, t_idx] = 1
    return r2mat


def get_gt(dataset):
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    gt = defaultdict(list)
    all_rdf = fact_rdf + train_rdf + valid_rdf + test_rdf
    for rdf in all_rdf:
        h, r, t = parse_rdf(rdf)
        gt[(h, r)].append(ent2idx[t])
    return gt


def kg_completion(rules, dataset, args):
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf

    gt = get_gt(dataset)

    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    rel2idx, idx2rel = rdict.rel2idx, rdict.idx2rel

    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    e_num = len(idx2ent)

    r2mat = construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf + train_rdf + valid_rdf)

    body2mat = {}

    rule_dataset = RuleDataset(r2mat, rules, e_num, idx2rel, args)
    rule_loader = DataLoader(
        rule_dataset,
        batch_size=args.data_batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),

        collate_fn=RuleDataset.collate_fn,
    )

    for epoch, sample in enumerate(rule_loader):
        heads, score_counts = sample
        for idx in range(len(heads)):
            head = heads[idx]
            score_count = score_counts[idx]
            body2mat[head] = score_count

    for key, value in body2mat.items():
        array_matrix = value.todense()
        body2mat[key] = array_matrix

    for q_i, query_rdf in enumerate(test_rdf):
        query = parse_rdf(query_rdf)
        q_h, q_r, q_t = query
        if q_r not in body2mat:
            continue
        print("{}\t{}\t{}".format(q_h, q_r, q_t))

        pred = np.squeeze(np.array(body2mat[q_r][ent2idx[q_h]]))
        pred_ranks = np.argsort(pred)[::-1]

        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t != ent2idx[q_t]]

        L = []
        H = []

        top_rank = 0
        mean_rank = 0

        for i in range(len(pred_ranks)):
            idx = pred_ranks[i]
            if idx not in truth and pred[idx] > pred[ent2idx[q_t]]:
                L.append(idx)
            if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
                H.append(idx)

        for i in range(len(L) + 1, len(H) + 1):
            mean_rank += i / (len(H) - len(L))
        Mean_rank['mrr'].append(1.0 / mean_rank)
        Mean_rank['hits_1'].append(1 if mean_rank <= 1 else 0)
        Mean_rank['hits_10'].append(1 if mean_rank <= 10 else 0)
        head2Mean_rank_mrr[q_r].append(1.0 / mean_rank)
        head2Mean_rank_hit_1[q_r].append(1 if mean_rank <= 1 else 0)
        head2Mean_rank_hit_10[q_r].append(1 if mean_rank <= 10 else 0)
        print("Number{}_use_mean_rank:{}".format(q_i, mean_rank))

        top_rank = len(L) + 1
        Top_rank['mrr'].append(1.0 / top_rank)
        Top_rank['hits_1'].append(1 if top_rank <= 1 else 0)
        Top_rank['hits_10'].append(1 if top_rank <= 10 else 0)
        head2Top_rank_mrr[q_r].append(1.0 / top_rank)
        head2Top_rank_hit_1[q_r].append(1 if top_rank <= 1 else 0)
        head2Top_rank_hit_10[q_r].append(1 if top_rank <= 10 else 0)
        print("Number{}_use_top_rank:{}".format(q_i, top_rank))

    print("{:<16}{:<20} Hits@1:{:<20} Hits@10:{:<20}\n{:>16}{:<20} Hits@1:{:<20} Hits@10:{:<20}\n".format(
        "expectation MRR:", np.mean(Mean_rank['mrr']), np.mean(Mean_rank['hits_1']),
        np.mean(Mean_rank['hits_10']), "TOP MRR:", np.mean(Top_rank['mrr']), np.mean(Top_rank['hits_1']),
        np.mean(Top_rank['hits_10'])))

    model_file = get_model_file(args)
    os.makedirs("../evaluate/{}/{}".format(args.model, args.datasets), exist_ok=True)
    with open("../evaluate/{}/{}/{}_{}-{}_[{}].txt".format(args.model, args.datasets, model_file, args.rule_len_low,
                                                           args.rule_len_high, args.threshold), 'w') as f:
        f.write("{:<16}{:<20} Hits@1:{:<20} Hits@10:{:<20}\n{:>16}{:<20} Hits@1:{:<20} Hits@10:{:<20}\n".format(
            "expectation MRR:", np.mean(Mean_rank['mrr']), np.mean(Mean_rank['hits_1']),
            np.mean(Mean_rank['hits_10']), "TOP MRR:", np.mean(Top_rank['mrr']), np.mean(Top_rank['hits_1']),
            np.mean(Top_rank['hits_10'])))

        f.write('\n{:<40}{:<20}{:<20}\n'.format("head", "expectation MRR", "TOP MRR"))
        for (head, mrr1), (head, mrr2) in zip(head2Mean_rank_mrr.items(), head2Top_rank_mrr.items()):
            f.write('{:<40}{:<20}{:<20}\n'.format(head, np.mean(mrr1), np.mean(mrr2)))


def feq(relation, fact_rdf):
    count = 0
    for rdf in fact_rdf:
        h, r, t = parse_rdf(rdf)
        if r == relation:
            count += 1
    return count
