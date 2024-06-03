import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample
import random
from torch.nn.utils import clip_grad_norm_
import time
import pickle
import argparse
import numpy as np
import os

from data import *
from utils import *
from model import *
from kg_completion import *

import debugpy

print("Waiting for debugger")

print("Attached! :)")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

CUDA_VISIBLE_DEVICES = 0, 1

print_msg(str(device))

rule_conf = {}
candidate_rule = {}


def sample_training_data(sample_max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict):
    print("Sampling training data...")
    anchors_rdf = []
    per_anchor_num = anchor_num // ((head_rdict.__len__() - 1) // 2)
    print("Number of head relation:{}".format((head_rdict.__len__() - 1) // 2))
    print("Number of per_anchor_num: {}".format(per_anchor_num))

    fact_dict = construct_fact_dict(fact_rdf)
    for head in head_rdict.rel2idx:

        if head != "None" and "inv_" not in head:
            sampled_rdf = sample_anchor_rdf(fact_dict[head], per_anchor_num)
            anchors_rdf.extend(sampled_rdf)

    print("Total_anchor_num", len(anchors_rdf))
    train_rule, train_rule_dict = [], {}
    len2train_rule_idx = {}

    sample_number = 0

    for anchor_rdf in anchors_rdf:
        rule_seq, record = construct_rule_seq(fact_rdf, anchor_rdf, entity2desced, sample_max_path_len,
                                              PRINT=False)

        sample_number += len(record)
        if len(rule_seq) > 0:
            train_rule += rule_seq
            for rule in rule_seq:
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                h = head_rdict.idx2rel[idx[-1].item()]

                if h not in train_rule_dict:
                    train_rule_dict[h] = []
                train_rule_dict[h].append(idx)
                body_len = len(idx) - 2
                if body_len in len2train_rule_idx.keys():
                    len2train_rule_idx[body_len] += [idx]
                else:
                    len2train_rule_idx[body_len] = [idx]

    print("# head:{}".format(len(train_rule_dict)))
    for h in train_rule_dict:
        print("head {}:{}".format(h, len(train_rule_dict[h])))
    rule_len_range = list(len2train_rule_idx.keys())
    print("Fact set number:{} Sample number:{}".format(len(fact_rdf), sample_number))
    for rule_len in rule_len_range:
        print("sampled examples for rule of length {}: {}".format(rule_len, len(len2train_rule_idx[rule_len])))
    print("length_of_train_rule:{}".format(len(train_rule)))

    return len2train_rule_idx


def train(args, dataset):
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    entity2desced = construct_descendant(all_rdf)

    relation_num = rdict.__len__()

    sample_max_path_len = args.sample_max_path_len
    anchor_num = args.anchor
    len2train_rule_idx = sample_training_data(sample_max_path_len, anchor_num, all_rdf, entity2desced, head_rdict)
    print_msg("  Start training  ")
    batch_size = args.batch_size
    emb_size = args.embedding_size
    n_epoch = args.epochs
    lr = args.learning_rate
    body_len_range = list(range(args.learned_rule_len_from_x_to_X, args.learned_rule_len_from_2_to_X + 1))
    print("body_len_range", body_len_range)
    model = Encoder(relation_num, emb_size, device)
    if torch.cuda.is_available():
        model = model.cuda()
        if args.parallel:
            device_ids = [0, 1]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    loss_func_head = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    start = time.time()

    train_acc = {}

    for rule_len in body_len_range:
        rule_ = len2train_rule_idx[rule_len]
        print("\nrule length:{}".format(rule_len))

        train_acc[rule_len] = []
        for epoch in range(n_epoch):
            model.zero_grad()

            if len(rule_) > batch_size:
                sample_rule_ = sample(rule_, batch_size)
            else:
                sample_rule_ = rule_
            body_ = [r_[0:-2] for r_ in sample_rule_]
            head_ = [r_[-1] for r_ in sample_rule_]

            inputs_h = body_
            targets_h = head_

            inputs_h = torch.stack(inputs_h, 0).to(device)
            targets_h = torch.stack(targets_h, 0).to(device)

            pred_head, _entropy_loss, relation_emb_weigth = model(inputs_h)

            loss_head = loss_func_head(pred_head, targets_h.reshape(-1))

            entropy_loss = _entropy_loss.mean()
            loss = args.alpha * loss_head + (1 - args.alpha) * entropy_loss

            if epoch % (n_epoch // 10) == 0:
                print("### epoch:{}\tloss_head:{:.3}\tentropy_loss:{:.3}\tloss:{:.3}\t".format(epoch, loss_head,
                                                                                               entropy_loss, loss))

            train_acc[rule_len].append(
                ((pred_head.argmax(dim=1) == targets_h.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy())

            clip_grad_norm_(model.parameters(), 0.5)
            loss.backward()
            optimizer.step()

        import matplotlib.pyplot as plt

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("LogicFormer Epoch vs Accurary")
        train_acc[rule_len] = [float(x) for x in train_acc[rule_len]]
        plt.plot(train_acc[rule_len])

        os.makedirs("../figures/{}/{}/{}".format(args.model, args.datasets, train_file_name), exist_ok=True)
        plt.savefig('../figures/{}/{}/{}/{}.png'.format(args.model, args.datasets, train_file_name, rule_len))

    end = time.time()
    print("Time usage: {:.2}".format(end - start))

    print("Saving model...")
    os.makedirs("../results/{}/{}".format(args.model, args.datasets, train_file_name), exist_ok=True)
    with open('../results/{}/{}/{}'.format(args.model, args.datasets, train_file_name), 'wb') as g:
        pickle.dump(model, g)


def enumerate_body(relation_num, rdict, body_len):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(range(relation_num), repeat=body_len))

    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body


def test(args, dataset):
    head_rdict = dataset.get_head_relation_dict()
    with open('../results/{}/{}/{}'.format(args.model, args.datasets, train_file_name),
              'rb') as g:
        if torch.cuda.is_available():
            model = pickle.load(g)
            model.to(device)
            print_msg(str(device))
        else:
            model = torch.load(g, map_location='cpu')
    print_msg("  Start Eval  ")
    model.eval()

    r_num = head_rdict.__len__() - 1

    batch_size = args.batch_size

    for i in range(args.learned_rule_len_from_x_to_X, args.learned_rule_len_from_2_to_X + 1):
        rule_len = i
        print("\nrule length:{}".format(rule_len))

        probs = []
        _, body = enumerate_body(r_num, head_rdict,
                                 body_len=rule_len)
        body_list = ["|".join(b) for b in body]
        candidate_rule[rule_len] = body_list
        n_epoches = math.ceil(float(len(body_list)) / batch_size)
        for epoches in range(n_epoches):
            bodies = body_list[epoches: (epoches + 1) * batch_size]
            if epoches == n_epoches - 1:
                bodies = body_list[epoches * batch_size:]
            else:
                bodies = body_list[epoches * batch_size: (epoches + 1) * batch_size]

            body_idx = body2idx(bodies, head_rdict)

            if torch.cuda.is_available():
                inputs = torch.LongTensor(np.array(body_idx)).to(device)
            else:
                inputs = torch.LongTensor(np.array(body_idx))

            with torch.no_grad():
                pred_head, _entropy_loss, relation_emb_weigth = model(inputs)

                prob_ = torch.softmax(pred_head, dim=-1)
                probs.append(prob_.detach().cpu())

        rule_conf[rule_len] = torch.cat(probs, dim=0)
        print("rule_conf[{}].shape:{}".format(rule_len, rule_conf[rule_len].shape))
        if args.get_rule:
            print_msg("Generate Rule!")
            head_rdict = dataset.get_head_relation_dict()
            n_rel = head_rdict.__len__() - 1
            os.makedirs("../rules/{}/{}/{}".format(args.model, args.datasets, model_file), exist_ok=True)
            rule_path = '../rules/{}/{}/{}/{}.txt'.format(args.model, args.datasets, model_file, rule_len)
            print("\nGenerate rule length:{}".format(rule_len))
            sorted_val, sorted_idx = torch.sort(rule_conf[rule_len], 0, descending=True)
            n_rules, _ = sorted_val.shape
            with open(rule_path, 'w') as g:
                for r in range(n_rel):

                    head = head_rdict.idx2rel[r]
                    idx = 0
                    while idx < args.get_top_k and idx < n_rules:
                        conf = sorted_val[idx, r]
                        body = candidate_rule[rule_len][sorted_idx[idx, r]]
                        msg = "{:.3f} ({:.3f})\t{} <-- ".format(conf, conf, head)
                        body = body.split('|')
                        msg += ", ".join(body)
                        g.write(msg + '\n')
                        idx += 1


def get_model_file(args):
    model_file = '[epochs{}][alpha{}][anchor{}][max_sample{}][emb{}][lr{}][batchsize{}][topk{}]'.format(
        args.epochs,
        args.alpha,
        args.anchor,
        args.sample_max_path_len,
        args.embedding_size,
        args.learning_rate,
        args.batch_size,
        args.get_top_k
    )
    return model_file


def get_train_file_name(args):
    train_file_name = '[epochs{}][alpha{}][anchor{}][max_sample{}][emb{}][lr{}][batchsize{}]'.format(
        args.epochs,
        args.alpha,
        args.anchor,
        args.sample_max_path_len,
        args.embedding_size,
        args.learning_rate,
        args.batch_size
    )
    return train_file_name


if __name__ == '__main__':
    msg = "First Order Logic Rule Mining"
    print_msg(msg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_rule", default="1", action="store_true", help="increase output verbosity")
    parser.add_argument("--gpu", type=int, default=1, help="increase output verbosity")
    parser.add_argument("--parallel", default="", help="increase output verbosity")
    parser.add_argument("--sparsity", type=float, default=1, help="increase output verbosity")
    parser.add_argument("--model", default="show", help="训练谁的模型")
    parser.add_argument("--datasets", default="umls", help="数据集")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--alpha", type=float, default=0.8, help="increase output verbosity")
    parser.add_argument("--anchor", type=int, default=5000, help="increase output verbosity")
    parser.add_argument("--sample_max_path_len", type=int, default=2, help="采样路径长度")
    parser.add_argument("--embedding_size", type=int, default=512, help="embedding_size")
    parser.add_argument("--learning_rate", type=int, default=0.0001, help="learning_rate")
    parser.add_argument("--batch_size", type=int, default=1000, help="increase output verbosity")
    parser.add_argument("--get_top_k", type=int, default=100, help="得到规则的get_top_k")
    parser.add_argument("--learned_rule_len_from_x_to_X", type=int, default=2, help="学习规则的长度")
    parser.add_argument("--learned_rule_len_from_2_to_X", type=int, default=2, help="学习规则的长度")
    parser.add_argument('--cpu_num', type=int, default=20)
    parser.add_argument("--data_batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--rule_len_low", type=int, default=2)
    parser.add_argument("--rule_len_high", type=int, default=2)
    args = parser.parse_args()
    model_file = get_model_file(args)
    train_file_name = get_train_file_name(args)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    data_path = '../datasets/{}/'.format(args.datasets)
    dataset = Dataset(data_root=data_path, sparsity=args.sparsity,
                      inv=True)
    print("Dataset:{}".format(data_path))
    model_path = "../results/{}/{}/{}".format(args.model, args.datasets, train_file_name)
    print("results at:{}".format(model_path))
    if not os.path.isfile("../results/{}/{}/{}".format(args.model, args.datasets, train_file_name)):
        print_msg("Train!")
        train(args, dataset)
    print_msg("Test!")
    test(args, dataset)
    all_rules = {}
    all_rule_heads = []
    for L in range(args.rule_len_low, args.rule_len_high + 1):
        file = "../rules/{}/{}/{}/{}.txt".format(args.model, args.datasets, model_file, L)
        load_rules("{}".format(file), all_rules, all_rule_heads)
    for head in all_rules:
        all_rules[head] = all_rules[head][:args.get_top_k * 5]
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    kg_completion(all_rules, dataset, args)
