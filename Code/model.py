import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=3):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores,
                                 device=V.device)
            scores.masked_fill_(attn_mask.mask,
                                -np.inf)
        attn = torch.softmax(scores,
                             dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn,
                                                                                                            V).type_as(
            context_in)
        if self.output_attention:
            attns = (torch.ones([B,
                                 H,
                                 L_V,
                                 L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries,
                                          keys,
                                          sample_k=U_part,
                                          n_top=u)
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context,
                                             values,
                                             scores_top,
                                             index,
                                             L_Q,
                                             attn_mask)
        return context.transpose(2, 1).contiguous(), attn


class Encoder(nn.Module):
    def __init__(self, relation_num, emb_size, device):

        super(Encoder, self).__init__()
        self.emb = nn.Embedding(relation_num + 1, emb_size, padding_idx=relation_num)
        self.hidden_size = emb_size
        self.relation_num = relation_num
        self.emb_size = emb_size
        self.device = device
        self.dropout_rate = 0.1
        self.num_heads = 8
        self.num_layers = 1
        self.slider_encoder = nn.LSTM(input_size=emb_size,
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers,
                                      batch_first=True,
                                      dropout=0
                                      )
        self.fc = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_1 = nn.Linear(emb_size, emb_size)
        self.w_2 = nn.Linear(emb_size, emb_size)
        self.relu = nn.ReLU()
        self.weight_w_1 = nn.Linear(self.num_heads * (relation_num + 1), self.num_heads * (relation_num + 1))
        self.weight_w_2 = nn.Linear(self.num_heads * (relation_num + 1), relation_num + 1)
        self.layernorm1 = nn.LayerNorm(relation_num + 1)
        self.weight_a_1 = nn.Linear(self.num_heads * (relation_num + 1), self.num_heads * (relation_num + 1))
        self.weight_a_2 = nn.Linear(self.num_heads * (relation_num + 1), relation_num + 1)
        self.layernorm2 = nn.LayerNorm(relation_num + 1)
        self.slider1 = nn.Linear(2 * emb_size, 2 * emb_size)
        self.slider2 = nn.Linear(2 * emb_size, emb_size)
        self.slider_3_1 = nn.Linear(2 * emb_size, 2 * emb_size)
        self.slider_3_2 = nn.Linear(2 * emb_size, emb_size)
        assert self.emb_size % self.num_heads == 0
        self.head_emb_size = self.emb_size // self.num_heads
        self.fc_q = nn.Linear(emb_size, emb_size)
        self.fc_k = nn.Linear(emb_size, emb_size)
        self.fc_v = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, emb_size)
        self.self_fc_q = nn.Linear(emb_size, emb_size)
        self.self_fc_k = nn.Linear(emb_size, emb_size)
        self.self_fc_v = nn.Linear(emb_size, emb_size)
        self.self_out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layernorm = nn.LayerNorm(emb_size)
        self.position_embedding = PositionalEmbedding(d_model=emb_size, max_len=3)
        self.probAttention = ProbAttention()

    def forward(self, inputs):
        inputs = self.emb(inputs)
        a = self.position_embedding(inputs)
        inputs = inputs + a
        batch_size, seq_len, emb_size = inputs.shape
        idx_ = torch.LongTensor(range(self.relation_num)).repeat(batch_size, 1).to(self.device)
        relation_emb_ori = self.emb(idx_)
        relation_emb = relation_emb_ori.reshape(batch_size, self.relation_num, self.num_heads, self.head_emb_size)
        relation_emb, relation_emb_weigth = self.probAttention(queries=relation_emb, keys=relation_emb,
                                                               values=relation_emb)
        relation_emb = relation_emb.reshape(batch_size, self.relation_num, -1)
        relation_emb = self.layernorm(relation_emb_ori + 0.1 * relation_emb)
        L = [inputs]
        loss_list = []
        idx = 0
        while idx < seq_len - 1:
            output, loss = self.reduce_rel_pairs(L[-1], relation_emb)
            L.append(output)
            loss_list.append(loss)
            idx += 1
        selected_rel_pair_after_attention_value, attn_weights, scores = self.multiHeadAttention(L[-1], relation_emb)
        loss = Categorical(probs=attn_weights).entropy()
        loss_list.append(loss)
        loss_tensor = torch.cat(loss_list, dim=-1)
        return self.predict_head(scores), loss_tensor, relation_emb_weigth

    def reduce_rel_pairs(self, inputs, relation_emb_with_attention):

        batch_size, seq_len, emb_size = inputs.shape
        if seq_len > 2:
            rel_pairs = []
            idx = 0
            while idx < seq_len - 1:
                rel_pairs_emb = inputs[:, idx:idx + 2, :]
                rel_pairs_emb = rel_pairs_emb.reshape(batch_size, -1)
                rel_pairs_emb = self.dropout(self.relu(self.slider1(rel_pairs_emb)))
                rel_pairs_emb = self.dropout(self.slider2(rel_pairs_emb))
                h = self.layernorm(rel_pairs_emb)
                rel_pairs.append(h)
                idx += 1
            rel_pairs = torch.stack(rel_pairs, dim=1)
            choice_rel_pairs = self.dropout(self.fc(rel_pairs)).squeeze(-1)
            choice_rel_pairs = self.sigmoid(choice_rel_pairs)
            selected_rel_pair_idx = torch.argmax(choice_rel_pairs,
                                                 dim=-1)
            full_batch = torch.arange(batch_size).to(self.device)
            selected_rel_pair = rel_pairs[full_batch, selected_rel_pair_idx, :]
            selected_rel_pair = selected_rel_pair.unsqueeze(1)
            selected_rel_pair_after_attention_value, attn_weights, scores = self.multiHeadAttention(selected_rel_pair,
                                                                                                    relation_emb_with_attention)
            selected_rel_pair_after_attention_value = self.feedForward(selected_rel_pair_after_attention_value)
            loss = Categorical(probs=attn_weights).entropy()
            selected_rel_pair_after_attention_value = selected_rel_pair_after_attention_value.squeeze(1)
            output = inputs.detach().clone()
            zero = torch.zeros(emb_size).to(self.device)
            output[full_batch, selected_rel_pair_idx, :] = selected_rel_pair_after_attention_value
            output[full_batch, selected_rel_pair_idx + 1, :] = zero
            output = output[~torch.all(output == 0, dim=-1)]
            output = output.reshape(batch_size, -1, emb_size)

        else:

            inter = inputs.reshape(batch_size, -1)
            inter = self.dropout(self.relu(self.slider1(inter)))
            output = self.dropout(self.slider2(inter))
            output = self.layernorm(output).unsqueeze(1)
            loss = torch.zeros((batch_size, 1)).to(self.device)
        return output, loss

    def multiHeadAttention(self, inputs, relation_emb, mask=None):
        batch_size, seq_len, emb_size = inputs.shape
        query = self.dropout(self.fc_q(inputs)).view(batch_size, seq_len, self.num_heads, self.head_emb_size).transpose(
            1, 2)
        key = self.dropout(self.fc_k(torch.cat((relation_emb, inputs), dim=1))).view(batch_size, -1, self.num_heads,
                                                                                     self.head_emb_size).transpose(1, 2)
        value = self.dropout(self.fc_v(torch.cat((relation_emb, inputs), dim=1))).view(batch_size, -1, self.num_heads,
                                                                                       self.head_emb_size).transpose(1,
                                                                                                                     2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_emb_size)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, value)

        output = self.out(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        scores = torch.mean(scores, dim=1)
        attn_weights = torch.mean(attn_weights, dim=1)

        return self.layernorm(output + inputs), attn_weights, scores

    def feedForward(self, inputs):
        inter = self.dropout(self.relu(self.w_1(inputs)))
        output = self.dropout(self.w_2(inter))
        return self.layernorm(output + inputs)

    def transformer_attention(self, inputs, relation_emb):
        batch_size, seq_len, emb_size = inputs.shape
        query = self.dropout(self.fc_q(inputs))
        key = self.dropout(self.fc_k(torch.cat((relation_emb, inputs), dim=1)))
        value = self.dropout(self.fc_v(torch.cat((relation_emb, inputs), dim=1)))
        scores_ori = torch.matmul(query, key.transpose(-2, -1)) \
                     / math.sqrt(self.emb_size)
        mask1 = torch.zeros((batch_size, seq_len, self.relation_num), dtype=torch.bool).to(self.device)
        I = torch.eye(seq_len).to(self.device)
        I = I.reshape((1, seq_len, seq_len))
        I = I.repeat(batch_size, 1, 1)
        mask2 = ~I.to(torch.bool)
        mask = torch.cat((mask1, mask2), dim=-1)
        scores = scores_ori
        scores[mask] = float('-inf')
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ value
        return self.layernorm(output), attn_weights, scores_ori

    def self_attention(self, relation_emb, mask=None):
        batch_size, seq_len, emb_size = relation_emb.shape

        query = self.dropout(self.self_fc_q(relation_emb)).view(batch_size, seq_len, self.num_heads,
                                                                self.head_emb_size).transpose(1, 2)
        key = self.dropout(self.self_fc_k(relation_emb)).view(batch_size, seq_len, self.num_heads,
                                                              self.head_emb_size).transpose(1, 2)
        value = self.dropout(self.self_fc_v(relation_emb)).view(batch_size, seq_len, self.num_heads,
                                                                self.head_emb_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_emb_size)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, value)

        output = self.self_out(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        scores = torch.mean(scores, dim=1)
        attn_weights = torch.mean(attn_weights, dim=1)

        return self.layernorm(output + relation_emb), attn_weights, scores

    def predict_head(self, prob):
        return prob.squeeze(1)

    def get_relation_emb(self, rel):
        return self.emb(rel)

    def weightedAverage(self, scores, inputs):
        batch_size, seq_len, _ = scores.shape
        mask1 = torch.zeros((batch_size, seq_len, self.relation_num), dtype=torch.bool).to(self.device)
        I = torch.eye(seq_len).to(self.device)
        I = I.reshape((1, seq_len, seq_len))
        I = I.repeat(batch_size, 1, 1)
        mask2 = ~I.to(torch.bool)
        mask = torch.cat((mask1, mask2), dim=-1)
        scores[mask] = float('-inf')
        prob = self.dropout(torch.softmax(scores, dim=-1))
        idx_ = torch.LongTensor(range(self.relation_num)).repeat(batch_size, 1).to(self.device)
        relation_emb = self.emb(idx_)
        all_emb = torch.cat((relation_emb, inputs), dim=1)
        out = prob @ all_emb
        return self.layernorm(out)
