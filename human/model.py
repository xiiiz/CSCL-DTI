import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
import copy
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from torch.nn.utils.weight_norm import weight_norm
import matplotlib.pyplot as plt
import pandas as pd
import time

def drop_tokens(embeddings, word_dropout):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - word_dropout)
    embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    return embeddings


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, batch, atom_dim, hid_dim, out_dim=128, dropout=0.2):
        super(GCN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.batch = batch

        self.gc1 = GraphConvolution(atom_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, hid_dim)

        self.out = torch.nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = self.relu(x)
        x = self.out(x)
        x = self.relu(x)
        return self.dropout(x)


class Clip(nn.Module):
    def __init__(self, temperature=0.05):
        super(Clip, self).__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, features_1, features_2):
        N = features_1.size()[0]
        cat_features_1 = torch.cat([features_1, features_2])
        cat_features_2 = torch.cat([features_2, features_1])
        features_1 = cat_features_1 / cat_features_1.norm(dim=1, keepdim=True)
        features_2 = cat_features_2 / cat_features_2.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_f1 = logit_scale * features_1 @ features_2.t()
        labels = torch.arange(2 * N).long().to(logits_per_f1.device)
        loss = self.loss_fun(logits_per_f1, labels) / 2
        return loss, logits_per_f1



def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Predictor(nn.Module):
    def __init__(self, hid_dim, device, dropout=0.2, atom_dim=75, batch=16,  bias=True):
        super(Predictor, self).__init__()
        self.device = device
        self.fusionsize = hid_dim
        self.max_d = 50
        self.max_p = 545
        self.input_dim_drug = 23532
        self.input_dim_targe = 16693
        self.n_layer = 2
        self.emb_size = hid_dim
        self.dropout_rate = 0.1

        self.hidden_size = hid_dim
        self.intermediate_size = 512
        self.num_attention_heads = 4
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.batch_size = batch


        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_targe, self.emb_size, self.max_p, self.dropout_rate)

        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)

        self.drug_GCN = GCN(self.batch_size, atom_dim=atom_dim, hid_dim=128, out_dim=128, dropout=dropout)

        self.clip = Clip()
        self.ban_heads = 2
        self.fusion1 = BANLayer(v_dim=hid_dim, q_dim=hid_dim, h_dim=hid_dim, h_out=self.ban_heads)
        self.fusion = AFF(self.fusionsize)

        self.out = nn.Sequential(
            nn.Linear(hid_dim*2 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.dropout = dropout
        self.do = nn.Dropout(dropout)
        self.atom_dim = atom_dim

        self.decoder_1 = nn.Sequential(
            nn.Linear(self.max_d * self.emb_size, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, hid_dim)
        )
        self.decoder_2 = nn.Sequential(
            nn.Linear(self.max_p * self.emb_size, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, hid_dim)
        )

        self.flatten = nn.Flatten()

        self.query_proj = nn.Linear(256, 256 * 2, bias=False)
        self.key_proj = nn.Linear(256, 256 * 2, bias=False)
        self.value_proj = nn.Linear(256, 256 * 2, bias=False)
        self.output_proj = nn.Linear(256 * 2, 256, bias=False)


    def forward(self, fts_1, adjs_1, num_size, de_1,pe_1,d_mask,p_mask):
        num_size = num_size.size(0)

        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)

        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0

        d_emb = self.demb(de_1)

        p_emb= self.pemb(pe_1)
        p_mask1 = drop_tokens(p_emb, 0.1)
        p_mask2 = drop_tokens(p_emb, 0.1)

        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())

        p_encoded_layers1 = self.p_encoder(p_mask1.float(), ex_p_mask.float())
        p_encoded_layers2 = self.p_encoder(p_mask2.float(), ex_p_mask.float())


        d1_trans_fts = d_encoded_layers.view(num_size, -1)
        d1_trans_fts_layer1 = self.decoder_1(d1_trans_fts)

        p1_trans_fts1 = p_encoded_layers1.view(num_size, -1)
        p1_trans_fts2 = p_encoded_layers2.view(num_size, -1)
        p_encoded_layers1 = self.decoder_2(p1_trans_fts1)
        p_encoded_layers2 = self.decoder_2(p1_trans_fts2)

        cmp_gnn_out = self.drug_GCN(fts_1, adjs_1)


        is_max = False
        if is_max:
            cmp_gnn_out = cmp_gnn_out.max(dim=1)[0]
        else:
            cmp_gnn_out = cmp_gnn_out.mean(dim=1)

        contrast_loss1, logits_per_f1 = self.clip(d1_trans_fts_layer1, cmp_gnn_out)
        contrast_loss2, logits_per_f2 = self.clip(p_encoded_layers1, p_encoded_layers2)

        d1_trans_fts_layer1 = d1_trans_fts_layer1.view(num_size, -1, self.hidden_size)
        cmp_gnn_out = cmp_gnn_out.view(num_size, -1, self.hidden_size)
        output1, _  = self.fusion1(d1_trans_fts_layer1, cmp_gnn_out)
        p_encoded_layers = self.fusion(p_encoded_layers1, p_encoded_layers2)
        final_fts_cat  = torch.cat((output1, p_encoded_layers), dim=-1)
        contrast_loss = (contrast_loss1+contrast_loss2) / 2
        torch.cuda.empty_cache()

        return self.out(final_fts_cat), contrast_loss

    def __call__(self, data, train=True):
        compound, adj, num_size, drug, protein, dmask_new, pmask_new, correct_interaction = data
        Loss = nn.CrossEntropyLoss()

        if train:
            predicted_interaction,CR_loss = self.forward(compound, adj, num_size, drug, protein, dmask_new, pmask_new)
            CE_loss = Loss(predicted_interaction, correct_interaction)
            loss = 0.1 * CR_loss + 1.5 *  CE_loss
            return loss
        else:
            predicted_interaction, CR_loss = self.forward(compound, adj, num_size, drug, protein,dmask_new, pmask_new)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores


class TextCNN2(nn.Module):
    def __init__(self, embed_dim, hid_dim):
        super(TextCNN2, self).__init__()
        self.convs_protein = nn.Sequential(
            nn.Conv1d(embed_dim, 512, kernel_size=3, padding=3),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=3, padding=3),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, hid_dim, kernel_size=3, padding=3),
        )


    def forward(self, protein):
        protein = protein.permute([0, 2, 1])

        protein_features = self.convs_protein(protein)
        protein_features = protein_features.permute([0, 2, 1])
        return protein_features


class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5], dropout_rate=0.5):
        super(TextCNN, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Linear(hid_dim * len(kernels), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, protein):
        protein = protein.permute([0, 2, 1])
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features = torch.cat((features1, features2), 1)
        features = features.max(dim=-1)[0]
        return self.conv(features)

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class AFF(nn.Module):
    def __init__(self, fusionsize, channels=128, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(1, inter_channels, kernel_size=(1, fusionsize), stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, fusionsize)),
            nn.Conv2d(1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size, feature_size = x.size()
        x = x.view(batch_size, 1, 1, feature_size)
        y = y.view(batch_size, 1, 1, feature_size)

        xy = x + y
        xl = self.local_att(xy)
        xg = self.global_att(xy)
        xlg = xl + xg

        wei = self.sigmoid(xlg.squeeze(dim=2).squeeze(dim=2))
        wei_new = wei.squeeze(dim=1)
        wei_new = torch.mean(wei_new, dim=1, keepdim=True)
        xo = x.squeeze(dim=2).squeeze(dim=2) * wei + y.squeeze(dim=2).squeeze(dim=2) * (1 - wei)
        xo = xo.squeeze(dim=1)
        return xo

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        b = torch.LongTensor(1, 2)
        b = b.cuda()
        input_ids = input_ids.type_as(b)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states


MAX_PROTEIN_LEN = 545
MAX_DRUG_LEN = 50

def data_to_device(data,device):
    atoms_new, adjs_new, num_size_new, d_new, p_new ,dmask_new ,pmask_new ,label_new = data

    atoms_new = atoms_new.to(device)
    adjs_new = adjs_new.to(device)
    num_size_new = num_size_new.to(device)
    d_new = d_new.to(device)
    p_new = p_new.to(device)
    dmask_new = dmask_new.to(device)
    pmask_new = pmask_new.to(device)
    label_new = label_new.to(device)

    return (atoms_new, adjs_new, num_size_new, d_new, p_new ,dmask_new ,pmask_new ,label_new)

from transformers import AdamW, get_cosine_schedule_with_warmup

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch, n_sample):
        self.model = model
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=10,
                                                            num_training_steps=n_sample // batch)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()

        for i, data_pack in enumerate(dataset):

            data_pack = data_to_device(data_pack, device)

            loss = self.model(data_pack)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []

        with torch.no_grad():
            for i, data_pack in enumerate(dataset):

                data_pack = data_to_device(data_pack, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(data_pack, train=False)

                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                torch.cuda.empty_cache()

        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, PRC, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)



