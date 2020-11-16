import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel
from data_load import idx2attitude, argument2idx
from consts import NONE
from utils import find_attitudes


class Net(nn.Module):
    def __init__(self, attitude_size=None, entity_size=None, argument_size=None, entity_embedding_dim=50, finetune=True, device=torch.device("cpu"), bert_size='base'):
        super().__init__()
        bert_string = 'bert-{}-uncased'.format(bert_size)
        embed_size = 768
        if bert_size == 'base':
            embed_size = 768
        elif bert_size == 'medium':
            embed_size = 512
        elif bert_size == 'large':
            embed_size = 1024
        else:
            print("unknown size: ", bert_size)
            raise RuntimeError



        self.bert = BertModel.from_pretrained(bert_string)
        self.entity_embed = MultiLabelEmbeddingLayer(
            num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1,
                           input_size=embed_size, hidden_size=embed_size // 2, batch_first=True)

        self.normal_embedding = nn.Embedding(num_embeddings=30522, embedding_dim=768, padding_idx=0)


        # hidden_size = 768 + entity_embedding_dim + postag_embedding_dim
        hidden_size = embed_size
        self.droplin = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
        )
        self.fc_attitude = nn.Sequential(
            nn.Linear(hidden_size, attitude_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
        )
        self.device = device
        self.finetune = finetune

    def predict_attitudes(self, tokens_x_2d, entities_x_3d, head_indexes_2d, attitudes_y_2d, arguments_2d, mode):

        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        attitudes_y_2d = torch.LongTensor(attitudes_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        # entity_x_2d = self.entity_embed(entities_x_3d)
        if mode == "BERT":
            if self.training and self.finetune:
                self.bert.train()
                encoded_layers, _ = self.bert(tokens_x_2d)
                enc = encoded_layers[-1]
            else:
                self.bert.eval()
                with torch.no_grad():
                    encoded_layers, _ = self.bert(tokens_x_2d)
                    enc = encoded_layers[-1]
        if mode == "BERT_twice":
            if self.training and self.finetune:
                self.bert.train()
                encoded_layers, _ = self.bert(tokens_x_2d)
                enc = encoded_layers[-1]
                enc = self.droplin(enc)
            else:
                self.bert.eval()
                with torch.no_grad():
                    encoded_layers, _ = self.bert(tokens_x_2d)
                    enc = encoded_layers[-1]
                    enc = self.droplin(enc)
        elif mode == "Embedding-only":
            embedded = self.normal_embedding(tokens_x_2d)
            enc = embedded

        elif mode == "LSTM":
            print("not implemented")
            return -1
            embedded = self.normal_embedding(tokens_x_2d)
            enc, (h_n, c_n) = self.rnn(embedded)


        # x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        x = enc
        #newx = torch.tensor(x.shape)
        #print(enc.shape)
        # logits = self.fc2(x + enc)

        batch_size = tokens_x_2d.shape[0]

        
        for i in range(batch_size):      
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])
        

        # x, (h_n, c_n) = self.rnn(x)

        attitude_logits = self.fc_attitude(x)
        attitude_hat_2d = attitude_logits.argmax(-1)

        argument_hidden, argument_keys =0,0

        return attitude_logits, attitudes_y_2d, attitude_hat_2d, argument_hidden, argument_keys





    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append(
                (e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
