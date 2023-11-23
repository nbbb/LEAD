#!/usr/bin python3
# -*- encoding: utf-8 -*-
# @Author : 犇犇
# @File : decoder.py
# @Time : 2021/11/9 9:21
import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_attention import TemporalAttention
# from .rnn_ntz import RNN

class Decoder(nn.Module):
    def __init__(self,config,embedding):
        super(Decoder, self).__init__()
        self.rnn_type = config.decoder_rnn_type
        self.num_layers= config.decoder_num_layers
        self.num_directions = config.decoder_num_directions
        self.feat_size = config.model_vision_senf_size
        self.v_feat_size = sum(config.dataset_backbones.values())
        # self.feat_size = config.dataset_n_vocabs
        self.embedding_size = config.decoder_embedding_size
        self.hidden_size = config.decoder_hidden_size
        self.attn_size = config.decoder_attn_size
        self.output_size = config.dataset_n_vocabs
        self.dropout = nn.Dropout(p=config.decoder_dropout)
        # self.embedding=embedding

        self.embedding = nn.Embedding(self.output_size, config.decoder_embedding_size)

        self.detach=config.training_detach
        self.attention_two = TemporalAttention(
                hidden_size=self.num_directions * self.hidden_size,
                feat_size=self.feat_size,
                bottleneck_size=self.attn_size)

        self.attention_v = TemporalAttention(
            hidden_size=self.num_directions * self.hidden_size,
            feat_size=self.v_feat_size,
            bottleneck_size=self.attn_size)

        RNN = nn.LSTMCell if self.rnn_type == 'LSTM' else nn.GRUCell

        self.one_step_rnn=RNN(self.v_feat_size + self.embedding_size*2+self.hidden_size, self.hidden_size)#+config.decoder_category_emb
        self.two_step_rnn = RNN(self.feat_size + self.embedding_size*2+self.hidden_size, self.hidden_size)
        self.out2= nn.Linear(self.num_directions * self.hidden_size, self.output_size)
        self.out1 = nn.Linear(self.num_directions * self.hidden_size, self.output_size)
        self.cate_emb = nn.Embedding(config.decoder_category_nums, config.decoder_category_emb)


    def get_last_hidden(self, hidden):
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        last_hidden = last_hidden.view(self.num_layers, self.num_directions,
                                       last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1),
                                       self.num_directions * last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden


    def forward(self, embedded0,embedded1, hidden, feats,v_feats,category_label):

        last_hidden_2 = self.get_last_hidden(hidden[1].unsqueeze(0))
        last_hidden_2 =self.dropout(last_hidden_2)
        feats_2,_=self.attention_two(last_hidden_2,feats)
        input_combined_2 = torch.cat((embedded0.squeeze(0).detach(), embedded1.squeeze(0),feats_2,hidden[0].detach()), dim=1)
        input_combined_2 =self.dropout(input_combined_2)
        h_2 = self.two_step_rnn(input_combined_2,hidden[1])
        h_2 = self.dropout(h_2)
        output_2 = self.out2(h_2)

        output_2_ls = F.log_softmax(output_2, dim=1)
        next_2 = output_2_ls.data.max(1)[1]
        next_2_embedded = self.embedding(next_2)

        last_hidden_1 = self.get_last_hidden(hidden[0].unsqueeze(0))
        last_hidden_1 = self.dropout(last_hidden_1)
        v_feats, _ = self.attention_v(last_hidden_1,v_feats)
        # feats_1=feats.mean(1)

        cate_embedded=self.cate_emb(category_label.squeeze(1))

        # input_combined_1 = torch.cat((v_feats,next_2_embedded,h_2,embedded0.squeeze(0),cate_embedded), dim=1)
        input_combined_1 = torch.cat((v_feats, next_2_embedded.detach(), h_2.detach(), embedded0.squeeze(0)), dim=1)
        input_combined_1 =self.dropout(input_combined_1)
        h_1 = self.one_step_rnn(input_combined_1, hidden[0])
        h_1 =self.dropout(h_1)
        output_1 = self.out1(h_1)
        # output_1_ls = F.log_softmax(output_1, dim=1)
        return [output_1,output_2], [h_1,h_2]