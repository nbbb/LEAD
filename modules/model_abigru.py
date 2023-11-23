#!/usr/bin python3
# -*- encoding: utf-8 -*-
# @Author : 犇犇
# @File : model_abigru.py
# @Time : 2021/11/8 21:20
import random
import numpy as np
import torch
from .model_vision import BaseVision
import torch.nn as nn
from torch.autograd import Variable
from .allennlp_beamsearch import BeamSearch
from .decoder import Decoder
import torch.nn.functional as F
from fastai.vision import ifnone
class ABI_GRUNet(nn.Module):
    def __init__(self,config):
        super(ABI_GRUNet, self).__init__()
        self.is_rl=config.rl_is_rl
        self.ce1_weight = ifnone(config.model_vision_ce1_weight, 1.0)
        self.ce2_weight = ifnone(config.model_vision_ce2_weight, 1.0)

        self.encoder = BaseVision(config)
        self.max_caption_len = config.dataset_max_length
        self.vocab_size=config.dataset_n_vocabs
        self.teacher_forcing_ratio=config.training_teacher_forcing_ratio
        self.embedding = nn.Embedding(self.vocab_size, config.decoder_embedding_size)
        if config.model_vocab_pretrain:
            glove_pretrain = np.load(config.model_vocab_pretrain)
            self.embedding.weight.data.copy_(torch.from_numpy(glove_pretrain))
        self.vocab_init_word2idx=config.dataset_init_word2idx
        self.beam_search = BeamSearch(  config.dataset_init_word2idx['<EOS>'],config.dataset_max_length,
                                         config.training_beam_size, per_node_beam_size=config.training_beam_size)
        self.decoder = Decoder(config,self.embedding)


    def get_rnn_init_hidden(self, batch_size, num_layers, num_directions, hidden_size):
        if self.decoder.rnn_type == 'LSTM':
            hidden = (
                torch.zeros(num_layers * num_directions, batch_size, hidden_size).cuda(),
                torch.zeros(num_layers * num_directions, batch_size, hidden_size).cuda())
        else:
            hidden = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
            hidden = hidden.cuda()
        return hidden

    def masked_logprobs(self,output, word0,word1):
        word0=word0.view(-1)
        word1 = word1.view(-1)
        batch_size=word0.size(0)
        logprobs_masks0=Variable(torch.cuda.BoolTensor(batch_size,self.vocab_size).fill_(False))
        logprobs_masks1 = Variable(torch.cuda.BoolTensor(batch_size, self.vocab_size).fill_(False))
        for bdash in range(batch_size):
            logprobs_masks0[bdash, word0[bdash]] = True
            logprobs_masks1[bdash, word1[bdash]] = True
        output0 = output[0].masked_fill(logprobs_masks0, -1e9)
        output1 = output[1].masked_fill(logprobs_masks1, -1e9)
        output=torch.stack((output0, output1))
        logprobs = F.log_softmax(output, dim=2)
        return logprobs
    def forward_decoder(self, batch_size, vocab_size, hidden, feats,v_feats, captions,category_label=None):
        out0puts = Variable(torch.zeros(self.max_caption_len + 3, batch_size, vocab_size)).cuda()
        out1puts = Variable(torch.zeros(self.max_caption_len + 3, batch_size, vocab_size)).cuda()
        output0 = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab_init_word2idx['<SOS>']))
        output1 = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab_init_word2idx['<SOS>']))
        embedded0 = self.embedding(output0.view(1, -1))
        embedded1= self.decoder.embedding(output1.view(1, -1))
        for t in range(1, self.max_caption_len + 2):
            output, hidden = self.decoder(embedded0,embedded1, hidden, feats,v_feats,category_label)
            out0put = F.log_softmax(output[0], dim=1)
            out1put = F.log_softmax(output[1], dim=1)
            out0puts[t] = out0put
            out1puts[t] = out1put
            is_teacher = random.random() < self.teacher_forcing_ratio if captions is not None else False
            top1_0= out0put.data.max(1)[1]
            output0 = Variable(captions.data[t] if is_teacher else top1_0).cuda()
            embedded0=self.embedding(output0.view(1, -1))

            tnext=t+1
            top1_1 = out1put.data.max(1)[1]
            if captions is not None:
                output1 = Variable(captions.data[tnext] if not is_teacher else top1_1).cuda()
            else:
                output1=top1_1
            embedded1 = self.decoder.embedding(output1.view(1, -1))

        return [out0puts,out1puts]


    def forward_decoder_sample(self, batch_size, vocab_size, hidden, feats,v_feats, captions,category_label=None):
        out0puts = Variable(torch.zeros(self.max_caption_len + 3, batch_size, vocab_size)).cuda()
        out1puts = Variable(torch.zeros(self.max_caption_len + 3, batch_size, vocab_size)).cuda()
        output0 = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab_init_word2idx['<SOS>']))
        output1 = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab_init_word2idx['<SOS>']))
        seqWords0 = Variable(torch.zeros(self.max_caption_len + 2, batch_size)).cuda()
        seqWords1 = Variable(torch.zeros(self.max_caption_len + 2,  batch_size)).cuda()
        seqLogprobs0 = Variable(torch.zeros(self.max_caption_len + 2,  batch_size)).cuda()
        seqLogprobs1 = Variable(torch.zeros(self.max_caption_len + 2,  batch_size)).cuda()
        embedded0 = self.embedding(output0.view(1, -1))
        embedded1= self.embedding(output1.view(1, -1))
        for t in range(1, self.max_caption_len + 2):
            output, hidden = self.decoder(embedded0,embedded1, hidden, feats,v_feats,category_label)
            out0put = F.log_softmax(output[0], dim=1)
            out1put = F.log_softmax(output[1], dim=1)
            out0puts[t] = out0put
            out1puts[t] = out1put
            word0 = torch.distributions.Categorical(logits=out0put.detach()).sample()
            sampleLogprobs0 = out0put.gather(1, word0.unsqueeze(1))
            embedded0 = self.embedding(word0.view(1, -1))

            word1 = torch.distributions.Categorical(logits=out1put.detach()).sample()
            sampleLogprobs1 = out1put.gather(1, word1.unsqueeze(1))
            embedded1 = self.embedding(word1.view(1, -1))
            # embedded1 = self.embedding(out1put.data.max(1)[1].view(1, -1))


            seqWords0[t] = word0
            seqWords1[t] = word1
            seqLogprobs0[t]=sampleLogprobs0.squeeze(1)
            seqLogprobs1[t] = sampleLogprobs1.squeeze(1)

        return [seqWords0,seqLogprobs0,seqWords1,seqLogprobs1],[out0puts,out1puts]



    def forward(self ,feats,captions=None,category_label=None):
        batch_size = feats.size(0)
        feats_vision = self.encoder(feats)
        if captions is not None:
            captions = captions.transpose(0, 1).contiguous()
        hidden_1 = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                          self.decoder.hidden_size)
        hidden_2 = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                          self.decoder.hidden_size)

        hidden=[hidden_1.squeeze(0),hidden_2.squeeze(0)]

        if not self.training or not self.is_rl:
            outputs = self.forward_decoder(batch_size, self.vocab_size, hidden, feats_vision['feature'],feats_vision['v_feature'], captions,category_label)
            outputs=[o.permute(1,0,2).contiguous() for o in outputs]

            return {'ce1_weight': self.ce1_weight, 'ce2_weight': self.ce2_weight,'outputs':outputs,
                 'name': 'all'} ,feats_vision
        else:
            outputs_rl,outputs_ce = self.forward_decoder_sample(batch_size, self.vocab_size, hidden,
                                                  feats_vision['feature'],feats_vision['v_feature'],
                                                  captions, category_label)
            outputs_rl = [o.permute(1,0).contiguous() for o in outputs_rl]
            outputs_ce = [o.permute(1,0,2).contiguous() for o in outputs_ce]
            return {'outputs_rl': outputs_rl, 'outputs_ce': outputs_ce,'name': 'rl'}


    def describe(self, feats,category_label):
        batch_size = feats.size(0)
        feats_vision = self.encoder(feats)
        hidden_1 = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                            self.decoder.hidden_size)
        hidden_2 = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                            self.decoder.hidden_size)



        output1= Variable(torch.cuda.FloatTensor(batch_size,self.vocab_size).fill_(0.99))
        output1[:, 1] = 1.
        output1 = F.log_softmax(output1, dim=1)
        start_state= {'hidden1':hidden_1.squeeze(0),
                      'hidden2': hidden_2.squeeze(0),
                          'feats':feats_vision['feature'],
                      'v_feats': feats_vision['v_feature'],
                      'category_label':category_label,
                      'output1': output1
                      }#feats_vision['logits']
        start_id = Variable(torch.cuda.LongTensor(batch_size).fill_(self.vocab_init_word2idx['<SOS>']))
        captions = self.beam_create_caption(start_id, start_state,batch_size)
        return captions

    def beam_create_caption(self, start_id, start_state,batch_size):
        outputs = []
        predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
        max_prob, max_index = torch.topk(log_prob, 1)
        max_index = max_index.squeeze(1)
        for i in range(batch_size):
            outputs.append(predictions[i, max_index[i], :])
        outputs = torch.stack(outputs)
        if outputs.size(1) < self.max_caption_len:
            pad_outs = Variable(torch.zeros(outputs.size(0), self.max_caption_len - outputs.size(1), dtype=int)).cuda()
            outputs = torch.cat((outputs, pad_outs), dim=1)  # 向量拼接
        return outputs


    def beam_step(self, last_predictions, current_state):
        group_size = last_predictions.size(0)
        batch_size = current_state['feats'].size(0)
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state

            hidden1 = current_state['hidden1'][:, i, :]
            hidden2 = current_state['hidden2'][:, i, :]
            feats = current_state['feats'][:, i, :]
            v_feats=current_state['v_feats'][:, i, :]
            category_label=current_state['category_label'][:, i, :]
            output1=current_state['output1'][:, i, :]
            # decoding stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.embedding(word_id.view(1, -1))

            top1_1 = output1.data.max(1)[1]
            emb1 = self.decoder.embedding(top1_1)
            output, hidden  = self.decoder(word,emb1.unsqueeze(0), [hidden1,hidden2], feats,v_feats,category_label)
            # output = self.masked_logprobs(output, word_id, top1_1)
            # store log probabilities
            out0put = F.log_softmax(output[0], dim=1)
            out1put = F.log_softmax(output[1], dim=1)

            log_probs.append(out0put)

            # update new state
            new_state['hidden1'].append(hidden[0])
            new_state['hidden2'].append(hidden[1])
            new_state['feats'].append(feats)
            new_state['v_feats'].append(v_feats)
            new_state['category_label'].append(category_label)
            new_state['output1'].append(out1put)
        # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size

        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)


