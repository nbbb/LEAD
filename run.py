#!/usr/bin python3
# -*- encoding: utf-8 -*-
# @Author : 犇犇
# @File : run.py
# @Time : 2022/1/18 14:55
from __future__ import print_function
import os

import torch
import os
from utils import dict_to_cls, get_predicted_captions2, save_result, score,load_checkpoint,count_parameters
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from torch import nn
from main import _get_model


class RunConfig:
    ckpt_fpath = "/data2/ntz/my_code/dingxing/GSGN_MSRVTT/best.ckpt"
    result_dpath = "/data2/ntz/my_code/dingxing/results"


def makeconfig(config):
    dataset_init_word2idx={'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    config.dataset_init_word2idx=dataset_init_word2idx
    dataset_backbones={ "B16_feats": 512}
    config.dataset_backbones=dataset_backbones
    config.model_vision_attention_mode=None
    config.model_vision_checkpoint=None
    return config
def run(ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)

    """ Load Config """
    config = dict_to_cls(checkpoint['config'])

    config = makeconfig(config)
    config.training_beam_size=3
    """ Build Data Loader """
    if config.dataset_corpus == "MSVD":
        corpus = MSVD(config)
    elif config.dataset_corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    train_iter, val_iter, test_iter,vocab = \
        corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader,corpus.vocab
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, config.dataset_min_count))

    """ Build Models """
    model = _get_model(config)
    model.cuda()
    model = nn.DataParallel(model)
    print("#params: ", count_parameters(model))
    load_checkpoint(model, ckpt_fpath)
    """ Test Set """
    test_vid2pred = get_predicted_captions2(test_iter, model , vocab , config)
    test_vid2GTs = test_iter.captions
    test_scores = score(test_vid2pred, test_vid2GTs)
    print("[TEST] {}".format(test_scores))

    test_save_fpath = os.path.join(RunConfig.result_dpath, "{}_{}.csv".format(config.dataset_corpus, 'test'))
    save_result(test_vid2pred, test_vid2GTs, test_save_fpath)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    run(RunConfig.ckpt_fpath)

