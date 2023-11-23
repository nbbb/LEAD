from utils import LossChecker,parse_batch
from tqdm import tqdm

import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
import os
import numpy as np
import torch.nn.functional as F
class suppress_stdout_stderr:
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



def train_rl(e, model, optimizer, train_iter, vocab, gradient_clip):
    loss_checker = LossChecker(1)
    t = tqdm(train_iter)
    vid2GTs = train_iter.captions
    PAD_idx = vocab.word2idx['<PAD>']
    ntz=1
    for batch in t:
        vids, feats, captions,senfs,category_label = parse_batch(batch)
        optimizer.zero_grad()
        # model.eval()
        # with torch.no_grad():
        #     output , _ = model(feats,category_label=category_label)
        #     greedy_res=output['outputs'][:,0]
        model.train()
        sample_output = model(feats, category_label=category_label)
        (sample0_reward, sample0_probs,sample0_masks),\
        (sample1_reward, sample1_probs,sample1_masks)\
            =calc_reward(vids,vocab,None,sample_output['outputs_rl'],vid2GTs,captions)
        reward0=(sample0_reward-0.1)*3#-greedy_scores
        reward0=np.repeat(reward0[:, np.newaxis], sample_output['outputs_rl'][0].shape[1], 1)
        reward0 = torch.from_numpy(reward0).float().cuda()
        loss0 = -reward0.contiguous().view(-1)*sample0_probs.contiguous().view(-1)*sample0_masks.contiguous().view(-1)
        loss0 = torch.sum(loss0)/torch.sum(sample0_masks)

        # reward1 = sample1_reward  # -greedy_scores
        # reward1 = np.repeat(reward1[:, np.newaxis], sample_output['outputs_rl'][1].shape[1], 1)
        # reward1 = torch.from_numpy(reward1).float().cuda()
        # loss1 = -reward1.contiguous().view(-1) * sample1_probs.contiguous().view(-1) * sample1_masks.contiguous().view(
        #     -1)
        # loss1 = torch.sum(loss1) / torch.sum(sample1_masks)

        # ce_1 = F.nll_loss(sample_output['outputs_ce'][0][:, 1:].contiguous().view(-1, vocab.n_vocabs),
        #                   captions[:, 1:].contiguous().view(-1),
        #                   ignore_index=PAD_idx)

        loss = loss0 #+ ce_1#+ loss1

        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item())
        t.set_description("[Epoch  #{0}] loss: {1:.3f}  ".format(e, *loss_checker.mean(last=10)))


    total_loss = loss_checker.mean()
    loss = {
        'total': total_loss
    }
    return loss


def calc_reward(vids,vocab,greedy_res,sample_output,vid2GTs,captions):
    vid_map={i: vid for i,vid in enumerate(vids)}
    EOS_idx = vocab.word2idx['<EOS>']
    cider=Cider()
    # METEOR = Meteor()
    # bleu=Bleu(4)
    # rouge=Rouge()
    '''greedy'''
    # if greedy_res is not None:
    #     greedyLogprobs, greedy_res = torch.max(greedy_res.data, 2)
    #     greedy_cm = [idxs_to_sentence_rl(caption, vocab.idx2word, EOS_idx) for caption in greedy_res[:,1:]]
    #     greedy_caps={i: [p[0]] for i,p in enumerate(greedy_cm)}
    #     greedy_masks=torch.stack([p[1] for p in greedy_cm])
    '''sample0'''
    sample_res0= sample_output[0].long()
    sample_Logprobs0=sample_output[1]
    sample_cm0 = [idxs_to_sentence_rl(caption, vocab.idx2word, EOS_idx) for caption in sample_res0[:, 1:]]
    sample_caps0 = {i: [p[0]] for i, p in enumerate(sample_cm0)}
    sample_masks0 = torch.stack([p[1] for p in sample_cm0])
    '''sample1'''
    sample_res1 = sample_output[ 2].long()
    sample_Logprobs1 = sample_output[3]
    sample_cm1 = [idxs_to_sentence_rl(caption, vocab.idx2word, EOS_idx) for caption in sample_res1[:, 1:]]
    sample_caps1 = {i: [p[0]] for i, p in enumerate(sample_cm1)}
    sample_masks1 = torch.stack([p[1] for p in sample_cm1])
    '''GTs'''
    GTs={id: vid2GTs[vid_map[id]] for id in vid_map.keys()}
    one_Gts=[idxs_to_sentence_rl(caption, vocab.idx2word, EOS_idx) for caption in captions[:,2:]]
    one_Gts = {i: [p[0]] for i, p in enumerate(one_Gts)}

    # with suppress_stdout_stderr():
        # _, greedy_scores = cider.compute_score(GTs, greedy_caps)

    _, sample0_scores = cider.compute_score(GTs, sample_caps0)#,verbose=0
    # sample0_scores=np.array(sample0_scores[3])

        # _, sample1_scores = cider.compute_score(GTs, sample_caps1)
    _, sample1_scores = cider.compute_score(one_Gts, sample_caps1)#,verbose=0
    # sample1_scores=np.array(sample1_scores[3])
    return (sample0_scores,sample_Logprobs0,sample_masks0),\
           (sample1_scores,sample_Logprobs1,sample_masks1)



from torch.autograd import Variable
def idxs_to_sentence_rl(idxs, idx2word, EOS_idx):
    words = []
    # for idx in idxs[1:]:
    for m,idx in enumerate(idxs):
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    mask=Variable(torch.cuda.BoolTensor(len(idxs)+1).fill_(False))
    mask[1:m+2]=True
    return sentence,mask