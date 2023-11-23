import torch
import torch.nn as nn
import torch.nn.functional as F




def vision_loss(pr,gt,PAD_idx):

    n_vocabs=pr.size(2)
    prep=F.log_softmax(pr, dim=2)
    loss=F.nll_loss(prep.view(-1, n_vocabs),
                        gt[:,1:].contiguous().view(-1),
                        ignore_index=PAD_idx
                                        )#weight=b_parms

    return loss

def vision_cos(pr,gt):
    # pr=pr / pr.norm(dim=-1, keepdim=True)
    # gt=gt / gt.norm(dim=-1, keepdim=True)
    loss = torch.cosine_similarity(pr, gt, dim=1).mean(0)
    return 1-loss