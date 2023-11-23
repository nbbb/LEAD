import logging
import os
import torch
import yaml
import random
import inspect
from tqdm import tqdm
import losses
import torch.nn.functional as F
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
class Logger(object):
    _handle = None
    _root = None

    @staticmethod
    def init(output_dir, name, phase):
        format = '[%(asctime)s %(filename)s:%(lineno)d %(levelname)s {}] ' \
                '%(message)s'.format(name)
        logging.basicConfig(level=logging.INFO, format=format)

        try: os.makedirs(output_dir)
        except: pass
        config_path = os.path.join(output_dir, f'{phase}.txt')
        Logger._handle = logging.FileHandler(config_path)
        Logger._root = logging.getLogger()

    @staticmethod
    def enable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.addHandler(Logger._handle)

    @staticmethod
    def disable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.removeHandler(Logger._handle)


class Config(object):

    def __init__(self, config_path, host=True):
        def __dict2attr(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    __dict2attr(v, f'{prefix}{k}_')
                else:
                    if k == 'phase':
                        assert v in ['train', 'test']
                    if k == 'stage':
                        assert v in ['pretrain-vision', 'pretrain-language',
                                     'train-semi-super', 'train-super']
                    self.__setattr__(f'{prefix}{k}', v)

        assert os.path.exists(config_path), '%s does not exists!' % config_path
        with open(config_path) as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        __dict2attr(config_dict)
        self.global_workdir = os.path.join(self.global_workdir, self.global_name)

    def __getattr__(self, item):
        attr = self.__dict__.get(item)
        if attr is None:
            attr = dict()
            prefix = f'{item}_'
            for k, v in self.__dict__.items():
                if k.startswith(prefix):
                    n = k.replace(prefix, '')
                    attr[n] = v
            return attr if len(attr) > 0 else None
        else:
            return attr

    def __repr__(self):
        str = 'ModelConfig(\n'
        for i, (k, v) in enumerate(sorted(vars(self).items())):
            str += f'\t({i}): {k} = {v}\n'
        str += ')'
        return str

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [ [] for _ in range(self.num_losses) ]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [ 0. for _ in range(self.num_losses) ]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses

def parse_batch(batch):
    vids, feats, captions, senfs = batch
    feats_f = [feat.cuda() for feat in feats[:-1]]
    feats_f = torch.cat(feats_f, dim=2)
    category_label=feats[-1].long().cuda()
    captions = captions.long().cuda()
    senfs = senfs.cuda()
    return vids, feats_f, captions, senfs,category_label



def train(e, model, optimizer, train_iter, vocab, gradient_clip, ntz_flag):
    model.train()
    loss_checker = LossChecker(4)
    PAD_idx = vocab.word2idx['<PAD>']
    t = tqdm(train_iter)
    for batch in t:
        _, feats, captions,senfs,category_label = parse_batch(batch)
        optimizer.zero_grad()
        output, feats_vision = model(feats,captions,category_label)
        #
        # labels_cur = captions[:, 1:].contiguous().view(-1).unsqueeze(1)
        # labels_cur = labels_cur.new_zeros(labels_cur.size(0),
        #                                     vocab.n_vocabs).scatter_(1, labels_cur, 1).float()
        #
        # output_cur=output['outputs'][:,0][:,1:].contiguous().view(-1, vocab.n_vocabs)
        #
        # kl_cur =  F.kl_div(output_cur,labels_cur,reduction='batchmean')
        ce_1 = F.nll_loss(output['outputs'][0][:,1:].contiguous().view(-1, vocab.n_vocabs),
                                        captions[:,1:].contiguous().view(-1),
                                        ignore_index=PAD_idx)

        # labels_next = captions[:, 2:].contiguous().view(-1).unsqueeze(1)
        # labels_next = labels_next.new_zeros(labels_next.size(0),
        #                                   vocab.n_vocabs).scatter_(1, labels_next, 1).float()
        # kl_next = F.kl_div(output['outputs'][:, 1][:, 1:-1].contiguous().view(-1, vocab.n_vocabs), labels_next,
        #                 reduction='batchmean')
        ce_2=F.nll_loss(output['outputs'][1][:,1:-1].contiguous().view(-1, vocab.n_vocabs),
                                        captions[:,2:].contiguous().view(-1),
                                        ignore_index=PAD_idx)

        #
        # kl= F.kl_div(output['outputs'][:, 0][:, 1:-1].contiguous().view(-1, vocab.n_vocabs),
        #                F.softmax(output['outputs'][:,3][:,2:].contiguous().view(-1, vocab.n_vocabs),dim=1),
        #                         reduction='batchmean')


        # labels = captions[:, 1:].contiguous().view(-1).unsqueeze(1)
        # labels=labels.new_zeros(labels.size(0), vocab.n_vocabs).scatter_(1, labels, 1).float()
        # l_mse=F.mse_loss(output['outputs'][:,0][:,1:].contiguous().view(-1, vocab.n_vocabs),
        #                                 labels)

        vision_loss =  losses.vision_cos(feats_vision['senf_feats'], senfs)

        # loss = output['ce1_weight'] * ce_1 + output['ce2_weight'] * ce_2\
        #        + feats_vision['loss_weight'] * vision_loss   #+ kl*e  #+ output['ce2_weight'] * ce_2
        if ntz_flag:
            loss= output['ce1_weight'] *ce_1   \
              +feats_vision['loss_weight']*vision_loss #+ kl*  + output['ce2_weight'] * ce_2
        else:

            loss =output['ce2_weight'] *ce_2  \
                   + feats_vision['loss_weight'] * vision_loss  # + kl*20

        # else:
        #     loss =  kl + feats_vision['loss_weight'] * vision_loss


        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item(),ce_1.item(),ce_2.item(),vision_loss.item())
        t.set_description("[Epoch  #{0}] loss: {4:.3f} = (CE_1: {1}*{5:.3f}+ CE_2: {2}*{6:.3f} +"
                          "  Sen_cos: {3}*{7:.3f} ) "
                          .format(e, output['ce1_weight'],output['ce2_weight'],
                                  feats_vision['loss_weight'],
                                  *loss_checker.mean(last=10))+"t {} ".format(round(model.module.teacher_forcing_ratio,5)))


    total_loss,ce_1 , ce_2,_ = loss_checker.mean()
    loss = {
        'total': total_loss,
        'ce_1': ce_1,
        'ce_2': ce_2,
    }
    return loss

def test(model, val_iter, vocab):
    model.eval()
    loss_checker = LossChecker(3)
    PAD_idx = vocab.word2idx['<PAD>']
    for b, batch in enumerate(val_iter, 1):
        _, feats, captions,senfs,category_label = parse_batch(batch)
        with torch.no_grad():
            output, feats_vision = model(feats,captions,category_label)

        ce_1 = F.cross_entropy(output['outputs'][ 0][:, 1:].contiguous().view(-1, vocab.n_vocabs),
                          captions[:,1:].contiguous().view(-1),
                          ignore_index=PAD_idx)

        ce_2 = F.cross_entropy(output['outputs'][ 1][:, 1:-1].contiguous().view(-1, vocab.n_vocabs),
                          captions[:, 2:].contiguous().view(-1),
                          ignore_index=PAD_idx)
        vision_loss = losses.vision_cos(feats_vision['senf_feats'], senfs)
        # vision_loss = losses.vision_loss(feats_vision['logit'], captions,PAD_idx)
        loss = output['ce1_weight'] * ce_1 + output['ce2_weight'] * ce_2 \
               + feats_vision['loss_weight'] * vision_loss

        loss_checker.update(loss.item(),ce_1.item(),ce_2.item())
    total_loss, ce_1, ce_2 = loss_checker.mean()
    loss = {
        'total': total_loss,
        'ce_1': ce_1,
        'ce_2': ce_2,
    }
    return loss
def get_predicted_captions2(data_iter, model, vocab,config):
    model.eval()
    vid2pred = {}
    EOS_idx = vocab.word2idx['<EOS>']

    for batch in iter(data_iter):
        vids, feats, _, _,category_label = parse_batch(batch)
        onlyonce_dataset = {}
        onlyonce_cate={}
        for vid, feat,cate_l in zip(vids, feats,category_label):
            if vid not in vid2pred and vid not in onlyonce_dataset:
                onlyonce_dataset[vid] = feat
                onlyonce_cate[vid]=cate_l
        vids = list(onlyonce_dataset.keys())
        feats = list(onlyonce_dataset.values())
        category_label=list(onlyonce_cate.values())
        if len(feats)==0:
            continue
        feats = torch.stack(feats)
        category_label=torch.stack(category_label)
        with torch.no_grad():
            captions = model.module.describe(feats,category_label)
        captions = [idxs_to_sentence(caption, vocab.idx2word, EOS_idx) for caption in captions]
        vid2pred.update({v: p for v, p in zip(vids, captions)})
    return vid2pred



def get_predicted_captions(data_iter, model, vocab,config):
    def build_onlyonce_iter(data_iter,config):
        onlyonce_dataset = {}
        for batch in iter(data_iter):
            vids, feats, _,_ = parse_batch(batch)
            for vid, feat in zip(vids, feats):
                if vid not in onlyonce_dataset:
                    onlyonce_dataset[vid] = feat
        onlyonce_iter = []
        vids = onlyonce_dataset.keys()
        feats = onlyonce_dataset.values()
        batch_size = config.dataset_test_batch_size
        while len(vids) > 0:
            onlyonce_iter.append(( list(vids)[:batch_size], torch.stack(list(feats)[:batch_size]) ))
            vids = list(vids)[batch_size:]
            feats = list(feats)[batch_size:]
        return onlyonce_iter

    model.eval()

    onlyonce_iter = build_onlyonce_iter(data_iter,config)

    vid2pred = {}
    EOS_idx = vocab.word2idx['<EOS>']
    for vids, feats in onlyonce_iter:
        with torch.no_grad():
            captions = model.module.describe(feats)
        captions = [ idxs_to_sentence(caption, vocab.idx2word, EOS_idx) for caption in captions ]
        vid2pred.update({ v: p for v, p in zip(vids, captions) })
    return vid2pred


def evaluate(data_iter, model, vocab,config):
    vid2pred = get_predicted_captions2(data_iter, model, vocab,config)
    vid2GTs = data_iter.captions
    scores = score(vid2pred, vid2GTs)
    return scores


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    # for idx in idxs[1:]:
    for idx in idxs:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = { v: i for i, v in enumerate(vid2pred.keys()) }
    refs = { vid2idx[vid]: GTs for vid, GTs in vid2GTs.items() }
    hypos = { vid2idx[vid]: [ pred ] for vid, pred in vid2pred.items() }

    for i in range(10):
        video_id = random.choice(list(vid2idx.keys()))

        print('\n', '%s:' % video_id, '\n ref: %s' % refs[vid2idx[video_id]][0],
              '\n pred: %s' % hypos[vid2idx[video_id]][0])
    print('\n')

    scores = calc_scores(refs, hypos)
    return scores


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    final_scores["METEOR"] = 0.28
    return final_scores

def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.module.load_state_dict(checkpoint['cap_gen'])
    return model


def save_checkpoint(e, model, ckpt_fpath,val_scores, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'scores':val_scores,
        'cap_gen': model.module.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)

def cls_to_dict(cls):
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([ str(vid), pred, GTs ])
            fout.write("{}\n".format(line))

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls