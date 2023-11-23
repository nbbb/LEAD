import argparse
import logging
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fastai.vision import *
import time
from torch.backends import cudnn
from utils import Config, Logger,count_parameters,\
    train,test,evaluate,save_checkpoint,load_checkpoint
from utils_rl import train_rl
def _set_random_seed(seed):
    assert seed=='random' or isinstance(seed, int) or seed is None
    if seed is None:
        return
    if isinstance(seed, int):
        seed = seed
    elif seed == 'random':
        seed = random.randint(0, 0xffffff)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    logging.warning('\nYou have chosen to seed training. \n'+
                        'The seed is {}\n'.format(seed)+
                        'This will slow down your training!')

def fix_param(model):
    for layer in model.named_parameters():
        if 'two' in layer[0]:
            layer[1].requires_grad= not layer[1].requires_grad
    return model


def _get_databaunch(config):
    assert config.dataset_corpus in [ 'MSVD', 'MSR-VTT' ], f"{config.dataset_corpus} is not a dataset."
    if config.dataset_corpus == "MSVD":
        corpus = MSVD(config)
    elif config.dataset_corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    logging.info(f'{len(corpus.train_data_loader)} training items found.')
    logging.info(f'{len(corpus.val_data_loader)} valid items found.')
    return corpus


def _get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    return model

def _log_train_val(loss,scores=None,phase='train'):
    if phase=='train':
        logging.info("train loss: {0:.3f} (CE_1: {1:.3f}  CE_2: {2:.3f} )"
                     .format(loss['total'], loss['ce_1'],loss['ce_2']))
    elif phase=='val':
        logging.info("val loss: {0:.3f} (CE_1: {1:.3f}  CE_2: {2:.3f} )"
                     .format(loss['total'], loss['ce_1'],
                    loss['ce_2']))

    if scores is not None:
        logging.info("\nscores: {}".format(scores))
def _comp2base(best_scores_base,best_val_scores,val_scores):
    current_scores=0.0
    best_scores=0.0

    for metrics,score in best_scores_base.items():
        current_scores+=val_scores[metrics]/score
        best_scores+=best_val_scores[metrics]/score
    save_flag=True if current_scores > best_scores else False
    return save_flag

def _trining_ce(model ,data,optimizer,lr_scheduler,config):
    best_val_scores = {'CIDEr': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0.}
    best_scores_base = config.dataset_best_scores_base
    best_epoch = 0
    best_ckpt_fpath = None

    for e in range(1, config.training_epochs + 1):
        #
        if e <= 15:
            ntz_flag = False
            model.module.teacher_forcing_ratio = 0.

        else:
            # if not ntz_flag:
            #     model = fix_param(model)
            #     opt_type = getattr(torch.optim, config.optimizer_type)
            #     optimizer = opt_type(model.parameters(), lr=config.optimizer_lr,
            #                          weight_decay=config.optimizer_wd,
            #                          amsgrad=True)
            # print('固定next层')
            ntz_flag = True
            model.module.teacher_forcing_ratio = 1.
            # model.module.teacher_forcing_ratio = config.optimizer_K / (e-13 + config.optimizer_K)
        train_loss = train(e, model, optimizer, data.train_data_loader, data.vocab, config.optimizer_clip_grad,
                           ntz_flag)
        _log_train_val(train_loss)
        if e >= config.save_checkpoint_save_from and e % config.save_checkpoint_save_every == 0:
            """ Validation """
            val_loss = test(model, data.val_data_loader, data.vocab)
            val_scores = evaluate(data.test_data_loader, model, data.vocab, config)
            _log_train_val(val_loss, val_scores, phase='val')
            if e >= config.optimizer_lr_decay_start_from:
                lr_scheduler.step(val_loss['total'])
        if e >= config.save_checkpoint_save_from and _comp2base(best_scores_base, best_val_scores, val_scores):
            ckpt_fpath = config.save_checkpoint_ckpt_fpath_tpl.format(
                (config.global_name + config.timestamp), 'best')
            logging.info("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
            save_checkpoint(e, model, ckpt_fpath,val_scores, config)
            best_epoch = e
            best_val_scores = val_scores
            best_ckpt_fpath = ckpt_fpath
    return best_val_scores,best_epoch,best_ckpt_fpath

def _training_rl(model ,data,optimizer,lr_scheduler,config):
    checkpoint = torch.load(config.rl_base_model_path)
    # val_scores = evaluate(data.test_data_loader, model, data.vocab, config)
    best_val_scores={'CIDEr': checkpoint['scores']['CIDEr'], 'Bleu_4': checkpoint['scores']['Bleu_4'],
                     'METEOR': checkpoint['scores']['METEOR'], 'ROUGE_L': checkpoint['scores']['ROUGE_L']}
    best_scores_base = config.dataset_best_scores_base
    best_epoch = checkpoint['epoch']
    best_ckpt_fpath = None
    for e in range(best_epoch, config.training_epochs + best_epoch):
        train_loss = train_rl(e, model, optimizer, data.train_data_loader, data.vocab, config.optimizer_clip_grad)
        logging.info("train loss: {} ".format(train_loss['total']))
        if e >= config.save_checkpoint_save_from and e % config.save_checkpoint_save_every == 0:
            """ Validation """
            val_loss = test(model, data.val_data_loader, data.vocab)
            val_scores = evaluate(data.test_data_loader, model, data.vocab, config)
            _log_train_val(val_loss, val_scores, phase='val')
            if e >= config.optimizer_lr_decay_start_from:
                lr_scheduler.step(val_loss['total'])
        if e >= config.save_checkpoint_save_from and _comp2base(best_scores_base, best_val_scores, val_scores):
            ckpt_fpath = config.save_checkpoint_ckpt_fpath_tpl.format(
                (config.global_name + config.timestamp), 'best_rl')
            logging.info("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
            save_checkpoint(e, model, ckpt_fpath, val_scores,config)
            best_epoch = e
            best_val_scores = val_scores
            best_ckpt_fpath = ckpt_fpath
    return best_val_scores,best_epoch,best_ckpt_fpath

def _taining_and_sorce(model ,data,optimizer,lr_scheduler,config):
    if not config.rl_is_rl:
        best_val_scores,best_epoch,best_ckpt_fpath=_trining_ce(model, data, optimizer, lr_scheduler, config)
    else:
        best_val_scores, best_epoch, best_ckpt_fpath = _training_rl(model, data, optimizer, lr_scheduler, config)

    """ Test with Best Model """
    logging.info("\n\n\n[BEST]")

    logging.info('The Best CIDEr is {},epoch is {}'.format(best_val_scores['CIDEr'], best_epoch))

    best_model = load_checkpoint(model, best_ckpt_fpath)
    test_scores = evaluate(data.test_data_loader, best_model, data.vocab,config)
    _log_train_val( None,test_scores,phase='test')


def _learning( model ,data,config):
    opt_type = getattr(torch.optim, config.optimizer_type)
    # optimizer = opt_type(model.parameters(), lr=config.optimizer_lr, weight_decay=config.optimizer_wd, amsgrad=True)
    optimizer = opt_type(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=config.optimizer_lr ,
                         weight_decay=config.optimizer_wd,
                         amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.optimizer_scheduler_gamma,
                                     patience=config.optimizer_scheduler_patience, verbose=True)

    _taining_and_sorce(model ,data,optimizer,lr_scheduler,config)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--phase', type=str, default=None, choices=['train', 'test'])
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    config = Config(args.config)
    if args.name is not None: config.global_name = args.name
    if args.phase is not None: config.global_phase = args.phase
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint

    timestamp = time.strftime("_%m%d-%H:%M:%S", time.gmtime())
    config.timestamp = timestamp

    Logger.init(config.global_workdir, config.global_name, config.global_phase+timestamp)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)

    logging.info('Construct dataset.')
    data  = _get_databaunch(config)
    config.dataset_n_vocabs=data.vocab.n_vocabs

    logging.info('Construct model.')
    model = _get_model(config)
    logging.info("#params: {} ".format(count_parameters(model)))
    model.cuda()
    model = nn.DataParallel(model)

    if config.rl_is_rl:
        assert  config.rl_base_model_path is not None
        load_checkpoint(model, config.rl_base_model_path)
        logging.info("Load checkpoint from {}".format(config.rl_base_model_path))
    logging.info("start learning")
    _learning( model ,data,config)



if __name__ == '__main__':
    print('cider +ce2')
    main()
