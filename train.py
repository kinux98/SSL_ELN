import argparse
import copy
import datetime
import os
import os.path
import random
import time
from os.path import join as pjn

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

import wandb
from dataset.common import ConfusionMatrix
from modeling.ELN import ELNetwork
from modeling.deeplab import DeepLab, Decoder
from utils.main_do import *


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class TrainManager(object):
    def __init__(
            self,
            enc,
            dec,
            eln,
            dec_list,
            optimizer,
            args,
            save_dir,
            labeled_loader,
            val_loader,
            unlabeled_loader=None,
            scaler=None,
            num_classes=None,
            epoch=None,
            ):
        self.enc2 = copy.deepcopy(enc).cuda()
        self.dec2 = copy.deepcopy(dec).cuda()
        
        self.enc = enc
        self.dec = dec

        self.dec_list = dec_list

        self.eln = eln

        for param in self.enc2.parameters():
            param.detach_()
        
        for param in self.dec2.parameters():
            param.detach_()

        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.args = args
        self.save_dir = save_dir
        self.orig_cwd = os.getcwd()
        self.scaler = scaler

        self.num_classes = num_classes
        self.start_epoch = epoch
        self.conf_mat = ConfusionMatrix(self.num_classes)
        self.conf_mat2 = ConfusionMatrix(self.num_classes)

        self.warm_up_epoch = self.args.pre_epoch+self.args.eln_epoch

    def validate(self, epoch, loader):

        if epoch < self.args.pre_epoch:
            standard = epoch % 10 
        elif (epoch >= self.args.pre_epoch) and (epoch < self.warm_up_epoch):
            standard = epoch % 30
        else:
            standard = epoch % 1
        
        if standard == 0:
            self.enc.eval()
            self.dec.eval()
            self.enc2.eval()
            self.dec2.eval()
            self.eln.eval()

            for sd in self.dec_list:
                sd.eval()


            with autocast():
                with torch.no_grad():
                    for idx, (image, target) in tqdm(enumerate(loader), leave=True, position=0, disable=True):
                        image, target = image.cuda() , target.cuda() 
                        x4, x1 = self.enc(image)
                        output, _  = self.dec(x4, x1)
                        self.conf_mat.update(target, output)
                        if epoch >= self.warm_up_epoch:
                            x4, x1 = self.enc2(image)
                            output2, _ = self.dec2(x4, x1)
                            self.conf_mat2.update(target, output2)
                acc_global, acc, iu = self.conf_mat.compute()
                global_correct = acc_global.item() * 100
                Iou = ['{:.2f}'.format(i) for i in (iu * 100).tolist()]
                mIou = iu.mean().item() * 100
                pix_acc, m_iou, iou = global_correct, mIou, Iou
                self.conf_mat.reset()
                wandb.log({'validation/student_net' : m_iou})

                # data = [[label, val] for (label, val) in zip(categories_voc, iou)]
                # ta = wandb.Table(data=data, columns = ["label", "pred_value"])
                # wandb.log({"Iou per Label" : wandb.plot.bar(ta, "label","pred_value", title="Validation")})
                if epoch >= self.warm_up_epoch:
                    acc_global, acc, iu = self.conf_mat2.compute()
                    global_correct = acc_global.item() * 100
                    Iou = ['{:.2f}'.format(i) for i in (iu * 100).tolist()]
                    mIou = iu.mean().item() * 100
                    pix_acc, m_iou, iou = global_correct, mIou, Iou
                    self.conf_mat2.reset()
                    wandb.log({'validation/teacher_net' : m_iou})
                        # data = [[label, val] for (label, val) in zip(categories_voc, iou)]
                        # ta = wandb.Table(data=data, columns = ["label", "pred_value"])
                        # wandb.log({"Iou per Label2" : wandb.plot.bar(ta, "label","pred_value", title="Validation2")})

    def update_teacher(self):
        _contrast_momentum = 0.995

        for mean_param, param in zip(self.enc2.parameters(), self.enc.parameters()):
                mean_param.data.mul_(_contrast_momentum).add_(1 - _contrast_momentum, param.data)
        for mean_param, param in zip(self.dec2.parameters(), self.dec.parameters()):
                mean_param.data.mul_(_contrast_momentum).add_(1 - _contrast_momentum, param.data)

    def train_both(self, epoch, dataloader, iter_per_epoch):
        self.enc .train()
        self.dec .train()
        self.enc2 .train()
        self.dec2 .train()
        self.eln .train()

        for sd in self.dec_list:
            sd.train()

        if epoch == self.warm_up_epoch:
            self.enc2.load_state_dict(self.enc.state_dict())
            self.dec2.load_state_dict(self.dec.state_dict())

        if epoch >= (self.warm_up_epoch+1):
            self.enc.load_state_dict(self.enc2.state_dict())
            self.dec.load_state_dict(self.dec2.state_dict())
        
        for batch_idx in range(iter_per_epoch):
            (inputs_l, labels_l), (inputs_ul, labels_ul, inputs_ul_aug) = next(dataloader)
            if epoch >= self.warm_up_epoch:
                inputs_ul = inputs_ul.cuda() 
                inputs_ul_aug = inputs_ul_aug.cuda() 
                labels_ul = labels_ul.cuda() 
            else:
                inputs_ul = inputs_ul.cuda()
                labels_ul = labels_ul.cuda() 

            inputs_l = inputs_l.cuda() 
            labels_l = labels_l.cuda() 

            losses_list = []

            with autocast():
                ##### sup part #####
                sup_loss = typical_segtrain(self, inputs_l, labels_l)

                
                losses_list.append(sup_loss)
                
                wandb.log({
                        "numeric_metric/sup_loss" : sup_loss,
                    })
                
                if epoch >= self.args.pre_epoch:
                    sub_seg_loss = typical_submodule_segtrain_loss_constrain(
                        self, inputs_l, labels_l, sup_loss
                    )
                    losses_list.append(sub_seg_loss)
                    losses_list2 = train_ELN(self, inputs_l, labels_l)
                    losses_list.append(losses_list2)

                ##### semi-sup part #####
                if epoch >= self.warm_up_epoch:
                    cls_output_unsup, feature_vec_un = get_cls_output_and_featuer_vec(self, inputs_ul)
                    cls_output_unsup_aug, feature_vec_un_aug = get_cls_output_and_featuer_vec(self, inputs_ul_aug)

                    with torch.no_grad():
                        # Get ELN binary map
                        a, b = self.enc2(inputs_ul)
                        cls_output_unsup_k, feature_vec_un_k = self.dec2(a,b, gt_mode=True)
                        correction_map = self.eln(inputs_ul, cls_output_unsup_k, gt_mode=True)
                        final_indices = cls_output_unsup_k.argmax(1).cuda()
                        final_candid = torch.round(torch.sigmoid(correction_map)).squeeze(1)
                        
                    ce_pseudo_loss_aug, _ = calculate_pseudo_loss(cls_output_unsup_aug, final_candid, final_indices)
                    ce_pseudo_loss, _ = calculate_pseudo_loss(cls_output_unsup, final_candid, final_indices)

                    pxl_dist_aug = pixelwisecontrastiveloss(self, feature_vec_un_k.detach(), feature_vec_un_aug,  final_candid, final_indices)
                    pxl_dist = pixelwisecontrastiveloss(self, feature_vec_un_k.detach(), feature_vec_un, final_candid, final_indices)

                    losses_list.append(ce_pseudo_loss_aug)
                    losses_list.append(ce_pseudo_loss)

                    losses_list.append(pxl_dist_aug)
                    losses_list.append(pxl_dist)

                    wandb.log({
                        "numeric_metric/pseudo_loss_aug" : ce_pseudo_loss_aug,
                        "numeric_metric/pseudo_loss" : ce_pseudo_loss,
                        "numeric_metric/pxl_contra_aug" : pxl_dist_aug,
                        "numeric_metric/pxl_loss" : pxl_dist
                    })

            
            self.optimizer.zero_grad()
            t_loss = total_loss(losses_list)
            if torch.isnan(t_loss).any():
                print("NAN!")
                exit(-1)
            self.scaler.scale(t_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if epoch >= self.warm_up_epoch:
                self.update_teacher()

            wandb.log({'epoch': epoch})

            del inputs_ul_aug
            del inputs_l, inputs_ul
            del labels_l, labels_ul
            
            if epoch >= self.warm_up_epoch:
                del feature_vec_un
                del cls_output_unsup
                del final_candid, final_indices
                del cls_output_unsup_k
                del pxl_dist, pxl_dist_aug

                   
    def train(self):
        start = time.time()
        epoch = 0
        iter_per_epoch = 266
        print(f"  The checkpoints files are saved to "
              f"'{os.path.relpath(self.save_dir, self.orig_cwd)}'")

        end_epoch = self.start_epoch + self.args.num_epochs

        print("  labeled data : ", len(self.labeled_loader))
        print("  unlabeled data :", len(self.unlabeled_loader))
        print("  training iter per epoch :", iter_per_epoch)
        print("  warm-up epoch : ", self.warm_up_epoch)
        print("  ELN train epoch : ", self.args.eln_epoch)
        print("  pre train epoch : ", self.args.pre_epoch)

        both_dataloader = iter(zip(cycle(self.labeled_loader), cycle(self.unlabeled_loader)))

        for epoch in tqdm(range(self.start_epoch, end_epoch), desc='epochs', leave=False):  
            self.train_both(epoch, both_dataloader, iter_per_epoch) 
            self.validate(epoch, self.val_loader) 

            if epoch >= self.warm_up_epoch:
                self.save_ckpt(epoch)
        end = time.time()
               
        print("Total training time : ", str(datetime.timedelta(seconds=(int(end)-int(start)))))
        print("Finish.")

    def save_ckpt(self, epoch):
        if epoch % self.args.save_ckpt == 0:
            nm = f'epoch_{epoch:04d}.pth'
            if not os.path.isdir(pjn('checkpoints', self.save_dir)):
                os.mkdir(pjn('checkpoints', self.save_dir))
            fpath=pjn('checkpoints', self.save_dir, nm)

            d = {
                    'epoch': epoch,

                    'enc_state_dict' : self.enc.state_dict(),
                    'dec_state_dict' : self.dec.state_dict(),

                    'scaler_state_dict' : self.scaler.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),

                    'eln_state_dict' : self.eln.state_dict(),
                }
            for idx, sd in enumerate(self.dec_list):
                d['dec_'+str(idx)] = sd.state_dict()

            torch.save(d, fpath)

def main(args):
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    wandb.init(project="ssl_v2")

    orig_cwd = os.getcwd()

    if args.dataset == 'voc':
        num_classes = num_classes_voc
        
    elif args.dataset == 'city':
        num_classes = num_classes_city
    else:
        raise ValueError

    enc = DeepLab(resnet_name=args.backbone_name).cuda()
    dec = Decoder(num_cls=num_classes).cuda()
    eln = ELNetwork(num_classes=num_classes).cuda()    

    trainable_params = [

        ## main model's parameters ##
        {'params': list(filter(lambda p:p.requires_grad, enc.get_backbone_params())), 'lr':args.lr/10},
        {'params': list(filter(lambda p:p.requires_grad, dec.get_other_params())), 'lr':args.lr},

        # ELN's parameters ##
        {'params': list(filter(lambda p:p.requires_grad, eln.get_other_params())), 'lr':args.lr},
        {'params': list(filter(lambda p:p.requires_grad, eln.get_backbone_params())), 'lr':args.lr/10},
        ]

    decoder_list = []
    print(" Temperature value : ", args.temp)
    print(" Generate 2 auxiliary decoders.")
    
    for _ in range(2):
        sdec = Decoder(num_cls=num_classes).cuda()
        decoder_list.append(sdec)
        trainable_params.append(
            {'params': list(filter(lambda p:p.requires_grad, sdec.get_other_params())), 'lr':args.lr},
        )
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() 
    epoch_save = 0
    if args.pretrained_ckpt:
        
        def load_state_dict(model, pretrain):
            model_dict = {}
            state_dict = model.state_dict()
            for k, v in pretrain.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)

        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        print(f"  Using pretrained model only and its checkpoint "
              f"'{args.pretrained_ckpt}'")
        
        epoch_save = loaded_struct['epoch']
        
        load_state_dict(enc, loaded_struct['enc_state_dict'])
        load_state_dict(dec, loaded_struct['dec_state_dict'])
        try:
            load_state_dict(eln, loaded_struct['eln_state_dict'])
        except:
            print("fail to load eln ")
            pass
            
        try:
            for idx, sd in enumerate(decoder_list):
                load_state_dict(sd, loaded_struct['dec_'+str(idx)])

        except:
            print("fail to load sub modules ")
            pass

        try:
            scaler.load_state_dict(loaded_struct['scaler_state_dict'])
        except:
            print("fail to load scaler's dict ")
            pass

        try:
            print(loaded_struct.keys())
            optimizer.load_state_dict(loaded_struct['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        except:
            print("fail to load optimizer's dict")
            pass
        
    now = datetime.datetime.now()
    ti = now.strftime('%Y-%m-%d-%H-%M-%S')

    if args.exp_name:
        save_dir = os.getcwd() + "/checkpoints/" + str(args.exp_name)
        wandb.run.name = str(args.exp_name)
    else:
        save_dir = os.getcwd() + "/checkpoints/" + str(ti)
        wandb.run.name = str(ti)

    labeled_loader, unlabeled_loader, val_loader = init(
        batch_size_labeled=args.batch_size_labeled,
        batch_size_unlabeled=args.batch_size_unlabeled,
        sets_id=args.sets_id,
        split=args.train_split,
        data_set=args.dataset,
    )

    trainer = TrainManager(
        enc = enc,
        dec = dec,
        eln = eln,
        dec_list = decoder_list,
        optimizer=optimizer,
        args=args,
        save_dir=save_dir,
        labeled_loader=labeled_loader,
        val_loader = val_loader,
        unlabeled_loader=unlabeled_loader,
        scaler=scaler,
        num_classes=num_classes,
        epoch=epoch_save
        )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Train/Evaluate on PASCAL VOC 2012(voc)/Cityscapes(city) (default: voc)')
    parser.add_argument('--seed', type=int, default=20170890,
                        help='Random seed ')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Load pretrained weight, write path to weight (default: None)')
    
    parser.add_argument('--batch-size-labeled', type=int, default=6,    
                        help='Batch size for labeled data (default: 4)')
    parser.add_argument('--batch-size-unlabeled', type=int, default=6,
                        help='Batch size for pseudo labeled data (default: 4)')

    parser.add_argument('--sets-id', type=int, default=0,
                        help='Different random splits(0/1/2) (default: 0)')
    parser.add_argument('--train-split', type=int, default=20,
                        help='percentage of splited training data(label) (default : 20 (5%, 1/20)')
    
    parser.add_argument('--save-ckpt', type=int, default=10,
                        help='number of epoch save current weight? (default: 10)')

    parser.add_argument('--backbone', type=str, default='resnet')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='start epoch (default: 0)')
    parser.add_argument('--num-epochs', type=int, default=500,
                        help='end epoch (default: 500)')
    
    parser.add_argument('--backbone_name', type=int, choices=[50,101], default=101)

    parser.add_argument('--pre_epoch', type=int,default=40) 
    parser.add_argument('--eln_epoch', type=int,default=50) 

    parser.add_argument('--temp', type=float, default=0.5)

    parser.add_argument('--lr', type=float, default=1e-4, #for adamw
                        help='Initial learning rate (default: 1e-4)')

    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for adamW (default: 1e-5)')
                
    args = parser.parse_args()
    main(args)
