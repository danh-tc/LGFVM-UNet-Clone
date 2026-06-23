import numpy as np
from tqdm import tqdm
from medpy import metric
from torch.cuda.amp import autocast as autocast
import torch
from utils import test_score
from colorama import Fore
import time
from sklearn.metrics import confusion_matrix

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    MultiScaleLoss = True,
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float().permute(0,3,1,2), targets.cuda(non_blocking=True).float()   
        if config.amp:
            with autocast():
                out, dec_outputs = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif MultiScaleLoss is True:
            out, dec_outputs = model(images)
            loss = criterion(out, targets, dec_outputs)
            loss.backward()


            optimizer.step()
        else:
            out, dec_outputs = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    mean_loss = np.mean(loss_list)
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss


def val_one_epoch(test_loader, model, epoch, logger, config):
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        for data in tqdm(test_loader):
            img, msk = data['image'], data['label']#B H W
            metric_i = test_score(img, msk, model, classes=config.num_classes,
                                  patch_size=[config.input_size_h, config.input_size_w])
            metric_list += np.array(metric_i)

        metric_list = metric_list[:, :-1] / metric_list[:, -1].reshape(-1, 1)

        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f mean_recall %f mean_IOU %f mean_acc %f mean_spe %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3], metric_list[i-1][4], metric_list[i-1][5]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_recall = np.mean(metric_list, axis=0)[2]
        mean_IOU = np.mean(metric_list, axis=0)[3]
        mean_acc = np.mean(metric_list, axis=0)[4]
        mean_spe = np.mean(metric_list, axis=0)[5]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95},  mean_recall: {mean_recall}, mean_IOU: {mean_IOU}, mean_acc: {mean_acc}, mean_spe: {mean_spe}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95

def val_one_epochV2(test_loader, model, epoch, logger, config):
    stime = time.time()
    model.eval()
    with torch.no_grad():
        total_pred = []
        total_target = []

        for data in tqdm(test_loader):
            img, msk = data['image'], data['label']#B H W
            model.eval()
            outputs, _ = model(img.permute(0, 3, 1, 2))
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            outputs = outputs.cpu().detach().numpy()#H W
            msk = msk.cpu().detach().numpy()
            total_pred.append(outputs)
            total_target.append(msk)

        dsc_avg = 0
        hd95_avg = 0
        sen_avg = 0
        miou_avg = 0
        acc_avg = 0
        spe_avg = 0
        TP_total = 0
        TN_total = 0
        FP_total = 0
        FN_total = 0
        for cur_num_classes in range(1, config.num_classes):
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            hd95_total = 0.0
            num = 0
            for pred_batch, target_batch in zip(total_pred, total_target):
                for i in range(pred_batch.shape[0]):
                    pred = pred_batch[i]  
                    target = target_batch[i]  
                    pred_flat = pred.ravel()
                    target_flat = target.ravel()
                    TP += np.sum((pred_flat == cur_num_classes) & (target_flat == cur_num_classes))
                    TN += np.sum((pred_flat != cur_num_classes) & (target_flat != cur_num_classes))
                    FP += np.sum((pred_flat == cur_num_classes) & (target_flat != cur_num_classes))
                    FN += np.sum((pred_flat != cur_num_classes) & (target_flat == cur_num_classes))
                    TP_total += TP
                    TN_total += TN
                    FP_total += FP
                    FN_total += FN
                    pred = (pred == cur_num_classes)
                    target = (target == cur_num_classes)
                    if pred.sum() == 0 and target.sum() > 0:
                        hd95_total += config.input_size_h * 1.414
                        num +=1
                    elif pred.sum() > 0 and target.sum() == 0:
                        hd95_total += config.input_size_h * 1.414
                        num +=1
                    elif pred.sum() > 0 and target.sum() > 0:
                        hd95_total += metric.binary.hd95(pred, target)
                        num +=1
                    else:
                        hd95_total += 0
                        num += 1
            epsilon = 1e-8
            dsc = (2 * TP) / (2 * TP + FP + FN + epsilon)
            dsc_avg += dsc
            sen = TP / (TP + FN + epsilon)
            sen_avg += sen
            spe = TN / (TN + FP + epsilon)
            spe_avg += spe
            acc = (TP + TN) / (TP + TN + FP + FN + epsilon)
            acc_avg += acc
            iou_foreground = TP / (TP + FP + FN + epsilon)
            miou = iou_foreground
            miou_avg += miou
            hd95 = hd95_total / num
            hd95_avg += hd95
            logger.info('Mean class %d mean_dice %f mean_hd95 %f mean_recall %f mean_IOU %f mean_acc %f mean_spe %f' % (cur_num_classes, dsc, hd95, sen, miou, acc, spe))      

        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {(2 * TP_total) / (2 * TP_total + FP_total + FN_total + epsilon)}, mean_hd95: {hd95_avg/ (config.num_classes - 1)},  mean_recall: {TP_total / (TP_total + FN_total + epsilon)}, mean_IOU: {TP_total / (TP_total + FP_total + FN_total + epsilon)}, mean_acc: {(TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total + epsilon)}, mean_spe: {TN_total / (TN_total + FP_total + epsilon)}, time(s): {etime-stime:.2f}'
        log_info = f'val epoch: {epoch}, mean_dice: {dsc_avg / (config.num_classes - 1)}, mean_hd95: {hd95_avg/ (config.num_classes - 1)},  mean_recall: {sen_avg / (config.num_classes - 1)}, mean_IOU: {miou_avg / (config.num_classes - 1)}, mean_acc: {acc_avg / (config.num_classes - 1)}, mean_spe: {spe_avg / (config.num_classes - 1)}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return dsc, hd95


