import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric
from thop import profile

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()
    


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
    

class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
    
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert len(weight) == self.n_classes, "Weight length must equal n_classes"
        assert inputs.size() == target.size(), f'Predict {inputs.size()} & target {target.size()} shape mismatch'
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        weight_sum = sum(weight)
        if weight_sum == 0:
            return torch.tensor(0.0, device=inputs.device)
        return loss / weight_sum


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target.long())
        loss_dice = self.diceloss(pred, target, softmax=True, weight=[0,1])
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss
    
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        assert target.dtype == torch.float32, "Target must be float32 with values 0.0 or 1.0"
        return self.bce(pred, target)

    
class BceDiceLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, smooth=1e-6):
        super(BceDiceLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth

    def forward(self, pred, target):
        _, C, _, _ = pred.size()
        if target.dtype == torch.float32:
            target = target.long()
        bce_loss = F.cross_entropy(pred, target)
        prob = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        intersection = (prob * target_one_hot).sum(dim=(2, 3))  # [B, C]
        union = prob.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff.mean() 

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
 


class AdaptiveHierarchicalLoss(nn.Module):
    def __init__(self, num_layers=4, tau=0.2, alpha=0.8, grad_clip=1.0, device=torch.device('cuda')):
        super().__init__()
        self.num_layers = num_layers
        self.tau = tau
        self.alpha = alpha
        self.grad_clip = grad_clip
        self.bce_dice_loss = BceDiceLoss()
        self.register_buffer('weights', torch.ones(num_layers) / num_layers)
        self.device = device or torch.device('cpu')  # 默认CPU
        self.weights = torch.ones(num_layers, device=self.device) / num_layers
        self.grad_buffer = []

    def forward(self, final_output, target, layer_outputs):

        assert len(layer_outputs) == self.num_layers
        losses = []
        self.grad_buffer.clear()  
        
        for l in range(self.num_layers):
            _, H, W, _= layer_outputs[l].shape
            target_resized = F.interpolate(target.unsqueeze(1).float(), size=(H, W), mode='bilinear').float()
            loss_l = self.bce_dice_loss(layer_outputs[l].permute(0, 3, 1, 2), target_resized.squeeze(1)).mean()
            loss_l.register_hook(self._grad_hook(l))
            losses.append(loss_l)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.parameters(),  
                    max_norm=self.grad_clip,
                    error_if_nonfinite=True
                )
        

        loss_final = self.bce_dice_loss(final_output, target)
        sum(losses).backward(retain_graph=True)
        
        with torch.no_grad():
            if len(self.grad_buffer) == self.num_layers:
                grads = torch.stack(self.grad_buffer)          
                grads = (grads - grads.mean()) / (grads.std() + 1e-8)  
                weights_new = torch.softmax(grads / self.tau, dim=0)
                self.weights = self.alpha * self.weights + (1 - self.alpha) * weights_new
                self.weights /= self.weights.sum()
        total_loss = sum(w * loss for w, loss in zip(self.weights, losses))
        total_loss += 0.75 * loss_final     
        return total_loss

    def _grad_hook(self, l):
        def hook(grad):
            grad_mag = grad.abs().mean()
            self.grad_buffer.append(grad_mag)
        return hook
    def get_weights(self):
        return self.weights.detach().cpu().numpy()





def calculate_tp_fp_tn_fn(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, fp, tn, fn

def calculate_metric_percase(pred, gt, threshold):
    if len(pred.shape) == 3 and pred.shape[0] > 1:
        assert pred.shape[0] == gt.shape[0]
        sum_dice, sum_hd95, sum_recall, sum_iou, sum_acc, sum_spe, sum_ignore = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for i in range(pred.shape[0]):
            metrics = calculate_metric_percase(pred[i], gt[i], threshold)
            sum_dice += metrics[0]
            sum_hd95 += metrics[1]
            sum_recall += metrics[2]
            sum_iou += metrics[3]
            sum_acc += metrics[4]
            sum_spe += metrics[5]
            sum_ignore += metrics[6]
        return sum_dice, sum_hd95, sum_recall, sum_iou, sum_acc, sum_spe, sum_ignore
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    tp, fp, tn, fn = calculate_tp_fp_tn_fn(pred, gt)
    total = tp + fp + tn + fn
    recall = tp / (tp + fn + 1e-10) 
    iou = tp / (tp + fp + fn + 1e-10)
    accuracy = (tp + tn) / (total + 1e-10)
    spe = tn / (tn + fp + 1e-10)
    

    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        sum_pred = np.sum(pred)
        sum_true = np.sum(gt)
        denominator = sum_pred + sum_true
        if denominator == 0:
            denominator = 1.0  
        return 2 * tp / (2*tp+fp+fn), hd95, recall, iou, accuracy, spe, 1
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, tn/(tn+fp), tn/(tn+fp+1e-10), 0 
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0.0, threshold, 0.0, 0.0, tn/(tn+fn), 1.0, 1
    else:
        return 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1  



def test_score(image, label, net, classes, patch_size=[224, 224]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3) 
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            
            net.eval()
            with torch.no_grad():
                outputs, _ = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    elif len(image.shape) == 4:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :, :]
            x, y = slice.shape[0], slice.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y, 1), order=0)  
            input = torch.from_numpy(slice).unsqueeze(0).float().cuda().permute(0,3,1,2)
            net.eval()
            with torch.no_grad():
                outputs, _ = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            xxx, _ = net(input)
            out = torch.argmax(torch.softmax(xxx, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    threshold = max(patch_size) * 1.414
    metric_list.append(calculate_metric_percase(prediction, label, threshold))
    return metric_list

class Early_stop:
    def __init__(self, patience, tolerance, save_dir):
        self.cnt = 0
        self.patience = 15
        self.best_loss = None
        self.tolerance = tolerance
        self.stop = False
        self.save_dir = save_dir

    def __call__(self, test_loss, epoch, model):
        if self.best_loss is None:
            self.best_loss = test_loss
            torch.save(model.state_dict(), self.save_dir + f'epoch_{epoch} test_loss {test_loss}.pth')
        elif test_loss > self.best_loss + self.tolerance:
            self.cnt += 1
            if self.cnt >= self.patience:
                self.stop = True
        elif test_loss < self.best_loss:
            self.best_loss = test_loss
            torch.save(model.state_dict(), self.save_dir + f'epoch_{epoch} test_loss {test_loss}.pth')
            self.cnt = 0
        else:
            self.cnt = 0
