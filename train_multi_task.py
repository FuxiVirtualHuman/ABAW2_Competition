import os

import torch
from models.pipeline5 import Pipeline
from models.minus_pipeline_512d_multitask import Multi_task_model
from models.multi_model_series import Multi_task_series_model
from data_new.ABAW2_data import compute_class_weight, ABAW2_Exp_data, ABAW2_VA_data, ABAW2_AU_data, \
    ABAW2_multitask_data2
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
import numpy as np
import torch.nn.functional as F
from eval_metrics import metric_for_AU, metric_for_Exp, metric_for_VA, metric_for_AU_mlce
import torchvision.transforms.transforms as transforms
from torch.utils import data
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.autograd import Variable


def CCC_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))+1e-10)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)+1e-10)
    return 1 - ccc

def CCC_SmoothL1(x,y):
    loss1 = SmoothL1Loss()(x,y)
    loss2 = CCC_loss(x,y)
    return loss1 + loss2

class FocalLoss_TOPK(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss_TOPK, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
            for i in range(class_num):
                self.alpha[i, :] = 0.25
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        batch_loss = torch.topk(torch.squeeze(batch_loss), int(inputs.shape[0] * 0.2))[0]

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
            for i in range(class_num):
                self.alpha[i, :] = 0.25
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def sCE_and_focal_loss(y_pred, y_true):
    loss1 = LabelSmoothingCrossEntropy()(y_pred, y_true)
    loss2 = FocalLoss(class_num=7)(y_pred, y_true)
    return loss1 + loss2


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = y_pred[:, :, 1]
    y_true = y_true[:, :, 1]
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)

    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return torch.mean(neg_loss + pos_loss)

def ml_ce_and_focal_topk_loss(y_pred,y_true):
    loss1 = FocalLoss_TOPK()(y_pred,y_true)
    loss2 = multilabel_categorical_crossentropy(y_pred,y_true)
    return loss1+loss2


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon * 2) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1).long()  
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label)  
    size.append(N)  
    return ones.view(*size)


def train(epoch, loader, net, optimizer, best_AU_score, best_VA_score, best_Exp_score,best_cate_score):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
    tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (au_img, va_img, exp_img, label_AU, label_V, label_A, label_exp) in t:

        if use_cuda:
            au_img, va_img, exp_img = au_img.cuda(), va_img.cuda(), exp_img.cuda()
            label_AU, label_V, label_A, label_exp = label_AU.cuda(), label_V.float().cuda(), label_A.float().cuda(), label_exp.cuda()

        optimizer.zero_grad()
        with autocast():
            if model_name == 'baseline':
                VA_out, _, _, _ = net(va_img, output_VA=True, output_AU=False, output_Exp=False)
            elif model_name == 'mutual':
                VA_out, _, _, _ = net.forward_mutual(va_img, output_VA=True, output_AU=False, output_Exp=False)
            elif model_name == 'dropout':
                VA_out, _, _, _ = net.forward_dropout(va_img, output_VA=True, output_AU=False, output_Exp=False)
            VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A)

        VA_out = VA_out.detach().cpu().numpy()
        tmp_V_prob.extend(VA_out[:, 0])
        tmp_V_label.extend(label_V.cpu().numpy())

        tmp_A_prob.extend(VA_out[:, 1])
        tmp_A_label.extend(label_A.cpu().numpy())

        with autocast():
            if model_name == 'baseline':
                _, _, _, Exp_out = net(exp_img, output_VA=False, output_AU=False, output_Exp=True)
            elif model_name == 'mutual':
                _, _, _, Exp_out = net.forward_mutual(exp_img, output_VA=False, output_AU=False, output_Exp=True)
            elif model_name == 'dropout':
                _, _, _, Exp_out = net.forward_dropout(va_img, output_VA=True, output_AU=False, output_Exp=False)
            # Exp_loss
            Exp_loss = crit_Exp(Exp_out, label_exp)

        Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
        for i in range(Exp_out.shape[0]):
            v = max(Exp_prediction[i])
            index = np.where(Exp_prediction[i] == v)[0][0]
            tmp_exp_prob.append(index)
            tmp_exp_label.append(label_exp[i].cpu().numpy())

        # AU_loss
        AU_loss = 0
        with autocast():
            if model_name == 'baseline':
                _, AU_out, final_AU_out, _ = net(au_img, output_VA=False, output_AU=True, output_Exp=False)
            elif model_name == 'mutual':
                _, AU_out, final_AU_out, _ = net.forward_mutual(au_img, output_VA=False, output_AU=True,
                                                                output_Exp=False)
            elif model_name == 'dropout':
                _, AU_out, final_AU_out, _ = net.forward_dropout(va_img, output_VA=True, output_AU=False,
                                                                 output_Exp=False)
            if AU_LOSS == 'ML_CE':
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            elif AU_LOSS == 'ML_CE_and_sCE':
                for i in range(12):
                    t_target = label_AU[:, i].long()
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * LabelSmoothingCrossEntropy()(t_input, t_target)
                    AU_loss += t_loss
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            else:
                for i in range(12):
                    t_target = label_AU[:, i].long()
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * crit_AU(t_input, t_target)
                    AU_loss += t_loss

        if 'ML_CE' not in AU_LOSS:
            prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
        else:
            prob = final_AU_out[:, :, 1]
        tmp_AU_prob.extend(prob.data.cpu().numpy())
        tmp_AU_label.extend(label_AU.data.cpu().numpy())

        loss = VA_loss + Exp_loss + AU_loss
        t.set_postfix(train_loss=loss.item(), VA_loss=VA_loss.item(), Exp_loss=Exp_loss.item(), AU_loss=AU_loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(step)

        loss_sum += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = loss_sum / 200

            # VA metric
            ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
            final_VA_score = (ccc_v + ccc_a) / 2
            # Exp metric
            Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_exp_label, tmp_exp_prob)
            final_Exp_score = 0.67 * Exp_F1 + 0.33 * Exp_acc
            # AU metric
            if 'ML_CE' not in AU_LOSS:
                AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU(tmp_AU_label, tmp_AU_prob)
            else:
                AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU_mlce(tmp_AU_label, tmp_AU_prob)
            final_AU_score = 0.5 * AU_F1 + 0.5 * AU_acc
            final_cate_score = 0.5 * AU_F1 + 0.5 * cate_acc

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print(
                '  train set - VA, Exp ,AU_strict,AU_cate score     = {:.8f} , {:.8f} , {:.8f}, {:.8f}'.format(final_VA_score, final_Exp_score,
                                                                                        final_AU_score, final_cate_score))
            with open(f"{log_save_path}/train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {0}, step: {1},  Loss: {2}, ccc_v: {3} ccc_a: {4} Exp_F1: {5} Exp_acc: {6} AU_F1: {7} AU_acc: {8} VA_score: {9} Exp_score: {10} AU_socre: {11} AU_cate_acc:{12} AU_cate_score:{13}\n".format(
                        epoch, step,
                        avg_loss,
                        ccc_v, ccc_a, Exp_F1, Exp_acc, AU_F1, AU_acc, final_VA_score, final_Exp_score, final_AU_score,cate_acc,final_cate_score))
            tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
            tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []
            loss_sum = 0.0

        if step % 3000 == 0:
            net = net.eval()
            best_VA_score = test_VA(epoch, VA_testloader, net, best_VA_score, step)
            net = net.train()
    return best_AU_score, best_VA_score, best_Exp_score,best_cate_score


def train_Exp(epoch, loader, net, optimizer, best_Exp_score):
    print("train {} epoch".format(epoch))
    tmp_exp_prob = []
    tmp_exp_label = []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (exp_img, label_exp, name) in t:
        if use_cuda:
            exp_img = exp_img.cuda()
            label_exp = label_exp.cuda()

        optimizer.zero_grad()

        _, _, _, Exp_out = net(exp_img, output_VA=False, output_AU=False, output_Exp=True)
        # Exp_loss
        Exp_loss = crit_Exp(Exp_out, label_exp)
        Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
        for i in range(Exp_out.shape[0]):
            v = max(Exp_prediction[i])
            index = np.where(Exp_prediction[i] == v)[0][0]
            tmp_exp_prob.append(index)
            tmp_exp_label.append(label_exp[i].cpu().numpy())

        loss = Exp_loss
        t.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = loss_sum / 100

            # Exp metric
            Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_exp_label, tmp_exp_prob)
            final_Exp_score = 0.67 * Exp_F1 + 0.33 * Exp_acc

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print('  train set - Exp     = {:.8f}'.format(final_Exp_score))
            with open(f"./log/ABAW2/Exp_{task_type}_task_train_2nd.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {}, Exp_F1: {} Exp_acc: {}  Exp_score: {} \n".format(epoch, step,
                                                                                                      avg_loss, Exp_F1,
                                                                                                      Exp_acc,
                                                                                                      final_Exp_score))
            tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
            tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []
            loss_sum = 0.0

        if step % 2500 == 0:
            net = net.eval()
            best_Exp_score = test_Exp(epoch, Exp_testloader, net, best_Exp_score, step)
            net = net.train()
    return best_Exp_score


def train_VA(epoch, loader, net, optimizer, best_VA_score):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob = [], []
    tmp_V_label, tmp_A_label = [], []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (va_img, label_V, label_A, name) in t:

        if use_cuda:
            va_img = va_img.cuda()
            label_V, label_A = label_V.float().cuda(), label_A.float().cuda()

        optimizer.zero_grad()
        VA_out, _, _, _ = net(va_img, output_VA=True, output_AU=False, output_Exp=False)

        # VA_loss
        VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A)
        VA_out = VA_out.detach().cpu().numpy()
        tmp_V_prob.extend(VA_out[:, 0])
        tmp_V_label.extend(label_V.cpu().numpy())

        tmp_A_prob.extend(VA_out[:, 1])
        tmp_A_label.extend(label_A.cpu().numpy())

        loss = VA_loss
        t.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = loss_sum / 100

            # VA metric
            ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
            final_VA_score = (ccc_v + ccc_a) / 2

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print('  train set - VA     = {:.8f}'.format(final_VA_score))
            with open(f"./log/ABAW2/VA_{task_type}_task_train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {}, ccc_v: {} ccc_a: {} VA_score: {} \n".format(epoch, step,
                                                                                                 avg_loss, ccc_v, ccc_a,
                                                                                                 final_VA_score))
            tmp_V_prob, tmp_A_prob = [], []
            tmp_V_label, tmp_A_label = [], []
            loss_sum = 0.0

        if step % 2500 == 0:
            net = net.eval()
            best_VA_score = test_VA(epoch, VA_testloader, net, best_VA_score, step)
            net = net.train()
    return best_VA_score


def train_AU(epoch, loader, net, optimizer, best_AU_score):
    print("train {} epoch".format(epoch))
    tmp_AU_prob = []
    tmp_AU_label = []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (au_img, label_AU, name) in t:
        if use_cuda:
            au_img = au_img.cuda()
            label_AU = label_AU.cuda()

        optimizer.zero_grad()

        # AU_loss
        AU_loss = 0
        _, AU_out, final_AU_out, _ = net(au_img, output_VA=False, output_AU=True, output_Exp=False)
        for i in range(12):
            t_input = AU_out[:, i, :]
            t_target = label_AU[:, i].long()
            t_input2 = final_AU_out[:, i, :]
            t_loss = AU_class_weight[i] * (crit_AU(t_input, t_target) + crit_AU(t_input2, t_target))
            AU_loss += t_loss
        prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
        tmp_AU_prob.extend(prob.data.cpu().numpy())
        tmp_AU_label.extend(label_AU.data.cpu().numpy())

        loss = AU_loss
        t.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = loss_sum / 100

            # AU metric
            AU_F1, AU_acc, AU_F1_per_class = metric_for_AU(tmp_AU_label, tmp_AU_prob)
            final_AU_score = 0.5 * AU_F1 + 0.5 * AU_acc

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print('  train set - AU score     =  {:.8f}'.format(final_AU_score))
            with open(f"./log/ABAW2/AU_{task_type}_task_train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {},  AU_F1: {} AU_acc: {}  AU_socre: {}\n".format(epoch, step,
                                                                                                   avg_loss,
                                                                                                   AU_F1, AU_acc,
                                                                                                   final_AU_score))
            tmp_AU_prob = []
            tmp_AU_label = []
            loss_sum = 0.0

        if step % 2500 == 0:
            net = net.eval()
            best_AU_score = test_AU(epoch, AU_testloader, net, best_AU_score, step)
            net = net.train()
    return best_AU_score


def test_VA(epoch, loader, net, best_acc, step=0):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob = [], []
    tmp_V_label, tmp_A_label = [], []
    net = net.eval()
    VA_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label_V, label_A, name) in t:

            if use_cuda:
                img = img.cuda()
                label_V, label_A = label_V.float().cuda(), label_A.float().cuda()
            if model_name == 'baseline':
                VA_out, _, _, _ = net(img, output_AU=False, output_Exp=False)
            elif model_name == 'mutual':
                VA_out, _, _, _ = net.forward_mutual(img, output_AU=False, output_Exp=False)
            elif model_name == 'dropout':
                VA_out, _, _, _ = net.forward_dropout(img, output_AU=False, output_Exp=False)
            VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A)
            VA_out = VA_out.detach().cpu().numpy()
            tmp_V_prob.extend(VA_out[:, 0])
            tmp_V_label.extend(label_V.cpu().numpy())

            tmp_A_prob.extend(VA_out[:, 1])
            tmp_A_label.extend(label_A.cpu().numpy())
            VA_loss_sum += VA_loss.item()
            t.set_postfix(test_VA_loss=VA_loss.item())

        ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
        final_VA_score = (ccc_v + ccc_a) / 2

        if final_VA_score > best_acc:
            best_acc = final_VA_score
            torch.save(net.state_dict(), f'{ck_save_path}/VA_best.pth')

        with open(f"{log_save_path}/VA_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, ccc_v: {2} ccc_a: {3}  VA_score: {4} \n".format(
                    epoch, VA_loss_sum / len(loader),
                    ccc_v, ccc_a, final_VA_score))
        return best_acc


def test_Exp(epoch, loader, net, best_acc, step=0):
    print("train {} epoch".format(epoch))
    tmp_Exp_prob, tmp_Exp_label = [], []
    net = net.eval()
    Exp_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label_exp, name) in t:

            if use_cuda:
                img = img.cuda()
                label_exp = label_exp.cuda()
            if model_name == 'baseline':
                _, _, _, Exp_out = net(img, output_VA=False, output_AU=False)
            elif model_name == 'mutual':
                _, _, _, Exp_out = net.forward_mutual(img, output_VA=False, output_AU=False)
            elif model_name == 'dropout':
                _, _, _, Exp_out = net.forward_dropout(img, output_VA=False, output_AU=False)
            Exp_loss = crit_Exp(Exp_out, label_exp)
            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
            for i in range(len(name)):
                v = max(Exp_prediction[i])
                index = np.where(Exp_prediction[i] == v)[0][0]
                tmp_Exp_prob.append(index)
                tmp_Exp_label.append(label_exp[i].cpu().numpy())
            t.set_postfix(test_Exp_loss=Exp_loss.item())
            Exp_loss_sum += Exp_loss.item()

        Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_Exp_label, tmp_Exp_prob)
        final_Exp_score = 0.67 * Exp_F1 + 0.33 * Exp_acc

        if final_Exp_score > best_acc:
            best_acc = final_Exp_score
            torch.save(net.state_dict(), f'{ck_save_path}/Exp_best.pth')

        # torch.save(net.state_dict(), f'{ck_step_save_path}/Exp_epoch{epoch}_step{step}.pth')

        with open(f"{log_save_path}/Exp_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {}, Loss: {}, Exp_F1: {} Exp_acc: {}  Exp_score: {} \n".format(
                    epoch, Exp_loss_sum / len(loader),
                    Exp_F1, Exp_acc, final_Exp_score))
            Exp_F1_per_class = [str(k) for k in Exp_F1_per_class]
            log_file.write(" ".join(Exp_F1_per_class))
            log_file.write("\n")
        return best_acc


def test_AU(epoch, loader, net, best_acc,best_acc_2, step=0):
    print("train {} epoch".format(epoch))
    tmp_AU_prob, tmp_AU_label = [], []
    net = net.eval()
    AU_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label_AU, name) in t:

            if use_cuda:
                img = img.cuda()
                label_AU = label_AU.cuda()
            if model_name == 'baseline':
                _, AU_out, final_AU_out, _ = net(img, output_VA=False, output_Exp=False)
            elif model_name == 'mutual':
                _, AU_out, final_AU_out, _ = net.forward_mutual(img, output_VA=False, output_Exp=False)
            elif model_name == 'dropout':
                _, AU_out, final_AU_out, _ = net.forward_dropout(img, output_VA=False, output_Exp=False)
            AU_loss = 0

            if AU_LOSS == 'ML_CE':
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            elif AU_LOSS == 'ML_CE_and_sCE':
                for i in range(12):
                    t_target = label_AU[:, i].long()
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * LabelSmoothingCrossEntropy()(t_input, t_target)
                    AU_loss += t_loss
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            else:
                for i in range(12):
                    t_target = label_AU[:, i].long()
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * crit_AU(t_input, t_target)
                    AU_loss += t_loss

            if 'ML_CE' not in AU_LOSS:
                prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
            else:
                prob = final_AU_out[:, :, 1]
            tmp_AU_prob.extend(prob.data.cpu().numpy())
            tmp_AU_label.extend(label_AU.data.cpu().numpy())
            t.set_postfix(test_AU_loss=AU_loss)
            AU_loss_sum += AU_loss

        if 'ML_CE' not in AU_LOSS:
            AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU(tmp_AU_label, tmp_AU_prob)
        else:
            AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU_mlce(tmp_AU_label, tmp_AU_prob)
        final_AU_score = 0.5 * AU_F1 + 0.5 * AU_acc
        final_cate_score = 0.5 * AU_F1 + 0.5 * cate_acc


        if final_AU_score > best_acc:
            best_acc = final_AU_score
            torch.save(net.state_dict(), f'{ck_save_path}/AU_best.pth')

        if final_cate_score > best_acc_2:
            best_acc_2 = final_cate_score
            torch.save(net.state_dict(), f'{ck_save_path}/AU_best_cate_score.pth')

        with open(f"{log_save_path}/AU_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, AU_F1: {2} AU_acc: {3}  AU_score: {4} AU_cate_acc:{5} AU_cate_score:{6}\n".format(
                    epoch, AU_loss_sum / len(loader),
                    AU_F1, AU_acc, final_AU_score, cate_acc, final_cate_score))
            AU_F1_per_class = [str(k) for k in AU_F1_per_class]
            log_file.write(" ".join(AU_F1_per_class))
            log_file.write("\n")
        return best_acc,best_acc_2


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = True

    task = 'MULTI'  # ['EXP','AU','VA','MULTI']
    if task in ['EXP', 'AU', 'VA']:
        task_type = 'single'
    elif task == 'MULTI':
        task_type = 'multi'

    model_name = 'baseline'  # ['baseline','mutual','dropout']
    VA_LOSS = 'CCC+SmoothL1'  # ['MAE','MSE','SmoothL1','CCC','CCC+SmoothL1']
    AU_LOSS = 'ML_CE_and_sCE'  # ['LabelSmoothCE','CE','WCE','ML_CE','ML_CE_FocalLoss_TOPK','ML_CE_and_sCE']
    EXP_LOSS = 'LabelSmoothCE'  # ['LabelSmoothCE','CE']
    comment = 'experiment_abaw2'
    
    remark = 'va:' + VA_LOSS + '_au:' + AU_LOSS + '_exp:' + EXP_LOSS + f'+{comment}'
    print(remark)

    ck_save_path = f'./checkpoints/ABAW2/{task_type}_task/{model_name}_{remark}'
    ck_step_save_path = f'./checkpoints/ABAW2/{task_type}_task/{model_name}_{remark}/epoch_weight'
    log_save_path = f'./log/ABAW2/{task_type}_task/{model_name}_{remark}'

    os.makedirs(ck_step_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    print(f'************** NOW IS {task} TASK. ******************')
    train_csv = "data_new/multi_data_new_1.csv"
    img_path = "ABAW2_dataset/original_dataset/crop_face_jpg/"

    AU_class_weight = compute_class_weight(train_csv, "AU").cuda()
    Exp_class_weight = compute_class_weight(train_csv, "Exp")
    print(AU_class_weight, Exp_class_weight)

    # LOSS
    if VA_LOSS == 'MAE':
        crit_VA = L1Loss()
    elif VA_LOSS == 'MSE':
        crit_VA = MSELoss()
    elif VA_LOSS == 'SmoothL1':
        crit_VA = SmoothL1Loss()
    elif VA_LOSS == 'CCC':
        crit_VA = CCC_loss
    elif VA_LOSS == 'CCC+SmoothL1':
        crit_VA = CCC_SmoothL1

    if AU_LOSS == 'CE':
        AU_class_weight = torch.ones_like(AU_class_weight) * 0.1
        crit_AU = CrossEntropyLoss()
    elif AU_LOSS == 'WCE':
        crit_AU = CrossEntropyLoss()
    elif AU_LOSS == 'LabelSmoothCE':
        crit_AU = LabelSmoothingCrossEntropy()
    elif AU_LOSS == 'ML_CE' or AU_LOSS == 'ML_CE_and_sCE':
        crit_AU = multilabel_categorical_crossentropy
    elif AU_LOSS == 'ML_CE_FocalLoss_TOPK':
        crit_AU = ml_ce_and_focal_topk_loss


    if EXP_LOSS == 'CE':
        crit_Exp = CrossEntropyLoss()
    elif EXP_LOSS == 'LabelSmoothCE':
        crit_Exp = LabelSmoothingCrossEntropy()
    elif EXP_LOSS == 'FocalLoss_TOPK':
        crit_Exp = FocalLoss_TOPK(class_num=7)

    # data
    transform1 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()])
    transform2 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(30),
        transforms.GaussianBlur(3),
        transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()])

    bz = 70

    if task == 'EXP':
        Exp_trainset = ABAW2_Exp_data("data_new/new_Exp_training_1.csv", img_path, transform)
        trainset = Exp_trainset
        Exp_testset = ABAW2_Exp_data("data_new/new_Exp_validation_1.csv", img_path, transform1)
        Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz * 3, num_workers=4)
    elif task == 'AU':
        AU_trainset = ABAW2_AU_data("data_new/new_AU_training_1.csv", img_path, transform)
        trainset = AU_trainset
        AU_testset = ABAW2_AU_data("data_new/new_AU_validation_1.csv", img_path, transform1)
        AU_testloader = data.DataLoader(AU_testset, batch_size=bz * 3, num_workers=4)
    elif task == 'VA':
        VA_trainset = ABAW2_VA_data("data_new/new_VA_training_1.csv", img_path, transform)
        trainset = VA_trainset
        VA_testset = ABAW2_VA_data("data_new/new_VA_validation_1.csv", img_path, transform1)
        VA_testloader = data.DataLoader(VA_testset, batch_size=bz * 3, num_workers=4)
    elif task == 'MULTI':
        trainset = ABAW2_multitask_data2(Exp_csv_file="data_new/ori_merge/Exp_training.csv",
                                         VA_csv_file="data_new/ori_merge/VA_training.csv",
                                         AU_csv_file="data_new/ori_merge/AU_training.csv", img_path=img_path,
                                         Exp_VA_transform=transform2, AU_transform=transform)
        Exp_testset = ABAW2_Exp_data("data_new/new_Exp_validation_1.csv", img_path, transform=transform1)
        Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz * 3, num_workers=3)
        AU_testset = ABAW2_AU_data("data_new/new_AU_validation_1.csv", img_path, transform=transform1)
        AU_testloader = data.DataLoader(AU_testset, batch_size=bz * 3, num_workers=3)
        VA_testset = ABAW2_VA_data("data_new/new_VA_validation_1.csv", img_path, transform=transform1)
        VA_testloader = data.DataLoader(VA_testset, batch_size=bz * 3, num_workers=3)

    trainloader = data.DataLoader(trainset, batch_size=bz, num_workers=8, shuffle=True)

    # model
    emb_net = Pipeline()
    net = Multi_task_series_model(emb_net)
    if use_cuda:
        net = net.cuda()

    # training parameters
    best_AU_score, best_VA_score, best_Exp_score,best_cate_score = 0, 0, 0,0
    lr = 0.001
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, 2, 1e-5)

    scaler = GradScaler()
    for i in range(50):
        if task == 'EXP':
            best_Exp_score = train_Exp(i, trainloader, net, optimizer, best_Exp_score)
            best_Exp_score = test_Exp(i, Exp_testloader, net, best_Exp_score)
        elif task == 'VA':
            best_VA_score = train_VA(i, trainloader, net, optimizer, best_VA_score)
            best_VA_score = test_VA(i, VA_testloader, net, best_VA_score)
        elif task == 'AU':
            best_AU_score = train_AU(i, trainloader, net, optimizer, best_AU_score)
            best_AU_score = test_AU(i, AU_testloader, net, best_AU_score)
        elif task == 'MULTI':
            best_AU_score, best_VA_score, best_Exp_score,best_cate_score = train(i, trainloader, net, optimizer, best_AU_score,
                                                                 best_VA_score, best_Exp_score,best_cate_score)
            best_VA_score = test_VA(i, VA_testloader, net, best_VA_score)
            





