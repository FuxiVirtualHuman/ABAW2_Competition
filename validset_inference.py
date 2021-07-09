import os

import torch
from models.pipeline5 import Pipeline
from models.multi_model_series import Multi_task_series_model
from data_new.ABAW2_data import compute_class_weight,ABAW2_test_data,ABAW2_Exp_data,ABAW2_AU_data,ABAW2_VA_data
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
import numpy as np
import torch.nn.functional as F
from eval_metrics import metric_for_AU, metric_for_Exp, metric_for_VA
import torchvision.transforms.transforms as transforms
from torch.utils import data
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import pandas as pd


def metric_for_AU_ml(gt, pred, class_num=12):
    # compute_F1,acc
    F1 = []
    gt = np.array(gt)
    pred = np.array(pred)
    cate_acc = np.sum((np.array(pred > 0, dtype=np.float)) == gt) / (gt.shape[0] * gt.shape[1])
    # print(pred.shape)
    for type in range(class_num):
        gt_ = gt[:, type]
        pred_ = pred[:, type]
        new_pred = ((pred_ >= 0.) * 1).flatten()
        F1.append(f1_score(gt_.flatten(), new_pred))
    F1_mean = np.mean(F1)
    # compute total acc
    counts = gt.shape[0]
    accs = 0
    for i in range(counts):
        pred_label = ((pred[i, :] >= 0.) * 1).flatten()
        gg = gt[i].flatten()
        j = 0
        for k in range(12):
            if int(gg[k]) == int(pred_label[k]):
                j += 1
        if j == 12:
            accs += 1

    acc = 1.0 * accs / counts

    return F1_mean, acc, F1, cate_acc


def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    # --------
    # >>> from sklearn.metrics import concordance_correlation_coefficient
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> concordance_correlation_coefficient(y_true, y_pred)
    # 0.97678916827853024
    # """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def CCC_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)
    return 1 - ccc


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

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
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
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
    '''
    label:(h,w), dtype:torch.long，可使用torch.LongTensor(mask)将shape为(h,w)的索引mask转为torch.long类型
    N:num_class，0也算一个类别
    '''
    size = list(label.size())
    label = label.view(-1).long()  # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def pred_testset_result(names_list, indexs_list, probs_list, labels_list,ck_model_name=None,cls=None):
    import pandas as pd
    exp_data = pd.DataFrame()
    exp_data['img'], exp_data['pred_label'],exp_data['probs'],exp_data['true_label']  = names_list, indexs_list,probs_list,labels_list
    os.makedirs(f'predict/valid_set/stage{stage}/{ck_model_name}', exist_ok=True)
    exp_data.to_csv(f'predict/valid_set/stage{stage}/{ck_model_name}/{cls}.csv')


def test_Exp(epoch, loader, net, cls=None,ck_model_name=None):
    print("train {} epoch".format(epoch))
    net = net.eval()
    names_list, indexs_list, probs_list = [], [], []
    labels_list =[]
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label,name) in t:
            # if batch_idx>1:
            #     break

            if use_cuda:
                img = img.cuda()
            if model_name == 'baseline':
                _, _, _, Exp_out = net(img, output_VA=False, output_AU=False)

            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()

            names_list.extend(list(name))
            indexs_list.extend(list(np.argmax(Exp_prediction, axis=1)))
            labels_list.extend(list(label.numpy()))
            probs_list.extend(list(Exp_prediction))

        pred_testset_result(names_list, indexs_list, probs_list, labels_list,ck_model_name=ck_model_name,cls=cls)


def test_AU(epoch, loader, net,cls=None,ck_model_name=None):
    print("train {} epoch".format(epoch))
    tmp_AU_prob, tmp_AU_label,tmp_AU_prob_label = [], [], []
    tmp_AU_prob_0 = []
    net = net.eval()
    names = []
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label,name) in t:
            # if batch_idx > 1:
            #     break

            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            if model_name == 'baseline':
                _, AU_out, final_AU_out, _ = net(img, output_VA=False, output_Exp=False)

            prob = final_AU_out[:,:,1]
            prob_0 = final_AU_out[:,:,0]
            tmp_AU_prob.extend(prob.data.cpu().numpy())
            tmp_AU_prob_0.extend(prob_0.data.cpu().numpy())
            tmp_AU_prob_label.extend(np.where(prob.data.cpu().numpy()>0,1,0))

            names.extend(name)
            tmp_AU_label.extend(label.data.cpu().numpy())
    data = pd.DataFrame()
    data["img"] = names
    data["AU"] = tmp_AU_prob_label
    data['prob'] = tmp_AU_prob
    data['prob_0'] = tmp_AU_prob_0
    data['label'] = tmp_AU_label
    os.makedirs(f'predict/valid_set/stage{stage}/{ck_model_name}', exist_ok=True)
    data.to_csv(f'predict/valid_set/stage{stage}/{ck_model_name}/{cls}.csv')

def test_VA(epoch, loader, net, ck_model_name=None,cls=None):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob = [], []
    net = net.eval()
    names = []
    data = pd.DataFrame()
    step = 0
    with torch.no_grad():
        for batch_idx, (img, label_v,label_a,name) in tqdm(enumerate(loader)):
            if use_cuda:
                img = img.cuda()
            VA_out, _, _, _ = net(img, output_AU=False, output_Exp=False)
            VA_out = VA_out.detach().cpu().numpy()
            tmp_V_prob.extend(VA_out[:, 0])
            tmp_A_prob.extend(VA_out[:,1])
            names.extend(name)
            step += 1

    data["img"] = names
    data["V"] = tmp_V_prob
    data['A'] = tmp_A_prob
    os.makedirs(f'predict/valid_set/stage{stage}/{ck_model_name}', exist_ok=True)
    data.to_csv(f'predict/valid_set/stage{stage}/{ck_model_name}/{cls}.csv')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ck_root_paths = ["checkpoints/ABAW2/multi_task/baseline_va:CCC+SmoothL1_au:ML_CE_and_sCE_exp:LabelSmoothCE+0.79exp_add_pseudo_DFEW_BP4D","checkpoints/ABAW2/multi_task/baseline_va:CCC+SmoothL1_au:ML_CE_and_sCE_exp:LabelSmoothCE+new_tri_tuple_0.766exp_add_pseudo_DFEW_BP4D"]

    stage = 1
    for ck_root_path in  ck_root_paths:
        print(ck_root_path)
        print('stage:{}'.format(stage))

        use_cuda = True
        task = 'MULTI'  # ['EXP','AU','VA','MULTI']
        if task in ['EXP', 'AU', 'VA']:
            task_type = 'single'
        elif task == 'MULTI':
            task_type = 'multi'

        model_name = 'baseline'  # ['baseline','mutual','dropout']
        VA_LOSS = 'CCC'  # ['MAE','MSE','SmoothL1','CCC']
        AU_LOSS = 'ML_CE'  # ['LabelSmoothCE','CE','WCE','ML_CE']
        EXP_LOSS = 'LabelSmoothCE'  # ['LabelSmoothCE','CE']


        # print(remark)
        print(f'************** NOW IS {task} TASK. ******************')

        img_path = "./"

        transform1 = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()])

        bz = 80

        if task == 'MULTI':
            Exp_testset = ABAW2_Exp_data("data_new/new_Exp_validation_1.csv", img_path, transform=transform1)
            Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz * 3, num_workers=8)
            AU_testset = ABAW2_AU_data("data_new/new_AU_validation_1.csv", img_path, transform=transform1)
            AU_testloader = data.DataLoader(AU_testset, batch_size=bz * 3, num_workers=8)
            VA_testset = ABAW2_VA_data("data_new/new_VA_validation_1.csv", img_path, transform=transform1)
            VA_testloader = data.DataLoader(VA_testset, batch_size=bz * 3, num_workers=8)

        # model
        emb_net = Pipeline()
        # net = Multi_task_model(emb_net)
        net = Multi_task_series_model(emb_net)
        if use_cuda:
            net = net.cuda()

        # training parameters
        best_AU_score, best_VA_score, best_Exp_score = 0, 0, 0
        lr = 0.002
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, 2, 1e-5)

        for cls in ['Exp', 'AU', 'VA']:
            if cls == 'AU':
                ck_path = ck_root_path + "/AU_best_cate_score.pth"
                if not os.path.exists(ck_path):
                    ck_path = ck_root_path + '/AU_best.pth'
            else:
                ck_path = ck_root_path + "/{}_best.pth".format(cls)

            state_dict = torch.load(ck_path)
            net.load_state_dict(state_dict)
            ck_model_name = ck_path.split('/')[-2]

            if cls == 'Exp':
                test_Exp(0, Exp_testloader, net,  cls=cls,ck_model_name=ck_model_name)
            elif cls == 'AU':
                test_AU(0,AU_testloader,net,cls=cls,ck_model_name=ck_model_name)
            elif cls == 'VA':
                test_VA(0,VA_testloader,net,cls=cls,ck_model_name=ck_model_name)
