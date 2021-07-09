import torch
from models.pipeline5 import Pipeline
from models.minus_pipeline_512d_multitask import Multi_task_model
from data.ABAW2_data import ABAW2_multitask_data,compute_class_weight,ABAW2_Exp_data,ABAW2_VA_data,ABAW2_AU_data
from torch.nn import MSELoss,CrossEntropyLoss
import numpy as np
import torch.nn.functional as F
from eval_metrics import metric_for_AU,metric_for_Exp,metric_for_VA
import torchvision.transforms.transforms as transforms
from torch.utils import data
from torch import optim


def train(epoch, loader, net, optimizer, best_AU_score,best_VA_score,best_Exp_score):
    print("train {} epoch".format(epoch))
    tmp_V_prob,tmp_A_prob,tmp_AU_prob,tmp_exp_prob = [], [], [],[]
    tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [],[]

    loss_sum = 0.0
    step = 1
    net = net.train()
    for batch_idx, (img,label_AU,label_V,label_A,label_exp,name) in enumerate(loader):
        if use_cuda:
            img = img.cuda()
            label_AU,label_V,label_A,label_exp = label_AU.cuda(),label_V.float().cuda(),label_A.float().cuda(),label_exp.cuda()
        optimizer.zero_grad()
        VA_out,AU_out,final_AU_out,Exp_out = net(img)

        #VA_loss
        VA_loss = crit_VA(VA_out[:,0],label_V) + crit_VA(VA_out[:,1],label_A)

        VA_out = VA_out.detach().cpu().numpy()
        tmp_V_prob.extend(VA_out[:,0])
        tmp_V_label.extend(label_V.cpu().numpy())

        tmp_A_prob.extend(VA_out[:, 1])
        tmp_A_label.extend(label_A.cpu().numpy())

        #Exp_loss
        Exp_loss = crit_Exp(Exp_out,label_exp)
        Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
        for i in range(len(name)):
            v = max(Exp_prediction[i])
            index = np.where(Exp_prediction[i] == v)[0][0]
            tmp_exp_prob.append(index)
            tmp_exp_label.append(label_exp[i].cpu().numpy())

        #AU_loss
        AU_loss = 0
        for i in range(12):
            t_input = AU_out[:, i, :]
            t_target = label_AU[:, i].long()
            t_input2 = final_AU_out[:, i, :]
            t_loss = AU_class_weight[i] * (crit_AU(t_input, t_target) + crit_AU(t_input2, t_target))
            AU_loss += t_loss
        prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
        tmp_AU_prob.extend(prob.data.cpu().numpy())
        tmp_AU_label.extend(label_AU.data.cpu().numpy())

        loss = VA_loss + Exp_loss + AU_loss
        print(loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        step += 1

        if step % 100 == 0:
            avg_loss = loss_sum / 100

            #VA metric
            ccc_v,ccc_a = metric_for_VA(tmp_V_label,tmp_A_label,tmp_V_prob,tmp_A_prob)
            final_VA_score = (ccc_v + ccc_a)/2
            #Exp metric
            Exp_F1,Exp_acc,Exp_F1_per_class = metric_for_Exp(tmp_exp_label,tmp_exp_prob)
            final_Exp_score = 0.67*Exp_F1 + 0.33*Exp_acc
            #AU metric
            AU_F1,AU_acc,AU_F1_per_class = metric_for_AU(tmp_AU_label,tmp_AU_prob)
            final_AU_score = 0.5*AU_F1 + 0.5*AU_acc


            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print('  train set - VA, Exp ,AU score     = {:.8f} , {:.8f} , {:.8f}'.format(final_VA_score,final_Exp_score,final_AU_score))
            with open("./log/ABAW2_multi_task_train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {0}, step: {1},  Loss: {2}, ccc_v: {3} ccc_a: {4} Exp_F1: {5} Exp_acc: {6} AU_F1: {7} AU_acc: {8} VA_score: {9} Exp_score: {10} AU_socre: {11}\n".format(epoch, step,
                                                                                                 avg_loss,
                                                                                                 ccc_v,ccc_a,Exp_F1,Exp_acc,AU_F1,AU_acc,final_VA_score,final_Exp_score,final_AU_score))
            tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
            tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []
            loss_sum = 0.0

        if step % 5000 == 0:
            net = net.eval()
            torch.save(net.state_dict(), './checkpoints/AU_multi_task_best.pth')
            best_AU_score = test_AU(epoch, AU_testloader, net, best_AU_score)
            best_VA_score = test_VA(epoch, VA_testloader, net, best_VA_score)
            best_Exp_score = test_Exp(epoch, Exp_testloader, net, best_Exp_score)
            net = net.train()
    return best_AU_score,best_VA_score,best_Exp_score


def test_VA(epoch, loader, net, best_acc):
    print("train {} epoch".format(epoch))
    tmp_V_prob,tmp_A_prob = [], []
    tmp_V_label, tmp_A_label = [], []
    net = net.eval()
    VA_loss_sum = 0
    with torch.no_grad():
        for batch_idx, (img, label_V, label_A, name) in enumerate(loader):
            if use_cuda:
                img = img.cuda()
                label_V, label_A= label_V.float().cuda(),label_A.float().cuda()
            VA_out, _, _,_ = net(img,output_AU=False,output_Exp=False)
            VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A)
            VA_out = VA_out.detach().cpu().numpy()
            tmp_V_prob.extend(VA_out[:, 0])
            tmp_V_label.extend(label_V.cpu().numpy())

            tmp_A_prob.extend(VA_out[:, 1])
            tmp_A_label.extend(label_A.cpu().numpy())
            VA_loss_sum += VA_loss.item()
            print(VA_loss.item())

        ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
        final_VA_score = (ccc_v + ccc_a) / 2

        if final_VA_score > best_acc:
            best_acc = final_VA_score
            torch.save(net.state_dict(), './checkpoints/VA_multi_task_best.pth')

        with open("./log/VA_multi_task_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, ccc_v: {2} ccc_a: {3}  VA_score: {4} \n".format(
                    epoch, VA_loss_sum/len(loader),
                          ccc_v, ccc_a,  final_VA_score))
        return best_acc


def test_Exp(epoch, loader, net, best_acc):
    print("train {} epoch".format(epoch))
    tmp_Exp_prob,tmp_Exp_label = [], []
    net = net.eval()
    Exp_loss_sum = 0
    with torch.no_grad():
        for batch_idx, (img, label_exp, name) in enumerate(loader):
            if use_cuda:
                img = img.cuda()
                label_exp= label_exp.cuda()
            _,_, _,Exp_out = net(img,output_VA=False,output_AU=False)
            Exp_loss = crit_Exp(Exp_out, label_exp)
            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
            for i in range(len(name)):
                v = max(Exp_prediction[i])
                index = np.where(Exp_prediction[i] == v)[0][0]
                tmp_Exp_prob.append(index)
                tmp_Exp_label.append(label_exp[i].cpu().numpy())
            print(Exp_loss.item())
            Exp_loss_sum += Exp_loss.item()

        Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_Exp_label, tmp_Exp_prob)
        final_Exp_score = 0.67 * Exp_F1 + 0.33 * Exp_acc

        if final_Exp_score > best_acc:
            best_acc = final_Exp_score
            torch.save(net.state_dict(), './checkpoints/Exp_multi_task_best.pth')

        with open("./log/Exp_multi_task_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, Exp_F1: {2} Exp_acc: {3}  Exp_score: {4} \n".format(
                    epoch, Exp_loss_sum/len(loader),
                          Exp_F1, Exp_acc,  final_Exp_score))
            Exp_F1_per_class = [str(k) for k in Exp_F1_per_class ]
            log_file.write(" ".join(Exp_F1_per_class))
            log_file.write("\n")
        return best_acc



def test_AU(epoch, loader, net, best_acc):
    print("train {} epoch".format(epoch))
    tmp_AU_prob,tmp_AU_label = [], []
    net = net.eval()
    AU_loss_sum = 0
    with torch.no_grad():
        for batch_idx, (img, label_AU, name) in enumerate(loader):
            if use_cuda:
                img = img.cuda()
                label_AU= label_AU.cuda()
            _, AU_out,final_AU_out, _ = net(img, output_VA=False, output_Exp=False)
            AU_loss = 0
            for i in range(12):
                t_input = AU_out[:, i, :]
                t_target = label_AU[:, i].long()
                t_input2 = final_AU_out[:, i, :]
                t_loss = AU_class_weight[i] * (crit_AU(t_input, t_target) + crit_AU(t_input2, t_target))
                AU_loss += t_loss
            prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
            tmp_AU_prob.extend(prob.data.cpu().numpy())
            tmp_AU_label.extend(label_AU.data.cpu().numpy())
            print(AU_loss)
            AU_loss_sum += AU_loss

        AU_F1, AU_acc, AU_F1_per_class = metric_for_AU(tmp_AU_label, tmp_AU_prob)
        final_AU_score = 0.5 * AU_F1 + 0.5 * AU_acc

        if final_AU_score > best_acc:
            best_acc = final_AU_score
            torch.save(net.state_dict(), './checkpoints/AU_multi_task_best.pth')

        with open("./log/AU_multi_task_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, AU_F1: {2} AU_acc: {3}  AU_score: {4} \n".format(
                    epoch, AU_loss_sum/len(loader),
                          AU_F1, AU_acc,  final_AU_score))
            AU_F1_per_class = [str(k) for k in AU_F1_per_class]
            log_file.write(" ".join(AU_F1_per_class))
            log_file.write("\n")
        return best_acc



if __name__ == '__main__':
    use_cuda = True
    train_csv = "data/multi_data.csv"
    img_path = "./"

    AU_class_weight = compute_class_weight(train_csv, "AU")
    Exp_class_weight = compute_class_weight(train_csv, "Exp")
    print(AU_class_weight,Exp_class_weight)

    # loss
    crit_VA = MSELoss()
    crit_AU = CrossEntropyLoss()
    crit_Exp = CrossEntropyLoss(Exp_class_weight.cuda())

    #data
    transform1 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()])

    trainset = ABAW2_multitask_data(train_csv, img_path,transform)
    AU_testset = ABAW2_AU_data("data/new_AU_validation.csv", img_path,transform1)
    VA_testset = ABAW2_VA_data("data/new_VA_validation.csv", img_path,transform1)
    Exp_testset = ABAW2_Exp_data("data/new_Exp_validation.csv", img_path,transform1)

    bz = 30
    trainloader = data.DataLoader(trainset, batch_size= bz, num_workers=0, shuffle=True)
    AU_testloader = data.DataLoader(AU_testset, batch_size=bz * 3, num_workers=0)
    VA_testloader = data.DataLoader(VA_testset, batch_size=bz * 3, num_workers=0)
    Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz * 3, num_workers=0)

    #model
    emb_net = Pipeline()
    net = Multi_task_model(emb_net)
    if use_cuda:
        net = net.cuda()

    #training parameters
    best_AU_score,best_VA_score,best_Exp_score = 0,0,0
    lr = 0.002
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                          weight_decay=5e-4)

    for i in range(50):
        best_AU_score,best_VA_score,best_Exp_score = train(i, trainloader, net, optimizer, best_AU_score,best_VA_score,best_Exp_score)
        best_AU_score = test_AU(i, AU_testloader, net,  best_AU_score)
        best_VA_score = test_VA(i, VA_testloader, net,  best_VA_score)
        best_Exp_score = test_Exp(i, Exp_testloader, net,  best_Exp_score)



