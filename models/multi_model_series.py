from models.pipeline5 import Pipeline
import torch.nn as nn
import torch


class Multi_task_series_model(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",freeze=False):
        super(Multi_task_series_model,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model
        self.emb_dim = emb_dim

        if pretrained:
            state_dict = torch.load(pretrained)
            self.exp_emb_net.load_state_dict(state_dict)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 7)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(24, 24)
        self.AU_inter = nn.Linear(192,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 7)

    def forward_baseline(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out


    def forward_dropout(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        final_AU_out = AU_out
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_inter = self.Exp_dropout(Exp_inter)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))

        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_inter = self.VA_dropout(VA_inter)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))
        VA_out = self.tanh1(VA_out)

        return VA_out,AU_out,final_AU_out,Exp_out


    def forward(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        final_AU_out = AU_out
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out

    def forward_mutual(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)

        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out_inter = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out_inter),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        Exp_VA_inter = torch.cat((Exp_out,VA_out_inter),dim=1)
        Exp_VA_inter = torch.relu(self.AU_linear_mutual(self.EXP_VA_BN(Exp_VA_inter)))
        Exp_VA_AU_inter = torch.cat((Exp_VA_inter,AU_out_),dim=1)
        final_AU_out = self.AU_final_linear_mutual(Exp_VA_AU_inter)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_inter_mutual = torch.cat((Exp_inter,VA_out_inter),dim=1)
        Exp_out = self.Exp_linear_mutual(self.Exp_BN_mutual(Exp_inter_mutual))

        return VA_out,AU_out,final_AU_out,Exp_out
