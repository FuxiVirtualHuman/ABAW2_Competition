import os
import cv2
import pandas as pd
import torch.utils.data as data
from PIL import Image
import torch
from PIL import ImageFile
import numpy as np
import torchvision.transforms.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ABAW2_multitask_data(data.dataset.Dataset):
    """
    Args:
        # this data type for images with three kinds of annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_AU = self.data['AU']
        self.labels_V = self.data['V']
        self.labels_A = self.data['A']
        self.labels_Exp = self.data['EXP']
        self.img_path = img_path

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path,anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        label_au = self.labels_AU[index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        label_exp = int(self.labels_Exp[index])


        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0

        label_au_ = torch.tensor(label_au_)
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        return anc_img,label_au_,label_V,label_A,label_exp,anc_list



class ABAW2_multitask_data2(data.dataset.Dataset):
    """
    Args:
        # this data type for images with specific annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, Exp_csv_file,VA_csv_file,AU_csv_file,img_path,Exp_VA_transform=None,AU_transform=None):
        self.Exp_VA_transform = Exp_VA_transform
        self.AU_transform = AU_transform
        self.Exp_pd_data = pd.read_csv(Exp_csv_file)
        self.AU_pd_data = pd.read_csv(AU_csv_file)
        self.VA_pd_data = pd.read_csv(VA_csv_file)
        self.Exp_data = self.Exp_pd_data.to_dict("list")
        self.VA_data = self.VA_pd_data.to_dict("list")
        self.AU_data = self.AU_pd_data.to_dict("list")
        self.Exp_imgs = self.Exp_data['img']
        self.AU_imgs = self.AU_data['img']
        self.VA_imgs = self.VA_data['img']
        self.labels_AU = self.AU_data['AU']
        self.labels_V = self.VA_data['V']
        self.labels_A = self.VA_data['A']
        self.labels_Exp = self.Exp_data['Expression']
        self.img_path = img_path

    def __len__(self):
        return max(len(self.Exp_data["img"]),len(self.VA_data["img"]),len(self.AU_data["img"]))

    def __getitem__(self, index):
        va_index,au_index,exp_index = index,index,index
        if index >= len(self.AU_data["img"]):
            au_index  = index - len(self.AU_data["img"])

        if index >= len(self.Exp_data["img"]) and index< 2*len(self.Exp_data["img"]):
            exp_index = index - len(self.Exp_data["img"])

        elif index >= 2*len(self.Exp_data["img"]) and index< 3*len(self.Exp_data["img"]):
            exp_index = index - 2*len(self.Exp_data["img"])
        
        elif index >= 3*len(self.Exp_data["img"]) and index< 4*len(self.Exp_data["img"]):
            exp_index = index - 3*len(self.Exp_data["img"])
        
        elif index >= 4*len(self.Exp_data["img"]) and index< 5*len(self.Exp_data["img"]):
            exp_index = index - 4*len(self.Exp_data["img"])

        if index >= len(self.VA_data["img"]) and index < 2 * len(self.VA_data["img"]):
            va_index = index - len(self.VA_data["img"])

        elif index >= 2 * len(self.VA_data["img"]) and index < 3 * len(self.VA_data["img"]):
            va_index = index - 2 * len(self.VA_data["img"])

        elif index >= 3 * len(self.VA_data["img"]) and index < 4 * len(self.VA_data["img"]):
            va_index = index - 3 * len(self.VA_data["img"])

        elif index >= 4 * len(self.VA_data["img"]) and index < 5 * len(self.VA_data["img"]):
            va_index = index - 4 * len(self.VA_data["img"])

        # exp_data
        exp_name_list = self.Exp_imgs[exp_index]
        exp_img = Image.open(os.path.join(self.img_path,exp_name_list))
        if exp_img.getbands()[0] != 'R':
            exp_img = exp_img.convert('RGB')
        label_exp = int(self.labels_Exp[exp_index])

        #AU_data
        au_name_list = self.AU_imgs[au_index]
        au_img = Image.open(os.path.join(self.img_path, au_name_list))
        if au_img.getbands()[0] != 'R':
            au_img = au_img.convert('RGB')
        label_au = self.labels_AU[au_index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]
        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0
        label_au_ = torch.tensor(label_au_)

        #VA_data
        va_name_list = self.VA_imgs[va_index]
        va_img = Image.open(os.path.join(self.img_path, va_name_list))
        if va_img.getbands()[0] != 'R':
            va_img = va_img.convert('RGB')
        label_V = float(self.labels_V[va_index])
        label_A = float(self.labels_A[va_index])

        if self.AU_transform is not None and self.Exp_VA_transform is not None:
            exp_img = self.Exp_VA_transform(exp_img)
            va_img = self.Exp_VA_transform(va_img)
            au_img = self.AU_transform(au_img)

        return au_img,va_img,exp_img,label_au_,label_V,label_A,label_exp
        
class ABAW2_multitask_embedding_data(data.dataset.Dataset):
    """
    Args:
        # this data type for images with three kinds of annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_AU = self.data['AU']
        self.labels_V = self.data['V']
        self.labels_A = self.data['A']
        self.labels_Exp = self.data['EXP']
        self.emb_before2, self.emb_before1, self.emb_current, self.emb_after1, self.emb_after2 = \
            self.data['emb_before2'], \
            self.data['emb_before1'], self.data['emb_current'], self.data['emb_after1'], self.data[
                'emb_after2']
        self.img_path = img_path
        #self.embs = self.data["emb"]
    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb
    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path , anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        label_au = self.labels_AU[index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        label_exp = int(self.labels_Exp[index])

        emb1, emb2, emb3, emb4, emb5 = self.emb_before2[index], self.emb_before1[
            index], self.emb_current[index], self.emb_after1[index], self.emb_after2[index]
        emb1, emb2, emb3, emb4, emb5 = self.process_emb(emb1), self.process_emb(
            emb2), self.process_emb(emb3), self.process_emb(emb4), self.process_emb(emb5)
        embs = np.concatenate((emb1, emb2, emb3, emb4, emb5), axis=0)

        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0

        label_au_ = torch.tensor(label_au_)
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        return anc_img,label_au_,label_V,label_A,label_exp,embs,anc_list


class ABAW2_multitask_embedding_data2(data.dataset.Dataset):
    """
    Args:
        # this data type for images with specific annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, Exp_csv_file,VA_csv_file,AU_csv_file,img_path,transform=None):
        self.transform = transform
        self.Exp_pd_data = pd.read_csv(Exp_csv_file)
        self.AU_pd_data = pd.read_csv(AU_csv_file)
        self.VA_pd_data = pd.read_csv(VA_csv_file)
        self.Exp_data = self.Exp_pd_data.to_dict("list")
        self.VA_data = self.VA_pd_data.to_dict("list")
        self.AU_data = self.AU_pd_data.to_dict("list")
        self.Exp_imgs = self.Exp_data['img']
        self.AU_imgs = self.AU_data['img']
        self.VA_imgs = self.VA_data['img']
        self.labels_AU = self.AU_data['AU']
        self.labels_V = self.VA_data['V']
        self.labels_A = self.VA_data['A']
        self.labels_Exp = self.Exp_data['Expression']

        self.VA_emb_before2,self.VA_emb_before1,self.VA_emb_current,self.VA_emb_after1,self.VA_emb_after2 = self.VA_data['emb_before2'],\
                                                                                                            self.VA_data['emb_before1'],self.VA_data['emb_current'],self.VA_data['emb_after1'],self.VA_data['emb_after2']
        self.AU_emb_before2, self.AU_emb_before1, self.AU_emb_current, self.AU_emb_after1, self.AU_emb_after2 = \
        self.AU_data['emb_before2'], \
        self.AU_data['emb_before1'], self.AU_data['emb_current'], self.AU_data['emb_after1'], self.AU_data['emb_after2']

        self.Exp_emb_before2, self.Exp_emb_before1, self.Exp_emb_current, self.Exp_emb_after1, self.Exp_emb_after2 = \
            self.Exp_data['emb_before2'], \
            self.Exp_data['emb_before1'], self.Exp_data['emb_current'], self.Exp_data['emb_after1'], self.Exp_data[
                'emb_after2']

        self.img_path = img_path

    def __len__(self):
        return max(len(self.Exp_data["img"]),len(self.VA_data["img"]),len(self.AU_data["img"]))
    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb

    def __getitem__(self, index):
        va_index,au_index,exp_index = index,index,index
        if index >= len(self.AU_data["img"]):
            au_index  = index -  len(self.AU_data["img"])

        if index >= len(self.Exp_data["img"]) and index< 2*len(self.Exp_data["img"]):
            exp_index = index - len(self.Exp_data["img"])

        elif index >= 2*len(self.Exp_data["img"]) and index< 3*len(self.Exp_data["img"]):
            exp_index = index - 2*len(self.Exp_data["img"])
        
        elif index >= 3*len(self.Exp_data["img"]) and index< 4*len(self.Exp_data["img"]):
            exp_index = index - 3*len(self.Exp_data["img"])
        
        elif index >= 4*len(self.Exp_data["img"]) and index< 5*len(self.Exp_data["img"]):
            exp_index = index - 4*len(self.Exp_data["img"])

        # exp_data
        exp_name_list = self.Exp_imgs[exp_index]
        exp_img = Image.open(os.path.join(self.img_path,exp_name_list))
        if exp_img.getbands()[0] != 'R':
            exp_img = exp_img.convert('RGB')
        label_exp = int(self.labels_Exp[exp_index])
        exp_emb1,exp_emb2,exp_emb3,exp_emb4,exp_emb5 = self.Exp_emb_before2[exp_index],self.Exp_emb_before1[exp_index],self.Exp_emb_current[exp_index],self.Exp_emb_after1[exp_index],self.Exp_emb_after2[exp_index]
        exp_emb1, exp_emb2, exp_emb3, exp_emb4, exp_emb5 = self.process_emb(exp_emb1),self.process_emb(exp_emb2),self.process_emb(exp_emb3),self.process_emb(exp_emb4),self.process_emb(exp_emb5)
        exp_embs = np.concatenate((exp_emb1,exp_emb2,exp_emb3,exp_emb4,exp_emb5),axis=0)

        #AU_data
        au_name_list = self.AU_imgs[au_index]
        au_img = Image.open(os.path.join(self.img_path, au_name_list))
        if au_img.getbands()[0] != 'R':
            au_img = au_img.convert('RGB')
        label_au = self.labels_AU[au_index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]
        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0
        label_au_ = torch.tensor(label_au_)
        AU_emb1, AU_emb2, AU_emb3, AU_emb4, AU_emb5 = self.AU_emb_before2[au_index], self.AU_emb_before1[
            au_index], self.AU_emb_current[au_index], self.AU_emb_after1[au_index], self.AU_emb_after2[au_index]
        AU_emb1, AU_emb2, AU_emb3, AU_emb4, AU_emb5 = self.process_emb(AU_emb1), self.process_emb(
            AU_emb2), self.process_emb(AU_emb3), self.process_emb(AU_emb4), self.process_emb(AU_emb5)
        au_embs = np.concatenate((AU_emb1, AU_emb2, AU_emb3, AU_emb4, AU_emb5), axis=0)

        #VA_data
        va_name_list = self.VA_imgs[va_index]
        va_img = Image.open(os.path.join(self.img_path, va_name_list))
        if va_img.getbands()[0] != 'R':
            va_img = va_img.convert('RGB')
        label_V = float(self.labels_V[va_index])
        label_A = float(self.labels_A[va_index])
        VA_emb1, VA_emb2, VA_emb3, VA_emb4, VA_emb5 = self.VA_emb_before2[va_index], self.VA_emb_before1[
            va_index], self.VA_emb_current[va_index], self.VA_emb_after1[va_index], self.VA_emb_after2[va_index]
        VA_emb1, VA_emb2, VA_emb3, VA_emb4, VA_emb5 = self.process_emb(VA_emb1), self.process_emb(
            VA_emb2), self.process_emb(VA_emb3), self.process_emb(VA_emb4), self.process_emb(VA_emb5)
        va_embs = np.concatenate((VA_emb1, VA_emb2, VA_emb3, VA_emb4, VA_emb5), axis=0)
        if self.transform is not None:
            exp_img = self.transform(exp_img)
            va_img = self.transform(va_img)
            au_img = self.transform(au_img)

        return au_img,va_img,exp_img,label_au_,label_V,label_A,label_exp,au_embs,va_embs,exp_embs


class ABAW2_VA_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,has_emb=False,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_V = self.data['V']
        self.labels_A = self.data['A']
        self.img_path = img_path
        self.has_emb = has_emb
        if has_emb:
            self.emb_before2, self.emb_before1, self.emb_current, self.emb_after1, self.emb_after2 = \
                self.data['emb_before2'], \
                self.data['emb_before1'], self.data['emb_current'], self.data['emb_after1'], self.data[
                    'emb_after2']


    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path, anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        if self.has_emb:
            emb1, emb2, emb3, emb4, emb5 = self.emb_before2[index], self.emb_before1[
                index], self.emb_current[index], self.emb_after1[index], self.emb_after2[index]
            emb1, emb2, emb3, emb4, emb5 = self.process_emb(emb1), self.process_emb(
                emb2), self.process_emb(emb3), self.process_emb(emb4), self.process_emb(emb5)
            embs = np.concatenate((emb1, emb2, emb3, emb4, emb5), axis=0)
            return anc_img,label_V,label_A,embs,anc_list


        return anc_img,label_V,label_A,anc_list


class ABAW2_AU_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,has_emb = False,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_AU = self.data['AU']
        self.img_path = img_path
        self.has_emb = has_emb
        print(has_emb)
        if self.has_emb:
            self.emb_before2, self.emb_before1, self.emb_current, self.emb_after1, self.emb_after2 = \
                self.data['emb_before2'], self.data['emb_before1'], self.data['emb_current'], self.data['emb_after1'], self.data['emb_after2']
        #self.embs = self.data["emb"]

    def __len__(self):
        return len(self.data["img"])

    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path, anc_list))

        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        label_au = self.labels_AU[index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]

        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0

        label_au_ = torch.tensor(label_au_)
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        if self.has_emb:
            emb1, emb2, emb3, emb4, emb5 = self.emb_before2[index], self.emb_before1[
                index], self.emb_current[index], self.emb_after1[index], self.emb_after2[index]
            emb1, emb2, emb3, emb4, emb5 = self.process_emb(emb1), self.process_emb(
                emb2), self.process_emb(emb3), self.process_emb(emb4), self.process_emb(emb5)
            embs = np.concatenate((emb1, emb2, emb3, emb4, emb5), axis=0)
            return anc_img,label_au_,embs,anc_list


        return anc_img,label_au_,anc_list



class ABAW2_Exp_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,has_emb=False,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_Exp = self.data['Expression']
        self.img_path = img_path
        self.has_emb = has_emb
        if has_emb:
            self.emb_before2, self.emb_before1, self.emb_current, self.emb_after1, self.emb_after2 = self.data['emb_before2'], self.data['emb_before1'], self.data['emb_current'], self.data['emb_after1'], self.data['emb_after2']

    def __len__(self):
        return len(self.data["img"])
    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb
    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path,anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')
        label_exp = int(self.labels_Exp[index])
        if self.transform is not None:
            anc_img = self.transform(anc_img)
        if self.has_emb:
            emb1, emb2, emb3, emb4, emb5 = self.emb_before2[index], self.emb_before1[
                index], self.emb_current[index], self.emb_after1[index], self.emb_after2[index]
            emb1, emb2, emb3, emb4, emb5 = self.process_emb(emb1), self.process_emb(
                emb2), self.process_emb(emb3), self.process_emb(emb4), self.process_emb(emb5)
            embs = np.concatenate((emb1, emb2, emb3, emb4, emb5), axis=0)
            return anc_img,label_exp,embs,anc_list

        return anc_img,label_exp,anc_list


class ABAW2_Exp_3dcnn_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_Exp = self.data['Expression']
        self.img_path = img_path
        #self.embs = self.data["emb"]

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_data = np.load(os.path.join(self.img_path,anc_list))
        anc_data = np.transpose(anc_data,(3,0,1,2))
        anc_data = torch.tensor(anc_data/255,dtype=torch.float32)
        label_exp = int(self.labels_Exp[index])

        return anc_data,label_exp,anc_list


def compute_class_weight(csv_data,type="AU"):
    csv_data = pd.read_csv(csv_data)
    if type=="AU":
        labels = csv_data["AU"]
        c = [0 for i in range(12)]
        N = len(labels)
        for i in range(len(labels)):
            l = labels[i].split(" ")
            for j in range(len(l)):
                if int(l[j]) > 0:
                    c[j] += 1
        r = [N / c[i] for i in range(12)]
        s = sum(r)
        r = [r[i] / s for i in range(12)]
        return torch.as_tensor(r, dtype=torch.float)
    elif type=="Exp":
        if "EXP" in csv_data.columns:
            labels = csv_data["EXP"].to_list()
        else:
            labels = csv_data["Expression"].to_list()
        counts = []
        for i in range(7):
            counts.append(labels.count(i))
        N = len(labels)
        r = [N / counts[i] for i in range(7)]
        s = sum(r)
        r = [r[i] / s for i in range(7)]
        return torch.as_tensor(r, dtype=torch.float)


class ABAW2_test_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,has_emb=False,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.img_path = img_path
        self.has_emb = has_emb
        if has_emb:
            self.emb_before2, self.emb_before1, self.emb_current, self.emb_after1, self.emb_after2 = \
                self.data['emb_before2'], \
                self.data['emb_before1'], self.data['emb_current'], self.data['emb_after1'], self.data[
                    'emb_after2']


    def __len__(self):
        return len(self.data["img"])
    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb
    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path,anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')
        if self.transform is not None:
            anc_img = self.transform(anc_img)
        if self.has_emb:
            emb1, emb2, emb3, emb4, emb5 = self.emb_before2[index], self.emb_before1[
                index], self.emb_current[index], self.emb_after1[index], self.emb_after2[index]
            emb1, emb2, emb3, emb4, emb5 = self.process_emb(emb1), self.process_emb(
                emb2), self.process_emb(emb3), self.process_emb(emb4), self.process_emb(emb5)
            embs = np.concatenate((emb1, emb2, emb3, emb4, emb5), axis=0)
            return anc_img,embs,anc_list

        return anc_img,anc_list
