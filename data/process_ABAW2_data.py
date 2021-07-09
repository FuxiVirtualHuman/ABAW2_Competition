import pandas as pd
import os
anno_path = r"E:\aff\aff_wild2\annotations"
img_path = r"D:\xiangmu\emotion\openface"
def produce_multi_task_videos():

    AU_train_videos = os.listdir(os.path.join(anno_path,"AU_Set","Train_Set"))
    AU_val_videos = os.listdir(os.path.join(anno_path,"AU_Set","Validation_Set"))

    VA_train_videos = os.listdir(os.path.join(anno_path, "VA_Set", "Train_Set"))
    VA_val_videos = os.listdir(os.path.join(anno_path, "VA_Set", "Validation_Set"))

    EXP_train_videos = os.listdir(os.path.join(anno_path, "EXPR_Set", "Train_Set"))
    EXP_val_videos = os.listdir(os.path.join(anno_path, "EXPR_Set", "Validation_Set"))


    train_videos = []
    train_videos.extend(AU_train_videos)
    train_videos.extend(VA_train_videos)
    train_videos.extend(EXP_train_videos)
    train_videos = list(set(train_videos))
    print(len(train_videos))

    return train_videos,AU_train_videos,VA_train_videos,EXP_train_videos

train_videos,AU_train_videos,VA_train_videos,EXP_train_videos = produce_multi_task_videos()



def get_names(id):
    name = ""
    if id>=0 and id<10:
        name = "0000" + str(id)
    elif id>=10 and id<100:
        name = "000" + str(id)
    elif id>=100 and id<1000:
        name = "00" + str(id)
    elif id>=1000 and id<10000:
        name = "0" + str(id)
    else:
        name = str(id)
    return name


def produce_multi_task_labels_for_one_video(video_name):
    label_dict = {}
    VA_flag,AU_flag,EXP_flag = True,True,True
    if video_name+".txt" not in VA_train_videos:
        VA_flag = False
    if video_name+".txt" not in AU_train_videos:
        AU_flag = False
    if video_name+".txt" not in EXP_train_videos:
        EXP_flag = False
    if not VA_flag and not AU_flag and not EXP_flag:
        return

    if VA_flag:
        f = open(os.path.join(anno_path,"VA_Set","Train_Set",video_name+".txt"))
        lines = f.readlines()[1:]
        for i in range(len(lines)):
            l = lines[i].strip().split(",")
            if l[0]=="-1" or l[1] == "-1":
                continue
            frame = get_names(i)
            if os.path.exists(os.path.join(img_path,video_name,video_name+"_aligned",frame+".jpg")):
                n = video_name.split(".")[0]+"/"+video_name+"_aligned"+"/"+frame+".jpg"
                if n not in label_dict.keys():
                    label_dict[n] = [float(l[0]),float(l[1]),-1,-1]
                else:
                    label_dict[n][0] = float(l[0])
                    label_dict[n][1] = float(l[1])

    if AU_flag:
        f = open(os.path.join(anno_path, "AU_Set", "Train_Set", video_name + ".txt"))
        lines = f.readlines()[1:]
        for i in range(len(lines)):
            l = lines[i].strip().split(",")
            if "-1" in l:
                continue
            frame = get_names(i)
            if os.path.exists(os.path.join(img_path, video_name, video_name + "_aligned",
                                           frame + ".jpg")):
                n = video_name.split(".")[0] + "/" + video_name + "_aligned" + "/" + frame + ".jpg"
                if n not in label_dict.keys():
                    label_dict[n] = [-1,-1," ".join(l),-1]
                else:
                    label_dict[n][2] = " ".join(l)
    if EXP_flag:
        f = open(os.path.join(anno_path, "EXPR_Set", "Train_Set", video_name + ".txt"))
        lines = f.readlines()[1:]
        for i in range(len(lines)):
            l = lines[i].strip()
            if "-1" in l:
                continue
            frame = get_names(i)
            if os.path.exists(os.path.join(img_path, video_name, video_name + "_aligned",
                                           frame + ".jpg")):
                n = video_name.split(".")[0] + "/" + video_name + "_aligned" + "/" + frame + ".jpg"
                if n not in label_dict.keys():
                    label_dict[n] = [-1,-1,-1,int(l)]
                else:
                    label_dict[n][-1] = int(l)
    return label_dict


def produce_anno_csvs():
    save_path = r"E:\aff\aff_wild2\annotations\multi_task\Train_Set"
    for video in train_videos:
        print(video)
        label_dict = produce_multi_task_labels_for_one_video(video.split(".")[0])
        data = pd.DataFrame()
        imgs,V,A,AU,EXP = [],[],[],[],[]
        for k,v in label_dict.items():
            imgs.append(k)
            V.append(v[0])
            A.append(v[1])
            AU.append(v[2])
            EXP.append(v[3])
        data["img"],data["AU"],data["EXP"],data["V"],data["A"] = imgs,AU,EXP,V,A
        data.to_csv(os.path.join(save_path,video.split(".")[0]+".csv"))

def produce_total_csvs():
    path = r"E:\aff\aff_wild2\annotations\multi_task\Train_Set"
    csv_data_list = os.listdir(path)
    total_data = pd.DataFrame()
    imgs,V,A,AU,EXP = [],[],[],[],[]
    for csv in csv_data_list:
        print(csv)
        data = pd.read_csv(os.path.join(path,csv))
        imgs.extend(data["img"].to_list())
        V.extend(data["V"].to_list())
        A.extend(data["A"].to_list())
        AU.extend(data["AU"].to_list())
        EXP.extend(data["EXP"].to_list())
    print(len(imgs),len(A),len(V),len(AU),len(EXP))
    total_data["img"],total_data["AU"],total_data["EXP"],total_data["V"],total_data["A"] = imgs,AU,EXP,V,A
    total_data.to_csv("ABAW2_multi_task_training.csv")

def produce_category_csvs():
    path = r"E:\aff\aff_wild2\annotations\multi_task\Train_Set"
    csv_data_list = os.listdir(path)
    AU_spec_data, VA_spec_data, Exp_spec_data, AU_VA_spec_data, AU_exp_spec_data, VA_exp_spec_data = [],[],[],[],[],[]
    multi_data = []
    for csv in csv_data_list:
        print(csv)
        total_data = pd.read_csv(os.path.join(path,csv))
        imgs, AU, EXP, V, A = total_data["img"],total_data["AU"],\
                          total_data["EXP"],total_data["V"],total_data["A"]
        for i in range(len(imgs)):
            if AU[i] != -1 and V[i] != -1 and A[i] != -1 and EXP[i] != -1:
                 multi_data.append(total_data.iloc[i, :])
            elif AU[i] != -1 and V[i] != -1 and A[i] != -1 and EXP[i] == -1:
                AU_VA_spec_data.append(total_data.iloc[i, :])
            elif AU[i] != -1 and V[i] == -1 and A[i] == -1 and EXP[i] != -1:
                AU_exp_spec_data.append(total_data.iloc[i, :])
            elif AU[i] == -1 and V[i] != -1 and A[i] != -1 and EXP[i] != -1:
                VA_exp_spec_data.append(total_data.iloc[i, :])
            elif AU[i] != -1 and V[i] == -1 and A[i] == -1 and EXP[i] == -1:
                AU_spec_data.append(total_data.iloc[i, :])
            elif AU[i] == -1 and V[i] != -1 and A[i] != -1 and EXP[i] == -1:
                VA_spec_data.append(total_data.iloc[i, :])
            elif AU[i] == -1 and V[i] == -1 and A[i] == -1 and EXP[i] != -1:
                Exp_spec_data.append(total_data.iloc[i, :])
        print(len(multi_data))
        print(len(AU_VA_spec_data))
        print(len(AU_exp_spec_data))
        print(len(VA_exp_spec_data))
        print(len(AU_spec_data))
        print(len(VA_spec_data))
        print(len(Exp_spec_data))
    multi_data = pd.DataFrame(multi_data)
    AU_spec_data = pd.DataFrame(AU_spec_data)
    VA_spec_data = pd.DataFrame(VA_spec_data)
    Exp_spec_data = pd.DataFrame(Exp_spec_data)
    AU_exp_spec_data = pd.DataFrame(AU_exp_spec_data)
    VA_exp_spec_data = pd.DataFrame(VA_exp_spec_data)
    AU_VA_spec_data = pd.DataFrame(AU_VA_spec_data)


    multi_data.to_csv("multi_data.csv")
    AU_spec_data.to_csv("AU_spec_data.csv")
    VA_spec_data.to_csv("VA_spec_data.csv")
    Exp_spec_data.to_csv("Exp_spec_data.csv")
    AU_VA_spec_data.to_csv("AU_VA_spec_data.csv")
    AU_exp_spec_data.to_csv("AU_exp_spec_data.csv")
    VA_exp_spec_data.to_csv("VA_exp_spec_data.csv")


def produce_training_data():
    total_data = pd.read_csv("ABAW2_multi_task_training.csv")
    imgs, AU, EXP, V, A =  total_data["img"], total_data["AU"], total_data["EXP"], total_data["V"], total_data["A"]

    for i in range(len(imgs)):
        if i%1000==0:
            print(i)
        if AU[i]==-1 or V[i]==-1 or A[i]==-1 or EXP[i]==-1:
            total_data.drop([i])

    print(len(total_data))


produce_training_data()

#produce_category_csvs()
#produce_total_csvs()
#produce_anno_csvs()








