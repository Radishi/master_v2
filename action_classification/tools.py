import math
import os.path as osp
import os
import cv2
import pandas as pd
import pickle
import numpy as np
import copy
#1.获取文件名列表
import torch


def get_video_name(folder_dir):
    """
    返回folder_dir路径下的 所有文件名(绝对路径)
    Args:
        folder_dir: 文件夹路径
    Returns: [file_dir1,file_dir1,...]  str
    """
    video_names = os.listdir(folder_dir)
    video_names = [osp.join(folder_dir, vid_n) for vid_n in video_names]
    return video_names

def process_img_skeleton_data(img_data):  #return everyone skeleton distance on image
    """
    计算关键点距离矩阵
    Args:
        img_data: top_down_img(img,det_model,pose_model) 的返回值，[{"bbox":[x1,x2,y1,y2,thr],"keypoints":[[k1x,k1y,thr],...]},.....]
    Returns: dis_data (people_num,1,11,11)
    """
    peoples_dis_matri = []
    for peopel_data in img_data:
        kp = peopel_data["keypoints"][:11] #只取上半身关键点
        dis_matri = []
        for kp1 in kp:
            dis = []
            for kp2 in kp:
                x1, y1, x2, y2 = kp1[0], kp1[1], kp2[0], kp2[1]
                d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                dis.append(d)
            dis_matri.append(dis)
        peoples_dis_matri.append(dis_matri)
    dis_data = torch.tensor(peoples_dis_matri).reshape((-1, 1, 11, 11))
    return dis_data

def process_raw_train_skeleton():
    with open(r"C:\Users\radishi\Desktop\data\process_data\train.pkl","rb") as fo:
        skeleton_data = pickle.load(fo,encoding="bytes")
    for i,video_data in enumerate(skeleton_data):
        video_data = [ frame_data[:11 ] for frame_data in video_data["keypoints"]]
        #calculate skeleton distance
        video_dis_matri = []
        for frame_data in video_data:
            dis_matri = []
            for kp1 in frame_data:
                dis = []
                for kp2 in frame_data:
                    x1,y1,x2,y2 = kp1[0],kp1[1],kp2[0],kp2[1]
                    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    dis.append(d)
                dis_matri.append(dis)
            video_dis_matri.append(dis_matri)
        skeleton_data[i]["keypoints"] = video_dis_matri
    with open(r"train_v3.pkl","wb") as fo:
        pickle.dump(skeleton_data,fo)

def data_match_lable():
    dis_data = read_pkl("train_v3.pkl")
    keypoints = []
    labels = []
    for video_data in dis_data:
        vid_label = int(video_data["label"])
        for kp in video_data["keypoints"]:
            labels.append(vid_label-1)
            keypoints.append([kp])
    #save keypoints and labels
    save_pkl("train_v4.pkl",dict(labels=labels,keypoints=keypoints))
    return keypoints,labels

def get_action_class(class_dir="action_classification/label_list.txt"):
    """ 返回动作类别
    Args:
        class_dir: 文件路径 str
    Returns:example : {"0":"sleep",....}
    """
    label_name_list = {}
    df = pd.read_csv(class_dir, sep=" ")
    for i in range(len(df)):
        label_name_list.update({df.loc[i]["num"]: df.loc[i]["class"]})
    return label_name_list

def model_process(pose_results,model):
    """
    将距离矩阵数据模型做识别，并将识别结果  action_class(动作类别) scores(置信度) box(人体检测框)
    Args:
        dis_data: 距离矩阵
        bbox:  检测框
        model_dir: 模型路径
    Returns: example: [{"action_class":"sleep","scores":0.892,"box":[x1,y1,x2,y2,sor]},...]
    """
    # caculate distance
    dis_data = process_img_skeleton_data(pose_results)
    if torch.cuda.is_available():
        dis_data = dis_data.cuda()
    bbox = [data["bbox"] for data in pose_results]
    nose_xys = [data["keypoints"][0][:2] for data in pose_results]
    out = model(dis_data)
    out = out.cpu().detach().numpy()
    res_index = out.argmax(1)
    action_class = get_action_class()
    labels = [dict(action_class=action_class[i],scores=res[i],box=box,nose_xy=nose_xy) for i,res,box,nose_xy in zip(res_index,out,bbox,nose_xys)]
    return labels

def vis_action_label(img,labels,thr=0.9):
    """
    Args:
        img: cv2 image
        labels: [{"action_class":"sleep","scores":0.892,"box":[x1,y1,x2,y2,sor]},...]
        thr: except object thr < 0.7
    Returns: img
    """
    for label in labels:
        action_class,scores,box = label["action_class"],label["scores"],label["box"]
        if scores > thr:
            x1, y1 = int(box[0]), int(box[1])
            cv2.putText(img, "action: "+action_class, (x1 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def read_pkl(file_dir):
    with open(file_dir,"rb") as fo:
        data = pickle.load(fo,encoding="bytes")
    return data

def save_pkl(file_name,data):
    with open(file_name,"wb") as fo:
        pickle.dump(data,fo)

if __name__=="__main__":

    data = read_pkl("../train_v4.pkl")
    kps = copy.deepcopy(data["keypoints"][-50:-1])
    labels = copy.deepcopy(data["labels"][-50:-1])
    l = np.append(data["labels"],labels)
    k = np.append(data["keypoints"],kps,axis=0)
    save_pkl("train_v5.pkl", dict(labels=l, keypoints=k))







