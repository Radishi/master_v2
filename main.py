import logging
import os

from action_classification.skeleton_pipeline import Action_Recognizer
import cv2
from action_classification.ADModel import ADModel
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from insightface.data import get_image as ins_get_image
import time

def action_match_face_and_head(faces,name_list,labels,head_poses):
    """
    动作识别结果与人脸识别结果绑定，通过判断鼻子关键点是否再人脸框内
    Args:
        faces: 人脸框
        name_list: 人脸识别的名字
        labels: 动作识别结果
        head_poses: 头部识别结果
    Returns: action_face_and_head_list [{"box":[],"action":"xxxx","name":"xxx","head_pose":[]}]
    """
    action_face_and_head_list = []
    for i,label in enumerate(labels):
        x,y = label["nose_xy"]
        for j,face in enumerate(faces):  #face顺序与head_poses一致
            x1,y1,x2,y2 = face.left(),face.top(),face.right(),face.bottom()
            if x1<x<x2 and y1<y<y2:  #动作与人脸匹配
                action_face_and_head_list.append(dict(box=label["box"],action=label["action_class"],scores=label["scores"],name=name_list[j],head_pose=head_poses[j]))
                break
            if j == len(faces)-1: #检测的到关键点，但是检测不到人脸框
                action_face_and_head_list.append(dict(box=label["box"],action=label["action_class"],scores=label["scores"],name="unknown",head_pose=None))
    return action_face_and_head_list


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter("output.mp4", fourcc, fps, size)
    app = FaceAnalysis(root="checkpoints/",
                       providers=['GPUExecutionProvider'])  # allowed_modules=['detection'],,'alignment'
    app.prepare(ctx_id=0, det_size=(1120, 1600))  # , det_size=(1120, 1600)
    action_recognizer = Action_Recognizer()
    num_frame = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        num_frame += 1
        faces = app.get(img)
        rimg = app.draw_on(img, faces)
        videoWriter.write(rimg)
        print("----------------process frame_num {}----------".format(num_frame))
    cap.release()
    videoWriter.release()


def process_img(img_path):
    #providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(root="checkpoints/",providers=['CPUExecutionProvider'])  # allowed_modules=['detection'],,'alignment'
    img = cv2.imread(img_path)
    app.prepare(ctx_id=0,det_size=(1120, 1600))#, det_size=(1120, 1600)

    faces = app.get(img)
    rimg = app.draw_on(img, faces,face_dis_thr=26)

    # recognizer = Action_Recognizer()
    #
    # ac_img, labels = recognizer.process(img,is_show_keypoints=False)  #labels中包含 人体bbox
    # cv2.imshow("123",rimg)
    # cv2.waitKey(0)
    output_path = "result_img/"+img_path.split(".")[0].split("/")[1]+"_out.jpg"
    cv2.imwrite(output_path,rimg)



if __name__ == '__main__':
    # img_list = os.listdir("test_img")
    # for img_name in img_list:
    #     if img_name.split('.')[-1] not in ['jpg','png']:
    #         continue
    #     img_path = "test_img"+"/"+img_name
    #     process_img(img_path)
    img_path = "test_img" + "/" + "test12.jpeg"
    process_img(img_path)

    # model = model_zoo.get_model("checkpoints/models/buffalo_l/det_10g.onnx",providers=['CPUExecutionProvider'])
    # print(model)


