from .pose_estimination import parser_top_down_args,init_top_down_model,top_down_img
import cv2
from .tools import process_img_skeleton_data, get_action_class, vis_action_label, model_process
import torch
from .ADModel import ADModel

class Action_Recognizer():
    def __init__(self):
        self.det_model, self.pose_model = init_top_down_model()
        self.model = torch.load("action_classification/model_latest_test_acc_1.0.pth")


    def process(self,frame,is_show_keypoints=False):
        img, pose_results,mmdet_results = top_down_img(frame, self.det_model, self.pose_model,is_show_keypoints=is_show_keypoints)
        labels = model_process(pose_results, self.model)
        # visualize skeleton and bbox
        if is_show_keypoints:
            vis_action_label(img, labels)
        return img,labels


def video_action_recognition(video_dir,is_show=False):
    det_model, pose_model = init_top_down_model()
    model = torch.load("action_classification/model_latest_test_acc_1.0.pth")
    cap = cv2.VideoCapture(video_dir)
    window_name = "action recognition"

    if not is_show:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter("output.mp4", fourcc, fps, size)
    num_frame = 0
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        img, pose_results = top_down_img(frame, det_model, pose_model)
        labels = model_process(pose_results, model)
        vis_action_label(img, labels)
        num_frame += 1
        if is_show:
            cv2.imshow(window_name, img)
        else:
            videoWriter.write(img)
        print("----------------process frame_num {}----------".format(num_frame))
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    if is_show:
        cv2.destroyAllWindows()
    else:
        videoWriter.release()


if __name__=="__main__":
    #get skeleton data
    img = cv2.imread("123.jpg")
    det_model, pose_model = init_top_down_model()
    img,pose_results = top_down_img(img,det_model,pose_model)
    model = torch.load("model_latest_test_acc_0.9888682745825603.pth")
    #inpuut model
    labels = model_process(pose_results,model)
    # visualize skeleton and bbox
    vis_action_label(img,labels)
    cv2.imshow(img)