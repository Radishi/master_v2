
# Copyright (c) OpenMMLab. All rights reserved.
import os
import abc
import warnings
import pickle
from mmpose.apis import (inference_top_down_pose_model,inference_pose_lifter_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)
from mmpose.datasets import DatasetInfo
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
import cv2
import os.path as osp
#from mmpose.apis.inference import init_pose_model

class Pose_3D_estimation():

    def __init__(self):

        self.pose_lifter_config = "action_classification/configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py"
        self.pose_lifter_checkpoint = "action_classification/checkpoints/simple3Dbaseline_h36m-f0ad73a4_20210419.pth"
        self.pose_config_2D = f'action_classification/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'  # noqa: E501
        # args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501
        self.pose_checkpoint_2D = 'action_classification/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501
        self.device = "cuda:0"
        self.det_config = f'action_classification/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
        # args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
        self.det_checkpoint = 'action_classification/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
        self.det_score_thr = 0.5
        self.det_cat_id = 1
        self.bbox_thr = 0.5
        self.kpt_thr = 0.5
        self.radius = 4
        self.thickness = 1
        self.out_img_root = None
        assert has_mmdet, 'Please install mmdet to run the demo.'


    def init_top_down_model(self):
        # 初始化模型
        assert self.det_config is not None
        assert self.det_checkpoint is not None
        #初始化人体检测模型
        self.det_model = init_detector(
            self.det_config, self.det_checkpoint, device=self.device.lower())
        # 初始化2d姿态估计模型
        self.pose_model_2D = init_pose_model(
            self.pose_config_2D, self.pose_checkpoint_2D, device=self.device.lower())
        self.dataset_2d = self.pose_model_2D.cfg.data['test']['type']
        self.dataset_info_2d = self.pose_model_2D.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info_2d is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info_2d = DatasetInfo(self.dataset_info_2d)

        #初始化3D姿态估计模型
        self.pose_lift_model = init_pose_model(
            self.pose_lifter_config,
            self.pose_lifter_checkpoint,
            device=self.device.lower())
        assert self.pose_lift_model.cfg.model.type == 'PoseLifter', 'Only' \
                                                               '"PoseLifter" model is supported for the 2nd stage ' \
                                                               '(2D-to-3D lifting)'
        self.dataset_3D = self.pose_lift_model.cfg.data['test']['type']
        self.dataset_info_3D = self.pose_lift_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info_3D is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info_3D = DatasetInfo(self.dataset_info_3D)



    def process_img(self,image, is_show_keypoints=False):
        """
        Args:
            image: cv2 img or path str
        Returns:
        """
        if isinstance(image,str):
            image_name = image.split("/")[-1].split(".")[0]
            image = cv2.imread(image)
            self.out_img_root = "result_img"

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, image)
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, self.det_cat_id)

        pose_det_results_list = []
        next_id = 0
        pose_det_results = []

        pose_det_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model_2D,
            image,
            person_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset_2d,
            dataset_info=self.dataset_info_2d,
            return_heatmap=False,
            outputs=None)

        pose_lift_results = inference_pose_lifter_model(
            self.pose_lift_model,
            pose_results_2d=[pose_det_results],
            dataset=self.dataset_3D,
            dataset_info=self.dataset_info_3D,
            with_track_id=False)

        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # rebase height (z-axis)
            # if args.rebase_keypoint_height:
            #     keypoints_3d[..., 2] -= np.min(
            #         keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # Add title
            res['title'] = f'Prediction ()'
            pose_lift_results_vis.append(res)


        # Visualization
        if self.out_img_root is None:
            out_file = None
        else:
            os.makedirs(self.out_img_root, exist_ok=True)
            out_file = osp.join(self.out_img_root, image_name+'_out.jpg')
        img = None

        if is_show_keypoints:
            img = vis_3d_pose_result(
                self.pose_lift_model,
                result=pose_lift_results,
                img=image,
                dataset_info=self.dataset_info_3D,
                out_file=out_file)
        # 增加返回人体检测框 mmdet_results 用来绑定其他信息
        return img, pose_lift_results, mmdet_results


#1.获取文件名列表
def get_file_names(folder_dir):
    file_names = os.listdir(folder_dir)
    file_names = [osp.join(folder_dir, vid_n) for vid_n in file_names]
    return file_names

def extract_pose_from_video(is_save_img=False):
    """
    从视频中提取骨骼数据，并将结果保存为图片
    Returns:
    """
    det_model,pose_model = init_top_down_model()
    file_names = get_file_names(r"/content/mmpose/data/train")
    train_data = []
    for file_name in file_names:
        label = int(file_name.split('/')[-1].split('A')[1][:3])
        cap = cv2.VideoCapture(file_name)
        total_frame = 0
        video_poses = []
        img_num = 0
        img_save_dir = file_name.split('.')[0]
        if not os.path.exists(img_save_dir) and is_save_img:
            os.mkdir(img_save_dir)
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            img, pose_results = top_down_img(frame,det_model,pose_model)
            total_frame += 1
            if len(pose_results) == 1 and total_frame%3 == 0:  #每3帧保存一次
                img_num += 1
                if is_save_img:
                    img_save_name = os.path.join(img_save_dir, str(img_num) + '.jpg')
                    cv2.imwrite(img_save_name, img)
                    print("save image to {}".format(img_save_name))
                video_poses.append(pose_results[0]["keypoints"])  # 0 每一帧只有一个人被识别
        train_data.append(dict(label=label,video_name=file_name,keypoints=video_poses))
        print("video:{}  process finished".format(file_name))
    with open("/content/mmpose/data/train/train.pkl", "wb") as fo:
        pickle.dump(train_data,fo)
    print("all video done!")

if __name__ == '__main__':
    file_name = "../123.jpg"
    estimationer = Pose_3D_estimation()
    estimationer.init_top_down_model()
    estimationer.process_img(file_name)


