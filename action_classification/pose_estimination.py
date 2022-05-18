
# Copyright (c) OpenMMLab. All rights reserved.
import os
import abc
import warnings
import pickle
from mmpose.apis import (inference_top_down_pose_model,inference_bottom_up_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
import cv2
import os.path as osp

args = abc.abstractproperty()
def parser_top_down_args():
    # top dowm config v1
    # args.pose_config = f'action_classification/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
    # args.pose_checkpoint = 'action_classification/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    # top dowm config v2
    args.pose_config = f'action_classification/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py'
    args.pose_checkpoint = 'action_classification/checkpoints/hrnet_w48_coco_384x288-314c8528_20200708.pth'

    # bottom up config v1
    # args.pose_config = f'action_classification/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_640x640.py'
    # args.pose_checkpoint = 'action_classification/checkpoints/higher_hrnet32_coco_640x640-a22fe938_20200712.pth'
    # bottom up config v2
    # args.pose_config = f'action_classification/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py'
    # args.pose_checkpoint = 'action_classification/checkpoints/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth'

    # detect v1
    args.det_config = f'action_classification/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
    args.det_checkpoint = 'action_classification/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
    # detect v2
    # args.det_config = f'action_classification/configs/faster_rcnn/faster_rcnn_r50_fpn_coco.py'  # noqa: E501
    # args.det_checkpoint = 'action_classification/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa: E501

    args.device = "cuda:0"
    # args.det_score_thr = 0.5
    args.det_cat_id = 1
    args.bbox_thr = 0.5
    args.kpt_thr = 0.5
    args.radius = 4
    args.thickness = 1
    assert has_mmdet, 'Please install mmdet to run the demo.'
    return args

def parser_bottom_up_args():

    args.pose_config = f"configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py"
    args.pose_checkpoint = "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth"
    args.device = "cuda:0"
    args.kpt_thr = 0.3
    args.pose_nms_thr =0.9
    args.radius = 4
    args.thickness = 1
    return args

def init_top_down_model():
    #初始化模型 ，返回人体检测和姿态估计模型
    args = parser_top_down_args()
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
    args.pose_config, args.pose_checkpoint, device=args.device.lower())
    return det_model,pose_model

def top_down_img(image,det_model,pose_model,is_show_keypoints=False):
    """
    Args:
        image: cv2 img
        det_model:
        pose_model:
    Returns:

    """
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, image)
    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
    # optional
    return_heatmap = False
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        image,
        person_results,
        bbox_thr=args.bbox_thr,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)


    if is_show_keypoints:
        # show the results
        image = vis_pose_result(
            pose_model,
            image,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness)
    # 增加返回人体检测框 mmdet_results 用来绑定其他信息
    return image,pose_results,mmdet_results


def bottom_up_img(image):
    """
    Args:
        img: ndarrays

    Returns: img_with_skeleton
    """
    args = parser_bottom_up_args()

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = False
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # test a single image, with a list of bboxes.
    pose_results, returned_outputs = inference_bottom_up_pose_model(
        pose_model,
        image,
        dataset=dataset,
        dataset_info=dataset_info,
        pose_nms_thr=args.pose_nms_thr,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    # show the results
    img_with_skeleton = vis_pose_result(
        pose_model,
        image,
        pose_results,
        radius=args.radius,
        thickness=args.thickness,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr)
    return img_with_skeleton

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
    extract_pose_from_video()
