from __future__ import division
import glob
import pickle
import os.path as osp
import os
import cv2
import onnxruntime
from insightface.model_zoo import model_zoo
from insightface.utils import ensure_available
from insightface.app.common import Face
class extractor:
    def __init__(self, name="buffalo_l", root='checkpoints/', allowed_modules=['detection','recognition'], **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available('models', name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']
        self.prepare(0)


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640,640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def features_extraction_to_csv(self, img_root,max_num=1):
        """
        img_root: 图片保存根目录
            img_root
                person_name
                    img1.jpg
                    img2.jpg
        """
        features_list = {}
        person_list = os.listdir(img_root) #名字列表
        for person in person_list:
            person_img_path = os.path.join(img_root,person)
            person_img_list = os.listdir(person_img_path)  #图片列表
            person_features = []
            for person_img in person_img_list:
                img_path = os.path.join(person_img_path,person_img)
                img = cv2.imread(img_path)
                bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
                if bboxes.shape[0] == 0:
                    print(person,person_img)
                    continue
                bbox = bboxes[0, 0:4]
                det_score = bboxes[0, 4]
                kps = None
                if kpss is not None:
                    kps = kpss[0]
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                for taskname, model in self.models.items():
                    if taskname=='detection':
                        continue
                    model.get(img, face)
                person_features.append(face["embedding"])
            features_list[person] = person_features
        return features_list


def read_pkl(file_dir):
    with open(file_dir,"rb") as fo:
        data = pickle.load(fo,encoding="bytes")
    return data

def save_pkl(file_name,data):
    with open(file_name,"wb") as fo:
        pickle.dump(data,fo)

if __name__=="__main__":
    et = extractor(providers=['CPUExecutionProvider'])  # providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    features_list = et.features_extraction_to_csv("face_database")
    file_name = "face_database.pkl"
    save_pkl(file_name,features_list)
    face_database = read_pkl(file_name)


