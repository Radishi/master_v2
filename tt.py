#
# import cv2
#
# cap = cv2.VideoCapture(r'C:\Users\radishi\Downloads\1652706566713video.mp4')
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     cv2.imshow('image', frame)
#     k = cv2.waitKey(40)
#     # q键退出
#     if (k & 0xff == ord('q')):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# import os
# import shutil
# from PIL import Image
# import cv2
# root = r"C:\Users\radishi\Desktop\4班\4班"
# name_list = os.listdir(r"C:\Users\radishi\Desktop\4班\4班")
#
# for name in name_list:
#     folder_path = os.path.join(root,name)
#     image_name = name+".png"
#     image_path = os.path.join(folder_path,image_name)
#     image = Image.open(image_path)
#     save_image_name = name+"_flip.png"
#     save_path = os.path.join(folder_path,save_image_name)
#     image.transpose(Image.FLIP_LEFT_RIGHT).save(save_path)
from extract_face_features import read_pkl

path = "face_database/luo/img1.jpg"
import cv2
from PIL import Image
data = read_pkl("face_database.pkl")
print(data.keys())

