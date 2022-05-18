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

import cv2
img_path = "face_database" + "/" + "luo/"+"img1.jpg"
img = cv2.imread(img_path)
new_shape = (50,60)
img = cv2.resize(img,new_shape)
cv2.imwrite("face_database/luo/low_resolution1.jpg",img)



