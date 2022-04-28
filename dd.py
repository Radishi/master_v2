#cuda version

import torch
#dlib state
import dlib

# print(torch.__version__)
#
# print(dlib.__version__)
#
# # 查看是否打开了CUDA加速
# print(dlib.DLIB_USE_CUDA)
#
# # 获取设备个数
# print(dlib.cuda.get_num_devices())
import cv2
img = cv2.imread("meeting.jpg")
img_rd = cv2.putText(img, "Radish",(200,200),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
cv2.imshow("123",img_rd)
cv2.waitKey(0)


# # Compute the 128D vector that describes the face in img identified by
# # shape.  In general, if two face descriptor vectors have a Euclidean
# # distance between them less than 0.6 then they are from the same
# # person, otherwise they are from different people. Here we just print
# # the vector to the screen.
# face_descriptor = facerec.compute_face_descriptor(img, shape)
# print(face_descriptor)
# # It should also be noted that you can also call this function like this:
# #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100, 0.25)
# # The version of the call without the 100 gets 99.13% accuracy on LFW
# # while the version with 100 gets 99.38%.  However, the 100 makes the
# # call 100x slower to execute, so choose whatever version you like.  To
# # explain a little, the 3rd argument tells the code how many times to
# # jitter/resample the image.  When you set it to 100 it executes the
# # face descriptor extraction 100 times on slightly modified versions of
# # the face and returns the average result.  You could also pick a more
# # middle value, such as 10, which is only 10x slower but still gets an
# # LFW accuracy of 99.3%.
# # 4th value (0.25) is padding around the face. If padding == 0 then the chip will
# # be closely cropped around the face. Setting larger padding values will result a looser cropping.
# # In particular, a padding of 0.5 would double the width of the cropped area, a value of 1.
# # would triple it, and so forth.
#
# # There is another overload of compute_face_descriptor that can take
# # as an input an aligned image.
# #
# # Note that it is important to generate the aligned image as
# # dlib.get_face_chip would do it i.e. the size must be 150x150,
# # centered and scaled.
# #
# # Here is a sample usage of that
#
# print("Computing descriptor on aligned image ..")
#
# # Let's generate the aligned image using get_face_chip
# face_chip = dlib.get_face_chip(img, shape)
#
# # Now we simply pass this chip (aligned image) to the api
# face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
# print(face_descriptor_from_prealigned_image)
#
# dlib.hit_enter_to_continue()