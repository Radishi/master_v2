img_path = r'F:\development\Project\deep_learning_demo_object\cocodataset\val2017\000000061471.jpg'
import cv2
import os
img = cv2.imread(img_path)
# [272.1, 200.23, 151.97, 279.77]
cv2.rectangle(img,(272,200),(272+152,200+279),(233,241,251),1)

cv2.imshow("123",img)
cv2.waitKey(0)


