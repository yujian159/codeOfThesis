#
# # 引入opencv模块
# import cv2 as cv
# # 引入numpy模块
# import numpy as np
# # 引入sys模块
# import sys
#
# # 对比度范围：0 ~ 0.3
# alpha = 0.3
# # 亮度范围0 ~ 100
# beta = 100
# img = cv.imread('.\orl\s1_0.jpg')
# img2 = cv.imread('.\orl\s1_1.jpg')
#
#
# def updateAlpha(x):
#     global alpha, img, img2
#     alpha = cv.getTrackbarPos('Alpha', 'image')
#     alpha = alpha * 0.01
#     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
#
#
# def updateBeta(x):
#     global beta, img, img2
#     beta = cv.getTrackbarPos('Beta', 'image')
#     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
#
#
# def img_test():
#     global beta, img, img2
#     # 判断是否读取成功
#     if img is None:
#         print("Could not read the image,may be path error")
#         return
#
#     # 创建窗口
#     cv.namedWindow('image', cv.WINDOW_NORMAL)
#     cv.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
#     cv.createTrackbar('Beta', 'image', 0, 255, updateBeta)
#     cv.setTrackbarPos('Alpha', 'image', 100)
#     cv.setTrackbarPos('Beta', 'image', 10)
#     while (True):
#         cv.imshow('image', img)
#         if cv.waitKey(1) == ord('q'):
#             break
#     cv.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     sys.exit(img_test() or 0)



import os
import random
import cv2 as cv
import numpy as np
import sys

# 对比度亮度调整
def img_contrast_bright(img, a, b, g):
    h, w, c = img.shape
    blank = np.zeros([h, w, c], img.dtype)
    dst = cv.addWeighted(img, a, blank, b, g)
    return dst

#调节程序
def img_ch():
    for filename in os.listdir(r'./orl'):
        print(filename)
        img = cv.imread('./orl/'+filename)
        if img is None:
            print("Could not read the image,may be path error")
            return

        a = 1+random.uniform(-0.4, 0.4)
        b = 1 - a
        g = 20 * random.uniform(-1.0, 1.0)
        img2 = img_contrast_bright(img, a, b, g)
        url='./orl_chlight/'+filename
        cv.imwrite(url,img2)

#主函数
if __name__ == '__main__':
    sys.exit(img_ch() or 0)