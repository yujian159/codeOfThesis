import os
import cv2

for filename in os.listdir(r'./orl'):
    print(filename)
    txt_out = './orl_cut/'+filename
    img = cv2.imread('./orl/'+filename)  # img_path为图片所在路径
    crop_img = img[30:90, 16:76]  # img[y0:y1,x0:x1],x0,y0为裁剪区域左上坐标；x1,y1为裁剪区域右下坐标
    cv2.imwrite(txt_out, crop_img)  # save_path为保存路径


