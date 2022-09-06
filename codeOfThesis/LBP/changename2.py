# coding=utf-8
import os
import shutil

# 将多个文件夹下的文件放入同一个文件中：2放入同一个文件夹下
#目标文件夹，此处为相对路径，也可以改为绝对路径
determination = './orl'
if not os.path.exists(determination):
    os.makedirs(determination)

#源文件夹路径
path = './image_c'
folders = os.listdir(path)
for folder in folders:
    dir = path + '/' + str(folder)
    files = os.listdir(dir)
    for file in files:
        source = dir + '/' + str(file)
        deter = determination + '/' + str(file)
        shutil.copyfile(source, deter)
