from PIL import Image             #改变图像名称
import os, glob


def batch_image(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)

    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    count = 0
    for files in glob.glob(in_dir + '/*'):
        filepath, filename = os.path.split(files)

        out_file =filename[0:9].replace('.pgm','') + '.jpg'

        # #print(filepath,',',filename, ',', out_file)
        im = Image.open(files)
        new_path = os.path.join(out_dir, out_file)
        print(count, ',', new_path)
        count = count + 1
        im.save(os.path.join(out_dir, out_file))


if __name__ == '__main__':
    num=0
    for i in range(1,41):
        txt_in="./image_c/s{}"
        txt_out="./orl_sqname"
        batch_image(txt_in.format(i),txt_out)
