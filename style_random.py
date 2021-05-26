

import numpy as np
from glob import glob
import cv2
import os
from PIL import Image


# 读取一张mask图片，并找出里面包含的所有语义标签
def get_labels(im):
    # im: rgb
    label_list = []
    for i in range(0, 19):
        index = np.where(im == i)
        if index and len(index[0]) > 20:
            label_list.append(i)
    return label_list


def main():
    label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/celeba_styles/test_label/"
    imgname_list = sorted(glob(label_dir + "*.png"))
    save_nums = [0 for m in range(0, 19)]
    # labels_tobechange = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    labels_tobechange = [0, 8, 9, 10, 13, 18]
    for imgname in imgname_list:
        im = cv2.imread(imgname, 0)   # read gray image
        name = imgname.split("/")[-1]
        name = name.replace("png", "jpg")
        exist_labels = get_labels(im)
        for label in labels_tobechange:
            if label in exist_labels:
                # 所包含的语义标签都分别读取对应的ACE文件
                ACE_filename = "styles_test/style_codes/" + name + "/" + str(label) + "/ACE.npy"
                code = np.load(ACE_filename)
                save_path = "styles_random3/" + str(label) + "/" + str(save_nums[label]) + "/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # 存储进入对应语义的文件夹，并编号
                np.save(save_path + "ACE.npy", code)
                np.savetxt(save_path + name.replace("jpg", "txt"), np.array([1]))
                save_nums[label] += 1
        print(name, save_nums)


def get_skinstyle_from_input():
    root = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front3/"
    lis = sorted(glob(root + "test_img_ori/*.png"))
    for i in range(0, len(lis)):
        raw_img = np.array(Image.open(lis[i]))
        label_full = np.array(Image.open(lis[i].replace("test_img_ori", "test_label_full")))

        index = np.where(np.all(raw_img == 255, axis=-1))
        label_full[index[0], index[1]] = 0
        Image.fromarray(label_full).save(lis[i].replace("test_img_ori", "test_label"))
        print(lis[i])

        # img = cv2.resize(raw_img, (0, 0), fx=2, fy=2)
        # name = lis[i].split("/")[-1].replace("png", "jpg")
        # Image.fromarray(img).save(root + "test_img2/" + name)



if __name__ == "__main__":
    main()
    # get_skinstyle_from_input()






