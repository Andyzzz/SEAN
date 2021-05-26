
import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np
from glob import glob
import copy
from data.base_dataset import get_params, get_transform
from PIL import Image
import random

opt = TestOptions().parse()
# opt.status = 'test'
opt.status = 'UI_mode'

dataloader = data.create_dataloader(opt)

# init deep model
model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))


def load_input_feature(GT_img_path):
    # GT_img_path就是真值图片的完整路径，最终只取图片名字，用于寻找style_codes
    # GT_img_path = os.path.join(opt.image_dir, os.path.basename(fileName)[:-4] + '.jpg')
    ############### load average features

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(GT_img_path)
    input_style_dic = {}
    label_count = []

    style_img_mask_dic = {}

    for i in range(19):
        input_style_dic[str(i)] = {}

        input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
        input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in average_category_folder_list]

        for style_code_path in average_category_list:
            if style_code_path in input_category_list:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy'))).cuda()

                if style_code_path == 'ACE':
                    style_img_mask_dic[str(i)] = GT_img_path
                    label_count.append(i)

            else:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()

    obj_dic = input_style_dic
    # self.obj_dic_back = copy.deepcopy(self.obj_dic)
    obj_dic_GT = copy.deepcopy(obj_dic)

    return obj_dic_GT

def load_average_feature():

    ############### load average features

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    input_style_dic = {}

    ############### hard coding for categories

    for i in range(19):
        input_style_dic[str(i)] = {}

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
                                 average_category_folder_list]

        for style_code_path in average_category_list:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()

    obj_dic = input_style_dic
    return obj_dic


def load_partial_feature(GT_img_path, style_list, input_style_dic):
    # GT_img_path就是真值图片的完整路径，最终只取图片名字，用于寻找style_codes
    # GT_img_path = os.path.join(opt.image_dir, os.path.basename(fileName)[:-4] + '.jpg')
    ############### load average features

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(GT_img_path)
    # input_style_dic = {}
    label_count = []

    style_img_mask_dic = {}
    # style_list = [0, 1, 11, 12, 13]
    # style_list = [i for i in range(1, 19)]

    # input_style_dic = load_average_feature()

    # for i in range(19):
    for i in style_list:
    #     input_style_dic[str(i)] = {}
        if str(i) not in input_style_dic:
            input_style_dic[str(i)] = {}

        input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
        input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in average_category_folder_list]

        for style_code_path in average_category_list:
            if style_code_path in input_category_list:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy'))).cuda()

                if style_code_path == 'ACE':
                    style_img_mask_dic[str(i)] = GT_img_path
                    label_count.append(i)

            else:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()

    obj_dic = input_style_dic
    # self.obj_dic_back = copy.deepcopy(self.obj_dic)
    obj_dic_GT = copy.deepcopy(obj_dic)

    return obj_dic_GT



def load_random_feature(GT_img_path):
    obj_dic0 = load_input_feature(GT_img_path)
    labels_tobechange = [0, 10, 13, 18]
    hair = 0
    for i in labels_tobechange:
        style_code_dir = "styles_random3/" + str(i) + "/"
        lis = sorted(glob(style_code_dir + "*/*.npy"))
        random_ind = random.randint(0, len(lis)-1)
        if i==18:
            # print(random_ind)
            random_ind = 7
        if i==13:
            hair = random_ind
        obj_dic0[str(i)]['ACE'] = torch.from_numpy(np.load(lis[random_ind])).cuda()
    return obj_dic0, hair



def load_skin_feature(GT_img_path, skin_img_path):
    obj_dic0 = load_input_feature(GT_img_path)
    obj_dic = load_partial_feature(skin_img_path, style_list=[1, 2, 4, 5, 6, 7, 10, 11, 12], input_style_dic=obj_dic0)

    # labels_tobechange = [0, 10, 13, 18]
    # for i in labels_tobechange:
    #     style_code_dir = "styles_random2/" + str(i) + "/"
    #     lis = sorted(glob(style_code_dir + "*/*.npy"))
    #     random_ind = random.randint(0, len(lis) - 1)
    #     if i == 18:
    #         # print(random_ind)
    #         random_ind = 7
    #     obj_dic[str(i)]['ACE'] = torch.from_numpy(np.load(lis[random_ind])).cuda()
    return obj_dic


count = 0
# f = open("results_front5/record.txt", mode='w')
# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    # data_i['obj_dic'] = load_input_feature("./datasets/celeba/test_img/28182.jpg")
    # data_i['obj_dic'] = load_partial_feature("./datasets/celeba/test_img/28898.jpg")

    # # 之前的测试代码
    img_path = data_i['path']
    data_i['obj_dic'], hair = load_random_feature("./datasets/celeba/test_img/29265.jpg")  # 29265  # 28182 man pose
    # data_i['obj_dic'] = load_input_feature("./datasets/celeba_green/test_img/656.jpg")  # 29265  # 28182 man pose

    #=================================================================================
    # lis = glob("/home/zhang/PycharmProjects/SEAN/datasets/celeba2/test_img/*.jpg")
    # random_ind = random.randint(0, len(lis) - 1)
    # data_i['obj_dic'] = load_input_feature(lis[random_ind])
    #=================================================================================

    # data_i['obj_dic'], hair = load_random_feature("./datasets/celeba2/test_img/66.jpg")  # 29265  # 28182 man pose
    # data_i['obj_dic'] = load_skin_feature("./datasets/celeba/test_img/29265.jpg", img_path[0].replace("png", "jpg"))

    # =================================================================================
    # # 0324
    # img_path = data_i['path']
    # # 先加载随机的头发、衣服风格编码
    # obj_dic0, hair = load_random_feature("./datasets/celeba/test_img/29265.jpg")
    # # 再加载皮肤区域风格编码
    # data_i['obj_dic'] = load_partial_feature(img_path[0].replace("png", "jpg"), style_list=[1, 2, 4, 5, 6, 7, 10, 11, 12, 17], input_style_dic=obj_dic0)
    # =================================================================================

    generated = model(data_i, mode='UI_mode')

    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        name = img_path[b].split("/")[-1]
        # f.writelines([name, " ", str(hair), "\n"])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        count += 1
        print(count)
    #     if count > 100:
    #         break
    # if count > 100:
    #     break

webpage.save()
# f.close()




#
# class BatchTest():
#     def __init__(self, opt):
#         super().__init__()
#         self.init_deep_model(opt)
#
#     def init_deep_model(self, opt):
#         self.opt = opt
#         self.model = Pix2PixModel(self.opt)
#         self.model.eval()
#
#     def run_deep_model(self):
#         torch.manual_seed(0)
#
#         data_i = self.get_single_input()
#
#         if self.obj_dic is not None:
#             data_i['obj_dic'] = self.obj_dic
#
#         generated = self.model(data_i, mode='UI_mode')
#         generated_img = self.convert_output_image(generated)
#         generated_img = generated_img.copy()
#
#         self.generated_img = generated_img
#
#     def load_average_feature(self):
#
#         ############### load average features
#
#         average_style_code_folder = 'styles_test/mean_style_code/mean/'
#         input_style_dic = {}
#
#         ############### hard coding for categories
#
#         for i in range(19):
#             input_style_dic[str(i)] = {}
#
#             average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
#             average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
#                                      average_category_folder_list]
#
#             for style_code_path in average_category_list:
#                 input_style_dic[str(i)][style_code_path] = torch.from_numpy(
#                     np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
#
#         self.obj_dic = input_style_dic
#
#     def load_input_feature(self):
#
#         ############### load average features
#
#         average_style_code_folder = 'styles_test/mean_style_code/mean/'
#         input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(self.GT_img_path)
#         input_style_dic = {}
#         self.label_count = []
#
#         self.style_img_mask_dic = {}
#
#         for i in range(19):
#             input_style_dic[str(i)] = {}
#
#             input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
#             input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]
#
#             average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
#             average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
#                                      average_category_folder_list]
#
#             for style_code_path in average_category_list:
#                 if style_code_path in input_category_list:
#                     input_style_dic[str(i)][style_code_path] = torch.from_numpy(
#                         np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
#
#                     if style_code_path == 'ACE':
#                         self.style_img_mask_dic[str(i)] = self.GT_img_path
#                         self.label_count.append(i)
#
#                 else:
#                     input_style_dic[str(i)][style_code_path] = torch.from_numpy(
#                         np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
#
#         self.obj_dic = input_style_dic
#         # self.obj_dic_back = copy.deepcopy(self.obj_dic)
#         self.obj_dic_GT = copy.deepcopy(self.obj_dic)
#
#     def load_input_feature(self):
#
#         ############### load average features
#
#         average_style_code_folder = 'styles_test/mean_style_code/mean/'
#         input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(self.GT_img_path)
#         input_style_dic = {}
#         self.label_count = []
#
#         self.style_img_mask_dic = {}
#
#         for i in range(19):
#             input_style_dic[str(i)] = {}
#
#             input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
#             input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]
#
#             average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
#             average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
#                                      average_category_folder_list]
#
#             for style_code_path in average_category_list:
#                 if style_code_path in input_category_list:
#                     input_style_dic[str(i)][style_code_path] = torch.from_numpy(
#                         np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
#
#                     if style_code_path == 'ACE':
#                         self.style_img_mask_dic[str(i)] = self.GT_img_path
#                         self.label_count.append(i)
#
#                 else:
#                     input_style_dic[str(i)][style_code_path] = torch.from_numpy(
#                         np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
#
#         self.obj_dic = input_style_dic
#         # self.obj_dic_back = copy.deepcopy(self.obj_dic)
#         self.obj_dic_GT = copy.deepcopy(self.obj_dic)
#
#     # get input images and labels
#     def get_single_input(self):
#
#         image_path = self.GT_img_path
#         # image = self.GT_img
#         label_img = self.mat_img[:, :, 0]
#
#         label = Image.fromarray(label_img)
#         params = get_params(self.opt, label.size)
#         transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
#         label_tensor = transform_label(label) * 255.0
#         label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
#         label_tensor.unsqueeze_(0)
#
#         image_tensor = torch.zeros([1, 3, 256, 256])
#
#         # if using instance maps
#         if self.opt.no_instance:
#             instance_tensor = torch.Tensor([0])
#
#         input_dict = {'label': label_tensor,
#                       'instance': instance_tensor,
#                       'image': image_tensor,
#                       'path': image_path,
#                       }
#
#         return input_dict
#
#
#




