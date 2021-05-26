"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # dataset_path = './datasets/celeba_green/'
        # dataset_path = './datasets/facescape_0324_addhair/'
        # dataset_path = './datasets/bg2/'
        dataset_path = "./datasets/facescape_0324_addhair_3/"
        # pose_ind = "5/"
        # dataset_path = './datasets/facescape_0325/' + pose_ind
        # result_path = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0325/" + pose_ind
        # result_path = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324_addhair_green/"
        # result_path = "./bg2/"
        # result_path = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324_addhair_bg2/"
        # result_path = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324_addhair_randomstyle/"

        dataset_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517_addhair/"
        result_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/results/0517_addhair/"

        parser.add_argument('--label_dir', type=str, default=dataset_path + 'test_label/', help='saves results here.')
        parser.add_argument('--image_dir', type=str, default=dataset_path + 'test_img/', help='saves results here.')
        parser.add_argument('--instance_dir', type=str, default=dataset_path, help='saves results here.')
        parser.add_argument('--results_dir', type=str, default=result_path, help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')

        parser.add_argument('--status', type=str, default='test')

        self.isTrain = False
        return parser
