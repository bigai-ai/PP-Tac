import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append('./')
from DataLoader_test import TongTacDatasetTest
from DataLoader_multi_sensor import TongTacDataset_multi_sensor

import os
import numpy as np
import cv2

MASK_CENTER = (250, 340)
MASK_RADIUS = 170

def build_dataset_train(opt):
    if opt.multi_sensor == False:
        # collate_fn
        dataset_train = TongTacDataset(opt.data_gt, opt.data_cap, opt)
        dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle = True, num_workers=1, pin_memory=True, drop_last=False)
        # If need unique validation dataset, in this case, we only have test set
        dataset_val = TongTacDataset(opt.data_gt_eval, opt.data_cap_eval, opt)
        dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle = False, num_workers=1, pin_memory=True, drop_last=False)
    else:
        dataset_train = TongTacDataset_multi_sensor(opt.data_gt, opt.data_cap, opt)
        dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle = True, num_workers=1, pin_memory=True, drop_last=False)
        # If need unique validation dataset, in this case, we only have test set
        dataset_val = TongTacDataset_multi_sensor(opt.data_gt_eval, opt.data_cap_eval, opt)
        dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle = False, num_workers=1, pin_memory=True, drop_last=False)
    
    return dataloader_train, dataloader_val

def build_dataset_test(opt):
    # collate_fn
    dataset_test = TongTacDatasetTest(opt.data_cap_test, opt)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle = False, num_workers=1, pin_memory=True, drop_last=False)
    return dataloader_test, dataset_test


class TongTacDataset(Dataset):
    def __init__(self, gt_depth_path, cap_gray_path, opt):
        super(TongTacDataset, self).__init__()
        self.opt = opt
        self.gaussian_blur = opt.gaussian_blur
        self.gt_depth_path = gt_depth_path
        self.cap_gray_path = cap_gray_path 

        self.image_name_list = self.load_image_name() 
        self.ref_gt_depth_img, self.ref_cap_gray_img = self.load_ref_img(opt)
        self.h = self.ref_gt_depth_img.shape[0] 
        self.w = self.ref_gt_depth_img.shape[1]
        self.mask = self.get_mask(opt)

        self.ref_gt_depth_img[self.mask==0] =0
        self.ref_cap_gray_img[self.mask==0] =0

        self.gt_depth, self.cap_gray = self.load_image()

        delta_max, delta_min  = self.get_statistic_delta()
        print('detal_max:', delta_max)
        print('detal_min:', delta_min)
        # delta_max : 43.0    delta_min : 0.0
        self.DELTAMAX = 50.0

    def normalize_forward(self, img):
        if self.opt.normalize == 'divide max':
            img = img/self.DELTAMAX
        elif self.opt.normalize == 'max to 20':
            img[img>20] = 20
            img = img/20.0
        return img

    def normalize_backward(self, img):
        if self.opt.normalize == 'divide max':
            img = img*self.DELTAMAX
        elif self.opt.normalize == 'max to 20':
            img = img*20.0
        return img
    
    def process_gaussian_blur(self, img):
        img = cv2.GaussianBlur(img, (5,5), 5)
        return img

    def load_image_name(self):
        gt_depth_name_list = [ i for i in os.listdir(self.gt_depth_path) if 'ref' not in i]
        cap_gray_name_list = [i for i in os.listdir(self.cap_gray_path) if 'ref' not in i]
        
        final_name_list = []
        for name in gt_depth_name_list:
            if name in cap_gray_name_list:
                final_name_list.append(name)
        return final_name_list

    def load_image(self):
        cap_gray_list = []
        gt_depth_list = []
        for i in self.image_name_list:
            img_path_cap_gray = os.path.join(self.cap_gray_path, i)
            img_path_gt_depth = os.path.join(self.gt_depth_path, i)

            img_cap_gray = cv2.imread(img_path_cap_gray,  cv2.IMREAD_GRAYSCALE)
            if self.gaussian_blur:
                img_cap_gray = self.process_gaussian_blur(img_cap_gray)
            img_cap_gray = img_cap_gray.astype(np.float32)
            img_cap_gray[self.mask==0] =0.

            img_gt_depth = cv2.imread(img_path_gt_depth, cv2.IMREAD_UNCHANGED)
            img_gt_depth = img_gt_depth.astype(np.float32)
            img_gt_depth[self.mask==0] =0.

            cap_gray_list.append(img_cap_gray)
            gt_depth_list.append(img_gt_depth)
        return gt_depth_list, cap_gray_list

    def load_ref_img(self, opt):
        ref_gt_depth_path =os.path.join(self.gt_depth_path, 'ref_depth.png')
        ref_gt_depth = cv2.imread(ref_gt_depth_path, cv2.IMREAD_UNCHANGED)
        ref_gt_depth = ref_gt_depth.astype(np.float32)

        ref_cap_gray_path =opt.cap_ref_path
        ref_cap_gray = cv2.imread(ref_cap_gray_path, cv2.IMREAD_GRAYSCALE)
        if self.gaussian_blur:
            ref_cap_gray = self.process_gaussian_blur(ref_cap_gray)
        ref_cap_gray = ref_cap_gray.astype(np.float32)
        return ref_gt_depth, ref_cap_gray

    def get_gt_delta_depth(self, gt_depth):
        delta = self.ref_gt_depth_img - gt_depth
        delta[delta<0] = 0
        return delta
    
    def get_statistic_delta(self):
        delta_max_list = []
        delta_min_list = []
        for img in self.gt_depth:
            delta = self.get_gt_delta_depth(img)
            delta_max = delta.max()
            delta_min = delta.min()

            delta_max_list.append(delta_max)
            delta_min_list.append(delta_min)
        
        delta_max = max(delta_max_list)
        delta_min = min(delta_min_list)
        return delta_max, delta_min

    
    def get_mask(self,opt):
        # (250, 340), 170
        mask = np.zeros(self.ref_gt_depth_img.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if np.linalg.norm([i-opt.MASK_CENTER[0], j-opt.MASK_CENTER[1]]) < opt.MASK_RADIUS:
                    mask[i, j] = 1
        return mask

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        ref_cap_gray_img = self.ref_cap_gray_img
        gt_depth = self.gt_depth[idx]
        cap_gray = self.cap_gray[idx]
        # Normalize the delta to [0,1]
        gt_depth_delta = self.normalize_forward(self.get_gt_delta_depth(gt_depth))

        # delta capture 
        MIN = self.opt.CAPTURE_DELTA_MIN
        MAX = self.opt.CAPTURE_DELTA_MAX
        cap_gray_delta = ref_cap_gray_img - cap_gray  
        cap_gray_delta = cap_gray_delta - MIN
        cap_gray_delta[cap_gray_delta < 0] = 0
        cap_gray_delta[cap_gray_delta > MAX] = 0
        cap_gray_delta = cap_gray_delta / MAX

        cap_gray_delta_mask = np.zeros(cap_gray_delta.shape)
        cap_gray_delta_mask[cap_gray_delta!=0] = 1

        # Normalize the cap to [0,1]
        cap_gray = cap_gray/255.0
        ref_cap_gray_img = ref_cap_gray_img/255.0
        

        #reshape the image to (h, w, 1)
        cap_gray_delta = torch.from_numpy(cap_gray_delta[:,:,np.newaxis]).permute(2, 0, 1)
        gt_depth = torch.from_numpy(gt_depth[:,:,np.newaxis]).permute(2, 0, 1)
        cap_gray = torch.from_numpy(cap_gray[:,:,np.newaxis]).permute(2, 0, 1)
        gt_depth_delta = torch.from_numpy(gt_depth_delta[:,:,np.newaxis]).permute(2, 0, 1)
        ref_cap_gray_img = torch.from_numpy(ref_cap_gray_img[:,:,np.newaxis]).permute(2, 0, 1)
        cap_gray_delta_mask = torch.from_numpy(cap_gray_delta_mask[:,:,np.newaxis]).permute(2, 0, 1)

        return gt_depth, cap_gray, gt_depth_delta, self.mask, ref_cap_gray_img, cap_gray_delta, cap_gray_delta_mask





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import parse_args, CompSet
    comp_opt = CompSet()
    opt = comp_opt.opt

    dataset = TongTacDataset(opt.data_gt, opt.data_cap, opt)
    print(len(dataset))

    dataloader_train, dataloader_val = build_dataset_train(opt)

    for batch_idx, data in enumerate(dataloader_train):
        img_gt, img_cap, img_gt_delta, mask , ref_cap_gray_img, cap_gray_delta, cap_gray_delta_mask= data


        cap_gray_delta = cap_gray_delta.squeeze(0).permute(1,2,0).numpy().reshape(dataset.h, dataset.w) 
        plt.imshow(cap_gray_delta, cmap=plt.cm.gray_r)
        plt.show()     
        waitkey = cv2.waitKey(0)
        print(cap_gray_delta.shape)
        print(cap_gray_delta.max())

        
        cap_gray_delta_mask = cap_gray_delta_mask.squeeze(0).permute(1,2,0).numpy().reshape(dataset.h, dataset.w) 
        plt.imshow(cap_gray_delta_mask, cmap=plt.cm.gray_r)
        plt.show()     
        waitkey = cv2.waitKey(0)
        print(cap_gray_delta_mask.shape)
        print(cap_gray_delta_mask.max())


        img_gt = img_gt.squeeze(0).permute(1,2,0).numpy().reshape(dataset.h, dataset.w) 
        import matplotlib.pyplot as plt
        plt.imshow(img_gt, cmap=plt.cm.gray_r)
        plt.show()     
        waitkey = cv2.waitKey(0)
        print(img_gt.shape)
        print(img_gt.max())

        img_cap = img_cap.squeeze(0).permute(1,2,0).numpy().reshape(dataset.h, dataset.w) 
        import matplotlib.pyplot as plt
        plt.imshow(img_cap, cmap=plt.cm.gray_r)
        plt.show()     
        waitkey = cv2.waitKey(0)
        print(img_cap.shape)
        print(img_cap.max())

        img_gt_delta = img_gt_delta.squeeze(0).permute(1,2,0).numpy().reshape(dataset.h, dataset.w) 
        import matplotlib.pyplot as plt
        plt.imshow(img_gt_delta, cmap=plt.cm.gray_r)
        plt.show()     
        waitkey = cv2.waitKey(0)
        print(img_gt_delta.shape)
        print(img_gt_delta.max())

        mask = mask.squeeze(0).numpy().reshape(dataset.h, dataset.w) 
        import matplotlib.pyplot as plt
        plt.imshow(mask, cmap=plt.cm.gray_r)
        plt.show()     
        waitkey = cv2.waitKey(0)
        print(mask.shape)

        exit(0)

    





