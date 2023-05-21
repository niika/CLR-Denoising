import glob
import random
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import os
import torchvision
import numpy as np
from random import sample 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from tqdm import tqdm
def show(imgs):
    if isinstance(imgs, torch.Tensor):
        imgs = make_grid(imgs)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False,figsize=(20,20))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def ls(directory):
    return [os.path.join(directory, path) for path in os.listdir(directory)]

class Concat(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length
    
    

class SIDAR(Dataset):
    
    def __init__(self, root_dir, sequence_length=10, resize= None):
        self.root_dir = root_dir
        self.sequence_dir = glob.glob(root_dir+"/*")
        self.sequence_dir.sort()
        
        
        errors = []
        for directory in glob.glob(root_dir+"/*"):
            for file in list(range(1,11)) + ["gt"]:
                if not os.path.exists(directory+f"/{file}.png"):
                    errors.append(directory)
                
        diff = []
        for element in self.sequence_dir:
            if element not in errors:
                diff.append(element)
        self.sequence_dir = diff
        
        self.images = []
        self.gt = []
        for path in tqdm(self.sequence_dir): 
            self.images.append([read_image(path+"/{}.png".format(i) , torchvision.io.ImageReadMode.RGB) for i in range(1,sequence_length+1) ])#+[path+"/front.png"]
            self.gt.append(read_image(path+"/gt.png",torchvision.io.ImageReadMode.RGB)) # <<<<<< Change this to real ground truth
        self.n = len(self.sequence_dir)
        self.sequence_length = sequence_length

        if resize is not None:
            self.resize = torchvision.transforms.Resize(resize)
        else:
            self.resize = torch.nn.Identity()

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):  
        

        images = random.sample(self.images[idx], self.sequence_length)
        images = [img/255. for img in images]
        gt = self.gt[idx]
        #if gt.size()[1]>gt.size()[2]:
        #    gt = F.rotate(gt, -90)

        #images = torch.stack(images)
        return images, gt