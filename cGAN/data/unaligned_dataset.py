import os
from data.base_dataset import BaseDataset, get_transform, get_transform_spec, get_transform_npy
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from torch import cat


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B_' + opt.spec_type)  # create a path '/path/to/data/trainB_<spec_type>'

        self.opt = opt
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.which_direction == 'BtoA'
        
        self.transform_A = get_transform(opt)
        if self.opt.npy_spec:
            self.transform_B = get_transform_npy()
        else:
            self.transform_B = get_transform_spec(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.random_batches:   # make sure index is within then range
            index_B = random.randint(0, self.B_size - 1)
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = index % self.B_size
        B_path = self.B_paths[index_B]

        if self.opt.output_nc == 1:
            mode = 'L' # for graysacel images
        else:
            mode = 'RGB' # for RGB images

        A_img = Image.open(A_path).convert(mode)

        if self.opt.npy_spec: # load spectrograms
            if self.opt.output_nc == 3:
                B_img = np.transpose(np.load(B_path), axes=(2, 0, 1))
            else:
                B_img = np.expand_dims(np.load(B_path)[:,:,0], 0)
        else: # load images
            B_img = Image.open(B_path).convert(mode)

        # apply transformations
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
