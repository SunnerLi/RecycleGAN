from addict import Dict
from glob import glob
import torch.utils.data as data
import numpy as np
import subprocess
import argparse
import random
import torch
import math
import os

down_sample = 0
over_sample = 1
with_tuple_form = 0
without_tuple_form = 1

def _domain2folder(domain):
    domain_list = domain.split('/')
    while True:
        if '.' in domain_list:
            domain_list.remove('.')
        elif '..' in domain_list:
            domain_list.remove('..')
        else:
            break
    return '_'.join(domain_list)       

def _file2folder(file_name):
    return '_'.join(file_name.split('.')[:-1])

def to_folder(name):
    if os.path.isdir(name):
        domain_list = name.split('/')
        while True:
            if '.' in domain_list:
                domain_list.remove('.')
            elif '..' in domain_list:
                domain_list.remove('..')
            else:
                break
        return '_'.join(domain_list)
    else:
        return '_'.join(name.split('.')[:-1])    

class VideoDataset(data.Dataset):
    def __init__(self, root, transform = None, T = 10, t = 2, rank_form = without_tuple_form,
                    use_cv = False, decode_root = './.decode', sample_method = down_sample, to_tensor = True):
        """
            The video version of ImageDataset, and the rank of return tensor is BTCHW
            If you assign t, then the rank of return tensor is BTtCHW
            Some notation are defined in the following:
                * Video : The image sequence, and the length of video is various and large
                * Tuple : The small image sequence, and the length of tuple is T

            Code notation:
                * root          : The list object
                                  The each "domain" can be obtained by accessing the list iteratively
                                  For example, ['./video/A', './video/B']
                * domain        : The str object
                                  The each "video" can be obtained
                                  For example, './video/A'
                * decode_root   : The str
                                  For example, '.decode'
                * decode_domain : The str which is transferred from "domain". 
                                  For example, 'video_A'

+-----------+----------------------------------+------------------------------------------+----------------------------------+-----------------------------+
|   Name    |           Operation1             |        Return after Operation1           |         Operation2               |    Return after Operation2  |
+-----------+----------------------------------+------------------------------------------+----------------------------------+-----------------------------+
|   root    |   Access the list iteratively    |                domain                    |             -                    |       -                     | 
+-----------+----------------------------------+------------------------------------------+----------------------------------+-----------------------------+            
|  domain   |     to_folder(domain)            |  the domain folder name after decoding   |       os.listdir(domain)         |       video                 |     
+-----------+----------------------------------+------------------------------------------+----------------------------------+-----------------------------+
|   video   |     to_folder(video)             |  the video folder name after decoding    |    os.listdir(self.decode_root,  |       frame                 |     
|           |                                  |                                          |                to_folder(domain),|                             |         
|           |                                  |                                          |                to_folder(video)  |                             |     
|           |                                  |                                          |    )                             |                             |     
+-----------+----------------------------------+------------------------------------------+----------------------------------+-----------------------------+
|   frame   |               -                  |                   -                      |             -                    |        -                    |
+-----------+----------------------------------+------------------------------------------+----------------------------------+-----------------------------+

            Arg:    root            - The list of different video folder
                    transform       - The transform.Compose object
                    decode_folder   - The temporal folder to store the decoded image
                    T               - The maximun image in single batch sequence
        """
        # Record the variable
        self.root = root
        self.transform = transform
        self.T = T
        self.t = t
        self.T_ext = self.T + self.t        # the number of frame in each batch.
        self.rank_form = rank_form
        self.use_cv = use_cv
        self.decode_root = decode_root
        self.sample_method = sample_method
        self.to_tensor = to_tensor
        self._len = None

        # Obtain the list of video list
        self.video_files = {}
        for domain in self.root:
            self.video_files[domain] = os.listdir(domain)

        # Check if the decode process should be conducted again
        should_decode = not os.path.exists(self.decode_root)
        if not should_decode:
            for domain in self.root:
                for video in os.listdir(domain):
                    if not os.path.exists(os.path.join(self.decode_root, to_folder(domain), to_folder(video))):
                        should_decode = True
                        break

        # Decode the video if needed
        if should_decode:
            if os.path.exists(self.decode_root):
                subprocess.call(['rm', '-rf', self.decode_root])
            os.mkdir(self.decode_root)
            self.decodeVideo()

        # Obtain the list of frame list
        self.frames = Dict()
        for domain in os.listdir(self.decode_root):
            self.frames[domain] = []
            for video in os.listdir(os.path.join(self.decode_root, domain)):
                self.frames[domain] += [glob(os.path.join(self.decode_root, domain, video, "*"))]

    def decodeVideo(self):
        """
            Decode the single video into a series of images, and store into particular folder

            Arg:    domain  - The str, the name of video domain
        """
        for domain in self.root:
            os.mkdir(os.path.join(self.decode_root, to_folder(domain)))
            for video in os.listdir(domain):
                os.mkdir(os.path.join(self.decode_root, to_folder(domain), to_folder(video)))
                source = os.path.join(domain, video)
                target = os.path.join(self.decode_root, to_folder(domain), to_folder(video), "%5d.png")
                # subprocess.call(['ffmpeg', '-i', source, '-vframes', str(100), target])
                subprocess.call(['ffmpeg', '-i', source, target])

    def __len__(self):
        """
            Return the number of video depend on the sample method
        """
        if self._len is None:
            self._len = len(os.listdir(self.root[0]))
            for domain in self.root:
                if self.sample_method == down_sample:
                    self._len = min(self._len, len(os.listdir(domain)))
                elif self.sample_method == over_sample:
                    self._len = max(self._len, len(os.listdir(domain)))
                else:
                    raise Exception("The sample method {} is not support".format(self.sample_method))
        return self._len

    def __getitem__(self, index):
        """
            Return single batch of data, and the rank is BTCHW
        """
        result = []
        for domain in self.root:
            film_sequence = []
            if self.rank_form == without_tuple_form:
                max_init_frame_idx = len(self.frames[to_folder(domain)][index]) - self.T_ext
                start_pos = random.randint(0, max_init_frame_idx)
                for i in range(self.T_ext):
                    img_path = self.frames[to_folder(domain)][index][start_pos + i]
                    if self.use_cv:
                        import cv2
                        img = cv2.imread(img_path)
                    else:
                        from PIL import Image
                        img = np.asarray(Image.open(img_path))
                    film_sequence.append(img)
            else:
                max_init_frame_idx = len(self.frames[to_folder(domain)][index]) - self.T
                start_pos = random.randint(self.t // 2, max_init_frame_idx)
                for i in range(self.T):
                    film_tuple = []
                    for j in range(self.t):
                        img_path = self.frames[to_folder(domain)][index][start_pos + i + j - self.t // 2]
                        if self.use_cv:
                            import cv2
                            img = cv2.imread(img_path)
                        else:
                            from PIL import Image
                            img = np.asarray(Image.open(img_path))
                        film_tuple.append(img)
                    film_tuple = np.asarray(film_tuple)
                    film_sequence.append(film_tuple)

            # Transform the film sequence
            film_sequence = np.asarray(film_sequence)
            # print('film_sequence size: ', film_sequence.shape)
            if self.transform:
                film_sequence = self.transform(film_sequence)

            # Transfer the object as torch.Tensor object
            if self.to_tensor:
                if type(film_sequence) is not torch.Tensor:
                    film_sequence = torch.from_numpy(film_sequence)
            result.append(film_sequence)
        return result

if __name__ == '__main__':
    loader = data.DataLoader(
        dataset = VideoDataset(
            root = ['../../dataset/A', '../../dataset/B'], 
            transform = None, 
            T = 10, 
            t = 3,
            use_cv = False,
        ), batch_size = 1, shuffle = True
    )
    for sequence_a, sequence_b in loader:
        print('sequence_a shape: ', sequence_a.size())