from addict import Dict
import torch.utils.data as data
import subprocess
import argparse
import random
import torch
import os

down_sample = 0
over_sample = 1

def _domain_2_decode_domain(domain):
    domain_list = domain.split('/')
    while True:
        if '.' in domain_list:
            domain_list.remove('.')
        elif '..' in domain_list:
            domain_list.remove('..')
        else:
            break
    return '_'.join(domain_list)       

class VideoDataset(data.Dataset):
    def __init__(self, root, transform = None, T = 10, use_cv = False, decode_root = './decode', sample_method = down_sample, to_tensor = True):
        """
            The video version of ImageDataset, and the rank of return tensor is BTCHW
            Some notation are defined in the following:
                * Video : The image sequence, and the length of video is various and large
                * Tuple : The small image sequence, and the length of tuple is T

            ------------------------------------------------------------------------------------------------
            In this class, we use 'info' object to store the whole information toward the video folders
            The structure of info is like this: 
            info = {

                # Store the information before decoding
                'origin' : {
                    'domains': ['dataset/A', 'dataset/B'],
                    'videos' : {
                        'dataset/A': ['video1.wav', 'video2.wav'],
                        'dataset/B': ['video1.wav', 'video2.wav'],
                    }
                },

                # Store the information after decoding
                'decode' : {
                    'domains': ['dataset_A', 'dataset_B'],
                    'videos' : {
                        'dataset_A': ['video1.wav', 'video2.wav'],
                        'dataset_B': ['video1.wav', 'video2.wav'],
                    },
                    'frames' : {
                        'video_A': ['frame01.png', 'frame02.png'],
                        'video_B': ['frame01.png', 'frame02.png'],
                    }
                }
            }
            Here is some definition of keys:
            * domains:  the list object, each element represent single domain name
            * videos :  the list object, each element represent the path of video folder
            * frames :  the dict object, the key of frames obj is the element in videos.
                        while the element is the list object which store the name of decoded frame

            Code notation:
                * root          : The list object which contains several "domain". 
                                  For example, ['./video/A', './video/B']
                * domain        : The str
                                  For example, './video/A'
                * decode_root   : The str
                                  For example, '.decode'
                * decode_domain : The str which is transferred from "domain". 
                                  For example, 'video_A'

            Arg:    root            - The list of different video folder
                    transform       - The transform.Compose object
                    decode_folder   - The temporal folder to store the decoded image
                    T               - The maximun image in single batch sequence
        """
        # Record the variable
        self.root = root
        self.transform = transform
        self.T = T
        self.use_cv = use_cv
        self.decode_root = decode_root
        self.sample_method = sample_method
        self.to_tensor = to_tensor
        self._len = None
        self.info = Dict({
            'domains': [],
            'videos': self.root,
            'frames': {}
        })

        # Obtain the list of video list
        self.videos = []
        for domain in self.root:
            self.videos.append(os.listdir(domain))

        # Check if the decode process should be conducted again
        should_decode = not os.path.exists(self.decode_root)
        if not should_decode:
            for domain in self.root:
                for video_name in domain:
                    folder_name = '.'.join(video_name.split('.')[:-1])
                    folder_path = os.path.join(self.decode_root, domain, folder_name)
                    if not os.path.exists(folder_path):
                        should_decode = True
                        break

        # Decode the video if needed
        if should_decode:
            if os.path.exists(self.decode_root):
                subprocess.call(['rm', '-rf', self.decode_root])
            os.mkdir(self.decode_root)
            for domain in self.root:
                self.decodeVideo(domain)

        # Obtain the list of frame list
        self.frames = []

    def decodeVideo(self, domain):
        """
            Decode the single video into a series of images, and store into particular folder

            Arg:    domain  - The str, the name of video domain
        """
        decode_domain = _domain_2_decode_domain(domain)
        decode_domain_path = os.path.join(self.decode_root, decode_domain)
        os.mkdir(decode_domain_path)
        for video_name in os.listdir(domain):
            folder_name = '.'.join(video_name.split('.')[:-1])
            folder_path = os.path.join(decode_domain_path, folder_name)
            source = os.path.join(domain, video_name)
            target = os.path.join(folder_path, "%5d.png")
            os.mkdir(folder_path)
            # print(' '.join(['ffmpeg', '-i', source, '-vframes', str(100), target]))
            subprocess.call(['ffmpeg', '-i', source, '-vframes', str(100), target])

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
        max_init_frame_idx = len(self.video_files[index]) - self.T - 1
        start_pos = random.randint(0, max_init_frame_idx)
        tuple_list = []
        for i in range(self.T):
            img_path = self.video_files[index][start_pos + i]
            if self.use_cv:
                import cv2
                img = cv2.imread(img_path)
            else:
                from PIL import Image
                img = Image.open(img_path)
            tuple_list.append(img)
        if self.transform:
            tuple_list = [self.transform(frame) for frame in tuple_list]
        if self.to_tensor:
            tuple_list = torch.cat(tuple_list, dim = 0)
        return tuple_list

if __name__ == '__main__':
    loader = data.DataLoader(
        dataset = VideoDataset(
            root = ['../../dataset/A', '../../dataset/B'], 
            transform = None, 
            T = 10, 
            use_cv = True
        ), batch_size = 1, shuffle = True
    )
    for sequence_a, sequence_b in loader:
        pass