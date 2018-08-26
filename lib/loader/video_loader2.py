from addict import Dict
import torch.utils.data as data
import subprocess
import argparse
import random
import torch
import os

down_sample = 0
over_sample = 1

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
            # info = {

            #     # Store the information before decoding
            #     'origin' : {
            #         'domains': ['dataset/A', 'dataset/B'],
            #         'videos' : {
            #             'dataset/A': ['video1.wav', 'video2.wav'],
            #             'dataset/B': ['video1.wav', 'video2.wav'],
            #         }
            #     },

            #     # Store the information after decoding
            #     'decode' : {
            #         'domains': ['dataset_A', 'dataset_B'],
            #         'videos' : {
            #             'dataset_A': ['video1', 'video2'],
            #             'dataset_B': ['video1', 'video2'],
            #         },
            #         'frames' : {
            #             'dataset_A': {
            #                 'video1': ['frame01.png', 'frame02.png']
            #                 'video2': ['frame01.png', 'frame02.png']
            #             },
            #             'dataset_B': {
            #                 'video1': ['frame01.png', 'frame02.png']
            #                 'video2': ['frame01.png', 'frame02.png']
            #             }
            #         }
            #     }
            # }

            info = {
                'domains': ['dataset/A', 'dataset/B'],
                'videos' : {
                    'dataset/A': ['dataset/A/video1.wav', 'dataset/A/video2.wav'],
                    'dataset/B': ['dataset/B/video1.wav', 'dataset/B/video2.wav'],
                }
                'frames' : {
                    'dataset/A': [['.decode/dataset_A/video1/1.png', '.decode/dataset_A/video1/2.png'], ['.decode/dataset_A/video2/1.png', '.decode/dataset_A/video2/2.png']]
                    'dataset/B': [['.decode/dataset_B/video1/1.png', '.decode/dataset_B/video1/2.png'], ['.decode/dataset_B/video2/1.png', '.decode/dataset_B/video2/2.png']]
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
        # TODO: Try to use the original style
        
        # Record the variable
        # self.root = root
        self.transform = transform
        self.T = T
        self.use_cv = use_cv
        self.decode_root = decode_root
        self.sample_method = sample_method
        self.to_tensor = to_tensor
        self._len = None
        # self.info = Dict({
        #     'origin' : {
        #         'domains': root,
        #         'videos' : {}
        #     },
        #     'decode' : {
        #         'domains': [],
        #         'videos' : {},
        #         'frames' : {}
        #     }
        # })
        self.info = Dict({
            'domains': root,
            'videos': {},
            'frames': {}
        })

        # ------------------------------------------------------------------
        # Complete the info object
        # ------------------------------------------------------------------
        from glob import glob
        # for domain in self.info.origin.domains:
        #     # origin.videos
        #     self.info.origin.videos[domain] = os.listdir(domain)

        #     # decode.domains
        #     domain_folder = _domain2folder(domain)
        #     self.info.decode.domains.append(domain_folder)

        #     # decode.videos
        #     self.info.decode.videos[domain] = [_file2folder(video_name) for video_name in os.listdir(domain)]
        for domain in self.info.domains:
            self.info.videos[domain] = glob(domain)

        # Check if the decode process should be conducted again
        should_decode = not os.path.exists(self.decode_root)
        if not should_decode:
            # for domain in self.info.origin.videos.keys():
            #     for video_name in self.info.origin.videos[domain]:
            #         folder_name = _file2folder(video_name)
            #         print(folder_name, self.info.decode.videos[domain])
            #         if folder_name not in self.info.decode.videos[domain]:
            #             should_decode = True
            #             break
            for domain in self.info.domains:
                for video in self.info.videos[domain]:
                    if _file2folder(video)

        # Decode the video if needed
        if should_decode:
            if os.path.exists(self.decode_root):
                subprocess.call(['rm', '-rf', self.decode_root])
            os.mkdir(self.decode_root)
            self.decodeVideo()

        # Obtain the list of frame list
        self.frames = []
        print(self.info)

        exit()

    def decodeVideo(self):
        """
            Decode the single video into a series of images, and store into particular folder

            Arg:    domain  - The str, the name of video domain
        """
        for domain_folder in self.info.decode.domains:
            os.mkdir(os.path.join(self.decode_root, domain_folder))
            for video_name in self.info.origin.videos[domain_folder]:
                os.mkdir(os.path.join(self.decode_root, domain_folder, _file2folder(video_name)))
                source = os.path.join(domain_folder, video_name)
                target = os.path.join(self.decode_root, domain_folder, _file2folder(video_name), "%5d.png")
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