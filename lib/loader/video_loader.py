from torch.utils import data
from lib.utils import INFO
from glob import glob
from tqdm import tqdm
import numpy as np
import subprocess
import torch
import cv2
import os

"""
    This script define the loader to deal with video data.
    We treat each video as single folder, and each folder contains bunch of images (frame)
    Since it is time consuming to decode and encode the video by OpenCV,
    we adopt ffmpeg to deal with this work

    You should install the ffmpeg by the cmd: 
    $ sudo apt install ffmpeg
"""

class videoLoader(data.Dataset):
    """
        nearest explain
    """
    def __init__(self, A_root, B_root, img_size = None, t = 2, T = 30, augmentations = None, temp_folder = './decode/'):
        """
        """
        # Record the parameters
        self.A_root = A_root
        self.B_root = B_root
        self.img_size = img_size
        self.T = T
        self.augmentations = augmentations
        self.temp_folder = temp_folder

        # -------------------------------------------------------------------------------------------
        # Decode the videos
        # There are two cases that the videos will be decoded again:
        #   1. The temp folder is not exist
        #   2. The folder with video name is missing
        # Before decoding the video, the temp folder will be removed completely if the folder is exist
        # 
        # ref: https://stackoverflow.com/questions/35771821/ffmpeg-decoding-mp4-video-file
        # -------------------------------------------------------------------------------------------
        should_decode = not os.path.exists(temp_folder)
        if not should_decode:
            for video_name in os.listdir(self.A_root):
                if not os.path.exists(os.path.join(temp_folder, video_name)) and not should_decode:
                    should_decode = True
                    break
            for video_name in os.listdir(self.B_root):
                if not os.path.exists(os.path.join(temp_folder, video_name)) and not should_decode:
                    should_decode = True
                    break
        if should_decode:
            if os.path.exists(temp_folder):                     # Remove the dummy (or broken) folder
                subprocess.call(["rm", "-rf", temp_folder])
            os.mkdir(temp_folder)
            self.decode(video_root = self.A_root, decode_folder = 'A')
            self.decode(video_root = self.B_root, decode_folder = 'B')

        # Show the video folder information
        self.files = dict()
        self.files['A'] = os.listdir(self.A_root)
        self.files['B'] = os.listdir(self.B_root)
        INFO("Dataset type : %s \t Video number in A: %d\t  Video number in B: %d" % \
                (self.__class__.__name__[:-6], len(self.files['A']), len(self.files['B'])))

    def decode(self, video_root, decode_folder):
        """
            Decode the video into the temp folder
            The structure is shown below:
            $ ls
            >> temp_folder --+-- decode_folder --+-- video_name_1
                                                 |
                                                 +-- video_name_2
                                                 |
                                                 +-- video_name_3
                                                 ...

            Arg:    video_root      - The root folder of video
                    decode_folder   - The decode folder under temp root folder
        """        
        INFO("---------- Start Decode Video ----------")
        video_list = os.listdir(video_root)
        for video_name in tqdm(video_list):
            # Create the folder for each video
            folder_name = ".".join(video_name.split('.')[:-1])      # remove the postfix
            os.mkdir(os.path.join(self.temp_folder, folder_name))

            # Decode the video
            video_path = os.path.join(video_root, video_name)
            subprocess.call(["ffmpeg", "-i", video_path, os.path.join(self.temp_folder, decode_folder, folder_name, "image-%05d.jpg")])
            # print("    ".join(
            #     ["ffmpeg", "-i", video_path, os.path.join(self.temp_folder, folder_name + "image-%05d.jpg")])
            # )

        INFO("---------- Finish Decode Video ----------")
        exit()

    def __len__(self):
        return len(self.files['A'])

    def __getitem__(self, index):
        # Initalize the parameters
        video_A_name = os.path.join(self.A_root, self.files['A'][index])
        video_B_name = os.path.join(self.B_root, self.files['B'][index])
        video_A_list = []
        video_B_list = []

        # Collect the video frame and resize
        cap = cv2.VideoCapture(video_A_name)
        counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if (self.T_max != -1 and counter > self.T_max) or frame is None:
                break
            if self.img_size is not None:
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
            video_A_list.append(frame)
            counter += 1

        cap = cv2.VideoCapture(video_B_name)
        counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if (self.T_max != -1 and counter > self.T_max) or frame is None:
                break
            if self.img_size is not None:
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
            video_B_list.append(frame)
            counter += 1


        # Argmentation toward the whole video
        if self.augmentations is not None:
            video_A_list, video_B_list = self.augmentations(video_A_list, video_B_list)

        # Normalize
        video_A_tensor = torch.from_numpy(np.asarray(video_A_list, dtype = np.float))
        video_B_tensor = torch.from_numpy(np.asarray(video_B_list, dtype = np.float))
        video_A_tensor = (video_A_tensor - 127.5) / 127.5
        video_B_tensor = (video_B_tensor - 127.5) / 127.5
        min_value = min(torch.min(video_A_tensor).item(), torch.min(video_B_tensor).item())
        max_value = max(torch.max(video_A_tensor).item(), torch.max(video_B_tensor).item())
        assert min_value >= -1 and max_value <= 1

        # Transform
        video_A_tensor = video_A_tensor.transpose(2, 3).transpose(1, 2)
        video_B_tensor = video_B_tensor.transpose(2, 3).transpose(1, 2)

        # Transfer as float tensor
        video_A_tensor = video_A_tensor.float()
        video_B_tensor = video_B_tensor.float()

        return video_A_tensor, video_B_tensor