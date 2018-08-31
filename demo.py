from parse import parse_demo_args
from lib.model.recycle_gan2 import ReCycleGAN
from tqdm import tqdm
import lib.augmentations as aug
import numpy as np
import subprocess
import torch
import cv2
import os

"""
    This script define the demo procedure to transfer the video to the opposite domain
    Currently, this script only support OpenCV backend only
"""


def demo(args):
    """
        Define the demo procedure

        Arg:    args    - The argparse argument
    """
    # TODO: complete demo main part
    
    # TODO: Re-order the folder path
    DECODE_FOLDER = 'demo_decode'
    RENDER_FOLDER = 'result_decode'
    
    # 1. Decode source video
    if not os.path.exists(DECODE_FOLDER):
        os.mkdir(DECODE_FOLDER)
    if not os.path.exists(RENDER_FOLDER):
        os.mkdir(RENDER_FOLDER)
    source = args.input
    target = os.path.join(DECODE_FOLDER, "%5d_img.jpg")
    subprocess.call(['ffmpeg', '-i', source, '-vframes', str(100), target])
    frame_path_list = sorted(os.listdir(DECODE_FOLDER))

    # Define the tensor preprocess
    aug_op = aug.Compose([
        aug.ToFloat(),
        aug.Transpose(aug.BHWC2BCHW),
        aug.Resize(size_tuple = (args.H, args.W)),
        aug.Normalize()
    ])

    # Create model
    model = ReCycleGAN(
        A_channel = args.A_channel, 
        B_channel = args.B_channel, 
        r = args.r, 
        t = 3, 
        device = args.device
    )
    if not os.path.exists(args.resume):
        raise Exception("You should ensure the pre-trained model: {} is exist" % args.resume)
    else:
        model.load_state_dict(torch.load(args.resume))
    model.eval()

    # For
    Q = []
    for i, img_name in enumerate(tqdm(frame_path_list)):
        # 2-1. Load the image
        img = cv2.imread(os.path.join(DECODE_FOLDER, img_name))
        img = torch.from_numpy(img).unsqueeze(0)
        img = aug_op(img)
        Q.append(img)
        # print(img.size(), torch.min(img), torch.max(img))

        # Render toward the specific direction

        # TODO: fix bug
        
        if i >= (args.t - 1):
            true_frame = Q[-1]
            true_tuple = Q[:args.t - 1]
            if args.direction == 'a2b':
                images = model(true_a = true_frame, true_a_tuple = true_tuple, warning = False)
            elif args.direction == 'b2a':
                images = model(true_b = true_frame, true_b_tuple = true_tuple, warning = False)
            else:
                raise Exception("Invalid direction: {}" % args.direction)
        else:
            continue

        # Save
        if args.direction == 'a2b':
            img = images['fake_b'][0]
        else:
            img = images['fake_a'][0]
        img = img.transpose(0, 1).transpose(1, 2)
        img = img * 127.5 + 127.5
        assert (torch.min(img) >= 0) and (torch.max(img) <= 255)
        cv2.imwrite(os.path.join(RENDER_FOLDER, img_name), img.astype(np.uint8))

    # Encode rendered images as video
    # TODO: unfinish...

if __name__ == '__main__':
    args = parse_demo_args()
    demo(args)