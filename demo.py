from parse import parse_demo_args
from lib.model.recycle_gan2 import ReCycleGAN
from lib.utils import get_frame_rate
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

    In the intermediate of procedure, the 'demo_temp' folder will be created.
    Ane the structure can be addressed as following:

    demo_temp --+-- input: Store the decode frame of the video
                |
                +-- output: Store the rendered frame of the video
"""

def demo(args):
    """
        Define the demo procedure

        Arg:    args    - The argparse argument
    """   
    # Create the folders to store the intermediate frame
    FOLDER = {'root': 'demo_temp', 'in': 'demo_temp/input', 'out': 'demo_temp/output'}
    for key, folder in FOLDER.items():
        if not os.path.exists(folder):
            subprocess.call(['mkdir', '-p', folder])
    
    # Decode source video
    source = args.input
    target = os.path.join(FOLDER['in'], "%5d_img.jpg")
    subprocess.call(['ffmpeg', '-i', source, '-vframes', str(100), target])
    fps = get_frame_rate(source)
    frame_path_list = sorted(os.listdir(FOLDER['in']))

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

    # ---------------------------------------------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------------------------------------------
    with torch.no_grad():
        Q = []
        for i, img_name in enumerate(tqdm(frame_path_list)):
            # Load the image
            img = cv2.imread(os.path.join(FOLDER['in'], img_name))
            img = torch.from_numpy(img).unsqueeze(0)
            img = aug_op(img)
            Q.append(img)

            # Render toward the specific direction
            if i >= (args.t - 1):
                true_frame = Q[-1]
                true_tuple = Q[:args.t - 1]
                if args.direction == 'a2b':
                    images = model(true_a = true_frame, true_a_seq = true_tuple, warning = False)
                elif args.direction == 'b2a':
                    images = model(true_b = true_frame, true_b_seq = true_tuple, warning = False)
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
            img = img.cpu().numpy()
            cv2.imwrite(os.path.join(FOLDER['out'], img_name), img.astype(np.uint8))

    # Encode rendered images as video and remove intermediate folder
    source = os.path.join(FOLDER['out'], "%5d_img.jpg")
    target = args.output
    subprocess.call(['ffmpeg', '-i', source, "-vf", "fps=" + str(fps), target])
    subprocess.call(['rm', '-rf', FOLDER['root']])

if __name__ == '__main__':
    args = parse_demo_args()
    demo(args)