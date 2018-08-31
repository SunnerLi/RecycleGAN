import numpy as np
import torch
import cv2
import os

def INFO(string):
    print("[ ReCycle ] %s" % (string))

def getParentFolder(path):
    """
        Get the parent folder path
        ==================================================
        * Notice: this function only support UNIX system!
        ==================================================
        Arg:    path - The path you want to examine
        Ret:    The path of parent folder 
    """
    path_list = path.split('/')
    if len(path_list) == 1:
        return '.'
    else:
        return os.path.join(*path_list[:-1])

def visualizeSingle(images, save_path = 'val.png'):
    """
        Visualize the render image in single time step

        Arg:    images  - The dict object to represent validation result.
                          The structure of images is like:
                          {
                              'real_a': <real_a>,
                              'fake_b': <fake_b>,
                              'reco_a': <reco_a>,
                              'real_a': <real_a>,
                              'fake_b': <fake_b>,
                              'reco_a': <reco_a>,
                          }
    """   
    # Back to numpy
    images_np = {}
    for key in images.keys():
        image = images[key][0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = image * 127.5 + 127.5
        # print(key, image.size(), torch.min(image), torch.max(image))
        assert (torch.min(image) >= 0) and (torch.max(image) <= 255)
        images_np[key] = image

    # Concat and show
    result1 = np.hstack((images_np['true_a'], images_np['fake_b'], images_np['reco_a']))
    result2 = np.hstack((images_np['true_b'], images_np['fake_a'], images_np['reco_b']))
    result = np.vstack((result1, result2))
    cv2.imwrite(save_path, result.astype(np.uint8))