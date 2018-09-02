from lib.utils import INFO
import torch.nn.functional as F
import numpy as np
import random
import torch
import cv2

"""
    This script defines some operation to do the augmentation toward video sequence
"""

BCHW2BHWC = 0
BHWC2BCHW = 1

def rotate(image, angle, center=None, scale=1.0):
    """
        Rotate the image with specific angle
        The code is borrowed from: https://www.jianshu.com/p/b5c29aeaedc7
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# =======================================================================================================
# The augmentation operators we define includes:
#   1. to tensor
#   2. to float
#   3. transpose
#   4. resize
#   5. normalize
#   6. random rotate
#   7. random horizontally flip
# =======================================================================================================

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, video_sequence):
        """
            Do the series of augmentation the video sequence

            Arg:    video_sequence_list - The list of video sequence
            Ret:    The list of argumented video sequence
        """
        for aug_op in self.augmentations:
            video_sequence = aug_op(video_sequence)
        return video_sequence

class ToTensor():
    def __call__(self, tensor):
        return torch.from_numpy(tensor)

class ToFloat():
    def __call__(self, tensor):
        return tensor.float()

class Transpose():
    def __init__(self, transfer_method = BHWC2BCHW):
        self.transfer_method = transfer_method

    def __call__(self, tensor):
        if self.transfer_method:
            return tensor.transpose(-1, -2).transpose(-2, -3)
        else:
            return tensor.transpose(-2, -3).transpose(-1, -2)

class Resize():
    def __init__(self, size_tuple, use_cv = True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv
        INFO("You should be aware that the tensor rank should be BTCHW or BTtCHW")

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """ 
        if len(tensor.size()) == 4:
            return F.interpolate(tensor, size = [self.size_tuple[0], self.size_tuple[1]])
        else:
            raise Exception("This function don't support the input whose rank is not 4...")

class Normalize():
    def __call__(self, tensor):
        """
            Normalized the tensor into the range [-1, 1]

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Normalized tensor
        """ 
        tensor = (tensor - 127.5) / 127.5
        assert (torch.min(tensor) >= -1) and (torch.max(tensor) <= 1)
        return tensor
        
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, tensor):
        """
            Do the rotation with random angle

            Arg:    tensor  - The np.ndarray or torch.Tensor obj whose rank is 4
                    seed    - The random seed which will be used to determine angle
            Ret:    Rotated tensor
        """       
        # Determine the rotation angle
        seed = random.random()
        is_tensor = tensor is torch.Tensor
        if is_tensor:
            tensor = tensor.numpy()
        rotate_degree = seed * 2 * self.degree - self.degree

        if len(tensor.shape) == 4:
            # Deal with rank=5 situation (BTCHW)
            video_sequence = []
            for frame in tensor:
                video_sequence.append(rotate(frame, rotate_degree))
        else:
            raise Exception("This function don't support the input whose rank is not 4...")

        # Transfer back to torch.Tensor if needed
        video_sequence = np.asarray(video_sequence)
        if is_tensor:
            video_sequence = torch.from_numpy(video_sequence)
        return video_sequence

class RandomHorizontallyFlip(object):
    def __call__(self, tensor):
        """
            Do the horizontally flip randomly

            Arg:    tensor  - The np.ndarray or torch.Tensor obj whose rank is 4
            Ret:    Flipped tensor
        """ 
        seed = random.random()
        if seed > 0.5:
            return tensor
        else:
            is_tensor = tensor is torch.Tensor
            if is_tensor:
                tensor = tensor.numpy()

            if len(tensor.shape) == 4:
                # Deal with rank=5 situation (BTCHW)
                video_sequence = []
                for frame in tensor:
                    video_sequence.append(cv2.flip(frame, 0))
            else:
                raise Exception("This function don't support the input whose rank is not 4...")
                
        # Transfer back to torch.Tensor if needed
        video_sequence = np.asarray(video_sequence)
        if is_tensor:
            video_sequence = torch.from_numpy(video_sequence)
        return video_sequence