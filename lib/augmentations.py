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
        https://www.jianshu.com/p/b5c29aeaedc7
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# ===========================================================
# TODO: Add transpose augmentation
# ===========================================================

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
        
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, tensor):
        """
            Do the rotation with random angle

            Arg:    tensor  - The np.ndarray or torch.Tensor obj, but the rank can be 5 or 6
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
        elif len(tensor.shape) == 5:
            # Deal with rank=6 situation (BTtCHW)
            video_sequence = []
            for _tuple in tensor:
                tuple_list = []
                for frame in _tuple:
                    tuple_list.append(rotate(frame, rotate_degree))
                video_sequence.append(np.asarray(tuple_list))
        else:
            raise Exception("This function don't support the input whose rank is neither 5 nor 6...")

        # Transfer back to torch.Tensor if needed
        video_sequence = np.asarray(video_sequence)
        if is_tensor:
            video_sequence = torch.from_numpy(video_sequence)
        return video_sequence

class RandomHorizontallyFlip(object):
    def __call__(self, tensor):
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
            elif len(tensor.shape) == 5:
                # Deal with rank=6 situation (BTtCHW)
                video_sequence = []
                for _tuple in tensor:
                    tuple_list = []
                    for frame in _tuple:
                        tuple_list.append(cv2.flip(frame, 0))
                    video_sequence.append(np.asarray(tuple_list))
            else:
                raise Exception("This function don't support the input whose rank is neither 4 nor 5...")
                
        # Transfer back to torch.Tensor if needed
        video_sequence = np.asarray(video_sequence)
        if is_tensor:
            video_sequence = torch.from_numpy(video_sequence)
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

    def __call__(self, tensor_hd):
        if len(tensor_hd.size()) == 4:
            return F.interpolate(tensor_hd, size = [self.size_tuple[0], self.size_tuple[1]])
        elif len(tensor_hd.size()) == 5:
            tensor_list = [tensor.squeeze(1) for tensor in torch.chunk(tensor_hd, tensor_hd.size(1), dim = 1)]  # T * tCHW
            result_tensor = []
            for tensor in tensor_list:
                result_tensor.append(F.interpolate(tensor, size = [self.size_tuple[0], self.size_tuple[1]]))
            result_tensor = torch.stack(result_tensor, dim = 1)
            return result_tensor

# TODO: Add normalize augmentation