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

# class TensorOP(object):
#     def work(self):
#         raise NotImplementedError("")

#     def __call__(self):
        
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

        if len(tensor.shape) == 5:
            # Deal with rank=5 situation (BTCHW)
            video_sequence = []
            for batch in tensor:
                video_sequence_batch = []
                for frame in batch:
                    video_sequence_batch.append(rotate(frame, rotate_degree))
                video_sequence.append(np.asarray(video_sequence_batch))
        elif len(tensor.shape) == 6:
            # Deal with rank=6 situation (BTtCHW)
            video_sequence = []
            for batch in tensor:
                video_sequence_batch = []
                for _tuple in batch:
                    tuple_list = []
                    for frame in _tuple:
                        tuple_list.append(rotate(frame, rotate_degree))
                    video_sequence_batch.append(np.asarray(tuple_list))
                video_sequence.append(np.asarray(video_sequence_batch))
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

            if len(tensor.shape) == 5:
                # Deal with rank=5 situation (BTCHW)
                video_sequence = []
                for batch in tensor:
                    video_sequence_batch = []
                    for frame in batch:
                        video_sequence_batch.append(cv2.flip(frame, 0))
                    video_sequence.append(np.asarray(video_sequence_batch))
            elif len(tensor.shape) == 6:
                # Deal with rank=6 situation (BTtCHW)
                video_sequence = []
                for batch in tensor:
                    video_sequence_batch = []
                    for _tuple in batch:
                        tuple_list = []
                        for frame in _tuple:
                            tuple_list.append(cv2.flip(frame, 0))
                        video_sequence_batch.append(np.asarray(tuple_list))
                    video_sequence.append(np.asarray(video_sequence_batch))
            else:
                raise Exception("This function don't support the input whose rank is neither 5 nor 6...")
                
        # Transfer back to torch.Tensor if needed
        video_sequence = np.asarray(video_sequence)
        if is_tensor:
            video_sequence = torch.from_numpy(video_sequence)
        return video_sequence

class ToTensor():
    def __call__(self, tensor):
        return torch.from_numpy(tensor)

class Transpose():
    def __init__(self, transfer_method = BHWC2BCHW):
        self.transfer_method = transfer_method

    def __call__(self, tensor):
        if self.transfer_method:
            return tensor.transpose(-1, -2).transpose(-2, -3)
        else:
            return tensor.transpose(-2, -3).transpose(-1, -2)