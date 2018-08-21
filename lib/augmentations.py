import random
import cv2

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

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, a_list, b_list):
        # Crop the video as the same length
        length = min(len(a_list), len(b_list))
        a_list = a_list[:length]
        b_list = b_list[:length]

        # Use the same random seed for whole video
        seed = random.random()

        # Do other augmentation
        aug_a_list = []
        aug_b_list = []
        for i in range(length):
            assert a_list[i].size == b_list[i].size
            for aug_op in self.augmentations:
                aug_a, aug_b = aug_op(a_list[i], b_list[i], seed)
            aug_a_list.append(aug_a)
            aug_b_list.append(aug_b)
        return aug_a_list, aug_b_list

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, a_frame, b_frame, seed = 0.0):
        rotate_degree = seed * 2 * self.degree - self.degree
        return rotate(a_frame, rotate_degree), rotate(b_frame, rotate_degree)

class RandomHorizontallyFlip(object):
    def __call__(self, a_frame, b_frame, seed):
        if seed < 0.5:
            return cv2.flip(a_frame, 0), cv2.flip(b_frame, 0)
        return a_frame, b_frame