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
    # TODO: Finish visualize method
    
    images_np = {}
    for key in images.keys():
        image = images[key][0]
        image = image.transpose(0, 1).transpose(1, 2)
        image
