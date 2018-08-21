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