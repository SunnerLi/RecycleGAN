from lib.loader.video_loader2 import *
from collections import Iterator

class InfiniteLoader(Iterator):
    def __init__(self, loader, max_iter = 1):
        """
            Constructor of the infinite loader
            The iteration object will create again while getting end
            Arg:    loader      - The torch.data.DataLoader object
                    max_iter    - The maximun iteration
        """
        super().__init__()
        self.loader = loader
        self.loader_iter = iter(self.loader)
        self.iter = 0
        self.max_iter = max_iter

    def __next__(self):
        try:
            data, target = next(self.loader_iter)
        except:
            self.loader_iter = iter(self.loader)
            data, target = next(self.loader_iter)
        self.iter += 1
        if self.iter <= self.max_iter:
            return data, target
        else:
            print()
            raise StopIteration()

def get_loader(name):
    """
        Return the corresponding data loader object
        Arg:    name    - The name of dataset you want to use
        Ret:    The target data loader object
    """
    try:
        loader = {
            'video': VideoDataset,
        }[name]
    except:
        INFO("The name of dataset [ %s ] is not support, you can choose one of them: [video]" % name)
        exit()
    return loader