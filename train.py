from lib.loader import get_loader, InfiniteLoader
from lib.model.recycle_gan import ReCycleGAN
import lib.augmentations as aug

from parse import parse_train_args

from torch.autograd import Variable
from torchvision import transforms
from torch.utils import data
from tqdm import tqdm

def train(args):
    # Create the data loader
    loader = InfiniteLoader(
        loader = data.DataLoader(
            dataset = get_loader(args.dataset)(
                root = [args.A, args.B], 
                transform = aug.Compose([
                    aug.RandomRotate(10),
                    aug.RandomHorizontallyFlip(),
                    aug.ToTensor(),
                    aug.Transpose(aug.BHWC2BCHW)
                ]), 
                T = 10, 
                t = 3,
                use_cv = False,
            ), batch_size = 1, shuffle = True
        ), max_iter = args.n_iter
    )

    # Create the model
    model = ReCycleGAN(A_channel = args.A_channel, B_channel = args.B_channel, r = args.r, t = 3).to(args.device)    
    for video_a, video_b in loader:
        model.setInput(video_a, video_b, device = args.device)
        model.backward()

if __name__ == '__main__':
    args = parse_train_args()
    train(args)
