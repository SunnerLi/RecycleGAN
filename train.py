from lib.loader import get_loader, InfiniteLoader
from lib.model.recycle_gan import ReCycleGAN
from lib.utils import visualizeSingle
from lib import augmentations as aug
from parse import parse_train_args

from torch.autograd import Variable
from torchvision import transforms
from torch.utils import data
from tqdm import tqdm
import torch
import os

"""
    This script define the training procedure of Re-cycle GAN
"""

def eval(args, model, video_a, video_b):
    """
        Render for the first valid time step and visualize

        Arg:    args        - The argparse object
                model       - The nn.Module represent the Re-cycle GAN model
                video_a     - The video sequence in domain A
                video_b     - The video sequence in domain B
    """
    # BTCHW -> T * BCHW
    true_a_seq = [frame.squeeze(1).to(args.device) for frame in torch.chunk(video_a, video_a.size(1), dim = 1)]
    true_b_seq = [frame.squeeze(1).to(args.device) for frame in torch.chunk(video_b, video_b.size(1), dim = 1)]
        
    # Form the input frame in original domain
    true_a = true_a_seq[args.t]
    true_b = true_b_seq[args.t]

    # Render single image
    model.eval()
    with torch.no_grad():
        images = model(
            true_a = true_a, 
            true_b = true_b, 
            true_a_seq = true_a_seq[:args.t], 
            true_b_seq = true_b_seq[:args.t], 
            warning = False
        )
        visualizeSingle(images)
    model.train()

def train(args):
    """
        This function define the training procedure

        Arg:    args    - The argparse argument
    """
    # Create the data loader
    loader = InfiniteLoader(
        loader = data.DataLoader(
            dataset = get_loader(args.dataset)(
                root = [args.A, args.B], 
                transform = aug.Compose([
                    # aug.RandomRotate(10),
                    aug.RandomHorizontallyFlip(),
                    aug.ToTensor(),
                    aug.ToFloat(),
                    aug.Transpose(aug.BHWC2BCHW),
                    aug.Resize(size_tuple = (args.H, args.W)),
                    aug.Normalize()
                ]), 
                T = args.T, 
                t = args.t,
                use_cv = True,
            ), batch_size = 1, shuffle = True
        ), max_iter = args.n_iter
    )

    # Create the model and initialize
    model = ReCycleGAN(A_channel = args.A_channel, B_channel = args.B_channel, T = args.T, r = args.r, t = args.t, device = args.device)
    if os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume))
    model.train()

    # Work!
    bar = tqdm(loader)
    for i, (video_a, video_b) in enumerate(bar):
        
        # Update parameters
        model.setInput(video_a, video_b)
        model.backward()
        bar.set_description("G: " + str(model.loss_G) + " D: " + str(model.loss_D) + " P: " + str(model.loss_P))
        bar.refresh()

        # Record render result
        if i % args.record_iter == 0 and i != 0:
            torch.save(model.state_dict(), args.det)
            eval(args, model, video_a, video_b)

if __name__ == '__main__':
    args = parse_train_args()
    train(args)