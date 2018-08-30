from lib.loader import get_loader, InfiniteLoader
from lib.model.recycle_gan2 import ReCycleGAN
from lib.utils import visualizeSingle
import lib.augmentations as aug

from parse import parse_train_args

from torch.autograd import Variable
from torchvision import transforms
from torch.utils import data
from tqdm import tqdm
import torch
import os

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
                    aug.ToFloat(),
                    aug.Transpose(aug.BHWC2BCHW),
                    aug.Resize(size_tuple = (224, 224)),
                ]), 
                T = args.T, 
                # t = 3,
                use_cv = False,
            ), batch_size = 1, shuffle = True
        ), max_iter = args.n_iter
    )

    # Create the model and initialize
    model = ReCycleGAN(A_channel = args.A_channel, B_channel = args.B_channel, T = args.T, r = args.r, t = 3).to(args.device)
    if os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume))
    model.train()

    # Work!
    bar = tqdm(loader)
    for i, (video_a, video_b) in enumerate(bar):
        
        # Update parameters
        model.setInput(video_a, video_b, device = args.device)
        model.backward()
        bar.set_description("G: " + str(model.loss_G.item()) + " D: " + str(model.loss_D.item()))
        bar.refresh()

        # Record render result
        # if i % args.record_iter == 0 and i != 0:
        if i % args.record_iter == 0:
            # torch.save(model.state_dict(), args.det)

            # TODO: Add render code
            model.eval()
            with torch.no_grad():
                images = model.validate()
                for key in images.keys():
                    print(key, images[key].size())
                visualizeSingle(images)
            model.train()

if __name__ == '__main__':
    args = parse_train_args()
    train(args)
