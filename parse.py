from lib.utils import INFO, getParentFolder
import argparse
import torch
import os

def __presentParameters(args_dict):
    """
        Print the parameters setting line by line

        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    INFO("========== Parameters ==========")
    for key in sorted(args_dict.keys()):
        INFO("{:>15} : {}".format(key, args_dict[key]))
    INFO("===============================")

def parse_train_args():
    """
        Parse the argument for training procedure
        ------------------------------------------------------
                        [Argument explain]
        
        <path>
            --A                 : The path of video folder in domain A
            --B                 : The path of video folder in domain B
            --resume            : The path of pre-trained model
            --det               : The path of destination model you want to store into
            
        <model>
            --A_channel         : The input channel of domain A
            --B_channel         : The input channel of domain B
            --H                 : The height of image
                                  Default is 240
            --W                 : The width of image
                                  Default is 320
            --r                 : The ratio of channel you want to reduce
                                  Default is 4
            --batch_size        : The batch size in single batch
                                  Default is 1 (due to memory limited)
            --n_iter            : Total iteration
                                  Default is 1 (30000 is recommand)
            --record_iter       : The period to record the render image and model parameters
                                  Default is 1 (200 is recommand)

        <temporal>
            --t                 : The length of tuple in a single sequence
                                  In the Re-cycle GAN paper
                                  Default is 2
            --T                 : The maximun time index we want to consider
                                  The memory burden will raise if set as -1 (without limit)
                                  Default is 30 (as same as vid2vid)

        <dataset>
            --dataset           : Candicate: [video]
                                  The name of dataset, you can choose one of them from the candicate
                                  The default is video            

        ------------------------------------------------------
        Ret:    The argparse object
    """
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--A'              , type = str, required = True)
    parser.add_argument('--B'              , type = str, required = True)
    parser.add_argument('--resume'         , type = str, default = 'result.pkl')
    parser.add_argument('--det'            , type = str, default = 'result.pkl')
    # model
    parser.add_argument('--A_channel'      , type = int, default = 3)
    parser.add_argument('--B_channel'      , type = int, default = 3)
    parser.add_argument('--H'              , type = int, default = 240)
    parser.add_argument('--W'              , type = int, default = 320)
    parser.add_argument('--r'              , type = int, default = 4)
    parser.add_argument('--batch_size'     , type = int, default = 1)
    parser.add_argument('--n_iter'         , type = int, default = 1)
    parser.add_argument('--record_iter'    , type = int, default = 1)
    # temporal
    parser.add_argument('--t'              , type = int, default = 2)
    parser.add_argument('--T'              , type = int, default = 30)
    # dataset
    parser.add_argument('--dataset'        , type = str, default = 'video')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.isdir(args.det):
        raise Exception("The <det> parameter only accept the .pkl path")
    else:
        parent_path = getParentFolder(args.det)
        try:
            if not os.path.exists(parent_path):
                os.mkdir(parent_path)
        except:
            raise Exception("The folder {} is not exist! You should create the folder first!".format(parent_path))
    __presentParameters(vars(args))
    return args

def parse_demo_args():
    """
        Parse the argument for demo procedure
        ------------------------------------------------------
                        [Argument explain]
        
        <path>
            --input             : The path of video you want to transfer
            --output            : The path of result video
            --direction         : The candidate is [a2b, b2a]
                                  The direction you want to proceed
            --resume            : The path of pre-trained model

        <model>
            --A_channel         : The input channel of domain A
            --B_channel         : The input channel of domain B
            --H                 : The height of image
                                  Default is 240
            --W                 : The width of image
                                  Default is 320
            --r                 : The ratio of channel you want to reduce
                                  Default is 4

        <temporal>
            --t                 : The length of tuple in a single sequence
                                  In the Re-cycle GAN paper
                                  Default is 2

        <dataset>
            --dataset           : Candicate: [video]
                                  The name of dataset, you can choose one of them from the candicate
                                  The default is video
    """
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--input'          , type = str, required = True)
    parser.add_argument('--output'         , type = str, default = 'det.mp4')
    parser.add_argument('--direction'      , type = str, default = 'a2b')
    parser.add_argument('--resume'         , type = str, default = None)
    # model
    parser.add_argument('--A_channel'      , type = int, default = 3)
    parser.add_argument('--B_channel'      , type = int, default = 3)
    parser.add_argument('--H'              , type = int, default = 240)
    parser.add_argument('--W'              , type = int, default = 320)
    parser.add_argument('--r'              , type = int, default = 4)
    # temporal
    parser.add_argument('--t'              , type = int, default = 2)
    # dataset
    parser.add_argument('--dataset'        , type = str, default = 'video')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.direction != 'a2b' and args.direction != 'b2a':
        raise Exception("The <direction> parameter you assign: {} is invalid, the candidate is 'a2b' or 'b2a'. " % args.direction)
    if os.path.isdir(args.output):
        raise Exception("The <output> parameter only accept the .mp4 path")
    __presentParameters(vars(args))
    return args