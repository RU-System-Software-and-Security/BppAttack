import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--model_filepath", type=str, default="./checkpoints")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--set_arch", type=str, default=None)
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--save_all", type=bool, default=False)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--scheduler_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--scheduler_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=1000)
    parser.add_argument("--num_workers", type=float, default=6)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--injection_rate", type=float, default=0.2)
    parser.add_argument("--neg_rate", type=float, default=0.2)
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--squeeze_num", type=int, default=8)
    parser.add_argument("--dithering", type=bool, default=False)

    return parser
