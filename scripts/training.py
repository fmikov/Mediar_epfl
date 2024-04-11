import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--valid_portion', type=float, required=True, help='Valid portion')
    parser.add_argument('--amplified', type=bool, default=False, help='Amplified')

    return parser.parse_args()

args=get_args()

strAmplified = ""
if(args.amplified):
    strAmplified = "--amplified"

os.system(f'python ./scripts/change_configs.py --root={args.root} --epoch={args.epoch} --batch_size={args.batch_size} --valid_portion={args.valid_portion} {strAmplified}')
os.system(f'python ./main.py --config_path="{args.root}/config/step2_finetuning/finetuning1.json" > training_output.txt')