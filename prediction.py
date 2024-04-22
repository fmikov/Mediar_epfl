import sys
import argparse
import os


import torch

import glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

from train_tools import *
from train_tools.models import MEDIARFormer
from core.MEDIAR import Predictor, EnsemblePredictor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--in_path', type=str, required=True, help='input path')
    parser.add_argument('--gt_path', type=str, required=True, help='ground truth path')
    #TODO add 1 more optional model path for ensemble prediction
    parser.add_argument('--model_path2', type=str, required=False, help='second model path for ensemble prediction')

    parser.add_argument('--tta', action='store_true', default=False, help='use TTA?')

    return parser.parse_args()

args=get_args()




def conduct_prediction(model_path, input_path, gt_path):
  model_name = (model_path.split('/')[-1]).split('.')[0]
  output_path = f"results/{model_name}"
  #!mkdir {output_path}
  os.system(f'mkdir {output_path}')

  weights = torch.load(model_path, map_location="cpu")
  model_args = {
    "classes": 3,
    "decoder_channels": [1024, 512, 256, 128, 64],
    "decoder_pab_channels": 256,
    "encoder_name": 'mit_b5',
    "in_channels": 3
  }
  model = MEDIARFormer(**model_args)
  model.load_state_dict(weights, strict=False)
  predictor = Predictor(model, "cuda:0", input_path, output_path, algo_params={"use_tta": args.tta})
  _ = predictor.conduct_prediction()
  print("Evaluation of model: ", model_name)
  os.system(f'python evaluate.py --pred_path={output_path} --gt_path={gt_path}')

def conduct_ensemble_prediction(model_path1, model_path2, input_path, output_path):
  model_name1 = (model_path1.split('/')[-1]).split('.')[0]

  model_name2 = (model_path2.split('/')[-1]).split('.')[0]

  output_path = f"results/ensemble_{model_name1}_and_{model_name2}"
  os.system(f'mkdir {output_path}')


  weights1 = torch.load(model_path1, map_location="cpu")
  weights2 = torch.load(model_path2, map_location="cpu")
  model_args = {
    "classes": 3,
    "decoder_channels": [1024, 512, 256, 128, 64],
    "decoder_pab_channels": 256,
    "encoder_name": 'mit_b5',
    "in_channels": 3
  }
  model1 = MEDIARFormer(**model_args)
  model1.load_state_dict(weights1, strict=False)
  model2 = MEDIARFormer(**model_args)
  model2.load_state_dict(weights2, strict=False)

  predictor3 = EnsemblePredictor(model1, model2, "cuda:0", input_path, output_path, algo_params={"use_tta": args.tta})
  _ = predictor3.conduct_prediction()

  print("Evaluation of ensemble model: ", model_name1, " and ", model_name2)
  os.system(f'python evaluate.py --pred_path={output_path} --gt_path={gt_path}')
# /home/mikoviny/MEDIAR/weights/yeast_trained/100_epochs.pth
# Example usage: python prediction.py --model_path='/home/mikoviny/MEDIAR/weights/yeast_trained/100_epochs.pth' --in_path='/home/mikoviny/MEDIAR/lab_test_separated/images' --gt_path='/home/mikoviny/MEDIAR/lab_ground_truth'
conduct_prediction(args.model_path, args.in_path, args.gt_path)