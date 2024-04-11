import sys
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory')

    return parser.parse_args()

args=get_args()

LOCAL_MEDIAR_PATH = args.root

os.system('mkdir weights')
os.system('mkdir weights/pretrained')
os.system('gdown --id 1NHDaYvsYz3G0OCqzegT-bkNcly2clPGR -O ./weights/pretrained/')
os.system('gdown --id 1v5tYYJDqiwTn_mV0KyX5UEonlViSNx4i -O ./weights/pretrained/')

os.system('mkdir weights/finetuned')
os.system('gdown --id 168MtudjTMLoq9YGTyoD2Rjl_d3Gy6c_L -O ./weights/finetuned/from_phase1.pth')
os.system('gdown --id 1JJ2-QKTCk-G7sp5ddkqcifMxgnyOrXjx -O ./weights/finetuned/from_phase2.pth')

os.system('gdown --id 1Drd7fEaxcL2TPMmqO2D2qJ5yN_KTLsdP')
os.system('unzip "lab dataset.zip"')
os.system('rm "lab dataset.zip"')

os.system('gdown --id 1hAUfb5RXWfYzDcCg481PY7Wv10m7ZYdx')
os.system('unzip "test_images.zip"')

os.system('mkdir lab_ground_truth')
os.system('mkdir temp')
