import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory')

    return parser.parse_args()

args=get_args()

LOCAL_MEDIAR_PATH = args.root

sys.path.append(LOCAL_MEDIAR_PATH)

# All imports we will need
import torch
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

from train_tools import *
from train_tools.models import MEDIARFormer
from core.MEDIAR import Predictor, EnsemblePredictor

import os
import imageio


import shutil
import cv2

import tifffile as tif


def process_and_save_image(input_path, output_path):
    # Read the image using imageio
    img = imageio.imread(input_path)

    if len(img.shape) != 3:
      img = img[np.newaxis, :,:]

    #Transpose the dimensions
    img_transformed = np.transpose(img, (1, 2, 0))
    rgb = np.stack([img_transformed]*3, axis=2)
    rgb = rgb[:,:,:,0]

    # Save the processed image to the output path
    imageio.imsave(output_path, rgb)

def process_images_in_directory(input_dir, output_dir):
    # Walk through the directory structure recursively
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            # Check if the file has a valid image extension
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):

                # Construct the full path to the input image
                input_image_path = os.path.join(root, filename)

                # Construct the corresponding output path
                output_image_path = os.path.join(output_dir, os.path.relpath(input_image_path, input_dir))

                # Ensure the output directory structure exists
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                try:
                  # Process and save the image
                  process_and_save_image(input_image_path, output_image_path)
                except:
                  img = imageio.imread(input_image_path)
                  print(img.shape)


def separate_images_in_directory(input_dir, output_dir_images, output_dir_labels):
    os.makedirs(LOCAL_MEDIAR_PATH + '/lab_data_separated', exist_ok = True)
    os.makedirs(output_dir_images, exist_ok = True)
    os.makedirs(output_dir_labels, exist_ok = True)
    # Walk through the directory structure recursively
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            # Check if the file has a valid image extension
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                # Construct the full path to the input image
                input_image_path = os.path.join(root, filename)

                # Determine the output directory based on the image filename
                if 'mask' in filename.lower():
                    output_dir = output_dir_labels
                else:
                    output_dir = output_dir_images


                # Construct the corresponding output path
                output_image_path = os.path.join(output_dir, filename)


                shutil.copy2(input_image_path, output_image_path)

def separate_images_in_directory_test(input_dir, output_dir_images, output_dir_labels):
    os.makedirs(LOCAL_MEDIAR_PATH + '/lab_test_separated', exist_ok = True)
    os.makedirs(output_dir_images, exist_ok = True)
    os.makedirs(output_dir_labels, exist_ok = True)
    # Walk through the directory structure recursively
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            # Check if the file has a valid image extension (you can modify this condition)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                # Construct the full path to the input image
                input_image_path = os.path.join(root, filename)

                # Determine the output directory based on the image filename
                if 'mask' in filename.lower():
                    output_dir = output_dir_labels
                else:
                    output_dir = output_dir_images

                #Even though tif and tiff are the same format, the tifffile library has trouble recognizing .tif as .tiff
                if filename.endswith('_im.tif'):
                    # Construct the new filename
                    filename = filename.replace('_im.tif', '_image.tiff')


                # Construct the corresponding output path
                output_image_path = os.path.join(output_dir, filename)

                shutil.copy2(input_image_path, output_image_path)


#Transform the labs data, this will be used ONLY for training the model and inference
input_directory = LOCAL_MEDIAR_PATH + '/YeaZ_universal_images_and_masks'
output_directory = LOCAL_MEDIAR_PATH + '/lab_data_transformed'

process_images_in_directory(input_directory, output_directory)

#Separate the transformed lab data into just two folders: images and labels
input_directory = LOCAL_MEDIAR_PATH + '/lab_data_transformed'
output_directory_images = LOCAL_MEDIAR_PATH + '/lab_data_separated/images'
output_directory_labels = LOCAL_MEDIAR_PATH + '/lab_data_separated/labels'

separate_images_in_directory(input_directory, output_directory_images, output_directory_labels)

#Transform the lab test dataset and separate. This is only used as input to our predictor
input_test_directory = LOCAL_MEDIAR_PATH + '/Test_images_for_ML_class_2023'
output_test_directory = LOCAL_MEDIAR_PATH +'/lab_test_transformed'

process_images_in_directory(input_test_directory, output_test_directory)

output_directory_images = LOCAL_MEDIAR_PATH + '/lab_test_separated/images'
output_directory_labels = LOCAL_MEDIAR_PATH + '/lab_test_separated/labels'

separate_images_in_directory_test(output_test_directory, output_directory_images, output_directory_labels)


#Create lab data ground truth folder, this is only used when evaluating our predictions (not in the predictor itself), because the evaluation needs lab data without any transformations
# that we did in all other cases
input_directory = LOCAL_MEDIAR_PATH + '/Test_images_for_ML_class_2023'
output_directory_images = LOCAL_MEDIAR_PATH + '/temp'
output_directory_labels = LOCAL_MEDIAR_PATH + '/lab_ground_truth'

# Delete temp folder, unused for prediction, only created when splitting the data
os.system('rm -r temp')

separate_images_in_directory(input_directory, output_directory_images, output_directory_labels)

# Directory path
directory_path = LOCAL_MEDIAR_PATH + '/lab_ground_truth'

#Rename ground truth labels so our evaluator can match the files correctly. NOTE: tif is the same as tiff, just renamed due to errors
# Iterate through files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('_mask.tif'):
        # Construct the new filename
        new_filename = os.path.join(directory_path, filename.replace('_mask.tif', '_image_label.tiff'))

        # Rename the file
        os.rename(os.path.join(directory_path, filename), new_filename)
