import argparse
import json
import sys
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--valid_portion', type=float, default=0.1, help='Valid portion')
    parser.add_argument('--amplified', action='store_true', default=False, help='Amplified')

    return parser.parse_args()

args=get_args()


LOCAL_MEDIAR_PATH = args.root

# Define your Python variables containing the new paths
new_root_path = LOCAL_MEDIAR_PATH

# Load the JSON data
with open(LOCAL_MEDIAR_PATH + "/config/step2_finetuning/finetuning1.json", "r") as file:
    config = json.load(file)

# Update file paths in the JSON data
config["data_setups"]["labeled"]["root"] = new_root_path
config["data_setups"]["labeled"]["amplified"] = args.amplified
config["data_setups"]["labeled"]["batch_size"] = args.batch_size
config["data_setups"]["labeled"]["valid_portion"] = args.valid_portion
config["data_setups"]["public"]["params"]["root"] = new_root_path
config["pred_setups"]["input_path"] = new_root_path + "/lab_test_separated/images"
config["train_setups"]["trainer"]["params"]["num_epochs"] = args.epoch

# Write the updated JSON back to the file
with open(LOCAL_MEDIAR_PATH + "/config/step2_finetuning/finetuning1.json", "w") as file:
    json.dump(config, file, indent=4)
