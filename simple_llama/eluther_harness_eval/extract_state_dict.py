import torch
import os
import argparse

from simple_llama.pretraining.config import TrainingConfig
torch.serialization.add_safe_globals({TrainingConfig})

"""
Used to extract out the model's state dict to a standalone .pt file, which is needed by lm-eval
"""


def extract_state_dict(input_path: str):
    """Logic is kind of brittle, but works for most cases"""

    checkpoint = torch.load(input_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # E.g. ./checkpoints/model_50B_2146L_4096MSQ.pth -> model_50B_2146L_4096MSQ.pth
    intermediate = input_path.split("/")[-1]
    # E.g. model_50B_2146L_4096MSQ.pth -> ['model_50B_2146L_4096MSQ', '.pth'] Assumes it only has a single '.', which is fine if user didn't modify anything
    intermediate = intermediate.split(".")

    output_sd_path = os.path.join(save_dir, f"{intermediate[0]}_sd.{intermediate[1]}")
    torch.save(state_dict, output_sd_path)
    print(f"State dict saved to: {output_sd_path}")

    config = checkpoint["config"]
    output_config_path = os.path.join(save_dir, f"{intermediate[0]}_config.{intermediate[1]}")
    torch.save(config, output_config_path)
    print(f"Config saved to: {output_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="path to .pth file, E.g. './checkpoints/model_50B_2146L_4096MSQ.pth'")

    input_path = parser.parse_args().i

    save_dir = "save_dir"
    os.makedirs(save_dir, exist_ok=True)

    extract_state_dict(input_path)

