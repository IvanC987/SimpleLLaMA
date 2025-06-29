import os
import torch
import re


def extract_token_count(filename):
    """Extract the token count from filename: model_500M_1234_2048.pth"""
    match = re.search(r"model_(\d+)([MB])_", filename)
    if not match:
        return -1  # Put unmatchable files at the end
    num, scale = match.groups()
    num = int(num)
    return num * (1_000_000 if scale == 'M' else 1_000_000_000)


def clean_checkpoints(dir_path):
    ckpt_files = [f for f in os.listdir(dir_path) if f.endswith(".pth")]
    if not ckpt_files:
        print("No .pth files found.")
        return

    # Sort files by token count extracted from filename
    ckpt_files.sort(key=extract_token_count)

    # Keep the one with the most tokens (last in sorted list)
    keep_full = ckpt_files[-1]
    print(f"Keeping full checkpoint: {keep_full}")
    input("Enter to confirm: ")

    for filename in ckpt_files[:-1]:
        full_path = os.path.join(dir_path, filename)
        try:
            checkpoint = torch.load(full_path, map_location="cpu")
            model_only = {"model_state_dict": checkpoint["model_state_dict"]}
            torch.save(model_only, full_path)
            print(f"Trimmed: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")


# Example usage
dir_path = "222.3M-LLaMA"
clean_checkpoints(dir_path)
