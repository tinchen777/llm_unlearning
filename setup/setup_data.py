import argparse
import os
import subprocess
from huggingface_hub import snapshot_download


def download_eval_data():
    snapshot_download(
        repo_id="open-unlearning/eval",
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="saves/eval",
    )


def download_idk_data():
    snapshot_download(
        repo_id="open-unlearning/idk",
        allow_patterns="*.jsonl",
        repo_type="dataset",
        local_dir="data",
    )


def download_wmdp():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["wget", url, "-O", zip_path], check=True)
    subprocess.run(["unzip", "-P", "wmdpcorpora", zip_path, "-d", dest_dir], check=True)


def download_lunar_reference():
    # LUNAR reference prompts D_ref (instruction-only): harmful (refused) / unverified (fictitious)
    base_url = "https://raw.githubusercontent.com/facebookresearch/LUNAR/main/dataset/splits"
    dest_dir = "data/lunar"
    os.makedirs(dest_dir, exist_ok=True)
    for name in ("harmful.json", "unverified.json"):
        subprocess.run(["wget", f"{base_url}/{name}", "-O", os.path.join(dest_dir, name)], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download and setup evaluation data.")
    parser.add_argument(
        "--eval_logs",
        action="store_true",
        help="Downloads TOFU, MUSE  - retain and finetuned models eval logs and saves them in saves/eval",
    )
    parser.add_argument(
        "--idk",
        action="store_true",
        help="Download idk dataset from HF hub and stores it data/idk.jsonl",
    )
    parser.add_argument(
        "--wmdp",
        action="store_true",
        help="Download and unzip WMDP dataset into data/wmdp",
    )
    parser.add_argument(
        "--lunar",
        action="store_true",
        help="Download LUNAR reference prompts (harmful/unverified) into data/lunar",
    )

    args = parser.parse_args()

    if args.eval_logs:
        download_eval_data()
    if args.idk:
        download_idk_data()
    if args.wmdp:
        download_wmdp()
    if args.lunar:
        download_lunar_reference()


if __name__ == "__main__":
    main()
