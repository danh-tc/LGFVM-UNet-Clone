"""
Upload a LGFVM-UNet result folder to Hugging Face Hub.

Usage:
    python upload_to_hf.py \
        --repo_id YOUR_USERNAME/lgfvm-unet-synapse \
        --folder   results/LGF-VMUNet_synapse_Tuesday_23_June_2026_12h_39m_55s \
        --token    hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        required=True,
        help="HuggingFace repo id, e.g. your-username/lgfvm-unet-synapse",
    )
    parser.add_argument(
        "--folder",
        default="results/LGF-VMUNet_synapse_Tuesday_23_June_2026_12h_39m_55s",
        help="Local result folder to upload.",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="HuggingFace Write token from huggingface.co/settings/tokens",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    api = HfApi(token=args.token)

    # Create repo
    print(f"Creating repository: {args.repo_id}")
    create_repo(
        repo_id=args.repo_id,
        token=args.token,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Upload entire folder
    folder_name = os.path.basename(args.folder)
    print(f"Uploading folder '{args.folder}' → repo path '{folder_name}/' ...")
    api.upload_folder(
        folder_path=args.folder,
        path_in_repo=folder_name,
        repo_id=args.repo_id,
        token=args.token,
        commit_message=f"Upload {folder_name}",
    )

    print(f"\nDone! View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
