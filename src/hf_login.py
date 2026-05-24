"""Authenticate with Hugging Face using the HF_TOKEN environment variable."""

import os
import sys

from huggingface_hub import login


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable is not set.")
        print("Set it with: export HF_TOKEN='your_token_here'")
        sys.exit(1)

    login(token=token)
    print("Successfully logged in to Hugging Face.")


if __name__ == "__main__":
    main()
