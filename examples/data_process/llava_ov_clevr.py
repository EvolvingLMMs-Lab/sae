import argparse

from datasets import load_dataset


def convert_llava_ov_dataset(example):
    """Convert LLaVA-OneVision dataset format to the expected format."""
    images = [example.pop("image")]
    conversations = example.pop("conversations")
    messages = []
    for conversation in conversations:
        role = conversation["from"]
        content = conversation["value"]
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        if "<image>" in content:
            content = content.replace("<image>", "")
            messages.append(
                {
                    "role": role,
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": content},
                    ],
                }
            )
        else:
            messages.append(
                {"role": role, "content": [{"type": "text", "text": content}]}
            )

    return {"images": images, "text": messages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_path", type=str, help="HuggingFace repository path to push to"
    )
    parser.add_argument(
        "--local_path", type=str, default="./data/llava_ov_clevr.parquet"
    )
    parser.add_argument(
        "--target_hf_repo_path", type=str, help="HuggingFace repository path to push to"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push dataset to HuggingFace Hub"
    )
    parser.add_argument("--subset", type=str, help="Subset of the dataset to process")
    parser.add_argument("--split", type=str, help="Split of the dataset to process")
    args = parser.parse_args()

    dataset = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data", name=args.subset, split=args.split
    )

    original_columns = dataset.column_names
    dataset = dataset.map(
        convert_llava_ov_dataset, remove_columns=original_columns, num_proc=32
    )

    if args.push_to_hub:
        if not args.target_hf_repo_path:
            raise ValueError(
                "--target_hf_repo_path is required when --push_to_hub is specified"
            )
        dataset.push_to_hub(args.target_hf_repo_path, config_name=args.subset)
        print(f"Dataset pushed to HuggingFace Hub: {args.target_hf_repo_path}")
    else:
        dataset.to_parquet(args.local_path)
        print(f"Dataset saved to {args.local_path}")
