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
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    return {"images": images, "text": messages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path", type=str, default="./data/llava_ov_clevr.parquet")
    args = parser.parse_args()

    dataset = load_dataset("lmms-lab/LLaVA-OneVision-Data", "CLEVR-Math(MathV360K)", split="train")

    dataset = dataset.map(convert_llava_ov_dataset, remove_columns=dataset.column_names, num_proc=32)
    dataset.to_parquet(args.local_path)
    print(f"Dataset saved to {args.local_path}")
