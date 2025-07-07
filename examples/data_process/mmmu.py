from datasets import Dataset, load_dataset

options = ["A", "B", "C", "D", "E", "F", "G"]

if __name__ == "__main__":
    dataset = load_dataset("lmms-lab/MMMU", split="validation")

    def gen():
        for da in dataset:
            question = da["question"]
            choices = da["options"]
            answer = da["answer"]

            image_list = [
                da["image_1"],
                da["image_2"],
                da["image_3"],
                da["image_4"],
                da["image_5"],
                da["image_6"],
                da["image_7"],
            ]
            image_list = [image for image in image_list if image is not None]

            question += "\n"
            for option, choice in zip(choices, options):
                question += f"{option}. {choice}\n"

            image_content = [{"type": "image"}] * len(image_list)

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}, *image_content],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]

            yield {"images": image_list, "text": messages}

    dataset = Dataset.from_generator(gen)
    dataset.to_parquet("./data/mmmu.parquet")
