import torch
import transformers
from accelerate import Accelerator
from typing import Any
from transformers import AutoModelForImageTextToText, AutoProcessor


def setup():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Transformers version: {transformers.__version__}")


def pick_model_dtype(device_type: str) -> torch.dtype:
    # Keep fp32 on non-CUDA backends for broader model/operator compatibility.
    if device_type == "cuda":
        return torch.float16
    return torch.float32


def build_message(image_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {
                    "type": "text",
                    "text": f"{prompt}",
                },
            ],
        }
    ]


if __name__ == "__main__":
    setup()

    device = Accelerator().device
    dtype = pick_model_dtype(device.type)
    print(f"Selected device: {device} (type={device.type})")
    print(f"Selected dtype: {dtype}")

    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")

    img_path = "data/benchmark/coco_subset/COCO_val2014_000000001171.jpg"
    prompt = f"Answer with only 'yes' or 'no'.\nQuestion: Is there a locomotive in the image?"
    message = build_message(img_path, prompt)

    text = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_len = len(text.input_ids[0])

    with torch.no_grad():
        generated_ids = model.generate(**text, max_new_tokens=200)
    generated_texts = processor.batch_decode(
        generated_ids[:, input_len:], skip_special_tokens=True
    )

    print(generated_texts)
