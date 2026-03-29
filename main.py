import torch
import transformers
def main():
    print(f"PyTorch version: {torch.__version__} and cuda available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
