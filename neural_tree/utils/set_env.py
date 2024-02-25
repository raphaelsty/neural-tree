import os

__all__ = ["set_env"]


def set_env() -> None:
    """Set environment variables."""
    os.environ["HF_HOME"] = os.environ["HOME"] + "/.cache/huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
