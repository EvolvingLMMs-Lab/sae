from enum import Enum


class SaeType(str, Enum):
    """
    Enum for different types of SAE (Sparse Adapter Embedding).
    """

    TOPK_SAE = "TOPK_SAE"
