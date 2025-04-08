from peft.utils import register_peft_method

from .config import KronFTConfig
from .layer import KronFTLayer, KronFTLinear
from .model import KronFTModel

__all__ = ["KronFTConfig", "KronFTLayer", "KronFTLinear", "KronFTModel"]

register_peft_method(name="kronft", model_cls=KronFTModel, config_cls=KronFTConfig, prefix="fourierft_")
