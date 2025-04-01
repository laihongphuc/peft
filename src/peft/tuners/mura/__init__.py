from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import register_peft_method

from .config import MuraConfig
from .layer import Linear, MuraLayer
from .model import MuraModel

__all__ = ["Linear", "MuraConfig", "MuraLayer", "MuraModel"]

register_peft_method(name="mura", config_cls=MuraConfig, model_cls=MuraModel, prefix="lora_")

def __getattr__(name):
    if (name == "Linear8bitLt") and is_bnb_available():
        from .bnb import Linear8bitLt

        return Linear8bitLt

    if (name == "Linear4bit") and is_bnb_4bit_available():
        from .bnb import Linear4bit

        return Linear4bit

    raise AttributeError(f"module {__name__} has no attribute {name}")