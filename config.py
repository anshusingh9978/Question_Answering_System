from typing import Dict, Union

from attrdict import AttrDict  # type: ignore

config: Dict[str, Union[int, str]] = {"framework": "pt", "NUM_SENT": 10}
config = AttrDict(config)

config = {
    "framework": "pt",        # or "tf" for TensorFlow
    "NUM_SENT": 5,            # number of wiki sentences to extract
    "model": "distilbert-base-uncased-distilled-squad"
}
