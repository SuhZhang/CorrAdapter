import os
import json
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors
import torch
from diffusers.utils import _get_model_file, logging
from safetensors import safe_open

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# >>> [CorrAdapter ADDED BEGIN: support sharded training checkpoints]
# Original MVAdapter loaded one adapter weight file. CorrAdapter fine-tuned
# checkpoints may be saved as Hugging Face-style shard folders containing
# pytorch_model.bin.index.json plus pytorch_model-*.bin shards.
def _load_sharded_state_dict(index_file: Union[str, os.PathLike]) -> Dict[str, torch.Tensor]:
    index_path = Path(index_file)
    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"No weight_map found in sharded checkpoint index: {index_path}")

    state_dict = {}
    shard_cache = {}
    for key, shard_name in weight_map.items():
        shard_path = index_path.parent / shard_name
        if shard_name not in shard_cache:
            shard = torch.load(shard_path, map_location="cpu")
            if isinstance(shard, dict) and "state_dict" in shard and isinstance(shard["state_dict"], dict):
                shard = shard["state_dict"]
            shard_cache[shard_name] = shard
        shard_state = shard_cache[shard_name]
        if key in shard_state:
            state_dict[key] = shard_state[key]

    return state_dict
# <<< [CorrAdapter ADDED END: support sharded training checkpoints]


class CustomAdapterMixin:
    def init_custom_adapter(self, *args, **kwargs):
        self._init_custom_adapter(*args, **kwargs)

    def _init_custom_adapter(self, *args, **kwargs):
        raise NotImplementedError

    def load_custom_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                user_agent=user_agent,
            )
            # >>> [CorrAdapter MODIFIED BEGIN: load sharded adapter checkpoints]
            # Original:
            #     if weight_name.endswith(".safetensors"):
            #         ...
            if weight_name.endswith(".index.json"):
                state_dict = _load_sharded_state_dict(model_file)
            elif weight_name.endswith(".safetensors"):
                state_dict = {}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
            # <<< [CorrAdapter MODIFIED END: load sharded adapter checkpoints]
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        self._load_custom_adapter(state_dict)

    def _load_custom_adapter(self, state_dict):
        raise NotImplementedError

    def save_custom_adapter(
        self,
        save_directory: Union[str, os.PathLike],
        weight_name: str,
        safe_serialization: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        if safe_serialization:

            def save_function(weights, filename):
                return safetensors.torch.save_file(
                    weights, filename, metadata={"format": "pt"}
                )

        else:
            save_function = torch.save

        # Save the model
        state_dict = self._save_custom_adapter(**kwargs)
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(
            f"Custom adapter weights saved in {os.path.join(save_directory, weight_name)}"
        )

    def _save_custom_adapter(self):
        raise NotImplementedError
