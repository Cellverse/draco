import os.path
from typing import Any

from omegaconf import DictConfig, OmegaConf

BASE_KEY = "_BASE_"
ROOT_KEY = "cfg"

class CfgNode(OmegaConf):
    """
    A wrapper around OmegaConf that provides some additional functionality.
    """

    @staticmethod
    def load_yaml_with_base(filename: str) -> DictConfig:
        cfg = OmegaConf.load(filename)

        def _load_with_base(base_cfg_file: str) -> dict[str, Any]:
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            return CfgNode.load_yaml_with_base(base_cfg_file)

        if BASE_KEY in cfg:
            if isinstance(cfg[BASE_KEY], list):
                base_cfg: dict[str, Any] = {}
                base_cfg_files = cfg[BASE_KEY]
                for base_cfg_file in base_cfg_files:
                    base_cfg = CfgNode.merge(base_cfg, _load_with_base(base_cfg_file))
            else:
                base_cfg_file = cfg[BASE_KEY]
                base_cfg = _load_with_base(base_cfg_file)
            del cfg[BASE_KEY]

            base_cfg = CfgNode.merge(base_cfg, cfg)
            return base_cfg

        if ROOT_KEY in cfg:
            return cfg[ROOT_KEY]
        return cfg

    @staticmethod
    def merge_with_dotlist(cfg: DictConfig, dotlist: list[str]) -> None:
        if len(dotlist) == 0:
            return

        new_dotlist = []
        for key, value in zip(dotlist[::2], dotlist[1::2]):
            new_dotlist.append(f"{key}={value}")
        cfg.merge_with_dotlist(new_dotlist)
