"""Config"""
import sys
import inspect
import importlib.util as imp_util
from pathlib import Path
import yaml


class Config(dict):
    """A wraped dict class, support accessw value through attributes.
    currently support load config from yaml file and python file.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        if config is not None:
            for key, value in config.items():
                self.add_item(key, value)

    def __getattr__(self, key):
        try:
            return super().__getitem__(key)
        except Exception:
            raise KeyError(f"{key} not in config")

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def add_item(self, key, value):
        """Add a key value pair to current dict.

        Args:
            key (Str): key to add.
            value (Any): value to add.
        """
        if isinstance(value, dict):
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key, value)

    def update(self, config):
        """Update current dict by config dict.

        Args:
            config (Config or Dict): config used to update.
        """
        assert isinstance(config, (dict, Config))
        for k, v in config.items():
            self.add_item(k, v)

    @staticmethod
    def from_yaml_file(filename):
        """Read a yaml file and construct Config object

        Args:
            filename (Str or Path): yaml file name.

        Raises:
            ValueError: if filename is not str or Path.

        Returns:
            Config: Config object.
        """
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()
        else:
            raise ValueError(
                f"filename type should be str or Path, but got {type(filename)}"
            )

        assert filepath.suffix == ".yaml", "only yaml file supported"

        with open(filepath, mode="r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        def handle(config):
            for k, v in config.items():
                if isinstance(v, dict):
                    handle(config[k])
                elif isinstance(v, str) and v == "None":
                    config[k] = None
            return config

        config = handle(config)
        return Config(config)

    @staticmethod
    def from_py_file(filename):
        """Read a python file and construct Config object.

        Args:
            filename (Str or Path): python file name.

        Raises:
            ValueError: filename is not str or Path

        Returns:
            Config: config object.
        """
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()
        else:
            raise ValueError(
                f"filename type should be str or Path, but got {type(filename)}"
            )

        assert filepath.suffix == ".py", "only python file supported"

        module_name = str(filepath.stem)
        spec = imp_util.spec_from_file_location(module_name, str(filepath))
        module = imp_util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        config = Config()
        for k, v in module.__dict__.items():
            if k.startswith("__") or inspect.ismodule(v) or inspect.isclass(v):
                continue
            config.add_item(k, v)

        del sys.modules[module_name]
        return config

    @staticmethod
    def from_file(filename: str):
        """Construct `Config` object according to given file.
        currently support:
            - .py file
            - .yaml file

        Args:
            filename (Str): config filename to construct.

        Raises:
            ValueError: if file is not .py or .yaml file.

        Returns:
            Config: config object.
        """
        assert isinstance(filename, str)
        filepath = Path(filename).absolute()

        assert filepath.exists(), f"{filename} not exist"

        if filepath.suffix == ".yaml":
            return Config.from_yaml_file(filepath)
        if filepath.suffix == ".py":
            return Config.from_py_file(filepath)

        raise ValueError(
            f"only .py and .yaml config file supported, but get {filepath.suffix} file"
        )
