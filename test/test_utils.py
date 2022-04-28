import os
from pathlib import Path
import shutil
from utils import Config
from utils import OSS


class TestUtils:
    def test_config(self):
        result_dict = dict(
            a=dict(
                b="a string", c=dict(c_b="b_string", d=dict(d_b="d string", x=1, y=2))
            )
        )
        py_config = Config.from_file("test/example_config.py")
        yaml_config = Config.from_file("test/example_config.yaml")
        assert str(py_config) == str(result_dict)
        assert str(yaml_config) == str(result_dict)

    def test_oss(self):
        data_url = "http://data-rainy.oss-cn-beijing.aliyuncs.com/data/exp3-data.zip"
        local_data_path = Path("test/test-data/exp3-data.zip")
        extract_path = Path("test/test-data/exp3-data")

        # step1: test download
        if local_data_path.exists():
            os.remove(str(local_data_path))
        assert not local_data_path.exists()
        OSS.download(data_url, "test/test-data")
        assert local_data_path.exists()

        # step2: test is_download
        assert OSS.is_download(data_url, "test/test-data")

        # step3: test extract
        if extract_path.exists():
            shutil.rmtree(str(extract_path))
        OSS.extract(local_data_path)
        assert extract_path.exists()

        # step4: clear
        shutil.rmtree("test/test-data")
