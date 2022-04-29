"""Manage data download and upload"""
# pylint: disable=logging-fstring-interpolation
import os
import logging
from pathlib import Path
import shutil
import zipfile
import wget


logger = logging.getLogger("OSS")


class OSS:
    """Mange data download and upload"""

    @staticmethod
    def extract(filename, target_path=None, remove=False):
        """Extract compressed file.
        Currently only support .zip file.

        Args:
            filename (str or Path): compressed file path.
            target_path (str or Path, optional): target path. Defaults to None.
            remove (bool, optional): whther remove original .zip file. Defaults to False.

        Raises:
            NotImplementedError: currently only .zip supported.
        """
        assert isinstance(filename, (str, Path))
        if isinstance(filename, str):
            filename = Path(filename)
        if isinstance(target_path, str):
            target_path = Path(target_path)

        if filename.suffix == ".zip":
            basename = OSS.basename(str(filename), ".zip")
            if target_path is None:
                extract_path = filename.parent.joinpath(basename)
            else:
                extract_path = target_path.parent.joinpath(basename)

            if extract_path.exists():
                shutil.rmtree(str(extract_path))
            with zipfile.ZipFile(str(filename)) as zip:
                zip.extractall(extract_path)

            if remove:
                # remove origin .zip file if required
                os.remove(str(filename))
        else:
            raise NotImplementedError

    @staticmethod
    def basename(basestr: str, suffix: str):
        """Get a base file name of path.
        E.g. basename of 'test/exp-data.zip' with suffix '.zip' is 'exp-data'

        Args:
            basestr (str): base file path.
            suffix (str): file suffix.

        Returns:
            str: base file name without suffix
        """
        basestr = basestr[basestr.rfind("/") + 1 :]
        if basestr.endswith(suffix):
            basestr = basestr[: len(basestr) - len(suffix)]
            return basestr
        return basestr

    @staticmethod
    def download(url, local_path):
        """DownLoad url data to local path and extract file.

        Args:
            url (str): remote data dir, starts with `http://`.
            local_path (str): local target path.

        Raises:
            NotImplementedError: if file not ends with `.zip`.
            RuntimeError: if download failed.

        Returns:
            str: name of downloaded file.
        """
        last_dir = os.getcwd()
        os.makedirs(local_path, exist_ok=True)
        os.chdir(local_path)
        try:
            filename = wget.download(url)
        except Exception:
            logger.error(f"failed to download {url}")
            os.chdir(last_dir)
            raise RuntimeError("failed to download")
        os.chdir(last_dir)
        return filename

    @staticmethod
    def is_download(url, local_path):
        """Check whther url is downloaded to local path.

        Args:
            url (str): remote date url.
            local_path (str): local path.

        Raises:
            NotImplementedError: only .zip supported.

        Returns:
            bool: True if downloaded.
        """
        if url.endswith(".zip"):
            basename = OSS.basename(url, ".zip")
            file_path = Path(local_path).joinpath(basename + ".zip")
            if file_path.exists():
                return True
            return False
        raise NotImplementedError
