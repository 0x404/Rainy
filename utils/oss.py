"""Manage data download and upload"""
# pylint: disable=logging-fstring-interpolation
import os
import logging
import zipfile
import wget


logger = logging.getLogger("OSS")


def filename_from_url(url):
    """Get filename from a given url"""
    if url.endswith(".zip"):
        start_pos = url.rindex("/") + 1
        file_name = url[start_pos : len(url) - 4]
    else:
        raise NotImplementedError("file format not supported")
    return file_name


def is_downloaded(url, local_path):
    """Check is url data been downloaded"""
    file_name = filename_from_url(url)
    file_path = os.path.join(local_path, file_name)
    if os.path.exists(file_path):
        return True
    return False


def download(url, local_path):
    """DownLoad url data to local path.

    Args:
        url (str): remote data dir, starts with `http://`.
        local_path (str): local target path.

    Raises:
        NotImplementedError: if file not ends with `.zip`.
        RuntimeError: if download failed.

    Returns:
        str: local file dir.
    """
    current_dir = os.getcwd()
    os.makedirs(local_path, exist_ok=True)
    os.chdir(local_path)
    try:
        file_name = wget.download(url)
        if file_name.endswith(".zip"):
            logger.info("unzip file ...")
            file_path = os.path.join(os.getcwd(), file_name[: len(file_name) - 4])
            with zipfile.ZipFile(file_name) as zip_ref:
                zip_ref.extractall(file_path)
            os.remove(file_name)
            logger.info(f"removed {os.path.join(os.getcwd(), file_name)}")
        else:
            logger.error(f"{url} file format not supported!")
            raise NotImplementedError("file format not supported")
    except Exception:
        logger.error(f"failed to download {url}")
        os.chdir(current_dir)
        raise RuntimeError("failed to download")
    os.chdir(current_dir)
    logger.info(f"successfull download {url} to {file_path}")
    return file_path
