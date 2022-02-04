# taken from https://github.com/mozilla/bugbug/tree/master/bugbug

import logging
import requests
from urllib.parse import urljoin
import os
import subprocess
import zstandard
import errno

logger = logging.getLogger(__name__)

# utils.py

def download_check_etag(url, path=None):
    r = requests.head(url, allow_redirects=True)

    if path is None:
        path = url.split("/")[-1]

    new_etag = r.headers["ETag"]

    try:
        with open(f"{path}.etag", "r") as f:
            old_etag = f.read()
    except IOError:
        old_etag = None

    if old_etag == new_etag:
        return False

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1048576):
            f.write(chunk)

    with open(f"{path}.etag", "w") as f:
        f.write(new_etag)

    return True
    
def zstd_decompress(path: str) -> None:
    dctx = zstandard.ZstdDecompressor()
    with open(f"{path}.zst", "rb") as input_f:
        with open(path, "wb") as output_f:
            dctx.copy_stream(input_f, output_f)

def extract_tar_zst(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    subprocess.run(["tar", "-I", "zstd", "-xf", path], check=True)

def extract_file(path: str) -> None:
    inner_path, _ = os.path.splitext(path)

    if str(path).endswith(".tar.zst"):
        extract_tar_zst(path)
    elif str(path).endswith(".zst"):
        zstd_decompress(inner_path)
    else:
        assert False, f"Unexpected compression type for {path}"

# db.py

DATABASES = {}

def register(path, url, version, support_files=[]):
    DATABASES[path] = {"url": url, "version": version, "support_files": support_files}

    # Create DB parent directory.
    os.makedirs(os.path.abspath(os.path.dirname(path)), exist_ok=True)

    if not os.path.exists(f"{path}.version"):
        with open(f"{path}.version", "w") as f:
            f.write(str(version))

def is_different_schema(path):
    url = urljoin(DATABASES[path]["url"], f"{os.path.basename(path)}.version")
    r = requests.get(url)

    if not r.ok:
        logger.info(f"Version file is not yet available to download for {path}")
        return True

    prev_version = int(r.text)

    return DATABASES[path]["version"] != prev_version

def download_support_file(path, file_name, extract=True):
    # If a DB with the current schema is not available yet, we can't download.
    if is_different_schema(path):
        return False

    try:
        url = urljoin(DATABASES[path]["url"], file_name)
        path = os.path.join(os.path.dirname(path), file_name)

        logger.info(f"Downloading {url} to {path}")
        updated = download_check_etag(url, path)

        if extract and updated and path.endswith(".zst"):
            extract_file(path)
            os.remove(path)

        return True
    except requests.exceptions.HTTPError:
        logger.info(
            f"{file_name} is not yet available to download for {path}", exc_info=True
        )
        return False

# Download and extract databases.
def download(path, support_files_too=False, extract=True):
    # If a DB with the current schema is not available yet, we can't download.
    if is_different_schema(path):
        return False

    zst_path = f"{path}.zst"

    url = DATABASES[path]["url"]
    try:
        logger.info(f"Downloading {url} to {zst_path}")
        updated = download_check_etag(url, zst_path)

        if extract and updated:
            extract_file(zst_path)
            os.remove(zst_path)

        successful = True
        if support_files_too:
            for support_file in DATABASES[path]["support_files"]:
                successful |= download_support_file(path, support_file, extract)

        return successful
    except requests.exceptions.HTTPError:
        logger.info(f"{url} is not yet available to download", exc_info=True)
        return False



if __name__ == "__main__":
    COMMITS_DB = "data/bugbug/commits.json"
    COMMIT_EXPERIENCES_DB = "commit_experiences.lmdb.tar.zst"
    register(
        COMMITS_DB,
        "https://community-tc.services.mozilla.com/api/index/v1/task/project.bugbug.data_commits.latest/artifacts/public/commits.json.zst",
        22,
        [COMMIT_EXPERIENCES_DB],
    )

    BUGS_DB = "data/bugbug/bugs.json"
    register(
        BUGS_DB,
        "https://community-tc.services.mozilla.com/api/index/v1/task/project.bugbug.data_bugs.latest/artifacts/public/bugs.json.zst",
        7,
    )

    # TODO
    # download(COMMITS_DB)
    # download(BUGS_DB)