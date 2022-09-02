from contextlib import contextmanager
from tqdm import tqdm
from typing import Callable, Optional
import gzip, os, subprocess, tarfile, urllib, zipfile


def _reporthook(pbar: tqdm) -> Callable:
    last_b = [0]
    def update_to(b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            pbar.total = tsize
        pbar.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return update_to


@contextmanager
def checkpoint(filepath: str) -> Optional[bool]:
    try:
        yield os.path.exists(filepath)
    finally:
        pass
    with open(filepath, 'a'):
        os.utime(filepath)


def download(url: str, target_dir: str, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = url.split('/')[-1].split('#')[0].split('?')[0]
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, filename)
    with tqdm(desc=f'Downloading: "{url}" to "{filepath}"', unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as pbar:
        urllib.request.urlretrieve(url, filepath, reporthook=_reporthook(pbar))
    return filepath


def unpack(filepath: str, target_dir: str) -> None:
    target_dir = os.path.abspath(target_dir)
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r') as tarf:
            for name in tarf.getnames():
                abs_path = os.path.abspath(os.path.join(target_dir, name))
                if not abs_path.startswith(target_dir):
                    raise RuntimeError(f'Archive tries to extract files outside {target_dir}')
            tarf.extractall(target_dir)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zipf:
            zipf.extractall(target_dir)
    else:
        root, ext = os.path.splitext(filepath)
        if ext.lower() == '.gz':
            with gzip.open(filepath, 'rb') as gzipf:
                with open(root, 'wb') as f:
                    f.write(gzipf.read())
        elif ext.lower() == '.z':
            if os.name != 'posix':
                raise NotImplementedError(f'Only Linux and Mac OS X support {ext} compression')
            retval = subprocess.Popen(f'gzip -d "{filepath}"', shell=True).wait()
            if retval != 0:
                raise RuntimeError(f'Archive file extraction failed for "{filepath}"')
        else:
            raise ValueError('"{filepath}" is not a supported archive file')
