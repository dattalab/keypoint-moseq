#!/usr/bin/env python3
"""
prepare_test_data.py - cross-platform persistent test data cache preparation

- Puts the unpacked dataset in ~/.cache/keypoint-moseq/test-assets
- Skips work if that cache already contains ".complete-{DATA_VERSION}"
- Source archive defaults to a file on O2 but can be overidden with the 
  KPMS_DATA_URL environment variable.
"""

import sys
import tarfile
from os import getenv
from shutil import copyfileobj
from urllib.request import urlopen
from keypoint_moseq._paths import TEST_DATA_ROOT

DATA_VERSION = "v1"
DEFAULT_SRC = f'file:///n/groups/datta/john/projects/kpms-test-data-{DATA_VERSION}.tar.gz'
SRC_URL = getenv("KPMS_DATA_URL", DEFAULT_SRC)

MARKER = TEST_DATA_ROOT / f'.complete-{DATA_VERSION}'
ARCHIVE = TEST_DATA_ROOT / f'{DATA_VERSION}.tar.gz'

if MARKER.exists():
    print(f'keypoint-moseq test data version {DATA_VERSION} already exists. Skipping copy.')
    sys.exit(0)

TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)

print(f'Copying test data from {SRC_URL}')
with urlopen(SRC_URL) as src, ARCHIVE.open('wb') as dst:
    copyfileobj(src, dst, length=1024 * 1024) # 1MB chunks

print(f'Extracting test data archive.')
with tarfile.open(ARCHIVE) as tf:
    tf.extractall(TEST_DATA_ROOT)

ARCHIVE.unlink()
MARKER.touch()
print(f'Done copying keypoint-moseq test data.')