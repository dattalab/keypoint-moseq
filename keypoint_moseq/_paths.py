import sys
from pathlib import Path
from os import getenv

def default_cache_dir(app_name: str) -> Path:
    """
    Returns the standard best practice cache dir for each operating system.

    Parameters
    ----------
    The name of the app to return the standard cache dir for

    Returns
    -------
    str:
        The cache dir for the user's system.
    """
    if sys.platform == 'win32': # Windows
        root = getenv('LOCALAPPDATA', Path.home())
        return Path(root) / app_name / 'Cache'
    elif sys.platform == 'darwin': # Mac
        return Path.home() / 'Library/Caches' / app_name
    else: # Linux + otherwise
        root =  getenv("XDG_CACHE_HOME", Path.home() / '.cache')
        return Path(root) / app_name
    
TEST_DATA_ROOT = default_cache_dir('keypoint-moseq') / 'test-assets'