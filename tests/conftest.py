import tempfile
import pytest
from selenium.webdriver.chrome.options import Options
from keypoint_moseq._paths import TEST_DATA_ROOT

def pytest_setup_options():
    opts = Options()
    opts.add_argument('--headless=new')
    profile_dir = tempfile.mkdtemp(prefix="chromium_profile_")
    opts.add_argument(f'--user-data-dir={profile_dir}')
    return opts

@pytest.fixture(scope='function')
def deeplabcut_2d_zenodo_dir():
    return TEST_DATA_ROOT / 'kpms-test-data/open_field_2d'

@pytest.fixture(scope='function')
def demo_project_dir(tmp_path):
    return tmp_path / 'demo_project'