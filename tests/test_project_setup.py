import os

import pytest
import yaml

import keypoint_moseq as kpms

TUTORIAL_BODYPARTS = [
    "tail",
    "spine4",
    "spine3",
    "spine2",
    "spine1",
    "head",
    "nose",
    "right ear",
    "left ear",
]

TUTORIAL_SKELETON = [
    ["tail", "spine4"],
    ["spine4", "spine3"],
    ["spine3", "spine2"],
    ["spine2", "spine1"],
    ["spine1", "head"],
    ["nose", "head"],
    ["left ear", "head"],
    ["right ear", "head"],
]


def plain_config(config):
    """Convert the jax index arrays so the config can be stored as YAML."""
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in config.items()
    }


def setup_tutorial_project(project_dir, config_edits):
    """Set up a project with the custom bodyparts and config edits from the
    modeling notebook."""
    kpms.setup_project(
        project_dir,
        video_dir="path/to/videos/",
        bodyparts=TUTORIAL_BODYPARTS,
        skeleton=TUTORIAL_SKELETON,
    )
    kpms.update_config(project_dir, **config_edits)


@pytest.fixture
def project_dir(tmp_path):
    return str(tmp_path / "demo_project")


@pytest.fixture
def dlc_config(dlc_project_dir):
    return os.path.join(dlc_project_dir, "config.yaml")


def test_config_without_project_raises(project_dir):
    config = lambda: kpms.load_config(project_dir)

    with pytest.raises(FileNotFoundError):
        config()


def test_setup_project_skips_existing_directory_without_config(
    project_dir, capsys
):
    os.makedirs(project_dir)

    kpms.setup_project(project_dir)

    assert "already exists" in " ".join(capsys.readouterr().out.split())
    assert not os.path.exists(os.path.join(project_dir, "config.yml"))
    with pytest.raises(FileNotFoundError):
        kpms.load_config(project_dir)


def test_default_config(project_dir, data_regression):
    kpms.setup_project(project_dir)
    config = lambda: kpms.load_config(project_dir)

    data_regression.check(plain_config(config()))


def test_tutorial_config(project_dir, tutorial_config_edits, data_regression):
    setup_tutorial_project(project_dir, tutorial_config_edits)
    config = lambda: kpms.load_config(project_dir)

    data_regression.check(plain_config(config()))


def test_dlc_config(project_dir, dlc_config, data_regression):
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config)

    assert os.listdir(project_dir) == ["config.yml"]
    data_regression.check(
        kpms.load_config(project_dir, check_if_valid=False, build_indexes=False)
    )


def test_dlc_config_after_tutorial_edits(tutorial_project_dir, data_regression):
    config = lambda: kpms.load_config(tutorial_project_dir)

    data_regression.check(plain_config(config()))


def test_config_after_dlc_setup_raises_on_placeholder_heading_bodyparts(
    project_dir, dlc_config, capsys
):
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config)
    config = lambda: kpms.load_config(project_dir)

    # The error comes from `list.index`, whose message names the missing item
    # before Python 3.14 and not from 3.14 on; the printed message below
    # names the bodypart on every version.
    with pytest.raises(ValueError, match="not in list"):
        config()

    printed = " ".join(capsys.readouterr().out.split())
    assert "`anterior_bodyparts` contains BODYPART1" in printed
    assert "`posterior_bodyparts` contains BODYPART3" in printed


@pytest.mark.parametrize("customized", [False, True], ids=["default", "tutorial"])
def test_rerunning_setup_keeps_existing_config(
    project_dir, tutorial_config_edits, customized
):
    if customized:
        setup_tutorial_project(project_dir, tutorial_config_edits)
    else:
        kpms.setup_project(project_dir)
    config = lambda: kpms.load_config(project_dir)
    original = plain_config(config())

    kpms.setup_project(project_dir, fps=120, bodyparts=["a", "b"])

    assert plain_config(config()) == original


@pytest.mark.parametrize("skeleton", ["empty", "removed"])
def test_config_without_a_skeleton(project_dir, tutorial_config_edits, skeleton, capsys):
    """`skeleton:` left empty in config.yml (YAML null), or the line removed,
    loads as a skeleton with no edges."""
    setup_tutorial_project(project_dir, tutorial_config_edits)
    path = os.path.join(project_dir, "config.yml")
    with open(path) as f:
        edited = yaml.safe_load(f)
    if skeleton == "empty":
        edited["skeleton"] = None
    else:
        del edited["skeleton"]
    with open(path, "w") as f:
        yaml.safe_dump(edited, f)

    assert kpms.load_config(project_dir)["skeleton"] == []
    assert "ACTION REQUIRED" not in capsys.readouterr().out


def test_update_config_keeps_keys_without_a_default(tmp_path):
    project_dir = str(tmp_path)
    kpms.generate_config(project_dir)
    path = tmp_path / "config.yml"
    config = yaml.safe_load(path.read_text())
    config["save_every_n_iters"] = 25
    config["my_custom_key"] = 3
    config["ar_hypparams"]["my_ar_key"] = 4
    path.write_text(yaml.safe_dump(config))

    kpms.update_config(project_dir, fps=60, new_key="x")

    updated = kpms.load_config(project_dir, check_if_valid=False, build_indexes=False)
    assert updated["fps"] == 60
    assert updated["save_every_n_iters"] == 25
    assert updated["my_custom_key"] == 3
    assert updated["new_key"] == "x"
    assert updated["ar_hypparams"]["my_ar_key"] == 4


def test_update_config_with_a_loaded_config(tmp_path):
    """load_config adds index arrays that are not settings."""
    project_dir = str(tmp_path)
    kpms.generate_config(project_dir)

    kpms.update_config(project_dir, **kpms.load_config(project_dir))

    config = kpms.load_config(project_dir, check_if_valid=False, build_indexes=False)
    assert "anterior_idxs" not in config
    assert "posterior_idxs" not in config
