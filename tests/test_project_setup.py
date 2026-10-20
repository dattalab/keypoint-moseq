import os

import pytest

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


def setup_tutorial_project(project_dir):
    """Set up a project with the custom bodyparts and config edits from the
    modeling notebook."""
    kpms.setup_project(
        project_dir,
        video_dir="path/to/videos/",
        bodyparts=TUTORIAL_BODYPARTS,
        skeleton=TUTORIAL_SKELETON,
    )
    kpms.update_config(
        project_dir,
        video_dir="dlc_project/videos/",
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["spine4"],
        use_bodyparts=[
            "spine4",
            "spine3",
            "spine2",
            "spine1",
            "head",
            "nose",
            "right ear",
            "left ear",
        ],
        fps=30,
    )


@pytest.fixture
def project_dir(tmp_path):
    return str(tmp_path / "demo_project")


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


def test_tutorial_config(project_dir, data_regression):
    setup_tutorial_project(project_dir)
    config = lambda: kpms.load_config(project_dir)

    data_regression.check(plain_config(config()))


@pytest.mark.parametrize("customized", [False, True], ids=["default", "tutorial"])
def test_rerunning_setup_keeps_existing_config(project_dir, customized):
    if customized:
        setup_tutorial_project(project_dir)
    else:
        kpms.setup_project(project_dir)
    config = lambda: kpms.load_config(project_dir)
    original = plain_config(config())

    kpms.setup_project(project_dir, fps=120, bodyparts=["a", "b"])

    assert plain_config(config()) == original
