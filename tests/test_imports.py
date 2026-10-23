def test_modeling_notebook_imports():
    import keypoint_moseq as kpms
    import matplotlib.pyplot as plt


def test_installation_check_imports_stay_out_of_the_package_namespace():
    # The tutorial names a variable `metadata`, which `from keypoint_moseq
    # import *` would otherwise replace with importlib.metadata.
    import keypoint_moseq as kpms

    for name in ["metadata", "site", "platform", "importlib", "Version", "InvalidVersion"]:
        assert not hasattr(kpms, name), name
