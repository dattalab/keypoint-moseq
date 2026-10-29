import pytest

import keypoint_moseq as kpms


def test_tutorial_sigmasq_loc(
    tutorial_sigmasq_loc, tutorial_outlier_removed, num_regression
):
    project_dir, _, _ = tutorial_outlier_removed
    config = lambda: kpms.load_config(project_dir)

    assert config()["cen_hypparams"]["sigmasq_loc"] == tutorial_sigmasq_loc
    num_regression.check(
        {"sigmasq_loc": [tutorial_sigmasq_loc]},
        default_tolerance=dict(rtol=1e-6),
    )


def hypparams_model():
    return {
        "hypparams": {
            "trans_hypparams": {
                "num_states": 100,
                "gamma": 1e3,
                "alpha": 5.7,
                "kappa": 1e6,
            },
            "ar_hypparams": {
                "latent_dim": 10,
                "nlags": 3,
                "S_0_scale": 0.01,
                "K_0_scale": 10.0,
                "num_states": 100,
                "nu_0": 12,
            },
        }
    }


@pytest.mark.parametrize(
    "key", ["num_states", "latent_dim", "nlags", "S_0_scale", "K_0_scale"]
)
def test_update_hypparams_refuses_keys_set_at_initialization(key):
    model = hypparams_model()

    with pytest.raises(ValueError, match=key):
        kpms.update_hypparams(model, **{key: 5})

    assert model == hypparams_model()


def test_update_hypparams_changes_kappa():
    model = kpms.update_hypparams(hypparams_model(), kappa=1e4)

    assert model["hypparams"]["trans_hypparams"]["kappa"] == 1e4
