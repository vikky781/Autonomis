import pytest

from autonomis.contracts import Recipe, TrainingDisabled, TrainingPermit


def test_recipe_hash_is_stable_and_parameters_cannot_be_mutated():
    source = {"C": 1.0}
    recipe = Recipe("lr", source)
    source["C"] = 10
    assert recipe.params["C"] == 1
    assert recipe.key == Recipe("lr", {"C": 1.0}).key
    assert recipe.key != Recipe("lr", {"C": 10.0}).key
    with pytest.raises(TypeError):
        recipe.params["C"] = 2


def test_gpu_catboost_recipe_uses_the_frozen_gpu_profile():
    recipe = Recipe("catboost", {}, backend="gpu")
    assert recipe.params["boosting_type"] == "Plain"
    assert recipe.params["border_count"] == 128
    assert recipe.params["devices"] == "0"
    assert recipe.params["gpu_ram_part"] == 0.75
    assert "bootstrap_type" not in recipe.params
    assert recipe.key != Recipe("catboost").key


def test_gpu_is_rejected_for_other_families_and_cpu_only_parameters():
    with pytest.raises(ValueError, match="only for CatBoost"):
        Recipe("lightgbm", {}, backend="gpu")
    with pytest.raises(ValueError, match="family/backend"):
        Recipe("catboost", {"bootstrap_type": "Bayesian"}, backend="gpu")


def test_training_permission_is_not_bound_to_a_hostname():
    with pytest.raises(TrainingDisabled):
        TrainingPermit().check()
    TrainingPermit(True).check()


@pytest.mark.parametrize("params", [{"C": float("nan")}, {"C": -1}, {"unexpected": 1}])
def test_invalid_recipes_are_rejected_before_reservation(params):
    with pytest.raises(ValueError):
        Recipe("lr", params)


def test_recipe_rejects_malformed_parameter_types():
    with pytest.raises(ValueError, match="parameter type"):
        Recipe("lr", {"C": "large"})
    with pytest.raises(ValueError, match="parameter type"):
        Recipe("catboost", {"depth": 6.5})
