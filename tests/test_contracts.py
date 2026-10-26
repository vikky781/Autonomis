import socket

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


def test_gpu_recipe_stops_at_explicit_integration_boundary():
    with pytest.raises(ValueError, match="GPU integration is deferred"):
        Recipe("catboost", {}, backend="gpu")


def test_training_requires_explicit_permission_on_named_host():
    for permit in (TrainingPermit(), TrainingPermit(True, "different-host")):
        with pytest.raises(TrainingDisabled):
            permit.check()
    TrainingPermit(True, socket.gethostname()).check()


@pytest.mark.parametrize("params", [{"C": float("nan")}, {"C": -1}, {"unexpected": 1}])
def test_invalid_recipes_are_rejected_before_reservation(params):
    with pytest.raises(ValueError):
        Recipe("lr", params)
