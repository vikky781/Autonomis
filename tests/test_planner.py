from autonomis.contracts import Observation, Recipe
from autonomis.planner import Planner, paired_winner


def anchors():
    return [Recipe("lr"), Recipe("catboost"), Recipe("lightgbm")]


def result(index, recipe, auc, train=None, parent=None, phase="search"):
    return Observation(index, recipe, auc, 0.6, train or auc, 1.0, 500, phase,
                       parent_id=parent)


def test_initial_anchors_are_unique_and_ordered():
    planner = Planner()
    tried = []
    for family in ("lr", "catboost", "lightgbm"):
        proposal = planner.propose([], tried)
        assert proposal.recipe.family == family
        tried.append(proposal.recipe)


def test_feedback_changes_regularization_direction_and_replay_is_deterministic():
    tried = anchors()
    normal = [result(1, tried[0], 0.7), result(2, tried[1], 0.82, 0.84),
              result(3, tried[2], 0.75)]
    overfit = [normal[0], result(2, tried[1], 0.82, 0.98), normal[2]]
    planner = Planner(feature_ablation=False)
    assert planner.propose(normal, tried).recipe.params["depth"] == 8
    proposal = planner.propose(overfit, tried)
    assert proposal.recipe.params["depth"] == 4
    assert proposal.to_dict() == planner.propose(overfit, tried).to_dict()
    assert proposal.parent_id == 2


def test_failed_configuration_is_not_proposed_again():
    tried = anchors()
    history = [result(1, tried[0], 0.7), result(2, tried[1], 0.82, 0.98),
               result(3, tried[2], 0.75)]
    planner = Planner(feature_ablation=False)
    failed = planner.propose(history, tried).recipe
    proposal = planner.propose(history, [*tried, failed])
    assert proposal.recipe.key not in {r.key for r in [*tried, failed]}


def test_single_supported_family_and_budget_exhaustion():
    planner = Planner(families=("lr",), feature_ablation=False)
    base = Recipe("lr")
    proposal = planner.propose([result(1, base, 0.8)], [base])
    assert proposal.recipe.family == "lr"
    assert planner.propose([], [base] * 11) is None


def test_feature_ablation_clones_the_exact_parent():
    tried = anchors()
    a = tried[1].changed(depth=4)
    b = tried[2].changed(num_leaves=31)
    history = [result(1, tried[0], 0.7), result(2, tried[1], 0.8),
               result(3, tried[2], 0.79), result(4, a, 0.83, parent=2),
               result(5, b, 0.81, parent=3)]
    proposal = Planner().propose(history, [*tried, a, b])
    assert proposal.recipe.feature_version == "V1"
    assert proposal.recipe.params == a.params
    assert proposal.parent_id == 4


def test_incomplete_pair_cannot_mix_different_validation_scores():
    a, b = Recipe("lr"), Recipe("catboost")
    first, second = result(1, a, 0.84), result(2, b, 0.83)
    probe = result(3, b, 0.95, phase="probe")
    assert paired_winner([first, second], [probe]).recipe.key == a.key
    both = [result(3, a, 0.8, phase="probe"), probe]
    assert paired_winner([first, second], both).recipe.key == b.key
