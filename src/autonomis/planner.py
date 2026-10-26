"""A finite sequential policy: all decisions depend on recorded development feedback."""

from dataclasses import dataclass, replace

from autonomis.contracts import Recipe

CATALOG = {
    "lr": (("C", (0.1, 1.0, 10.0), -1),),
    "catboost": (("depth", (4, 6, 8), -1), ("l2_leaf_reg", (3.0, 10.0, 30.0), 1)),
    "lightgbm": (("num_leaves", (7, 15, 31), -1),
                 ("min_child_samples", (20, 50, 100), 1),
                 ("reg_lambda", (1.0, 10.0, 30.0), 1)),
}


def rank(observation):
    return (-round(observation.roc_auc, 8), -round(observation.average_precision, 8),
            observation.fit_seconds, observation.recipe.key)


def finalists(history):
    unique = {}
    for row in sorted((o for o in history if o.phase == "search"), key=rank):
        unique.setdefault(row.recipe.key, row)
    return list(unique.values())[:2]


def paired_winner(search_finalists, probes):
    if not search_finalists:
        raise ValueError("No valid finalist")
    wanted = {o.recipe.key for o in search_finalists}
    paired = [o for o in probes if o.phase == "probe" and o.recipe.key in wanted]
    if len(wanted) == 2 and {o.recipe.key for o in paired} == wanted and len(paired) == 2:
        return min(paired, key=rank)
    return min(search_finalists, key=rank)


@dataclass(frozen=True)
class Proposal:
    recipe: Recipe
    parent_id: int | None
    reason: str
    alternatives: tuple[str, ...] = ()

    def to_dict(self):
        return {"recipe": self.recipe.to_dict(), "parent_id": self.parent_id,
                "reason": self.reason, "alternatives": list(self.alternatives)}


class Planner:
    def __init__(self, *, families=("lr", "catboost", "lightgbm"), feature_ablation=True,
                 seed=20261026, threads=4, backend="cpu"):
        if not families or set(families) - CATALOG.keys():
            raise ValueError("No supported family")
        if backend not in {"cpu", "gpu"}:
            raise ValueError("Backend must be cpu or gpu")
        self.anchors = [Recipe(f, backend=backend if f == "catboost" else "cpu",
                               seed=seed, threads=threads) for f in families]
        self.feature_ablation = feature_ablation

    def propose(self, history, attempted):
        if len(attempted) >= 11:
            return None
        tried = {recipe.key for recipe in attempted}
        for anchor in self.anchors:
            if anchor.key not in tried:
                return Proposal(anchor, None, "Initial complementary anchor")
        search = [o for o in history if o.phase == "search"]
        anchor_keys = {r.key for r in self.anchors}
        initial = sorted((o for o in search if o.recipe.key in anchor_keys), key=rank)
        families = list(dict.fromkeys(o.recipe.family for o in initial))[:2]
        if not families:
            return None
        incumbents = {f: min((o for o in search if o.recipe.family == f), key=rank)
                      for f in families}
        moved = {f: sum(r.family == f and r.key not in anchor_keys for r in attempted)
                 for f in families}
        ablated = any(r.feature_version == "V1" for r in attempted)
        if (self.feature_ablation and not ablated and all(moved.values())):
            parent = min(incumbents.values(), key=rank)
            recipe = replace(parent.recipe, feature_version="V1")
            return Proposal(recipe, parent.attempt_id, "Single fixed-parent feature ablation")
        parents = {o.attempt_id: o for o in search}
        adaptive = [o for o in search if o.parent_id in parents]
        resolved = not self.feature_ablation or ablated
        if resolved and all(count >= 2 for count in moved.values()) and len(adaptive) >= 4:
            ordered = sorted(search, key=lambda o: o.attempt_id)
            gains, best = [], 0.0
            for row in ordered:
                if row.parent_id is not None:
                    gains.append(row.roc_auc - best)
                best = max(best, row.roc_auc)
            if len(gains) >= 4 and all(gain < 0.002 for gain in gains[-4:]):
                return None
        order = sorted(families, key=lambda f: (moved[f] > 0, rank(incumbents[f])))
        leading = order[0]
        recent = sorted((o for o in adaptive if o.recipe.family == leading),
                        key=lambda o: o.attempt_id)[-2:]
        if (len(order) == 2 and all(moved.values()) and len(recent) == 2 and
                all(o.roc_auc - parents[o.parent_id].roc_auc < 0.002 for o in recent)):
            order.reverse()
        for family in order:
            parent = incumbents[family]
            choices, direction = self._neighbors(parent)
            available = [r for r in choices if r.key not in tried]
            if available:
                return Proposal(available[0], parent.attempt_id, direction,
                                tuple(r.key for r in available[1:]))
        return None

    @staticmethod
    def _neighbors(parent):
        recipe = parent.recipe
        choices = []
        cap_name = {"catboost": "iterations", "lightgbm": "n_estimators"}.get(recipe.family)
        if cap_name and recipe.params[cap_name] == 1000 and (parent.tree_count or 0) >= 900:
            choices.append(recipe.changed(**{cap_name: 2000}))
        overfit = parent.train_roc_auc - parent.roc_auc > 0.05
        for preferred in (True, False):
            for name, values, regularize_direction in CATALOG[recipe.family]:
                direction = regularize_direction * (1 if overfit else -1)
                direction *= 1 if preferred else -1
                index = values.index(recipe.params[name]) + direction
                if 0 <= index < len(values):
                    choices.append(recipe.changed(**{name: values[index]}))
        explanation = ("Train/Search gap > 0.05: test lower capacity or stronger regularization"
                       if overfit else "Test a modest capacity increase or weaker regularization")
        if choices and cap_name and choices[0].params[cap_name] == 2000:
            explanation = "Best tree count reached 90% of cap: test a longer horizon"
        return choices, explanation
