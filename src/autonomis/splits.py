"""Persist row identities, never resample validation silently on restart."""

from sklearn.model_selection import train_test_split

from autonomis.contracts import fingerprint
from autonomis.storage import read_json, write_json

GROUPS = ("train", "search", "audit", "probe_train", "probe_valid")


def create_splits(data, seed=20261026):
    if data.y is None or data.y.nunique() != 2 or not data.sha256:
        raise ValueError("Splitting requires a fingerprinted binary labelled table")
    indices = list(range(len(data.ids)))
    dev, audit = train_test_split(indices, test_size=0.2, stratify=data.y, random_state=seed)
    train, search = train_test_split(dev, test_size=0.25, stratify=data.y.iloc[dev],
                                     random_state=seed + 1)
    probe_train, probe_valid = train_test_split(dev, test_size=0.25,
        stratify=data.y.iloc[dev], random_state=seed + 2)
    groups = dict(zip(GROUPS, (train, search, audit, probe_train, probe_valid), strict=True))
    manifest = {"dataset_hash": data.sha256, "seed": seed, "policy": "stratified_customer_rows",
                **{name: data.ids.iloc[rows].tolist() for name, rows in groups.items()}}
    validate_splits(manifest, data)
    return manifest


def validate_splits(manifest, data):
    if manifest["dataset_hash"] != data.sha256:
        raise ValueError("Split source fingerprint changed")
    universe = set(data.ids)
    sets = {}
    target = dict(zip(data.ids, data.y, strict=True))
    for name in GROUPS:
        values = manifest[name]
        group = set(values)
        if not values or len(group) != len(values) or group - universe:
            raise ValueError(f"Invalid membership: {name}")
        if {target[value] for value in values} != {0, 1}:
            raise ValueError(f"One-class partition: {name}")
        sets[name] = group
    train, search, audit = (sets[name] for name in ("train", "search", "audit"))
    if train & search or train & audit or search & audit or train | search | audit != universe:
        raise ValueError("Main partitions overlap or omit customers")
    if (sets["probe_train"] & sets["probe_valid"] or
            sets["probe_train"] | sets["probe_valid"] != train | search):
        raise ValueError("Probe partitions overlap or include audit customers")


def save_splits(path, manifest):
    return write_json(path, manifest)


def read_splits(path, data):
    manifest = read_json(path)
    validate_splits(manifest, data)
    return manifest


def split_hash(manifest):
    return fingerprint(manifest)


def positions(data, identities):
    lookup = {value: index for index, value in enumerate(data.ids)}
    try:
        return [lookup[value] for value in identities]
    except KeyError as error:
        raise ValueError("Unknown customer in split manifest") from error
