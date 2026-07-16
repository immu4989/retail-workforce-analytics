import warnings

import pytest

from workforce_analytics import (
    SimulationConfig,
    TurnoverModel,
    build_snapshots,
    generate,
    time_split,
)

warnings.filterwarnings("ignore")

# One medium simulation shared by the whole suite (session-scoped: the
# generator is the slowest step and the tests only read from it).
CFG = SimulationConfig(n_districts=6, n_months=48, seed=11)


@pytest.fixture(scope="session")
def cfg():
    return CFG


@pytest.fixture(scope="session")
def sim():
    return generate(CFG)


@pytest.fixture(scope="session")
def snapshots(sim):
    return build_snapshots(sim.person_months, horizons=(3, 6, 12))


@pytest.fixture(scope="session")
def splits(snapshots):
    return time_split(snapshots, train_end=28, val_end=34, train_stride=2)


@pytest.fixture(scope="session")
def hourly_model(splits):
    train, val, _ = splits
    return TurnoverModel("hourly", horizons=(3, 6)).fit(train, val)
