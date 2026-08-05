"""The released JEPA-WAM LIBERO training mixture."""

from prismatic.vla.datasets.rlds.oxe.configs import LIBERO_DATASETS

OXE_NAMED_MIXTURES = {
    "libero_4_task_suites_no_noops": [(name, 1.0) for name in LIBERO_DATASETS],
}
