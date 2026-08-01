""" Various algorithms for recommender systems that use the same base """
from .recommender import BaseRecommender
from .bass import BassModel
from .content import ContentFiltering
from .social import SocialFiltering
from .popularity import PopularityRecommender
from .random import RandomRecommender

# ImplicitMF depends on `lenskit`, an optional heavy dependency whose older
# `lenskit.algorithms` API is unavailable on modern releases (and does not build
# on recent Python/numpy). Import it lazily so a missing/incompatible lenskit does
# not break the rest of the library. `trecs.models.ImplicitMF` remains available
# whenever a compatible lenskit is installed.
try:
    from .mf import ImplicitMF
except ImportError as _mf_import_error:  # pragma: no cover - depends on optional dep
    import warnings as _warnings

    _warnings.warn(
        "trecs.models.ImplicitMF is unavailable because its optional dependency "
        f"'lenskit' could not be imported ({_mf_import_error}). All other models "
        "are unaffected. Install a compatible 'lenskit' to enable ImplicitMF."
    )
