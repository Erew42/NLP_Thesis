from __future__ import annotations


class ThesisAssetsError(Exception):
    """Base error for thesis asset scaffold failures."""


class RegistryError(ThesisAssetsError):
    """Raised when registry definitions are invalid."""


class ResolutionError(ThesisAssetsError):
    """Raised when a source run root cannot be resolved unambiguously."""


class MissingArtifactError(ThesisAssetsError):
    """Raised when a required source artifact is unavailable."""


class SampleContractError(ThesisAssetsError):
    """Raised when a sample contract cannot be satisfied safely."""


class AssetBuildError(ThesisAssetsError):
    """Raised when an asset builder cannot complete successfully."""
