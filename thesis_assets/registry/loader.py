from __future__ import annotations

from importlib import import_module

from thesis_assets.config.constants import REGISTRY_MODULES
from thesis_assets.errors import RegistryError
from thesis_assets.specs import AssetSpec


def load_registry() -> tuple[AssetSpec, ...]:
    assets: list[AssetSpec] = []
    seen_ids: dict[str, str] = {}

    for module_name in REGISTRY_MODULES:
        module = import_module(f"thesis_assets.registry.{module_name}")
        module_assets = getattr(module, "ASSETS", None)
        if module_assets is None:
            raise RegistryError(f"Registry module {module_name!r} is missing ASSETS.")
        for asset in module_assets:
            if not isinstance(asset, AssetSpec):
                raise RegistryError(f"Registry module {module_name!r} contains a non-AssetSpec entry.")
            previous_module = seen_ids.get(asset.asset_id)
            if previous_module is not None:
                raise RegistryError(
                    f"Duplicate asset_id {asset.asset_id!r} found in {module_name!r} and {previous_module!r}."
                )
            seen_ids[asset.asset_id] = module_name
            assets.append(asset)
    return tuple(assets)


def load_assets_by_chapter(chapter: str) -> tuple[AssetSpec, ...]:
    assets = tuple(asset for asset in load_registry() if asset.chapter == chapter)
    if not assets:
        raise RegistryError(f"No assets registered for chapter {chapter!r}.")
    return assets


def load_asset_by_id(asset_id: str) -> AssetSpec:
    for asset in load_registry():
        if asset.asset_id == asset_id:
            return asset
    raise RegistryError(f"Unknown asset_id: {asset_id!r}.")
