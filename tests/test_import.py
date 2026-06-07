def test_import():
    import deepEmulator
    assert deepEmulator.__version__


def test_core_modules_import():
    from deepEmulator.core import cartridge, env, registry  # noqa: F401


def test_registry_empty_lookup_raises():
    from deepEmulator.core import registry
    import pytest
    with pytest.raises(KeyError):
        registry.get("DOES_NOT_EXIST")
