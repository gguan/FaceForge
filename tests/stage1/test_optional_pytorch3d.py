import importlib
import sys


def _drop_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def test_visualization_imports_without_pytorch3d(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytorch3d", None)
    monkeypatch.setitem(sys.modules, "pytorch3d.renderer", None)
    monkeypatch.setitem(sys.modules, "pytorch3d.structures", None)
    _drop_modules("faceforge.stage1.visualization")

    visualization = importlib.import_module("faceforge.stage1.visualization")

    assert visualization._PYTORCH3D_AVAILABLE is False


def test_stage1_package_imports_without_pytorch3d(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytorch3d", None)
    monkeypatch.setitem(sys.modules, "pytorch3d.renderer", None)
    monkeypatch.setitem(sys.modules, "pytorch3d.structures", None)
    _drop_modules("faceforge.stage1")

    stage1 = importlib.import_module("faceforge.stage1")

    assert stage1.Stage1Pipeline is not None
