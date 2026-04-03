import sys
from types import ModuleType
from pathlib import Path


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def test_import_faceboxes_detector_supports_script_style_imports_without_leaking_utils(tmp_path):
    from faceforge.stage1 import pipnet_inference

    code_base = tmp_path / 'pixel3dmm_repo'
    src_root = code_base / 'src'
    other_root = tmp_path / 'other_modules'

    _write(src_root / 'pixel3dmm' / '__init__.py', '')
    _write(src_root / 'pixel3dmm' / 'env_paths.py', 'CODE_BASE = ""\n')
    _write(src_root / 'pixel3dmm' / 'preprocessing' / '__init__.py', '')
    _write(src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / '__init__.py', '')
    _write(src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / '__init__.py', '')
    _write(src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / 'utils' / '__init__.py', '')
    _write(src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / 'utils' / 'config.py', 'cfg = {"ok": True}\n')
    _write(src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / 'utils' / 'nms' / '__init__.py', '')
    _write(
        src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / 'utils' / 'nms' / 'py_cpu_nms.py',
        'def py_cpu_nms(dets, thresh):\n'
        '    return ["fallback", dets, thresh]\n',
    )
    _write(
        src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / 'utils' / 'nms_wrapper.py',
        'from .nms.cpu_nms import cpu_nms, cpu_soft_nms\n'
        'def nms(dets, thresh):\n'
        '    return cpu_nms(dets, thresh)\n',
    )
    _write(
        src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / 'detector.py',
        'class Detector:\n    pass\n',
    )
    _write(
        src_root / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2' / 'faceboxes_detector.py',
        'from detector import Detector\n'
        'from utils.config import cfg\n'
        'from utils.nms_wrapper import nms\n'
        'class FaceBoxesDetector(Detector):\n'
        '    config = cfg\n',
    )
    _write(other_root / 'utils' / '__init__.py', '')
    _write(other_root / 'utils' / 'masking.py', 'MARKER = "mica-utils"\n')

    added = []
    try:
        sys.path.insert(0, str(other_root))
        added.append(str(other_root))

        added.extend(pipnet_inference._ensure_pixel3dmm_src_path(str(code_base)))
        detector_cls = pipnet_inference._import_faceboxes_detector_class(str(code_base))
        from utils.masking import MARKER

        assert detector_cls.__name__ == 'FaceBoxesDetector'
        assert detector_cls.config == {'ok': True}
        assert detector_cls.__module__.endswith('faceboxes_detector')
        assert MARKER == 'mica-utils'
        assert any(Path(entry) == src_root for entry in added)
        assert 'detector' not in sys.modules
    finally:
        for entry in reversed(added):
            if entry in sys.path:
                sys.path.remove(entry)
        for name in list(sys.modules):
            if name.startswith('pixel3dmm') or name.startswith('utils') or name == 'detector':
                sys.modules.pop(name, None)


def test_ensure_scipy_simps_compat_adds_alias_when_missing(monkeypatch):
    from faceforge.stage1 import pipnet_inference

    integrate_module = ModuleType('scipy.integrate')

    def _simpson(*args, **kwargs):
        return ('simpson', args, kwargs)

    integrate_module.simpson = _simpson
    monkeypatch.delitem(sys.modules, 'scipy.integrate', raising=False)

    scipy_module = ModuleType('scipy')
    scipy_module.integrate = integrate_module
    monkeypatch.setitem(sys.modules, 'scipy', scipy_module)
    monkeypatch.setitem(sys.modules, 'scipy.integrate', integrate_module)

    pipnet_inference._ensure_scipy_simps_compat()

    assert integrate_module.simps is _simpson
