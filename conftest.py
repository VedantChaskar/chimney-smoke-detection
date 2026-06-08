# conftest.py — root-level pytest configuration
"""
Add repo root to sys.path so all test subdirectories can resolve their
own sys.path.insert() calls regardless of where pytest is invoked from.

Also fixes the `src`/`config` namespace collision: temporal_reference/,
gemini_inpainting/, and smokescreen/ each expose top-level `src` and `config`
packages with different contents. The hook below evicts all modules whose
__file__ belongs to a different subproject and pins the correct subproject to
the front of sys.path immediately before each test module in those directories
is imported.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_PHYSICS_ROOT = _ROOT / 'physics_methods'

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# physics_methods/ is the parent of smoke_opacity and vitmatte packages
if str(_PHYSICS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHYSICS_ROOT))

# Subprojects that each own their own `src` + `config` namespace
_SUBPROJECT_DIRS = [
    str(_PHYSICS_ROOT / 'temporal_reference'),
    str(_PHYSICS_ROOT / 'gemini_inpainting'),
    str(_PHYSICS_ROOT / 'smokescreen'),
    str(_PHYSICS_ROOT / 'smoke_opacity'),
    str(_PHYSICS_ROOT / 'vitmatte'),
]

_active_sp = [None]  # track which subproject is currently active


def pytest_pycollect_makemodule(module_path, parent):
    """Pin the correct subproject namespace before each test module is imported."""
    mp = str(module_path)
    for sp in _SUBPROJECT_DIRS:
        if mp.startswith(sp + '/'):
            if _active_sp[0] == sp:
                break  # already set up, no action needed
            # Evict all modules whose __file__ lives inside any subproject dir
            for key in list(sys.modules.keys()):
                mod = sys.modules.get(key)
                mod_file = getattr(mod, '__file__', None) or ''
                if any(mod_file.startswith(d + '/') for d in _SUBPROJECT_DIRS):
                    del sys.modules[key]
            # Bring the owning subproject to the front of sys.path
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)
            _active_sp[0] = sp
            break
    return None  # let pytest create the default Module collector
