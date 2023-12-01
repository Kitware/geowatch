"""
A package that holds third-party libraries that some geowatch tasks require.

To regenerate static submodule:

    # Checkout submodules

    # Run script:
    python ~/code/watch/geowatch_tpl/snapshot_submodules.py
"""

import os
import sys
# Adds the "modules" subdirectory to the python path.
# See https://gitlab.kitware.com/smart/watch/-/merge_requests/148#note_1050127
# for discussion of how to refactor this in the future.


# By default a few modules get added, but we should port these to the
# submodule_static repo and require they be handled on an as-needed basis.
TPL_DPATH = os.path.dirname(__file__)
MODULE_DPATH = os.path.join(os.path.dirname(__file__), 'modules')
sys.path.append(MODULE_DPATH)


FORCE_STATIC = int(os.environ.get('GEOWATCH_STATIC_TPL', '0'))


STATIC_SUBMODULES = {
    'scalemae': {
        'rel_dpath': 'scale-mae/scalemae',
        'ignore': [
            'scale-mae/scalemae/splits',
        ]
    },
    'torchview': {
        'rel_dpath': 'torchview/torchview',
    },
    'lop':
    {
        'rel_dpath': 'loss-of-plasticity/lop',
        'parts': [
            'loss-of-plasticity/lop/__init__.py',
            'loss-of-plasticity/lop/algos',
        ]
    },
    'segment_anything': {
        'rel_dpath': 'segment-anything/segment_anything',
        'parts': [
            'segment-anything/segment_anything',
            'segment-anything/LICENSE',
        ]
    },
}


def import_submodule(submod_name):
    """
    If the developer version of the submodule exists, use that.
    Otherwise use the static version.

    Args:
        submod_name (str): registered TPL module name

    Returns:
        ModuleType

    CommandLine:
        FORCE_STATIC=1 xdoctest -m geowatch_tpl import_submodule
        FORCE_STATIC=0 xdoctest -m geowatch_tpl import_submodule

    Example:
        >>> import geowatch_tpl
        >>> import ubelt as ub
        >>> for key in geowatch_tpl.STATIC_SUBMODULES:
        >>>     module = geowatch_tpl.import_submodule(key)
        >>>     print('{} module = {}'.format(key, ub.urepr(module, nl=1)))
    """
    import ubelt as ub
    if submod_name in sys.modules:
        return sys.modules[submod_name]

    tpl_dpath = ub.Path(TPL_DPATH)

    dev_mod_dpath = tpl_dpath / 'modules'
    dev_submod_dpath = tpl_dpath / 'submodules'
    static_submod_dpath = tpl_dpath / 'submodules_static'

    if submod_name in STATIC_SUBMODULES:
        info = STATIC_SUBMODULES[submod_name]
        rel_dpath = info['rel_dpath']
        cand1 = dev_submod_dpath / rel_dpath
        cand2 = static_submod_dpath / rel_dpath
        if cand1.exists() and not FORCE_STATIC:
            new_module_dpath = cand1
        else:
            assert cand2 is not None and cand2.exists()
            new_module_dpath = cand2
    else:
        # Assume we have a submodule with the same repo name if
        # it is unregistered here.
        import warnings
        cand_old = dev_mod_dpath / submod_name
        cand_new = dev_submod_dpath / submod_name / submod_name
        if cand_old.exists():
            warnings.warn('Warning: Unregistered submodule (old style)')
            new_module_dpath = cand_old
        else:
            warnings.warn('Warning: Unregistered submodule (new style)')
            new_module_dpath = cand_new
        assert new_module_dpath.exists()

    new_sys_dpath = os.fspath(new_module_dpath.parent)
    sys.path.append(new_sys_dpath)
    module = ub.import_module_from_name(submod_name)
    return module
