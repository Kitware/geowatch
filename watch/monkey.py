"""
See Also:
    ~/code/watch/watch/tasks/fusion/monkey.py
"""


def _monkeypatch_pil():
    # Monkeypatch PIL
    import PIL
    if not hasattr(PIL, 'PILLOW_VERSION') and hasattr(PIL, '__version__'):
        PIL.PILLOW_VERSION = PIL.__version__

_monkeypatch_pil()
