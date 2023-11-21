def fix_pil_version():
    # Monkeypatch PIL
    import PIL
    if not hasattr(PIL, 'PILLOW_VERSION') and hasattr(PIL, '__version__'):
        PIL.PILLOW_VERSION = PIL.__version__
