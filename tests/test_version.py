def test_version():
    """ Test that the version attribute exists """
    import watch
    assert isinstance(watch.__version__, str)
