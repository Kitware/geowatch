def disable_tensorflow_warnings():
    import warnings
    warnings.filterwarnings(
        action='ignore',
        message='.*Call to deprecated create function.*',
        module=r'.*tensorboard.*'
    )
