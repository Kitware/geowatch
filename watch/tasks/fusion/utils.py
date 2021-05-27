import inspect
def filter_args(args, func):
    signature = inspect.signature(func)
    return {
        key: value
        for key, value in args.items()
        if key in signature.parameters.keys()
    }