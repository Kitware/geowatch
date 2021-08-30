"""
Do a better job with default argparse

TODO: work on this
"""


def parse_docstring_args(cls):
    import inspect
    from xdoctest.docstr import docscrape_google
    from pytorch_lightning.utilities.argparse import (
        get_init_arguments_and_types,
        str_to_bool,
        str_to_bool_or_int,
        str_to_bool_or_str,
        _gpus_allowed_type,
        _int_or_float_type,
    )

    symbol = cls.__init__
    if symbol.__doc__ is None:
        arg_infos = []
    else:
        arg_infos = list(docscrape_google.parse_google_args())

    if not arg_infos:
        # Try cls instead
        arg_infos = list(docscrape_google.parse_google_args(cls.__doc__))

    ignore_arg_names = ["self", "args", "kwargs"]
    if hasattr(cls, "get_deprecated_arg_names"):
        ignore_arg_names += cls.get_deprecated_arg_names()

    # Get symbols from cls or init function.
    args_and_types = get_init_arguments_and_types(symbol)

    sig_lut = {name: (sig_type, sig_default)
               for name, sig_type, sig_default in args_and_types}

    for arg_info in arg_infos:
        name = arg_info['name']
        arg_info['str_type'] = arg_info['type']
        if name in sig_lut:
            sig_type, sig_default = sig_lut[name]
            arg_info['sig_type'] = sig_type
            arg_info['sig_default'] = sig_default

    type_lut = {
        'int': int,
        'float': float,
        'str': str,
    }

    return_infos = []

    for arg_info in arg_infos:
        name = arg_info['name']

        if name in ignore_arg_names:
            continue

        str_type = arg_info['str_type']
        sig_types = arg_info.get('sig_type', inspect._empty)

        arg_types = []
        for sig_type in sig_types:
            if sig_type is not inspect._empty:
                arg_types.append(sig_type)

        resolved = type_lut.get(str_type, inspect._empty)
        if resolved is not inspect._empty:
            arg_types.append(resolved)

        arg_kwargs = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type = str_to_bool
            elif int in arg_types:
                use_type = str_to_bool_or_int
            elif str in arg_types:
                use_type = str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            if len(arg_types) == 0:
                use_type = inspect._empty
            else:
                use_type = arg_types[0]

        if name == "gpus" or name == "tpu_cores":
            use_type = _gpus_allowed_type

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if name == "track_grad_norm":
            use_type = float

        arg_info['use_type'] = use_type
        arg_info['arg_kwargs'] = arg_kwargs
        return_infos.append(arg_info)

    return return_infos


def add_arginfos_to_parser(parent_parser, arg_infos):
    import inspect
    for arg_info in arg_infos:
        name = arg_info['name']
        name = arg_info['name']
        arg_default = arg_info['sig_default']
        use_type = arg_info['use_type']
        desc = arg_info['desc']
        arg_kwargs = arg_info['arg_kwargs'].copy()
        if use_type is not inspect._empty:
            arg_kwargs['type'] = use_type
        parent_parser.add_argument(
            f"--{name}", dest=name, default=arg_default, help=desc, **arg_kwargs
        )
    return parent_parser


def add_argparse_args(cls, parent_parser):
    arg_infos = parse_docstring_args(cls)
    add_arginfos_to_parser(parent_parser, arg_infos)
