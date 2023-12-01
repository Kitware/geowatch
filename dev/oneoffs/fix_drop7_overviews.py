def main():
    """
    Fix the overviews for the QA bands.
    """
    import ubelt as ub
    dvc_dpath = ub.Path('/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc/')
    dpath = dvc_dpath / 'Aligned-Drop7'
    all_qa_fpaths = list(dpath.glob('*/*/*/*/*_quality.tif'))

    from simple_dvc import SimpleDVC
    dvc = SimpleDVC.coerce(dvc_dpath)
    dvc.unprotect(all_qa_fpaths)

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=16, name='addo-queue')

    for fpath in all_qa_fpaths:
        command = f'gdaladdo -r nearest {fpath}'
        queue.submit(command)

    # queue.print_commands()
    queue.run()

    sidecars = set()
    for fpath in all_qa_fpaths:
        sidecars.update(dvc.sidecar_paths(fpath))

    tracked_paths = [p.parent / p.stem for p in sidecars]
    dvc.add(tracked_paths, verbose=2)

    # stat = fpath.readlink().stat()
    # octal_to_string(stat.st_mode)
    # print(ub.cmd(f'ls -alL {fpath}').stdout)


def octal_to_string(octal):
    """
    https://pythoncircle.com/post/716/python-program-to-convert-linux-file-permissions-from-octal-number-to-rwx-string/
    """
    result = ""
    value_letters = [(4, "r"), (2, "w"), (1, "x")]
    # Iterate over each of the digits in octal
    for digit in [int(n) for n in str(octal)]:
        # Check for each of the permissions values
        for value, letter in value_letters:
            if digit >= value:
                result += letter
                digit -= value
            else:
                result += '-'
    return result
