"""
cd ~/code/watch/dev/mwe

# Normal behavior
python rich_tee.py

# Works fine
python rich_tee.py 2>&1

# How do make this work?
python rich_tee.py 2>&1 | tee out.txt

# Unbuffering python helps
python -u rich_tee.py 2>&1 | tee out.txt

"""


def main():
    import sys
    import time

    print('demo simple progress')
    sys.stdout.flush()
    for i in range(100):
        time.sleep(0.01)
        sys.stdout.write('\rsimple progress {:03d}'.format(i))
        sys.stdout.flush()
    sys.stdout.write('\n')

    print('demo progiter')
    from watch.utils import util_progress
    pman = util_progress.ProgressManager('progiter')
    with pman:
        for _ in pman.progiter(range(1000)):
            time.sleep(0.01)
            sys.stdout.flush()
            ...

    from watch.utils import util_progress
    pman = util_progress.ProgressManager('rich')
    with pman:
        for _ in pman.progiter(range(1000)):
            time.sleep(0.01)
            sys.stdout.flush()
            ...

    print('\ndemo rich progress')
    from rich.live import Live
    from rich import get_console
    from rich.progress import Progress
    from rich.progress import BarColumn, TextColumn

    prog = Progress(
        TextColumn("task"),
        BarColumn(),
    )
    task_id = prog.add_task('task', total=100)

    console = get_console()
    print(f'console={console}')
    print(f'console.is_interactive={console.is_interactive}')
    print(f'console.is_alt_screen={console.is_alt_screen}')
    print(f'console.is_terminal={console.is_terminal}')
    print(f'console.is_dumb_terminal={console.is_dumb_terminal}')
    sys.stdout.flush()

    with Live(prog, console=console):
        for i in range(100):
            time.sleep(0.1)
            prog.update(task_id, advance=1)


if __name__ == '__main__':
    main()
