#!/bin/bash
# Make mlops not take up so much RAM
#
python -c "if 1:
    import ubelt as ub
    import psutil

    a = 0
    import time
    now = time.time()

    candidates = []
    for p in psutil.process_iter():
        existing_time = (now - p.create_time())
        if existing_time > 10:
            if p.name() == 'python':
                cmdline = p.cmdline()
                if cmdline[0:2] == ['python', '-m']:
                    if any('watch.' in x for x in cmdline):
                        candidates.append(p)

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='serial')
    for p in candidates:
        queue.submit(f'sudo renice -n 1 -p {p.pid}')

    queue.print_commands()
    queue.run(system=True)
"
#renice
