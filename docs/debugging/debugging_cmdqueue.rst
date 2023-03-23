Debugging Cmd-Queue Powered Scripts
-----------------------------------

When running a cmd-queue based script it is important to understand that it is
primarilly a mechanism for building a sequence of commands to accomplish a
task. The execution of those commands is secondary.

Idomatically, most cmd-queue scripts will have the options: ``--run`` and
``--print_commands``. By setting  ``--run=0`` and  ``--print_commands=1`` it
will simply show you the commands that it would execute in order. You can
manually execute these commands step-by-step to diagnose problems.

Also there is usually a ``--backend`` option. Often this is set to
``--backend=tmux``, but if you use ``--backend=serial`` it will run everything
in the foreground terminal session allowing you to see everything happen one at
a time. This is very similar to running commands one-by-one, but there are
minor differences so its important to be aware of both strategies.

When using the tmux cmd-queue backend, if jobs are failing you can use tmux to
inspect the status of the jobs. Use ``tmux a`` to attach to one of the tmux
sessions, and you can use ``CTRL-b`` followed by ``s`` to show an interactive
overview of all sessions. You can navigate to inspect what each session is
doing. You will also notice that each tmux session has an associate script that
it sources at the very beginning. You can look at that to see the details of
what is happening in each tmux session (Note: there is a lot of bash
boilerplate in these files, and most of it can be ignored: look for the part of
the code with the ``# command`` comment).
