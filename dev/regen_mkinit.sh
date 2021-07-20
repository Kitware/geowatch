# Mkinit commands for generating __init__ files

mkinit -m watch.tasks.fusion --noattrs -w
mkinit -m watch.tasks.fusion.models -w --noattrs
mkinit -m watch.tasks.fusion.datasets -w --nomods

mkinit -m watch.tasks.fusion.methods -w


## Test:
python -c "import watch.tasks.fusion"
python -c "import xdev; xdev.make_warnings_print_tracebacks(); import watch.tasks.fusion"
