# Mkinit commands for generating __init__ files

mkinit -m watch.tasks.fusion --noattrs -w
mkinit -m watch.tasks.fusion.models -w --noattrs
mkinit -m watch.tasks.fusion.datasets -w --noattrs

mkinit -m watch.tasks.fusion.methods -w


## Test:
python -c "import watch.tasks.fusion"
