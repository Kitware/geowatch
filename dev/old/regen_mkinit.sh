# Mkinit commands for generating __init__ files

mkinit -m watch.tasks.fusion --noattrs -w
mkinit -m watch.tasks.fusion.models -w --noattrs
mkinit -m watch.tasks.fusion.datamodules -w --nomods

mkinit -m watch.tasks.fusion.methods -w


#  
mkinit -m watch.gis --noattrs --lazy
mkinit -m watch.utils --noattrs --lazy
mkinit -m watch.datasets --noattrs --lazy
mkinit -m watch.demo --noattrs --lazy
mkinit -m watch.cli --noattrs --lazy

mkinit -m watch.tasks --noattrs --lazy

mkinit -m watch.tasks.landcover --noattrs --lazy

mkinit -m watch.tasks.uky_temporal_prediction --noattrs --lazy
mkinit -m watch.tasks.uky_temporal_prediction.spacenet --noattrs --lazy
mkinit -m watch.tasks.uky_temporal_prediction.models --noattrs --lazy

## Test:
EAGER_IMPORT=1 python -c "import watch.tasks.fusion"
python -c "import xdev; xdev.make_warnings_print_tracebacks(); import watch.tasks.fusion"
