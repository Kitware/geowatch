#!/bin/bash
__doc__='
Only execute the safest autopep8 options

E225 - Fix missing whitespace around operator.
E226 - Fix missing whitespace around arithmetic operator.
E227 - Fix missing whitespace around bitwise/shift operator.
E228 - Fix missing whitespace around modulo operator.
E231 - Add missing whitespace.
E241 - Fix extraneous whitespace around keywords.
E242 - Remove extraneous whitespace around operator.
E251 - Remove whitespace around parameter '=' sign.
E252 - Missing whitespace around parameter equals.
E26  - Fix spacing after comment hash for inline comments.
E265 - Fix spacing after comment hash for block comments.
E266 - Fix too many leading # for block comments.
E27  - Fix extraneous whitespace around keywords.
E301 - Add missing blank line.
E302 - Add missing 2 blank lines.
E303 - Remove extra blank lines.
E304 - Remove blank line following function decorator.
E305 - Expected 2 blank lines after end of function or class.
E306 - Expected 1 blank line before a nested definition.

W291 - Remove trailing whitespace.
W292 - Add a single newline at the end of the file.
W293 - Remove trailing whitespace on blank line.
W391 - Remove trailing blank lines.

chmod +x ~/code/watch/safe_autopep8.sh:
~/code/watch/safe_autopep8.sh ~/code/watch/watch/tasks/fusion/fit_bigvoter.py --diff

Example:

    ~/code/watch/safe_autopep8.sh ~/code/watch/watch/datasets/extern --in-place
    ~/code/watch/safe_autopep8.sh ~/code/watch/watch/tasks/fusion/ --in-place

'

autopep8 --select E225,E226,E227,E228,E231,E251,E252,E301,E302,E303,E304,E305,E306,W291,W292,W293,W391 --recursive $@
