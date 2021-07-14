__doc__='
There seems to be a regression in our internal build of gdal that makes the
import time for osgeo very slow. This script helps test that in a reproducible
way.

Relevant conversation with David Manthey:

Jon Crall, Fri 5:06 PM
I looped through a few different versions of Python and gdal with this method.
3.0.0 seems like the last version with fast import time. 3.1.0 is still fast,
but a noticeably slower, 3.1.1 introduces an additional slowdown, and 3.1.4
introduces the final slowdown that seems to persist through 3.4.0

David Manthey, 8:42 AM
It has something to do with the armadillo library -- if I dont include it, GDAL
starts fast.  armadillo switch from version 9.x to version 10.x in the interval
when it became slower.  Im hunting what changed and why it slows down start up.

David Manthey, 6 min
I still dont know the underlying change that caused this.  armadillo optionally
pulls in OpenBLAS and SuperLU.  If I opt not to pull those in, it is fast.
Even if I pull in older versions of those, it is slow.  OpenBLAS depends on
OpenMP, so maybe the version of that included in the manylinux2010 docker image
is at fault.  For now, Ive rebuilt the GDAL 3.3.1 and latest wheels without
OpenBLAS and SuperLU.  You should be able to try them out (and I expect that
3.3.1 will be fast on your system).

'

# Fast: 0.187 seconds on namek, 0.140s on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==2.4.1 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Fast: 0.118 seconds on namek, 0.058s on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.0.0 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Fast: ~1 second on namek, 0.636s on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.1.0 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Slower 7 seconds on namek, 1.9 seconds on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.1.1 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Slower 7 seconds on namek.
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.1.2 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Slower 7 seconds on namek.
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.1.3 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Slow 14 second on namek, 4 seconds on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.1.4 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Actually seems to point to (3.3.0dev-77d7389), regression exists
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.2.0 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.2.1 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.2.2 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Slow 14 seconds on namek; 4 seconds on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.3.0 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# New version is fast 0.15 seconds on namek
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.3.1 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

# Slow 14 seconds on namek; 4 seconds on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.4.0 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'

