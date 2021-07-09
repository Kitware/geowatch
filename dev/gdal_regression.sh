
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

# Slow 14 seconds on namek; 4 seconds on horologic
docker run -it python:3.6 /bin/bash -c 'pip install GDAL==3.4.0 --find-links https://girder.github.io/large_image_wheels --no-index && time python -c "import osgeo; print(osgeo.__file__, osgeo.__version__)"'
