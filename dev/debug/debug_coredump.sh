__notes__="

https://askubuntu.com/questions/1349047/where-do-i-find-core-dump-files-and-how-do-i-view-and-analyze-the-backtrace-st/1349048#1349048 

ulimit -c

service apport start

cat /proc/sys/kernel/core_pattern

echo '/tmp/core.%t.%e.%p' | tee /proc/sys/kernel/core_pattern

apt update
apt install gdb
apt install valgrind

python -m xdoctest watch/cli/coco_add_watch_fields.py

valgrind --track-origins=yes python -m xdoctest watch/cli/coco_add_watch_fields.py
gdb python /root/watch/vgcore.1118

gdb python 

apt install binutils
objdump -s /root/watch/vgcore.1118

python -m xdoctest --global-exec 'import faulthandler; faulthandler.enable()' watch/cli/coco_add_watch_fields.py
python -m xdoctest --global-exec 'from osgeo import gdal' watch/cli/coco_add_watch_fields.py




"

python -c 'import kwimage, numpy; kwimage.imwrite("foo.tif", kwimage.grab_test_image(), backend="gdal")'

notes2 = '


gdb python
r -c "import torch"
r -c "from osgeo import gdal"



docker run -it gitlab.kitware.com:4567/computer-vision/ci-docker/miniconda3

conda create -y -n py38 python=3.8
conda activate py38

pip install torch
pip install numpy
pip install GDAL==3.3.3 --find-links https://girder.github.io/large_image_wheels
pip install kwimage[headless]
pip install opencv-python-headless
pip install xdoctest


# Works
python -c "from osgeo import gdal; import torch"

# Does not work
python -c "import torch; from osgeo import gdal"


python -c "from osgeo import gdal; import kwcoco; kwcoco.CocoDataset.demo()"
python -c "from osgeo import gdal; import kwcoco; kwcoco.CocoDataset.demo()"

'

python -c "from osgeo import gdal; import xdoctest, kwimage; import kwcoco; xdoctest.doctest_module(kwimage.im_io, command='_gdal_to_numpy_dtype', style='google'); from osgeo import gdal" --network
python -c "import xdoctest, kwimage; import kwcoco; xdoctest.doctest_module(kwimage.im_io, command='_gdal_to_numpy_dtype', style='google'); from osgeo import gdal" 



# Minimal code to reproduce the failure
docker run -it gitlab.kitware.com:4567/computer-vision/ci-docker/miniconda3

# Inside docker run
conda create -y -n py38 python=3.8
conda activate py38

pip install torch
pip install GDAL==3.3.3 --find-links https://girder.github.io/large_image_wheels

# Basic symptom (not a segfault, but it seems related)

# This command works
python -c "from osgeo import gdal; import torch"
# This command fails
python -c "import torch; from osgeo import gdal"


# Slightly more complex - but allows me to reproduce the segfault outside of WATCH
pip install kwimage[headless]
pip install xdoctest

# This will work when we import gdal first
python -c "from osgeo import gdal; import xdoctest, kwimage; xdoctest.doctest_module(kwimage.im_io, command='_gdal_to_numpy_dtype', style='google'); from osgeo import gdal" --network

# This start gives the same no _gdal error, and then crashes with a segfault
python -c "import xdoctest, kwimage; xdoctest.doctest_module(kwimage.im_io, command='_gdal_to_numpy_dtype', style='google'); from osgeo import gdal" --network
