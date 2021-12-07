pip install johnnydep

johnnydep GDAL==3.3.1 --index-url https://girder.github.io/large_image_wheels --extra-index-url https://pypi.python.org/simple

pip uninstall shapely pygeos GDAL

# Get abidiff
sudo apt install abigail-tools -y
sudo apt install libabigail libabigail-dev -y

pip install GDAL>=3.3.1,!=3.4.0,!=3.3.3 --find-links https://girder.github.io/large_image_wheels -U
pip install pygeos
pip install shapely
#pip install --no-binary shapely shapely


GEOS_CONFIG=/path/to/geos-config

SITE_PACKAGE_DPATH=$(python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib())")
find $SITE_PACKAGE_DPATH -iname "*libgeos_c*" | sort
find $SITE_PACKAGE_DPATH -iname "*libgeos-*" | sort
find $SITE_PACKAGE_DPATH -iname "*geos-config*"
find . -iname "*geos-config*"


abidiff \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/GDAL.libs/libgeos-74ae207e.so.3.9.1dev \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pygeos.libs/libgeos-558e4cfc.so.3.10.0 \
    --stat

abidiff \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/GDAL.libs/libgeos-74ae207e.so.3.9.1dev \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/rasterio.libs/libgeos--no-undefined-dcd1b562.so \
    --stat

abidiff \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/rasterio.libs/libgeos--no-undefined-dcd1b562.so \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/shapely/.libs/libgeos--no-undefined-b94097bf.so \
    --stat



abidiff \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/shapely/.libs/libgeos--no-undefined-b94097bf.so \
    /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/GDAL.libs/libgeos-74ae207e.so.3.9.1dev \
    --stat
    | c++filt


abidiff /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pygeos.libs/libgeos_c-e4a00d47.so.1.16.0 /home/joncrall/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/rasterio.libs/libgeos_c-74dec7a7.so.1.14.2 | c++filt

python -c "from osgeo import gdal; import pyproj"

python -m watch
python -c "from osgeo import gdal; import pyproj"

xdoctest -m watch.utils.util_gis



pip list | grep -i "GDAL\|pygeos\|shapely\|geopandas\|rasterio\|rtree\|pyproj\|fiona"
