pip install johnnydep

johnnydep GDAL==3.3.1 --index-url https://girder.github.io/large_image_wheels --extra-index-url https://pypi.python.org/simple

pip uninstall shapely pygeos GDAL

pip install GDAL>=3.3.1,!=3.4.0,!=3.3.3 --find-links https://girder.github.io/large_image_wheels -U
pip install pygeos
pip install shapely
#pip install --no-binary shapely shapely


GEOS_CONFIG=/path/to/geos-config

SITE_PACKAGE_DPATH=$(python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib())")
find $SITE_PACKAGE_DPATH -iname "*libgeos*"
find $SITE_PACKAGE_DPATH -iname "*geos-config*"
find . -iname "*geos-config*"

python -c "from osgeo import gdal; import pyproj"

python -m watch
python -c "from osgeo import gdal; import pyproj"

xdoctest -m watch.utils.util_gis



pip list | grep -i "GDAL\|pygeos\|shapely\|geopandas\|rasterio\|rtree\|pyproj\|fiona"
