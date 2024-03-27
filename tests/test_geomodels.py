import pytest
import ubelt as ub
from geowatch.geoannots.geomodels import RegionModel
from geowatch.geoannots.geomodels import RegionHeader


def test_dump_load():
    from geowatch.geoannots import geomodels
    # Create arguments to the script
    dpath = ub.Path.appdir('geowatch/tests/geomodels/test_dump_load').delete().ensuredir()

    region = geomodels.RegionModel.random()
    fpath1 = dpath / 'region1.geojson'
    fpath2 = dpath / 'region2.geojson'

    region.dump(fpath1)

    with open(fpath2, 'w') as file:
        region.dump(file)

    assert fpath1.exists()
    assert fpath2.exists()

    recon1 = geomodels.RegionModel.load(fpath1)

    with open(fpath2, 'r') as file:
        recon2 = geomodels.RegionModel.load(file)

    assert recon1 == region
    assert recon2 == region


def test_dumps_loads():
    from geowatch.geoannots import geomodels
    region = geomodels.RegionModel.random()
    text = region.dumps()
    recon = region.loads(text)
    assert recon == region


def test_infer_region_header():
    # Case: RegionModel has None header info
    self = RegionModel.random()
    self.header['properties']['mgrs'] = None
    self.header['geometry'] = None
    assert self.header is not None
    self.infer_header()
    assert self.header is not None
    assert self.header['geometry'] is not None
    assert self.header['properties']['mgrs'] is not None

    # Case: RegionModel has missing header info
    self = RegionModel.random()
    del self.header['properties']['mgrs']
    del self.header['geometry']
    assert self.header is not None
    self.infer_header()
    assert self.header is not None
    assert self.header['geometry'] is not None
    assert self.header['properties']['mgrs'] is not None

    # Case: RegionModel has no header
    self = RegionModel.random()
    self.features.remove(self.header)
    assert self.header is None
    # Infer the header using site summaries
    self.infer_header()
    assert self.header is not None
    assert self.header['properties']['region_id'] is None, 'should not be able to infer region id here'

    # Case: RegionModel has no header, and a partial header is given
    self = RegionModel.random()
    self.features.remove(self.header)
    assert self.header is None
    # Infer the header using site summaries
    self.infer_header(RegionHeader(properties={'region_id': 'foobar'}))
    assert self.header is not None
    assert self.header['properties']['region_id'] == 'foobar'

    # Case: RegionModel has a header and a partial header is given
    self = RegionModel.random()
    with pytest.raises(ValueError):
        self.infer_header(RegionHeader(properties={'region_id': 'foobar'}))

    # Case: RegionModel has a header
    self = RegionModel.random()
    old_header = self.header
    old_header_str = str(old_header)
    self.infer_header()
    assert self.header is old_header, 'should not change anything'
    assert old_header_str == str(self.header), 'should not change anything'


def test_infer_region_header_conflict_mgrs():
    self1 = RegionModel.random(rng=123)
    self2 = RegionModel.random(rng=321)

    # Make a region model that spans multiple MGRS tiles
    self1.features.extend(list(self2.site_summaries()))

    # Remove the header
    self1.features.remove(self1.header)

    # Infered header should handle conflicting mgrs gracefully
