import decimal
import simplejson
import json


class fakefloat(float):
    def __init__(self, value):
        self._value = value
    def __repr__(self):
        return str(self._value)
    def __str__(self):
        return str(self._value)


class DecimalEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/1960516/python-json-serialize-a-decimal-object
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return fakefloat(o)
        return super(DecimalEncoder, self).default(o)


def defaultencode(o):
    if isinstance(o, decimal.Decimal):
        # Subclass float with custom repr?
        return fakefloat(o)
    raise TypeError(repr(o) + " is not JSON serializable")


def main():
    from watch.utils import util_gis
    from watch.geoannots import geomodels
    dpath = '/media/joncrall/flash1/smart_data_dvc/submodules/annotations/region_models'
    region_model_fpaths = util_gis.coerce_geojson_paths(dpath)
    _iter = iter(region_model_fpaths)
    for fpath in _iter:
        if fpath.stem in {'AE_C002', 'AE_C003', 'BR_T001', 'BR_T002', 'PY_C001'}:
            continue
        region_model = geomodels.RegionModel.coerce(fpath, parse_float=decimal.Decimal)
        # region_model._validate_parts()
        try:
            region_model.validate()
        except Exception:
            fix_region_model(region_model)
            region_model.validate()
            if '_C' in fpath.stem:
                # Try to minimize the diff by outputing in a similar style
                new_text = special_dumps(region_model)
            else:
                # T&E regions seem to be normal json outputs
                new_text = simplejson.dumps(region_model, indent='    ')
            fpath.write_text(new_text)

        # old_text = fpath.read_text()
        # print(new_text.split('\n')[:6])
        # print(old_text.split('\n')[:6])
        # import xdev
        # print(xdev.difftext(old_text, new_text, colored=True))


def special_dumps(region_model):
    # import ubelt as ub
    t = region_model.copy()
    features = t.pop('features', None)
    lines = ['{']
    def oneline_dict(val):
        # import json
        # text = json.dumps(val, cls=DecimalEncoder)
        text = simplejson.dumps(val)
        # cls=DecimalEncoder)
        # text = ub.urepr(v, trailsep=False, nl=0)
        text = text.replace('{', '{ ')
        text = text.replace('}', ' }')
        text = text.replace('[', '[ ')
        text = text.replace(']', ' ]')
        # text = text.replace("'", '"')
        return text

    for k, v in t.items():
        text = oneline_dict(v)
        lines.append(f'"{k}": ' + text + ',')

    last_feat = features[-1]

    lines.append('"features": [')
    for feat in features:
        text = oneline_dict(feat)
        if feat is last_feat:
            lines.append(text)
        else:
            lines.append(text + ',')

    lines.append(']')
    lines.append('}')
    text = '\n'.join(lines) + '\n'
    return text


def fix_region_model(region_model):
    import mgrs
    import kwimage
    (lon,), (lat,) = region_model.geometry.centroid.xy
    mgrs_code = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)

    import kwplot
    import ubelt as ub
    kwplot.autompl()

    def draw_bad_region(region_model, region_poly):
        fig = kwplot.figure(fnum=1)
        ax = fig.gca()
        ax.cla()
        region_poly.draw(setlim=1, ax=ax, alpha=0.5)
        ax.set_title('Region Geometry: {}'.format(region_model.region_id))
        dpath = (ub.Path.home() / 'tmpfig/bad_regions').ensuredir()
        fig.savefig(dpath / region_model.region_id + '_bounds.png')

    def draw_bad_site(region_model, feat, site_poly):
        fig = kwplot.figure(fnum=1)
        ax = fig.gca()
        ax.cla()
        site_id = feat['properties']['site_id']
        site_poly.draw(setlim=1, ax=ax, alpha=0.5)
        ax.set_title('Site Geometry: {}'.format(site_id))
        dpath = (ub.Path.home() / 'tmpfig/bad_sites').ensuredir()
        fig.savefig(dpath / site_id + '_bounds.png')

    if region_model.geometry.geom_type == 'MultiPolygon':
        region_geom = region_model.geometry
        region_poly = kwimage.MultiPolygon.from_shapely(region_geom)
        parts = list(region_geom.geoms)
        assert len(parts) == 1
        draw_bad_region(region_model, region_poly)
        poly = kwimage.Polygon.from_shapely(parts[0])
        print('Fix region header geom')
        region_model.header['geometry'] = poly.to_geojson()

    if region_model.geometry.geom_type == 'Polygon':
        for ring in region_model.header['geometry']['coordinates']:
            for pt in ring:
                if len(pt) != 2:
                    assert pt[2] == 0
                    pt[:] = pt[0:2]

    for feat in region_model.features:
        props = feat['properties']
        props['mgrs'] = mgrs_code

        if 'score' in props:
            if isinstance(props['score'], str):
                props['score'] = float(props['score'])

        if 'model_cont' in props:
            props['model_content'] = props.pop('model_cont')

        if 'model_content' in props:
            if props['model_content'] is None:
                props['model_content'] = 'annotation'

        if 'originator' in props:
            if props['originator'] == 'imerit':
                props['originator'] = 'iMERIT'
        if 'orginator' in props:
            props['originator'] = props.pop('orginator')

        props['type'] = props['type'].replace(' ', '')

        if props['type'] == 'site_summary':
            if props['version'] is None:
                props['version'] = region_model.header['properties']['version']
            if 'version' in props:
                props['version'] = props['version'].strip()
            props['site_id'] = props['site_id'].replace(' ', '')
            props['status'] = props['status'].strip().lower()
            if 'validated' in props:
                props['validated'] = props['validated'].strip()
            if feat['geometry']['type'] == 'MultiPolygon':
                print('Fix site summary geom')
                site_poly = kwimage.MultiPolygon.coerce(feat['geometry'])
                draw_bad_site(region_model, feat, site_poly)
                parts = list(site_poly.to_shapely().geoms)
                assert len(parts) == 1
                poly = kwimage.Polygon.from_shapely(parts[0])
                feat['geometry'] = poly.to_geojson()
            if 'annotation_cache' not in props:
                props['annotation_cache'] = {}

            if feat['geometry']['type'] == 'Polygon':
                for ring in feat['geometry']['coordinates']:
                    for pt in ring:
                        if len(pt) != 2:
                            assert pt[2] == 0
                            pt[:] = pt[0:2]

            if feat['geometry']['type'] == 'MultiPolygon':
                for poly in feat['geometry']['coordinates']:
                    for ring in poly:
                        for pt in ring:
                            if len(pt) != 2:
                                raise Exception
    # region_model._validate_parts()
