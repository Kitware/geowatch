def _check_rasterio_stuff():

    mask = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]).astype(np.uint8) * 255
    import rasterio
    import shapely
    rio_poly = kwimage.Polygon.from_geojson(list(rasterio.features.shapes(mask))[0][0])
    kw_poly = kwimage.Mask(mask, 'c_mask').to_multi_polygon().data[0]

    import kwplot
    kwplot.autompl()
    _, ax = kwplot.imshow(mask, alpha=0.5, show_ticks=True, doclf=1)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.grid(True)

    rio_poly.translate(-0.5).draw(color='orange', fill=0, border=1, ax=ax, linewidth=8)
    kw_poly.draw(color='blue', fill=0, border=1, ax=ax, linewidth=4)
