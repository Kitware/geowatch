def associate_images(dset1, dset2):
    """
    Hueristic for getting pairs of images that correspond between two datasets
    """

    dset1_img_names = set(dset1.index.name_to_img)
    dset2_img_names = set(dset2.index.name_to_img)
    common_names = dset1_img_names & dset2_img_names
    dset1_missing_img_names = dset1_img_names - common_names
    dset2_missing_img_names = dset2_img_names - common_names
    report = {}
    report.update({
        'num_name_common': len(common_names),
        'num_name_missing1': len(dset1_missing_img_names),
        'num_name_missing2': len(dset2_missing_img_names),
    })

    dset1_img_file_names = set(dset1.index.file_name_to_img)
    dset2_img_file_names = set(dset2.index.file_name_to_img)
    common_file_names = dset1_img_file_names & dset2_img_file_names
    dset1_missing_img_file_names = dset1_img_file_names - common_file_names
    dset2_missing_img_file_names = dset2_img_file_names - common_file_names
    report.update({
        'num_file_name_common': len(common_file_names),
        'num_file_name_missing1': len(dset1_missing_img_file_names),
        'num_file_name_missing2': len(dset2_missing_img_file_names),
    })

    gids1 = []
    gids2 = []
    for name in common_names:
        img1 = dset1.index.name_to_img[name]
        img2 = dset2.index.name_to_img[name]
        gids1.append(img1['id'])
        gids2.append(img2['id'])

    if 0:
        import kwimage
        @ub.memoize
        def img_hueristic_info(dset, gid):
            img = dset.index.imgs[gid]
            video = dset.index.videos[img['video_id']]
            utm_corners = kwimage.Polygon.coerce(np.array(img['utm_corners']))

            info = {
                'vidname': video['name'],
                'dsize': (img['width'], img['height']),
                'date_captured': img['date_captured'],
                'utm_corners': utm_corners.to_shapely(),
            }

            for aux in img.get('auxiliary', []):
                if aux['channels'] == 'coastal':
                    info['parent_coastal'] = aux['parent_file_name']
            return info

        # Debug if names just changes slightly
        dset1_missing_img_names = list(dset1_missing_img_names)
        dset2_missing_img_names = list(dset2_missing_img_names)
        import xdev
        import numpy as np
        import kwarray
        distances = xdev.edit_distance(dset1_missing_img_names,
                                       dset2_missing_img_names)

        # Add in extra constraints
        distances = np.array(distances).astype(np.float32)
        flags = distances < 50
        distances[~flags] = np.inf
        ious = {}
        for idx1, idx2 in zip(*np.where(flags)):
            name1 = dset1_missing_img_names[idx1]
            name2 = dset2_missing_img_names[idx2]
            gid1 = dset1.index.name_to_img[name1]['id']
            gid2 = dset2.index.name_to_img[name2]['id']

            dist = distances[idx1, idx2]
            info1 = img_hueristic_info(dset1, gid1)
            info2 = img_hueristic_info(dset2, gid2)

            flag = 1
            if info1['date_captured'] == info2['date_captured']:
                isect = info1['utm_corners'].intersection(info2['utm_corners'])
                union = info1['utm_corners'].union(info2['utm_corners'])
                iou = isect.area / union.area
                ious[(idx1, idx2)] = iou
                if iou > 0.99:
                    flag = 0

            if flag:
                distances[idx1, idx2] = np.inf

        assignment, _ = kwarray.mincost_assignment(distances)
        candidates = []
        for idx1, idx2 in assignment:
            dist = distances[idx1, idx2]
            name1 = dset1_missing_img_names[idx1]
            name2 = dset2_missing_img_names[idx2]
            iou = ious[(idx1, idx2)]
            candidates.append((dist, iou, name1, name2))
        print(len(candidates))
        print(ub.repr2(sorted(candidates)[0:1000], compact_brace=10, nl=3))

    # identifier_to_gid = []
    # for img in dset1.index.imgs.values():
    #     name = img.get('name', None)
    #     file_name = img.get('file_name', None)
    #     gid = img['id']
    return gids1, gids2, report
