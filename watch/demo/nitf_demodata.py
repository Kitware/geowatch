"""
Access sample NITF images from [1]_ for testing and demo purposes.

References:
    .. [1] https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/scen_2_1.html
"""
import ssl
import ubelt as ub


DEFAULT_KEY = 'i_3034c.ntf'  # use a small image as the default


_TEST_IMAGES = [
    {'key': 'ns3114a.nsf', 'sha512': '5605cd0b0187900c1b43130bf157c6ed', 'size_bytes': 680, 'enabled': False},
    {'key': 'i_3034c.ntf', 'sha512': '5f42ab1034f20756bdf15043b58c828f', 'size_bytes': 933},
    {'key': 'ns3034d.nsf', 'sha512': 'aaf65232611bdc53934fa7f58790529a', 'size_bytes': 937},
    {'key': 'i_3034f.ntf', 'sha512': '2c2c3a918fe2805dc78a090164d671dc', 'size_bytes': 948},
    {'key': 'i_3051e.ntf', 'sha512': 'a67e9c4172310faaadf34a0e6ce01a72', 'size_bytes': 1436, 'enabled': False},
    {'key': 'i_3052a.ntf', 'sha512': 'bb96983f58b3ec5891ef2f096d97b84e', 'size_bytes': 1520, 'enabled': False},
    {'key': 'ns3051v.nsf', 'sha512': 'ca438abdb6e67776a99bd14cc4154fa6', 'size_bytes': 1592, 'enabled': False},
    {'key': 'i_3063f.ntf', 'sha512': '68bb22e59739f31c1f5cc66cc5ac415b', 'size_bytes': 1596, 'enabled': False},
    {'key': 'ns3063h.nsf', 'sha512': '7b85023c46414b4b36c2a2795e59e2bd', 'size_bytes': 1606, 'enabled': False},
    {'key': 'i_3060a.ntf', 'sha512': '9e42e2d8e9fe6c07380501fcf5339c3e', 'size_bytes': 1624, 'enabled': False},
    {'key': 'i_3068a.ntf', 'sha512': 'aa4913146f4512b41ba2d6717a375a94', 'size_bytes': 1658, 'enabled': False},
    {'key': 'ns3061a.nsf', 'sha512': '86013922c8caf816943296a8b3dcbbd5', 'size_bytes': 1668, 'enabled': False},
    {'key': 'ns3059a.nsf', 'sha512': 'c84eed6be08e60921920b4bfe621e0b6', 'size_bytes': 1766, 'enabled': False},
    {'key': 'i_3114e.ntf', 'sha512': '709933f92fb86a0022fed6f4edd32626', 'size_bytes': 1776, 'enabled': False},
    {'key': 'ns3073a.nsf', 'sha512': 'a70636dcc455307f4cea235fb05d7ddd', 'size_bytes': 1854, 'enabled': False},
    {'key': 'ns3101b.nsf', 'sha512': '7535d935a56ac1d1a7ac85e3d6c03983', 'size_bytes': 2144, 'enabled': False},
    {'key': 'i_3025b.ntf', 'sha512': '582bb41bd308535ef81cc822ad68d90f', 'size_bytes': 2199},
    {'key': 'i_3076a.ntf', 'sha512': '97200f22508e02e130799fedc64bf2a2', 'size_bytes': 2246, 'enabled': False},
    {'key': 'i_3018a.ntf', 'sha512': '6e8539c992f289ed7292401ce4a394a1', 'size_bytes': 3564},
    {'key': 'ns3050a.nsf', 'sha512': '1a87ae99cd1952360159a43e9f0fd269', 'size_bytes': 4071},
    {'key': 'ns3022b.nsf', 'sha512': '876d0e4435132cb667ef7f7b662e4ac2', 'size_bytes': 4502},
    {'key': 'i_3015a.ntf', 'sha512': '6f7e8c8e5c93f3bfe4795dafb194cd13', 'size_bytes': 6074},
    {'key': 'ns3038a.nsf', 'sha512': '77e3270feb072d8173547cde48b8128c', 'size_bytes': 7018},
    {'key': 'ns3417c.nsf', 'sha512': 'd28b53eee6afb3eee010447dbf3b952a', 'size_bytes': 8836, 'enabled': False},
    {'key': 'ns3010a.nsf', 'sha512': 'dddede2a6bc30d9dad64ee062668b14a', 'size_bytes': 10711},
    {'key': 'ns3017a.nsf', 'sha512': 'f074fe40a4ddfee2627b2561f6c8f360', 'size_bytes': 11052},
    {'key': 'i_3008a.ntf', 'sha512': '3a6a693f2ad48de56494017d0cb7cd5e', 'size_bytes': 21212},
    {'key': 'ns3114i.nsf', 'sha512': '3381b75a976da70022eb1564e07b5a73', 'size_bytes': 26664, 'enabled': False},
    {'key': 'i_3201c.ntf', 'sha512': '55789ec5e5cf772c7b186bd6a0b57d6b', 'size_bytes': 48497},
    {'key': 'ns3033b.nsf', 'sha512': '85633cb72f4c10fcf8c98f6caffd5b9b', 'size_bytes': 53061},
    {'key': 'i_3041a.ntf', 'sha512': 'bdb9215995a602231003caf7aca0d3b7', 'size_bytes': 64682},
    {'key': 'i_3113g.ntf', 'sha512': '143d866fb1bc4ee5e16b458f5e8c1e24', 'size_bytes': 70765, 'enabled': False},
    {'key': 'ns3301j.nsf', 'sha512': '813992b3600624b9748847180a007889', 'size_bytes': 95605},
    {'key': 'ns3005b.nsf', 'sha512': '49702adf3d64648b8a29b7ef7ca7a132', 'size_bytes': 129920},
    {'key': 'i_3301h.ntf', 'sha512': 'faf89333207b840e11ac06b44b0ff09c', 'size_bytes': 140837},
    {'key': 'ns3201a.nsf', 'sha512': 'ab5cc412ddfba05463f5d84593c726da', 'size_bytes': 170590},
    {'key': 'ns3302a.nsf', 'sha512': 'c3445af2d5f529e42be7d66bde52311c', 'size_bytes': 197477},
    {'key': 'ns3310a.nsf', 'sha512': '62cec9952b7d98de3f550e410099ab93', 'size_bytes': 197477},
    {'key': 'ns3301e.nsf', 'sha512': '2659e9f504b2427ebe064fa7bbfceeea', 'size_bytes': 197504},
    {'key': 'ns5600a.nsf', 'sha512': 'f7e27013e66568e02b0c4039120ff155', 'size_bytes': 219974},
    {'key': 'i_3128b.ntf', 'sha512': 'e4fa986099dcfdaaa74e66f68f48abbc', 'size_bytes': 248762},
    {'key': 'i_3004g.ntf', 'sha512': '5285370808d15ffcbc57a53105ff7a1b', 'size_bytes': 263047, 'has_crs': True, 'has_rpc': False},
    {'key': 'ns3004f.nsf', 'sha512': '0bfbd6d378dfd0f0e8cad703498fa6c8', 'size_bytes': 263047, 'has_crs': True, 'has_rpc': False},
    {'key': 'ns3090i.nsf', 'sha512': '9be0244363bbe63c71b55217a90c346b', 'size_bytes': 264083, 'enabled': False},
    {'key': 'i_3090m.ntf', 'sha512': '30017da0f1c9c41130e79c18f99aba97', 'size_bytes': 264083},
    {'key': 'i_3090u.ntf', 'sha512': 'a3f57adee4e5e25f03131891e6948da4', 'size_bytes': 264091},
    {'key': 'ns3090q.nsf', 'sha512': '3c4f69ed8298f40e9a4845ae8321b056', 'size_bytes': 264091},
    {'key': 'ns3361c.nsf', 'sha512': '67123e051a8d02909e6b53b703330db9', 'size_bytes': 264592, 'has_crs': True, 'has_rpc': False},
    {'key': 'ns3321a.nsf', 'sha512': '5a82d19b8a903537bee14e4d3a7cdc55', 'size_bytes': 281130},
    {'key': 'ns3118b.nsf', 'sha512': 'c95f99cf7bdd0ae2f802a282b772e339', 'size_bytes': 362407},
    {'key': 'i_5012c.ntf', 'sha512': 'de025f0a3da3b4f9e7279a45b7cd02e5', 'size_bytes': 594601},
    {'key': 'ns3304a.nsf', 'sha512': '5b0418c2f2cae7038eebdf3764514129', 'size_bytes': 701687, 'enabled': False},
    {'key': 'i_3309a.ntf', 'sha512': '099d017dbfee8f703c4e6d76b9810a0e', 'size_bytes': 722432},
    {'key': 'i_3001a.ntf', 'sha512': 'e5cdb23c612cbe28f0b994b69285aa49', 'size_bytes': 1049479, 'has_crs': True, 'has_rpc': False},
    {'key': 'ns3119b.nsf', 'sha512': '14b7ee574116538f88b0a7aa7758c88b', 'size_bytes': 1051108, 'enabled': False},
    {'key': 'i_3430a.ntf', 'sha512': 'c52c72adf654c5e02fb5bb19174c5e99', 'size_bytes': 1573707},
    {'key': 'i_3301k.ntf', 'sha512': 'daf36eec7d4eb4be006c6f098d0efea8', 'size_bytes': 1770388},
    {'key': 'ns3301b.nsf', 'sha512': '9a51ed8ec667b2618d8d1f357a84b1e5', 'size_bytes': 1770388},
    {'key': 'i_3301c.ntf', 'sha512': '07de02322f46f9ba90318fb97b8ca759', 'size_bytes': 1770460},
    {'key': 'i_3405a.ntf', 'sha512': 'fde7c7c42b42ce3bf6f6193f9108edf1', 'size_bytes': 2097995},
    {'key': 'i_3450c.ntf', 'sha512': 'bea077c7d21d85e69f87fb4e565388d6', 'size_bytes': 2097995},
    {'key': 'ns3450e.nsf', 'sha512': 'ffa6e6923a2573d0aefb52908f078762', 'size_bytes': 2097995},
    {'key': 'ns3437a.nsf', 'sha512': '85a1bdcda593373326e5a0a498a79f95', 'size_bytes': 2656587, 'enabled': False},
    {'key': 'i_3301a.ntf', 'sha512': 'f669994ddaab9f08e1064f24dc0e1580', 'size_bytes': 3146597, 'enabled': False},
    {'key': 'i_3117ax.ntf', 'sha512': 'c9ab95cc2cd4711677a0cce78122b703', 'size_bytes': 3489726},
    {'key': 'i_3303a.ntf', 'sha512': 'e13a7aa57775b71d10e5a420e7a13214', 'size_bytes': 4195147, 'enabled': False},
    {'key': 'i_3311a.ntf', 'sha512': '74e58d3b921555544678ae3488cb6a35', 'size_bytes': 4679168},
    {'key': 'ns3229b.nsf', 'sha512': '1004f108fd4b2841d3f7362ae4077e28', 'size_bytes': 5659571},
    {'key': 'ns3228b.nsf', 'sha512': '2e9445fa3876e2e09aaa362f25f3018d', 'size_bytes': 6292578},
    {'key': 'i_3228c.ntf', 'sha512': '2b059b564911c0f7c93b3ab3e332e480', 'size_bytes': 6292578},
    {'key': 'i_3228e.ntf', 'sha512': '3d9814143e2281241923904c4132859c', 'size_bytes': 6292578},
    {'key': 'ns3228d.nsf', 'sha512': 'a9b4ebab56101935eccc0b27b1060810', 'size_bytes': 6292578},
]


_FNAME_TO_INFO = {row['key']: row for row in _TEST_IMAGES if row.get('enabled', True)}


_FNAME_TO_DESC = {
    'i_3001a.ntf': 'Can the system handle an uncompressed 1024x1024 8-bit mono image and file contains GEO data? (AIRFIELD)',
    'i_3004g.ntf': 'Checks a system to see how it applies GEO data around 00, 180.',
    'i_3008a.ntf': 'Checks a JPEG-compressed, 256x256 8-bit mono image, Q4, COMRAT 00.4 with general purpose tables embedded. File also contains image comments. (TANK)',
    'i_3015a.ntf': 'Can the system handle a JPEG-compressed 256x256 8-bit mono image with comment in the JPEG stream before frame marker? (TANK)',
    'i_3018a.ntf': 'Checks a JPEG-compressed 231x191 8-bit mono image with a corrupted restart marker occurring too early. (BLIMP)',
    'i_3025b.ntf': 'Checks to see if a viewer can read a JPEG stream with fill bytes (FF) in the JPEG stream before FFD8. (LINCOLN)',
    'i_3034c.ntf': 'Checks a 1-bit RGB/LUT with an arrow, the value of 1 mapped to green and the background value of 0 mapped to red, and no mask table.',
    'i_3034f.ntf': 'Checks a 1-bit RGB/LUT (green arrow) with a mask table (pad pixels having value of 0x00) and a transparent pixel value of 1 being mapped to green by the LUT.',
    'i_3041a.ntf': 'Checks a bi-level compressed at 2DS 512x512 FAX image. (SHIP)',
    'i_3051e.ntf': 'Checks to see if a system can render CGM Text in the proper location.',
    'i_3052a.ntf': 'Checks to see if the system renders a basic Circle.',
    'i_3060a.ntf': 'Checks for rendering CGM polylines (types 1 through 5.)',
    'i_3063f.ntf': 'Checks for rendering CGM polygons with hatch style 5.',
    'i_3068a.ntf': 'Checks for rendering CGM rectangles with starting point in Lower Right of rectangle.',
    'i_3076a.ntf': 'Checks for rendering various CGM elliptical arc cords.',
    'i_3090m.ntf': 'CIRARCC5 checks for proper interpretation of upper left VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'i_3090u.ntf': 'CIRARCCD checks for proper interpretation of upper right VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'i_3113g.ntf': 'Can system display a Low Bite Rate (LBR) file with an uncompressed image overlay?',
    'i_3114e.ntf': 'Checks to see if the system recognizes all UT1 values 0xA0 to 0xFF.',
    'i_3117ax.ntf': 'Can the system render an NSIF file having the maximum total bytes in 32 text segments each of 99,998 bytes with an image segment? (Text shows 1 of 32 identical text segments.)',
    'i_3128b.ntf': 'This file contains PIAE TREs version 2.0 to include three PEA TREs. If the system supports PIAE TREs, can they find each TRE to include all 3 PEA TREs?',
    'i_3201c.ntf': 'Checks a systems ability to handle a single block IMODE R image, 126x126',
    'i_3228c.ntf': 'MS IMODE P RGB, multi-blocked image, not all bands displayed.',
    'i_3228e.ntf': 'MS IMODE R RGB, multi-blocked image, not all bands displayed.',
    'i_3301a.ntf': 'Checks an uncompressed 1024x1024 24-bit multi-blocked (IMode-S) color image. (HELO)',
    'i_3301c.ntf': 'Checks an IMODE S image with a data mask subheader, the subheader with padded pixels, having a pad pixel value of 0x00 displaying as transparent, 3x3 blocks.',
    'i_3301h.ntf': 'Can the system display a multi block 6x6 IMODE R image and 216x216?',
    'i_3301k.ntf': 'Checks an IMODE R image with a data mask subheader, with padded pixels, a pad pixel value of 0x00 displaying as transparent, and 3x3 blocks.',
    'i_3303a.ntf': 'Can the system display an uncompressed 2048x2048 8-bit multi-blocked mono image? (CAMELS)',
    'i_3309a.ntf': 'Can the system display a JPEG-compressed 2048x2048 8-bit multi-blocked (256x256) mono image w/QFAC=3, RSTI=16, and IMODE=B? (CAMELS)',
    'i_3311a.ntf': 'Can the system display a JPEG 2048x2048 24-bit PI block color w/QFAC=3,RSTI=32,IMODE=P, blocked (512x512)? (JET)',
    'i_3405a.ntf': 'Can the system handle a multi-blocked 1024x1024 image with 11/16 (ABPP=11, NBPP=16)? (AIRSTRIP)',
    'i_3430a.ntf': 'Can the system handle an NSIF file with an uncompressed image with 12-bit back to back data, ABPP = 12, and NBPP = 12?',
    'i_3450c.ntf': 'Can the system read a 32-bit real image?',
    'i_5012c.ntf': 'Can the system handle an NSIF file with 100 images, 100 symbols and 32 text elements, images 1, 25, 50, 75 and 100 attached to "000", symbol 12 and text 29 attached to image 25, symbol 32 and text 30 attached to image 50, symbol 86 and text 31 attached to image 75, symbol 90 and text 32 attached to image 100, and all other segments attached to image 1?',
    'ns3004f.nsf': 'Checks a system to see how it applies GEO data around 00, 000.',
    'ns3005b.nsf': 'Checks a JPEG-compressed 1024x1024 8-bit mono image compressed with visible 8-bit tables and COMRAT 01.1. (AIRFIELD)',
    'ns3010a.nsf': 'Can the system handle a JPEG-compressed 231x191 8-bit mono image that is non-divide by 8, and file also contains image comments? (BLIMP)',
    'ns3017a.nsf': 'Checks a JPEG-compressed 231x191 8-bit mono image with a corrupted restart marker occurring too late. (BLIMP)',
    'ns3022b.nsf': 'Checks a JPEG-compressed 181 x 73 8-bit mono image with split Huffman tables 1 DC 1 AC having separate marker for each. (JET)',
    'ns3033b.nsf': 'Checks a JPEG-compressed 512x512 8-bit mono image with APP7 marker in JPEG stream. (LENNA)',
    'ns3034d.nsf': 'Checks a 1-bit mono with mask table having (0x00) black as transparent with white arrow.',
    'ns3038a.nsf': 'Checks all run lengths on a bi-level compressed at 1D and 1024x1024 FAX imagery. (SEMAPHORE)',
    'ns3050a.nsf': 'Checks all run lengths on a bi-level compressed at 2DH and 1024x1024 FAX imagery. (SEMAPHORE)',
    'ns3051v.nsf': 'Checks to see if the system can render CGM polygon sets properly and two polygons that do not intersect.',
    'ns3059a.nsf': 'Checks for rendering CGM ellipses with edge width of 50.',
    'ns3061a.nsf': 'Checks an IMODE S image with a data mask subheader, the subheader with padded pixels, having a color value of 0x00, 0x00, 0x00 displaying as transparent, and 3x3 blocks.',
    'ns3063h.nsf': 'Checks for rendering CGM polygons with hatch style 1 with auxiliary color.',
    'ns3073a.nsf': 'Checks for rendering various CGM circular arcs.',
    'ns3090i.nsf': 'CIRARCC1 checks for proper interpretation of lower left VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'ns3090q.nsf': 'CIRARCC9 checks for proper interpretation of lower right VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'ns3101b.nsf': 'Checks to see what CGM fonts are supported by the system. The display image is shown with limited font support.',
    'ns3114a.nsf': 'Can the render an NSIF file with a single (STA) text segment with only one byte of data?',
    'ns3114i.nsf': 'Can the system render a U8S character set (this text segment is in an HTML format)? (To verify data, ensure your web browser is set to properly display Unicode UT8-F.)',
    'ns3118b.nsf': 'Can the system render an embedded MTF file is the second text segment. Text shows MTF text segment.Can the system render an embedded MTF file that is the second text segment? (Text shows MTF text segment.)',
    'ns3119b.nsf': 'Can the system render the maximum CGM total bytes for a clevel 3 file (total bytes 1,048,576 in 8 CGM segments)?',
    'ns3201a.nsf': 'Checks a systems ability to handle an RGB/LUT. (LUT has 128 entries.)',
    'ns3228b.nsf': 'MS IMODE S RGB, multi-blocked image, not all bands displayed.',
    'ns3228d.nsf': 'MS IMODE B RGB, multi-blocked image, not all bands displayed.',
    'ns3229b.nsf': 'Nine band MS image, PVTYPE=SI, ABPP=16 in NBPP=16, IMODE B. Band 1, 2 & 3 have been enhanced for viewing, image is naturally dark.',
    'ns3301b.nsf': 'Checks an IMODE B image with a data mask subheader, the subheader with padded pixels, having a pad pixel value of 0x00 displaying as transparent, 3x3 blocks.',
    'ns3301e.nsf': 'Checks an IMODE P image with a data mask subheader, the subheader with padded pixels, having a pad pixel value of 0x7F displaying as determined by the ELT, 4x4 blocks.',
    'ns3301j.nsf': 'Can the system display a mono JPEG image with mask blocks?',
    'ns3302a.nsf': 'Can the system display an uncompressed 256x256 24-bit multi-blocked (IMode-B) image? (TRACKER)',
    'ns3304a.nsf': 'Can the system display a JPEG-compressed 2048x2048 8-bit multi-blocked (512x512) mono image w/QFAC=3, RSTI=32, and IMODE=B? (CAMELS)',
    'ns3310a.nsf': 'Can the system display an uncompressed, 244x244 24-bit IMODE P multi-blocked (128x128) color image? (BIRDS)',
    'ns3321a.nsf': 'Can the system handle an NSIF file containing a streaming file header (in which the image size was unknown at the time of production) and the main header has replacement data?',
    'ns3361c.nsf': 'How does the system handle multi-images with GEO data?',
    'ns3417c.nsf': 'Can the system handle a 98x208 mono image with custom 12-bit JPEG SAR tables and COMRAT 03.5?',
    'ns3437a.nsf': 'Can the system handle a 12-bit JPEG C5 (Lossless) ES implementation multi-blocked 1024x2048 image with APP6 in each displayable block?',
    'ns3450e.nsf': 'Can the system read a 64-bit real image?',
    'ns5600a.nsf': 'Can the system handle a MS, 31 Band image, 42 by 42 pixels, 32bpp Float, and IREPBANDS all blank?',
}


def grab_nitf_fpath(key=None, safe=True):
    """
    Args:
        key (str | None): the name the nitf to grab.
            Use ``grab_nitf_fpath.keys()`` to list available keys.
            If None, ``DEFAULT_KEY`` is used.

        safe (bool): if True, only only allow access if we have the propert
            certificates. Setting to False is a security risk.  Note, in the
            past the certs did not seem to be provided by default authorities,
            but now they do seem to work with default SSL.

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.demo.nitf_demodata import *  # NOQA
        >>> fpath = grab_nitf_fpath()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> data = kwimage.imread(fpath)
        >>> kwplot.imshow(data)
        >>> kwplot.show_if_requested()
    """
    base = 'https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/'

    if key is None:
        key = DEFAULT_KEY

    if key in _FNAME_TO_INFO:
        info = _FNAME_TO_INFO[key]
    else:
        raise KeyError(key)

    fname = info['key']
    sha512 = info['sha512']
    url = base + fname
    try:
        fpath = ub.grabdata(url, appname='smart_watch/demodata/nitf',
                            hash_prefix=sha512)
    except Exception:
        if safe:
            raise
        else:
            # Disable SSL verification. This is unsafe.
            print('SAFE GRAB FAILED. FALLBACK TO UNSAFE GRAB (IGNORE SSL CERTS)')
            _orig_context = ssl._create_default_https_context
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                fpath = ub.grabdata(url, appname='smart_watch/demodata/nitf',
                                    hash_prefix=sha512)
            finally:
                # Restore ssl context if we hacked it
                ssl._create_default_https_context = _orig_context
                print('RESTORED SSL CONTEXT')
    return fpath

grab_nitf_fpath.keys = _FNAME_TO_INFO.keys


def _dev_build_description_table():
    """
    Developer function used to help populate data in this file.
    Unused at during runtime.

    Requirements:
        !pip install bs4
    """
    import bs4
    import requests
    resp = requests.get('https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/scen_2_1.html', verify=False)
    soup = bs4.BeautifulSoup(resp.text, 'html.parser')
    tables = soup.findAll('table')

    name_to_fname = {row['key'].split('.')[0]: row['key'] for row in _TEST_IMAGES}
    names_noext = list(name_to_fname.keys())

    name = None
    fname_to_desc = {}

    for tab in tables:
        for td in tab.findAll('td'):
            if name is not None:
                desc = td.text.strip()
                fname = name_to_fname[name]
                fname_to_desc[fname] = desc.replace('\r', '').replace('\n', '').replace('\t', '').replace('\xa0', '')
                name = None
            elif td.text.strip() in names_noext:
                name = td.text.strip()
    print(ub.repr2(fname_to_desc, nl=1))


def _build_test_image_table():
    """ dev function for generating expected hashes """
    import os
    test_image_table = []
    for row in _TEST_IMAGES:
        fname = row['key']
        # fpath = grab_nitf_fpath(fname, safe=True)
        base = 'https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/'
        url = base + fname
        fpath = ub.grabdata(url, appname='smart_watch/demodata/nitf')
        sha512 = ub.hash_file(fpath)[0:32]
        os.stat(fpath)
        new_row = ub.dict_union({
            'key': fname,
            'sha512': sha512,
            'size_bytes': os.stat(fpath).st_size,
        }, row)
        try:
            import kwimage
            kwimage.imread(fpath)
        except Exception:
            # Disable data that kwimage cant read
            new_row['enabled'] = False

        test_image_table.append(new_row)

    test_image_table = sorted(test_image_table, key=lambda x: x['size_bytes'])
    print('_TEST_IMAGES = {}'.format(ub.repr2(test_image_table, nl=1, sort=False)))


def _check_properties():
    from watch.gis.geotiff import geotiff_crs_info  # NOQA
    infos = []
    for row in _TEST_IMAGES:
        if row.get('enabled', True):
            print('----')
            fname = row['key']
            fpath = grab_nitf_fpath(fname, safe=True)
            out = ub.cmd('gdalinfo {}'.format(fpath), verbose=3)

            # if 'rpc' in out['out'].lower():
            #     break
            try:
                info = geotiff_crs_info(fpath)
                infos.append(info)
            except Exception as ex:
                print('ex = {!r}'.format(ex))
            else:
                row['has_crs'] = True
                row['has_rpc'] = info['is_rpc']
                print('info = {}'.format(ub.repr2(info, nl=1, sort=False)))
                _ = ub.cmd('gdalinfo {}'.format(fpath), verbose=3)

    [x['wld_crs_type'] for x in infos]

    print('_TEST_IMAGES = {}'.format(ub.repr2(_TEST_IMAGES, nl=1, sort=False)))

    # i_3001a.ntf

