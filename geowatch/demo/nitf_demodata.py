"""
Access sample NITF images from [1]_ for testing and demo purposes.

References:
    .. [1] https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/scen_2_1.html


NOTE:
    The data provided is no longer available, but a mirror exists on IPFS and
    the wayback machine.

    https://web.archive.org/web/20190501060607/http://www.gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/scen_2_1.html

    # IPFS folder containing all demo NITFs
    QmWApzbAX2W8cobWqKPjsM1kj82TvMGvyZ75PnbUeNHktW demo_nitf

    ipfs pin add --progress --name demo_nitf_data QmWApzbAX2W8cobWqKPjsM1kj82TvMGvyZ75PnbUeNHktW
    ipfs pin ls --type="recursive" --names
"""
import ubelt as ub


DEFAULT_KEY = 'i_3034c.ntf'  # use a small image as the default


_TEST_IMAGES = [
    {'key': 'ns3114a.nsf', 'sha512': '5605cd0b0187900c1b43130bf157c6ed', 'size_bytes': 680, 'enabled': False, 'CID': 'QmQKr5fd5cp1b1LYFgYRJKT6jdNHyecEN1XqRMKp4JJ4SM'},
    {'key': 'i_3034c.ntf', 'sha512': '5f42ab1034f20756bdf15043b58c828f', 'size_bytes': 933, 'CID': 'QmSRGeqnkmMGfXFDZS22yWt3TAbxun533kNCWjuw6rqqpC', 'alt_url': 'https://data.kitware.com/api/v1/file/62757dff4acac99f42ce826b/download'},
    {'key': 'ns3034d.nsf', 'sha512': 'aaf65232611bdc53934fa7f58790529a', 'size_bytes': 937, 'CID': 'QmStHdJpmTtmwP1cV589p8yMr9aGprukNfaKxGWHT2gXLk'},
    {'key': 'i_3034f.ntf', 'sha512': '2c2c3a918fe2805dc78a090164d671dc', 'size_bytes': 948, 'CID': 'QmUHHD92Fecr1wviDihGUkEomVoF7XHSPYgK2vZN26Et9i'},
    {'key': 'i_3051e.ntf', 'sha512': 'a67e9c4172310faaadf34a0e6ce01a72', 'size_bytes': 1436, 'enabled': False, 'CID': 'QmZLE6WudN5ZRCwpMDnpK43BnERds3qNPanzqgCpuX1BZC'},
    {'key': 'i_3052a.ntf', 'sha512': 'bb96983f58b3ec5891ef2f096d97b84e', 'size_bytes': 1520, 'enabled': False, 'CID': 'QmSkjd7DcqE34L8ThpAmpEPJNKa1ewSi2npQhrcjtreHmv'},
    {'key': 'ns3051v.nsf', 'sha512': 'ca438abdb6e67776a99bd14cc4154fa6', 'size_bytes': 1592, 'enabled': False, 'CID': 'QmRDMx5yDx8eBWFggj8556ke5iCtuvymLRfhBRonS2usba'},
    {'key': 'i_3063f.ntf', 'sha512': '68bb22e59739f31c1f5cc66cc5ac415b', 'size_bytes': 1596, 'enabled': False, 'CID': 'QmUFrYdz5xHrvMbcEWCpMurrqbXago7RK5U76EfnUrdniV'},
    {'key': 'ns3063h.nsf', 'sha512': '7b85023c46414b4b36c2a2795e59e2bd', 'size_bytes': 1606, 'enabled': False, 'CID': 'QmcFNheBVnjUVakDi5So5UtpurAaPXxYwdEj6pxqRCX57e'},
    {'key': 'i_3060a.ntf', 'sha512': '9e42e2d8e9fe6c07380501fcf5339c3e', 'size_bytes': 1624, 'enabled': False, 'CID': 'QmWCRDnbEkX7Nqju5vfUjetFTDek5GLHLjg2UwvHMsuffy'},
    {'key': 'i_3068a.ntf', 'sha512': 'aa4913146f4512b41ba2d6717a375a94', 'size_bytes': 1658, 'enabled': False, 'CID': 'QmbxM3NB8iFJs5gQk2QJi4mFHbfFBcexpJEN6WsKq7Rq19'},
    {'key': 'ns3061a.nsf', 'sha512': '86013922c8caf816943296a8b3dcbbd5', 'size_bytes': 1668, 'enabled': False, 'CID': 'QmUfNxVJttBDsJcjeQWpk8pY1FgxHkGbepy3WC4zsVTAjs'},
    {'key': 'ns3059a.nsf', 'sha512': 'c84eed6be08e60921920b4bfe621e0b6', 'size_bytes': 1766, 'enabled': False, 'CID': 'QmdjWZHsjBKa5YKTcNBaEuxZb1tXzrVy3JSnXfHCGjomBA'},
    {'key': 'i_3114e.ntf', 'sha512': '709933f92fb86a0022fed6f4edd32626', 'size_bytes': 1776, 'enabled': False, 'CID': 'QmcAakF7RN533TiU8qnhGSUoKZuScCDXZVLeJmdcq7C9PG'},
    {'key': 'ns3073a.nsf', 'sha512': 'a70636dcc455307f4cea235fb05d7ddd', 'size_bytes': 1854, 'enabled': False, 'CID': 'QmXUjPur9Cbe1sNgKkPP3sBfqkWfymKuiehNqHu8WuFU3A'},
    {'key': 'ns3101b.nsf', 'sha512': '7535d935a56ac1d1a7ac85e3d6c03983', 'size_bytes': 2144, 'enabled': False, 'CID': 'QmTjPsxGvQzdvH9T61ewEuoJKkczRL8U4CSTnqJuZhH6DL'},
    {'key': 'i_3025b.ntf', 'sha512': '582bb41bd308535ef81cc822ad68d90f', 'size_bytes': 2199, 'CID': 'QmfKK7NPfggEJAmVfv8VfFrzL6P8rpxTBUtX33YhHqmRF5'},
    {'key': 'i_3076a.ntf', 'sha512': '97200f22508e02e130799fedc64bf2a2', 'size_bytes': 2246, 'enabled': False, 'CID': 'QmUKSJRYKKDhsvZwdFgPnHLwzbEB22TFwwsoGxwCtB1Wne'},
    {'key': 'i_3018a.ntf', 'sha512': '6e8539c992f289ed7292401ce4a394a1', 'size_bytes': 3564, 'CID': 'QmfVBure4Jbmo7ESt5aotYbr5MnesWJxMFfRVLRoHqAfDd'},
    {'key': 'ns3050a.nsf', 'sha512': '1a87ae99cd1952360159a43e9f0fd269', 'size_bytes': 4071, 'CID': 'QmVQgiFEV1w1GGw9AM3abNqU3fhPFbLTrgSLsbGXPzhWXG'},
    {'key': 'ns3022b.nsf', 'sha512': '876d0e4435132cb667ef7f7b662e4ac2', 'size_bytes': 4502, 'CID': 'QmT3TpNozoLqS96dA5WCiWyZU1j3CwiAtkPmnU4WFNNe5B'},
    {'key': 'i_3015a.ntf', 'sha512': '6f7e8c8e5c93f3bfe4795dafb194cd13', 'size_bytes': 6074, 'CID': 'QmPaDjuweukFaDiM44rxkHaSMwcvLu5gM6U8EY2XU6XzAF'},
    {'key': 'ns3038a.nsf', 'sha512': '77e3270feb072d8173547cde48b8128c', 'size_bytes': 7018, 'CID': 'QmYG9PdWmJLRxZj6Qesv4uAj5gjeq3WZMofaDCQZKRsbaa'},
    {'key': 'ns3417c.nsf', 'sha512': 'd28b53eee6afb3eee010447dbf3b952a', 'size_bytes': 8836, 'enabled': False, 'CID': 'QmQDsLkqiZ2yKbF7qdShEZupiAuuzYsL7UJQoWiakJ6zbL'},
    {'key': 'ns3010a.nsf', 'sha512': 'dddede2a6bc30d9dad64ee062668b14a', 'size_bytes': 10711, 'CID': 'QmS8zETJ9fpJqAjSSiTMbTMEDcZJPjs5LG3EMhdGwYURef'},
    {'key': 'ns3017a.nsf', 'sha512': 'f074fe40a4ddfee2627b2561f6c8f360', 'size_bytes': 11052, 'CID': 'QmNTmkLffWuu4dKXz7WhAe99ErtvFMpiaogLQaRVDy3GSt'},
    {'key': 'i_3008a.ntf', 'sha512': '3a6a693f2ad48de56494017d0cb7cd5e', 'size_bytes': 21212, 'CID': 'QmVEfXgbQrHYVYbjiek5TEvxmvyWuLSLas5vnCmxBCJ5CG'},
    {'key': 'ns3114i.nsf', 'sha512': '3381b75a976da70022eb1564e07b5a73', 'size_bytes': 26664, 'enabled': False, 'CID': 'QmS3rjo6yuhGVVMtXpjkoR8HhbzXv9B7nJ3CbE7uZedSrj'},
    {'key': 'i_3201c.ntf', 'sha512': '55789ec5e5cf772c7b186bd6a0b57d6b', 'size_bytes': 48497, 'CID': 'QmNhrFM4SYsqUk333MCdTgzzwN5j1nvmCAu2Kjmq3ohEDm'},
    {'key': 'ns3033b.nsf', 'sha512': '85633cb72f4c10fcf8c98f6caffd5b9b', 'size_bytes': 53061, 'CID': 'QmXjnNXcAEwje3VFu6VYgVeeb9PacD6NRSRamV6Pr8z33s'},
    {'key': 'i_3041a.ntf', 'sha512': 'bdb9215995a602231003caf7aca0d3b7', 'size_bytes': 64682, 'CID': 'QmdNAsELfULbtXHTZhmfQ1svZVY24thttPpn4GNCZ9M2PE'},
    {'key': 'i_3113g.ntf', 'sha512': '143d866fb1bc4ee5e16b458f5e8c1e24', 'size_bytes': 70765, 'enabled': False, 'CID': 'QmRSrjcm75KiCw5zjJFvWGJXSKaLLvRNpj7y8Bqd67uBnk'},
    {'key': 'ns3301j.nsf', 'sha512': '813992b3600624b9748847180a007889', 'size_bytes': 95605, 'CID': 'QmW6Dpgrr2x94DGse6rRVuyAajMPnN7r6U9UgmEzeEQ8Uz'},
    {'key': 'ns3005b.nsf', 'sha512': '49702adf3d64648b8a29b7ef7ca7a132', 'size_bytes': 129920, 'CID': 'QmViVzFPjWe7KEF1eFDniBDggRrd2V7vAomnghyvtWYjbm'},
    {'key': 'i_3301h.ntf', 'sha512': 'faf89333207b840e11ac06b44b0ff09c', 'size_bytes': 140837, 'CID': 'QmSjNfgd2iue4ZoPbhzoTcp7isVK1aVBcDbPqYFsdRwTxg'},
    {'key': 'ns3201a.nsf', 'sha512': 'ab5cc412ddfba05463f5d84593c726da', 'size_bytes': 170590, 'CID': 'QmTibUeWxAW2YpC5WL6egku4aq95VfXTGC6CiJfcX3KLaJ'},
    {'key': 'ns3302a.nsf', 'sha512': 'c3445af2d5f529e42be7d66bde52311c', 'size_bytes': 197477, 'CID': 'QmVysZ9HexrwZ131U8iu4HVXgBtVGXazcJupA3yU7avp7n'},
    {'key': 'ns3310a.nsf', 'sha512': '62cec9952b7d98de3f550e410099ab93', 'size_bytes': 197477, 'CID': 'QmfMzFPqdeYvqpk33qRDsuhkzQbqydqXgQkvCMynMAtAnJ'},
    {'key': 'ns3301e.nsf', 'sha512': '2659e9f504b2427ebe064fa7bbfceeea', 'size_bytes': 197504, 'CID': 'QmVMwtyKUzzeUiHvNPWWhZV91rJbngy9F22wqkdFBBDUpW'},
    {'key': 'ns5600a.nsf', 'sha512': 'f7e27013e66568e02b0c4039120ff155', 'size_bytes': 219974, 'CID': 'Qmf6HT32uBxgPnYanYnaBrDyQdob9FuM5dv2mDSDejBxvK'},
    {'key': 'i_3128b.ntf', 'sha512': 'e4fa986099dcfdaaa74e66f68f48abbc', 'size_bytes': 248762, 'CID': 'QmQrnGbJ8XbUWipiw582WHbvpPqF2gLZsHj2tvRFgudC7H'},
    {'key': 'i_3004g.ntf', 'sha512': '5285370808d15ffcbc57a53105ff7a1b', 'size_bytes': 263047, 'has_crs': True, 'has_rpc': False, 'CID': 'QmSZkJKidBsWAUyPRFRbHFYxy1d4DBArhXD8utgavNASDp',
     'alt_url': 'https://data.kitware.com/api/v1/file/627579e94acac99f42ce69b1/download'},
    {'key': 'ns3004f.nsf', 'sha512': '0bfbd6d378dfd0f0e8cad703498fa6c8', 'size_bytes': 263047, 'has_crs': True, 'has_rpc': False, 'CID': 'QmXNq8HGVk5cMRTPSc9br685tpJqhBvky8Wmd6KGBKGA6t'},
    {'key': 'ns3090i.nsf', 'sha512': '9be0244363bbe63c71b55217a90c346b', 'size_bytes': 264083, 'enabled': False, 'CID': 'QmPfNJhP4Ytgx6D566NTt4CYZG2FJDamibNZaZYSLjDVG1'},
    {'key': 'i_3090m.ntf', 'sha512': '30017da0f1c9c41130e79c18f99aba97', 'size_bytes': 264083, 'CID': 'QmXhiB3uFeNHT91PFSbSNEV3zbDrkGKr9FTGA3GeVrMbfU'},
    {'key': 'i_3090u.ntf', 'sha512': 'a3f57adee4e5e25f03131891e6948da4', 'size_bytes': 264091, 'CID': 'QmYcA5UuwpMu6C7T92tLMMpmkMR69fzv3BdNt7y5KfDnTk'},
    {'key': 'ns3090q.nsf', 'sha512': '3c4f69ed8298f40e9a4845ae8321b056', 'size_bytes': 264091, 'CID': 'QmZBz5vKwygBtdfmRDG6xrbG2RtU8HWf5L7aAP5h4AF5gV'},
    {'key': 'ns3361c.nsf', 'sha512': '67123e051a8d02909e6b53b703330db9', 'size_bytes': 264592, 'has_crs': True, 'has_rpc': False, 'CID': 'QmSQBWmZm6WNzzuSySEYvSEiqKt3SpPTuuevVSjQh6g3DT'},
    {'key': 'ns3321a.nsf', 'sha512': '5a82d19b8a903537bee14e4d3a7cdc55', 'size_bytes': 281130, 'CID': 'Qmf9LVfsMTyFfvLpcXJAqCkZEoGG5KFA4QX8BoaXnz8JjX'},
    {'key': 'ns3118b.nsf', 'sha512': 'c95f99cf7bdd0ae2f802a282b772e339', 'size_bytes': 362407, 'CID': 'QmRmjCMyhHYBN5WfXJteTnAxYoJSQ9iK3KWbjDTe6DoTav'},
    {'key': 'i_5012c.ntf', 'sha512': 'de025f0a3da3b4f9e7279a45b7cd02e5', 'size_bytes': 594601, 'CID': 'QmbgWEVKE78tspn7ZYnmtPs2QYZFpirtEQxGYNQJkVghYw'},
    {'key': 'ns3304a.nsf', 'sha512': '5b0418c2f2cae7038eebdf3764514129', 'size_bytes': 701687, 'enabled': False, 'CID': 'QmUq7tL1EH2KyByNCFvRQnrCUkds4ZjpoPi78YKc1ZA94o'},
    {'key': 'i_3309a.ntf', 'sha512': '099d017dbfee8f703c4e6d76b9810a0e', 'size_bytes': 722432, 'CID': 'QmTgPnfTaHiz8zma8jDQ9n72yVeWLqApihmgQWrA4Q5wpk'},
    {'key': 'i_3001a.ntf', 'sha512': 'e5cdb23c612cbe28f0b994b69285aa49', 'size_bytes': 1049479, 'has_crs': True, 'has_rpc': False, 'CID': 'QmfNRAZdn3VtEHQSzXRYdsFCZozMSVUuHv85yjyY6EPmka'},
    {'key': 'ns3119b.nsf', 'sha512': '14b7ee574116538f88b0a7aa7758c88b', 'size_bytes': 1051108, 'enabled': False, 'CID': 'QmZzY3Vn93xKv5KEh7cSRLUEjFx29k8pxwD49e7fA4zfUx'},
    {'key': 'i_3430a.ntf', 'sha512': 'c52c72adf654c5e02fb5bb19174c5e99', 'size_bytes': 1573707, 'CID': 'QmU58g5BUDUc6k5UXX8CBgjUcR6yhvxLwicjdqMWkbADXj'},
    {'key': 'i_3301k.ntf', 'sha512': 'daf36eec7d4eb4be006c6f098d0efea8', 'size_bytes': 1770388, 'CID': 'QmZP22WEv6X3Mr5XeyZDP1XeX5u7w7RrHhD3U3de5kbvo9'},
    {'key': 'ns3301b.nsf', 'sha512': '9a51ed8ec667b2618d8d1f357a84b1e5', 'size_bytes': 1770388, 'CID': 'QmQxKrj1SppQcXHhAJBv4GccQxjdXfSf8psWH5qAjXnqLc'},
    {'key': 'i_3301c.ntf', 'sha512': '07de02322f46f9ba90318fb97b8ca759', 'size_bytes': 1770460, 'CID': 'QmYYwyHfQnx4jdpsXuwQU1jTtHVWeaLsCq7ftFP6PkLP7H'},
    {'key': 'i_3405a.ntf', 'sha512': 'fde7c7c42b42ce3bf6f6193f9108edf1', 'size_bytes': 2097995, 'CID': 'QmTYF1Hn9wAUoo8aTbEnnr9ty2NCTAnC82Sys7My9D38mi'},
    {'key': 'i_3450c.ntf', 'sha512': 'bea077c7d21d85e69f87fb4e565388d6', 'size_bytes': 2097995, 'CID': 'QmRiwn3nDxTmsjwiKgbyztMT7vwVqaKo2NP2N7Y42wy5MY'},
    {'key': 'ns3450e.nsf', 'sha512': 'ffa6e6923a2573d0aefb52908f078762', 'size_bytes': 2097995, 'CID': 'QmQSNZY3kGTxSCGw3gJftrqUHjdpDnnq9hJL5BFq4aRxbu'},
    {'key': 'ns3437a.nsf', 'sha512': '85a1bdcda593373326e5a0a498a79f95', 'size_bytes': 2656587, 'enabled': False, 'CID': 'QmQhKDL4S7mr6WkHPx9g2YgpUodUNq5nmYkSEzfKcispYa'},
    {'key': 'i_3301a.ntf', 'sha512': 'f669994ddaab9f08e1064f24dc0e1580', 'size_bytes': 3146597, 'enabled': False, 'CID': 'QmWqspPteMi38xHoohYLAhSqbcdNepWNNvS4f711uzicp8'},
    {'key': 'i_3117ax.ntf', 'sha512': 'c9ab95cc2cd4711677a0cce78122b703', 'size_bytes': 3489726, 'CID': 'QmYw4wppfG4uARMcaNrfrCCwrnTQmBb7cpW7VNZFRM7sXF'},
    {'key': 'i_3303a.ntf', 'sha512': 'e13a7aa57775b71d10e5a420e7a13214', 'size_bytes': 4195147, 'enabled': False, 'CID': 'QmVmCk77Dw4sTHvHKRBDxsto4RDQ1FQrf2roS8ntrALuCv'},
    {'key': 'i_3311a.ntf', 'sha512': '74e58d3b921555544678ae3488cb6a35', 'size_bytes': 4679168, 'CID': 'QmeQ3y7pAVgP7KxNwwaUuJyZQjvryL4yTisGneAoPS1eyJ'},
    {'key': 'ns3229b.nsf', 'sha512': '1004f108fd4b2841d3f7362ae4077e28', 'size_bytes': 5659571, 'CID': 'QmTJHhqvnMNnTMhDkq3rbRvQ8Bw6by5UUYvH8FrBUTxCqQ'},
    {'key': 'ns3228b.nsf', 'sha512': '2e9445fa3876e2e09aaa362f25f3018d', 'size_bytes': 6292578, 'CID': 'QmZuqBAUYEtYiRZ8hHoDJ3iYRf86UzcJHfSh5f6HLjmveU'},
    {'key': 'i_3228c.ntf', 'sha512': '2b059b564911c0f7c93b3ab3e332e480', 'size_bytes': 6292578, 'CID': 'QmeKe7twmNEq5iFTF6RZegVtuLBnm2GaVSnfLdbtLRkanh'},
    {'key': 'i_3228e.ntf', 'sha512': '3d9814143e2281241923904c4132859c', 'size_bytes': 6292578, 'CID': 'QmYwNVAaC9bZNCN6T5Qjojjsub2C2sVWy7y5RD3ah7VtA2'},
    {'key': 'ns3228d.nsf', 'sha512': 'a9b4ebab56101935eccc0b27b1060810', 'size_bytes': 6292578, 'CID': 'QmRrdi31As3dguMeGUZC1CroMmuj4cD3AiBmZXyWJYX6wg'},
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


def grab_nitf_fpath(key=None):
    """
    Args:
        key (str | None): the name the nitf to grab.
            Use ``grab_nitf_fpath.keys()`` to list available keys.
            If None, ``DEFAULT_KEY`` is used.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(--network)
        >>> from geowatch.demo.nitf_demodata import *  # NOQA
        >>> fpath = grab_nitf_fpath()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> data = kwimage.imread(fpath)
        >>> kwplot.imshow(data)
        >>> kwplot.show_if_requested()

    Ignore:
        for key in grab_nitf_fpath.keys():
            fpath = grab_nitf_fpath(key)
    """
    # base = 'https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/'
    if key is None:
        key = DEFAULT_KEY

    if key in _FNAME_TO_INFO:
        info = _FNAME_TO_INFO[key]
    else:
        raise KeyError(key)

    if 'alt_url' in info:
        try:
            fname = info['key']
            sha512 = info['sha512']
            fpath = ub.grabdata(info['alt_url'], appname='geowatch/demodata/nitf',
                                fname=fname, hash_prefix=sha512)
            return fpath
        except Exception:
            pass

    if 'CID' in info:
        # Use IPFS instead
        # https://ipfs.github.io/public-gateway-checker/
        ipfs_gateways = [
            'https://ipfs.io/ipfs',
            'https://dweb.link/ipfs',
            'https://gateway.pinata.cloud/ipfs',
        ]
        url_fname = info['CID']
        fname = info['key']
        sha512 = info['sha512']
        import urllib
        fpath = None
        # try different gateways if the first one times out
        for try_idx, gateway in enumerate(ipfs_gateways):
            url = gateway + '/' + url_fname
            try:
                fpath = ub.grabdata(url, appname='geowatch/demodata/nitf',
                                    fname=fname, hash_prefix=sha512)
            except urllib.error.HTTPError as ex:
                print('caught error ex = {!r}'.format(ex))
                if try_idx == len(ipfs_gateways) - 1:
                    # ipfs_exe = ub.find_exe('ipfs')
                    # print(f'ipfs_exe={ipfs_exe}')
                    # if ipfs_exe:
                    #     fpath = ub.Path.appdir('geowatch/demodata/nitf') / fname
                    #     print(f'fpath={fpath}')
                    #     ub.cmd([ipfs_exe, 'get', '-o', str(fpath), str(url_fname)], verbose=3)
                    # else:
                    raise
                else:
                    print('Try again...')
        if fpath is None:
            raise AssertionError('should not happen')
    else:
        url_fname = info['key']
        raise Exception('Requires update to IPFS support. Old URLs are dead')
    return fpath


grab_nitf_fpath.keys = _FNAME_TO_INFO.keys


def _dev_build_description_table():
    """
    Developer function used to help populate data in this file.
    Unused at during runtime.

    Notes:
        No longer works, the URL is dead!

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
    print(ub.urepr(fname_to_desc, nl=1))


def _build_test_image_table():
    """
    dev function for generating expected hashes

    Notes:
        No longer works, the URL is dead!
    """
    import os
    test_image_table = []
    for row in _TEST_IMAGES:
        fname = row['key']
        # fpath = grab_nitf_fpath(fname, safe=True)
        base = 'https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/'
        url = base + fname
        fpath = ub.grabdata(url, appname='geowatch/demodata/nitf')
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
    print('_TEST_IMAGES = {}'.format(ub.urepr(test_image_table, nl=1, sort=False)))


def _check_properties():
    from geowatch.gis.geotiff import geotiff_crs_info  # NOQA

    cid_map = ub.codeblock(
        '''
        QmfNRAZdn3VtEHQSzXRYdsFCZozMSVUuHv85yjyY6EPmka i_3001a.ntf
        QmSZkJKidBsWAUyPRFRbHFYxy1d4DBArhXD8utgavNASDp i_3004g.ntf
        QmVEfXgbQrHYVYbjiek5TEvxmvyWuLSLas5vnCmxBCJ5CG i_3008a.ntf
        QmPaDjuweukFaDiM44rxkHaSMwcvLu5gM6U8EY2XU6XzAF i_3015a.ntf
        QmfVBure4Jbmo7ESt5aotYbr5MnesWJxMFfRVLRoHqAfDd i_3018a.ntf
        QmfKK7NPfggEJAmVfv8VfFrzL6P8rpxTBUtX33YhHqmRF5 i_3025b.ntf
        QmSRGeqnkmMGfXFDZS22yWt3TAbxun533kNCWjuw6rqqpC i_3034c.ntf
        QmUHHD92Fecr1wviDihGUkEomVoF7XHSPYgK2vZN26Et9i i_3034f.ntf
        QmdNAsELfULbtXHTZhmfQ1svZVY24thttPpn4GNCZ9M2PE i_3041a.ntf
        QmZLE6WudN5ZRCwpMDnpK43BnERds3qNPanzqgCpuX1BZC i_3051e.ntf
        QmSkjd7DcqE34L8ThpAmpEPJNKa1ewSi2npQhrcjtreHmv i_3052a.ntf
        QmWCRDnbEkX7Nqju5vfUjetFTDek5GLHLjg2UwvHMsuffy i_3060a.ntf
        QmUFrYdz5xHrvMbcEWCpMurrqbXago7RK5U76EfnUrdniV i_3063f.ntf
        QmbxM3NB8iFJs5gQk2QJi4mFHbfFBcexpJEN6WsKq7Rq19 i_3068a.ntf
        QmUKSJRYKKDhsvZwdFgPnHLwzbEB22TFwwsoGxwCtB1Wne i_3076a.ntf
        QmXhiB3uFeNHT91PFSbSNEV3zbDrkGKr9FTGA3GeVrMbfU i_3090m.ntf
        QmYcA5UuwpMu6C7T92tLMMpmkMR69fzv3BdNt7y5KfDnTk i_3090u.ntf
        QmRSrjcm75KiCw5zjJFvWGJXSKaLLvRNpj7y8Bqd67uBnk i_3113g.ntf
        QmcAakF7RN533TiU8qnhGSUoKZuScCDXZVLeJmdcq7C9PG i_3114e.ntf
        QmYw4wppfG4uARMcaNrfrCCwrnTQmBb7cpW7VNZFRM7sXF i_3117ax.ntf
        QmQrnGbJ8XbUWipiw582WHbvpPqF2gLZsHj2tvRFgudC7H i_3128b.ntf
        QmNhrFM4SYsqUk333MCdTgzzwN5j1nvmCAu2Kjmq3ohEDm i_3201c.ntf
        QmeKe7twmNEq5iFTF6RZegVtuLBnm2GaVSnfLdbtLRkanh i_3228c.ntf
        QmYwNVAaC9bZNCN6T5Qjojjsub2C2sVWy7y5RD3ah7VtA2 i_3228e.ntf
        QmWqspPteMi38xHoohYLAhSqbcdNepWNNvS4f711uzicp8 i_3301a.ntf
        QmYYwyHfQnx4jdpsXuwQU1jTtHVWeaLsCq7ftFP6PkLP7H i_3301c.ntf
        QmSjNfgd2iue4ZoPbhzoTcp7isVK1aVBcDbPqYFsdRwTxg i_3301h.ntf
        ''')

    name_to_cid = {}
    for line in cid_map.split('\n'):
        line = line.strip()
        if line:
            cid, name = line.split(' ')
            name_to_cid[name] = cid

    infos = []
    for row in _TEST_IMAGES:
        if 'CID' not in row:
            row['CID'] = name_to_cid[row['key']]
        if row.get('enabled', True):
            print('----')
            fname = row['key']
            fpath = grab_nitf_fpath(fname, safe=True)
            out = ub.cmd('gdalinfo {}'.format(fpath), verbose=3)  # NOQA

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
                print('info = {}'.format(ub.urepr(info, nl=1, sort=False)))
                _ = ub.cmd('gdalinfo {}'.format(fpath), verbose=3)
    [x['wld_crs_type'] for x in infos]
    print('_TEST_IMAGES = {}'.format(ub.urepr(_TEST_IMAGES, nl=1, sort=False)))
