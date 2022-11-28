"""
Describe how to interpret QA bands.

References:
    https://smart-research.slack.com/?redir=%2Ffiles%2FU028UQGN1N0%2FF04B998ANRL%2Faccenture_ta1_productdoc_phaseii_20211117.pptx%3Forigin_team%3DTN3QR7WAH%26origin_channel%3DC03QTAXU7GF
"""


ACC_V1_S2_QA = {

    'processing_level': 'ACC-1',
    'sensor': 'S2',
    'spec_date': '2022-11-28',
    'dtype': {
        'kind': 'u',
        'itemsize': 2,  # in bytes
    },

    'bits': [
        {
            'bit_number': 0,
            'qa_description': 'combined qa mask',
            'bit_value': [{'value': 1, 'description': 'use-pixel'}]
        },

        {
            'bit_number': 1,
            'qa_description': 'cloud',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 2,
            'qa_description': 'adjacent to cloud/shadow',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 3,
            'qa_description': 'cloud shadow',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 3,
            'qa_description': 'cloud shadow',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 4,
            'qa_description': 'snow / ice',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 5,
            'qa_description': 'water',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 6,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 7,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 8,
            'qa_description': 'filled value',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 9,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 15,
            'qa_description': 'reserved for future use',
        },


    ]
}


ACC_V1_L8_QA = {

    'processing_level': 'ACC-1',
    'sensor': 'L8',
    'spec_date': '2022-11-28',
    'dtype': {
        'kind': 'u',
        'itemsize': 2,  # in bytes
    },

    'bits': [
        {
            'bit_number': 0,
            'qa_description': 'combined qa mask',
            'bit_value': [{'value': 1, 'description': 'use-pixel'}]
        },

        {
            'bit_number': 1,
            'qa_description': 'cloud',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 2,
            'qa_description': 'adjacent to cloud/shadow',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 3,
            'qa_description': 'cloud shadow',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 3,
            'qa_description': 'cloud shadow',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 4,
            'qa_description': 'snow / ice',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 5,
            'qa_description': 'water',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 6,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 7,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 8,
            'qa_description': 'filled value',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 9,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 15,
            'qa_description': 'reserved for future use',
        },


    ]

}


"""
Note: In Ver 2 processing, WorldView VNIR products will include a 2nd QA file
containing QA information per multispectral band -Filename: *_QA2.tif -Format
same as original QA file
"""
ACC_V1_WV_QA = {

    'processing_level': 'ACC-1',
    'sensor': 'WV',
    'spec_date': '2022-11-28',
    'dtype': {
        'kind': 'u',
        'itemsize': 2,  # in bytes
    },

    'bits': [
        {
            'bit_number': 0,
            'qa_description': 'combined qa mask',
            'bit_value': [
                {'value': 1, 'description': 'use-pixel'},
                {'value': 0, 'description': 'ignore-pixel'},
            ],
        },

        {
            'bit_number': 1,
            'qa_description': 'cloud',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 2,
            'qa_description': 'cloud shadow',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 3,
            'qa_description': 'thin cloud',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 4,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 7,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 8,
            'qa_description': 'filled value / suspicious pixel',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 9,
            'qa_description': 'AOD Source',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 10,
            'qa_description': 'climatology source',
            'bit_value': [{'value': 1, 'description': 'climatology'}, {'value': 0, 'description': 'MODIS'}]
        },

        {
            'bit_number': 11,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 15,
            'qa_description': 'reserved for future use',
        },


    ]

}


ACC_V1_PD_QA = {

    'processing_level': 'ACC-1',
    'sensor': 'PD',
    'spec_date': '2022-11-28',
    'dtype': {
        'kind': 'u',
        'itemsize': 2,  # in bytes
    },

    'bits': [
        {
            'bit_number': 0,
            'qa_description': 'combined qa mask',
            'bit_value': [
                {'value': 1, 'description': 'use-pixel'},
                {'value': 0, 'description': 'ignore-pixel'},
            ],
        },

        {
            'bit_number': 1,
            'qa_description': 'cloud',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 2,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 7,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 8,
            'qa_description': 'filled value / suspicious pixel',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 9,
            'qa_description': 'AOD Source',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },

        {
            'bit_number': 10,
            'qa_description': 'climatology source',
            'bit_value': [{'value': 1, 'description': 'climatology'}, {'value': 0, 'description': 'MODIS'}]
        },

        {
            'bit_number': 11,
            'qa_description': 'reserved for future use',
        },

        {
            'bit_number': 15,
            'qa_description': 'reserved for future use',
        },


    ]

}
