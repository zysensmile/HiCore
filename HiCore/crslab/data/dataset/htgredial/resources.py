

from crslab.download import DownloadableFile

resources = {
    'pkuseg': {
        'version': '0.31',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EdVnNcteOkpAkLdNL-ejvAABPieUd8jIty3r1jcdJvGLzw?download=1',
            'redial_nltk.zip',
            '01dc2ebf15a0988a92112daa7015ada3e95d855e80cc1474037a86e536de3424',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0
        },
    },
    'bert': {
        'version': '0.31',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EXe_sjFhfqpJoTbNcoUPJf8Bl_4U-lnduct0z8Dw5HVCPw?download=1',
            'redial_bert.zip',
            'fb55516c22acfd3ba073e05101415568ed3398c86ff56792f82426b9258c92fd',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 101,
            'end': 102,
            'unk': 100,
            'sent_split': 2,
            'word_split': 3,
            'pad_entity': 0,
            'pad_word': 0,
        },
    },
    'gpt2': {
        'version': '0.31',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EQHOlW2m6mFEqHgt94PfoLsBbmQQeKQEOMyL1lLEHz7LvA?download=1',
            'redial_gpt2.zip',
            '37b1a64032241903a37b5e014ee36e50d09f7e4a849058688e9af52027a3ac36',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'sent_split': 4,
            'word_split': 5,
            'pad_entity': 0,
            'pad_word': 0
        },
    }
}
