"""
Fork of python-slugify.

https://pypi.org/project/python-slugify/1.2.2/
"""
import re
import unicodedata
# import types
import sys

# try:
#     from htmlentitydefs import name2codepoint
#     _unicode = unicode  # NOQA
#     _unicode_type = types.UnicodeType
# except ImportError:
from html.entities import name2codepoint
_unicode = str
_unicode_type = str
unichr = chr

try:
    import text_unidecode as unidecode
except ImportError:
    import unidecode

__all__ = ['slugify', 'smart_truncate']


CHAR_ENTITY_PATTERN = re.compile(r'&(%s);' % '|'.join(name2codepoint))
DECIMAL_PATTERN = re.compile(r'&#(\d+);')
HEX_PATTERN = re.compile(r'&#x([\da-fA-F]+);')
QUOTE_PATTERN = re.compile(r'[\']+')
ALLOWED_CHARS_PATTERN = re.compile(r'[^-a-z0-9]+')
ALLOWED_CHARS_PATTERN_WITH_UPPERCASE = re.compile(r'[^-a-zA-Z0-9]+')
DUPLICATE_DASH_PATTERN = re.compile(r'-{2,}')
NUMBERS_PATTERN = re.compile(r'(?<=\d),(?=\d)')
DEFAULT_SEPARATOR = '-'


def _trunc_op(string, max_length, trunc_loc):
    """
    max_length

    string = 'DarnOvercastSculptureTipperBlazerConcaveUnsuitedDerangedHexagonRockband'
    max_length = 16
    trunc_loc = 0.5
    _trunc_op(string, max_length, trunc_loc)
    """
    total_len = len(string)
    mid_pos = int(total_len * trunc_loc)

    num_remove = max(total_len - max_length, 1)
    import ubelt as ub
    import numpy as np
    recommend = min(max(4, int(np.ceil(np.log(num_remove)))), 32)
    hash_len = min(max_length, min(num_remove, recommend))
    num_insert = hash_len + 2

    actual_remove = num_remove + num_insert

    low_pos = max(0, (mid_pos - (actual_remove) // 2))
    high_pos = min(total_len, (mid_pos + (actual_remove) // 2))
    if low_pos <= 0:
        n_extra = actual_remove - (high_pos - low_pos)
        high_pos += n_extra
    if high_pos >= total_len:
        n_extra = actual_remove - (high_pos - low_pos)
        low_pos -= n_extra

    really_removed = (high_pos - low_pos)
    high_pos += (really_removed - actual_remove)

    begin = string[:low_pos]
    mid = string[low_pos:high_pos]
    end = string[high_pos:]

    mid = ub.hash_data(string)[0:hash_len]
    trunc_text = ''.join([begin, '~', mid, '~', end])
    return trunc_text


def smart_truncate(string, max_length=0, word_boundary=False, separator=' ', save_order=False, trunc_loc=0.5):
    """
    Truncate a string.
    :param string (str): string for modification
    :param max_length (int): output string length
    :param word_boundary (bool):
    :param save_order (bool): if True then word order of output string is like input string
    :param separator (str): separator between words
    :param trunc_loc (float): fraction of location where to remove the text
    :return:
    """

    string = string.strip(separator)

    if not max_length:
        return string

    if len(string) < max_length:
        return string

    if not word_boundary:
        return _trunc_op(string, max_length, trunc_loc).strip(separator)

    if separator not in string:
        return _trunc_op(string, max_length, trunc_loc)

    # hack
    truncated = ''
    # for word in string.split(separator):
    #     if word:
    #         next_len = len(truncated) + len(word)
    #         if next_len < max_length:
    #             truncated += '{}{}'.format(word, separator)
    #         elif next_len == max_length:
    #             truncated += '{}'.format(word)
    #             break
    #         else:
    #             if save_order:
    #                 break

    if not truncated:  # pragma: no cover
        truncated = _trunc_op(string, max_length, trunc_loc)
    return truncated.strip(separator)


def slugify(text, entities=True, decimal=True, hexadecimal=True, max_length=0, word_boundary=False,
            separator=DEFAULT_SEPARATOR, save_order=False, stopwords=(), regex_pattern=None, lowercase=True,
            replacements=(), trunc_loc=1.0):
    """
    Make a slug from the given text.
    :param text (str): initial text
    :param entities (bool): converts html entities to unicode
    :param decimal (bool): converts html decimal to unicode
    :param hexadecimal (bool): converts html hexadecimal to unicode
    :param max_length (int): output string length
    :param word_boundary (bool): truncates to complete word even if length ends up shorter than max_length
    :param save_order (bool): if parameter is True and max_length > 0 return whole words in the initial order
    :param separator (str): separator between words
    :param stopwords (iterable): words to discount
    :param regex_pattern (str): regex pattern for allowed characters
    :param lowercase (bool): activate case sensitivity by setting it to False
    :param replacements (iterable): list of replacement rules e.g. [['|', 'or'], ['%', 'percent']]
    :return (str):

    # Example:
    #     >>> from watch.utils.slugify_ext import slugify  # NOQA
    #     >>> import ubelt as ub
    #     >>> text = ub.cmd('diceware -n 12')['out'].strip()
    #     >>> print('text = {!r}'.format(text))
    #     >>> slug = slugify(text, max_length=10, lowercase=0, trunc_loc=1.0)
    #     >>> print('slug = {!r}'.format(slug))
    #     >>> slug = slugify(text, max_length=10, lowercase=0, trunc_loc=0.8)
    #     >>> print('slug = {!r}'.format(slug))
    #     >>> slug = slugify(text, max_length=10, lowercase=0, trunc_loc=0.5)
    #     >>> print('slug = {!r}'.format(slug))
    #     >>> slug = slugify(text, max_length=10, lowercase=0, trunc_loc=0.2)
    #     >>> print('slug = {!r}'.format(slug))
    #     >>> slug = slugify(text, max_length=10, lowercase=0, trunc_loc=0.0)
    #     >>> print('slug = {!r}'.format(slug))
    """

    # user-specific replacements
    if replacements:
        for old, new in replacements:
            text = text.replace(old, new)

    # ensure text is unicode
    if not isinstance(text, _unicode_type):
        text = _unicode(text, 'utf-8', 'ignore')

    # replace quotes with dashes - pre-process
    text = QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)

    # decode unicode
    text = unidecode.unidecode(text)

    # ensure text is still in unicode
    if not isinstance(text, _unicode_type):
        text = _unicode(text, 'utf-8', 'ignore')

    # character entity reference
    if entities:
        text = CHAR_ENTITY_PATTERN.sub(lambda m: unichr(name2codepoint[m.group(1)]), text)

    # decimal character reference
    if decimal:
        try:
            text = DECIMAL_PATTERN.sub(lambda m: unichr(int(m.group(1))), text)
        except Exception:
            pass

    # hexadecimal character reference
    if hexadecimal:
        try:
            text = HEX_PATTERN.sub(lambda m: unichr(int(m.group(1), 16)), text)
        except Exception:
            pass

    # translate
    text = unicodedata.normalize('NFKD', text)
    if sys.version_info < (3,):
        text = text.encode('ascii', 'ignore')

    # make the text lowercase (optional)
    if lowercase:
        text = text.lower()

    # remove generated quotes -- post-process
    text = QUOTE_PATTERN.sub('', text)

    # cleanup numbers
    text = NUMBERS_PATTERN.sub('', text)

    # replace all other unwanted characters
    if lowercase:
        pattern = regex_pattern or ALLOWED_CHARS_PATTERN
    else:
        pattern = regex_pattern or ALLOWED_CHARS_PATTERN_WITH_UPPERCASE
    text = re.sub(pattern, DEFAULT_SEPARATOR, text)

    # remove redundant
    text = DUPLICATE_DASH_PATTERN.sub(DEFAULT_SEPARATOR, text).strip(DEFAULT_SEPARATOR)

    # remove stopwords
    if stopwords:
        if lowercase:
            stopwords_lower = [s.lower() for s in stopwords]
            words = [w for w in text.split(DEFAULT_SEPARATOR) if w not in stopwords_lower]
        else:
            words = [w for w in text.split(DEFAULT_SEPARATOR) if w not in stopwords]
        text = DEFAULT_SEPARATOR.join(words)

    # finalize user-specific replacements
    if replacements:
        for old, new in replacements:
            text = text.replace(old, new)

    # smart truncate if requested
    if max_length > 0:
        text = smart_truncate(text, max_length, word_boundary, DEFAULT_SEPARATOR, save_order, trunc_loc=trunc_loc)

    if separator != DEFAULT_SEPARATOR:
        text = text.replace(DEFAULT_SEPARATOR, separator)

    return text
