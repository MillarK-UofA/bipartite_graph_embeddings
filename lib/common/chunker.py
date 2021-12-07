# encoding: utf-8
# module chunker.py
# from affiliation_graphs
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
Functions that split up a series of elements into multiple chunks. Useful for splitting a list/dict into multiple chunks
for multiprocessing.
"""


def enumerate_chunk_list(l, n):
    """
    Iterates through a list and yields 'n' sized chunks. If the list does not divide evenly into chunks of length 'n'
    the last chunk will have fewer elements.

    **Parameters**
    > **l:** ``list`` -- The list to split into multiple check.

    > **n:** ``int`` -- The length of the chunk.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunk_list(l, n):
    """
    Splits a list into a series of chunks of length 'n'. If the list does not divide evenly into chunks of length 'n'
    the last chunk will have fewer elements.

    **Parameters**
    > **l:** ``list`` -- The list to split into multiple check.

    > **n:** ``int`` -- The length of the chunk.

    **Returns**
    > **results:** ``list`` -- A list of chunks from the original list.
    """
    return list(enumerate_chunk_list(l, n))


def chunk_dict(d, n):
    """
    Splits a dictionary into a series of chunks of length 'n'. If the dict does not divide evenly into chunks of
    length 'n' the last chunk will have fewer elements.

    **Parameters**
    > **d:** ``dict`` -- The dict to split into multiple check.

    > **n:** ``int`` -- The length of the chunk.

    **Returns**
    > **results:** ``list`` -- A list of chunks from the original dict.
    """

    l = list(d.items())
    return chunk_list(l, n)


# Testing
if __name__ == '__main__':
    # Test List.
    k = list(range(1, 10))
    m = chunk_list(k, 2)
    print(m)

    # Test Dict
    d = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8,
        'i': 9,
        'j': 10
    }
    e = chunk_dict(d, 5)
    print(e)