"""Bounded private padding policy for an isolated B4/T4 MTP study.

This policy alone does not qualify model arithmetic or serving performance.
"""


def use_padded_mtp(limits):
    if type(limits) is not list or not 1<=len(limits)<=4:
        raise ValueError('Expected one to four real request limits')
    if any(type(n) is not int or not 2<=n<=1024 for n in limits):
        raise ValueError('Invalid real request output limit')
    return len(set(limits))==1


def is_public_mtp_row(row,real_rows):
    if type(real_rows) is not int or not 1<=real_rows<=4:
        raise ValueError('Invalid real row count')
    if type(row) is not int or not 0<=row<4:
        raise ValueError('Invalid physical MTP row index')
    return row<real_rows
