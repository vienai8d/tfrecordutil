import pytest

import tensorflow as tf

from tempfile import NamedTemporaryFile
from tfrecordutil.cli import csv2tfrecord


def test_csv2tfrecort():
    lines = [
        'int,float,string\n',
        '1,1.0,a\n',
        ',2.0,b\n',
        '3,,c\n',
        '4,4.0,\n',
    ]
    src = NamedTemporaryFile()
    dst = NamedTemporaryFile()
    with open(src.name, 'w') as f:
        f.writelines(lines)
    csv2tfrecord([src.name, dst.name])
    with open(dst.name, 'rb') as f:
        assert len(f.readlines()) > 0