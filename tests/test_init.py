import pytest

import tensorflow as tf
import pandas as pd

from tempfile import NamedTemporaryFile

# pylint: disable=import-error,no-name-in-module
from tensorflow.python.framework import ops

from tfrecordutil import create_example_schema, write_example_tfrecord, read_example_tfrecord


def test_tfrecordutil_when_disabling_eager_execution():
    expected = 'do not disable eager execution'
    ops.reset_default_graph()
    ops.disable_eager_execution()
    with pytest.raises(ValueError) as e:
        write_example_tfrecord(None, None, None)
    assert str(e.value) == expected
    ops.enable_eager_execution()

def test_tfrecordutil():
    tmp = NamedTemporaryFile()
    filename = tmp.name
    data = [{
        'k-int': i,
        'k-float': float(i),
        'k-str': str(i),
        'k-bytes': str(i).encode(),
        'k-bool': i%2 == 0,
    } for i in range(0, 9)]
    df = pd.DataFrame(data)
    schema = create_example_schema(df)
    input_ds = tf.data.Dataset.from_tensor_slices(dict(df))
    write_example_tfrecord(filename, input_ds, schema)
    output_ds = read_example_tfrecord(filename, schema)
    for ir, dr in zip(input_ds, output_ds):
        assert set(ir.keys()) == set(dr.keys())
        for k in ir.keys():
            assert ir[k].numpy() == dr[k].numpy()
