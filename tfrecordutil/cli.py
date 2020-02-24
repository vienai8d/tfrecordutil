import os
import tensorflow as tf
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from tfrecordutil import create_example_schema, write_example_tfrecord

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # WARNING

logger = tf.get_logger()

OK = 0
ERROR = 1

def csv2tfrecord(arguments=None):
    parser = ArgumentParser()
    parser.add_argument('input', help='CSV file')
    parser.add_argument('output', help='TFRrecord file')
    args = parser.parse_args(arguments)

    try:
        df = pd.read_csv(args.input)

        for k, v in df.dtypes.items():
            if v != np.object_:
                continue
            if df[k].isnull().sum() > 0:
                df[k] = df[k].fillna('')

        schema = create_example_schema(df)
        dataset = tf.data.TFRecordDataset(args.input)
        write_example_tfrecord(args.output, dataset, schema)
    except Exception as e:
        logger.error(e)
        return ERROR
    return OK
