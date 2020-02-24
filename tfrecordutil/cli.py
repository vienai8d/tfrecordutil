import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser
from tfrecordutil import create_example_schema, write_example_tfrecord

OK = 0
ERROR = 1

def csv2tfrecord(arguments=None):
    parser = ArgumentParser()
    parser.add_argument('input', help='CSV file')
    parser.add_argument('output', help='TFRrecord file')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        schema = create_example_schema(df)
        dataset = tf.data.Dataset(args.input)
        write_example_tfrecord(args.output, dataset, schema)
    except Exception as e:
        print(e)
        return ERROR
    return OK
