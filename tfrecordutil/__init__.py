import tensorflow as tf
import numpy as np

def create_example_schema(df):
    schema = {}
    for k in df.keys():
        _dtype = df[k].dtype
        if _dtype in (np.int64, np.bool):
            schema[k] = tf.int64
        elif _dtype in (np.float32, np.float64):
            schema[k] = tf.float32
        elif _dtype == np.float64:
            schema[k] = tf.float32
        elif _dtype == np.object_:
            schema[k] = tf.string
        else:
            raise ValueError(f'unexpected dtype: column={k}, dtype={_dtype}')
    return schema

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _create_example_serialization(schema):
    ser = {}
    for k, v in schema.items():
        if v ==  tf.int64:
            ser[k] = _int64_feature
        elif v ==  tf.float32:
            ser[k] = _float_feature
        elif v ==  tf.string:
            ser[k] = _bytes_feature
        else:
            raise ValueError(f'unexpected dtype: column={k}, dtype={v}')
    return ser

def write_example_tfrecord(filename, dataset, schema):

    if not tf.executing_eagerly():
        raise ValueError('do not disable eager execution')
    example_serialization = _create_example_serialization(schema)
    
    def serialize_example(features):
        feature = {k: example_serialization[k](v) for k, v in features.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def generator():
        for features in dataset:
            yield serialize_example(features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

def _create_example_deserialization(schema):
    return {k: tf.io.FixedLenFeature([], v) for k, v in  schema.items()}

def read_example_tfrecord(filename, schema):
    example_deserialization = _create_example_deserialization(schema)
    def deserialize_example(features):
        return tf.io.parse_example(features, example_deserialization)
    return tf.data.TFRecordDataset(filename).map(deserialize_example)
