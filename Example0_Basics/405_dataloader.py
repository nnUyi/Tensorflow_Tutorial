# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''

import tensorflow as tf
import numpy as np
import scipy.misc
import os
from tqdm import tqdm
from glob import glob

# crop image in center location
def centercrop(image, side_length=255):
    height, width, channels = image.shape
    if height < side_length or width < side_length:
        return image
    height_offset = int((height-side_length)/2)
    width_offset = int((width-side_length)/2)
    img = image[height_offset:height_offset+side_length, width_offset:width_offset+side_length]
    return img

# write image data as train.tfrecords format
def tf_record_writer():
    # change [your path] to the real data_path to your image data
    filename = glob(os.path.join('[your path]', '*.*'))
    writer = tf.python_io.TFRecordWriter('train.tfrecords')
    for index, data in enumerate(filename):
        img = scipy.misc.imread(data)
        img = centercrop(img)
        img = img.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

# read image data from train.tfrecords
def tf_record_reader():
    file_queue = tf.train.string_input_producer(['train.tfrecords'])
    
    reader = tf.TFRecordReader()
    key, example = reader.read(file_queue)
    
    features = tf.parse_single_example(example, features={
        'label':tf.FixedLenFeature([], tf.int64),
        'img_raw':tf.FixedLenFeature([], tf.string)
    })
    
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [255,255,3])
    label = tf.cast(features['label'], tf.int32)
    return img, label

def main(_):
    tf_record_writer()
    img, label = tf_record_reader()
    batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=75, capacity=1000, min_after_dequeue=100)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        threads = tf.train.start_queue_runners(sess=sess)
        for i in tqdm(range(100000)):
            img_, label_ = sess.run([batch, label_batch])
            
if __name__=='__main__':
    # choose which device to use
    with tf.device('/cpu:0'):
        tf.app.run()
