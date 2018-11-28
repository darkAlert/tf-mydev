import tensorflow as tf
import cv2
import sys


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def pack_to_tfrecord(paths, labels, dst_path):
	"""Create TFRecord from paths and labels and save it"""
	assert len(paths) == len(labels)
	writer = tf.python_io.TFRecordWriter(dst_path)

	count = 0
	for i in range(len(paths)):
		with open(paths[i], 'rb') as fp:
			#Load the image data:
			img = fp.read()

			#Convert string label to int one:
			label = list(map(int, labels[i]))

			#Create a feature:
			feature = {'image': _bytes_feature(img),
					   'label': _int64_feature(label)}

			#Create an example protocol buffer
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			
			#Serialize to string and write on the file:
			writer.write(example.SerializeToString())

			count += 1

		if not i % 1000:
			print('Packing: {}/{}'.format(i, len(paths)))
			sys.stdout.flush()

	writer.close()
	sys.stdout.flush()

	return count


def unpack_from_tfrecord(src_path):
	"""Unpack images and labels from a TFRecord"""
	feature = {'image': tf.FixedLenFeature([], tf.string),
			   'label': tf.FixedLenFeature([], tf.int64)}


	#Create a list of filenames and pass it to a queue:
	filename_queue = tf.train.string_input_producer([src_path], num_epochs=1)

	#Define a reader and read the next record:
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	#Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)
	
	#Convert the image data from string back to the numbers
	# image = tf.decode_raw(features['train/image'], tf.float32)
	
	# Cast label data into int32
	label = tf.cast(features['label'], tf.int32)

	print (features['label'])