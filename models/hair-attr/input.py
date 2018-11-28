import cv2
import tensorflow as tf

TRAIN_REC_PATH = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/records/hair-attr5-train*.tfrecord'

FLAGS = type("FLAGS", (), {"shuffle_buffer_size":0, "batch_size":0, "prefetch_buffer_size":0, "num_parallel_calls":0})
FLAGS.shuffle_buffer_size = 1000
FLAGS.batch_size = 128
FLAGS.prefetch_buffer_size = 256
FLAGS.num_parallel_calls = 6


def _parser(record):
	"""Parse TFExample records and perform simple data augmentation"""
	record_fmt = {
		"image": tf.FixedLenFeature([], dtype = tf.string, default_value = ""),
		"label": tf.VarLenFeature(tf.int64)
	}
	parsed = tf.parse_single_example(record, record_fmt)

	#Perform additional preprocessing on the parsed data:
	image = tf.image.decode_image(parsed["image"])#, tf.uint8)
	label = tf.cast(parsed['label'], tf.int32)

	# image = _augment_helper(image)  # augments image using slice, reshape, resize_bilinear

	return image, label


def input():
	files = tf.data.Dataset.list_files(TRAIN_REC_PATH)
	dataset = files.interleave(tf.data.TFRecordDataset, cycle_length = 1)
	dataset = dataset.shuffle(buffer_size = FLAGS.shuffle_buffer_size)
	dataset = dataset.map(map_func = _parser, num_parallel_calls = FLAGS.num_parallel_calls)

	# dataset = dataset.map(
		# lambda record: tuple(tf.py_func(_parser_py, [record], [tf.uint8, tf.int64])))#,
		# num_parallel_calls = FLAGS.num_parallel_calls)

	dataset = dataset.batch(batch_size = FLAGS.batch_size)
	# dataset = dataset.prefetch(buffer_size = FLAGS.prefetch_buffer_size)

	return dataset


if __name__ == '__main__':
	dataset = input()
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	with tf.Session() as sess:
		img_batch,label = sess.run(next_element)
		print(img_batch.shape)
		print(label)

		img = img_batch[0,:]
		cv2.imshow('img', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()