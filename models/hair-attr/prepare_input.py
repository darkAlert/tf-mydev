
import sys
import os
from random import shuffle
import data_provider.csv as csv
from data_provider.csv import CsvObject
import data_provider.preprocessing_offline as pre
import data_provider.packer as packer



DATASET_DIR = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23'
SRC_DATA_PATH = '/home/darkalert/Desktop/face_prod/normalized/2018-09-23/'
DST_DATA_PATH = DATASET_DIR + '/data/'
PICKLE_PATH = '/home/darkalert/Desktop/face_prod/normalized/2018-09-23/landmarks68p.pickle'

FULL_CSV_PATH = DATASET_DIR + '/hair-attr5-full.csv'
TRAIN_CSV_PATH = DATASET_DIR + '/hair-attr5-train.csv'
TEST_CSV_PATH = DATASET_DIR + '/hair-attr5-test.csv'
VAL_CSV_PATH = DATASET_DIR + '/hair-attr5-val.csv'

TRAIN_REC_PATH = DATASET_DIR + '/records/hair-attr5-train.tfrecord'
TEST_REC_PATH = DATASET_DIR + '/records/hair-attr5-test.tfrecord'
VAL_REC_PATH = DATASET_DIR + '/records/hair-attr5-val.tfrecord'

DATASET_RATIO = {'train':0.8, 'val':0.15, 'test':0.05}


def _preprocess_images():
	#Preprocess images from pickle and save them on disk:
	paths, labels = csv.parse_pickle(PICKLE_PATH)
	count = pre.preprocess(
		paths, labels,
		src_dir = SRC_DATA_PATH,
		dst_dir = DST_DATA_PATH,
		color = 'bgr', crop = 'head', size = (160, 200))
	print('Preprocessing is done. Total images:', count)


def _make_csv():
	#Set CsvObjects:
	csv_objs = []
	csv_objs.append(CsvObject(DATASET_DIR + '/result-from-toloka/fixing.csv',
							  name = 'fixing', label_dict = {0:'no', 1:'yes'}))
	csv_objs.append(CsvObject(DATASET_DIR + '/result-from-toloka/fringe.csv',
							  name = 'fringe', label_dict = {0:'no', 1:'yes'}))
	csv_objs.append(CsvObject(DATASET_DIR + '/result-from-toloka/length.csv',
							  name = 'length', label_dict = {0:'bald', 1:'one_cm', 2:'short', 3:'long'}))
	csv_objs.append(CsvObject(DATASET_DIR + '/result-from-toloka/parting.csv',
							  name = 'parting', label_dict = {0:'none', 1:'side', 2:'centre'}))
	csv_objs.append(CsvObject(DATASET_DIR + '/result-from-toloka/srtucture.csv',
							  name = 'srtucture', label_dict = {0:'curls', 1:'dreadlocks', 2:'other'}))
	  
	#Generate a csv-file containing paths and labels and save it:
	paths, labels, rejceted = csv.merge_csv(csv_objs)
	header = csv.generate_header(csv_objs)
	count = csv.write_csv(
		paths, labels, 
		dst_path = FULL_CSV_PATH,
		data_path_prefix = DST_DATA_PATH,
		header = header)
	print('Accepted samples:', len(paths), ', rejected:', len(rejceted[0]))
	print('Saved samples:', count, ', skipped:', len(paths) - count)


def _split_csv_on_train_test_val():
	#Open full csv:
	paths, labels, header = csv.parse_csv(FULL_CSV_PATH)

	#Split the full dataset on a tran/validation/test subsets:
	assert DATASET_RATIO['train'] + DATASET_RATIO['val'] + DATASET_RATIO['test'] == 1.0
	train_paths, train_labels, paths, labels = pre.split(paths, labels, DATASET_RATIO['train'])
	val_ratio = DATASET_RATIO['val'] / (DATASET_RATIO['val'] +  DATASET_RATIO['test'])
	val_paths, val_labels, test_paths, test_labels = pre.split(paths, labels, val_ratio)

	#Save csv:
	csv.write_csv(train_paths, train_labels, dst_path = TRAIN_CSV_PATH)
	csv.write_csv(val_paths, val_labels, dst_path = VAL_CSV_PATH)
	csv.write_csv(test_paths, test_labels, dst_path = TEST_CSV_PATH)

	print('FULL csv has been splitted into TRAIN/VAL/TEST with ratio {}/{}/{}:'.
		format(DATASET_RATIO['train'],DATASET_RATIO['val'],DATASET_RATIO['test']))
	print('Train: {}, val: {}, test: {}'.format(len(train_paths), len(val_paths),len(test_paths)))


def _pack_csv_to_records():
	#Pack images and labels to records:
	paths, labels, _ = csv.parse_csv(TRAIN_CSV_PATH)
	count = packer.pack_to_tfrecord(paths, labels, TRAIN_REC_PATH)
	print('Train recs:', count)

	paths, labels, _ = csv.parse_csv(VAL_CSV_PATH)
	count = packer.pack_to_tfrecord(paths, labels, VAL_REC_PATH)
	print('Val recs:', count)

	paths, labels, _ = csv.parse_csv(TEST_CSV_PATH)
	count = packer.pack_to_tfrecord(paths, labels, TEST_REC_PATH)
	print('Test recs:', count)


def input():
	print('>Run preprocessing...')

	_preprocess_images()
	_make_csv()
	_split_csv_on_train_test_val()
	_pack_csv_to_records()

	print('>All done.')


if __name__ == '__main__':
	input()
	