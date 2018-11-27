
import sys
import os
import data_provider.prepare_csv as csv
from data_provider.prepare_csv import CsvObject
import data_provider.preprocessing_offline as pre



def input():
	#Preprocess images from pickle and save them on disk:
	count = pre.preprocess_from_pickle(
		'/home/darkalert/Desktop/face_prod/normalized/2018-09-23/landmarks68p.pickle',
		src_dir = '/home/darkalert/Desktop/face_prod/normalized/2018-09-23/',
		dst_dir = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/data/',
		color = 'bgr', crop = 'head', size = (160, 200))
	print('Preprocessing is done. Total images:', count)

	#Set CsvObjects:
	csv_objs = []
	csv_objs.append(CsvObject('/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/result-from-toloka/fixing.csv',
							  name = 'fixing', label_dict = {0:'no', 1:'yes'}))
	csv_objs.append(CsvObject('/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/result-from-toloka/fringe.csv',
							  name = 'fringe', label_dict = {0:'no', 1:'yes'}))
	csv_objs.append(CsvObject('/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/result-from-toloka/length.csv',
							  name = 'length', label_dict = {0:'bald', 1:'one_cm', 2:'short', 3:'long'}))
	csv_objs.append(CsvObject('/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/result-from-toloka/parting.csv',
							  name = 'parting', label_dict = {0:'none', 1:'side', 2:'centre'}))
	csv_objs.append(CsvObject('/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/result-from-toloka/srtucture.csv',
							  name = 'srtucture', label_dict = {0:'curls', 1:'dreadlocks', 2:'other'}))
	  
	#Generate a csv-file containing paths and labels and save it:
	samples, rejected = csv.merge_csv(csv_objs)
	header = csv.generate_header(csv_objs)
	count = csv.write_csv(
		samples, 
		csv_path = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/hair-attr5-full.csv',
		data_path = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/data/',
		header = header)
	print('accepted samples:', len(samples), ', rejected:', len(rejected))
	print('recorded samples:', count, ', skipped:', len(samples) - count)

	#Pack images and labels to records:


def test():
	paths, labels, header = csv.parse_csv('/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/hair-attr5-full.csv')


if __name__ == '__main__':
	# print('Run preprocessing...')
	# input()
	# print('All done.')
	test()