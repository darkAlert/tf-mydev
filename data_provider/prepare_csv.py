import os
import sys
import pickle


class CsvObject:
	"""CsvObject"""
	def __init__(self, path, name = 'label', label_dict = None):
		self.path = path
		self.name = name
		if label_dict is not None:
			self.label_dict = label_dict


def merge_csv(csv_objs, delimiter = ';'):
	"""Merging csv-files into single sample list with checking the completeness of labels"""
	samples = {}

	#Read csv's and merge into samples dictionary:
	for csv in csv_objs:
		lines = [line.rstrip('\n').split(delimiter) for line in open(csv.path)]
		assert len(lines) > 0 and len(lines[0]) > 1
		for l in lines:
			if l[0] in samples:
				samples[l[0]].append([l[i] for i in range(1,len(l))])
			else:
				samples[l[0]] = [[l[i] for i in range(1,len(l))]]	

	#Accept a sample if it has a completed label, and reject one otherwise:
	num_labels = len(csv_objs)
	accepted = []
	rejected = []
	for key in samples:
		if len(samples[key]) == num_labels:
			accepted.append([key, samples[key]])
		else:
			rejected.append([key, samples[key]])

	return accepted, rejected


def generate_header(csv_objs):
	"""Generate header for csv-file"""
	header = '#path'
	for obj in csv_objs:
		header += str(',') + obj.name
		if obj.label_dict is not None:
			header += str('[')
			first = True
			for key in obj.label_dict:
				if not first: header += str(',')
				header += str(key) + str(':') + obj.label_dict[key]
				first = False
			header += str(']')

	return header


def write_csv(samples, csv_path, data_path = '', delimiter = ',', header = None):
	"""Write samples to csv-file"""
	dst_csv = open(csv_path,'w')
	
	if header is not None:
		dst_csv.write(header + '\n')
	
	count = 0
	for s in samples:
		line = data_path + s[0]

		if os.path.isfile(line) == False:
			print('write_csv<warning>: File', line, 'not found and will be skipped!')
			continue

		for label in s[1]:
			line += delimiter + str(label[0])
		dst_csv.write(line + '\n')
		count += 1

	dst_csv.close

	return count


def parse_pickle(pickle_path):
	"""Extract paths and labels from pickle-file"""
	pickle_data = pickle.load(open(pickle_path, 'rb'))
	paths = []
	labels = []

	for pic_path, pick_label in pickle_data.items():
		paths.append(pic_path)
		pick_label = pick_label.flatten()
		label = []
		for l in pick_label:
			label.append(l)
		labels.append(label)

	return paths, labels



def main():
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
	  
	samples, rejected = merge_csv(csv_objs)
	header = generate_header(csv_objs)
	count = write_csv(samples, 
		              csv_path = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/hair-attr5-full.csv',
					  data_path = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/data/',
					  header = header)
	print('accepted samples:', len(samples), ', rejected:', len(rejected))
	print('recorded samples:', count, ', skipped:', len(samples) - count)

if __name__ == '__main__':
	main()