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
				s = [l[i] for i in range(1,len(l))]
				samples[l[0]] = samples[l[0]] + s
			else:
				samples[l[0]] = [l[i] for i in range(1,len(l))]	

	#Split the samples into a paths and labels, and reject those ones if a label is uncompleted:
	num_labels = len(csv_objs)
	paths = []
	labels = []
	rejected_paths = []
	rejected_labels = []
	for key in samples:
		if len(samples[key]) == num_labels:
			paths.append(key)
			labels.append(samples[key])
		else:
			rejected_paths.append(key)
			rejected_labels.append(samples[key])

	assert len(paths) == len(labels)
	assert len(rejected_paths) == len(rejected_labels)

	return paths, labels, [rejected_paths, rejected_labels]


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


def write_csv(paths, labels, dst_path, data_path_prefix = '', delimiter = ',', header = None):
	"""Write samples to csv-file"""
	assert len(paths) == len(labels)

	dst_csv = open(dst_path, 'w')
	if header is not None:
		dst_csv.write(header + '\n')
	
	count = 0
	for i in range(len(paths)):
		line = data_path_prefix + paths[i]

		if os.path.isfile(line) == False:
			print('write_csv<warning>: File', line, 'not found and will be skipped!')
			continue

		for l in labels[i]:
			line += delimiter + str(l)

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

	assert len(paths) == len(labels)

	return paths, labels


def parse_csv(csv_path, delimiter = ','):
	"""Extract paths and labels (and header) from csv-file"""
	lines = [line.rstrip('\n')for line in open(csv_path)]
	header = ''
	if lines[0][0] == '#':
		header = lines[0]
		lines = [line.split(delimiter) for line in lines[1:]]  # skip header
	else:
		lines = [line.split(delimiter) for line in lines]

	paths = [line[0] for line in lines]
	labels = [line[1:] for line in lines]
	
	return paths, labels, header