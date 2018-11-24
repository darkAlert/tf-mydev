import os
import numpy as np
import cv2
import prepare_csv as csv


def norm_face(img, pts, crop_type='default', max_size=500):
	if len(pts) > 2:
		pts = [pts[36], pts[45]]
	pts = np.array(pts)
	v = pts[1] - pts[0]
	v_len = np.sqrt(np.sum(v ** 2))
	angle_cos = v[0] / v_len
	angle_sin = v[1] / v_len
	cx = pts[0][0]
	cy = pts[0][1]

	if type(crop_type) is tuple and len(crop_type) == 4:
		kx, ky, target_size_w, target_size_h = crop_type
	elif crop_type == 'default':
		kx = 0.32
		ky = 0.395
		target_size_w = min(img.shape[1], max_size)
		target_size_h = int(target_size_w * 5 / 4.0)
	elif crop_type == 'exp01':
		kx = 0.33
		ky = 0.44
		target_size_w = min(img.shape[1], max_size)
		target_size_h = int(target_size_w * 5 / 4.0)
	elif crop_type == 'face':
		kx = 19.0 / 128.0
		ky = 31.0 / 128.0
		target_size_w = 128
		target_size_h = 128
	elif crop_type == 'head':
		kx = 41.0 / 128.0
		ky = 59.0 / 128.0
		target_size_w = 128
		target_size_h = 128
	elif crop_type == '3d_landmarks':
		kx = 70.0 / 240.0
		ky = 73.0 / 240.0
		target_size_w = 160
		target_size_h = 160
	elif crop_type == 'hopenet':
		kx = 41.0 / 128.0 # 0.36
		ky = 62.0 / 128.0 # 0.52
		target_size_w = 224
		target_size_h = 224

	target_dist = target_size_w * (1 - 2 * kx)
	scale = target_dist / v_len

	alpha = scale * angle_cos
	beta = scale * angle_sin
	shift_x = (1 - alpha) * cx - beta * cy
	shift_y = beta * cx + (1 - alpha) * cy

	tx = kx * target_size_w
	ty = ky * target_size_h
	dx = tx - (cx * alpha + cy * beta + shift_x)
	dy = ty - (-cx * beta + alpha * cy + shift_y)

	M = [
		[alpha, beta, shift_x + dx],
		[-beta, alpha, shift_y + dy]
	]
	M = np.array(M)

	return cv2.warpAffine(img, M, (target_size_w, target_size_h)), M


def preprocess(path, points = None, color = 'bgr', crop = None, size = None):
	"""Preprocess image by normalization, cropping, color and size"""
	invert_color = False
	if color == 'bgr':
		color_t = cv2.IMREAD_COLOR
	elif color == 'rgb':
		color_t = cv2.IMREAD_COLOR
		invert_color = True
	elif color == 'grayscale' or color == 'gray':
		color_t = cv2.IMREAD_GRAYSCALE
	else:
		color_t = cv2.IMREAD_UNCHANGED
		print('preprocess<warning>: Undefined color type', color_t)

	#Open image:
	img = cv2.imread(path, color_t)

	if size is None:
		size = (img.shape[1], img.shape[0]) #width, height

	#Normalize face:
	if crop is not None:
		assert points is not None
		pts = [points[i:i+2] for i in range(0, len(points), 2)]
		norm_face(img, pts, crop_type = crop)

	#Resize:
	actual_size = (img.shape[1], img.shape[0])
	if size[0] != actual_size[0] or size[1] != actual_size[1]:
		if actual_size[0] > size[0]: 
			inter = cv2.INTER_AREA 
		else: 
			inter = cv2.INTER_LINEAR
		img =  cv2.resize(img, (size), interpolation = inter)

	#BGR to RGB:
	if invert_color:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img



def preprocess_from_pickle(pickle_path, src_dir, dst_dir, color = 'bgr', crop = None, size = None):
	"""Read paths and labels from pickle-file and preprocess them"""
	paths, labels = csv.parse_pickle(pickle_path)
	assert len(paths) == len(labels)

	if os.path.exists(dst_dir) == False:
		os.makedirs(dst_dir)

	#Load, preprocess and save image:
	for i in range(len(paths)):
		print('Processing', paths[i])
		src_path = src_dir + paths[i]
		img = preprocess(src_path, labels[i], color = color, crop = crop, size = size)

		dst_path = dst_dir + paths[i]
		cv2.imwrite(dst_path, img)

	return len(paths)


if __name__ == '__main__':
	count = preprocess_from_pickle('/home/darkalert/Desktop/face_prod/normalized/2018-09-23/landmarks68p.pickle',
				                   src_dir = '/home/darkalert/Desktop/face_prod/normalized/2018-09-23/',
				                   dst_dir = '/home/darkalert/MirrorJob/Datasets/Processed/hairs-v2/hair-2018-09-23/data/',
				                   color = 'bgr', crop = 'head', size = (160, 200))
	print('Done. Total images:', count)