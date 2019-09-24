import numpy as np
from glob import glob
import pickle
import cv2

import lycon
import os
from random import randint

from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("split", "train","")
flags.DEFINE_string("dataset_path", "./datasets/","path to the entire 20bn-something-something-v1 dataset")
flags.DEFINE_string("out_dataset_path", "./datasets/","path where the generated dataset is saved")


flags.DEFINE_integer("num_frames", 32, "number of frames in a video")
flags.DEFINE_boolean("crop", False,"crop a part of the every frame")
flags.DEFINE_boolean("center", False,"center the cropped frame")
flags.DEFINE_integer("crop_dim", 224, "crop a DxD frame")
flags.DEFINE_integer("num_temporal_samplings", 1, "number of different sampling for each clips: 1 or 2")
flags.DEFINE_boolean("more_space", False, "select num_more_space_clips clips for every video, each at different space location")
flags.DEFINE_integer("num_more_space_clips", 11, "select num_more_space_clips clips")
flags.DEFINE_integer("smaller_side", 256, "resize the video such that this is the size of the smaller dim")

flags.DEFINE_integer("N", 6, "maximum workers")
flags.DEFINE_integer("index", 0, "index of current worker")


def read_video_from_images(video_folder):
	clip_images = glob(video_folder + "/*")
	# files must be in increasing order 
	clip_images.sort()
	video_frames = []
	for i,image in enumerate(clip_images):
		img = lycon.load(image)
		video_frames.append(img)
	video_frames = np.array(video_frames)
	return video_frames

def read_video_file_rand256_NL_SomethingV1(filename,num_frames=10, H = 240, W = 320, duration=-1,
	 center=True,crop=True,crop_dim=224,smaller_side=256, sample_offset=False):
	# read video, cut and resize
	# if max_duration > 0 read random 'duration' seconds
	# if fisrt_stride: sample uniform with offset
	# read video from images
	# create the used clip with fixed size and number of frames from the whole video by sampling, and cropping
	# the video is resized, sampled, and cropped
	# the video is sampled uniformly 
	#		(if videos have different lengths sample at different fps)
	# 
	clip = read_video_from_images(filename)
	
	# resize smaller size to a fixed (256) size 
	height = clip.shape[1]
	width = clip.shape[2]
	
	ratio = width / height

	if width  < height:
		new_width = smaller_side
		new_height = int(new_width/ratio)
	else:
		new_height = smaller_side
		new_width = int(ratio * new_height)

	sub_clip = clip
	new_clip = []
	for frame in sub_clip:
		new_clip.append(lycon.resize(frame, width=new_width, height=new_height, interpolation=lycon.Interpolation.CUBIC))
	
	# sample num_frame frames uniform from the entire duration or by cropping duration
	video_duration = len(new_clip)
	
	frames = []
	time_frames = aa = np.linspace(0, video_duration-1, num_frames,dtype='int')
	if sample_offset:
		start = int(np.round(0.5 * video_duration / num_frames))
		# select the frames with minimum overlap with the first sampling
		# normal sampleing: |   |   |   |   |   |   |
		# offset  sampling:   |   |   |   |   |   |   |
		# repeat last freame if necesarry
		time_frames = bb = np.clip(np.linspace(start, video_duration -1 + start, num_frames, dtype='int'),0,video_duration-1)
	
	frames = np.array([new_clip[i] for i in time_frames])
	
	real_height = frames.shape[1] 
	real_width = frames.shape[2]
	
	num_frames = frames.shape[0]
	bigger_frames = np.zeros([num_frames, new_height, new_width, 3], dtype=np.uint8)
	bigger_frames[:num_frames,:real_height, :real_width] = frames

	# crop only the center patch of every frame / or a random crom
	if center:
		center_h = bigger_frames.shape[1] // 2 - (crop_dim // 2)
		center_w = bigger_frames.shape[2] // 2 - (crop_dim // 2)
	else:
		center_h = int(np.random.rand() * ( bigger_frames.shape[1] - crop_dim))
		center_w = int(np.random.rand() * ( bigger_frames.shape[2] - crop_dim))

	if crop:
		final_video = bigger_frames[:,center_h:center_h+crop_dim, center_w:center_w+crop_dim]
	else:
		final_video = bigger_frames

	return final_video



def read_save_videos_buffer_256_NL_SomethingV1(save_dir,vids_init,index,N,
		num_frames=252, center=True, crop=True,
		crop_dim=224, smaller_side=256,
		num_temporal_samplings=1,
		more_space=False,num_more_space_clips=3):
	# save each video as a large jpeg image
	# every frame of the video would be stacked vertically
	#	the resulted image would have H * number_frames rows
	# optionally: we could stack horizontally multiple clips (different crops / different samplig)
	#	the resulted image would have W * number_of_clips columns 	


	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir+'/tmp_replace/'):
		os.makedirs(save_dir+'/tmp_replace/')

	vids = sorted(vids_init)
	# if N > 1, we se multiple processes, each processing part of the videos
	# take videos in current process
	segm = len(vids) // N 
	if index == N - 1:
		vids = vids[index * segm:]
	else:
		vids = vids[index * segm : (index + 1) * segm]

	videos = []
	for ind, vid in enumerate(vids):
		if ind % 10 == 1:
			print(f'Process {index} number of saved videos {ind}')
		video_name = vid.split('/')[-1]

		stride = 1

		all_video_frames_batch = []
		sample_offset = False
		for i in range(num_temporal_samplings):
			sample_offset = i > 0
			if more_space:
				crop = False
			# get the clip in the desired format
			video_frames = np.array(read_video_file_rand256_NL_SomethingV1(vid,num_frames=num_frames,
				center=center,
				crop=crop,crop_dim=crop_dim,
				smaller_side=smaller_side,
				sample_offset=sample_offset))

			nr_fr = video_frames.shape[0]
			vid_h = video_frames.shape[1]
			vid_w = video_frames.shape[2]

			video_batch = 0
			video_frames_batch = video_frames[::stride,:,:,:]
			# we can select multiple spatial clips by cropping square patches from a lanscape / portrait video 
			if more_space:
				dl = np.linspace(0, video_frames.shape[1] - 256, num_more_space_clips).astype(np.int32)
				dc = np.linspace(0, video_frames.shape[2] - 256, num_more_space_clips).astype(np.int32)						
				video_frames_batch_spatial = []
				for ii in range(num_more_space_clips):
					video_frames_batch_spatial.append(video_frames_batch[:,dl[ii]:dl[ii]+256, dc[ii]:dc[ii] + 256])
				video_frames_batch = np.concatenate(video_frames_batch_spatial,axis=2)
				
			video_frames_batch = np.reshape(video_frames_batch,
				[video_frames_batch.shape[0] * video_frames_batch.shape[1],video_frames_batch.shape[2],video_frames_batch.shape[3]])

			
			all_video_frames_batch.append(video_frames_batch)
		# if we use multiple samplings of the videos, we stack them horizontally
		if len(all_video_frames_batch) > 1:
			all_video_frames_batch = np.concatenate(all_video_frames_batch,axis=1)
		else:
			all_video_frames_batch = all_video_frames_batch[0]
		
		ok = True
		iii = 0
		# retry writting the final image until success or until maximum retries is reached
		image = save_dir + f'{nr_fr}_{vid_h}_{vid_w}_'+ video_name + f'_batch_{video_batch}.jpeg'
		tmp_image = save_dir + '/tmp_replace/' + image.split('/')[-1]
		good_img = False
		while(ok):
			iii += 1
			ok = False
			# use cv2 instead of lycon because lycon crasses after 35000 file written
			cv2.imwrite(tmp_image,cv2.cvtColor(all_video_frames_batch, cv2.COLOR_RGB2BGR),[int(cv2.IMWRITE_JPEG_QUALITY), 75])
			
			with open(tmp_image, 'rb') as f:
				check_chars = f.read()[-2:]
			if check_chars != b'\xff\xd9':
				print(f'Image {tmp_image} NOT complete, retry')
				ok = True
			else:
				good_img = True
			if iii == 20:
				ok = False
		if good_img:
			os.rename(tmp_image, image)

# code for creating a dataset in the format used by us for training / evaluation
# for training we use square videos: 224x224 sampled from the e
# we use different formats for training / testing
# training:
# 	a single clip
# 	random 224x224 patch cropped from video with 256 smaller scale
# testing
# 	optionally multiple clips
#	center 224x224 patch cropped from video with 256 smaller scale

if __name__ == '__main__':
	from sys import argv
	N 			= FLAGS.N
	index 		= FLAGS.index

	split     	= FLAGS.split
	print(f'index_{index}_N_{N}')
	crop 		= FLAGS.crop
	center 		= FLAGS.center
	crop_dim 	= FLAGS.crop_dim
	num_temporal_samplings 	= FLAGS.num_temporal_samplings
	more_space  = FLAGS.more_space
	num_frames  = FLAGS.num_frames
	smaller_side= FLAGS.smaller_side
	num_more_space_clips 	= FLAGS.num_more_space_clips
	# always take central crop
	if 'train' not in split:
		center 				= True
	save_dir = f'{FLAGS.out_dataset_path}/smt-smt_dataset_{num_frames}frames_crop-{crop_dim}x{crop_dim}_center-{center}_smaller_size-{smaller_side}x{smaller_side}'
	if not crop:
		save_dir = f'{FLAGS.out_dataset_path}/smt-smt_dataset_{num_frames}frames_crop-{crop}_center-{center}_smaller_size-{smaller_side}x{smaller_side}'
	if num_temporal_samplings > 1:
		save_dir = save_dir + f'_num_temporal_samplings-{num_temporal_samplings}'
	if more_space:
		save_dir = save_dir + '_more_space'
		if num_more_space_clips != 3:
			save_dir = save_dir + f'-{num_more_space_clips}'
	save_dir = save_dir + f'_{split}/'

	vids = glob(f'{FLAGS.dataset_path}/20bn-something-something-v1/{split}/*')
	print(f'Save dir: {save_dir}')
	
	read_save_videos_buffer_256_NL_SomethingV1(save_dir,vids,
				index,N, num_frames=num_frames,
				center=center, crop=crop, crop_dim=crop_dim,
				smaller_side=smaller_side,
				num_temporal_samplings=num_temporal_samplings, more_space=more_space,
				num_more_space_clips=num_more_space_clips,
				)

