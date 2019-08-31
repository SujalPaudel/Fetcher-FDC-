#This code is to download videos of bottle-ml bucket, specifically date based
import argparse
import imutils
import time
import cv2
import json
import boto3
import os
import botocore
import glob, os
import bkp_start

s3 = boto3.resource('s3', aws_access_key_id = 'XXXXXXXXXXXXXXX', aws_secret_access_key = '+24saxYvPJiEunsVXHRJaQTpDAy6LIk55E3VVuDs')
s3client = boto3.client('**')

def run_fetch(batch, video_folder):	

	f = open("batches_batch.txt", "a")
	f.write(str(batch) + "\n")
	f.write("*********************************************************************" + "\n")
	f.close()		

	for s in range(len(batch)):
		print(batch[s])

		f = open("batches_singular.txt", "a")
		f.write(batch[s] + "\n")
		f.close()

		real_name = batch[s].split("/")[1]

		real_name = str(real_name)

		date = batch[s].rsplit('_', 1)[-1]
	
		hrs = date[8:10]

		mins = date[10:12]
		
		secs = date[12:14]

		vid_strt_time = hrs + ":" + mins + ":" + secs

		hrs = int(hrs)

		if(10 <= hrs <= 22):
			f = open("log_of_streams_downloaded.txt", "a")
			f.write(batch[s] + "|" + video_folder + "\n")
			f.close()

			print("Downloading " + batch[s])

			s3.Bucket('rtmp-v1').download_file(batch[s], video_folder + '/' + 'chapter_video.mp4')

			vid_name = str(batch[s])

			bkp_start.main('lb_NB_', vid_name, vid_strt_time, video_folder)


def remove_from_usb():
	test = 'usb/*'
	r = glob.glob(test)
	for i in r:
   		os.remove(i)

	print("The FDC operation is completed!!")
	print(time.time())

        
