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
import threading
import common_batches


s3 = boto3.resource('s3', aws_access_key_id = 'XXXXXXXXXXXXXXX', aws_secret_access_key = 'XXXXXXXXXXXXXXXXXXX')
s3client = boto3.client('**')

bucket = s3.Bucket('XXXXX')

files_in_s3 = bucket.objects.all() 

logfile = []

log_list = []

filter_log = []

all_logs = {}

count = 0

totalCount = 0

def batch_process():	
	g = input("Insert the date of Videos: ") 
	print(g)
	print("Donwloading Process Started...................")

	bet_suffix = g + "%"

	for j in files_in_s3:
		if j not in logfile:
			logfile.append(j.key)

	print("Appending Completed")

	for s in range(len(logfile)):
		suffix = g
		bet_suffix = suffix + "%"
		endfix1 = "mp4"
		endfix2 = "flv"
		flag1 = logfile[s].startswith(suffix)
		flag2 = logfile[s].endswith(endfix1) or logfile[s].endswith(endfix2)


		if(flag1 and flag2):

			filter_log.append(logfile[s])


	len_filter_log = len(filter_log)

	length = len(filter_log)

	firster = round(length/2)

	first_half = filter_log[0:firster]

	sec_half = filter_log[firster:len_filter_log]


	first_half_length = len(first_half)

	first_half_half = round(first_half_length/2)

	first_half_first_half = filter_log[0:first_half_half]

	first_half_second_half = filter_log[first_half_half:first_half_length]

	sec_half_length = len(sec_half)

	sec_half_half = round(sec_half_length/2)


	firster_ender = first_half_length + sec_half_half

	sec_half_first_half = filter_log[firster:firster_ender]

	sec_half_second_half = filter_log[firster_ender:len_filter_log]

	total = len(first_half_first_half) + len(first_half_second_half) + len(sec_half_first_half) + len(sec_half_second_half)

	first_half_first_half = first_half_first_half
	first_half_second_half = first_half_second_half
	sec_half_first_half = sec_half_first_half
	sec_half_second_half = sec_half_second_half
	

	print("Thread 1 initiated")
	t1 = threading.Thread(target = common_batches.run_fetch, args = (first_half_first_half, 'first_one'))
	
	print("Thread 2 initiated")	
	t2 = threading.Thread(target = common_batches.run_fetch, args = (first_half_second_half, 'first_sec'))
	
	print("Thread 3 initiated")	
	t3 = threading.Thread(target = common_batches.run_fetch, args = (sec_half_first_half, 'second_one'))
	
	print("Thread 4 initiated")
	t4 = threading.Thread(target = common_batches.run_fetch, args = (sec_half_second_half, 'second_sec'))

	t1.start()
	t2.start()
	t3.start()
	t4.start()

	t1.join()
	print("Thread 1 complete")

	t2.join()
	print("Thread 2 complete")

	t3.join()
	print("Thread 3 complete")

	t4.join()
	print("Thread 4 complete")

	common_batches.remove_from_usb()

batch_process()
