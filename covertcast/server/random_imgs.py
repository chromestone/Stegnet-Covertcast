import os
import threading

import numpy as np
from PIL import Image

# basically make 8 pixel thick border black
offset = 8
width = 1280-16
height_yt = 720 - 16
frame_time_yt = 0.5

badpractice_stop = False

the_rng = np.random.default_rng(12345)

def gen_img():

	if not badpractice_stop:

		timer = threading.Timer(frame_time_yt, gen_img)
		timer.start()

	#imageData=np.zeros((720,1280,channels),np.uint8)
	temp_arr = np.zeros((720, 1280, 3), dtype=np.uint8)

	random_stuff = the_rng.integers(2, size=(height_yt // 8, width // 8, 6))
	rgb_stuff = np.zeros((*random_stuff.shape[:2], 3), dtype=np.uint8)
	for i in range(0, 3):

		rgb_stuff[:, :, i] = 160 * random_stuff[:, :, 2 * i] + 32 * random_stuff[:, :, 2 * i + 1]
	for i in range(0, 8):

		for j in range(0, 8):

			temp_arr[(i + offset):(height_yt + offset):8, (j + offset):(width + offset):8] = rgb_stuff

	# temp = Image.fromarray(temp, mode="YCbCr")
	temp = Image.fromarray(temp_arr, mode="RGB")
	temp.save("temp.jpg","JPEG",quality=100)
	os.rename("temp.jpg","out.jpg")

gen_img()

while True:

	the_input = input()
	if the_input == 'q':

		badpractice_stop = True
		break
