"""
TODO
"""

import os
import sys
import threading

assert len(sys.argv) > 1, 'Specify input dir.'
assert len(sys.argv) > 2, 'Specify number of frames.'

input_dir = sys.argv[1]
number_of_frames = int(sys.argv[2])

frame_time_yt = 1.0

badpractice_stop = False

frame_no = 0

def gen_img():

	global frame_no

	print(badpractice_stop)

	if not badpractice_stop:

		timer = threading.Timer(frame_time_yt, gen_img)
		timer.start()

	os.symlink(f'file_{frame_no}.png', 'temp.png')
	os.rename('temp.png', 'out.png')

	frame_no += 1
	frame_no %= number_of_frames

gen_img()

while True:

	the_input = input()
	if the_input == 'q':

		badpractice_stop = True
		break
