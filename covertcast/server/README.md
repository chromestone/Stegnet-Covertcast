# Server

This code should encode messages and deliver them as images to a YouTube live stream (preferrable in a loop.)

Make sure you write the image in a temporary spot and only mv them to out.jpg when ready.

There are two scripts here:

* `broadcast-yt.sh` and `random_imgs.py` that simulates covertcast (we used this to do some preliminary packet captures and decided it wasn't worth our time to analyze that low level).
* Standalone `broadcast-multiple-yt.sh` which loops a series of images at 1 image per second (broadcast fps is 30).

Scripts derived from https://github.com/rfmcpherson/CovertCast .
