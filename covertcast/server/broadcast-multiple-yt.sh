#!/bin/sh

# Source:
# https://github.com/rfmcpherson/CovertCast/blob/master/covertcast/broadcast-yt.sh

INPUT=""
# Read 1 image per second
INPUT="$INPUT -r 1"
# Loop over input stream
INPUT="$INPUT -loop 1"
# Input file
INPUT="$INPUT -i file_%d.png"

AUDIO=""
# Read input as native framerate
AUDIO="$AUDIO -re"
# Audio codec (lavfi is better for bandwidth?)
AUDIO="$AUDIO -f lavfi"
# Input (dummy)
AUDIO="$AUDIO -i aevalsrc=0"

OUTPUT=""
# Video codec
OUTPUT="$OUTPUT -c:v libx264"
# Video resolution
OUTPUT="$OUTPUT -video_size 1280x720"
# Pixel format
#OUTPUT="$OUTPUT -pix_fmt yuv420p"
# Sets H.265 profile to high efficieny lossless whatever
#OUTPUT="$OUTPUT -preset ultrafast -qp 0"
# Attempt to force bitrate at 3000k
OUTPUT="$OUTPUT -b:v 1500k -maxrate 1500k -minrate 1500k"
# frames per second
OUTPUT="$OUTPUT -r 30"
# quality
OUTPUT="$OUTPUT -q:v 20"
# Audio codec
OUTPUT="$OUTPUT -c:a aac"
# GOP size (calculates intraframes)
OUTPUT="$OUTPUT -g 30"
# Format
OUTPUT="$OUTPUT -f flv"
# Allow use of "experimental" encoders
OUTPUT="$OUTPUT -strict experimental"
# Testing
OUTPUT="$OUTPUT rtmp://a.rtmp.youtube.com/live2/DO_NOT_LEAK_YOUR_KEY"

ffmpeg $INPUT $AUDIO -threads 8 -y $OUTPUT
