"""
This transforms a data split so that torchvision's ImageFolders can ingest it.
Source:
https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
"""

import sys

assert len(sys.argv) > 1, 'Specify data split path.'
assert len(sys.argv) > 2, 'Specify annotation path.'

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(sys.argv[1], 'images')

# Open and read val annotations text file
with open(os.path.join(sys.argv[2]), 'r') as fp:

	data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
	words = line.split('\t')
	val_img_dict[words[0]] = words[1]

# Display first 10 entries of resulting val_img_dict dictionary
{k: val_img_dict[k] for k in list(val_img_dict)[:10]}

# Create subfolders (if not present) for validation images based on label,
# and move images into the respective folders
for img, folder in val_img_dict.items():
	newpath = (os.path.join(val_img_dir, folder))
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	if os.path.exists(os.path.join(val_img_dir, img)):
		os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
