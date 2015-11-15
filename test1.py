import os
rootdir = '/home/shivani/Pictures/cohn-kanade-images/S005'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print os.path.join(subdir, file)