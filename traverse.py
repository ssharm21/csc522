import os
import cv2
import stasm
import xlwt
	
rootDir = 'home/shivani/Pictures/cohn-kanade-images/S097/001'
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")

row = 0
for subdir, dirs, files in os.walk(rootDir):
	for file in files:
		path =  os.path.join(subdir, file)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			print("Cannot load", path)
			raise SystemExit
		
		landmarks = stasm.search_single(img)
		
		landmarks = stasm.force_points_into_image(landmarks, img)
		col=0
		sheet1.write(row,col,file)
		col=col+1
		for point in landmarks:
			sheet1.write(row,col,round(point[1]))
			sheet1.write(row,col+1,round(point[0]))
			col=col+2
		row=row+1
book.save("trial.xls")