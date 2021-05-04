import glob
import os
import numpy as np
import pandas as pd

IOU_THRESHOLD = 0.5

def compute_iou(groundtruth_box, detection_box):
	g_xmin, g_ymin, g_xmax, g_ymax = map(int, groundtruth_box[1:])
	d_xmin, d_ymin, d_xmax, d_ymax = map(int, detection_box[2:])
	
	xa = max(g_xmin, d_xmin)
	ya = max(g_ymin, d_ymin)
	xb = min(g_xmax, d_xmax)
	yb = min(g_ymax, d_ymax)

	intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

	boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
	boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

	return intersection / float(boxAArea + boxBArea - intersection)

def main():
	n = int(input("Number of classes (positive integer): "))			#no. of classes
	confusion_matrix = np.zeros(shape=(n, n))
	base_path = os.getcwd()
	filename_path = base_path+"/inputs/ground-truth/"
	os.chdir(filename_path)
	filenames = glob.glob("*.txt")
	os.chdir(base_path)
	class_file = str(input("Path to text file with class names: "))
	with open(class_file, "r") as f_t:
		classnames = f_t.readlines()
	classnames = [x.strip() for x in classnames]
	class_dict = dict((classnames[i],i) for i in range(9))
	for file in filenames:
		gt_path = "inputs/ground-truth/"+file
		with open(gt_path, "r") as f_gt:
			groundtruth_boxes = f_gt.readlines()
		groundtruth_boxes = [x.strip() for x in groundtruth_boxes]
	
		det_path = "inputs/detection-results/"+file
		with open(det_path, "r") as f_det:
			detection_boxes = f_det.readlines()
		detection_boxes = [x.strip() for x in detection_boxes]
		for i in range(len(groundtruth_boxes)):
			for j in range(len(detection_boxes)):
				groundtruth_box = groundtruth_boxes[i].split()
				detection_box = detection_boxes[j].split()
				iou = compute_iou(groundtruth_box, detection_box)
				   
				if iou > IOU_THRESHOLD:
					confusion_matrix[int(class_dict[groundtruth_box[0]])][int(class_dict[detection_box[0]])] += 1
					break
				#confusion_matrix[n][int(groundtruth_box[0])] += 1
		for i in range(n):
			for j in range(n):
				confusion_matrix[i][j] = int(confusion_matrix[i][j])
	confusion_matrix = confusion_matrix/np.sum(confusion_matrix, axis = 0)
	d = {"gt / pred": classnames}
	temp1 = pd.DataFrame(data = d)
	d.clear()
	for i in range(9):
		d.update({classnames[i] : confusion_matrix[i]})
		temp = pd.DataFrame(data = d)
		temp1 = pd.concat([temp1,temp], axis = 1)
		d.clear()
	temp1.to_csv("confusion_matrix.csv")
	#print(confusion_matrix)
# 	print(temp1)

if __name__ == '__main__':
	main()
