import glob
import os
import numpy as np

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
	n = 9			#no. of classes
	confusion_matrix = np.zeros(shape=(n, n))
	base_path = os.getcwd()
	filename_path = base_path+"/inputs/ground-truth/"
	os.chdir(filename_path)
	filenames = glob.glob("*.txt")
	os.chdir(base_path)
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
					confusion_matrix[int(groundtruth_box[0])][int(detection_box[0])] += 1
	confusion_matrix = confusion_matrix/np.linalg.norm(confusion_matrix, axis = 0)
	np.savetxt("confusion_matrix.csv", confusion_matrix, delimiter=",")
	print(confusion_matrix)

if __name__ == '__main__':
	main()
