# object detection on images 

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import cv2
import fileinput
from utils import label_map_util
from utils import visualization_utils as vis_util

model_name = 'inference_graph'

current_path = os.getcwd()
print(current_path)
path_to_ckpt = os.path.join(current_path,model_name,'frozen_inference_graph.pb')

path_to_labels = os.path.join(current_path,'training','labelmap.pbtxt')

path_to_input_image = input()
number_labels = 2

label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=number_labels, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()

def tf_graph():
		model_used = path_to_ckpt
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
            # Works up to here.
			with tf.gfile.GFile(model_used, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			number_classes = detection_graph.get_tensor_by_name('num_detections:0')
		sess = tf.Session(graph=detection_graph) 
		
		image = cv2.imread(path_to_input_image)
		image_expanded = np.expand_dims(image, axis=0)
		
		(boxes, scores, classes, num) = sess.run(
            [detection_boxes,detection_scores,detection_classes,number_classes],
            feed_dict={image_tensor: image_expanded})
		
		print("confidence scores :" ,np.min(scores))
		
		#visualizing the graph
		vis_util.visualize_boxes_and_labels_on_image_array(
				image,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8,
				min_score_thresh=0.40)

#	 	All the results have been drawn on image. Now display the image.
		cv2.imshow('Object detector', image)
			
		return image_tensor,detection_boxes,detection_classes,detection_scores,number_classes,sess


#calling the function
tf_graph()

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
