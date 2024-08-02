import time 
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import supervision as sv
import cv2 
import numpy as np

import json

from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score
#nltk.download('punkt')  # Ensure the tokenizer is downloaded
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tabulate import tabulate
from functions import calculate_iou, drawing_boxes, calculate_predi_time
import replicate
import matplotlib.patches as patches
from PIL import Image

######## OLD ##################
#temp_img = "gt_bordny8_ny.jpg"
#temp_img = "bordny8.jpg"
# temp_img = "bordny10.jpg"

######## NEW ##################
#temp_img = "gt_bordny8_yolo.jpg"
temp_img = "gt_ordning11.jpg"
#temp_img = "gt_ordning12.jpg"
#temp_img = "gt_ordning14.jpg"

#temp_img = "gt_ordning15.jpg"
start = time.time()
f = open('YOLO_data.txt', 'a')

######================ Model YOLOx - x ================###########
image = cv2.imread(temp_img)
input_image = open(f"{temp_img}", "rb")
input_data={
    "nms": 0.01,
    "conf": 0.23,
    "tsize": 640,
    "model_name": "yolox-x",
    "input_image": input_image,
    "return_json": True
}
value_of_conf = input_data["conf"]
str_bThresh = f"Confidence value of YOLOx-x: {value_of_conf}"
value = 23
PredimgToBeStored = f"YOLOx_pred3_bordny11_bt{value}.jpg"
BlueimgToBeStored = f"YOLOx_blue3_bordny11_bt{value}.jpg"
output = replicate.run(
    "daanelson/yolox:ae0d70cebf6afb2ac4f5e4375eb599c178238b312c8325a9a114827ba869e3e9",
    input=input_data
)
print(output)
######================ end Model YOLOx - x ================###########
end = time.time()

# Remove the extra quotes and parse the JSON string
json_str = output['json_str'].strip('"')
detections = json.loads(json_str.replace('\'', '"'))  # Replace single quotes with double quotes to form valid JSON

###=========== draw the ground truth =====================##############
# image_path = image
# image = cv2.imread(image_path)
# numOfTimes = 10
# title = "objects"
# coordinates = []
# for _ in range(numOfTimes):
#     x_val, y_val, w_val, h_val = drawing_boxes(title,image)
#     coordinates.append([x_val,y_val, (x_val + w_val), (y_val+h_val)])
#     #print("Num of iter: ", x)

# ##Add two list together:
# # result = coordinates[0] + coordinates[1]
# # print("\n",result)

# result_gt = []
# sz = len(coordinates)
# for x in range(sz):
#     #merged_result = []

#     # Add each element of list_float to itself and append to merged_result
#     # for num in list_float:
#     #     merged_result.append(num + num)

#     result_gt.extend(coordinates[x])
#     #result_gt.append(coordinates[x])

#     # Append the merged_result to mrgd_list_float
#     # mrgd_lis_float.append(merged_result)

# print("\nresult ground truth bounding box",result_gt)


# #selectROI USED TO DRAW THE BOUNDING BOX WITH THE HELP OF CV2 AND OBTAIN THE COORDINATES 
# #x,y,w,h = cv2.selectROI("cat", img, fromCenter=False, showCrosshair=True)

# for coord in coordinates:
#     temp = cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)

# cv2.imshow('Detections', image)
# cv2.imwrite("yolotest3_bboxes.jpg", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# bordny9
#gt = [[195, 328, 354, 483], [88, 385, 125, 520], [129, 82, 238, 169],[42, 407, 414, 601],  [0, 0, 437, 187]]
#gt = [195, 328, 354, 483, 88, 385, 125, 520, 42, 407, 414, 601, 129, 82, 238, 169, 0, 0, 437, 187]
# bordny8
#gt = [[197, 323, 362, 485], [78, 392, 116, 535], [43, 415, 441, 623], [0, 2, 457, 175], [90, 78, 195, 156], [217, 83, 323, 161]]  

# bordny8_yolo
#gt =  [[190, 319, 366, 487], [74, 389, 122, 540], [38, 411, 445, 629], [1, 1, 440, 177], [83, 76, 202, 153], [215, 79, 328, 161]] 
#   BORDNY11.JPG      orden: cuchiaio, forchetta, tabla de picar, dish rack, bicchiere nero, shelf 
gt = [[328, 395, 355, 465], [295, 394, 323, 463], [77, 319, 267, 543], [49, 427, 423, 631], [262, 127, 337, 206], [1, 2, 444, 220]]
#   BORDNY12.JPG        orden: forchetta,cuchiaio, forbice, dish rack, bicchiere nero, grater, shelf
#gt = [[259, 397, 285, 463], [289, 397, 314, 460], [127, 424, 196, 526], [28, 426, 371, 619], [56, 154, 124, 225], [202, 71, 304, 226], [0, 0, 376, 235]]
#   BORDNY13.JPG    orden: forbice, coperchio, dish rack, bicchiere nero, tazza bianca, shelf
#gt = [[262, 399, 324, 466], [92, 369, 251, 522], [39, 432, 382, 624], [85, 162, 153, 231], [223, 161, 320, 228], [0, 2, 385, 238]]
#   BORDNY14.JPG orden: forchetta, forbici, dish rack, grater, shelf
#gt = [[294, 377, 318, 453], [116, 420, 186, 536], [11, 415, 380, 611], [141, 39, 244, 195], [0, 1, 385, 209]]
#   BORDNY15        orden: grater, fork, dish rack, black mug, whit mug, shelf
#gt = [[225, 402, 311, 551], [91, 455, 125, 561], [31, 430, 406, 632], [105, 124, 171, 196], [252, 127, 353, 199], [0, 0, 415, 214]]
###=========== end draw the ground truth =====================##############


###=========== drawing the predic bounding boxes=====================##############
print(f"dete {detections}\n")
pred_bboxes = []
labels = []
score = []
if image is not None:
    # Loop through the detections and draw each bounding box
    for det_key, det in detections.items():
        x0 = int(det['x0'])
        y0 = int(det['y0'])
        x1 = int(det['x1'])
        y1 = int(det['y1'])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label = f"{det['cls']}: {det['score']:.2f}"
        labels.append(det['cls'])
        score.append(det['score'])
        #cv2.putText(image, (x0, y0 - 10), label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(image, f'{label}', (x0, y0  - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        pred_bboxes.append([x0, y0, x1, y1])

   # Display the result
    print(f"labels: {labels} and score: {score}")
    cv2.imshow('Detections', image)
    cv2.imwrite(PredimgToBeStored, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image could not be loaded.")
###=========== drawing the predic bounding boxes =====================##############

###=============== stolpdiagram för objekten som detekteras =================#####
# Plot the bar chart with confidence and labels
#bordny8
# color_map = {
#         'knife': 'green',
#         'plate': 'green',
#         'cup': 'green',
#         'dish rack': 'green',
#         'cabinet':'green',
#         'mug':'green',
#         'cupboard':'green'
# }
#bordny11
color_map = {
        'cup': 'green',
        'dish rack': 'green',
        'shelf':'green',
        'cupboard':'green',
        'white dish rack': 'green',
        'spoon':'green',
        'fork':'green',
        'cutting board':'green',
        'black cup':'green',
        'cabinet':'green',
        'board':'green'                                
}
#gt_ordning14.jpg
# color_map = {
#         'dish rack': 'green',
#         'shelf':'green',
#         'cupboard':'green',
#         'white dish rack': 'green',
#         'fork':'green',
#         'cabinet':'green',
#         'scissors':'green',
#         'grater':'green'                                
# }
#gt_ordning15.jpg
# color_map = {
#         'dish rack': 'green',
#         'shelf':'green',
#         'cupboard':'green',
#         'white dish rack': 'green',
#         'fork':'green',
#         'cabinet':'green',
#         'black cup':'green',
#         'grater':'green',
#         'white cup':'green',
#         'cup':'green'                                
# }
default_color = 'gray'
colors = [color_map.get(label, default_color) for label in labels]
plt.figure(figsize=(10, 6))
plt.bar(range(len(labels)), score, color=colors, tick_label=labels)
plt.xlabel('Objects')
plt.ylabel('Confidence')
plt.axhline(y=value_of_conf, color='red', linestyle='--', label=str_bThresh)
plt.text(0, value_of_conf, f'{value_of_conf}', color='red', va='center', ha='left', backgroundcolor='white')
plt.title('Confidence levels of different objects generated by YOLOx storlek x')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
ground_truth_patch = mpatches.Patch(color='green', label='Ground Truth Objects')
non_ground_truth_patch = mpatches.Patch(color='gray', label='Non Ground Truth Objects')
plt.legend(handles=[ground_truth_patch, non_ground_truth_patch])
# Save the plot to a file
plot_path = fr"C:\Users\gianf\OneDrive\Skrivbord\YOLOv8\YoloX_confidence2_bar_chart{value}.png"
plt.savefig(plot_path)
###=============== end stolpdiagram för objekten som detekteras =================#####

###============= CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####
#pr_boxes = [pred_bboxes[i:i+4] for i in range(0, len(pred_bboxes), 4)]
print("Predicted bboxes",pred_bboxes, "\n")
true_positives = 0
false_positives = 0
iou_threshold = 0.75
matches = []
# matched_ground_truth = []  # Lista för att hålla reda på vilka GT-boxar som matchats
gt_matched = set()  # För att hålla reda på vilka GT-boxar som matchats
p_bboxes = []
# Calculate True Positives and False Positives
for p_idx, p_box in enumerate(pred_bboxes):
    match_found = False
    for gt_idx, gt_box in enumerate(gt):
        iou = calculate_iou(p_box, gt_box)
        print(f"Comparing Ground Truth Box: {gt_box} with Predicted Box: {p_box}, IoU: {round(iou, 3)}")  # Add this line to visualize all comparisons
        if iou >= iou_threshold:
            true_positives += 1
            gt_matched.add(gt_idx)
            print(f"Match Found! Ground Truth Box: {gt_box}, Predicted Box: {p_box}, IoU: {round(iou, 3)}")
            match_found = True
            p_bboxes.append(p_box)
            break
    if not match_found:
        print(f"No match found for Predicted Box: {p_box}")
        false_positives += 1

# Calculate False Negatives
# false_negatives = len(gt) - (true_positives + false_positives)
false_negatives = len(gt) - true_positives
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
print("p_bboxes: ", p_bboxes, "\n")
for coord1 in p_bboxes:
    cv2.rectangle(image, (coord1[0], coord1[1]), (coord1[2], coord1[3]), (255, 0,0), 2)

print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
cv2.imshow("bounding boxes", image)
cv2.imwrite(BlueimgToBeStored,image)      
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Matched ground truth: {gt_matched}")
# ###============= END CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####

####============== CALCULATING THE PREDICTION TIME  =========####
Total_cost, Predic_time = calculate_predi_time("ae0d70cebf6afb2ac4f5e4375eb599c178238b312c8325a9a114827ba869e3e9",input_data,0.000225) 
####============== END OF CALCULATING THE PREDICTION TIME  =========####

mydata = [
    ["Execution time of DINO", f"{end - start} second"], 
    ["Predict time usage of Nvidia T4 GPU hardware (YOLOV8x)", f"{Predic_time} seconds"],
    ["Cost of the usage Nvidia T4 GPU hardware (YOLOV8x)", f"{Predic_time} * 0.000225/s = {Total_cost} dollar"],
    ["Precision", f"{precision}"],
    ["Recall", f"{recall}"],
    ["DINO predicted coordinates of objects: ", f"{pred_bboxes}"],
    ["True positive", f"{true_positives}"],
    ["False positive", f"{false_positives}"],
    ["False negative", f"{false_negatives}"],
    ["Matched ground truth:", f"{gt_matched}"],
    ["iou threshold:", f"{iou_threshold}"],
    ["coordinates", f"{gt}"],
    ["Result score:", f"{score}"],
    ["labels: ", f"{labels}"],
    ["Conf: ", f"{value_of_conf}"]
]
 
# create header
head = [f"{input_image}","YOLOv8x"]
print(tabulate(mydata, headers=head, tablefmt="grid"))
f.write(tabulate(mydata, headers=head, tablefmt="grid"))
