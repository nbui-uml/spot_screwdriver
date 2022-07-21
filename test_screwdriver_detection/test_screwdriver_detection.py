import cv2
import numpy as np

#https://github.com/doleron/yolov5-opencv-cpp-python/ referenced

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.5

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


net = cv2.dnn.readNet("/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_yolo5.onnx")
img = cv2.imread("/home/csrobot/catkin_ws/src/spot_screwdriver/test_screwdriver_detection/img/7121.jpg")


img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


inputImage = format_yolov5(img)
outs = detect(inputImage, net)

class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = (0,0,255)
        cv2.rectangle(img, box, color, 2)
        cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(img, f"screwdriver: {confidence}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

cv2.imshow("output", img)

cv2.waitKey(0)
cv2.destroyAllWindows()