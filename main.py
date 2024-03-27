import argparse #for parsing command line arguments

import cv2.dnn #deep learning module opencv
import numpy as np # numpy for numerical operations

#import utilities for handling assets and configurations
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

#load class labels from a yaml file pretrained from yolov8
CLASSES = yaml_load(check_yaml("coco128.yaml"))["names"]
#color generation for each class label
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# draws bounding boxes around objects
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main(onnx_model):
    #load the ONNX model
    model = cv2.dnn.readNetFromONNX(onnx_model)

    #start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, original_image = cap.read() #read frames
        if not ret:
            break

        [height, width, _] = original_image.shape

        #prep a square image for inference as before
        length = max(height, width)
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        #calculate scale factor and preprocess the image
        scale = length / 640
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        model.setInput(blob)

        #perform inference and process outputs as before
        outputs = model.forward()
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        #lists to store objects details
        boxes = []
        scores = []
        class_ids = []

        #process the model output
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25: #filter out detections with low confidence
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        #non-max suppression to refine the detections and prevent overlapping bounding boxes
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        #draw bounding boxes for the remaining detections
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        #display the image with bounding boxes
        cv2.imshow("YOLOv8 Real-Time Detection", original_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #release camera and close cv2 windows
    cap.release()
    cv2.destroyAllWindows()

#parse cmd line arguments for model path
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.onnx", help="Path to your ONNX model.")
    args = parser.parse_args()
    main(args.model)
