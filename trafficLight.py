import cv2 
import numpy as np
import imutils
from shapely.geometry import Polygon


cap = cv2.VideoCapture(0)
objName = [] #list object
objFile = 'coco.names'

with open(objFile, 'rt') as f:
        objName = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  
weightPath= 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

#classIds, cofs, bbox = net.detect()
while True:
        result, frame = cap.read()
        classIds, confs, bbox = net.detect(frame, confThreshold=0.65)
        #print(classIds, bbox)

        if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):       
                        if(objName[classId-1] == "traffic light"):
                                frame_tl = cv2.rectangle(frame, box, color = (100,255,220), thickness=2) #Take sub frame from traffic light
                                cv2.putText(frame, objName[classId-1], (box[0] +10, box[1] +30), cv2.FONT_HERSHEY_COMPLEX,1,(100,255,220),2)
                                
                                hsvFrame = cv2.cvtColor(frame_tl, cv2.COLOR_BGR2HSV)
                                ###################Red
                                red_lower = np.array([0, 100, 100], np.uint8)
                                red_upper = np.array([10, 255, 255], np.uint8)
                                red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
                                print(classId, box)

                        
                                # area_tl = abs(box[0]-box[2])*abs(box[1]-box[3])
                                # print(area_tl)

                                contours_red, hierarchy_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                
        
                                # gather_area_red = []
                                # for pic, contour in enumerate(contours_red):
                                #         gather_area_red.append(cv2.contourArea(contour))

                                for pic, contour in enumerate(contours_red):
                                                area_red = cv2.contourArea(contour)
                                                print(area_red)
                                                if area_red > 3000:
                                                        x, y, w, h = cv2.boundingRect(contour)
                                                        frame_color_red = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (0, 0, 255), 2)   
                                                        cv2.putText(frame_color_red, "Red", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255))

                                # #################################Grreen
                                green_lower = np.array([50, 100, 100], np.uint8)
                                green_upper = np.array([70, 255, 255], np.uint8)
                                green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
                                
                                # Creating contour to track red color
                                contours_green, hierarchy_green = cv2.findContours(green_mask,
                                                                cv2.RETR_TREE,
                                                                cv2.CHAIN_APPROX_SIMPLE)

                                for pic, contour in enumerate(contours_green):
                                                area_green = cv2.contourArea(contour)
                                                print(area_green)
                                                if area_green > 3000:
                                                        x, y, w, h = cv2.boundingRect(contour)
                                                        frame_color_green = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (0, 255, 0), 2)   
                                                        cv2.putText(frame_color_green, "Green", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 255, 0))

                                ####################Yellow
                                yellow_lower = np.array([28, 100, 100], np.uint8)
                                yellow_upper = np.array([30, 255, 255], np.uint8)
                                yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
                                
                                # Creating contour to track red color
                                contours_yellow, hierarchy_yellow = cv2.findContours(yellow_mask,
                                                                cv2.RETR_TREE,
                                                                cv2.CHAIN_APPROX_SIMPLE)

                                for pic, contour in enumerate(contours_yellow):
                                                area_yellow = cv2.contourArea(contour)
                                                print(area_yellow)
                                                if area_yellow > 3000:
                                                        x, y, w, h = cv2.boundingRect(contour)
                                                        frame_color_yellow = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (250, 253, 15), 2)   
                                                        cv2.putText(frame_color_yellow, "Yellow", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(250, 253, 15))

 
                                

        cv2.imshow("Traffic light", frame)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()