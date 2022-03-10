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

def PolyArea(x,y): # implementation of Shoelace formula 
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#classIds, cofs, bbox = net.detect()
while True:
        result, frame = cap.read()
        classIds, confs, bbox = net.detect(frame, confThreshold=0.65)
        #print(classIds, bbox)

        if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):       
                        if(objName[classId-1] == "traffic light"):
                                frame_tl = cv2.rectangle(frame, box, color = (255,0,0), thickness=2) #Take sub frame from traffic light
                                cv2.putText(frame, objName[classId-1], (box[0] +10, box[1]-9), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                                cv2.putText(frame, str(confidence), (box[0] -40, box[1] -35), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

                                hsvFrame = cv2.cvtColor(frame_tl, cv2.COLOR_BGR2HSV)
                                ###################Red
                                red_lower = np.array([0, 100, 100], np.uint8)
                                red_upper = np.array([10, 255, 255], np.uint8)
                                red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
                                print(classId, box)
                                x_tl = np.array([box[0], box[2], box[2], box[0]])
                                y_tl = np.array([box[1], box[1], box[3], box[3]])
                                S_tl = PolyArea(x_tl, y_tl)
                                edge_tl = abs(box[3] - box[1])
                                #print(S_tl)
                                # area_tl = abs(box[0]-box[2])*abs(box[1]-box[3])
                                # print(area_tl)

                                contours_red, hierarchy_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                
        
                                # gather_area_red = []
                                # for pic, contour in enumerate(contours_red):
                                #         gather_area_red.append(cv2.contourArea(contour))

                                for pic, contour in enumerate(contours_red):
                                                area_red = cv2.contourArea(contour)
                                                #print(area_red)
                                                #if area_red > 3000:
                                                x, y, w, h = cv2.boundingRect(contour)
                                                x_color = np.array([x , x+w, x+w, x])
                                                y_color = np.array([y, y, y+h, y+h])
                                                S_color = PolyArea(x_color, y_color)
                                                print(h/w)

                                                if  h/edge_tl > 0.24: #Condition for determinating traffic light signal
                                                        if w/h  <= 0.8:
                                                                frame_color_red = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (0, 0, 255), 2)   
                                                                cv2.putText(frame_color_red, "Invalid", (x ,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 2) 
                                                        else:
                                                                frame_color_red = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (0, 0, 255), 2)   
                                                                cv2.putText(frame_color_red, "Red", (x ,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 2)

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
                                                #print(area_green)
                                                x, y, w, h = cv2.boundingRect(contour)
                                                x_color = np.array([x , x+w, x+w, x])
                                                y_color = np.array([y, y, y+h, y+h])
                                                S_color = PolyArea(x_color, y_color)
                                                print(h/w)

                                                if  h/edge_tl > 0.24: #Condition for determinating traffic light signal
                                                        if w/h  <= 0.8:
                                                                frame_color_green = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (0, 255, 0), 2)   
                                                                cv2.putText(frame_color_green, "Invalid", (x ,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 255, 0), 2) 
                                                        else:
                                                                frame_color_red = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (0, 0, 255), 2)   
                                                                cv2.putText(frame_color_green, "green", (x ,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 255, 0), 2)

                                ####################Yellow
                                yellow_lower = np.array([28, 100, 100], np.uint8)
                                yellow_upper = np.array([30, 255, 255], np.uint8)
                                yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
                                
                                # Creating contour to track red color
                                contours_yellow, hierarchy_yellow = cv2.findContours(yellow_mask,
                                                                cv2.RETR_TREE,
                                                                cv2.CHAIN_APPROX_SIMPLE)
                                #100,255,220
                                for pic, contour in enumerate(contours_yellow):
                                                area_yellow = cv2.contourArea(contour)
                                                #print(area_yellow)
                                                x, y, w, h = cv2.boundingRect(contour)
                                                x_color = np.array([x , x+w, x+w, x])
                                                y_color = np.array([y, y, y+h, y+h])
                                                S_color = PolyArea(x_color, y_color)
                                                print(h/w)

                                                if  h/edge_tl > 0.24: #Condition for determinating traffic light signal
                                                        if w/h  <= 0.8:
                                                                frame_color_yellow = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (100,255,220), 2)   
                                                                cv2.putText(frame_color_yellow, "Invalid", (x ,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(100,255,220), 2) 
                                                        else:
                                                                frame_color_yellow = cv2.rectangle(frame_tl, (x, y), (x + w, y + h), (100,255,220), 2)   
                                                                cv2.putText(frame_color_yellow, "Yellow", (x ,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(100,255,220), 2)

 
                                

        cv2.imshow("Traffic light", frame)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

