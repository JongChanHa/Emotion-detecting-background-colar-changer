import cv2
import numpy as np
from openvino.runtime import Core


def face_detection_model(frame):
    ie = Core()
    frame = cv2.resize(src=frame, dsize=(256, 256))
    frame=np.expand_dims(frame.transpose(2,0,1),0)


    face_detect_model_name = "intel\\face-detection-0200\FP16\\face-detection-0200.xml"
    face_detect_model = ie.read_model(model=face_detect_model_name)
    compiled_model = ie.compile_model(model=face_detect_model, device_name="CPU")

    output_layer = compiled_model.output(0)
    result_infer=compiled_model([frame])[output_layer]

    x1=int(result_infer[0][0][0][3]*256)
    y1=int(result_infer[0][0][0][4]*256)
    x2=int(result_infer[0][0][0][5]*256)
    y2=int(result_infer[0][0][0][6]*256)

    return x1,x2,y1,y2


def emotion_model(input_image):

    #face recognization
    #reshape to model input shape
    input_image = cv2.resize(src=input_image, dsize=(64, 64))
    input_image=np.expand_dims(input_image.transpose(2,0,1),0)
    ie = Core()

    emotion_model_name = "intel\emotions-recognition-retail-0003\FP16-INT8\emotions-recognition-retail-0003.xml"
    emotion_model = ie.read_model(model=emotion_model_name)
    compiled_model = ie.compile_model(model=emotion_model, device_name="CPU")

    output_layer = compiled_model.output(0)
    result_infer=compiled_model([input_image])[output_layer]
    result_index=np.argmax(result_infer)
  
    #print("r_index: {}".format(emotion(result_index)))
    return result_index

def emotion(input):
    if input==0:
        return "neutral"
    elif input==1:
        return "happy"
    elif input==2:
        return "sad"
    elif input==3:
        return "suprise"
    elif input==4:
        return "anger"


def emotion_color_map(emotion_index):
    if emotion_index==0:
        return [255,255,255]
    
    if emotion_index==1:
        return [0,255,0]
    
    if emotion_index==2:
        return [255,0,0]
    
    if emotion_index==3:
        return [0,255,255]
    
    if emotion_index==4:
        return [0,0,255]
    else:
        return [0,0,0]






