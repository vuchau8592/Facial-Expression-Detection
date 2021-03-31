import streamlit as st
import numpy as np
import cv2
import PIL
import tensorflow as tf
import tensorflow.keras as keras
from module.helper import *
from imutils.face_utils import rect_to_bb
import imutils



DEFAULT_CONFIDENCE_THRESHOLD = 50

def load_model():
    """
    function to load and cache object detector
    """
    model = keras.models.load_model(f'/app/model/model.h5')
    
    return model

def main():
    st.set_page_config(page_title = "Facial Expression detection")
    st.markdown("""<h1 style='text-align:center;background-color:#FFC0CB;color:black;padding:20px'>Emotion Detection</h1>
                <hr>
                        <h2 style='color:#003399'><b>Instruction </b></h2>
                            <p>
                                <li> Select one of our options on your left: upload an image or start a live video.
                                <li> Note: if you select a live video, make sure that your device has a webcam  </p>
                            <p class='container'>
                <hr>
                """, unsafe_allow_html=True)

    
    choose_app_mode = "Choose an app mode"
    expression_recognition_from_image = "Expression recognition from image"
    expression_recognition_from_webcam = "Real-time expression recognition"
    

    app_mode = st.sidebar.selectbox("",
                                    [choose_app_mode, 
                                     expression_recognition_from_image,
                                     expression_recognition_from_webcam])

    if app_mode == expression_recognition_from_image:
        confidence_threshold = st.sidebar.slider("Confidence threshold", 0, 100, DEFAULT_CONFIDENCE_THRESHOLD, 1)
        uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = PIL.Image.open(uploaded_file)
            img = np.array(image)
          
        if st.sidebar.button("Click Here to Classify"):
            if uploaded_file is None:
                st.sidebar.write("Please upload an Image to Classify")
            else:
                result = image_recognition(img, confidence_threshold/100.)
                st.markdown("""<h2 style='color:#003399'><b>Result\n </b></h2>""",  unsafe_allow_html=True)
                st.image(result)
        
    elif app_mode == expression_recognition_from_webcam:
        confidence_threshold = st.sidebar.slider("Confidence threshold", 0, 100, DEFAULT_CONFIDENCE_THRESHOLD, 1)
        
        if st.sidebar.button("Start","Start"):
            webcam_recognition(confidence_threshold/100.)
            

def image_recognition(img, conf_threshold):
    """
    Detect the emotion of presenting face in an image
    """
    model = load_model()
    emo_list = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredLeftEye=(0.23, 0.23), desiredFaceWidth=224)
    
    image = imutils.resize(img, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 2)
    for rect in rects:
        # extract the ROI of the *original* face, then align the face using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)
        pred = model.predict(faceAligned[np.newaxis,...])
        if np.max(pred) >= conf_threshold:
            emo = emo_list[np.argmax(pred)]
            image = cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            image = cv2.putText(image, emo, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1)
            
    return image


def webcam_recognition(conf_threshold):
    """
    Detect the emotion of presenting face in an image
    """
    model = load_model()
    emo_list = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredLeftEye=(0.23, 0.23), desiredFaceWidth=224)
    FRAME_WINDOW = st.image([])
    cap=cv2.VideoCapture(0)  
    cv2.namedWindow("ED",cv2.WINDOW_AUTOSIZE)
    
    while st.sidebar.button("Stop") == False:
           
        while True:
	    # captures frame and returns boolean value and captured image 
            ret,frame=cap.read()
            captured_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            rects = detector(captured_frame,2)
            # extract the ROI of the *original* face, then align the face using facial landmarks
            if rects:
                for rect in rects:
                    (x, y, w, h) = rect_to_bb(rect)
                    faceAligned = fa.align(frame, captured_frame, rect)
                    pred = model.predict(faceAligned[np.newaxis,...])
                    if np.max(pred) >= conf_threshold:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=2)
                        emo = emo_list[np.argmax(pred)]   
                        cv2.putText(frame, emo, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                        
            FRAME_WINDOW.image(frame,channels="BGR")
      
        cv2.destroyAllWindows

main()

col1, col2 = st.beta_columns(2)
with col1:
    st.header("From Image")
    st.image("https://www.pantechelearning.com/wp-content/uploads/2021/01/Face-Emotion-Recognition-using-CNN-OpenCV-and-Python.jpg",
            width=280)
with col2:
    st.header("From a live video")
    video_file = open('/app/demo/video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
