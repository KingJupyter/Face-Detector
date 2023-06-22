import streamlit as st
import cv2
import numpy as np 
import tempfile

def main():
    st.set_page_config(page_title="Facial Detection")
    st.title("Facial Recognition Web App")
    st.caption("Powered by OpenCV, Streamlit")
    face_cascade = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = face_cascade.detectMultiScale(gray_frame)
        for (fx, fy, fw, fh) in face_coordinates:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        frame_placeholder.image(frame,channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
