import streamlit as st
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import time

# Implements softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def Run_Inference(img,net):
    cv2.imwrite("test.jpg",img);
    im2= cv2.imread("test.jpg",0)
    blob=cv2.dnn.blobFromImage(im2,1/255,(28,28))

    net.setInput(blob)
    out=net.forward()
    out=softmax(out.flatten())
    classId=np.argmax(out)
    confifidence=out[classId]
    return classId,confifidence

if __name__ == '__main__':
    net=cv2.dnn.readNetFromONNX('digit.onnx')

    st.title("Digit Recognizer")
    st.write("\n\n")
    st.write("Draw a digit below and click on Predict button")
    st.write("\n")
    st.write("To clear the digit, uncheck checkbox, double click on the digit or refresh the page")
    st.write("To draw the digit, check the checkbox")

    # Draw or clear?
    drawing_mode = st.checkbox("Draw or clear?",True)

    # Create a canvas component
    image_data = st_canvas( 15, '#FFF', '#000', height=280,width=280, drawing_mode=drawing_mode, key="canvas")

    if image_data is not None:
        if st.button('Predict'):
            # Model inference
            classId,confifidence = Run_Inference(image_data,net)
            st.write('Recognized Digit: {}'.format(classId))
            st.write('Confidence: {:.2f}'.format(confifidence))
            time.sleep(10)



