import onnxruntime
import torch
import torchvision.models as models
import os
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np;

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def transform_image(image):
    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    return transform(image)


def model_inference(model,img_pil):
    img_t = transform_image(img_pil)
    batch_t=torch.unsqueeze(img_t,0)

    session= onnxruntime.InferenceSession(model,None);
    input_name=session.get_inputs()[0].name;
    output_name=session.get_outputs()[0].name;
    result=session.run([output_name],{input_name: batch_t.numpy()})[0]

    with open('imagenet_classes.txt') as f:
        classes=[line.strip() for line in f.readlines()]

    class_name=classes[np.argmax(result.ravel())]
    return class_name


def read_image(image_path):
    image=Image.open(image_path)

    return image;

def convert_model(SaveOnnx,SavePT):
    # Use a breakpoint in the code line below to debug your script.
    model=models.resnet50(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]

    if SaveOnnx == True:
        torch.onnx.export(model,dummy_input,"resnet50.onnx",verbose=False,input_names=input_names,output_names=output_names)

    if SavePT == True:
        traced_script_module=torch.jit.trace(model,dummy_input)
        traced_script_module.save("resnet50.pt")


def RunInference():
    st.title(" My Image Classification")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    model = "resnet50.onnx"
    convert_model(True,False)
    #image_path="dog.jpg"
    image_path = st.file_uploader("",type=["png", "jpg", "jpeg"])
    if image_path is not None:
        image=read_image(image_path)
        class_name=model_inference(model,image)
        st.image(image,use_column_width=True)
        st.title("Class recognized: {}".format(class_name))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    RunInference()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
