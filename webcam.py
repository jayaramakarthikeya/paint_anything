import os
from model.unet_cyclegan import UnetGenerator
import cv2
import torch
import numpy as np
from dataset.finetune_utils import ImageLoader


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetGenerator(input_channels=3,output_channels=3).to(device)
    model.eval()

    
    #start video/webcamsetup
    webcam = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not webcam.isOpened():
        raise IOError("Cannot open webcam")

    while True:

        #ret is bool returned by cap.read() -> whether or not frame was captured succesfully
        #if captured correctly, store in frame
        ret, frame = webcam.read()

        #resize frame
        frame = cv2.resize(frame, (128,128), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #model wants batchsize * channels * h * w
        #gives it a dimension for batch size
        frame = np.array([frame])
        #now shape is batchsize * channels * h * w
        frame = frame.transpose([0,3,1,2])

        inp_model = torch.FloatTensor(frame)
        generated_img = model(inp_model).squeeze().permute(1,2,0)
        cv2.imshow('style', generated_img)

        #ASCII value of Esc is 27.
        c = cv2.waitKey(1)
        if c == 27:
            break
      
        
    webcam.release()
    cv2.destroyAllWindows()