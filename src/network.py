import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image


header = st.beta_container()
body = st.beta_container()
foot = st.beta_container()

classes=(['Cat', 'Dog', 'Panda'])

#CNN Network
class ConvNet(nn.Module):
    def __init__(self,num_classes=3):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
                
        #Feed forwad function
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            #Above output will be in matrix form, with shape (256,32,75,75) 
        output=output.view(-1,32*75*75)
        output=self.fc(output)
        return output



checkpoint = torch.load('../src/hightraindata.model')
model = ConvNet(num_classes=3)
model.load_state_dict(checkpoint)
model.eval()


transformer=transforms.Compose([
transforms.Resize((150,150)),
transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                    [0.5,0.5,0.5])
])

#prediction function
def prediction(img_path,transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)  
    input = Variable(image_tensor)    
    output = model(input)    
    index = output.data.numpy().argmax()    
    pred = classes[index]    
    return pred


    
with header:
    st.title('What is in the image? a Cat or Dog or Panda?')
    st.markdown('* This is a front end for a CNN(Convolutional neural network) using streamlit-API')
    st.markdown('* This CNN classifies an image of animal in to cat, dog or panda classes')
    
with body:
    st.title("Upload + Classification Example")
    left, right = st.beta_columns(2)    
    uploaded_file = right.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        left.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
   
with foot:
    st.title('Result')
    if uploaded_file is not None:
        st.write("And the picture is of a...")
        pred = prediction(uploaded_file, transformer)
        st.markdown(pred) 
    
    
        
        
        

    
    
    
    