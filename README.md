# **Spot Distracted Drivers using PyTorch**

## Can Computer Vision Spot Distracted Drivers?

  **Identify the Problem:**<br>
  What is the goal? -> The main goal is to detect the distracted drivers.
  Why do we need that? -> A scenario would be the safe driving.
  Six percent of all drivers involved in fatal crashes in 2019 were reported as distracted at the time of the crashes. Nine percent of drivers 15 to 20 years old
  involved in fatal crashes were reported as distracted. [Full Report here.](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813111).

  **Success Criteria:**<br>
  Measure the Effectiveness of the solution.
  We will use accuracy as a success criteria.
  
  **Technique Used:**<br>

  More precisely we want to clasify the causes of distraction.
  We have 10 classes to predict the causes of distraction.
  ```
  c0: normal driving
  c1: texting - right
  c2: talking on the phone - right
  c3: texting - left
  c4: talking on the phone - left
  c5: operating the radio
  c6: drinking
  c7: reaching behind
  c8: hair and makeup
  c9: talking to passenger  
  ```
  
## Dataset
  We will be using State Farm Distracted Driver Detection Dataset.
  The Dataset publicly avaliable at [kaggle platform.](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview).
  
  The sample from dataset.
  <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/5048/media/drivers_statefarm.png"><br>
  source: Taken from the same page of dataset.[ dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview).
  
## Simple Model
  We will define a ConvBlock and use that ConvBlock 4 times.
  The ConvBlocks are;
  ```
  >Input(image) size is [64, 3, 64, 64].
  > ConvBlock(in_channels=3, out_channels=64)(image) >>> output size is [64, 64, 32, 32].
  > ConvBlock(64, 128)(out) >>> [64, 128, 16, 16].
  > ConvBlock(128, 256)(out) >>> [64, 256, 8, 8].
  > ConvBlock(256, 512)(out) >>> [64, 512, 4, 4].
  
  
  After ConvBlocks we have fully connected layer.
  > Flatten                     >>> [64, 512*4*4] = [64, 8192]
  > Linear(512*4*4, 500)        >>> [64, 500]
  > Linear(500, 10)             >>> [64, 10]
  ```
  
## Result
  The Models got  almost same accuracy.<br>
  Simple Architecture got ~%98.40. The best accuracy was ~%98.69 at 26th epoch. <br>
  Pretrained ResNet34 Architecture got ~%98.78. The best accuracy was ~%99.09 at 7th epoch.<br>
  We saved these as val_acc and train_loss in history(simple architecture) and history2(pretrained ResNet34 architecture).<br>
  There are subtle differences between loss changes in simple architecture. If we plot these losses the line would look like straight line.
