we develop our inference app in this directory

# 1. prepare face detector model 
1 download the weights from the [link](https://drive.google.com/file/d/127N01CeSd78vf9ayMAb3ZUXilUl6T69f/view?usp=sharing)

2 put the weight file "R50-retinaface.onnx" into the "./asset/det/" (you may need to create the folder) 

# 2. prepare face recognition model

1 download the [MS1V2-protagolabsweights](https://drive.google.com/file/d/127N01CeSd78vf9ayMAb3ZUXilUl6T69f/view?usp=sharing](https://drive.google.com/file/d/1P7FZU16MOthOQ2cMXg1DZwXrYn0Js2wJ/view?usp=sharing)

2 put the weight file "R50_MS1MV2_PW.onnx" into the "./asset/face_reg/" (you may need to create the folder) 

# 3. Ackowledgement
we adopt the implementing code from [WebFace260M Track of ICCV21-MFR](https://github.com/WebFace260M/webface260m-iccv21-mfr)
