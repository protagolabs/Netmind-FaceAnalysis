we develop our inference app in this directory

# Prepare Environment (Optional)

1 create a new python environment by conda
```
conda create --name face python=3.8
```

2 install the devs
```
conda activate face
pip install -r requirements.txt
```


# Prepare face detector model 
1 download the weights from the [link](https://drive.google.com/file/d/127N01CeSd78vf9ayMAb3ZUXilUl6T69f/view?usp=sharing)

2 put the weight file "R50-retinaface.onnx" into the "./asset/det/" (you may need to create the folder) 

# Prepare face recognition model

1 download the [MS1V2-protagolabsweights](https://drive.google.com/file/d/1P7FZU16MOthOQ2cMXg1DZwXrYn0Js2wJ/view?usp=sharing)

2 put the weight file "R50_MS1MV2_PW.onnx" into the "./asset/face_reg/" (you may need to create the folder) 

# Important Notes

    In this demo, we provide a face detection model (/assets/det/R50-retinaface.onnx) and a face recognition model (/assets/face_reg/R50_MS1MV2_PW.onnx). Participants should replace them with the models trained by themselves and modify the model name in faceverfication_implement.py

    def load(self, rdir):
        det_model = os.path.join(rdir, 'det', 'R50-retinaface.onnx')
        self.detector = face_detection.get_retinaface(det_model)
        print('use onnx-model:', det_model)

        self.detector.prepare(use_gpu=self.use_gpu)
        max_time_cost = 1000

        self.model_file = os.path.join(rdir, 'face_reg', "R18.onnx")
        print('use onnx-model:', self.model_file)


     Please also judge whether the value of input_mean and input_std in faceverfication_implement.py is right for their own face recognition models. Generally, when the model have preprocessing steps in itself, it do not need any other operations (input_mean = 0.0 and input_std = 1.0). Otherwise, it may be input_mean = 127.5 and input_std = 127.5. Participants must adjust the script accordingly.
    
    If necessary, please compare the feature obtained by this repo with the one obtained from training model by using one image, e.g. demo/0.png, to make sure the feature are almost the same. Otherwise, there may be other issues.


# Results:

Dataset | Backbone | IJBC (AUC) | IJBC (1E-4) | IJBC (1E-5) | MFR-all | link
--- | --- | --- | --- |--- | --- | ---
MS1MV3 | mobilefacenet | 99.55 | 94.27 | 91.49 | - | [download](https://drive.google.com/file/d/1Kd2fUdrpAUUERTi4jFasUPk8I8caa7QW/view?usp=sharing)
MS1MV2 | IResnet50 | - | -| - | - | [download](https://drive.google.com/file/d/1P7FZU16MOthOQ2cMXg1DZwXrYn0Js2wJ/view?usp=sharing)
MS1MV3 | IResnet100 | 99.64 | 97.56 | 96.12 | 94.12 | [download](https://drive.google.com/file/d/136rmns71yjuhZ9i-tTHDs81rbR71Wagj/view?usp=sharing)
WF42M-PFC-0.2 | IResnet100 | 99.67 | 97.60 | 96.03 | 95.66 | [download](https://drive.google.com/file/d/1-_YIsfC9U2QSdEd67_4SqGe9DU7tjC97/view?usp=sharing)

# Ackowledgement
we adopt the implementing code from [WebFace260M Track of ICCV21-MFR](https://github.com/WebFace260M/webface260m-iccv21-mfr)
