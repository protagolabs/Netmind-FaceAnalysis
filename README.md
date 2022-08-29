# Netmind-FaceAnalysis

## Updates
 * 08/29/2022 Model updated: MobileFacenet on MS1MV3 and IResent100 trained on MS1MV3 and WebFace42M.
 * 06/02/2022 The face verification code updated. The model is [IResent50](https://arxiv.org/abs/2004.04989) trained on [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) and validated on [IJBC](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).


Dataset | Backbone | IJBC (AUC) | IJBC (1E-4) | IJBC (1E-5) | MFR-all | link
--- | --- | --- | --- |--- | --- | ---
MS1MV3 | mobilefacenet | 99.55 | 94.27 | 91.49 | - | [download](https://drive.google.com/file/d/1Kd2fUdrpAUUERTi4jFasUPk8I8caa7QW/view?usp=sharing)
MS1MV2 | IResnet50 | - | -| - | - | [download](https://drive.google.com/file/d/1P7FZU16MOthOQ2cMXg1DZwXrYn0Js2wJ/view?usp=sharing)
MS1MV3 | IResnet100 | 99.64 | 97.56 | 96.12 | 94.12 | [download](https://drive.google.com/file/d/136rmns71yjuhZ9i-tTHDs81rbR71Wagj/view?usp=sharing)
WF42M-PFC-0.2 | IResnet100 | 99.67 | 97.60 | 96.03 | 95.66 | [download](https://drive.google.com/file/d/1-_YIsfC9U2QSdEd67_4SqGe9DU7tjC97/view?usp=sharing)
