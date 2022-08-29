# Netmind-FaceAnalysis

## Updates
 * 08/29/2022 Model updated: MobileFacenet on MS1MV3 and IResent100 trained on MS1MV3 and WebFace42M.
 * 06/02/2022 The face verification code updated. The model is [IResent50](https://arxiv.org/abs/2004.04989) trained on [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) and validated on [IJBC](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

Dataset | Backbone | IJBC (AUC) | IJBC (1E-4) | IJBC (1E-5) | MFR-all
--- | --- | --- | --- |--- 
MS1MV3 | mobilefacenet | 99.55 | 94.27 | 91.49 | - 
MS1MV3 | IResnet100 | 99.64 | 97.56 | 96.12 | 94.12 
WebFace42M | IResnet100 | 99.67 | 97.60 | 96.03 | 95.66
