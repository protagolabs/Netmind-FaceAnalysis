# Netmind-FaceAnalysis

## Updates
 * 11/15/2022 Model updated: face_attributes_40 is fine-tuned from pretrain on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
 * 10/04/2022 We provide the face attributes estimation 
 * 08/29/2022 Model updated: MobileFacenet on MS1MV3 and IResent100 trained on MS1MV3 and WebFace42M.
 * 06/02/2022 The face verification code updated. The model is [IResent50](https://arxiv.org/abs/2004.04989) trained on [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) and validated on [IJBC](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).


## Requirements

Pytorch >= 1.9.0 
```bash
# CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```


