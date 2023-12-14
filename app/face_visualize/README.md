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

* 1 download the R50 weights from the [insightface torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) through this [link](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC)

* 2 unzip the weights by
```
unzip ms1mv3_arcface_r50_fp16.zip -d ./weights
``` 
# Visualization
 * run the code 
 ```
 python visualization.py --imageA-path "your source image path here" --imageB-path "your target image path here" --method 'gradcam' (default)
 ```
 the results will show in output folder

 * Or you can run the scripts for all the methods

 ```
 bash visualization.sh
 ```

# Ackowledgement
we adopt the implementing code from [insightface](https://insightface.ai/) and [Grad-Cam](https://github.com/jacobgil/pytorch-grad-cam)
