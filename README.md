# similar-detector

pytorch implement of arcface 



## データセットの準備 (in data/)
```
raw_data
└── 20200807
    ├── blank
    │   ├── BLANK
    │   │   ├── a15_BLANK.png
    │   │   ├── a15_BLANK_mesh.png
    │   │   ├── d75_BLANK.png
    │   │   ├── d75_BLANK_mesh.png
    │   │   ├── g44_BLANK.png
    │   │   └── g44_BLANK_mesh.png
    │   ├── ElementID
    │   │   ├── a15_BLANK_ElementID.csv
    │   │   ├── d75_BLANK_ElementID.csv
    │   │   └── g44_BLANK_ElementID.csv
    │   └── NodeID
    │       ├── a15_BLANK_NodeID.csv
    │       ├── d75_BLANK_NodeID.csv
    │       └── g44_BLANK_NodeID.csv
    ├── conters
    │   ├── フォンミューゼス応力　#この名前は任意
    │   │   ├── a15_FM1.csv
    │   │   └── g54_FM1.csv
    │   └── 板減率
    │       ├── a15_FM1.csv
    │       └── g54_FM1.csv
    └── dynain
        ├── a15_FM1_dynain
        ├── a21_FM2_dynain
   
```
上図のようなディレクトリ構造でデータを格納する


その後、データ準備用のdocker コンテナ内を作成
```bash
docker build -t prepare ./prepare_data/docker/
docker run -v $PWD:/prepare/similar-detector -it -name prepare prepare bash
```
in docker container (image from prepare_data/docker/Dockerfile)
```bash 
python prepare_data.py
```
を実行し、データセットリストを作成する

# 実行環境（学習）


gpu環境で
```bash
cd similar-detector
docker build -t similar ./docker
docker run --gpus all -t -v $PWD:/workspace/similar-detector --shm-size=4gb --name similar -d -p 8097:8097 -p 8888:8888 similar  bash
docker exec -it similar bash
```

port 8888はjupyter用、 8097はvisdom用

docker 内で
```bash
# 学習状況のプロット用（必要なければconfigでdisply = False でoff)
python -m visdom.server
python train.py
```

# visualize(jupyter使用)(for remote server)

## 学習モデルを使う時
 ### 1 ssh でremote serverに入る

    例
    ```bash
    ssh msel@welding_server 
    ```
 ### 2 学習用docker に入る
 ```commandline
docker exec -it similar bash
```

### 3 jupyterを立ち上げる
```commandline
jupyter notebook --ip=0.0.0.0
```

### 4 access to http://remote:8888/
最初はjupyter立ち上げ時に出るtokenを使用


[http://welding_server:8888/](http://welding_server:8888/)

##### --ip はリモートアクセスでクロスドメインゆえ、指定 


## 実験に使ったmulti label datasets
celebA
 
https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8


## using openmax to predict

1 secondly train to make full connected layer
```bash
python train.py --train_second
```
2 prepare features of evaluation images
```bash
python imageNet_Features.py
python MAV_Compute.py
python compute_distances.py
```
3 predict criteria by openmax
```bash
python estimate_openmax.py
```


### evaluate用には必要なライブラリは別途用意必要



# References
https://github.com/deepinsight/insightface

https://github.com/auroua/InsightFace_TF

https://github.com/MuggleWang/CosFace_pytorch

# pretrained model and lfw test dataset
the pretrained model and the lfw test dataset can be download here. link: https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA pwd: b2ec
the pretrained model use resnet-18 without se. Please modify the path of the lfw dataset in config.py before you run test.py.
