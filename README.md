# ETAnoxP
Code and trained model for our paper **ETAnoxP: Antioxidant Peptide Prediction via Fine-Tuned Prorein Large Language Models and Traditional Descriptors**
![Overall FrameWork](https://github.com/zhangzhangleilei/ETAnoxP/blob/main/fig.png)

<br>

## Installation
ETAnoxP can be downloaded by following the commands below.
```bash
git clone https://github.com/zhangzhangleilei/ETAnoxP.git
cd ETAnoxP
conda env create -f environment.yml -n ETAnoxP
conda activate ETAnoxP
```

<br>

## Data
The dataset used can be downloaded from [data](https://github.com/zhangzhangleilei/ETAnoxP/tree/main/data). 

<br>

## Predict
We have provided ETAnoxP model for you to use [predict](https://github.com/zhangzhangleilei/ETAnoxP/tree/main/predict)
If you want to use our model, please first generate the features of your own dataset according to the following code.
```bash
cd embedding

python esm_embedding.py [path]

python trad_embedding.py.py [save_path] [input_path]
```
The above command will generate fea1.csv and fea2.csv. Execute the following command to complete the prediction.
```bash
cd predict

python predict.py [fea1] [fea2] [path] [k] #k, feature selection index
```
<br>

## Web
We have developed a web server for the above process to facilitate its usage.
```bash
http://aidd.bioai-global.com/anoxp/
or
http://218.244.151.86/anoxp/
```
<br>

## Contact
If you have any problems with downloading or using the model, please contact zhangleilei0327@163.com. We will reply in a timely manner upon seeing your message.
