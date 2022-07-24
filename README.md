# ATS
**ATS (adaptive test selection)** could be used to select an effective subset from massive unlabelled data adaptively, for saving the cost of DNN testing. 

![](https://github.com/SATE-Lab/ATS/blob/master/result/overview.png)

## Installation

```
pip install -r requirements.txt
```

## The structure of the repository 

In the experiment, our method and all baselines are conducted upon `Keras 2.3.1` with `TensorFlow 1.13.1`. All experiments are performed on a `Ubuntu 18.04.3 LTS server` with `two NVIDIA Tesla V100 GPU`, one 10-core processor "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz", and `120GB memory`.

main_folder:

```
├── ATS "adaptive test selection method"
├── gen_data/ "load data"
├── gen_model/ "to get the model in experiment" 
├── selection_method/ "test select method"
├── utils/ "some tool functions"
├── mop/  "data mutation operators"
├── exp_fault.py "RQ1 and RQ2"
├── exp_retrain_*.py "RQ3"
├── exp_utils.py "some experiment utils"
├── init.py "init dirs and download svhn data"
```

others:

```
├── result_raw/ "the raw output of experiment"
├── result/ "pictures and tables of experimental results"
├── data/ "svhn data"
├── dau/ "Data Augmentation"
├── model/ "keras model"
├── temp/ "temp files"
├── README.md
└── requirements.txt
```

## Usage

We prepared a demo for ATS

- `python demo.py`.

If you want to reproduce our experiment:

1. initial  models and datasets

   - you can download by this link

     link： https://pan.baidu.com/s/1Bi6qcN6Tc0esxcH7RbMWiQ
     
     code：atse

   - or initial  by python scrips

     1. initial dirs , svhn data and models

        `python init.py`

     2. data augmentation

        `python -m gen_data.{MnistDau}/{CifarDau}/{FashionDau}/{SvhnDau}`

2. experiment

   - `python exp_retrain_cov.py`

     `python exp_retrain_rank.py`

     `python exp_retrain_all.py`

     Here, we get the priority sequence of all selection methods and the results of rq3.

   - `python exp_fault.py`

     Here, we get the information of fault number and fault diversity of all priority sequences


## Experimental result

1. Fault detection

    ![](https://github.com/SATE-Lab/ATS/blob/master/result/tab1.png)

2. Fault diversity

   ![](https://github.com/SATE-Lab/ATS/blob/master/result/fig/diverse_errors/mnist_LeNet5.png)



3. Optimization Effectivenes

   ![](https://github.com/SATE-Lab/ATS/blob/master/result/tab3.png)

## Citation

Please cite the following paper if `ATS(adaptive test selection)` helps you on the research:

```.
@inproceedings{ATS,
  title={Adaptive test selection for deep neural networks},
  author={Gao, Xinyu and Feng, Yang and Yin, Yining and Liu, Zixi and Chen, Zhenyu and Xu, Baowen},
  booktitle={2022 IEEE/ACM 44th International Conference on Software Engineering (ICSE)},
  pages={73--85},
  year={2022},
  organization={IEEE}
}
```


