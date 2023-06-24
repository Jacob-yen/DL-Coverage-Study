# Revisiting Deep Neural Network Test Coverage from the Test Effectiveness Perspective (Experience Paper)

📢 This is the implementation repository of our submission: Revisiting Deep Neural Network Test Coverage from the Test Effectiveness Perspective. 

## Description  

In Deep Neural Network (DNN) testing, many test coverage metrics have been proposed to measure test effectiveness, including structural coverage and non-structural coverage (which are classified according to whether considering structural elements are covered during testing). These test coverage metrics are proposed based on the fundamental assumption: they are correlated with test effectiveness. However, the fundamental assumption is still not validated sufficiently and reasonably, which brings question on the usefulness of DNN test coverage. In this work, we conducted the most comprehensive study revisiting the existing DNN test coverage from the test effectiveness perspective, which can effectively validate the fundamental assumption. Here, we carefully considered the diversity of subjects (10 pairs of models and datasets), three test effectiveness criteria, and both typical and state-of-the-art test coverage metrics. Different from all the existing studies that deliver negative conclusions on the usefulness of existing DNN test coverage, we obtained positive conclusions on their usefulness from the test effectiveness perspective in general. In particular, we found the complementary relationship between structural coverage and non-structural coverage and identified the practical usage scenarios and promising research directions for these existing test coverage.

### Datasets & Model

| Dataset         | Model        | Accuracy(%) | Domain | Network |
| --------------- | ------------ | ----------- | ------ | ------- |
| MNIST           | LeNet5       | 98.72       | image  | CNN     |
| Fashion-MNIST   | LeNet5       | 91.07       | image  | CNN     |
| CIFAR-10        | AlexNet      | 76.64       | image  | CNN     |
| CIFAR-10        | ResNet-20    | 91.21       | image  | CNN     |
| CIFAR-10        | VGG-16       | 87.41       | image  | CNN     |
| CIFAR-100       | ResNet-32    | 70.52       | image  | CNN     |
| Driving         | Dave-orig    | 90.33       | image  | CNN     |
| Driving         | Dave-dropout | 91.82       | image  | CNN     |
| Speech-Commands | DeepSpeech   | 94.53       | speech | RNN     |
| 20-Newsgroups   | TextCNN      | 77.68       | text   | CNN     |

**Notes:** We use 1 - MSE (Mean Square Error) as the accuracy of regression models. 

### Environment configuration
The version of related libraries are in the `requirements.txt`. You can install related libraries with the following command：

```shell
pip install -r requirements.txt
```

### Scripts 

```
.
├── LICENSE
├── README.md
├── coverage  
│   ├── __init__.py
|   ├── get_coverage.py
|   ├── files
│   │   ├── boundary_values
│   │   └── models
│   ├── case_study
│   │   ├── __init__.py
│   │   └── case_sc_U.py # script to investigate the influence of U on performance of SA-based coverage
│   ├── config
│   │   └── exp.conf
│   ├── raw_results # raw results of all rqs
│   │   ├── rq1-results
│   │   ├── rq2-results
│   │   └── rq3-results
│   ├── rq1 # runner of rq1
│   │   ├── __init__.py
│   │   ├── rq1_runner.py
│   │   └── rq1_script.py
│   ├── rq2
│   │   ├── __init__.py
│   │   ├── rq2_runner.py
│   │   └── rq2_script.py
│   ├── rq3
│   │   ├── __init__.py
│   │   ├── rq3_runner.py
│   │   └── rq3_script.py
│   ├── rq4
│   │   ├── __init__.py
│   │   ├── rq4_runner.py
│   │   └── rq4_script.py
│   └── tools
│       ├── __init__.py
│       ├── __pycache__
│       ├── adv_image_runner.py # to generate adversarial examples
│       ├── boundary.py # to get boundary for DeepGauge
│       ├── common_utils.py 
│       ├── coverage_utils.py # implementations of all coverages
│       ├── dataloader.py # to load dataset
│       ├── deepspeech # folder of deepspeech model
│       ├── driving # folder of driving dataset
│       ├── idc # folder of IDC
│       ├── model_utils.py # to load model
│       └── surprise_adequacy
├── requirements.txt # python packages
├── run_rq1.sh # commands to run rq1
├── run_rq2.sh # commands to run rq2
├── run_rq3.sh # commands to run rq3 with one diversity unit
└── run_rq4.sh # commands to run rq3 with saturated diversity
```
**Note:** `rq3` refers to the experiments with one diversity unit in `research question3` and `rq4` refers to experiments with saturated diversity in `research question3`


📢 If your goal is to obtain coverage information for DNN models using this artifact, without the need to reproduce our experiments, you can directly run the `coverage/get_coverage.py`.
> python -u coverage/get_coverage.py 

### Running

It's easy to reproduce our experiments since we have put all the commands of scripts in `.sh` files. Taking rq1 as an example, you can simply execute the experiment with the following command (in the parent folder of the `coverage`):

> mkdir logs
>
> bash coverage/run_rq1.sh

Alternatively, you can run the experiment on a specific model instead of all models:

> nohup python -u -m coverage.rq1.rq1_runner --size 800 --dataset_models speech-commands_deepspeech  > logs/rq1.speech\`date "+%Y%m%d%H%M%S"\`.out 2>&1 &

### Results

All results of our experiments can be reproduced by executing the commands provided in Chapter `Running`. Additionally, we uploaded all our experimental results and figures in `coverage/raw_results` due to the limited space in the paper.

### Hyperparameter Setting
Regarding `m`, `n`, and `α`, we experimented to investigate their influence on the correlation between the error-revealing capability of the test set and the coverage. According to the figures below, the results demonstrate that their settings hardly influence the conclusion of our study. We have discussed it in the paper. 
![image](https://user-images.githubusercontent.com/98631517/206959630-d8d42af1-7357-400a-8539-2526429187f7.png)
![image (1)](https://user-images.githubusercontent.com/98631517/206959706-4ede87eb-f3bf-4bfb-b49e-64379a7dd9d3.png)
![image (1)](https://user-images.githubusercontent.com/98631517/206959769-8e05079d-c740-483b-b059-7305de5f9361.png)



