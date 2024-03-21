# Taillight_action_classification
Taillight action classification for for the paper **“Action-State Joint Learning-Based Vehicle Taillight Recognition in Diverse Actual Traffic Scenes”.**
If you use this project for your academic research, please consider citing the follow
+ Song, Wenjie, et al. "Action-state joint learning-based vehicle taillight recognition in diverse actual traffic scenes." IEEE transactions on intelligent transportation systems 23.10 (2022): 18088-18099.

Bibtex

```
@article{song2022action,
  title={Action-state joint learning-based vehicle taillight recognition in diverse actual traffic scenes},
  author={Song, Wenjie and Liu, Shixian and Zhang, Ting and Yang, Yi and Fu, Mengyin},
  journal={IEEE transactions on intelligent transportation systems},
  volume={23},
  number={10},
  pages={18088--18099},
  year={2022},
  publisher={IEEE}
}
```



## Dependencies
This project has been tested on Ubuntu 18.04. 

Install required packages

+ python==3.8
+ cuda==11.1
+ torch==1.10.0
+ torchaudio==0.10.0
+ torchvision==0.11.0
+ pandas==2.0.3

## Train
Train the network
`python ALSTM.py`

## Test
Test the network
`python testAlstm.py`


This project is my work during my master's degree. I have graduated now. Due to the loss of the original data and code, the source code is irrecoverable. I found and recovered some of them according to open source data, and achieved partial classification effects. Effects and codes are for reference only. Your own dataset path needs to be added to `iterater.py`
