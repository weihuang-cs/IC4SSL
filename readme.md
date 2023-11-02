## Inherit Consistency


### Usage

1. Clone the repo:
```
git clone https://github.com/HiLab-git/DTC.git 
cd DTC
```
2. Put the data in [data/2018LA_Seg_Training Set](https://github.com/Luoxd1996/DTC/tree/master/data/2018LA_Seg_Training%20Set).

3. Train the model
```
cd code
python train_la_dtc.py
```

4. Test the model
```
python test_LA.py
```
Our pre-trained models are saved in the model dir [DTC_model](https://github.com/Luoxd1996/DTC/tree/master/model)

## Acknowledgement
* This code is adapted from [SSL4MIS]([https://github.com/yulequan/UA-MT](https://github.com/HiLab-git/SSL4MIS)), [SS-Net](https://github.com/ycwu1997/SS-Net)

