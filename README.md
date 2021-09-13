# DF-TAR
# A Deep Fusion Network for Citywide Traffic AccidentRisk Prediction with Dangerous Driving Behavior

This is the implementation of the paper _published_ in TheWebConf 2021 [[Paper](https://dl.acm.org/doi/10.1145/3442381.3450003)] [[Slide](https://docs.google.com/presentation/d/1UD-e4H6eAsabZ4UBe31kajLMJ2oYdPloXOC-072oUXU/edit?usp=sharing)] [[Video](https://www.youtube.com/watch?v=XARHYVIvYPo)]

## Abstract
Because traffic accidents cause huge social and economic losses, it is of prime importance to precisely predict the traffic accident risk for reducing future accidents. In this paper, we propose a Deep Fusion network for citywide Traffic Accident Risk prediction (DF-TAR) with _dangerous driving statistics_ that contain the frequencies of various dangerous driving offences in each region. Our unique contribution is to exploit these statistics, obtained by processing the data from in- vehicle sensors, for modeling the traffic accident risk. Toward this goal, we first examine the correlation between dangerous driving offences and traffic accidents, and the analysis shows a strong correlation between them in terms of both location and time. Specifically, quick start (0.83), rapid acceleration (0.76), and sharp turn (0.76) are the top three offences that have the highest average correlation scores. We then train the DF-TAR model using the dangerous driving statistics as well as external environmental features. By extensive experiments on various frameworks, the DF-TAR model is shown to improve the accuracy of the baseline models by up to 54% by virtue of the integration of dangerous driving into the modeling of traffic accident risk.


## Note for Driving Record Data
- **Digital Tachograph (Driving Log) Data**: _cannot be publicly accessible because of the non-disclosure agreement_ 
  - For demonstration purpose, we partially provide the aggregated number of _classified_ dangerous driving cases in the `datasets` folder. 
  - If you are interested in the original data, there is a sample file provided [here](https://www.data.go.kr/en/data/15050068/fileData.do) by Korea Transportation Safety Authority.

## Example Run
For package installation: `pip install -r requirements.txt` 

For model testing: `Example Run.ipynb`

## Citation
```
@inproceedings{trirat2021dftar,
  title={DF-TAR: A Deep Fusion Network for Citywide Traffic Accident Risk Prediction with Dangerous Driving Behavior},
  author={Trirat, Patara and Lee, Jae-Gil},
  booktitle={Proceedings of the Web Conference 2021},
  pages={1146--1156},
  year={2021}
}
```
