Pattern Graph Insight-State Difference Mutual Information
=======
``PGI-SDMI algorithm is used to select the RIFs to improve the prediction accuracy of marine accident severity. This is the code repository for our publication ''Prediction of the severity of marine accidents using improved machine learning'' that is published on Transportation Research Part E.``
## Motivation
This study aims to optimise the predictive performance of the model by utilising FS techniques to enhance the prediction accuracy. Among the commonly employed FS algorithms, methods such as Removing Features with Low Variance (RFLV) and Traditional Mutual Information (TMI) are prevalent for feature subset selection in classification problems. However, factors such as problem dependency, feature complexity, and combinatorial explosion can hinder these methods from yielding the optimal feature subset for a specific problem, thus limiting improvements in model prediction performance. Consequently, the primary objective of this study is to develop an advanced FS method aimed at maximising prediction accuracy, which serves as the core focus of this research.
## Methodology
* PGI-SDMI algorithm structure:
![image](https://github.com/FengYinLeo/PGI-SDMI/assets/108978874/3d7d9a94-1fda-49a7-97c8-eff70c003136)
This figure shows the main structure of the PGI-SDMI algorithm. It mainly includes two stages: Pattern Graph Insight (PGI) and State Difference Mutual Information (SDMI). Initially, the PGI examines and ranks features, capturing their intricate interactions within a complex network landscape. Subsequently, the SDMI focuses on assessing the significance of features excluded from the PGI ranking across different target variable states, generating an additional feature ranking. By combining these rankings, the comprehensive approach ensures a holistic consideration of both complex feature interactions (identified by the PGI) and nuanced feature importance across diverse states (analysed by the SDMI), thereby facilitating the identification of crucial features with multifaceted relevance.
## Run
* We add "PGI.py", "SDMI.py", and "PGI_SDMI.py" files to folder "feature selection algorithm" to implement the proposed PGI, SDMI, and PGI-SDMI feature selection algorithms respectively.
* We give an example of calling this research algorithm for feature selection in the "Run_the_example.ipynb" file.
## Citation
Please cite our paper if this repository is helpful for your study.
{Yinwei Feng, Xinjian Wang, Qilei Chen, Zaili Yang, Jin Wang, Huanhuan Li, Guoqing Xia, Zhengjiang Liu,Prediction of the severity of marine accidents using improved machine learning,Transportation Research Part E: Logistics and Transportation Review,2024,103647.https://doi.org/10.1016/j.tre.2024.103647.}
## License
This project is released under the MIT license.
