# The Randomized-Supervised Time Series Forest (r-STSF)

This repository contains the source code for the r-STSF time series classifier, which derives from the article:

Cabello, N., Naghizade, E., Qi, J. et al. Fast, accurate and explainable time series classification through randomization. Data Min Knowl Disc (2023). https://doi.org/10.1007/s10618-023-00978-w



The arXiv version of r-STSF can be found in:
Nestor Cabello, Elham Naghizade, Jianzhong Qi, and Lars Kulik. [Fast, Accurate and Interpretable Time Series Classification Through Randomization.](https://arxiv.org/abs/2105.14876) arXiv e-prints, [arXiv:2105.14876](https://arxiv.org/abs/2105.14876) (2021)

The folder **history_arxiv** contains the results presented in the arXiv version of r-STSF.

# Abstract

Time series classification (TSC) aims to predict the class label of a given time series, which is critical to a rich set of application areas such as economics and medicine. State-of-the-art TSC methods have mostly focused on classification accuracy, without considering classification speed. However, efficiency is important for big data analysis. Datasets with a large training size or long series challenge the use of the current highly accurate methods, because they are usually computationally expensive. Similarly, classification explainability, which is an important property required by modern big data applications such as appliance modeling and legislation such as the European General Data Protection Regulation, has received little attention. To address these gaps, we propose a novel TSC method – the Randomized-Supervised Time Series Forest (r-STSF). r-STSF is extremely fast and achieves state-of-the-art classification accuracy. It is an efficient interval-based approach that classifies time series according to aggregate values of the discriminatory sub-series (intervals). To achieve state-of-the-art accuracy, r-STSF builds an ensemble of randomized trees using the discriminatory sub-series. It uses four time series representations, nine aggregation functions and a supervised binary-inspired search combined with a feature ranking metric to identify highly discriminatory sub-series. The discriminatory sub-series enable explainable classifications. Experiments on extensive datasets show that r-STSF achieves state-of-the-art accuracy while being orders of magnitude faster than most existing TSC methods and enabling for explanations on the classifier decision.


# Usage

For a working example run --> code/r-STSF.ipynb

For a demo on r-STSF's explanability run --> code/r-STSF_explainability_demo.ipynb
