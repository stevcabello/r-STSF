# The Randomized-Supervised Time Series Forest (r-STSF)

This repository contains the source code for the r-STSF algorithm, which derives from the article:

Nestor Cabello, Elham Naghizade, Jianzhong Qi, and Lars Kulik. [Fast, Accurate and Interpretable Time Series Classification Through Randomization.](https://arxiv.org/abs/2105.14876) arXiv e-prints, [arXiv:2105.14876](https://arxiv.org/abs/2105.14876) (2021)


# Abstract

Time series classification (TSC) aims to predict the class label of a given time series, which is critical to a rich set of application areas such as economics and medicine. State-of-the-art TSC methods have mostly focused on classification accuracy and efficiency, without considering the interpretability of their classifications, which is an important property required by modern applications such as appliance modeling and legislation such as the European General Data Protection Regulation. To address this gap, we propose a novel TSC method – the Randomized-Supervised Time Series Forest (r-STSF). r-STSF is highly efficient, achieves state-of-the-art classification accuracy and enables interpretability. r-STSF takes an efficient interval-based approach to classify time series according to aggregate values of discriminatory sub-series (intervals). To achieve state-of-the-art accuracy, r-STSF builds an ensemble of randomized trees using the discriminatory sub-series. It uses four time series representations, nine aggregation functions and a supervised binary-inspired search combined with a feature ranking metric to identify highly discriminatory sub-series. The discriminatory sub-series enable interpretable classifications. Experiments on extensive real datasets show that r-STSF achieves state-of-the-art accuracy while being orders of magnitude faster than most existing TSC methods. It is the only classifier from the state-of-the-art group that enables interpretability. Our findings also highlight that r-STSF is the best TSC method when classifying complex time series datasets.


# Usage

For a working example run --> code/r-STSF.ipynb

For a demo on r-STSF's interpretability run --> code/r-STSF-interpretability demo.ipynb
