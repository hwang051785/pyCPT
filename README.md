# pyCPT #

> A Bayesian unsupervised learning method for geotechnical soil stratification identification.

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)]()
[![Python 3.6.x](https://img.shields.io/badge/Python-3.6.x-blue.svg)]()

## Contents

+ [Introduction](#introduction)
+ [Examples](#examples)
  - [1D: Segmentation of geophysical well log data](#NGES example)  
+ [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Cloning directly from GitHub](#cloning-directly-from-github)
+ [Getting Started](#getting-started)
+ [References](#references)
+ [Contact](#contact)

## Introduction

This package presents a novel perspective to understand the spatial and statistical patterns of a cone penetration dataset and an automatic approach to identify soil stratification. Both local consistency in physical space (i.e., along depth) and statistical similarity in feature space (i.e., logQt – logFr space or the Robertson chart) between data points are considered simultaneously. The proposed approach is, in essence, consist of two parts: 1) a pattern detection approach using Bayesian inferential framework, and 2) a pattern interpretation protocol using Robertson chart. The first part is the mathematical core of the proposed approach, which infers both spatial pattern in physical space and statistical pattern in feature space from the input dataset; the second part converts the abstract patterns into intuitive spatial configurations of multiple soil layers having different soil behavior types. The advantages of this approach include probabilistic soil classification, and identifying soil stratification in an automatic and fully unsupervised manner. This approach has been tested using various datasets including both synthetic and real-world CPT soundings.

The package is based on the algorithm developed by [Wang et al., 2017](https://link.springer.com/article/10.1007/s11004-016-9663-9) and combines Hidden Markov Random Fields with Gaussian Mixture Models in a Bayesian inference framework.

## Examples

You can try out this method by using an interactive Jupyter Notebook in your own web browser, enabled by Binder:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/hwang051785/pyCPT/master?filepath=jupyter_notebooks%2FpyCPT_test.ipynb)

## Installation

As the library is still in early development, the current way to install it is to clone this repository
and then import it manually to your projects.

#### Dependencies

pyCPT depends on the following existing packages:

* `numpy` for efficient numerical implementation
* `scikit-learn` for finite mixture models
* `scipy` for its statistical functionality
* `matplotlib` for plotting
* `tqdm` provides convenient progress meters

#### Cloning directly from GitHub

First clone the repository using the command (or by manually downloading the zip file from the GitHub page)

    git clone https://github.com/hwang051785/pyCPT.git

then append the path to the repository:
    
    import sys
    sys.path.append("path/to/cloned/repository")
    
to import the module:

    import pre_process
    import pyCPT
    
## Getting Started

Construct data structure

    cpt = pyCPT.read_cpt_data('path/to/dataset')
    
Model selection for the optimal number of clusters
    
    mod_sel = pre_process.model_selection(cpt.feat, 10, plot=True)
    
Extract soil segments
    
    pyCPT.segmentation(cpt, n=200, n_labels=mod_sel[1], start_iter=100)
    
Soil interpretation
    
    layer_info = pyCPT.detect_layers(cpt)
    
## References

* Wang, H., Wellmann, J. F., Li, Z., Wang, X., & Liang, R. Y. (2017). A Segmentation Approach for Stochastic Geological Modeling Using Hidden Markov Random Fields. Mathematical Geosciences, 49(2), 145-177.
    
