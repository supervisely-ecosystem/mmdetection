
<div align="center" markdown>
<img src="https://i.imgur.com/GRGQrAy.png"/>  

# Serve MMDetection

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#How-To-Use-Custom-Model-Outside-The-Platform">How To Use Custom Model Outside The Platform</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/mmdetection/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmdetection)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/mmdetection/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/mmdetection/serve.png)](https://supervisely.com)

</div>


# Overview

Serve MMDetection model as Supervisely Application. MMDetection is an open source toolbox based on PyTorch. Learn more about MMDetection and available models [here](https://github.com/open-mmlab/mmdetection).

Application key points:
- All Object Detection and Instance Segmentation models from MM Toolbox are available
- Deployed on GPU or CPU


# How to Run

**Step 1.** Add [Serve MMDetection](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/mmdetection/serve) to your team

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection/serve" src="https://i.imgur.com/R5nCKt2.png" width="350px" style='padding-bottom: 10px'/>

**Step 2.** Run the application from Plugins & Apps page

<img src="https://i.imgur.com/2FvYoTy.png" width="80%" style='padding-top: 10px'>  

# How to Use

**Pretrained models**

**Step 1.** Select Deep Learning problem to solve

<img src="https://i.imgur.com/fKoqM3Q.png" width="80%">  

**Step 2.** Select architecture, pretrained model, deploying device and press the **Serve** button

<img src="https://i.imgur.com/PFmbbtp.png" width="80%">  

**Step 3.** Wait for the model to deploy

<img src="https://i.imgur.com/1H2Uwsd.png" width="80%">

**Custom models**

Model and directory structure must be acquired via [Train MMDetection](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/mmdetection/train) app or manually created with the same directory structure

<img src="https://github.com/supervisely-ecosystem/mmdetection/releases/download/v0.0.1/copy-path-min.gif"/>  


# How To Use Custom Model Outside The Platform

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. You just need to download model weights (.pth) and a config file from Team Files, and then you can build and use the model as a normal model in mmdetection. See this [Jupyter Notebook](https://github.com/supervisely-ecosystem/mmdetection/blob/main/inference_outside_supervisely.ipynb) for details.


# Related Apps


- [Train MMDetection](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/mmdetection/train) - app allows to play with different inference options, monitor metrics charts in real time, and save training artifacts to Team Files.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection/train" src="https://i.imgur.com/92PoYV7.png" height="60px" margin-bottom="20px"/>

- [Apply NN to Images Project](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" height="70px" margin-bottom="20px"/>  

- [Apply NN to Videos Project](https://ecosystem.supervisely.com/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [NN Image Labeling](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>


# Acknowledgment

This app is based on the great work `MMDetection` ([github](https://github.com/open-mmlab/mmdetection)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection?style=social)

