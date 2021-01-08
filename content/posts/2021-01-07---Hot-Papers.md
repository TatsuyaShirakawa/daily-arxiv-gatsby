---
title: Hot Papers 2021-01-07
date: 2021-01-08T09:21:40.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-07"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-07"
socialImage: "/media/flying-marine.jpg"

---

# 1. AutoDropout: Learning Dropout Patterns to Regularize Deep Networks

Hieu Pham, Quoc V. Le

- retweets: 595, favorites: 163 (01/08/2021 09:21:40)

- links: [abs](https://arxiv.org/abs/2101.01761) | [pdf](https://arxiv.org/pdf/2101.01761)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Neural networks are often over-parameterized and hence benefit from aggressive regularization. Conventional regularization methods, such as Dropout or weight decay, do not leverage the structures of the network's inputs and hidden states. As a result, these conventional methods are less effective than methods that leverage the structures, such as SpatialDropout and DropBlock, which randomly drop the values at certain contiguous areas in the hidden states and setting them to zero. Although the locations of dropout areas random, the patterns of SpatialDropout and DropBlock are manually designed and fixed. Here we propose to learn the dropout patterns. In our method, a controller learns to generate a dropout pattern at every channel and layer of a target network, such as a ConvNet or a Transformer. The target network is then trained with the dropout pattern, and its resulting validation performance is used as a signal for the controller to learn from. We show that this method works well for both image recognition on CIFAR-10 and ImageNet, as well as language modeling on Penn Treebank and WikiText-2. The learned dropout patterns also transfers to different tasks and datasets, such as from language model on Penn Treebank to Engligh-French translation on WMT 2014. Our code will be available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AutoDropout: Learning Dropout Patterns to Regularize Deep Networks<br>pdf: <a href="https://t.co/ET6O8dg9AF">https://t.co/ET6O8dg9AF</a><br>abs: <a href="https://t.co/obFBnuhdqu">https://t.co/obFBnuhdqu</a> <a href="https://t.co/Di5eMzvZYC">pic.twitter.com/Di5eMzvZYC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347004696171573248?ref_src=twsrc%5Etfw">January 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Line Segment Detection Using Transformers without Edges

Yifan Xu, Weijian Xu, David Cheung, Zhuowen Tu

- retweets: 600, favorites: 150 (01/08/2021 09:21:40)

- links: [abs](https://arxiv.org/abs/2101.01909) | [pdf](https://arxiv.org/pdf/2101.01909)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we present a holistically end-to-end algorithm for line segment detection with transformers that is post-processing and heuristics-guided intermediate processing (edge/junction/region detection) free. Our method, named LinE segment TRansformers (LETR), tackles the three main problems in this domain, namely edge element detection, perceptual grouping, and holistic inference by three highlights in detection transformers (DETR) including tokenized queries with integrated encoding and decoding, self-attention, and joint queries respectively. The transformers learn to progressively refine line segments through layers of self-attention mechanism skipping the heuristic design in the previous line segmentation algorithms. We equip multi-scale encoder/decoder in the transformers to perform fine-grained line segment detection under a direct end-point distance loss that is particularly suitable for entities such as line segments that are not conveniently represented by bounding boxes. In the experiments, we show state-of-the-art results on Wireframe and YorkUrban benchmarks. LETR points to a promising direction for joint end-to-end detection of general entities beyond the standard object bounding box representation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Line Segment Detection Using Transformers without Edges<br>pdf: <a href="https://t.co/1jDGpB7ZHA">https://t.co/1jDGpB7ZHA</a><br>abs: <a href="https://t.co/TERaiNAvyW">https://t.co/TERaiNAvyW</a> <a href="https://t.co/X4YPxiVRhW">pic.twitter.com/X4YPxiVRhW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347005937387757568?ref_src=twsrc%5Etfw">January 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Transformer Guided Geometry Model for Flow-Based Unsupervised Visual  Odometry

Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, Wanqing Li

- retweets: 62, favorites: 48 (01/08/2021 09:21:41)

- links: [abs](https://arxiv.org/abs/2101.02143) | [pdf](https://arxiv.org/pdf/2101.02143)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Existing unsupervised visual odometry (VO) methods either match pairwise images or integrate the temporal information using recurrent neural networks over a long sequence of images. They are either not accurate, time-consuming in training or error accumulative. In this paper, we propose a method consisting of two camera pose estimators that deal with the information from pairwise images and a short sequence of images respectively. For image sequences, a Transformer-like structure is adopted to build a geometry model over a local temporal window, referred to as Transformer-based Auxiliary Pose Estimator (TAPE). Meanwhile, a Flow-to-Flow Pose Estimator (F2FPE) is proposed to exploit the relationship between pairwise images. The two estimators are constrained through a simple yet effective consistency loss in training. Empirical evaluation has shown that the proposed method outperforms the state-of-the-art unsupervised learning-based methods by a large margin and performs comparably to supervised and traditional ones on the KITTI and Malaga dataset.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Transformer Guided Geometry Model for Flow-Based Unsupervised Visual Odometry<br>pdf: <a href="https://t.co/KT1VZrWJWM">https://t.co/KT1VZrWJWM</a><br>abs: <a href="https://t.co/sE8jiq0HDF">https://t.co/sE8jiq0HDF</a> <a href="https://t.co/CQ9qCwm2pF">pic.twitter.com/CQ9qCwm2pF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347006876173676545?ref_src=twsrc%5Etfw">January 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Latency Overhead of ROS2 for Modular Time-Critical Systems

Tobias Kronauer, Joshwa Pohlmann, Maximilian Matthe, Till Smejkal, Gerhard Fettweis

- retweets: 49, favorites: 23 (01/08/2021 09:21:41)

- links: [abs](https://arxiv.org/abs/2101.02074) | [pdf](https://arxiv.org/pdf/2101.02074)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent) | [cs.SE](https://arxiv.org/list/cs.SE/recent)

Robot Operating System 2 (ROS2) targets distributed real-time systems. Especially in tight real-time control loops, latency in data processing and communication can lead to instabilities. As ROS2 encourages splitting of the data-processing pipelines into several modules, it is important to understand the latency implications of such modularization. In this paper, we investigate the end-to-end latency of ROS2 data-processing pipeline with different Data Distribution Service (DDS) middlewares. In addition, we profile the ROS2 stack and point out latency bottlenecks. Our findings indicate that end-to-end latency strongly depends on the used DDS middleware. Moreover, we show that ROS2 can lead to 50 % latency overhead compared to using low-level DDS communications. Our results imply guidelines for designing modular ROS2 architectures and indicate possibilities for reducing the ROS2 overhead.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Latency Overhead of ROS2 for Modular Time-Critical Systems&quot; -- this paper examines end-to-end latency in ROS 2 using different RMW implementations. <br><br>Check it out: <a href="https://t.co/5vxC1RKjJd">https://t.co/5vxC1RKjJd</a></p>&mdash; Open Robotics (@OpenRoboticsOrg) <a href="https://twitter.com/OpenRoboticsOrg/status/1347202534851670016?ref_src=twsrc%5Etfw">January 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Generating Masks from Boxes by Mining Spatio-Temporal Consistencies in  Videos

Bin Zhao, Goutam Bhat, Martin Danelljan, Luc Van Gool, Radu Timofte

- retweets: 42, favorites: 29 (01/08/2021 09:21:41)

- links: [abs](https://arxiv.org/abs/2101.02196) | [pdf](https://arxiv.org/pdf/2101.02196)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Segmenting objects in videos is a fundamental computer vision task. The current deep learning based paradigm offers a powerful, but data-hungry solution. However, current datasets are limited by the cost and human effort of annotating object masks in videos. This effectively limits the performance and generalization capabilities of existing video segmentation methods. To address this issue, we explore weaker form of bounding box annotations.   We introduce a method for generating segmentation masks from per-frame bounding box annotations in videos. To this end, we propose a spatio-temporal aggregation module that effectively mines consistencies in the object and background appearance across multiple frames. We use our resulting accurate masks for weakly supervised training of video object segmentation (VOS) networks. We generate segmentation masks for large scale tracking datasets, using only their bounding box annotations. The additional data provides substantially better generalization performance leading to state-of-the-art results in both the VOS and more challenging tracking domain.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating Masks from Boxes by Mining Spatio-Temporal Consistencies in Videos<br>pdf: <a href="https://t.co/TIqxJyzm0P">https://t.co/TIqxJyzm0P</a><br>abs: <a href="https://t.co/vKLdTdfSXM">https://t.co/vKLdTdfSXM</a> <a href="https://t.co/NRuEPLLfLd">pic.twitter.com/NRuEPLLfLd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347016440323440643?ref_src=twsrc%5Etfw">January 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Weakly-Supervised Multi-Face 3D Reconstruction

Jialiang Zhang, Lixiang Lin, Jianke Zhu, Steven C.H. Hoi

- retweets: 24, favorites: 35 (01/08/2021 09:21:41)

- links: [abs](https://arxiv.org/abs/2101.02000) | [pdf](https://arxiv.org/pdf/2101.02000)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

3D face reconstruction plays a very important role in many real-world multimedia applications, including digital entertainment, social media, affection analysis, and person identification. The de-facto pipeline for estimating the parametric face model from an image requires to firstly detect the facial regions with landmarks, and then crop each face to feed the deep learning-based regressor. Comparing to the conventional methods performing forward inference for each detected instance independently, we suggest an effective end-to-end framework for multi-face 3D reconstruction, which is able to predict the model parameters of multiple instances simultaneously using single network inference. Our proposed approach not only greatly reduces the computational redundancy in feature extraction but also makes the deployment procedure much easier using the single network model. More importantly, we employ the same global camera model for the reconstructed faces in each image, which makes it possible to recover the relative head positions and orientations in the 3D scene. We have conducted extensive experiments to evaluate our proposed approach on the sparse and dense face alignment tasks. The experimental results indicate that our proposed approach is very promising on face alignment tasks without fully-supervision and pre-processing like detection and crop. Our implementation is publicly available at \url{https://github.com/kalyo-zjl/WM3DR}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Weakly-Supervised Multi-Face 3D Reconstruction<br>pdf: <a href="https://t.co/wZXdmnW6xK">https://t.co/wZXdmnW6xK</a><br>abs: <a href="https://t.co/Bm4Wqoz3bT">https://t.co/Bm4Wqoz3bT</a><br>github: <a href="https://t.co/asXE9VVHGv">https://t.co/asXE9VVHGv</a> <a href="https://t.co/i6C4lE4egd">pic.twitter.com/i6C4lE4egd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347013173283606528?ref_src=twsrc%5Etfw">January 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



