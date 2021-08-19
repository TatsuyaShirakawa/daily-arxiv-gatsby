---
title: Hot Papers 2021-08-19
date: 2021-08-20T06:55:02.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-19"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-19"
socialImage: "/media/flying-marine.jpg"

---

# 1. Global Pooling, More than Meets the Eye: Position Information is Encoded  Channel-Wise in CNNs

Md Amirul Islam, Matthew Kowal, Sen Jia, Konstantinos G. Derpanis, Neil D. B. Bruce

- retweets: 798, favorites: 177 (08/20/2021 06:55:02)

- links: [abs](https://arxiv.org/abs/2108.07884) | [pdf](https://arxiv.org/pdf/2108.07884)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we challenge the common assumption that collapsing the spatial dimensions of a 3D (spatial-channel) tensor in a convolutional neural network (CNN) into a vector via global pooling removes all spatial information. Specifically, we demonstrate that positional information is encoded based on the ordering of the channel dimensions, while semantic information is largely not. Following this demonstration, we show the real world impact of these findings by applying them to two applications. First, we propose a simple yet effective data augmentation strategy and loss function which improves the translation invariance of a CNN's output. Second, we propose a method to efficiently determine which channels in the latent representation are responsible for (i) encoding overall position information or (ii) region-specific positions. We first show that semantic segmentation has a significant reliance on the overall position channels to make predictions. We then show for the first time that it is possible to perform a `region-specific' attack, and degrade a network's performance in a particular part of the input. We believe our findings and demonstrated applications will benefit research areas concerned with understanding the characteristics of CNNs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs<br>Authors: all tagged + Neil D. B. Bruce<br>tl;dr: CNNs with global average pooling encode the position in channel order. Nicely formulated and tested hypothesis  <a href="https://t.co/3qE46jprMS">https://t.co/3qE46jprMS</a> <a href="https://t.co/vi5aZWiEh7">pic.twitter.com/vi5aZWiEh7</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1428324571992711175?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Accepted <a href="https://twitter.com/hashtag/ICCV2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICCV2021</a> paper!<br><br>Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs<br><br>TLDR: CNNs with global average pooling layers encode position information in the channel order! 🧐😮<a href="https://t.co/2je3zeZFrN">https://t.co/2je3zeZFrN</a> <a href="https://t.co/lrrPo6YIET">pic.twitter.com/lrrPo6YIET</a></p>&mdash; Matthew Kowal (@MatthewKowal9) <a href="https://twitter.com/MatthewKowal9/status/1428386120358838274?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Deep Hybrid Self-Prior for Full 3D Mesh Generation

Xingkui Wei, Zhengqing Chen, Yanwei Fu, Zhaopeng Cui, Yinda Zhang

- retweets: 708, favorites: 136 (08/20/2021 06:55:02)

- links: [abs](https://arxiv.org/abs/2108.08017) | [pdf](https://arxiv.org/pdf/2108.08017)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a deep learning pipeline that leverages network self-prior to recover a full 3D model consisting of both a triangular mesh and a texture map from the colored 3D point cloud. Different from previous methods either exploiting 2D self-prior for image editing or 3D self-prior for pure surface reconstruction, we propose to exploit a novel hybrid 2D-3D self-prior in deep neural networks to significantly improve the geometry quality and produce a high-resolution texture map, which is typically missing from the output of commodity-level 3D scanners. In particular, we first generate an initial mesh using a 3D convolutional neural network with 3D self-prior, and then encode both 3D information and color information in the 2D UV atlas, which is further refined by 2D convolutional neural networks with the self-prior. In this way, both 2D and 3D self-priors are utilized for the mesh and texture recovery. Experiments show that, without the need of any additional training data, our method recovers the 3D textured mesh model of high quality from sparse input, and outperforms the state-of-the-art methods in terms of both the geometry and texture quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Hybrid Self-Prior for Full 3D Mesh Generation<br>abs: <a href="https://t.co/REt7IRQik5">https://t.co/REt7IRQik5</a><br>project page: <a href="https://t.co/WedpbgJvJn">https://t.co/WedpbgJvJn</a><br><br>without the need of any additional training data,<br>method recovers the 3D textured mesh model of high quality from sparse input <a href="https://t.co/iou3kGBfHq">pic.twitter.com/iou3kGBfHq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428160381231501316?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. ARCH++: Animation-Ready Clothed Human Reconstruction Revisited

Tong He, Yuanlu Xu, Shunsuke Saito, Stefano Soatto, Tony Tung

- retweets: 483, favorites: 131 (08/20/2021 06:55:03)

- links: [abs](https://arxiv.org/abs/2108.07845) | [pdf](https://arxiv.org/pdf/2108.07845)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present ARCH++, an image-based method to reconstruct 3D avatars with arbitrary clothing styles. Our reconstructed avatars are animation-ready and highly realistic, in both the visible regions from input views and the unseen regions. While prior work shows great promise of reconstructing animatable clothed humans with various topologies, we observe that there exist fundamental limitations resulting in sub-optimal reconstruction quality. In this paper, we revisit the major steps of image-based avatar reconstruction and address the limitations with ARCH++. First, we introduce an end-to-end point based geometry encoder to better describe the semantics of the underlying 3D human body, in replacement of previous hand-crafted features. Second, in order to address the occupancy ambiguity caused by topological changes of clothed humans in the canonical pose, we propose a co-supervising framework with cross-space consistency to jointly estimate the occupancy in both the posed and canonical spaces. Last, we use image-to-image translation networks to further refine detailed geometry and texture on the reconstructed surface, which improves the fidelity and consistency across arbitrary viewpoints. In the experiments, we demonstrate improvements over the state of the art on both public benchmarks and user studies in reconstruction quality and realism.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ARCH++: Animation-Ready Clothed Human Reconstruction Revisited<br>pdf: <a href="https://t.co/i0hTesxVFM">https://t.co/i0hTesxVFM</a><br>abs: <a href="https://t.co/AqAAz6Jfo1">https://t.co/AqAAz6Jfo1</a><br><br>ARCH++ produces results which have high-level<br>fidelity and are animation-ready for many AR/VR applications <a href="https://t.co/oTLGUdAJmE">pic.twitter.com/oTLGUdAJmE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428158858225229825?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Pixel-Perfect Structure-from-Motion with Featuremetric Refinement

Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson, Marc Pollefeys

- retweets: 432, favorites: 153 (08/20/2021 06:55:03)

- links: [abs](https://arxiv.org/abs/2108.08291) | [pdf](https://arxiv.org/pdf/2108.08291)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Finding local features that are repeatable across multiple views is a cornerstone of sparse 3D reconstruction. The classical image matching paradigm detects keypoints per-image once and for all, which can yield poorly-localized features and propagate large errors to the final geometry. In this paper, we refine two key steps of structure-from-motion by a direct alignment of low-level image information from multiple views: we first adjust the initial keypoint locations prior to any geometric estimation, and subsequently refine points and camera poses as a post-processing. This refinement is robust to large detection noise and appearance changes, as it optimizes a featuremetric error based on dense features predicted by a neural network. This significantly improves the accuracy of camera poses and scene geometry for a wide range of keypoint detectors, challenging viewing conditions, and off-the-shelf deep features. Our system easily scales to large image collections, enabling pixel-perfect crowd-sourced localization at scale. Our code is publicly available at https://github.com/cvg/pixel-perfect-sfm as an add-on to the popular SfM software COLMAP.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pixel-Perfect Structure-from-Motion with Featuremetric Refinement <a href="https://t.co/GpdlXuJLPl">https://t.co/GpdlXuJLPl</a><br><br>Cool new <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> paper from ETHZ showing how to improve over classical detect-once-and-never-refine local features. Also now a part of the popular colmap <a href="https://twitter.com/hashtag/SfM?src=hash&amp;ref_src=twsrc%5Etfw">#SfM</a> library. <a href="https://twitter.com/hashtag/iccv2021?src=hash&amp;ref_src=twsrc%5Etfw">#iccv2021</a> <a href="https://t.co/129cgReK8g">pic.twitter.com/129cgReK8g</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1428185462611619845?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pixel-Perfect Structure-from-Motion with Featuremetric Refinement<br>tl;dr: refine sparse matches with dense descriptors+optimization, twice: before RANSAC and after SfM.<br><br>Philipp Lindenberger <a href="https://twitter.com/pesarlin?ref_src=twsrc%5Etfw">@pesarlin</a> Viktor Larsson <a href="https://twitter.com/mapo1?ref_src=twsrc%5Etfw">@mapo1</a> <a href="https://t.co/oymHcEVMwI">https://t.co/oymHcEVMwI</a><br>1/2 <a href="https://t.co/m15Et5fbvz">pic.twitter.com/m15Et5fbvz</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1428249990246305794?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Overfitting the Data: Compact Neural Video Delivery via Content-aware  Feature Modulation

Jiaming Liu, Ming Lu, Kaixin Chen, Xiaoqi Li, Shizun Wang, Zhaoqing Wang, Enhua Wu, Yurong Chen, Chuang Zhang, Ming Wu

- retweets: 288, favorites: 67 (08/20/2021 06:55:03)

- links: [abs](https://arxiv.org/abs/2108.08202) | [pdf](https://arxiv.org/pdf/2108.08202)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Internet video delivery has undergone a tremendous explosion of growth over the past few years. However, the quality of video delivery system greatly depends on the Internet bandwidth. Deep Neural Networks (DNNs) are utilized to improve the quality of video delivery recently. These methods divide a video into chunks, and stream LR video chunks and corresponding content-aware models to the client. The client runs the inference of models to super-resolve the LR chunks. Consequently, a large number of models are streamed in order to deliver a video. In this paper, we first carefully study the relation between models of different chunks, then we tactfully design a joint training framework along with the Content-aware Feature Modulation (CaFM) layer to compress these models for neural video delivery. {\bf With our method, each video chunk only requires less than $1\% $ of original parameters to be streamed, achieving even better SR performance.} We conduct extensive experiments across various SR backbones, video time length, and scaling factors to demonstrate the advantages of our method. Besides, our method can be also viewed as a new approach of video coding. Our primary experiments achieve better video quality compared with the commercial H.264 and H.265 standard under the same storage cost, showing the great potential of the proposed method. Code is available at:\url{https://github.com/Neural-video-delivery/CaFM-Pytorch-ICCV2021}

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Overfitting the Data: Compact Neural Video Delivery via Content-aware Feature Modulation<br>abs: <a href="https://t.co/WaecE0Ignq">https://t.co/WaecE0Ignq</a><br>github: <a href="https://t.co/d7HIwfAxiP">https://t.co/d7HIwfAxiP</a><br><br>each video chunk only requires less than 1% of original parameters to be streamed, achieving even better SR performance <a href="https://t.co/pZ7QsD6b6P">pic.twitter.com/pZ7QsD6b6P</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428183648914886668?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. AdapterHub Playground: Simple and Flexible Few-Shot Learning with  Adapters

Tilman Beck, Bela Bohlender, Christina Viehmann, Vincent Hane, Yanik Adamson, Jaber Khuri, Jonas Brossmann, Jonas Pfeiffer, Iryna Gurevych

- retweets: 100, favorites: 39 (08/20/2021 06:55:03)

- links: [abs](https://arxiv.org/abs/2108.08103) | [pdf](https://arxiv.org/pdf/2108.08103)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The open-access dissemination of pretrained language models through online repositories has led to a democratization of state-of-the-art natural language processing (NLP) research. This also allows people outside of NLP to use such models and adapt them to specific use-cases. However, a certain amount of technical proficiency is still required which is an entry barrier for users who want to apply these models to a certain task but lack the necessary knowledge or resources. In this work, we aim to overcome this gap by providing a tool which allows researchers to leverage pretrained models without writing a single line of code. Built upon the parameter-efficient adapter modules for transfer learning, our AdapterHub Playground provides an intuitive interface, allowing the usage of adapters for prediction, training and analysis of textual data for a variety of NLP tasks. We present the tool's architecture and demonstrate its advantages with prototypical use-cases, where we show that predictive performance can easily be increased in a few-shot learning scenario. Finally, we evaluate its usability in a user study. We provide the code and a live interface at https://adapter-hub.github.io/playground.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do you want to use SOTA NLP models *without writing a single line of code*?<br><br>Check out the AdapterHub Playground!<br><br>Predictions are directly written into your google sheets! <br><br>📜: <a href="https://t.co/ZUrvu8yHVC">https://t.co/ZUrvu8yHVC</a><br>🌐: <a href="https://t.co/GEwtp4iFdQ">https://t.co/GEwtp4iFdQ</a><br>📽️: <a href="https://t.co/yjfHClF9TP">https://t.co/yjfHClF9TP</a><br><br>More applications 👇 <a href="https://t.co/jWfWuEDLXu">pic.twitter.com/jWfWuEDLXu</a></p>&mdash; AdapterHub (@AdapterHub) <a href="https://twitter.com/AdapterHub/status/1428294107617693707?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Moser Flow: Divergence-based Generative Modeling on Manifolds

Noam Rozen, Aditya Grover, Maximilian Nickel, Yaron Lipman

- retweets: 72, favorites: 50 (08/20/2021 06:55:03)

- links: [abs](https://arxiv.org/abs/2108.08052) | [pdf](https://arxiv.org/pdf/2108.08052)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We are interested in learning generative models for complex geometries described via manifolds, such as spheres, tori, and other implicit surfaces. Current extensions of existing (Euclidean) generative models are restricted to specific geometries and typically suffer from high computational costs. We introduce Moser Flow (MF), a new class of generative models within the family of continuous normalizing flows (CNF). MF also produces a CNF via a solution to the change-of-variable formula, however differently from other CNF methods, its model (learned) density is parameterized as the source (prior) density minus the divergence of a neural network (NN). The divergence is a local, linear differential operator, easy to approximate and calculate on manifolds. Therefore, unlike other CNFs, MF does not require invoking or backpropagating through an ODE solver during training. Furthermore, representing the model density explicitly as the divergence of a NN rather than as a solution of an ODE facilitates learning high fidelity densities. Theoretically, we prove that MF constitutes a universal density approximator under suitable assumptions. Empirically, we demonstrate for the first time the use of flow models for sampling from general curved surfaces and achieve significant improvements in density estimation, sample quality, and training complexity over existing CNFs on challenging synthetic geometries and real-world benchmarks from the earth and climate sciences.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Moser Flow: Divergence-based Generative Modeling<br>on Manifolds<br>abs: <a href="https://t.co/p7EygdJyxb">https://t.co/p7EygdJyxb</a><br><br>flow models for sampling from general curved surfaces and achieve significant improvements in density estimation, sample quality, and training complexity over<br>existing CNFs <a href="https://t.co/ZvvFpMFZqV">pic.twitter.com/ZvvFpMFZqV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428155983658897409?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Quantitative Uniform Stability of the Iterative Proportional Fitting  Procedure

George Deligiannidis, Valentin De Bortoli, Arnaud Doucet

- retweets: 72, favorites: 40 (08/20/2021 06:55:03)

- links: [abs](https://arxiv.org/abs/2108.08129) | [pdf](https://arxiv.org/pdf/2108.08129)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.OC](https://arxiv.org/list/math.OC/recent) | [math.PR](https://arxiv.org/list/math.PR/recent)

We establish the uniform in time stability, w.r.t. the marginals, of the Iterative Proportional Fitting Procedure, also known as Sinkhorn algorithm, used to solve entropy-regularised Optimal Transport problems. Our result is quantitative and stated in terms of the 1-Wasserstein metric. As a corollary we establish a quantitative stability result for Schr\"odinger bridges.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our new preprint <a href="https://t.co/Sm4EDpUIDl">https://t.co/Sm4EDpUIDl</a> with <a href="https://twitter.com/ValentinDeBort1?ref_src=twsrc%5Etfw">@ValentinDeBort1</a> and <a href="https://twitter.com/ArnaudDoucet1?ref_src=twsrc%5Etfw">@ArnaudDoucet1</a> we give a quantitative stability result for the Schrödinger bridge and the IPFP iterates wrt perturbations of the marginals.</p>&mdash; George Deligiannidis (@GeorgeDeligian9) <a href="https://twitter.com/GeorgeDeligian9/status/1428233561065365509?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



