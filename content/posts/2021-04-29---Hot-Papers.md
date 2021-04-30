---
title: Hot Papers 2021-04-29
date: 2021-04-30T11:05:38.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-29"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-29"
socialImage: "/media/flying-marine.jpg"

---

# 1. PyTorch Tabular: A Framework for Deep Learning with Tabular Data

Manu Joseph

- retweets: 9602, favorites: 441 (04/30/2021 11:05:38)

- links: [abs](https://arxiv.org/abs/2104.13638) | [pdf](https://arxiv.org/pdf/2104.13638)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

In spite of showing unreasonable effectiveness in modalities like Text and Image, Deep Learning has always lagged Gradient Boosting in tabular data - both in popularity and performance. But recently there have been newer models created specifically for tabular data, which is pushing the performance bar. But popularity is still a challenge because there is no easy, ready-to-use library like Sci-Kit Learn for deep learning. PyTorch Tabular is a new deep learning library which makes working with Deep Learning and tabular data easy and fast. It is a library built on top of PyTorch and PyTorch Lightning and works on pandas dataframes directly. Many SOTA models like NODE and TabNet are already integrated and implemented in the library with a unified API. PyTorch Tabular is designed to be easily extensible for researchers, simple for practitioners, and robust in industrial deployments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It&#39;s exciting to see the performance improvements of deep learning on tabular data.<br><br>It&#39;s even more exciting to see that there is a new <a href="https://twitter.com/PyTorch?ref_src=twsrc%5Etfw">@PyTorch</a> based library for applying SOTA deep learning models to tabular data.<br><br>repo: <a href="https://t.co/ZYh6Wp1CQW">https://t.co/ZYh6Wp1CQW</a><br>paper: <a href="https://t.co/u2JUIGGwTL">https://t.co/u2JUIGGwTL</a> <a href="https://t.co/uEnSbeX2zP">pic.twitter.com/uEnSbeX2zP</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1387734006960115712?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges

Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković

- retweets: 8038, favorites: 17 (04/30/2021 11:05:38)

- links: [abs](https://arxiv.org/abs/2104.13478) | [pdf](https://arxiv.org/pdf/2104.13478)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CG](https://arxiv.org/list/cs.CG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The last decade has witnessed an experimental revolution in data science and machine learning, epitomised by deep learning methods. Indeed, many high-dimensional learning tasks previously thought to be beyond reach -- such as computer vision, playing Go, or protein folding -- are in fact feasible with appropriate computational scale. Remarkably, the essence of deep learning is built from two simple algorithmic principles: first, the notion of representation or feature learning, whereby adapted, often hierarchical, features capture the appropriate notion of regularity for each task, and second, learning by local gradient-descent type methods, typically implemented as backpropagation.   While learning generic functions in high dimensions is a cursed estimation problem, most tasks of interest are not generic, and come with essential pre-defined regularities arising from the underlying low-dimensionality and structure of the physical world. This text is concerned with exposing these regularities through unified geometric principles that can be applied throughout a wide spectrum of applications.   Such a 'geometric unification' endeavour, in the spirit of Felix Klein's Erlangen Program, serves a dual purpose: on one hand, it provides a common mathematical framework to study the most successful neural network architectures, such as CNNs, RNNs, GNNs, and Transformers. On the other hand, it gives a constructive procedure to incorporate prior physical knowledge into neural architectures and provide principled way to build future architectures yet to be invented.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m excited about progress in GNNs and geometric deep learning in general because of its potential to address more complex problems.<a href="https://twitter.com/mmbronstein?ref_src=twsrc%5Etfw">@mmbronstein</a> et al. recently published a 150+ pages book on Geometric Deep Learning. This is a must-read for ML students.<a href="https://t.co/ysoRJp93lk">https://t.co/ysoRJp93lk</a> <a href="https://t.co/cIeHNl8Uv0">pic.twitter.com/cIeHNl8Uv0</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1387738916200005632?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Zero-Shot Detection via Vision and Language Knowledge Distillation

Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, Yin Cui

- retweets: 4060, favorites: 109 (04/30/2021 11:05:38)

- links: [abs](https://arxiv.org/abs/2104.13921) | [pdf](https://arxiv.org/pdf/2104.13921)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Zero-shot image classification has made promising progress by training the aligned image and text encoders. The goal of this work is to advance zero-shot object detection, which aims to detect novel objects without bounding box nor mask annotations. We propose ViLD, a training method via Vision and Language knowledge Distillation. We distill the knowledge from a pre-trained zero-shot image classification model (e.g., CLIP) into a two-stage detector (e.g., Mask R-CNN). Our method aligns the region embeddings in the detector to the text and image embeddings inferred by the pre-trained model. We use the text embeddings as the detection classifier, obtained by feeding category names into the pre-trained text encoder. We then minimize the distance between the region embeddings and image embeddings, obtained by feeding region proposals into the pre-trained image encoder. During inference, we include text embeddings of novel categories into the detection classifier for zero-shot detection. We benchmark the performance on LVIS dataset by holding out all rare categories as novel categories. ViLD obtains 16.1 mask AP$_r$ with a Mask R-CNN (ResNet-50 FPN) for zero-shot detection, outperforming the supervised counterpart by 3.8. The model can directly transfer to other datasets, achieving 72.2 AP$_{50}$, 36.6 AP and 11.8 AP on PASCAL VOC, COCO and Objects365, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can we use free-form text to detect any object, especially long-tailed objects?<br><br>Yes!<br><br>We train Mask R-CNN by distilling from CLIP to enable zero-shot detection. The model achieves higher AP compared to its supervised counterpart on rare classes.<a href="https://t.co/ZAE7UtLcv5">https://t.co/ZAE7UtLcv5</a> <a href="https://t.co/fuT7PFAHE7">pic.twitter.com/fuT7PFAHE7</a></p>&mdash; Yin Cui (@YinCui1) <a href="https://twitter.com/YinCui1/status/1387782820664279040?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Zero-Shot Detection via Vision and Language Knowledge Distillation<br><br>Distills the knowledge from CLIP into Mask R-CNN to perform zero-shot detection. Outperforms the supervised counterpart.  <a href="https://t.co/QcrK1TKp0f">https://t.co/QcrK1TKp0f</a> <a href="https://t.co/mplUfEeT8Z">pic.twitter.com/mplUfEeT8Z</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1387572420341497860?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. KAMA: 3D Keypoint Aware Body Mesh Articulation

Umar Iqbal, Kevin Xie, Yunrong Guo, Jan Kautz, Pavlo Molchanov

- retweets: 484, favorites: 164 (04/30/2021 11:05:39)

- links: [abs](https://arxiv.org/abs/2104.13502) | [pdf](https://arxiv.org/pdf/2104.13502)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present KAMA, a 3D Keypoint Aware Mesh Articulation approach that allows us to estimate a human body mesh from the positions of 3D body keypoints. To this end, we learn to estimate 3D positions of 26 body keypoints and propose an analytical solution to articulate a parametric body model, SMPL, via a set of straightforward geometric transformations. Since keypoint estimation directly relies on image clues, our approach offers significantly better alignment to image content when compared to state-of-the-art approaches. Our proposed approach does not require any paired mesh annotations and is able to achieve state-of-the-art mesh fittings through 3D keypoint regression only. Results on the challenging 3DPW and Human3.6M demonstrate that our approach yields state-of-the-art body mesh fittings.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">KAMA: 3D Keypoint Aware Body Mesh Articulation<br>pdf: <a href="https://t.co/bw17lV3A4S">https://t.co/bw17lV3A4S</a><br>abs: <a href="https://t.co/OAsNTxcl9T">https://t.co/OAsNTxcl9T</a> <a href="https://t.co/e0jeOCg0ou">pic.twitter.com/e0jeOCg0ou</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387583334507859971?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Can the Wikipedia moderation model rescue the social marketplace of  ideas?

Taha Yasseri, Filippo Menczer

- retweets: 342, favorites: 67 (04/30/2021 11:05:39)

- links: [abs](https://arxiv.org/abs/2104.13754) | [pdf](https://arxiv.org/pdf/2104.13754)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [physics.data-an](https://arxiv.org/list/physics.data-an/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

Facebook announced a community review program in December 2019 and Twitter launched a community-based platform to address misinformation, called Birdwatch, in January 2021. We provide an overview of the potential affordances of such community based approaches to content moderation based on past research. While our analysis generally supports a community-based approach to content moderation, it also warns against potential pitfalls, particularly when the implementation of the new infrastructures does not promote diversity. We call for more multidisciplinary research utilizing methods from complex systems studies, behavioural sociology, and computational social science to advance the research on crowd-based content moderation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How come today Wikipedia is &quot;last best place on the Internet&quot; &amp; Social Media are to blame for polarization, misinformation &amp; hate speech; both designed on same promise?<br>What can we learn from former to fix latter?<br>Comment by Fil Menczer of <a href="https://twitter.com/OSoMe_IU?ref_src=twsrc%5Etfw">@OSoMe_IU</a> &amp; I <a href="https://t.co/yF9r9b112T">https://t.co/yF9r9b112T</a> <a href="https://t.co/UlZa7hTcxZ">pic.twitter.com/UlZa7hTcxZ</a></p>&mdash; Taha Yasseri (@TahaYasseri) <a href="https://twitter.com/TahaYasseri/status/1387779938397347846?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Gradient-based Adversarial Attacks against Text Transformers

Chuan Guo, Alexandre Sablayrolles, Hervé Jégou, Douwe Kiela

- retweets: 240, favorites: 52 (04/30/2021 11:05:39)

- links: [abs](https://arxiv.org/abs/2104.13733) | [pdf](https://arxiv.org/pdf/2104.13733)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose the first general-purpose gradient-based attack against transformer models. Instead of searching for a single adversarial example, we search for a distribution of adversarial examples parameterized by a continuous-valued matrix, hence enabling gradient-based optimization. We empirically demonstrate that our white-box attack attains state-of-the-art attack performance on a variety of natural language tasks. Furthermore, we show that a powerful black-box transfer attack, enabled by sampling from the adversarial distribution, matches or exceeds existing methods, while only requiring hard-label outputs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Gradient-based Adversarial Attacks against Text Transformers<br>pdf: <a href="https://t.co/3PpQ3P4I1A">https://t.co/3PpQ3P4I1A</a><br>abs: <a href="https://t.co/YKzDuzSeYk">https://t.co/YKzDuzSeYk</a><br>github: <a href="https://t.co/he8vXqnFkW">https://t.co/he8vXqnFkW</a><br><br>the first general-purpose gradient based attack against transformer models <a href="https://t.co/Sz0ddaHQkq">pic.twitter.com/Sz0ddaHQkq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387575640094871560?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. ConTNet: Why not use convolution and transformer at the same time?

Haotian Yan, Zhe Li, Weijian Li, Changhu Wang, Ming Wu, Chuang Zhang

- retweets: 210, favorites: 69 (04/30/2021 11:05:39)

- links: [abs](https://arxiv.org/abs/2104.13497) | [pdf](https://arxiv.org/pdf/2104.13497)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Although convolutional networks (ConvNets) have enjoyed great success in computer vision (CV), it suffers from capturing global information crucial to dense prediction tasks such as object detection and segmentation. In this work, we innovatively propose ConTNet (ConvolutionTransformer Network), combining transformer with ConvNet architectures to provide large receptive fields. Unlike the recently-proposed transformer-based models (e.g., ViT, DeiT) that are sensitive to hyper-parameters and extremely dependent on a pile of data augmentations when trained from scratch on a midsize dataset (e.g., ImageNet1k), ConTNet can be optimized like normal ConvNets (e.g., ResNet) and preserve an outstanding robustness. It is also worth pointing that, given identical strong data augmentations, the performance improvement of ConTNet is more remarkable than that of ResNet. We present its superiority and effectiveness on image classification and downstream tasks. For example, our ConTNet achieves 81.8% top-1 accuracy on ImageNet which is the same as DeiT-B with less than 40% computational complexity. ConTNet-M also outperforms ResNet50 as the backbone of both Faster-RCNN (by 2.6%) and Mask-RCNN (by 3.2%) on COCO2017 dataset. We hope that ConTNet could serve as a useful backbone for CV tasks and bring new ideas for model design

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ConTNet: Why not use convolution and transformer at the same time?<br>pdf: <a href="https://t.co/e8n3xcZ0yg">https://t.co/e8n3xcZ0yg</a><br>abs: <a href="https://t.co/adcsBGvLL1">https://t.co/adcsBGvLL1</a> <a href="https://t.co/EhoLAorfrT">pic.twitter.com/EhoLAorfrT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387574738134913029?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. An optical neural network using less than 1 photon per multiplication

Tianyu Wang, Shi-Yuan Ma, Logan G. Wright, Tatsuhiro Onodera, Brian Richard, Peter L. McMahon

- retweets: 121, favorites: 96 (04/30/2021 11:05:39)

- links: [abs](https://arxiv.org/abs/2104.13467) | [pdf](https://arxiv.org/pdf/2104.13467)
- [physics.optics](https://arxiv.org/list/physics.optics/recent) | [cs.ET](https://arxiv.org/list/cs.ET/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Deep learning has rapidly become a widespread tool in both scientific and commercial endeavors. Milestones of deep learning exceeding human performance have been achieved for a growing number of tasks over the past several years, across areas as diverse as game-playing, natural-language translation, and medical-image analysis. However, continued progress is increasingly hampered by the high energy costs associated with training and running deep neural networks on electronic processors. Optical neural networks have attracted attention as an alternative physical platform for deep learning, as it has been theoretically predicted that they can fundamentally achieve higher energy efficiency than neural networks deployed on conventional digital computers. Here, we experimentally demonstrate an optical neural network achieving 99% accuracy on handwritten-digit classification using ~3.2 detected photons per weight multiplication and ~90% accuracy using ~0.64 photons (~$2.4 \times 10^{-19}$ J of optical energy) per weight multiplication. This performance was achieved using a custom free-space optical processor that executes matrix-vector multiplications in a massively parallel fashion, with up to ~0.5 million scalar (weight) multiplications performed at the same time. Using commercially available optical components and standard neural-network training methods, we demonstrated that optical neural networks can operate near the standard quantum limit with extremely low optical powers and still achieve high accuracy. Our results provide a proof-of-principle for low-optical-power operation, and with careful system design including the surrounding electronics used for data storage and control, open up a path to realizing optical processors that require only $10^{-16}$ J total energy per scalar multiplication -- which is orders of magnitude more efficient than current digital processors.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our lab has two papers out on the arXiv this evening, on two different but related topics: *optical neural networks* <a href="https://t.co/ZaeVBMgcF2">https://t.co/ZaeVBMgcF2</a> and *physical neural networks* <a href="https://t.co/62U1fUotcF">https://t.co/62U1fUotcF</a>. Since they&#39;re distinct projects, I&#39;m going to post two separate threads about them.</p>&mdash; Peter McMahon (@peterlmcmahon) <a href="https://twitter.com/peterlmcmahon/status/1387568993662382085?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Congratulations to Tianyu, who led the project, and Shiyuan and everyone else involved! <a href="https://t.co/ZaeVBMgcF2">https://t.co/ZaeVBMgcF2</a> 3/3</p>&mdash; Peter McMahon (@peterlmcmahon) <a href="https://twitter.com/peterlmcmahon/status/1387569518663319555?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Deep physical neural networks enabled by a backpropagation algorithm for  arbitrary physical systems

Logan G. Wright, Tatsuhiro Onodera, Martin M. Stein, Tianyu Wang, Darren T. Schachter, Zoey Hu, Peter L. McMahon

- retweets: 121, favorites: 91 (04/30/2021 11:05:39)

- links: [abs](https://arxiv.org/abs/2104.13386) | [pdf](https://arxiv.org/pdf/2104.13386)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [cs.ET](https://arxiv.org/list/cs.ET/recent) | [physics.optics](https://arxiv.org/list/physics.optics/recent)

Deep neural networks have become a pervasive tool in science and engineering. However, modern deep neural networks' growing energy requirements now increasingly limit their scaling and broader use. We propose a radical alternative for implementing deep neural network models: Physical Neural Networks. We introduce a hybrid physical-digital algorithm called Physics-Aware Training to efficiently train sequences of controllable physical systems to act as deep neural networks. This method automatically trains the functionality of any sequence of real physical systems, directly, using backpropagation, the same technique used for modern deep neural networks. To illustrate their generality, we demonstrate physical neural networks with three diverse physical systems-optical, mechanical, and electrical. Physical neural networks may facilitate unconventional machine learning hardware that is orders of magnitude faster and more energy efficient than conventional electronic processors.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our lab has two papers out on the arXiv this evening, on two different but related topics: *optical neural networks* <a href="https://t.co/ZaeVBMgcF2">https://t.co/ZaeVBMgcF2</a> and *physical neural networks* <a href="https://t.co/62U1fUotcF">https://t.co/62U1fUotcF</a>. Since they&#39;re distinct projects, I&#39;m going to post two separate threads about them.</p>&mdash; Peter McMahon (@peterlmcmahon) <a href="https://twitter.com/peterlmcmahon/status/1387568993662382085?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Neural Ray-Tracing: Learning Surfaces and Reflectance for Relighting and  View Synthesis

Julian Knodt, Seung-Hwan Baek, Felix Heide

- retweets: 134, favorites: 73 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13562) | [pdf](https://arxiv.org/pdf/2104.13562)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent neural rendering methods have demonstrated accurate view interpolation by predicting volumetric density and color with a neural network. Although such volumetric representations can be supervised on static and dynamic scenes, existing methods implicitly bake the complete scene light transport into a single neural network for a given scene, including surface modeling, bidirectional scattering distribution functions, and indirect lighting effects. In contrast to traditional rendering pipelines, this prohibits changing surface reflectance, illumination, or composing other objects in the scene.   In this work, we explicitly model the light transport between scene surfaces and we rely on traditional integration schemes and the rendering equation to reconstruct a scene. The proposed method allows BSDF recovery with unknown light conditions and classic light transports such as pathtracing. By learning decomposed transport with surface representations established in conventional rendering methods, the method naturally facilitates editing shape, reflectance, lighting and scene composition. The method outperforms NeRV for relighting under known lighting conditions, and produces realistic reconstructions for relit and edited scenes. We validate the proposed approach for scene editing, relighting and reflectance estimation learned from synthetic and captured views on a subset of NeRV's datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Ray-Tracing: Learning Surfaces and Reflectance for Relighting and View Synthesis<br>pdf: <a href="https://t.co/pFTcapkuy5">https://t.co/pFTcapkuy5</a><br>abs: <a href="https://t.co/a0jAmxFdSx">https://t.co/a0jAmxFdSx</a> <a href="https://t.co/X5mskPkY1V">pic.twitter.com/X5mskPkY1V</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387621675710177280?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Causes of Effects: Learning individual responses from population data

Scott Mueller, Ang Li, Judea Pearl

- retweets: 143, favorites: 30 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13730) | [pdf](https://arxiv.org/pdf/2104.13730)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The problem of individualization is recognized as crucial in almost every field. Identifying causes of effects in specific events is likewise essential for accurate decision making. However, such estimates invoke counterfactual relationships, and are therefore indeterminable from population data. For example, the probability of benefiting from a treatment concerns an individual having a favorable outcome if treated and an unfavorable outcome if untreated. Experiments conditioning on fine-grained features are fundamentally inadequate because we can't test both possibilities for an individual. Tian and Pearl provided bounds on this and other probabilities of causation using a combination of experimental and observational data. Even though those bounds were proven tight, narrower bounds, sometimes significantly so, can be achieved when structural information is available in the form of a causal model. This has the power to solve central problems, such as explainable AI, legal responsibility, and personalized medicine, all of which demand counterfactual logic. We analyze and expand on existing research by applying bounds to the probability of necessity and sufficiency (PNS) along with graphical criteria and practical applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How to harness population data for personalized decision making was discussed several time on this education channel. Our latest thinking on this issue is now posted here: <a href="https://t.co/8WkTojnZNt">https://t.co/8WkTojnZNt</a><br>I hope you see its potentials for personalized medicine and precision marketing.</p>&mdash; Judea Pearl (@yudapearl) <a href="https://twitter.com/yudapearl/status/1387604036002533376?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Twins: Revisiting Spatial Attention Design in Vision Transformers

Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, Chunhua Shen

- retweets: 64, favorites: 42 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13840) | [pdf](https://arxiv.org/pdf/2104.13840)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins-PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks including imagelevel classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks. Our code will be released soon at https://github.com/Meituan-AutoML/Twins .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Twins: Revisiting Spatial Attention Design in Vision Transformers<br>pdf: <a href="https://t.co/q8bqQ1orgx">https://t.co/q8bqQ1orgx</a><br>abs: <a href="https://t.co/83aL2GG4wo">https://t.co/83aL2GG4wo</a> <a href="https://t.co/nQolcyoa0l">pic.twitter.com/nQolcyoa0l</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387576242707255298?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Extreme Rotation Estimation using Dense Correlation Volumes

Ruojin Cai, Bharath Hariharan, Noah Snavely, Hadar Averbuch-Elor

- retweets: 72, favorites: 32 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13530) | [pdf](https://arxiv.org/pdf/2104.13530)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a technique for estimating the relative 3D rotation of an RGB image pair in an extreme setting, where the images have little or no overlap. We observe that, even when images do not overlap, there may be rich hidden cues as to their geometric relationship, such as light source directions, vanishing points, and symmetries present in the scene. We propose a network design that can automatically learn such implicit cues by comparing all pairs of points between the two input images. Our method therefore constructs dense feature correlation volumes and processes these to predict relative 3D rotations. Our predictions are formed over a fine-grained discretization of rotations, bypassing difficulties associated with regressing 3D rotations. We demonstrate our approach on a large variety of extreme RGB image pairs, including indoor and outdoor images captured under different lighting conditions and geographic locations. Our evaluation shows that our model can successfully estimate relative rotations among non-overlapping images without compromising performance over overlapping image pairs.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Extreme Rotation Estimation using Dense Correlation Volumes<br>pdf: <a href="https://t.co/MurGLIdHhS">https://t.co/MurGLIdHhS</a><br>abs: <a href="https://t.co/x4H07QTVz4">https://t.co/x4H07QTVz4</a><br>project page: <a href="https://t.co/V6PjCHWuIU">https://t.co/V6PjCHWuIU</a> <a href="https://t.co/EyLtLeHMrU">pic.twitter.com/EyLtLeHMrU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387580838028718080?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Domain Adaptive Semantic Segmentation with Self-Supervised Depth  Estimation

Qin Wang, Dengxin Dai, Lukas Hoyer, Olga Fink, Luc Van Gool

- retweets: 36, favorites: 52 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13613) | [pdf](https://arxiv.org/pdf/2104.13613)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Domain adaptation for semantic segmentation aims to improve the model performance in the presence of a distribution shift between source and target domain. Leveraging the supervision from auxiliary tasks~(such as depth estimation) has the potential to heal this shift because many visual tasks are closely related to each other. However, such a supervision is not always available. In this work, we leverage the guidance from self-supervised depth estimation, which is available on both domains, to bridge the domain gap. On the one hand, we propose to explicitly learn the task feature correlation to strengthen the target semantic predictions with the help of target depth estimation. On the other hand, we use the depth prediction discrepancy from source and target depth decoders to approximate the pixel-wise adaptation difficulty. The adaptation difficulty, inferred from depth, is then used to refine the target semantic segmentation pseudo-labels. The proposed method can be easily implemented into existing segmentation frameworks. We demonstrate the effectiveness of our proposed approach on the benchmark tasks SYNTHIA-to-Cityscapes and GTA-to-Cityscapes, on which we achieve the new state-of-the-art performance of $55.0\%$ and $56.6\%$, respectively. Our code is available at \url{https://github.com/qinenergy/corda}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Domain Adaptive Semantic Segmentation with Self-Supervised Depth Estimation<br>pdf: <a href="https://t.co/4GufWHzk0u">https://t.co/4GufWHzk0u</a><br>abs: <a href="https://t.co/5o6MePKHnl">https://t.co/5o6MePKHnl</a> <a href="https://t.co/OZX2ikyZAr">pic.twitter.com/OZX2ikyZAr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387624838718709761?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. HOTR: End-to-End Human-Object Interaction Detection with Transformers

Bumsoo Kim, Junhyun Lee, Jaewoo Kang, Eun-Sol Kim, Hyunwoo J. Kim

- retweets: 49, favorites: 39 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13682) | [pdf](https://arxiv.org/pdf/2104.13682)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Human-Object Interaction (HOI) detection is a task of identifying "a set of interactions" in an image, which involves the i) localization of the subject (i.e., humans) and target (i.e., objects) of interaction, and ii) the classification of the interaction labels. Most existing methods have indirectly addressed this task by detecting human and object instances and individually inferring every pair of the detected instances. In this paper, we present a novel framework, referred to by HOTR, which directly predicts a set of <human, object, interaction> triplets from an image based on a transformer encoder-decoder architecture. Through the set prediction, our method effectively exploits the inherent semantic relationships in an image and does not require time-consuming post-processing which is the main bottleneck of existing methods. Our proposed algorithm achieves the state-of-the-art performance in two HOI detection benchmarks with an inference time under 1 ms after object detection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">HOTR: End-to-End Human-Object Interaction Detection with Transformers<br>pdf: <a href="https://t.co/Mjll43pERZ">https://t.co/Mjll43pERZ</a><br>abs: <a href="https://t.co/7wDYuSeLsq">https://t.co/7wDYuSeLsq</a> <a href="https://t.co/U5iJ3a72qD">pic.twitter.com/U5iJ3a72qD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387575089353347072?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. PAFNet: An Efficient Anchor-Free Object Detector Guidance

Ying Xin, Guanzhong Wang, Mingyuan Mao, Yuan Feng, Qingqing Dang, Yanjun Ma, Errui Ding, Shumin Han

- retweets: 56, favorites: 30 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13534) | [pdf](https://arxiv.org/pdf/2104.13534)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Object detection is a basic but challenging task in computer vision, which plays a key role in a variety of industrial applications. However, object detectors based on deep learning usually require greater storage requirements and longer inference time, which hinders its practicality seriously. Therefore, a trade-off between effectiveness and efficiency is necessary in practical scenarios. Considering that without constraint of pre-defined anchors, anchor-free detectors can achieve acceptable accuracy and inference speed simultaneously. In this paper, we start from an anchor-free detector called TTFNet, modify the structure of TTFNet and introduce multiple existing tricks to realize effective server and mobile solutions respectively. Since all experiments in this paper are conducted based on PaddlePaddle, we call the model as PAFNet(Paddle Anchor Free Network). For server side, PAFNet can achieve a better balance between effectiveness (42.2% mAP) and efficiency (67.15 FPS) on a single V100 GPU. For moblie side, PAFNet-lite can achieve a better accuracy of (23.9% mAP) and 26.00 ms on Kirin 990 ARM CPU, outperforming the existing state-of-the-art anchor-free detectors by significant margins. Source code is at https://github.com/PaddlePaddle/PaddleDetection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PAFNet: An Efficient Anchor-Free Object Detector Guidance<br>pdf: <a href="https://t.co/rPEMTj5JpR">https://t.co/rPEMTj5JpR</a><br>abs: <a href="https://t.co/lUit4eJMIz">https://t.co/lUit4eJMIz</a> <a href="https://t.co/amv3rL7zfQ">pic.twitter.com/amv3rL7zfQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387613096026443779?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Distributional Gaussian Process Layers for Outlier Detection in Image  Segmentation

Sebastian G. Popescu, David J. Sharp, James H. Cole, Konstantinos Kamnitsas, Ben Glocker

- retweets: 8, favorites: 48 (04/30/2021 11:05:40)

- links: [abs](https://arxiv.org/abs/2104.13756) | [pdf](https://arxiv.org/pdf/2104.13756)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a parameter efficient Bayesian layer for hierarchical convolutional Gaussian Processes that incorporates Gaussian Processes operating in Wasserstein-2 space to reliably propagate uncertainty. This directly replaces convolving Gaussian Processes with a distance-preserving affine operator on distributions. Our experiments on brain tissue-segmentation show that the resulting architecture approaches the performance of well-established deterministic segmentation algorithms (U-Net), which has never been achieved with previous hierarchical Gaussian Processes. Moreover, by applying the same segmentation model to out-of-distribution data (i.e., images with pathology such as brain tumors), we show that our uncertainty estimates result in out-of-distribution detection that outperforms the capabilities of previous Bayesian networks and reconstruction-based approaches that learn normative distributions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work from <a href="https://twitter.com/SebastianGP13?ref_src=twsrc%5Etfw">@SebastianGP13</a> introducing distributional Gaussian process layers for image segmentation with outlier detection<br><br>Coming up <a href="https://twitter.com/ipmi2021?ref_src=twsrc%5Etfw">@ipmi2021</a> with pre-print available here <a href="https://t.co/rVVjI5wf0t">https://t.co/rVVjI5wf0t</a><a href="https://twitter.com/ICComputing?ref_src=twsrc%5Etfw">@ICComputing</a> <a href="https://twitter.com/BioMedIAICL?ref_src=twsrc%5Etfw">@BioMedIAICL</a> <a href="https://t.co/ARPNXyxPWE">https://t.co/ARPNXyxPWE</a></p>&mdash; Ben Glocker (@GlockerBen) <a href="https://twitter.com/GlockerBen/status/1387684825914650625?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Autoregressive Dynamics Models for Offline Policy Evaluation and  Optimization

Michael R. Zhang, Tom Le Paine, Ofir Nachum, Cosmin Paduraru, George Tucker, Ziyu Wang, Mohammad Norouzi

- retweets: 6, favorites: 47 (04/30/2021 11:05:41)

- links: [abs](https://arxiv.org/abs/2104.13877) | [pdf](https://arxiv.org/pdf/2104.13877)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Standard dynamics models for continuous control make use of feedforward computation to predict the conditional distribution of next state and reward given current state and action using a multivariate Gaussian with a diagonal covariance structure. This modeling choice assumes that different dimensions of the next state and reward are conditionally independent given the current state and action and may be driven by the fact that fully observable physics-based simulation environments entail deterministic transition dynamics. In this paper, we challenge this conditional independence assumption and propose a family of expressive autoregressive dynamics models that generate different dimensions of the next state and reward sequentially conditioned on previous dimensions. We demonstrate that autoregressive dynamics models indeed outperform standard feedforward models in log-likelihood on heldout transitions. Furthermore, we compare different model-based and model-free off-policy evaluation (OPE) methods on RL Unplugged, a suite of offline MuJoCo datasets, and find that autoregressive dynamics models consistently outperform all baselines, achieving a new state-of-the-art. Finally, we show that autoregressive dynamics models are useful for offline policy optimization by serving as a way to enrich the replay buffer through data augmentation and improving performance using model-based planning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am excited to share our ICLR paper on autoregressive dynamics models for RL. Explicitly capturing conditional dependencies between state dimensions improves forward dynamics and reward prediction, yielding SOTA off-policy evaluation on control tasks.<a href="https://t.co/xELsmbhYBE">https://t.co/xELsmbhYBE</a> <a href="https://t.co/7Kw8bJuhzg">pic.twitter.com/7Kw8bJuhzg</a></p>&mdash; Michael Zhang (@michaelrzhang) <a href="https://twitter.com/michaelrzhang/status/1387830192983330816?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. FrameExit: Conditional Early Exiting for Efficient Video Recognition

Amir Ghodrati, Babak Ehteshami Bejnordi, Amirhossein Habibian

- retweets: 14, favorites: 36 (04/30/2021 11:05:41)

- links: [abs](https://arxiv.org/abs/2104.13400) | [pdf](https://arxiv.org/pdf/2104.13400)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this paper, we propose a conditional early exiting framework for efficient video recognition. While existing works focus on selecting a subset of salient frames to reduce the computation costs, we propose to use a simple sampling strategy combined with conditional early exiting to enable efficient recognition. Our model automatically learns to process fewer frames for simpler videos and more frames for complex ones. To achieve this, we employ a cascade of gating modules to automatically determine the earliest point in processing where an inference is sufficiently reliable. We generate on-the-fly supervision signals to the gates to provide a dynamic trade-off between accuracy and computational cost. Our proposed model outperforms competing methods on three large-scale video benchmarks. In particular, on ActivityNet1.3 and mini-kinetics, we outperform the state-of-the-art efficient video recognition methods with 1.3$\times$ and 2.1$\times$ less GFLOPs, respectively. Additionally, our method sets a new state of the art for efficient video understanding on the HVU benchmark.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How many frames are needed to reliably recognize an action?<a href="https://twitter.com/hashtag/FrameExit?src=hash&amp;ref_src=twsrc%5Etfw">#FrameExit</a> uses self-supervised gates to adjust the computation to the difficulty of the input video. Check out our <a href="https://twitter.com/hashtag/Oral?src=hash&amp;ref_src=twsrc%5Etfw">#Oral</a> <a href="https://twitter.com/hashtag/CVPR?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR</a> paper: <a href="https://t.co/m46sXn56GN">https://t.co/m46sXn56GN</a> <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a><br>Great collaboration with <a href="https://twitter.com/ghodrati?ref_src=twsrc%5Etfw">@ghodrati</a> &amp; <a href="https://twitter.com/amir_habibian?ref_src=twsrc%5Etfw">@amir_habibian</a> <a href="https://t.co/JYoRxaV85v">pic.twitter.com/JYoRxaV85v</a></p>&mdash; Babak Ehteshami Bejnordi (@BabakEht) <a href="https://twitter.com/BabakEht/status/1387779307569885205?ref_src=twsrc%5Etfw">April 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



