---
title: Hot Papers 2020-07-24
date: 2020-07-25T13:30:54.Z
template: "post"
draft: false
slug: "hot-papers-2020-07-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-07-24"
socialImage: "/media/42-line-bible.jpg"

---

# 1. Contact and Human Dynamics from Monocular Video

Davis Rempe, Leonidas J. Guibas, Aaron Hertzmann, Bryan Russell, Ruben Villegas, Jimei Yang

- retweets: 145, favorites: 533 (07/25/2020 13:30:54)

- links: [abs](https://arxiv.org/abs/2007.11678) | [pdf](https://arxiv.org/pdf/2007.11678)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Existing deep models predict 2D and 3D kinematic poses from video that are approximately accurate, but contain visible errors that violate physical constraints, such as feet penetrating the ground and bodies leaning at extreme angles. In this paper, we present a physics-based method for inferring 3D human motion from video sequences that takes initial 2D and 3D pose estimates as input. We first estimate ground contact timings with a novel prediction network which is trained without hand-labeled data. A physics-based trajectory optimization then solves for a physically-plausible motion, based on the inputs. We show this process produces motions that are significantly more realistic than those from purely kinematic methods, substantially improving quantitative measures of both kinematic and dynamic plausibility. We demonstrate our method on character animation and pose estimation tasks on dynamic motions of dancing and sports with complex contact patterns.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contact and Human Dynamics from Monocular Video<br>pdf: <a href="https://t.co/ECy6u7VOcq">https://t.co/ECy6u7VOcq</a><br>abs: <a href="https://t.co/SqN67KcaUw">https://t.co/SqN67KcaUw</a><br>project page: <a href="https://t.co/0ez3bjGunw">https://t.co/0ez3bjGunw</a> <a href="https://t.co/2tIAjrHvQD">pic.twitter.com/2tIAjrHvQD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1286466935379697668?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new <a href="https://twitter.com/hashtag/ECCV2020?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV2020</a> paper uses learned foot contact estimation and trajectory optimization to reconstruct physically-plausible 3D human motion from video. Check it out!<br><br>Paper: <a href="https://t.co/1JDnFYpwy5">https://t.co/1JDnFYpwy5</a><br>Video: <a href="https://t.co/IO9MrHob6t">https://t.co/IO9MrHob6t</a><br>Project: <a href="https://t.co/GTijegtYD1">https://t.co/GTijegtYD1</a> <a href="https://t.co/9zc5PqIf3r">pic.twitter.com/9zc5PqIf3r</a></p>&mdash; Davis Rempe (@davrempe) <a href="https://twitter.com/davrempe/status/1286692429576327170?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial  and Multi-Map SLAM

Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, Juan D. Tardós

- retweets: 114, favorites: 399 (07/25/2020 13:30:54)

- links: [abs](https://arxiv.org/abs/2007.11898) | [pdf](https://arxiv.org/pdf/2007.11898)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

This paper presents ORB-SLAM3, the first system able to perform visual, visual-inertial and multi-map SLAM with monocular, stereo and RGB-D cameras, using pin-hole and fisheye lens models. The first main novelty is a feature-based tightly-integrated visual-inertial SLAM system that fully relies on Maximum-a-Posteriori (MAP) estimation, even during the IMU initialization phase. The result is a system that operates robustly in real-time, in small and large, indoor and outdoor environments, and is 2 to 5 times more accurate than previous approaches. The second main novelty is a multiple map system that relies on a new place recognition method with improved recall. Thanks to it, ORB-SLAM3 is able to survive to long periods of poor visual information: when it gets lost, it starts a new map that will be seamlessly merged with previous maps when revisiting mapped areas. Compared with visual odometry systems that only use information from the last few seconds, ORB-SLAM3 is the first system able to reuse in all the algorithm stages all previous information. This allows to include in bundle adjustment co-visible keyframes, that provide high parallax observations boosting accuracy, even if they are widely separated in time or if they come from a previous mapping session. Our experiments show that, in all sensor configurations, ORB-SLAM3 is as robust as the best systems available in the literature, and significantly more accurate. Notably, our stereo-inertial SLAM achieves an average accuracy of 3.6 cm on the EuRoC drone and 9 mm under quick hand-held motions in the room of TUM-VI dataset, a setting representative of AR/VR scenarios. For the benefit of the community we make public the source code.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ORB-SLAM3 ?! !!<br><br>ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM<a href="https://t.co/sPWH5dK3pe">https://t.co/sPWH5dK3pe</a><br><br>videos <a href="https://t.co/JBhgAfi2J3">https://t.co/JBhgAfi2J3</a></p>&mdash; Giseop Kim (@GiseopK) <a href="https://twitter.com/GiseopK/status/1286518860447850496?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are proud to announce ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM.<br>Paper: <a href="https://t.co/y77D4UmzNN">https://t.co/y77D4UmzNN</a><br>Code: <a href="https://t.co/1fwh9MKCM9">https://t.co/1fwh9MKCM9</a><br>Videos: <a href="https://t.co/wWO863EqW4">https://t.co/wWO863EqW4</a></p>&mdash; Mingo Tardos (@mingo_tardos) <a href="https://twitter.com/mingo_tardos/status/1286593477048926209?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Whole-Body Human Pose Estimation in the Wild

Sheng Jin, Lumin Xu, Jin Xu, Can Wang, Wentao Liu, Chen Qian, Wanli Ouyang, Ping Luo

- retweets: 42, favorites: 227 (07/25/2020 13:30:55)

- links: [abs](https://arxiv.org/abs/2007.11858) | [pdf](https://arxiv.org/pdf/2007.11858)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper investigates the task of 2D human whole-body pose estimation, which aims to localize dense landmarks on the entire human body including face, hands, body, and feet. As existing datasets do not have whole-body annotations, previous methods have to assemble different deep models trained independently on different datasets of the human face, hand, and body, struggling with dataset biases and large model complexity. To fill in this blank, we introduce COCO-WholeBody which extends COCO dataset with whole-body annotations. To our best knowledge, it is the first benchmark that has manual annotations on the entire human body, including 133 dense landmarks with 68 on the face, 42 on hands and 23 on the body and feet. A single-network model, named ZoomNet, is devised to take into account the hierarchical structure of the full human body to solve the scale variation of different body parts of the same person. ZoomNet is able to significantly outperform existing methods on the proposed COCO-WholeBody dataset. Extensive experiments show that COCO-WholeBody not only can be used to train deep models from scratch for whole-body pose estimation but also can serve as a powerful pre-training dataset for many different tasks such as facial landmark detection and hand keypoint estimation. The dataset is publicly available at https://github.com/jin-s13/COCO-WholeBody.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Whole-Body Human Pose Estimation in the Wild<br>pdf: <a href="https://t.co/M3WDRjKHGu">https://t.co/M3WDRjKHGu</a><br>abs: <a href="https://t.co/CdiEIztSjk">https://t.co/CdiEIztSjk</a><br>github: <a href="https://t.co/9c8PqBDowz">https://t.co/9c8PqBDowz</a> <a href="https://t.co/FlspnRTAXu">pic.twitter.com/FlspnRTAXu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1286463112942690304?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Online Robust and Adaptive Learning from Data Streams

Shintaro Fukushima, Atsushi Nitanda, Kenji Yamanishi

- retweets: 25, favorites: 68 (07/25/2020 13:30:55)

- links: [abs](https://arxiv.org/abs/2007.12160) | [pdf](https://arxiv.org/pdf/2007.12160)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In online learning from non-stationary data streams, it is both necessary to learn robustly to outliers and to adapt to changes of underlying data generating mechanism quickly. In this paper, we refer to the former nature of online learning algorithms as robustness and the latter as adaptivity. There is an obvious tradeoff between them. It is a fundamental issue to quantify and evaluate the tradeoff because it provides important information on the data generating mechanism. However, no previous work has considered the tradeoff quantitatively. We propose a novel algorithm called the Stochastic approximation-based Robustness-Adaptivity algorithm (SRA) to evaluate the tradeoff. The key idea of SRA is to update parameters of distribution or sufficient statistics with the biased stochastic approximation scheme, while dropping data points with large values of the stochastic update. We address the relation between two parameters, one of which is the step size of the stochastic approximation, and the other is the threshold parameter of the norm of the stochastic update. The former controls the adaptivity and the latter does the robustness. We give a theoretical analysis for the non-asymptotic convergence of SRA in the presence of outliers, which depends on both the step size and the threshold parameter. Since SRA is formulated on the majorization-minimization principle, it is a general algorithm including many algorithms, such as the online EM algorithm and stochastic gradient descent. Empirical experiments for both synthetic and real datasets demonstrated that SRA was superior to previous methods.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">プレプリントを公開しました．時系列データのオンライン学習において，外れ値への頑健性(robustness)と変化への適応性(adaptivity)の間に存在するトレードオフを評価しています．確率的近似(stochastic approximation)の収束性を用いて上記トレードオフを議論しています．<a href="https://t.co/u76LYR5eAg">https://t.co/u76LYR5eAg</a></p>&mdash; Shintaro FUKUSHIMA (@shifukushima) <a href="https://twitter.com/shifukushima/status/1286457796431470592?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. TSIT: A Simple and Versatile Framework for Image-to-Image Translation

Liming Jiang, Changxu Zhang, Mingyang Huang, Chunxiao Liu, Jianping Shi, Chen Change Loy

- retweets: 19, favorites: 67 (07/25/2020 13:30:55)

- links: [abs](https://arxiv.org/abs/2007.12072) | [pdf](https://arxiv.org/pdf/2007.12072)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

We introduce a simple and versatile framework for image-to-image translation. We unearth the importance of normalization layers, and provide a carefully designed two-stream generative model with newly proposed feature transformations in a coarse-to-fine fashion. This allows multi-scale semantic structure information and style representation to be effectively captured and fused by the network, permitting our method to scale to various tasks in both unsupervised and supervised settings. No additional constraints (e.g., cycle consistency) are needed, contributing to a very clean and simple method. Multi-modal image synthesis with arbitrary style control is made possible. A systematic study compares the proposed method with several state-of-the-art task-specific baselines, verifying its effectiveness in both perceptual quality and quantitative evaluations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TSIT: A Simple and Versatile Framework for Image-to-Image Translation<br>pdf: <a href="https://t.co/mEUvgIF1q5">https://t.co/mEUvgIF1q5</a><br>abs: <a href="https://t.co/1QdmjapdWb">https://t.co/1QdmjapdWb</a><br>github: <a href="https://t.co/SxcLoIsHGY">https://t.co/SxcLoIsHGY</a> <a href="https://t.co/vaG2mSP4Fa">pic.twitter.com/vaG2mSP4Fa</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1286464303768403968?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Funnel Activation for Visual Recognition

Ningning Ma, Xiangyu Zhang, Jian Sun

- retweets: 6, favorites: 67 (07/25/2020 13:30:55)

- links: [abs](https://arxiv.org/abs/2007.11824) | [pdf](https://arxiv.org/pdf/2007.11824)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a conceptually simple but effective funnel activation for image recognition tasks, called Funnel activation (FReLU), that extends ReLU and PReLU to a 2D activation by adding a negligible overhead of spatial condition. The forms of ReLU and PReLU are y = max(x, 0) and y = max(x, px), respectively, while FReLU is in the form of y = max(x,T(x)), where T(x) is the 2D spatial condition. Moreover, the spatial condition achieves a pixel-wise modeling capacity in a simple way, capturing complicated visual layouts with regular convolutions. We conduct experiments on ImageNet, COCO detection, and semantic segmentation tasks, showing great improvements and robustness of FReLU in the visual recognition tasks.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Funnel Activation for Visual Recognition<a href="https://t.co/J0AVaaD50D">https://t.co/J0AVaaD50D</a><br>良さげ。 <a href="https://t.co/bxP7EZEPDi">pic.twitter.com/bxP7EZEPDi</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1286665225723244544?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval

Andrew Brown, Weidi Xie, Vicky Kalogeiton, Andrew Zisserman

- retweets: 14, favorites: 47 (07/25/2020 13:30:56)

- links: [abs](https://arxiv.org/abs/2007.12163) | [pdf](https://arxiv.org/pdf/2007.12163)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Optimising a ranking-based metric, such as Average Precision (AP), is notoriously challenging due to the fact that it is non-differentiable, and hence cannot be optimised directly using gradient-descent methods. To this end, we introduce an objective that optimises instead a smoothed approximation of AP, coined Smooth-AP. Smooth-AP is a plug-and-play objective function that allows for end-to-end training of deep networks with a simple and elegant implementation. We also present an analysis for why directly optimising the ranking based metric of AP offers benefits over other deep metric learning losses. We apply Smooth-AP to standard retrieval benchmarks: Stanford Online products and VehicleID, and also evaluate on larger-scale datasets: INaturalist for fine-grained category retrieval, and VGGFace2 and IJB-C for face retrieval. In all cases, we improve the performance over the state-of-the-art, especially for larger-scale datasets, thus demonstrating the effectiveness and scalability of Smooth-AP to real-world scenarios.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We boost the performance of image retrieval systems by directly optimising a Smoothed Average Precision loss, Smooth-AP! By <a href="https://twitter.com/AndrewB88429285?ref_src=twsrc%5Etfw">@AndrewB88429285</a> <a href="https://twitter.com/WeidiXie?ref_src=twsrc%5Etfw">@WeidiXie</a> <a href="https://twitter.com/VickyKalogeiton?ref_src=twsrc%5Etfw">@VickyKalogeiton</a> and Andrew Zisserman. <br><br>website: <a href="https://t.co/3LAvbzl7Hk">https://t.co/3LAvbzl7Hk</a><br>arXiv: <a href="https://t.co/sXZOWzsVAm">https://t.co/sXZOWzsVAm</a><a href="https://twitter.com/hashtag/ECCV2020?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV2020</a> <a href="https://t.co/gQvgs6l53n">pic.twitter.com/gQvgs6l53n</a></p>&mdash; Visual Geometry Group (VGG) (@Oxford_VGG) <a href="https://twitter.com/Oxford_VGG/status/1286603905397293057?ref_src=twsrc%5Etfw">July 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



