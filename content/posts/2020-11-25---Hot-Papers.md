---
title: Hot Papers 2020-11-25
date: 2020-11-26T10:46:11.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. MonoRec: Semi-Supervised Dense Reconstruction in Dynamic Environments  from a Single Moving Camera

Felix Wimbauer, Nan Yang, Lukas von Stumberg, Niclas Zeller, Daniel Cremers

- retweets: 1504, favorites: 320 (11/26/2020 10:46:11)

- links: [abs](https://arxiv.org/abs/2011.11814) | [pdf](https://arxiv.org/pdf/2011.11814)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we propose MonoRec, a semi-supervised monocular dense reconstruction architecture that predicts depth maps from a single moving camera in dynamic environments. MonoRec is based on a MVS setting which encodes the information of multiple consecutive images in a cost volume. To deal with dynamic objects in the scene, we introduce a MaskModule that predicts moving object masks by leveraging the photometric inconsistencies encoded in the cost volumes. Unlike other MVS methods, MonoRec is able to predict accurate depths for both static and moving objects by leveraging the predicted masks. Furthermore, we present a novel multi-stage training scheme with a semi-supervised loss formulation that does not require LiDAR depth values. We carefully evaluate MonoRec on the KITTI dataset and show that it achieves state-of-the-art performance compared to both multi-view and single-view methods. With the model trained on KITTI, we further demonstrate that MonoRec is able to generalize well to both the Oxford RobotCar dataset and the more challenging TUM-Mono dataset recorded by a handheld camera. Training code and pre-trained model will be published soon.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MonoRec: Semi-Supervised Dense Reconstruction in Dynamic Environments from a Single Moving Camera<br>pdf: <a href="https://t.co/iC7DyrDmLD">https://t.co/iC7DyrDmLD</a><br>abs: <a href="https://t.co/gy9XCyjOLF">https://t.co/gy9XCyjOLF</a><br>project page: <a href="https://t.co/8wMfO9AgR4">https://t.co/8wMfO9AgR4</a> <a href="https://t.co/i15K5bFPPW">pic.twitter.com/i15K5bFPPW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331468407112339458?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">MonoRecは単眼カメラからの密な深度推定による三次元復元でSOTA性能を達成。複数フレームからコストボリュームを構築、動的物体を推定/除去し深度推定。学習時は他の深度/動的物体推定結果を利用しブートストラップ、LIDARセンサを必要としない  <a href="https://t.co/fTQPpeC0KZ">https://t.co/fTQPpeC0KZ</a> <a href="https://t.co/GAZ3udeGEE">https://t.co/GAZ3udeGEE</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1331735718134042624?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Is a Green Screen Really Necessary for Real-Time Human Matting?

Zhanghan Ke, Kaican Li, Yurou Zhou, Qiuhua Wu, Xiangyu Mao, Qiong Yan, Rynson W.H. Lau

- retweets: 1056, favorites: 246 (11/26/2020 10:46:11)

- links: [abs](https://arxiv.org/abs/2011.11961) | [pdf](https://arxiv.org/pdf/2011.11961)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

For human matting without the green screen, existing works either require auxiliary inputs that are costly to obtain or use multiple models that are computationally expensive. Consequently, they are unavailable in real-time applications. In contrast, we present a light-weight matting objective decomposition network (MODNet), which can process human matting from a single input image in real time. The design of MODNet benefits from optimizing a series of correlated sub-objectives simultaneously via explicit constraints. Moreover, since trimap-free methods usually suffer from the domain shift problem in practice, we introduce (1) a self-supervised strategy based on sub-objectives consistency to adapt MODNet to real-world data and (2) a one-frame delay trick to smooth the results when applying MODNet to video human matting.   MODNet is easy to be trained in an end-to-end style. It is much faster than contemporaneous matting methods and runs at 63 frames per second. On a carefully designed human matting benchmark newly proposed in this work, MODNet greatly outperforms prior trimap-free methods. More importantly, our method achieves remarkable results in daily photos and videos. Now, do you really need a green screen for real-time human matting?

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Is a Green Screen Really Necessary for Real-Time Human Matting?<br>pdf: <a href="https://t.co/qmjULLGpJB">https://t.co/qmjULLGpJB</a><br>abs: <a href="https://t.co/NA9ctiRBiS">https://t.co/NA9ctiRBiS</a> <a href="https://t.co/1rXV7W0EJ9">pic.twitter.com/1rXV7W0EJ9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331434102587875329?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Differentially Private Learning Needs Better Features (or Much More  Data)

Florian Tramèr, Dan Boneh

- retweets: 258, favorites: 100 (11/26/2020 10:46:11)

- links: [abs](https://arxiv.org/abs/2011.11660) | [pdf](https://arxiv.org/pdf/2011.11660)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent)

We demonstrate that differentially private machine learning has not yet reached its "AlexNet moment" on many canonical vision tasks: linear models trained on handcrafted features significantly outperform end-to-end deep neural networks for moderate privacy budgets. To exceed the performance of handcrafted features, we show that private learning requires either much more private data, or access to features learned on public data from a similar domain. Our work introduces simple yet strong baselines for differentially private learning that can inform the evaluation of future progress in this area.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Current algorithms for training neural nets with differential privacy greatly hurt model accuracy.<br><br>Can we do better? Yes!<br>With <a href="https://twitter.com/danboneh?ref_src=twsrc%5Etfw">@danboneh</a> we show how to get better private models by...not using deep learning!<br><br>Paper: <a href="https://t.co/5jMfcq2NXZ">https://t.co/5jMfcq2NXZ</a><br>Code: <a href="https://t.co/ZnudaQrZ9Q">https://t.co/ZnudaQrZ9Q</a> <a href="https://t.co/qFTpJeJ8WC">pic.twitter.com/qFTpJeJ8WC</a></p>&mdash; Florian Tramèr (@florian_tramer) <a href="https://twitter.com/florian_tramer/status/1331680382803034112?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color  Histograms

Mahmoud Afifi, Marcus A. Brubaker, Michael S. Brown

- retweets: 218, favorites: 96 (11/26/2020 10:46:11)

- links: [abs](https://arxiv.org/abs/2011.11731) | [pdf](https://arxiv.org/pdf/2011.11731)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

While generative adversarial networks (GANs) can successfully produce high-quality images, they can be challenging to control. Simplifying GAN-based image generation is critical for their adoption in graphic design and artistic work. This goal has led to significant interest in methods that can intuitively control the appearance of images generated by GANs. In this paper, we present HistoGAN, a color histogram-based method for controlling GAN-generated images' colors. We focus on color histograms as they provide an intuitive way to describe image color while remaining decoupled from domain-specific semantics. Specifically, we introduce an effective modification of the recent StyleGAN architecture to control the colors of GAN-generated images specified by a target color histogram feature. We then describe how to expand HistoGAN to recolor real images. For image recoloring, we jointly train an encoder network along with HistoGAN. The recoloring model, ReHistoGAN, is an unsupervised approach trained to encourage the network to keep the original image's content while changing the colors based on the given target histogram. We show that this histogram-based approach offers a better way to control GAN-generated and real images' colors while producing more compelling results compared to existing alternative strategies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms<br>pdf: <a href="https://t.co/ejPdYZCPE0">https://t.co/ejPdYZCPE0</a><br>abs: <a href="https://t.co/saIMtEYXE4">https://t.co/saIMtEYXE4</a> <a href="https://t.co/wuNESux4XE">pic.twitter.com/wuNESux4XE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331413656270237699?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. MicroNet: Towards Image Recognition with Extremely Low FLOPs

Yunsheng Li, Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu, Lu Yuan, Zicheng Liu, Lei Zhang, Nuno Vasconcelos

- retweets: 86, favorites: 79 (11/26/2020 10:46:12)

- links: [abs](https://arxiv.org/abs/2011.12289) | [pdf](https://arxiv.org/pdf/2011.12289)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this paper, we present MicroNet, which is an efficient convolutional neural network using extremely low computational cost (e.g. 6 MFLOPs on ImageNet classification). Such a low cost network is highly desired on edge devices, yet usually suffers from a significant performance degradation. We handle the extremely low FLOPs based upon two design principles: (a) avoiding the reduction of network width by lowering the node connectivity, and (b) compensating for the reduction of network depth by introducing more complex non-linearity per layer. Firstly, we propose Micro-Factorized convolution to factorize both pointwise and depthwise convolutions into low rank matrices for a good tradeoff between the number of channels and input/output connectivity. Secondly, we propose a new activation function, named Dynamic Shift-Max, to improve the non-linearity via maxing out multiple dynamic fusions between an input feature map and its circular channel shift. The fusions are dynamic as their parameters are adapted to the input. Building upon Micro-Factorized convolution and dynamic Shift-Max, a family of MicroNets achieve a significant performance gain over the state-of-the-art in the low FLOP regime. For instance, MicroNet-M1 achieves 61.1% top-1 accuracy on ImageNet classification with 12 MFLOPs, outperforming MobileNetV3 by 11.3%.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MicroNet: Towards Image Recognition with Extremely Low FLOPs<a href="https://t.co/wpBenRPA00">https://t.co/wpBenRPA00</a> <a href="https://t.co/xoMfUBtQKg">pic.twitter.com/xoMfUBtQKg</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1331421258651602946?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MicroNet: Towards Image Recognition with Extremely Low FLOPs<br>pdf: <a href="https://t.co/R0jqr1E6F5">https://t.co/R0jqr1E6F5</a><br>abs: <a href="https://t.co/p1Z5B4gc3n">https://t.co/p1Z5B4gc3n</a> <a href="https://t.co/yisKFisAiI">pic.twitter.com/yisKFisAiI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331450411061022721?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Benchmarking Image Retrieval for Visual Localization

Noé Pion, Martin Humenberger, Gabriela Csurka, Yohann Cabon, Torsten Sattler

- retweets: 92, favorites: 72 (11/26/2020 10:46:12)

- links: [abs](https://arxiv.org/abs/2011.11946) | [pdf](https://arxiv.org/pdf/2011.11946)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Visual localization, i.e., camera pose estimation in a known scene, is a core component of technologies such as autonomous driving and augmented reality. State-of-the-art localization approaches often rely on image retrieval techniques for one of two tasks: (1) provide an approximate pose estimate or (2) determine which parts of the scene are potentially visible in a given query image. It is common practice to use state-of-the-art image retrieval algorithms for these tasks. These algorithms are often trained for the goal of retrieving the same landmark under a large range of viewpoint changes. However, robustness to viewpoint changes is not necessarily desirable in the context of visual localization. This paper focuses on understanding the role of image retrieval for multiple visual localization tasks. We introduce a benchmark setup and compare state-of-the-art retrieval representations on multiple datasets. We show that retrieval performance on classical landmark retrieval/recognition tasks correlates only for some but not all tasks to localization performance. This indicates a need for retrieval approaches specifically designed for localization tasks. Our benchmark and evaluation protocols are available at https://github.com/naver/kapture-localization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Benchmarking Image Retrieval for Visual Localization&quot; from <a href="https://twitter.com/Poyonoz?ref_src=twsrc%5Etfw">@Poyonoz</a> <a href="https://twitter.com/SattlerTorsten?ref_src=twsrc%5Etfw">@SattlerTorsten</a>  <br>is so cool, that I wrote an overview.<br><br>And my take-home messages are different from the paper conclusions ;)<a href="https://t.co/ffUmtMkWQL">https://t.co/ffUmtMkWQL</a><br><br>paper: <a href="https://t.co/HQghxgU3oF">https://t.co/HQghxgU3oF</a><br>tl;dr below:<br>1/4 <a href="https://t.co/dKzvzi64qv">pic.twitter.com/dKzvzi64qv</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1331554541813329921?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">If you are attending <a href="https://twitter.com/hashtag/3DV2020?src=hash&amp;ref_src=twsrc%5Etfw">#3DV2020</a>, please stop by our poster today: <a href="https://twitter.com/Poyonoz?ref_src=twsrc%5Etfw">@Poyonoz</a>, Martin Humenberger, Gabriela Csurka, Yohann Cabon, Torsten Sattler, Benchmarking Image Retrieval for Visual Localization, 3DV 2020, <a href="https://t.co/GOQrMwvAWr">https://t.co/GOQrMwvAWr</a><br>Times (CET): 7am and 5:30pm</p>&mdash; Torsten Sattler (@SattlerTorsten) <a href="https://twitter.com/SattlerTorsten/status/1331484691652874241?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Adversarial Generation of Continuous Images

Ivan Skorokhodov, Savva Ignatyev, Mohamed Elhoseiny

- retweets: 30, favorites: 47 (11/26/2020 10:46:12)

- links: [abs](https://arxiv.org/abs/2011.12026) | [pdf](https://arxiv.org/pdf/2011.12026)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In most existing learning systems, images are typically viewed as 2D pixel arrays. However, in another paradigm gaining popularity, a 2D image is represented as an implicit neural representation (INR) -- an MLP that predicts an RGB pixel value given its (x,y) coordinate. In this paper, we propose two novel architectural techniques for building INR-based image decoders: factorized multiplicative modulation and multi-scale INRs, and use them to build a state-of-the-art continuous image GAN. Previous attempts to adapt INRs for image generation were limited to MNIST-like datasets and do not scale to complex real-world data. Our proposed architectural design improves the performance of continuous image generators by x6-40 times and reaches FID scores of 6.27 on LSUN bedroom 256x256 and 16.32 on FFHQ 1024x1024, greatly reducing the gap between continuous image GANs and pixel-based ones. To the best of our knowledge, these are the highest reported scores for an image generator, that consists entirely of fully-connected layers. Apart from that, we explore several exciting properties of INR-based decoders, like out-of-the-box superresolution, meaningful image-space interpolation, accelerated inference of low-resolution images, an ability to extrapolate outside of image boundaries and strong geometric prior. The source code is available at https://github.com/universome/inr-gan

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Adversarial Generation of Continuous Images<br>pdf: <a href="https://t.co/PPc0geUIKL">https://t.co/PPc0geUIKL</a><br>abs: <a href="https://t.co/AHyozdAHXc">https://t.co/AHyozdAHXc</a> <a href="https://t.co/10c4zH558N">pic.twitter.com/10c4zH558N</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331417131469115394?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. GIRAFFE: Representing Scenes as Compositional Generative Neural Feature  Fields

Michael Niemeyer, Andreas Geiger

- retweets: 36, favorites: 29 (11/26/2020 10:46:12)

- links: [abs](https://arxiv.org/abs/2011.12100) | [pdf](https://arxiv.org/pdf/2011.12100)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep generative models allow for photorealistic image synthesis at high resolutions. But for many applications, this is not enough: content creation also needs to be controllable. While several recent works investigate how to disentangle underlying factors of variation in the data, most of them operate in 2D and hence ignore that our world is three-dimensional. Further, only few works consider the compositional nature of scenes. Our key hypothesis is that incorporating a compositional 3D scene representation into the generative model leads to more controllable image synthesis. Representing scenes as compositional generative neural feature fields allows us to disentangle one or multiple objects from the background as well as individual objects' shapes and appearances while learning from unstructured and unposed image collections without any additional supervision. Combining this scene representation with a neural rendering pipeline yields a fast and realistic image synthesis model. As evidenced by our experiments, our model is able to disentangle individual objects and allows for translating and rotating them in the scene as well as changing the camera pose.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields<br>pdf: <a href="https://t.co/0zfGaDDRWQ">https://t.co/0zfGaDDRWQ</a><br>abs: <a href="https://t.co/xxlpLZbVGx">https://t.co/xxlpLZbVGx</a> <a href="https://t.co/5WraAgsh83">pic.twitter.com/5WraAgsh83</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331447877458141184?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Energy-Based Models for Continual Learning

Shuang Li, Yilun Du, Gido M. van de Ven, Antonio Torralba, Igor Mordatch

- retweets: 8, favorites: 54 (11/26/2020 10:46:12)

- links: [abs](https://arxiv.org/abs/2011.12216) | [pdf](https://arxiv.org/pdf/2011.12216)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We motivate Energy-Based Models (EBMs) as a promising model class for continual learning problems. Instead of tackling continual learning via the use of external memory, growing models, or regularization, EBMs have a natural way to support a dynamically-growing number of tasks or classes that causes less interference with previously learned information. We find that EBMs outperform the baseline methods by a large margin on several continual learning benchmarks. We also show that EBMs are adaptable to a more general continual learning setting where the data distribution changes without the notion of explicitly delineated tasks. These observations point towards EBMs as a class of models naturally inclined towards the continual learning regime.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our work investigating energy-based models for continual learning and how they are naturally less prone to catastrophic forgetting: <a href="https://t.co/IFwBaJwVnw">https://t.co/IFwBaJwVnw</a> with fantastic collaborators <a href="https://twitter.com/ShuangL13799063?ref_src=twsrc%5Etfw">@ShuangL13799063</a> <a href="https://twitter.com/du_yilun?ref_src=twsrc%5Etfw">@du_yilun</a> <a href="https://twitter.com/GMvandeVen?ref_src=twsrc%5Etfw">@GMvandeVen</a> and A. Torralba</p>&mdash; Igor Mordatch (@IMordatch) <a href="https://twitter.com/IMordatch/status/1331656487312130049?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Energy-based models are a class of flexible, powerful models with applications in many areas of deep learning. Could energy-based models also be useful for continual learning?<br><br>Yes! <a href="https://t.co/bp2Huaz9iV">https://t.co/bp2Huaz9iV</a> Work led by <a href="https://twitter.com/ShuangL13799063?ref_src=twsrc%5Etfw">@ShuangL13799063</a>.</p>&mdash; Gido van de Ven (@GMvandeVen) <a href="https://twitter.com/GMvandeVen/status/1331574498827624451?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. From Pixels to Legs: Hierarchical Learning of Quadruped Locomotion

Deepali Jain, Atil Iscen, Ken Caluwaerts

- retweets: 20, favorites: 32 (11/26/2020 10:46:13)

- links: [abs](https://arxiv.org/abs/2011.11722) | [pdf](https://arxiv.org/pdf/2011.11722)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Legged robots navigating crowded scenes and complex terrains in the real world are required to execute dynamic leg movements while processing visual input for obstacle avoidance and path planning. We show that a quadruped robot can acquire both of these skills by means of hierarchical reinforcement learning (HRL). By virtue of their hierarchical structure, our policies learn to implicitly break down this joint problem by concurrently learning High Level (HL) and Low Level (LL) neural network policies. These two levels are connected by a low dimensional hidden layer, which we call latent command. HL receives a first-person camera view, whereas LL receives the latent command from HL and the robot's on-board sensors to control its actuators. We train policies to walk in two different environments: a curved cliff and a maze. We show that hierarchical policies can concurrently learn to locomote and navigate in these environments, and show they are more efficient than non-hierarchical neural network policies. This architecture also allows for knowledge reuse across tasks. LL networks trained on one task can be transferred to a new task in a new environment. Finally HL, which processes camera images, can be evaluated at much lower and varying frequencies compared to LL, thus reducing computation times and bandwidth requirements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">From Pixels to Legs: Hierarchical Learning of Quadruped Locomotion<br>pdf: <a href="https://t.co/R9bu202ERa">https://t.co/R9bu202ERa</a><br>abs: <a href="https://t.co/beZtPYJMmq">https://t.co/beZtPYJMmq</a> <a href="https://t.co/JCb6mKbHoj">pic.twitter.com/JCb6mKbHoj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331475097694183424?ref_src=twsrc%5Etfw">November 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



