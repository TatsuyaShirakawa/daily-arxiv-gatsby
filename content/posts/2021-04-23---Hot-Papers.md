---
title: Hot Papers 2021-04-23
date: 2021-04-24T17:20:33.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-23"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-23"
socialImage: "/media/flying-marine.jpg"

---

# 1. VATT: Transformers for Multimodal Self-Supervised Learning from Raw  Video, Audio and Text

Hassan Akbari, Linagzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, Boqing Gong

- retweets: 5490, favorites: 256 (04/24/2021 17:20:33)

- links: [abs](https://arxiv.org/abs/2104.11178) | [pdf](https://arxiv.org/pdf/2104.11178)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

We present a framework for learning multimodal representations from unlabeled data using convolution-free Transformer architectures. Specifically, our Video-Audio-Text Transformer (VATT) takes raw signals as inputs and extracts multimodal representations that are rich enough to benefit a variety of downstream tasks. We train VATT end-to-end from scratch using multimodal contrastive losses and evaluate its performance by the downstream tasks of video action recognition, audio event classification, image classification, and text-to-video retrieval. Furthermore, we study a modality-agnostic single-backbone Transformer by sharing weights among the three modalities. We show that the convolution-free VATT outperforms state-of-the-art ConvNet-based architectures in the downstream tasks. Especially, VATT's vision Transformer achieves the top-1 accuracy of 82.1% on Kinetics-400, 83.6% on Kinetics-600,and 41.1% on Moments in Time, new records while avoiding supervised pre-training. Transferring to image classification leads to 78.7% top-1 accuracy on ImageNet compared to 64.7% by training the same Transformer from scratch, showing the generalizability of our model despite the domain gap between videos and images. VATT's audio Transformer also sets a new record on waveform-based audio event recognition by achieving the mAP of 39.4% on AudioSet without any supervised pre-training.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing Video-Audio-Text Transformer (VATT)!<br><br>VATT is a conv-free Transformer trained from scratch on unlabeled raw video, audio waveform and text, achieving fine-tuning accuracies of 82.1% on Kinetics-400, 39.4% on AudioSet and 78.7% on ImageNet.<a href="https://t.co/ZKd8eCx8uM">https://t.co/ZKd8eCx8uM</a> <a href="https://t.co/Gy5knh9Y5a">pic.twitter.com/Gy5knh9Y5a</a></p>&mdash; Yin Cui (@YinCui1) <a href="https://twitter.com/YinCui1/status/1385592727174205440?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text<br><br>Proposes a model for learning multimodal representations from unlabeled data using convolution-free Transformer architectures. <a href="https://t.co/4q7hiNYIBP">https://t.co/4q7hiNYIBP</a> <a href="https://t.co/UYhsOmtO38">pic.twitter.com/UYhsOmtO38</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1385396656821075969?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Multiscale Vision Transformers

Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer

- retweets: 2642, favorites: 320 (04/24/2021 17:20:33)

- links: [abs](https://arxiv.org/abs/2104.11227) | [pdf](https://arxiv.org/pdf/2104.11227)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present Multiscale Vision Transformers (MViT) for video and image recognition, by connecting the seminal idea of multiscale feature hierarchies with transformer models. Multiscale Transformers have several channel-resolution scale stages. Starting from the input resolution and a small channel dimension, the stages hierarchically expand the channel capacity while reducing the spatial resolution. This creates a multiscale pyramid of features with early layers operating at high spatial resolution to model simple low-level visual information, and deeper layers at spatially coarse, but complex, high-dimensional features. We evaluate this fundamental architectural prior for modeling the dense nature of visual signals for a variety of video recognition tasks where it outperforms concurrent vision transformers that rely on large scale external pre-training and are 5-10x more costly in computation and parameters. We further remove the temporal dimension and apply our model for image classification where it outperforms prior work on vision transformers. Code is available at: https://github.com/facebookresearch/SlowFast

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multiscale Vision Transformers<br><br>Outperforms 5-10x more costly vision transformers by combining the idea of multiscale feature hierarchies with Transformer.<br><br>abs: <a href="https://t.co/pDLzWcxBOg">https://t.co/pDLzWcxBOg</a><br>code: <a href="https://t.co/iLAhjCu12c">https://t.co/iLAhjCu12c</a> <a href="https://t.co/wG1TIUBrJV">pic.twitter.com/wG1TIUBrJV</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1385394672147734528?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multiscale Vision Transformers<br><br>&quot;We present Multiscale Vision Transformers (MViT) for<br>video and image recognition, by connecting the seminal idea of multiscale feature hierarchies with transformer models&quot;<br><br>pdf: <a href="https://t.co/fZ2Ftg1RiJ">https://t.co/fZ2Ftg1RiJ</a><br>abs: <a href="https://t.co/YyJXuAcZrY">https://t.co/YyJXuAcZrY</a> <a href="https://t.co/tfX5S4gyRp">pic.twitter.com/tfX5S4gyRp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385396764673462272?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Provable Limitations of Acquiring Meaning from Ungrounded Form: What  will Future Language Models Understand?

William Merrill, Yoav Goldberg, Roy Schwartz, Noah A. Smith

- retweets: 2358, favorites: 263 (04/24/2021 17:20:34)

- links: [abs](https://arxiv.org/abs/2104.10809) | [pdf](https://arxiv.org/pdf/2104.10809)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Language models trained on billions of tokens have recently led to unprecedented results on many NLP tasks. This success raises the question of whether, in principle, a system can ever "understand" raw text without access to some form of grounding. We formally investigate the abilities of ungrounded systems to acquire meaning. Our analysis focuses on the role of "assertions": contexts within raw text that provide indirect clues about underlying semantics. We study whether assertions enable a system to emulate representations preserving semantic relations like equivalence. We find that assertions enable semantic emulation if all expressions in the language are referentially transparent. However, if the language uses non-transparent patterns like variable binding, we show that emulation can become an uncomputable problem. Finally, we discuss differences between our formal model and natural language, exploring how our results generalize to a modal setting and other semantic relations. Together, our results suggest that assertions in code or language do not provide sufficient signal to fully emulate semantic representations. We formalize ways in which ungrounded language models appear to be fundamentally limited in their ability to "understand".

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Is it possible for GPT-n to &quot;understand&quot; the semantics of English? What about Python?<br><br>I&#39;m excited to finally share work  formalizing this question! We give formal languages that are *provably* un-understandable by LMs (within our setup, at least)<a href="https://t.co/fa9NqOuA63">https://t.co/fa9NqOuA63</a></p>&mdash; Will Merrill (@lambdaviking) <a href="https://twitter.com/lambdaviking/status/1385415712034803713?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Analyzing Monotonic Linear Interpolation in Neural Network Loss  Landscapes

James Lucas, Juhan Bae, Michael R. Zhang, Stanislav Fort, Richard Zemel, Roger Grosse

- retweets: 2100, favorites: 297 (04/24/2021 17:20:34)

- links: [abs](https://arxiv.org/abs/2104.11044) | [pdf](https://arxiv.org/pdf/2104.11044)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Linear interpolation between initial neural network parameters and converged parameters after training with stochastic gradient descent (SGD) typically leads to a monotonic decrease in the training objective. This Monotonic Linear Interpolation (MLI) property, first observed by Goodfellow et al. (2014) persists in spite of the non-convex objectives and highly non-linear training dynamics of neural networks. Extending this work, we evaluate several hypotheses for this property that, to our knowledge, have not yet been explored. Using tools from differential geometry, we draw connections between the interpolated paths in function space and the monotonicity of the network - providing sufficient conditions for the MLI property under mean squared error. While the MLI property holds under various settings (e.g. network architectures and learning problems), we show in practice that networks violating the MLI property can be produced systematically, by encouraging the weights to move far from initialization. The MLI property raises important questions about the loss landscape geometry of neural networks and highlights the need to further study their global properties.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Linear interpolation from initial to final neural net params typically decreases the loss monotonically<br><br>We investigate this phenomenon empirically and theoretically in new preprint <a href="https://t.co/sEmOpcME93">https://t.co/sEmOpcME93</a><br>w/ <a href="https://twitter.com/juhan_bae?ref_src=twsrc%5Etfw">@juhan_bae</a>, <a href="https://twitter.com/michaelrzhang?ref_src=twsrc%5Etfw">@michaelrzhang</a>, <a href="https://twitter.com/stanislavfort?ref_src=twsrc%5Etfw">@stanislavfort</a>, Rich Zemel, <a href="https://twitter.com/RogerGrosse?ref_src=twsrc%5Etfw">@RogerGrosse</a> <a href="https://t.co/ZrwocW34CU">pic.twitter.com/ZrwocW34CU</a></p>&mdash; James  Lucas (@james_r_lucas) <a href="https://twitter.com/james_r_lucas/status/1385586850061037568?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new paper is out <a href="https://t.co/OUU3UHfmTa">https://t.co/OUU3UHfmTa</a> led by <a href="https://twitter.com/james_r_lucas?ref_src=twsrc%5Etfw">@james_r_lucas</a>! Linear interpolation init-&gt;optimum often has ~monotonically decreasing loss &amp; this also holds on paths from unrelated(!) inits. Joint work w <a href="https://twitter.com/james_r_lucas?ref_src=twsrc%5Etfw">@james_r_lucas</a>, <a href="https://twitter.com/juhan_bae?ref_src=twsrc%5Etfw">@juhan_bae</a>, <a href="https://twitter.com/michaelrzhang?ref_src=twsrc%5Etfw">@michaelrzhang</a>, Rich Zemel, <a href="https://twitter.com/RogerGrosse?ref_src=twsrc%5Etfw">@RogerGrosse</a> <a href="https://t.co/fycMVKE0Bm">https://t.co/fycMVKE0Bm</a></p>&mdash; Stanislav Fort (@stanislavfort) <a href="https://twitter.com/stanislavfort/status/1385642187073806336?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. "I'm a Professor, which isn't usually a dangerous job":  Internet-Facilitated Harassment and its Impact on Researchers

Periwinkle Doerfler, Andrea Forte, Emiliano De Cristofaro, Gianluca Stringhini, Jeremy Blackburn, Damon McCoy

- retweets: 577, favorites: 95 (04/24/2021 17:20:34)

- links: [abs](https://arxiv.org/abs/2104.11145) | [pdf](https://arxiv.org/pdf/2104.11145)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

While the Internet has dramatically increased the exposure that research can receive, it has also facilitated harassment against scholars. To understand the impact that these attacks can have on the work of researchers, we perform a series of systematic interviews with researchers including academics, journalists, and activists, who have experienced targeted, Internet-facilitated harassment. We provide a framework for understanding the types of harassers that target researchers, the harassment that ensues, and the personal and professional impact on individuals and academic freedom. We then study preventative and remedial strategies available, and the institutions that prevent some of these strategies from being more effective. Finally, we discuss the ethical structures that could facilitate more equitable access to participating in research without serious personal suffering.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New pre-print out: <a href="https://t.co/mkDplEnDxZ">https://t.co/mkDplEnDxZ</a><br><br>Tl;dr <a href="https://twitter.com/PeriwinkleID?ref_src=twsrc%5Etfw">@PeriwinkleID</a> et al. present a study of online harassment targeted at researchers (N=17). We examine negative effects on careers and roles of institutions, and discuss a set of recommendations. <a href="https://t.co/TiNpXYryev">pic.twitter.com/TiNpXYryev</a></p>&mdash; Emiliano DC (@emilianoucl) <a href="https://twitter.com/emilianoucl/status/1385419618764922886?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control

Tomas Jakab, Richard Tucker, Ameesh Makadia, Jiajun Wu, Noah Snavely, Angjoo Kanazawa

- retweets: 462, favorites: 89 (04/24/2021 17:20:35)

- links: [abs](https://arxiv.org/abs/2104.11224) | [pdf](https://arxiv.org/pdf/2104.11224)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We introduce KeypointDeformer, a novel unsupervised method for shape control through automatically discovered 3D keypoints. We cast this as the problem of aligning a source 3D object to a target 3D object from the same object category. Our method analyzes the difference between the shapes of the two objects by comparing their latent representations. This latent representation is in the form of 3D keypoints that are learned in an unsupervised way. The difference between the 3D keypoints of the source and the target objects then informs the shape deformation algorithm that deforms the source object into the target object. The whole model is learned end-to-end and simultaneously discovers 3D keypoints while learning to use them for deforming object shapes. Our approach produces intuitive and semantically consistent control of shape deformations. Moreover, our discovered 3D keypoints are consistent across object category instances despite large shape variations. As our method is unsupervised, it can be readily deployed to new object categories without requiring annotations for 3D keypoints and deformations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control<br>pdf: <a href="https://t.co/ixhc6KrrZD">https://t.co/ixhc6KrrZD</a><br>abs: <a href="https://t.co/JdZQG7ibVf">https://t.co/JdZQG7ibVf</a><br>project page: <a href="https://t.co/cLrqBt4E12">https://t.co/cLrqBt4E12</a> <a href="https://t.co/YnxN7Xvp9m">pic.twitter.com/YnxN7Xvp9m</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385399150104227842?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Pri3D: Can 3D Priors Help 2D Representation Learning?

Ji Hou, Saining Xie, Benjamin Graham, Angela Dai, Matthias Nießner

- retweets: 302, favorites: 113 (04/24/2021 17:20:35)

- links: [abs](https://arxiv.org/abs/2104.11225) | [pdf](https://arxiv.org/pdf/2104.11225)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent advances in 3D perception have shown impressive progress in understanding geometric structures of 3Dshapes and even scenes. Inspired by these advances in geometric understanding, we aim to imbue image-based perception with representations learned under geometric constraints. We introduce an approach to learn view-invariant,geometry-aware representations for network pre-training, based on multi-view RGB-D data, that can then be effectively transferred to downstream 2D tasks. We propose to employ contrastive learning under both multi-view im-age constraints and image-geometry constraints to encode3D priors into learned 2D representations. This results not only in improvement over 2D-only representation learning on the image-based tasks of semantic segmentation, instance segmentation, and object detection on real-world in-door datasets, but moreover, provides significant improvement in the low data regime. We show a significant improvement of 6.0% on semantic segmentation on full data as well as 11.9% on 20% data against baselines on ScanNet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our work Pri3D, learning 3D priors for 2D scene understanding tasks. Pre-training on 3D learns complementary features to ImageNet pre-trained model, leading to an 11.9% improvement on 20% data.<a href="https://t.co/ep2thjGEhL">https://t.co/ep2thjGEhL</a><br>w/ <a href="https://twitter.com/sainingxie?ref_src=twsrc%5Etfw">@sainingxie</a> ben <a href="https://twitter.com/angelaqdai?ref_src=twsrc%5Etfw">@angelaqdai</a> <a href="https://twitter.com/MattNiessner?ref_src=twsrc%5Etfw">@MattNiessner</a> <a href="https://t.co/QImPmzHV2K">pic.twitter.com/QImPmzHV2K</a></p>&mdash; Ji Hou (@sekunde_) <a href="https://twitter.com/sekunde_/status/1385657969321418761?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pri3D: Can 3D Priors Help 2D Representation Learning?<br>pdf: <a href="https://t.co/HUTaUVNQMl">https://t.co/HUTaUVNQMl</a><br>abs: <a href="https://t.co/3b3RMaRgsS">https://t.co/3b3RMaRgsS</a> <a href="https://t.co/9t3idKy2sQ">pic.twitter.com/9t3idKy2sQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385413422741966850?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. The Road Less Travelled: Trying And Failing To Generate Walking  Simulators

Michael Cook

- retweets: 289, favorites: 106 (04/24/2021 17:20:35)

- links: [abs](https://arxiv.org/abs/2104.10789) | [pdf](https://arxiv.org/pdf/2104.10789)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Automated game design is a rapidly growing area of research, yet many aspects of game design lie largely unexamined still, as most systems focus on two-dimensional games with clear objectives and goal-oriented gameplay. This paper describes several attempts to build an automated game designer for 3D games more focused on space, atmosphere and experience. We describe our attempts to build these systems, why they failed, and what steps and future work we believe would be useful for future attempts by others.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New Paper: The Road Less Travelled: Trying And Failing To Generate Walking Simulators<br><br>I wrote a reflection on several abandoned projects I made between 2014 and 2016, trying to generate 3D walking simulators and similar games. <br><br>🌲🐕🌲 <a href="https://t.co/me8ZIrISZZ">https://t.co/me8ZIrISZZ</a> <a href="https://t.co/5c4lqEGnal">pic.twitter.com/5c4lqEGnal</a></p>&mdash; mike cook (@mtrc) <a href="https://twitter.com/mtrc/status/1385585989528268802?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation

Gaurav Parmar, Richard Zhang, Jun-Yan Zhu

- retweets: 180, favorites: 51 (04/24/2021 17:20:35)

- links: [abs](https://arxiv.org/abs/2104.11222) | [pdf](https://arxiv.org/pdf/2104.11222)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We investigate the sensitivity of the Fr\'echet Inception Distance (FID) score to inconsistent and often incorrect implementations across different image processing libraries. FID score is widely used to evaluate generative models, but each FID implementation uses a different low-level image processing process. Image resizing functions in commonly-used deep learning libraries often introduce aliasing artifacts. We observe that numerous subtle choices need to be made for FID calculation and a lack of consistencies in these choices can lead to vastly different FID scores. In particular, we show that the following choices are significant: (1) selecting what image resizing library to use, (2) choosing what interpolation kernel to use, (3) what encoding to use when representing images. We additionally outline numerous common pitfalls that should be avoided and provide recommendations for computing the FID score accurately. We provide an easy-to-use optimized implementation of our proposed recommendations in the accompanying code.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation<br>pdf: <a href="https://t.co/Amq6mkBE0i">https://t.co/Amq6mkBE0i</a><br>abs: <a href="https://t.co/ZPnBxQfA6p">https://t.co/ZPnBxQfA6p</a><br>project page: <a href="https://t.co/jbFCmdrk8X">https://t.co/jbFCmdrk8X</a><br>github: <a href="https://t.co/AYSp4N6Zmf">https://t.co/AYSp4N6Zmf</a> <a href="https://t.co/ztelgFZHmu">pic.twitter.com/ztelgFZHmu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385402352295944197?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Token Labeling: Training a 85.5% Top-1 Accuracy Vision Transformer with  56M Parameters on ImageNet

Zihang Jiang, Qibin Hou, Li Yuan, Daquan Zhou, Xiaojie Jin, Anran Wang, Jiashi Feng

- retweets: 170, favorites: 53 (04/24/2021 17:20:35)

- links: [abs](https://arxiv.org/abs/2104.10858) | [pdf](https://arxiv.org/pdf/2104.10858)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper provides a strong baseline for vision transformers on the ImageNet classification task. While recent vision transformers have demonstrated promising results in ImageNet classification, their performance still lags behind powerful convolutional neural networks (CNNs) with approximately the same model size. In this work, instead of describing a novel transformer architecture, we explore the potential of vision transformers in ImageNet classification by developing a bag of training techniques. We show that by slightly tuning the structure of vision transformers and introducing token labeling -- a new training objective, our models are able to achieve better results than the CNN counterparts and other transformer-based classification models with similar amount of training parameters and computations. Taking a vision transformer with 26M learnable parameters as an example, we can achieve a 84.4% Top-1 accuracy on ImageNet. When the model size is scaled up to 56M/150M, the result can be further increased to 85.4%/86.2% without extra data. We hope this study could provide researchers with useful techniques to train powerful vision transformers. Our code and all the training details will be made publicly available at https://github.com/zihangJiang/TokenLabeling.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Token Labeling: Training a 85.5% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet<br>pdf: <a href="https://t.co/SRb1inymP0">https://t.co/SRb1inymP0</a><br>abs: <a href="https://t.co/7Fd0RNdSYb">https://t.co/7Fd0RNdSYb</a> <a href="https://t.co/EjMwxNxlhV">pic.twitter.com/EjMwxNxlhV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385403022449205252?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. ImageNet-21K Pretraining for the Masses

Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, Lihi Zelnik-Manor

- retweets: 169, favorites: 44 (04/24/2021 17:20:35)

- links: [abs](https://arxiv.org/abs/2104.10972) | [pdf](https://arxiv.org/pdf/2104.10972)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

ImageNet-1K serves as the primary dataset for pretraining deep learning models for computer vision tasks. ImageNet-21K dataset, which contains more pictures and classes, is used less frequently for pretraining, mainly due to its complexity, and underestimation of its added value compared to standard ImageNet-1K pretraining. This paper aims to close this gap, and make high-quality efficient pretraining on ImageNet-21K available for everyone. % Via a dedicated preprocessing stage, utilizing WordNet hierarchies, and a novel training scheme called semantic softmax, we show that various models, including small mobile-oriented models, significantly benefit from ImageNet-21K pretraining on numerous datasets and tasks. We also show that we outperform previous ImageNet-21K pretraining schemes for prominent new models like ViT. % Our proposed pretraining pipeline is efficient, accessible, and leads to SoTA reproducible results, from a publicly available dataset. The training code and pretrained models are available at: https://github.com/Alibaba-MIIL/ImageNet21K

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ImageNet-21K Pretraining for the Masses<br>pdf: <a href="https://t.co/Iqx7HYXbi7">https://t.co/Iqx7HYXbi7</a><br>abs: <a href="https://t.co/9HNKp3pA8L">https://t.co/9HNKp3pA8L</a><br>github: <a href="https://t.co/dguO6v9mlg">https://t.co/dguO6v9mlg</a> <a href="https://t.co/K29B4NpAzA">pic.twitter.com/K29B4NpAzA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385407457674006530?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. So-ViT: Mind Visual Tokens for Vision Transformer

Jiangtao Xie, Ruiren Zeng, Qilong Wang, Ziqi Zhou, Peihua Li

- retweets: 87, favorites: 41 (04/24/2021 17:20:36)

- links: [abs](https://arxiv.org/abs/2104.10935) | [pdf](https://arxiv.org/pdf/2104.10935)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recently the vision transformer (ViT) architecture, where the backbone purely consists of self-attention mechanism, has achieved very promising performance in visual classification. However, the high performance of the original ViT heavily depends on pretraining using ultra large-scale datasets, and it significantly underperforms on ImageNet-1K if trained from scratch. This paper makes the efforts toward addressing this problem, by carefully considering the role of visual tokens. First, for classification head, existing ViT only exploits class token while entirely neglecting rich semantic information inherent in high-level visual tokens. Therefore, we propose a new classification paradigm, where the second-order, cross-covariance pooling of visual tokens is combined with class token for final classification. Meanwhile, a fast singular value power normalization is proposed for improving the second-order pooling. Second, the original ViT employs the naive embedding of fixed-size image patches, lacking the ability to model translation equivariance and locality. To alleviate this problem, we develop a light-weight, hierarchical module based on off-the-shelf convolutions for visual token embedding. The proposed architecture, which we call So-ViT, is thoroughly evaluated on ImageNet-1K. The results show our models, when trained from scratch, outperform the competing ViT variants, while being on par with or better than state-of-the-art CNN models. Code is available at https://github.com/jiangtaoxie/So-ViT

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">So-ViT: Mind Visual Tokens for Vision Transformer<br>pdf: <a href="https://t.co/MIRQsEH8BT">https://t.co/MIRQsEH8BT</a><br>abs: <a href="https://t.co/d6Kr1RKhbj">https://t.co/d6Kr1RKhbj</a><br>&quot;when trained from scratch, outperform the competing ViT variants, while being on par with or better than state-of-the-art CNN models&quot; <a href="https://t.co/hzaYDz8T4M">pic.twitter.com/hzaYDz8T4M</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385395579652644876?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Cross-Domain and Disentangled Face Manipulation with 3D Guidance

Can Wang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao

- retweets: 56, favorites: 39 (04/24/2021 17:20:36)

- links: [abs](https://arxiv.org/abs/2104.11228) | [pdf](https://arxiv.org/pdf/2104.11228)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Face image manipulation via three-dimensional guidance has been widely applied in various interactive scenarios due to its semantically-meaningful understanding and user-friendly controllability. However, existing 3D-morphable-model-based manipulation methods are not directly applicable to out-of-domain faces, such as non-photorealistic paintings, cartoon portraits, or even animals, mainly due to the formidable difficulties in building the model for each specific face domain. To overcome this challenge, we propose, as far as we know, the first method to manipulate faces in arbitrary domains using human 3DMM. This is achieved through two major steps: 1) disentangled mapping from 3DMM parameters to the latent space embedding of a pre-trained StyleGAN2 that guarantees disentangled and precise controls for each semantic attribute; and 2) cross-domain adaptation that bridges domain discrepancies and makes human 3DMM applicable to out-of-domain faces by enforcing a consistent latent space embedding. Experiments and comparisons demonstrate the superiority of our high-quality semantic manipulation method on a variety of face domains with all major 3D facial attributes controllable: pose, expression, shape, albedo, and illumination. Moreover, we develop an intuitive editing interface to support user-friendly control and instant feedback. Our project page is https://cassiepython.github.io/sigasia/cddfm3d.html.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Cross-Domain and Disentangled Face Manipulation with 3D Guidance<br>pdf: <a href="https://t.co/sOV82Guhnv">https://t.co/sOV82Guhnv</a><br>abs: <a href="https://t.co/wcAn8xQD0A">https://t.co/wcAn8xQD0A</a><br>project page: <a href="https://t.co/ObVTIsDZex">https://t.co/ObVTIsDZex</a> <a href="https://t.co/mtCiAhXCQg">pic.twitter.com/mtCiAhXCQg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385400562259271684?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Hierarchical Motion Understanding via Motion Programs

Sumith Kulal, Jiayuan Mao, Alex Aiken, Jiajun Wu

- retweets: 42, favorites: 50 (04/24/2021 17:20:36)

- links: [abs](https://arxiv.org/abs/2104.11216) | [pdf](https://arxiv.org/pdf/2104.11216)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Current approaches to video analysis of human motion focus on raw pixels or keypoints as the basic units of reasoning. We posit that adding higher-level motion primitives, which can capture natural coarser units of motion such as backswing or follow-through, can be used to improve downstream analysis tasks. This higher level of abstraction can also capture key features, such as loops of repeated primitives, that are currently inaccessible at lower levels of representation. We therefore introduce Motion Programs, a neuro-symbolic, program-like representation that expresses motions as a composition of high-level primitives. We also present a system for automatically inducing motion programs from videos of human motion and for leveraging motion programs in video synthesis. Experiments show that motion programs can accurately describe a diverse set of human motions and the inferred programs contain semantically meaningful motion primitives, such as arm swings and jumping jacks. Our representation also benefits downstream tasks such as video interpolation and video prediction and outperforms off-the-shelf models. We further demonstrate how these programs can detect diverse kinds of repetitive motion and facilitate interactive video editing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical Motion Understanding via Motion Programs<br>pdf: <a href="https://t.co/oMJZwjRzi2">https://t.co/oMJZwjRzi2</a><br>abs: <a href="https://t.co/np7iRz8zVB">https://t.co/np7iRz8zVB</a><br>project page: <a href="https://t.co/zHw0Kjxtyw">https://t.co/zHw0Kjxtyw</a> <a href="https://t.co/viwsR1JXGW">pic.twitter.com/viwsR1JXGW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385406108014161921?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Automated Tackle Injury Risk Assessment in Contact-Based Sports -- A  Rugby Union Example

Zubair Martin, Amir Patel, Sharief Hendricks

- retweets: 42, favorites: 24 (04/24/2021 17:20:36)

- links: [abs](https://arxiv.org/abs/2104.10916) | [pdf](https://arxiv.org/pdf/2104.10916)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Video analysis in tackle-collision based sports is highly subjective and exposed to bias, which is inherent in human observation, especially under time constraints. This limitation of match analysis in tackle-collision based sports can be seen as an opportunity for computer vision applications. Objectively tracking, detecting and recognising an athlete's movements and actions during match play from a distance using video, along with our improved understanding of injury aetiology and skill execution will enhance our understanding how injury occurs, assist match day injury management, reduce referee subjectivity. In this paper, we present a system of objectively evaluating in-game tackle risk in rugby union matches. First, a ball detection model is trained using the You Only Look Once (YOLO) framework, these detections are then tracked by a Kalman Filter (KF). Following this, a separate YOLO model is used to detect persons/players within a tackle segment and then the ball-carrier and tackler are identified. Subsequently, we utilize OpenPose to determine the pose of ball-carrier and tackle, the relative pose of these is then used to evaluate the risk of the tackle. We tested the system on a diverse collection of rugby tackles and achieved an evaluation accuracy of 62.50%. These results will enable referees in tackle-contact based sports to make more subjective decisions, ultimately making these sports safer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New Paper: Automated Tackle Injury Risk Assessment in Contact-Based Sports -- A Rugby Union Example with <a href="https://twitter.com/UnitAfrican?ref_src=twsrc%5Etfw">@UnitAfrican</a> accepted in <a href="https://twitter.com/IEEEXplore?ref_src=twsrc%5Etfw">@IEEEXplore</a> Computer Society Conference on Computer Vision and Pattern Recognition Workshops <a href="https://t.co/Y7gyjKCtiR">https://t.co/Y7gyjKCtiR</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> 🏉🔬 <a href="https://t.co/JJcGALbqGh">pic.twitter.com/JJcGALbqGh</a></p>&mdash; Sharief Hendricks (@Sharief_H) <a href="https://twitter.com/Sharief_H/status/1385607983284080644?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Scalable Benchmarks for Gate-Based Quantum Computers

Arjan Cornelissen, Johannes Bausch, András Gilyén

- retweets: 6, favorites: 54 (04/24/2021 17:20:36)

- links: [abs](https://arxiv.org/abs/2104.10698) | [pdf](https://arxiv.org/pdf/2104.10698)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

In the near-term "NISQ"-era of noisy, intermediate-scale, quantum hardware and beyond, reliably determining the quality of quantum devices becomes increasingly important: users need to be able to compare them with one another, and make an estimate whether they are capable of performing a given task ahead of time. In this work, we develop and release an advanced quantum benchmarking framework in order to help assess the state of the art of current quantum devices. Our testing framework measures the performance of universal quantum devices in a hardware-agnostic way, with metrics that are aimed to facilitate an intuitive understanding of which device is likely to outperform others on a given task. This is achieved through six structured tests that allow for an immediate, visual assessment of how devices compare. Each test is designed with scalability in mind, making this framework not only suitable for testing the performance of present-day quantum devices, but also of those released in the foreseeable future. The series of tests are motivated by real-life scenarios, and therefore emphasise the interplay between various relevant characteristics of quantum devices, such as qubit count, connectivity, and gate and measurement fidelity. We present the benchmark results of twenty-one different quantum devices from IBM, Rigetti and IonQ.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Congrats to my student Arjan Cornelissen and his collaborators Johannes Bausch and András Gilyén on their paper that benchmarks 21 different quantum devices!<a href="https://t.co/ASLjo1vn1t">https://t.co/ASLjo1vn1t</a> <a href="https://t.co/gG5BMHlHnq">pic.twitter.com/gG5BMHlHnq</a></p>&mdash; Māris Ozols (@enclanglement) <a href="https://twitter.com/enclanglement/status/1385628712658472965?ref_src=twsrc%5Etfw">April 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



