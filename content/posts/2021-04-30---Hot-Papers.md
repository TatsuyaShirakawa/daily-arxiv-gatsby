---
title: Hot Papers 2021-04-30
date: 2021-05-01T18:50:48.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-30"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-30"
socialImage: "/media/flying-marine.jpg"

---

# 1. What Are Bayesian Neural Network Posteriors Really Like?

Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, Andrew Gordon Wilson

- retweets: 10066, favorites: 78 (05/01/2021 18:50:48)

- links: [abs](https://arxiv.org/abs/2104.14421) | [pdf](https://arxiv.org/pdf/2104.14421)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The posterior over Bayesian neural network (BNN) parameters is extremely high-dimensional and non-convex. For computational reasons, researchers approximate this posterior using inexpensive mini-batch methods such as mean-field variational inference or stochastic-gradient Markov chain Monte Carlo (SGMCMC). To investigate foundational questions in Bayesian deep learning, we instead use full-batch Hamiltonian Monte Carlo (HMC) on modern architectures. We show that (1) BNNs can achieve significant performance gains over standard training and deep ensembles; (2) a single long HMC chain can provide a comparable representation of the posterior to multiple shorter chains; (3) in contrast to recent studies, we find posterior tempering is not needed for near-optimal performance, with little evidence for a "cold posterior" effect, which we show is largely an artifact of data augmentation; (4) BMA performance is robust to the choice of prior scale, and relatively similar for diagonal Gaussian, mixture of Gaussian, and logistic priors; (5) Bayesian neural networks show surprisingly poor generalization under domain shift; (6) while cheaper alternatives such as deep ensembles and SGMCMC methods can provide good generalization, they provide distinct predictive distributions from HMC. Notably, deep ensemble predictive distributions are similarly close to HMC as standard SGLD, and closer than standard variational inference.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What are Bayesian neural network posteriors really like? With high fidelity HMC, we study approximate inference quality, generalization, cold posteriors, priors, and more. <a href="https://t.co/BhyOLlggoB">https://t.co/BhyOLlggoB</a><br>With <a href="https://twitter.com/Pavel_Izmailov?ref_src=twsrc%5Etfw">@Pavel_Izmailov</a>, <a href="https://twitter.com/sharadvikram?ref_src=twsrc%5Etfw">@sharadvikram</a>, and Matthew D. Hoffman. 1/10 <a href="https://t.co/4ByB6QA07g">pic.twitter.com/4ByB6QA07g</a></p>&mdash; Andrew Gordon Wilson (@andrewgwils) <a href="https://twitter.com/andrewgwils/status/1387929898099134473?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Emerging Properties in Self-Supervised Vision Transformers

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

- retweets: 3141, favorites: 249 (05/01/2021 18:50:49)

- links: [abs](https://arxiv.org/abs/2104.14294) | [pdf](https://arxiv.org/pdf/2104.14294)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we question if self-supervised learning provides new properties to Vision Transformer (ViT) that stand out compared to convolutional networks (convnets). Beyond the fact that adapting self-supervised methods to this architecture works particularly well, we make the following observations: first, self-supervised ViT features contain explicit information about the semantic segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets. Second, these features are also excellent k-NN classifiers, reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the importance of momentum encoder, multi-crop training, and the use of small patches with ViTs. We implement our findings into a simple self-supervised method, called DINO, which we interpret as a form of self-distillation with no labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on ImageNet in linear evaluation with ViT-Base.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Emerging Properties in Self-Supervised Vision Transformers <a href="https://t.co/UbIaCJLi9Q">https://t.co/UbIaCJLi9Q</a><br><br>object segmentation emerges out of ViT networks trained with self-supervision. This information is directly accessible in the self-attention modules of the last block. <a href="https://t.co/KCCpIg87z9">pic.twitter.com/KCCpIg87z9</a></p>&mdash; Ankur Handa (@ankurhandos) <a href="https://twitter.com/ankurhandos/status/1388152700458606594?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Experts, Errors, and Context: A Large-Scale Study of Human Evaluation  for Machine Translation

Markus Freitag, George Foster, David Grangier, Viresh Ratnakar, Qijun Tan, Wolfgang Macherey

- retweets: 2190, favorites: 189 (05/01/2021 18:50:49)

- links: [abs](https://arxiv.org/abs/2104.14478) | [pdf](https://arxiv.org/pdf/2104.14478)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Human evaluation of modern high-quality machine translation systems is a difficult problem, and there is increasing evidence that inadequate evaluation procedures can lead to erroneous conclusions. While there has been considerable research on human evaluation, the field still lacks a commonly-accepted standard procedure. As a step toward this goal, we propose an evaluation methodology grounded in explicit error analysis, based on the Multidimensional Quality Metrics (MQM) framework. We carry out the largest MQM research study to date, scoring the outputs of top systems from the WMT 2020 shared task in two language pairs using annotations provided by professional translators with access to full document context. We analyze the resulting data extensively, finding among other results a substantially different ranking of evaluated systems from the one established by the WMT crowd workers, exhibiting a clear preference for human over machine output. Surprisingly, we also find that automatic metrics based on pre-trained embeddings can outperform human crowd workers. We make our corpus publicly available for further research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to announce our most recent paper: Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation <a href="https://t.co/P8vDbxdTy9">https://t.co/P8vDbxdTy9</a> - We<br>re-evaluated the outputs from WMT20 using in-context MQM annotations from professional translators. <a href="https://t.co/On3VQEMJcI">pic.twitter.com/On3VQEMJcI</a></p>&mdash; Markus Freitag (@markuseful) <a href="https://twitter.com/markuseful/status/1388232296310874114?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. With a Little Help from My Friends: Nearest-Neighbor Contrastive  Learning of Visual Representations

Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, Andrew Zisserman

- retweets: 1625, favorites: 284 (05/01/2021 18:50:49)

- links: [abs](https://arxiv.org/abs/2104.14548) | [pdf](https://arxiv.org/pdf/2104.14548)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Self-supervised learning algorithms based on instance discrimination train encoders to be invariant to pre-defined transformations of the same instance. While most methods treat different views of the same image as positives for a contrastive loss, we are interested in using positives from other instances in the dataset. Our method, Nearest-Neighbor Contrastive Learning of visual Representations (NNCLR), samples the nearest neighbors from the dataset in the latent space, and treats them as positives. This provides more semantic variations than pre-defined transformations.   We find that using the nearest-neighbor as positive in contrastive losses improves performance significantly on ImageNet classification, from 71.7% to 75.6%, outperforming previous state-of-the-art methods. On semi-supervised learning benchmarks we improve performance significantly when only 1% ImageNet labels are available, from 53.8% to 56.5%. On transfer learning benchmarks our method outperforms state-of-the-art methods (including supervised learning with ImageNet) on 8 out of 12 downstream datasets. Furthermore, we demonstrate empirically that our method is less reliant on complex data augmentations. We see a relative reduction of only 2.1% ImageNet Top-1 accuracy when we train using only random crops.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Nearest-Neighbor Contrastive Learning of Visual Representations<br><br>Finds that using the nearest-neighbor as positive<br>in contrastive loss improves performance significantly<br>on ImageNet classification, from 71.7% to 75.6%, outperforming previous SotA methods.<a href="https://t.co/HxGZ1Z67FE">https://t.co/HxGZ1Z67FE</a> <a href="https://t.co/a7c1heelRn">pic.twitter.com/a7c1heelRn</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1387941266579550209?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations<br>pdf: <a href="https://t.co/DdianuJ0fG">https://t.co/DdianuJ0fG</a><br>abs: <a href="https://t.co/iTAhX2yTYS">https://t.co/iTAhX2yTYS</a><br><br>(NNCLR), samples the nearest neighbors from the dataset in the latent space, and treats them as positives <a href="https://t.co/aekyc8zXxM">pic.twitter.com/aekyc8zXxM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387960211135242240?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Dynabench: Rethinking Benchmarking in NLP

Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, Zhiyi Ma, Tristan Thrush, Sebastian Riedel, Zeerak Waseem, Pontus Stenetorp, Robin Jia, Mohit Bansal, Christopher Potts, Adina Williams

- retweets: 911, favorites: 148 (05/01/2021 18:50:50)

- links: [abs](https://arxiv.org/abs/2104.14337) | [pdf](https://arxiv.org/pdf/2104.14337)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We introduce Dynabench, an open-source platform for dynamic dataset creation and model benchmarking. Dynabench runs in a web browser and supports human-and-model-in-the-loop dataset creation: annotators seek to create examples that a target model will misclassify, but that another person will not. In this paper, we argue that Dynabench addresses a critical need in our community: contemporary models quickly achieve outstanding performance on benchmark tasks but nonetheless fail on simple challenge examples and falter in real-world scenarios. With Dynabench, dataset creation, model development, and model assessment can directly inform each other, leading to more robust and informative benchmarks. We report on four initial NLP tasks, illustrating these concepts and highlighting the promise of the platform, and address potential objections to dynamic benchmarking as a new standard for the field.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Dynabench paper, accepted at <a href="https://twitter.com/hashtag/NAACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#NAACL2021</a>, is out! The paper introduces our unified research platform for dynamic benchmarking on (so far) four initial NLU tasks.<a href="https://t.co/xsqwDWZBKO">https://t.co/xsqwDWZBKO</a><br><br>We also address some potential concerns and talk about future plans.<br><br>(1/4) <a href="https://t.co/TWPAh3uYku">pic.twitter.com/TWPAh3uYku</a></p>&mdash; Dynabench (@DynabenchAI) <a href="https://twitter.com/DynabenchAI/status/1388207516723515400?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">AI分野で人気なベンチマークにおいて，人間レベルのスコアに到達するまでの期間が短くなっている。特に言語の理解に着目して作成されたマルチタスクベンチマーク「GLUE」ではその期間が9ヶ月だった<br><br>Dynabench: Rethinking Benchmarking in NLP<a href="https://t.co/epxF2Muaaq">https://t.co/epxF2Muaaq</a> <a href="https://t.co/FgnM3SpBof">pic.twitter.com/FgnM3SpBof</a></p>&mdash; 小猫遊りょう（たかにゃし・りょう） (@jaguring1) <a href="https://twitter.com/jaguring1/status/1387970959164932099?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dynabench: Rethinking Benchmarking in NLP<br><br>Proposes Dynabench, an open-source platform for dynamic dataset creation and model benchmarking to address the problem that models quickly achieve good perf on benchmark tasks but falter in real-world scenarios.<a href="https://t.co/2hMuRZp01h">https://t.co/2hMuRZp01h</a> <a href="https://t.co/JKuyrxKBm9">pic.twitter.com/JKuyrxKBm9</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1387936801034170372?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. MarioNette: Self-Supervised Sprite Learning

Dmitriy Smirnov, Michael Gharbi, Matthew Fisher, Vitor Guizilini, Alexei A. Efros, Justin Solomon

- retweets: 418, favorites: 100 (05/01/2021 18:50:50)

- links: [abs](https://arxiv.org/abs/2104.14553) | [pdf](https://arxiv.org/pdf/2104.14553)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Visual content often contains recurring elements. Text is made up of glyphs from the same font, animations, such as cartoons or video games, are composed of sprites moving around the screen, and natural videos frequently have repeated views of objects. In this paper, we propose a deep learning approach for obtaining a graphically disentangled representation of recurring elements in a completely self-supervised manner. By jointly learning a dictionary of texture patches and training a network that places them onto a canvas, we effectively deconstruct sprite-based content into a sparse, consistent, and interpretable representation that can be easily used in downstream tasks. Our framework offers a promising approach for discovering recurring patterns in image collections without supervision.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MarioNette: Self-Supervised Sprite Learning<br>pdf: <a href="https://t.co/aSZVuf2f0g">https://t.co/aSZVuf2f0g</a><br>abs: <a href="https://t.co/SM8NzzPnBQ">https://t.co/SM8NzzPnBQ</a> <a href="https://t.co/FV3YSbxJc1">pic.twitter.com/FV3YSbxJc1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387958030361407493?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. LightTrack: Finding Lightweight Neural Networks for Object Tracking via  One-Shot Architecture Search

Bin Yan, Houwen Peng, Kan Wu, Dong Wang, Jianlong Fu, Huchuan Lu

- retweets: 380, favorites: 94 (05/01/2021 18:50:50)

- links: [abs](https://arxiv.org/abs/2104.14545) | [pdf](https://arxiv.org/pdf/2104.14545)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Object tracking has achieved significant progress over the past few years. However, state-of-the-art trackers become increasingly heavy and expensive, which limits their deployments in resource-constrained applications. In this work, we present LightTrack, which uses neural architecture search (NAS) to design more lightweight and efficient object trackers. Comprehensive experiments show that our LightTrack is effective. It can find trackers that achieve superior performance compared to handcrafted SOTA trackers, such as SiamRPN++ and Ocean, while using much fewer model Flops and parameters. Moreover, when deployed on resource-constrained mobile chipsets, the discovered trackers run much faster. For example, on Snapdragon 845 Adreno GPU, LightTrack runs $12\times$ faster than Ocean, while using $13\times$ fewer parameters and $38\times$ fewer Flops. Such improvements might narrow the gap between academic models and industrial deployments in object tracking task. LightTrack is released at https://github.com/researchmm/LightTrack.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search<br>pdf: <a href="https://t.co/nLhkQkyhKt">https://t.co/nLhkQkyhKt</a><br>abs: <a href="https://t.co/H7CkX85ERW">https://t.co/H7CkX85ERW</a> <a href="https://t.co/DIEe3XtC0B">pic.twitter.com/DIEe3XtC0B</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387970781552984064?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A Large-Scale Study on Unsupervised Spatiotemporal Representation  Learning

Christoph Feichtenhofer, Haoqi Fan, Bo Xiong, Ross Girshick, Kaiming He

- retweets: 301, favorites: 120 (05/01/2021 18:50:50)

- links: [abs](https://arxiv.org/abs/2104.14558) | [pdf](https://arxiv.org/pdf/2104.14558)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a large-scale study on unsupervised spatiotemporal representation learning from videos. With a unified perspective on four recent image-based frameworks, we study a simple objective that can easily generalize all these methods to space-time. Our objective encourages temporally-persistent features in the same video, and in spite of its simplicity, it works surprisingly well across: (i) different unsupervised frameworks, (ii) pre-training datasets, (iii) downstream datasets, and (iv) backbone architectures. We draw a series of intriguing observations from this study, e.g., we discover that encouraging long-spanned persistency can be effective even if the timespan is 60 seconds. In addition to state-of-the-art results in multiple benchmarks, we report a few promising cases in which unsupervised pre-training can outperform its supervised counterpart. Code is made available at https://github.com/facebookresearch/SlowFast

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning<br><br>In addition to SotA results in multiple benchmarks, they report a few promising cases in which unsupervised pre-training can outperform its supervised counterpart.<a href="https://t.co/42mWYxnZzh">https://t.co/42mWYxnZzh</a> <a href="https://t.co/ms4551LXYj">pic.twitter.com/ms4551LXYj</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1387929417360429058?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning<br>pdf: <a href="https://t.co/pbGlTeCDgP">https://t.co/pbGlTeCDgP</a><br>abs: <a href="https://t.co/2JidPsSsqP">https://t.co/2JidPsSsqP</a><br><br>We present a large-scale study on unsupervised spatiotem- poral representation learning from videos <a href="https://t.co/rBP3PYX6R1">pic.twitter.com/rBP3PYX6R1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387934133276581896?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth

Jamie Watson, Oisin Mac Aodha, Victor Prisacariu, Gabriel Brostow, Michael Firman

- retweets: 240, favorites: 57 (05/01/2021 18:50:51)

- links: [abs](https://arxiv.org/abs/2104.14540) | [pdf](https://arxiv.org/pdf/2104.14540)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Self-supervised monocular depth estimation networks are trained to predict scene depth using nearby frames as a supervision signal during training. However, for many applications, sequence information in the form of video frames is also available at test time. The vast majority of monocular networks do not make use of this extra signal, thus ignoring valuable information that could be used to improve the predicted depth. Those that do, either use computationally expensive test-time refinement techniques or off-the-shelf recurrent networks, which only indirectly make use of the geometric information that is inherently available.   We propose ManyDepth, an adaptive approach to dense depth estimation that can make use of sequence information at test time, when it is available. Taking inspiration from multi-view stereo, we propose a deep end-to-end cost volume based approach that is trained using self-supervision only. We present a novel consistency loss that encourages the network to ignore the cost volume when it is deemed unreliable, e.g. in the case of moving objects, and an augmentation scheme to cope with static cameras. Our detailed experiments on both KITTI and Cityscapes show that we outperform all published self-supervised baselines, including those that use single or multiple frames at test time.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth<br>pdf: <a href="https://t.co/nd41xIriCn">https://t.co/nd41xIriCn</a><br>abs: <a href="https://t.co/MurTMx1xbr">https://t.co/MurTMx1xbr</a><br>github: <a href="https://t.co/94FLcCJ2wV">https://t.co/94FLcCJ2wV</a> <a href="https://t.co/R2rMjxjE5g">pic.twitter.com/R2rMjxjE5g</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1388325519515607042?ref_src=twsrc%5Etfw">May 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Ensembling with Deep Generative Views

Lucy Chai, Jun-Yan Zhu, Eli Shechtman, Phillip Isola, Richard Zhang

- retweets: 143, favorites: 65 (05/01/2021 18:50:51)

- links: [abs](https://arxiv.org/abs/2104.14551) | [pdf](https://arxiv.org/pdf/2104.14551)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent generative models can synthesize "views" of artificial images that mimic real-world variations, such as changes in color or pose, simply by learning from unlabeled image collections. Here, we investigate whether such views can be applied to real images to benefit downstream analysis tasks such as image classification. Using a pretrained generator, we first find the latent code corresponding to a given real input image. Applying perturbations to the code creates natural variations of the image, which can then be ensembled together at test-time. We use StyleGAN2 as the source of generative augmentations and investigate this setup on classification tasks involving facial attributes, cat faces, and cars. Critically, we find that several design decisions are required towards making this process work; the perturbation procedure, weighting between the augmentations and original image, and training the classifier on synthesized images can all impact the result. Currently, we find that while test-time ensembling with GAN-based augmentations can offer some small improvements, the remaining bottlenecks are the efficiency and accuracy of the GAN reconstructions, coupled with classifier sensitivities to artifacts in GAN-generated images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Ensembling with Deep Generative Views<br>pdf: <a href="https://t.co/H1x0oOThBc">https://t.co/H1x0oOThBc</a><br>abs: <a href="https://t.co/yXdukf31ze">https://t.co/yXdukf31ze</a><br>project page: <a href="https://t.co/bgBTJNJ9O9">https://t.co/bgBTJNJ9O9</a> <a href="https://t.co/ttCbaryGre">pic.twitter.com/ttCbaryGre</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387932709817929728?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. PPFL: Privacy-preserving Federated Learning with Trusted Execution  Environments

Fan Mo, Hamed Haddadi, Kleomenis Katevas, Eduard Marin, Diego Perino, Nicolas Kourtellis

- retweets: 134, favorites: 37 (05/01/2021 18:50:51)

- links: [abs](https://arxiv.org/abs/2104.14380) | [pdf](https://arxiv.org/pdf/2104.14380)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose and implement a Privacy-preserving Federated Learning (PPFL) framework for mobile systems to limit privacy leakages in federated learning. Leveraging the widespread presence of Trusted Execution Environments (TEEs) in high-end and mobile devices, we utilize TEEs on clients for local training, and on servers for secure aggregation, so that model/gradient updates are hidden from adversaries. Challenged by the limited memory size of current TEEs, we leverage greedy layer-wise training to train each model's layer inside the trusted area until its convergence. The performance evaluation of our implementation shows that PPFL can significantly improve privacy while incurring small system overheads at the client-side. In particular, PPFL can successfully defend the trained model against data reconstruction, property inference, and membership inference attacks. Furthermore, it can achieve comparable model utility with fewer communication rounds (0.54x) and a similar amount of network traffic (1.002x) compared to the standard federated learning of a complete model. This is achieved while only introducing up to ~15% CPU time, ~18% memory usage, and ~21% energy consumption overhead in PPFL's client-side.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Preprint of our new paper proposing the 1st Privacy-Preserving Federated Learning Framework with TEEs, accepted <a href="https://twitter.com/ACMMobiSys?ref_src=twsrc%5Etfw">@ACMMobiSys</a> 2021, here: <a href="https://t.co/KwfCVRzrIH">https://t.co/KwfCVRzrIH</a><a href="https://twitter.com/VincentMo6?ref_src=twsrc%5Etfw">@VincentMo6</a>,<a href="https://twitter.com/realhamed?ref_src=twsrc%5Etfw">@realhamed</a>,<a href="https://twitter.com/minoskt?ref_src=twsrc%5Etfw">@minoskt</a>,<a href="https://twitter.com/_EduardMarin_?ref_src=twsrc%5Etfw">@_EduardMarin_</a>,<a href="https://twitter.com/Diego_Perino?ref_src=twsrc%5Etfw">@Diego_Perino</a> <br>Powered by <a href="https://twitter.com/TEFresearch?ref_src=twsrc%5Etfw">@TEFresearch</a>,<a href="https://twitter.com/concordiah2020?ref_src=twsrc%5Etfw">@concordiah2020</a>,<a href="https://twitter.com/accordion_h2020?ref_src=twsrc%5Etfw">@accordion_h2020</a> <a href="https://t.co/RSXdfz2L0B">pic.twitter.com/RSXdfz2L0B</a></p>&mdash; Nicolas Kourtellis (@kourtellis) <a href="https://twitter.com/kourtellis/status/1388033706967904258?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. ELSD: Efficient Line Segment Detector and Descriptor

Haotian Zhang, Yicheng Luo, Fangbo Qin, Yijia He, Xiao Liu

- retweets: 72, favorites: 48 (05/01/2021 18:50:51)

- links: [abs](https://arxiv.org/abs/2104.14205) | [pdf](https://arxiv.org/pdf/2104.14205)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present the novel Efficient Line Segment Detector and Descriptor (ELSD) to simultaneously detect line segments and extract their descriptors in an image. Unlike the traditional pipelines that conduct detection and description separately, ELSD utilizes a shared feature extractor for both detection and description, to provide the essential line features to the higher-level tasks like SLAM and image matching in real time. First, we design the one-stage compact model, and propose to use the mid-point, angle and length as the minimal representation of line segment, which also guarantees the center-symmetry. The non-centerness suppression is proposed to filter out the fragmented line segments caused by lines' intersections. The fine offset prediction is designed to refine the mid-point localization. Second, the line descriptor branch is integrated with the detector branch, and the two branches are jointly trained in an end-to-end manner. In the experiments, the proposed ELSD achieves the state-of-the-art performance on the Wireframe dataset and YorkUrban dataset, in both accuracy and efficiency. The line description ability of ELSD also outperforms the previous works on the line matching task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ELSD: Efficient Line Segment Detector and Descriptor<br><br>Haotian Zhang, Yicheng Luo, Fangbo Qin, Yijia He, Xiao Liu<a href="https://t.co/PLtiqEysWp">https://t.co/PLtiqEysWp</a><br><br>tl;dr: another way of detect lines with convnets. New things: center point detection &amp; nms.  + lots of small decision choices. <a href="https://t.co/LBvwISBluL">pic.twitter.com/LBvwISBluL</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1388098840574693376?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. MeerCRAB: MeerLICHT Classification of Real and Bogus Transients using  Deep Learning

Zafiirah Hosenie, Steven Bloemen, Paul Groot, Robert Lyon, Bart Scheers, Benjamin Stappers, Fiorenzo Stoppa, Paul Vreeswijk, Simon De Wet, Marc Klein Wolt, Elmar Körding, Vanessa McBride, Rudolf Le Poole, Kerry Paterson, Daniëlle L. A. Pieterse, Patrick Woudt

- retweets: 67, favorites: 22 (05/01/2021 18:50:51)

- links: [abs](https://arxiv.org/abs/2104.13950) | [pdf](https://arxiv.org/pdf/2104.13950)
- [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [astro-ph.GA](https://arxiv.org/list/astro-ph.GA/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Astronomers require efficient automated detection and classification pipelines when conducting large-scale surveys of the (optical) sky for variable and transient sources. Such pipelines are fundamentally important, as they permit rapid follow-up and analysis of those detections most likely to be of scientific value. We therefore present a deep learning pipeline based on the convolutional neural network architecture called $\texttt{MeerCRAB}$. It is designed to filter out the so called 'bogus' detections from true astrophysical sources in the transient detection pipeline of the MeerLICHT telescope. Optical candidates are described using a variety of 2D images and numerical features extracted from those images. The relationship between the input images and the target classes is unclear, since the ground truth is poorly defined and often the subject of debate. This makes it difficult to determine which source of information should be used to train a classification algorithm. We therefore used two methods for labelling our data (i) thresholding and (ii) latent class model approaches. We deployed variants of $\texttt{MeerCRAB}$ that employed different network architectures trained using different combinations of input images and training set choices, based on classification labels provided by volunteers. The deepest network worked best with an accuracy of 99.5$\%$ and Matthews correlation coefficient (MCC) value of 0.989. The best model was integrated to the MeerLICHT transient vetting pipeline, enabling the accurate and efficient classification of detected transients that allows researchers to select the most promising candidates for their research goals.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our PhD student <a href="https://twitter.com/ZHosenie?ref_src=twsrc%5Etfw">@ZHosenie</a> has had another paper accepted for publication. It presents a <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> pipeline which allows for more efficient automated detection &amp; classification in large-scale sky surveys💫<br><br>Congrats Zafiirah! 👩🏾‍💻 <a href="https://twitter.com/bstappers?ref_src=twsrc%5Etfw">@bstappers</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a><a href="https://t.co/srMZUR2MFN">https://t.co/srMZUR2MFN</a> <a href="https://t.co/Gvk9tHyNQh">pic.twitter.com/Gvk9tHyNQh</a></p>&mdash; DARA Big Data (@DARABigData) <a href="https://twitter.com/DARABigData/status/1388070433224802310?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Optimal training of variational quantum algorithms without barren  plateaus

Tobias Haug, M.S. Kim

- retweets: 12, favorites: 65 (05/01/2021 18:50:52)

- links: [abs](https://arxiv.org/abs/2104.14543) | [pdf](https://arxiv.org/pdf/2104.14543)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Variational quantum algorithms (VQAs) promise efficient use of near-term quantum computers. However, training these algorithms often requires an extensive amount of time and suffers from the barren plateau problem where the magnitude of the gradients vanishes with increasing number of qubits. Here, we show how to optimally train a VQA for learning quantum states. Parameterized quantum circuits can form Gaussian kernels, which we use to derive optimal adaptive learning rates for gradient ascent. We introduce the generalized quantum natural gradient that features stability and optimized movement in parameter space. Both methods together outperform other optimization routines and can enhance VQAs as well as quantum control techniques. The gradients of the VQA do not vanish when the fidelity between the initial state and the state to be learned is bounded from below. We identify a VQA for quantum simulation with such a constraint that can be trained free of barren plateaus. Finally, we propose the application of Gaussian kernels for quantum machine learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">On arXiv today: How to optimally train variational quantum algorithms for learning quantum states.<br>We show the conditions to train free of barren plateaus.<a href="https://t.co/vOFsmf05qq">https://t.co/vOFsmf05qq</a><a href="https://t.co/XPmhNyWqZP">https://t.co/XPmhNyWqZP</a><a href="https://t.co/IQKYguiu9x">https://t.co/IQKYguiu9x</a></p>&mdash; Tobias Haug (@TobiasHaug_Q) <a href="https://twitter.com/TobiasHaug_Q/status/1388079884942073862?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Learned Spatial Representations for Few-shot Talking-Head Synthesis

Moustafa Meshry, Saksham Suri, Larry S. Davis, Abhinav Shrivastava

- retweets: 42, favorites: 26 (05/01/2021 18:50:52)

- links: [abs](https://arxiv.org/abs/2104.14557) | [pdf](https://arxiv.org/pdf/2104.14557)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a novel approach for few-shot talking-head synthesis. While recent works in neural talking heads have produced promising results, they can still produce images that do not preserve the identity of the subject in source images. We posit this is a result of the entangled representation of each subject in a single latent code that models 3D shape information, identity cues, colors, lighting and even background details. In contrast, we propose to factorize the representation of a subject into its spatial and style components. Our method generates a target frame in two steps. First, it predicts a dense spatial layout for the target image. Second, an image generator utilizes the predicted layout for spatial denormalization and synthesizes the target frame. We experimentally show that this disentangled representation leads to a significant improvement over previous methods, both quantitatively and qualitatively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learned Spatial Representations for Few-shot Talking-Head Synthesis<br>pdf: <a href="https://t.co/CDnfILOziy">https://t.co/CDnfILOziy</a><br>abs: <a href="https://t.co/X7FqmyotFp">https://t.co/X7FqmyotFp</a><br>project page: <a href="https://t.co/iZpuwfIVgw">https://t.co/iZpuwfIVgw</a> <a href="https://t.co/MssL7Ey3bV">pic.twitter.com/MssL7Ey3bV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387949295836479490?ref_src=twsrc%5Etfw">April 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



