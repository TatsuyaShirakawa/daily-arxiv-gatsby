---
title: Hot Papers 2021-02-16
date: 2021-02-17T09:21:49.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-16"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-16"
socialImage: "/media/flying-marine.jpg"

---

# 1. TransGAN: Two Transformers Can Make One Strong GAN

Yifan Jiang, Shiyu Chang, Zhangyang Wang

- retweets: 9669, favorites: 7 (02/17/2021 09:21:49)

- links: [abs](https://arxiv.org/abs/2102.07074) | [pdf](https://arxiv.org/pdf/2102.07074)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The recent explosive interest on transformers has suggested their potential to become powerful "universal" models for computer vision tasks, such as classification, detection, and segmentation. However, how further transformers can go - are they ready to take some more notoriously difficult vision tasks, e.g., generative adversarial networks (GANs)? Driven by that curiosity, we conduct the first pilot study in building a GAN \textbf{completely free of convolutions}, using only pure transformer-based architectures. Our vanilla GAN architecture, dubbed \textbf{TransGAN}, consists of a memory-friendly transformer-based generator that progressively increases feature resolution while decreasing embedding dimension, and a patch-level discriminator that is also transformer-based. We then demonstrate TransGAN to notably benefit from data augmentations (more than standard GANs), a multi-task co-training strategy for the generator, and a locally initialized self-attention that emphasizes the neighborhood smoothness of natural images. Equipped with those findings, TransGAN can effectively scale up with bigger models and high-resolution image datasets. Specifically, our best architecture achieves highly competitive performance compared to current state-of-the-art GANs based on convolutional backbones. Specifically, TransGAN sets \textbf{new state-of-the-art} IS score of 10.10 and FID score of 25.32 on STL-10. It also reaches competitive 8.64 IS score and 11.89 FID score on Cifar-10, and 12.23 FID score on CelebA $64\times64$, respectively. We also conclude with a discussion of the current limitations and future potential of TransGAN. The code is available at \url{https://github.com/VITA-Group/TransGAN}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TransGAN: Two Transformers Can Make One Strong GAN<br>pdf: <a href="https://t.co/EUTkjaEvF4">https://t.co/EUTkjaEvF4</a><br>abs: <a href="https://t.co/vRTwLMNDPm">https://t.co/vRTwLMNDPm</a><br>github: <a href="https://t.co/jSt9t1tYUN">https://t.co/jSt9t1tYUN</a> <a href="https://t.co/DC6wZVOikq">pic.twitter.com/DC6wZVOikq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361500966198124545?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. DOBF: A Deobfuscation Pre-Training Objective for Programming Languages

Baptiste Roziere, Marie-Anne Lachaux, Marc Szafraniec, Guillaume Lample

- retweets: 8664, favorites: 435 (02/17/2021 09:21:49)

- links: [abs](https://arxiv.org/abs/2102.07492) | [pdf](https://arxiv.org/pdf/2102.07492)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Recent advances in self-supervised learning have dramatically improved the state of the art on a wide variety of tasks. However, research in language model pre-training has mostly focused on natural languages, and it is unclear whether models like BERT and its variants provide the best pre-training when applied to other modalities, such as source code. In this paper, we introduce a new pre-training objective, DOBF, that leverages the structural aspect of programming languages and pre-trains a model to recover the original version of obfuscated source code. We show that models pre-trained with DOBF significantly outperform existing approaches on multiple downstream tasks, providing relative improvements of up to 13% in unsupervised code translation, and 24% in natural language code search. Incidentally, we found that our pre-trained model is able to de-obfuscate fully obfuscated source files, and to suggest descriptive variable names.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper on code de-obfuscation: <a href="https://t.co/vs3Y2E2Hfn">https://t.co/vs3Y2E2Hfn</a><br>We show that if you obfuscate the name of identifiers in source code, a model can retrieve the original names with very high accuracy. It even works when you remove the name of each variable / function! 1/3 <a href="https://t.co/2ISslynQn6">pic.twitter.com/2ISslynQn6</a></p>&mdash; Guillaume Lample (@GuillaumeLample) <a href="https://twitter.com/GuillaumeLample/status/1361663915072118784?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Intermediate Layer Optimization for Inverse Problems using Deep  Generative Models

Giannis Daras, Joseph Dean, Ajil Jalal, Alexandros G. Dimakis

- retweets: 3510, favorites: 428 (02/17/2021 09:21:50)

- links: [abs](https://arxiv.org/abs/2102.07364) | [pdf](https://arxiv.org/pdf/2102.07364)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose Intermediate Layer Optimization (ILO), a novel optimization algorithm for solving inverse problems with deep generative models. Instead of optimizing only over the initial latent code, we progressively change the input layer obtaining successively more expressive generators. To explore the higher dimensional spaces, our method searches for latent codes that lie within a small $l_1$ ball around the manifold induced by the previous layer. Our theoretical analysis shows that by keeping the radius of the ball relatively small, we can improve the established error bound for compressed sensing with deep generative models. We empirically show that our approach outperforms state-of-the-art methods introduced in StyleGAN-2 and PULSE for a wide range of inverse problems including inpainting, denoising, super-resolution and compressed sensing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Intermediate Layer Optimization for Inverse Problems using Deep Generative Models<br>pdf: <a href="https://t.co/kzM10WHfnq">https://t.co/kzM10WHfnq</a><br>abs: <a href="https://t.co/rj8xvuYbNM">https://t.co/rj8xvuYbNM</a><br>github: <a href="https://t.co/eQiaBZYX2g">https://t.co/eQiaBZYX2g</a> <a href="https://t.co/b1WiG1TCLc">pic.twitter.com/b1WiG1TCLc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361505296561143820?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper: &quot;Intermediate Layer Optimization for Inverse Problems using Deep Generative Models&quot;.<br><br>Paper: <a href="https://t.co/kjfTfwvveh">https://t.co/kjfTfwvveh</a><br><br>Code: <a href="https://t.co/b4bMkve7ps">https://t.co/b4bMkve7ps</a><br><br>Colab: <a href="https://t.co/euiCEXXXXi">https://t.co/euiCEXXXXi</a><br><br>Below a video of the Mona Lisa with inpainted eyes and a thread🧵 <a href="https://t.co/oe908LEZZJ">pic.twitter.com/oe908LEZZJ</a></p>&mdash; Giannis Daras (@giannis_daras) <a href="https://twitter.com/giannis_daras/status/1361494657746804737?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. NeRF$--$: Neural Radiance Fields Without Known Camera Parameters

Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, Victor Adrian Prisacariu

- retweets: 2580, favorites: 355 (02/17/2021 09:21:50)

- links: [abs](https://arxiv.org/abs/2102.07064) | [pdf](https://arxiv.org/pdf/2102.07064)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper tackles the problem of novel view synthesis (NVS) from 2D images without known camera poses and intrinsics. Among various NVS techniques, Neural Radiance Field (NeRF) has recently gained popularity due to its remarkable synthesis quality. Existing NeRF-based approaches assume that the camera parameters associated with each input image are either directly accessible at training, or can be accurately estimated with conventional techniques based on correspondences, such as Structure-from-Motion. In this work, we propose an end-to-end framework, termed NeRF--, for training NeRF models given only RGB images, without pre-computed camera parameters. Specifically, we show that the camera parameters, including both intrinsics and extrinsics, can be automatically discovered via joint optimisation during the training of the NeRF model. On the standard LLFF benchmark, our model achieves comparable novel view synthesis results compared to the baseline trained with COLMAP pre-computed camera parameters. We also conduct extensive analyses to understand the model behaviour under different camera trajectories, and show that in scenarios where COLMAP fails, our model still produces robust results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeRF−−: Neural Radiance Fields Without Known Camera Parameters<br>pdf: <a href="https://t.co/99g3MJpjM7">https://t.co/99g3MJpjM7</a><br>abs: <a href="https://t.co/VURgtj3wdt">https://t.co/VURgtj3wdt</a><br>project page: <a href="https://t.co/KWr7cRQrzp">https://t.co/KWr7cRQrzp</a> <a href="https://t.co/smW9u7GyBW">pic.twitter.com/smW9u7GyBW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361511297775505411?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out NeRF--: NeRF without camera parameters!<br>We show that camera poses, intrinsics and NeRF can be jointly optimised from scratch given only RGB images.<br><br>🚩 <a href="https://t.co/WHMPcN2e5g">https://t.co/WHMPcN2e5g</a><br>📄 <a href="https://t.co/J6R3pb8obm">https://t.co/J6R3pb8obm</a><br><br>w/ <a href="https://twitter.com/elliottszwu?ref_src=twsrc%5Etfw">@elliottszwu</a> <a href="https://twitter.com/WeidiXie?ref_src=twsrc%5Etfw">@WeidiXie</a> Min Chen <a href="https://twitter.com/viprad?ref_src=twsrc%5Etfw">@viprad</a> <a href="https://twitter.com/AVLOxford?ref_src=twsrc%5Etfw">@AVLOxford</a> <a href="https://twitter.com/Oxford_VGG?ref_src=twsrc%5Etfw">@Oxford_VGG</a> <a href="https://t.co/Lq6lp5TN93">pic.twitter.com/Lq6lp5TN93</a></p>&mdash; Zirui Wang (@ziruiwang_ox) <a href="https://twitter.com/ziruiwang_ox/status/1361684012209733635?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them

Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Küttler, Aleksandra Piktus, Pontus Stenetorp, Sebastian Riedel

- retweets: 1549, favorites: 172 (02/17/2021 09:21:50)

- links: [abs](https://arxiv.org/abs/2102.07033) | [pdf](https://arxiv.org/pdf/2102.07033)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Open-domain Question Answering models which directly leverage question-answer (QA) pairs, such as closed-book QA (CBQA) models and QA-pair retrievers, show promise in terms of speed and memory compared to conventional models which retrieve and read from text corpora. QA-pair retrievers also offer interpretable answers, a high degree of control, and are trivial to update at test time with new knowledge. However, these models lack the accuracy of retrieve-and-read systems, as substantially less knowledge is covered by the available QA-pairs relative to text corpora like Wikipedia. To facilitate improved QA-pair models, we introduce Probably Asked Questions (PAQ), a very large resource of 65M automatically-generated QA-pairs. We introduce a new QA-pair retriever, RePAQ, to complement PAQ. We find that PAQ preempts and caches test questions, enabling RePAQ to match the accuracy of recent retrieve-and-read models, whilst being significantly faster. Using PAQ, we train CBQA models which outperform comparable baselines by 5%, but trail RePAQ by over 15%, indicating the effectiveness of explicit retrieval. RePAQ can be configured for size (under 500MB) or speed (over 1K questions per second) whilst retaining high accuracy. Lastly, we demonstrate RePAQ's strength at selective QA, abstaining from answering when it is likely to be incorrect. This enables RePAQ to ``back-off" to a more expensive state-of-the-art model, leading to a combined system which is both more accurate and 2x faster than the state-of-the-art model alone.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚨 New work 🚨 “PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them”. <br>Read the paper here: <a href="https://t.co/MA8BTtWBi6">https://t.co/MA8BTtWBi6</a>, and check out the thread below <br>w/ <a href="https://twitter.com/mindjimmy?ref_src=twsrc%5Etfw">@mindjimmy</a>,<a href="https://twitter.com/likicode?ref_src=twsrc%5Etfw">@likicode</a>, <a href="https://twitter.com/PMinervini?ref_src=twsrc%5Etfw">@PMinervini</a>, <a href="https://twitter.com/HeinrichKuttler?ref_src=twsrc%5Etfw">@HeinrichKuttler</a>,<a href="https://twitter.com/olapiktus?ref_src=twsrc%5Etfw">@olapiktus</a>, Pontus Stenetorp, <a href="https://twitter.com/riedelcastro?ref_src=twsrc%5Etfw">@riedelcastro</a>. 1/N <a href="https://t.co/MdY3Iv54WZ">pic.twitter.com/MdY3Iv54WZ</a></p>&mdash; Patrick Lewis (@PSH_Lewis) <a href="https://twitter.com/PSH_Lewis/status/1361682746641436677?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Understanding self-supervised Learning Dynamics without Contrastive  Pairs

Yuandong Tian, Xinlei Chen, Surya Ganguli

- retweets: 1286, favorites: 303 (02/17/2021 09:21:51)

- links: [abs](https://arxiv.org/abs/2102.06810) | [pdf](https://arxiv.org/pdf/2102.06810)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Contrastive approaches to self-supervised learning (SSL) learn representations by minimizing the distance between two augmented views of the same data point (positive pairs) and maximizing the same from different data points (negative pairs). However, recent approaches like BYOL and SimSiam, show remarkable performance {\it without} negative pairs, raising a fundamental theoretical question: how can SSL with only positive pairs avoid representational collapse? We study the nonlinear learning dynamics of non-contrastive SSL in simple linear networks. Our analysis yields conceptual insights into how non-contrastive SSL methods learn, how they avoid representational collapse, and how multiple factors, like predictor networks, stop-gradients, exponential moving averages, and weight decay all come into play. Our simple theory recapitulates the results of real-world ablation studies in both STL-10 and ImageNet. Furthermore, motivated by our theory we propose a novel approach that \emph{directly} sets the predictor based on the statistics of its inputs. In the case of linear predictors, our approach outperforms gradient training of the predictor by $5\%$ and on ImageNet it performs comparably with more complex two-layer non-linear predictors that employ BatchNorm. Code is released in https://github.com/facebookresearch/luckmatters/tree/master/ssl.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/ New preprint: &quot;Understanding self-supervised learning without contrastive pairs&quot; w/ awesome collaborators <a href="https://twitter.com/tydsh?ref_src=twsrc%5Etfw">@tydsh</a> and <a href="https://twitter.com/endernewton?ref_src=twsrc%5Etfw">@endernewton</a> <a href="https://twitter.com/facebookai?ref_src=twsrc%5Etfw">@facebookai</a> <a href="https://t.co/pcAiIdy6k1">https://t.co/pcAiIdy6k1</a> We develop a theory for how self-supervised learning without negative pairs (i.e. BYOL/SimSiam) can work... <a href="https://t.co/EgUQNw41k9">pic.twitter.com/EgUQNw41k9</a></p>&mdash; Surya Ganguli (@SuryaGanguli) <a href="https://twitter.com/SuryaGanguli/status/1361737378650492928?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Understanding self-supervised Learning Dynamics without Contrastive Pairs<br>pdf: <a href="https://t.co/TCnhb48Cvk">https://t.co/TCnhb48Cvk</a><br>abs: <a href="https://t.co/PNwme8BAbq">https://t.co/PNwme8BAbq</a> <a href="https://t.co/abeO2Qo2Aw">pic.twitter.com/abeO2Qo2Aw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361529404426489857?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our work <a href="https://t.co/NeasPSMKcb">https://t.co/NeasPSMKcb</a> analyzes SSL with linear predictor w/o neg pairs. It analytically explains how w decay, predictor LR and EMA affect training dynamics, supported by extensive experiments. Following the analysis, direct setting the predictor does well (or better)!</p>&mdash; Yuandong Tian (@tydsh) <a href="https://twitter.com/tydsh/status/1361717556634034178?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. End-to-End Egospheric Spatial Memory

Daniel Lenton, Stephen James, Ronald Clark, Andrew J. Davison

- retweets: 182, favorites: 66 (02/17/2021 09:21:51)

- links: [abs](https://arxiv.org/abs/2102.07764) | [pdf](https://arxiv.org/pdf/2102.07764)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Spatial memory, or the ability to remember and recall specific locations and objects, is central to autonomous agents' ability to carry out tasks in real environments. However, most existing artificial memory modules have difficulty recalling information over long time periods and are not very adept at storing spatial information. We propose a parameter-free module, Egospheric Spatial Memory (ESM), which encodes the memory in an ego-sphere around the agent, enabling expressive 3D representations. ESM can be trained end-to-end via either imitation or reinforcement learning, and improves both training efficiency and final performance against other memory baselines on both drone and manipulator visuomotor control tasks. The explicit egocentric geometry also enables us to seamlessly combine the learned controller with other non-learned modalities, such as local obstacle avoidance. We further show applications to semantic segmentation on the ScanNet dataset, where ESM naturally combines image-level and map-level inference modalities. Through our broad set of experiments, we show that ESM provides a general computation graph for embodied spatial reasoning, and the module forms a bridge between real-time mapping systems and differentiable memory architectures.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">End-to-End Egospheric Spatial Memory. <a href="https://t.co/HR2SR3tOBa">https://t.co/HR2SR3tOBa</a> <a href="https://twitter.com/hashtag/robotics?src=hash&amp;ref_src=twsrc%5Etfw">#robotics</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://twitter.com/hashtag/ICLR2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICLR2021</a> <br><br>The paper proposes a parameter-free module, Egospheric Spatial Memory (ESM), which encodes the memory in an ego-sphere around the agent, enabling expressive 3D representations. <a href="https://t.co/GO63BrW2wO">pic.twitter.com/GO63BrW2wO</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1361503524161478659?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Understanding Negative Samples in Instance Discriminative  Self-supervised Representation Learning

Kento Nozawa, Issei Sato

- retweets: 122, favorites: 71 (02/17/2021 09:21:51)

- links: [abs](https://arxiv.org/abs/2102.06866) | [pdf](https://arxiv.org/pdf/2102.06866)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Instance discriminative self-supervised representation learning has been attracted attention thanks to its unsupervised nature and informative feature representation for downstream tasks. Self-supervised representation learning commonly uses more negative samples than the number of supervised classes in practice. However, there is an inconsistency in the existing analysis; theoretically, a large number of negative samples degrade supervised performance, while empirically, they improve the performance. We theoretically explain this empirical result regarding negative samples. We empirically confirm our analysis by conducting numerical experiments on CIFAR-10/100 datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new preprint is out <a href="https://t.co/pm8H2IOo4k">https://t.co/pm8H2IOo4k</a><br>w/ <a href="https://twitter.com/issei_sato?ref_src=twsrc%5Etfw">@issei_sato</a> | Understanding Negative Samples in Instance Discriminative Self-supervised Representation Learning</p>&mdash; Kento Nozawa (@nzw0301) <a href="https://twitter.com/nzw0301/status/1361506508488413188?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">はい<a href="https://t.co/pm8H2IOo4k">https://t.co/pm8H2IOo4k</a></p>&mdash; Kento Nozawa (@nzw0301) <a href="https://twitter.com/nzw0301/status/1361496050352091137?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Learning Intra-Batch Connections for Deep Metric Learning

Jenny Seidenschwarz, Ismail Elezi, Laura Leal-Taixé

- retweets: 72, favorites: 78 (02/17/2021 09:21:52)

- links: [abs](https://arxiv.org/abs/2102.07753) | [pdf](https://arxiv.org/pdf/2102.07753)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The goal of metric learning is to learn a function that maps samples to a lower-dimensional space where similar samples lie closer than dissimilar ones. In the case of deep metric learning, the mapping is performed by training a neural network. Most approaches rely on losses that only take the relations between pairs or triplets of samples into account, which either belong to the same class or to two different classes. However, these approaches do not explore the embedding space in its entirety. To this end, we propose an approach based on message passing networks that takes into account all the relations in a mini-batch. We refine embedding vectors by exchanging messages among all samples in a given batch allowing the training process to be aware of the overall structure. Since not all samples are equally important to predict a decision boundary, we use dot-product self-attention during message passing to allow samples to weight the importance of each neighbor accordingly. We achieve state-of-the-art results on clustering and image retrieval on the CUB-200-2011, Cars196, Stanford Online Products, and In-Shop Clothes datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Tired of looking for the best pairs or triplets to train your net for deep metric learning? Look no further! <a href="https://t.co/NEGHcgvx3U">https://t.co/NEGHcgvx3U</a> How? You will find transformers somewhere in there! <a href="https://twitter.com/JennySeidensch1?ref_src=twsrc%5Etfw">@JennySeidensch1</a> <a href="https://twitter.com/Ismail_Elezi?ref_src=twsrc%5Etfw">@Ismail_Elezi</a></p>&mdash; Laura Leal-Taixe (@lealtaixe) <a href="https://twitter.com/lealtaixe/status/1361716308811911175?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Beyond QUIC v1 -- A First Look at Recent Transport Layer IETF  Standardization Efforts

Mike Kosek, Tanya Shreedhar, Vaibhav Bajpai

- retweets: 121, favorites: 25 (02/17/2021 09:21:52)

- links: [abs](https://arxiv.org/abs/2102.07527) | [pdf](https://arxiv.org/pdf/2102.07527)
- [cs.NI](https://arxiv.org/list/cs.NI/recent)

The transport layer is ossified. With most of the research and deployment efforts in the past decade focussing on the Transmission Control Protocol (TCP) and its extensions, the QUIC standardization by the Internet Engineering Task Force (IETF) is to be finalized in early 2021. In addition to addressing the most urgent issues of TCP, QUIC ensures its future extendibility and is destined to drastically change the transport protocol landscape. In this work, we present a first look at emerging protocols and their IETF standardization efforts beyond QUIC v1. While multiple proposed extensions improve on QUIC itself, Multiplexed Application Substrate over QUIC Encryption (MASQUE) as well as WebTransport present different approaches to address long-standing problems, and their interplay extends on QUIC's take to address transport layer ossification challenges.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">With QUIC v1 in the <a href="https://twitter.com/ietf?ref_src=twsrc%5Etfw">@ietf</a> RFC editor queue, what are the next big things beyond QUIC v1? See our paper at <a href="https://t.co/6TuRFP2Pza">https://t.co/6TuRFP2Pza</a> <a href="https://t.co/cRUdeBHTAf">pic.twitter.com/cRUdeBHTAf</a></p>&mdash; Mike Kosek (@MikeKosek) <a href="https://twitter.com/MikeKosek/status/1361665227507593219?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Fluctuation-response theorem for Kullback-Leibler divergences to  quantify causation

Andrea Auconi, Benjamin M. Friedrich, Andrea Giansanti

- retweets: 100, favorites: 23 (02/17/2021 09:21:52)

- links: [abs](https://arxiv.org/abs/2102.06839) | [pdf](https://arxiv.org/pdf/2102.06839)
- [cs.IT](https://arxiv.org/list/cs.IT/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent)

We define a new measure of causation from a fluctuation-response theorem for Kullback-Leibler divergences, based on the information-theoretic cost of perturbations. This information response has both the invariance properties required for an information-theoretic measure and the physical interpretation of a propagation of perturbations. In linear systems, the information response reduces to the transfer entropy, providing a connection between Fisher and mutual information.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How to quantify &#39;causation&#39;, e.g., how X influences Y in complex systems? New preprint that combines the two fundamental perspectives on this old problem: &#39;flow of information&#39; versus &#39;fluctuation-response theory&#39; by <a href="https://twitter.com/Auconi?ref_src=twsrc%5Etfw">@Auconi</a> at <a href="https://twitter.com/cfaed_TUD?ref_src=twsrc%5Etfw">@cfaed_TUD</a> &amp; <a href="https://twitter.com/PoLDresden?ref_src=twsrc%5Etfw">@PoLDresden</a>  <a href="https://t.co/Phzux313Ry">https://t.co/Phzux313Ry</a></p>&mdash; Benjamin Friedrich (@friedrich_group) <a href="https://twitter.com/friedrich_group/status/1361699058058952706?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Prompt Programming for Large Language Models: Beyond the Few-Shot  Paradigm

Laria Reynolds, Kyle McDonell

- retweets: 54, favorites: 50 (02/17/2021 09:21:52)

- links: [abs](https://arxiv.org/abs/2102.07350) | [pdf](https://arxiv.org/pdf/2102.07350)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Prevailing methods for mapping large generative language models to supervised tasks may fail to sufficiently probe models' novel capabilities. Using GPT-3 as a case study, we show that 0-shot prompts can significantly outperform few-shot prompts. We suggest that the function of few-shot examples in these cases is better described as locating an already learned task rather than meta-learning. This analysis motivates rethinking the role of prompts in controlling and evaluating powerful language models. In this work, we discuss methods of prompt programming, emphasizing the usefulness of considering prompts through the lens of natural language. We explore techniques for exploiting the capacity of narratives and cultural anchors to encode nuanced intentions and techniques for encouraging deconstruction of a problem into components before producing a verdict. Informed by this more encompassing theory of prompt programming, we also introduce the idea of a metaprompt that seeds the model to generate its own natural language prompts for a range of tasks. Finally, we discuss how these more general methods of interacting with language models can be incorporated into existing and future benchmarks and practical applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Great to see that work is going into investigating better zero-shot prompts for GPT-3—zero-shot prompting for locating tasks in task-space does seem to be the future, with few-shot prompting merely as a brief transition to that future.<a href="https://t.co/fLJZnm2G5B">https://t.co/fLJZnm2G5B</a></p>&mdash; Leo Gao (@nabla_theta) <a href="https://twitter.com/nabla_theta/status/1361536299472015361?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Does Standard Backpropagation Forget Less Catastrophically Than Adam?

Dylan R. Ashley, Sina Ghiassian, Richard S. Sutton

- retweets: 49, favorites: 46 (02/17/2021 09:21:52)

- links: [abs](https://arxiv.org/abs/2102.07686) | [pdf](https://arxiv.org/pdf/2102.07686)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Catastrophic forgetting remains a severe hindrance to the broad application of artificial neural networks (ANNs), however, it continues to be a poorly understood phenomenon. Despite the extensive amount of work on catastrophic forgetting, we argue that it is still unclear how exactly the phenomenon should be quantified, and, moreover, to what degree all of the choices we make when designing learning systems affect the amount of catastrophic forgetting. We use various testbeds from the reinforcement learning and supervised learning literature to (1) provide evidence that the choice of which modern gradient-based optimization algorithm is used to train an ANN has a significant impact on the amount of catastrophic forgetting and show that--surprisingly--in many instances classical algorithms such as vanilla SGD experience less catastrophic forgetting than the more modern algorithms such as Adam. We empirically compare four different existing metrics for quantifying catastrophic forgetting and (2) show that the degree to which the learning systems experience catastrophic forgetting is sufficiently sensitive to the metric used that a change from one principled metric to another is enough to change the conclusions of a study dramatically. Our results suggest that a much more rigorous experimental methodology is required when looking at catastrophic forgetting. Based on our results, we recommend inter-task forgetting in supervised learning must be measured with both retention and relearning metrics concurrently, and intra-task forgetting in reinforcement learning must--at the very least--be measured with pairwise interference.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Does Standard Backpropagation Forget Less Catastrophically Than Adam?<br>pdf: <a href="https://t.co/ZuwRBbI5w9">https://t.co/ZuwRBbI5w9</a><br>abs: <a href="https://t.co/iOPE2CBfu8">https://t.co/iOPE2CBfu8</a> <a href="https://t.co/dK3Rp9HtKe">pic.twitter.com/dK3Rp9HtKe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361558913993310214?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Graph Convolution for Semi-Supervised Classification: Improved Linear  Separability and Out-of-Distribution Generalization

Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath

- retweets: 49, favorites: 42 (02/17/2021 09:21:52)

- links: [abs](https://arxiv.org/abs/2102.06966) | [pdf](https://arxiv.org/pdf/2102.06966)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Recently there has been increased interest in semi-supervised classification in the presence of graphical information. A new class of learning models has emerged that relies, at its most basic level, on classifying the data after first applying a graph convolution. To understand the merits of this approach, we study the classification of a mixture of Gaussians, where the data corresponds to the node attributes of a stochastic block model. We show that graph convolution extends the regime in which the data is linearly separable by a factor of roughly $1/\sqrt{D}$, where $D$ is the expected degree of a node, as compared to the mixture model data on its own. Furthermore, we find that the linear classifier obtained by minimizing the cross-entropy loss after the graph convolution generalizes to out-of-distribution data where the unseen data can have different intra- and inter-class edge probabilities from the training data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">“Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization”. <a href="https://t.co/C2axUVwYfp">https://t.co/C2axUVwYfp</a> <a href="https://t.co/XOIi8JcuaN">pic.twitter.com/XOIi8JcuaN</a></p>&mdash; Kimon Fountoulakis (@kfountou) <a href="https://twitter.com/kfountou/status/1361552009531428864?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Good-case Latency of Byzantine Broadcast: a Complete Categorization

Ittai Abraham, Kartik Nayak, Ling Ren, Zhuolun Xiang

- retweets: 72, favorites: 18 (02/17/2021 09:21:52)

- links: [abs](https://arxiv.org/abs/2102.07240) | [pdf](https://arxiv.org/pdf/2102.07240)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent)

This paper explores the problem good-case latency of Byzantine fault-tolerant broadcast, motivated by the real-world latency and performance of practical state machine replication protocols. The good-case latency measures the time it takes for all non-faulty parties to commit when the designated broadcaster is non-faulty. We provide a complete characterization of tight bounds on good-case latency, in the authenticated setting under both synchrony and asynchrony. Some of our new results may be surprising, e.g, 2-round PBFT-style asynchronous reliable broadcast is possible if and only if $n\geq 5f-1$, and a tight bound for good-case latency under $n/3<f<n/2$ under synchrony is not an integer multiple of the delay bound.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work from lead author Zhuolun Xiang with <a href="https://twitter.com/kartik1507?ref_src=twsrc%5Etfw">@kartik1507</a> and Ling Ren:<br><br>Good-case Latency of Byzantine Broadcast: a Complete Categorization <a href="https://t.co/IH0e3dJl9l">https://t.co/IH0e3dJl9l</a></p>&mdash; Ittai Abraham (@ittaia) <a href="https://twitter.com/ittaia/status/1361625517963833344?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Learning Speech-driven 3D Conversational Gestures from Video

Ikhsanul Habibie, Weipeng Xu, Dushyant Mehta, Lingjie Liu, Hans-Peter Seidel, Gerard Pons-Moll, Mohamed Elgharib, Christian Theobalt

- retweets: 58, favorites: 29 (02/17/2021 09:21:53)

- links: [abs](https://arxiv.org/abs/2102.06837) | [pdf](https://arxiv.org/pdf/2102.06837)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose the first approach to automatically and jointly synthesize both the synchronous 3D conversational body and hand gestures, as well as 3D face and head animations, of a virtual character from speech input. Our algorithm uses a CNN architecture that leverages the inherent correlation between facial expression and hand gestures. Synthesis of conversational body gestures is a multi-modal problem since many similar gestures can plausibly accompany the same input speech. To synthesize plausible body gestures in this setting, we train a Generative Adversarial Network (GAN) based model that measures the plausibility of the generated sequences of 3D body motion when paired with the input audio features. We also contribute a new way to create a large corpus of more than 33 hours of annotated body, hand, and face data from in-the-wild videos of talking people. To this end, we apply state-of-the-art monocular approaches for 3D body and hand pose estimation as well as dense 3D face performance capture to the video corpus. In this way, we can train on orders of magnitude more data than previous algorithms that resort to complex in-studio motion capture solutions, and thereby train more expressive synthesis algorithms. Our experiments and user study show the state-of-the-art quality of our speech-synthesized full 3D character animations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Speech-driven 3D Conversational Gestures from Video<br>pdf: <a href="https://t.co/iuVUXwCUUr">https://t.co/iuVUXwCUUr</a><br>abs: <a href="https://t.co/VyIqmOcruA">https://t.co/VyIqmOcruA</a> <a href="https://t.co/RA1dfi2EiN">pic.twitter.com/RA1dfi2EiN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361509043722805250?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Learning low-rank latent mesoscale structures in networks

Hanbaek Lyu, Yacoub H. Kureh, Joshua Vendrow, Mason A. Porter

- retweets: 42, favorites: 45 (02/17/2021 09:21:53)

- links: [abs](https://arxiv.org/abs/2102.06984) | [pdf](https://arxiv.org/pdf/2102.06984)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.OC](https://arxiv.org/list/math.OC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

It is common to use networks to encode the architecture of interactions between entities in complex systems in the physical, biological, social, and information sciences. Moreover, to study the large-scale behavior of complex systems, it is important to study mesoscale structures in networks as building blocks that influence such behavior. In this paper, we present a new approach for describing low-rank mesoscale structure in networks, and we illustrate our approach using several synthetic network models and empirical friendship, collaboration, and protein--protein interaction (PPI) networks. We find that these networks possess a relatively small number of `latent motifs' that together can successfully approximate most subnetworks at a fixed mesoscale. We use an algorithm that we call "network dictionary learning" (NDL), which combines a network sampling method and nonnegative matrix factorization, to learn the latent motifs of a given network. The ability to encode a network using a set of latent motifs has a wide range of applications to network-analysis tasks, such as comparison, denoising, and edge inference. Additionally, using our new network denoising and reconstruction (NDR) algorithm, we demonstrate how to denoise a corrupted network by using only the latent motifs that one learns directly from the corrupted networks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Here is one of my new papers out on arXiv tonight: &quot;Learning Low-Rank Latent Mesoscale Structures in Networks&quot;: <a href="https://t.co/KsUfvQGu42">https://t.co/KsUfvQGu42</a><br><br>by Hanbaek Lyu, Yacoub H. Kureh, Joshua Vendrow, and MAP<br><br>This paper is about &quot;Network Dictionary Learning&quot; <a href="https://t.co/B1PXZF3k4r">pic.twitter.com/B1PXZF3k4r</a></p>&mdash; Mason Porter (@masonporter) <a href="https://twitter.com/masonporter/status/1361494696271302660?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Geometric feature performance under downsampling for EEG classification  tasks

Bryan Bischof, Eric Bunch

- retweets: 42, favorites: 32 (02/17/2021 09:21:53)

- links: [abs](https://arxiv.org/abs/2102.07669) | [pdf](https://arxiv.org/pdf/2102.07669)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.AT](https://arxiv.org/list/math.AT/recent)

We experimentally investigate a collection of feature engineering pipelines for use with a CNN for classifying eyes-open or eyes-closed from electroencephalogram (EEG) time-series from the Bonn dataset. Using the Takens' embedding--a geometric representation of time-series--we construct simplicial complexes from EEG data. We then compare $\epsilon$-series of Betti-numbers and $\epsilon$-series of graph spectra (a novel construction)--two topological invariants of the latent geometry from these complexes--to raw time series of the EEG to fill in a gap in the literature for benchmarking. These methods, inspired by Topological Data Analysis, are used for feature engineering to capture local geometry of the time-series. Additionally, we test these feature pipelines' robustness to downsampling and data reduction. This paper seeks to establish clearer expectations for both time-series classification via geometric features, and how CNNs for time-series respond to data of degraded resolution.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A couple years ago, <a href="https://twitter.com/ericbunch0?ref_src=twsrc%5Etfw">@ericbunch0</a> &amp; I found a paper discussing the combination of Topological Data Analysis, time-series, and CNNs; we thought it would be fun to read and implement, and maybe get inspired...<br><br>After a long journey here&#39;s our paper: <a href="https://t.co/siqu78aItF">https://t.co/siqu78aItF</a><br>🧵 (1/7) <a href="https://t.co/yTI1PVYJh0">pic.twitter.com/yTI1PVYJh0</a></p>&mdash; Dr. Donut ☕️ (@BEBischof) <a href="https://twitter.com/BEBischof/status/1361707097939865600?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. Private learning implies quantum stability

Srinivasan Arunachalam, Yihui Quek, John Smolin

- retweets: 30, favorites: 32 (02/17/2021 09:21:53)

- links: [abs](https://arxiv.org/abs/2102.07171) | [pdf](https://arxiv.org/pdf/2102.07171)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Learning an unknown $n$-qubit quantum state $\rho$ is a fundamental challenge in quantum computing. Information-theoretically, it is known that tomography requires exponential in $n$ many copies of $\rho$ to estimate it up to trace distance. Motivated by computational learning theory, Aaronson et al. introduced many (weaker) learning models: the PAC model of learning states (Proceedings of Royal Society A'07), shadow tomography (STOC'18) for learning "shadows" of a state, a model that also requires learners to be differentially private (STOC'19) and the online model of learning states (NeurIPS'18). In these models it was shown that an unknown state can be learned "approximately" using linear-in-$n$ many copies of rho. But is there any relationship between these models? In this paper we prove a sequence of (information-theoretic) implications from differentially-private PAC learning, to communication complexity, to online learning and then to quantum stability.   Our main result generalizes the recent work of Bun, Livni and Moran (Journal of the ACM'21) who showed that finite Littlestone dimension (of Boolean-valued concept classes) implies PAC learnability in the (approximate) differentially private (DP) setting. We first consider their work in the real-valued setting and further extend their techniques to the setting of learning quantum states. Key to our results is our generic quantum online learner, Robust Standard Optimal Algorithm (RSOA), which is robust to adversarial imprecision. We then show information-theoretic implications between DP learning quantum states in the PAC model, learnability of quantum states in the one-way communication model, online learning of quantum states, quantum stability (which is our conceptual contribution), various combinatorial parameters and give further applications to gentle shadow tomography and noisy quantum state learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;I&#39;m not a quantum person, but...&quot; this paper by Arunachalam, Quek (<a href="https://twitter.com/quekpottheories?ref_src=twsrc%5Etfw">@quekpottheories</a>), and Smolin seems potentially really cool: &quot;Private learning implies quantum stability&quot; <a href="https://t.co/33D7SNPK8r">https://t.co/33D7SNPK8r</a><br><br>(h/t <a href="https://twitter.com/thegautamkamath?ref_src=twsrc%5Etfw">@thegautamkamath</a> for pointing it out to me) <a href="https://t.co/IduFLfqqTj">pic.twitter.com/IduFLfqqTj</a></p>&mdash; Clément Canonne (@ccanonne_) <a href="https://twitter.com/ccanonne_/status/1361518089674706950?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. Machine Learning for Mechanical Ventilation Control

Daniel Suo, Udaya Ghai, Edgar Minasyan, Paula Gradu, Xinyi Chen, Naman Agarwal, Cyril Zhang, Karan Singh, Julienne LaChance, Tom Zadjel, Manuel Schottdorf, Daniel Cohen, Elad Hazan

- retweets: 25, favorites: 33 (02/17/2021 09:21:53)

- links: [abs](https://arxiv.org/abs/2102.06779) | [pdf](https://arxiv.org/pdf/2102.06779)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We consider the problem of controlling an invasive mechanical ventilator for pressure-controlled ventilation: a controller must let air in and out of a sedated patient's lungs according to a trajectory of airway pressures specified by a clinician.   Hand-tuned PID controllers and similar variants have comprised the industry standard for decades, yet can behave poorly by over- or under-shooting their target or oscillating rapidly.   We consider a data-driven machine learning approach: First, we train a simulator based on data we collect from an artificial lung. Then, we train deep neural network controllers on these simulators.We show that our controllers are able to track target pressure waveforms significantly better than PID controllers.   We further show that a learned controller generalizes across lungs with varying characteristics much more readily than PID controllers do.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The preliminary paper is out!!!  <a href="https://t.co/awSyIv4MyX">https://t.co/awSyIv4MyX</a><br>very excited about this collaboration as well as the work ahead of us to fulfill the potential of this technology in health! <a href="https://t.co/RHBo9HeoiX">https://t.co/RHBo9HeoiX</a></p>&mdash; Elad Hazan (@HazanPrinceton) <a href="https://twitter.com/HazanPrinceton/status/1361706660515876867?ref_src=twsrc%5Etfw">February 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 21. STruD: Truss Decomposition of Simplicial Complexes

Giulia Preti, Gianmarco De Francisci Morales, Francesco Bonchi

- retweets: 42, favorites: 15 (02/17/2021 09:21:53)

- links: [abs](https://arxiv.org/abs/2102.07564) | [pdf](https://arxiv.org/pdf/2102.07564)
- [cs.SI](https://arxiv.org/list/cs.SI/recent)

A simplicial complex is a generalization of a graph: a collection of n-ary relationships (instead of binary as the edges of a graph), named simplices. In this paper, we develop a new tool to study the structure of simplicial complexes: we generalize the graph notion of truss decomposition to complexes, and show that this more powerful representation gives rise to different properties compared to the graph-based one. This power, however, comes with important computational challenges derived from the combinatorial explosion caused by the downward closure property of complexes. Drawing upon ideas from itemset mining and similarity search, we design a memory-aware algorithm, dubbed STruD, which is able to efficiently compute the truss decomposition of a simplicial complex. STruD adapts its behavior to the amount of available memory by storing intermediate data in a compact way. We then devise a variant that computes directly the n simplices of maximum trussness. By applying STruD to several datasets, we prove its scalability, and provide an analysis of their structure. Finally, we show that the truss decomposition can be seen as a filtration, and as such it can be used to study the persistent homology of a dataset, a method for computing topological features at different spatial resolutions, prominent in Topological Data Analysis.



