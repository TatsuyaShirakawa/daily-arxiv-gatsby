---
title: Hot Papers 2020-07-15
date: 2020-07-16T09:44:51.Z
template: "post"
draft: false
slug: "hot-papers-2020-07-15"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-07-15"
socialImage: "/media/42-line-bible.jpg"

---

# 1. Closed-Form Factorization of Latent Semantics in GANs

Yujun Shen, Bolei Zhou

- retweets: 112, favorites: 294 (07/16/2020 09:44:51)

- links: [abs](https://arxiv.org/abs/2007.06600) | [pdf](https://arxiv.org/pdf/2007.06600)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

A rich set of semantic attributes has been shown to emerge in the latent space of the Generative Adversarial Networks (GANs) trained for synthesizing images. In order to identify such latent semantics for image manipulation, previous methods annotate a collection of synthesized samples and then train supervised classifiers in the latent space. However, they require a clear definition of the target attribute as well as the corresponding manual annotations, severely limiting their applications in practice. In this work, we examine the internal representation learned by GANs to reveal the underlying variation factors in an unsupervised manner. By studying the essential role of the fully-connected layer that takes the latent code into the generator of GANs, we propose a general closed-form factorization method for latent semantic discovery. The properties of the identified semantics are further analyzed both theoretically and empirically. With its fast and efficient implementation, our approach is capable of not only finding latent semantics as accurately as the state-of-the-art supervised methods, but also resulting in far more versatile semantic classes across multiple GAN models trained on a wide range of datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Closed-Form Factorization of Latent Semantics in GANs<br>pdf: <a href="https://t.co/pYda24esEJ">https://t.co/pYda24esEJ</a><br>abs: <a href="https://t.co/TTx4gwm5Xl">https://t.co/TTx4gwm5Xl</a> <a href="https://t.co/7YuVWnuquW">pic.twitter.com/7YuVWnuquW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1283221912949264386?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Multitask Learning Strengthens Adversarial Robustness

Chengzhi Mao, Amogh Gupta, Vikram Nitin, Baishakhi Ray, Shuran Song, Junfeng Yang, Carl Vondrick

- retweets: 33, favorites: 140 (07/16/2020 09:44:51)

- links: [abs](https://arxiv.org/abs/2007.07236) | [pdf](https://arxiv.org/pdf/2007.07236)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Although deep networks achieve strong accuracy on a range of computer vision benchmarks, they remain vulnerable to adversarial attacks, where imperceptible input perturbations fool the network. We present both theoretical and empirical analyses that connect the adversarial robustness of a model to the number of tasks that it is trained on. Experiments on two datasets show that attack difficulty increases as the number of target tasks increase. Moreover, our results suggest that when models are trained on multiple tasks at once, they become more robust to adversarial attacks on individual tasks. While adversarial defense remains an open challenge, our results suggest that deep networks are vulnerable partly because they are trained on too few tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What causes adversarial examples? Latest <a href="https://twitter.com/hashtag/ECCV2020?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV2020</a> paper from <a href="https://twitter.com/ChengzhiM?ref_src=twsrc%5Etfw">@ChengzhiM</a> and Amogh shows that deep networks are vulnerable partly because they are trained on too few tasks. Just by increasing tasks, we strengthen  robustness for each task individually. <a href="https://t.co/zQzpS9pWtE">https://t.co/zQzpS9pWtE</a> <a href="https://t.co/B5LoWkrHgg">pic.twitter.com/B5LoWkrHgg</a></p>&mdash; Carl Vondrick (@cvondrick) <a href="https://twitter.com/cvondrick/status/1283229668573884416?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Transposer: Universal Texture Synthesis Using Feature Maps as Transposed  Convolution Filter

Guilin Liu, Rohan Taori, Ting-Chun Wang, Zhiding Yu, Shiqiu Liu, Fitsum A. Reda, Karan Sapra, Andrew Tao, Bryan Catanzaro

- retweets: 26, favorites: 73 (07/16/2020 09:44:51)

- links: [abs](https://arxiv.org/abs/2007.07243) | [pdf](https://arxiv.org/pdf/2007.07243)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Conventional CNNs for texture synthesis consist of a sequence of (de)-convolution and up/down-sampling layers, where each layer operates locally and lacks the ability to capture the long-term structural dependency required by texture synthesis. Thus, they often simply enlarge the input texture, rather than perform reasonable synthesis. As a compromise, many recent methods sacrifice generalizability by training and testing on the same single (or fixed set of) texture image(s), resulting in huge re-training time costs for unseen images. In this work, based on the discovery that the assembling/stitching operation in traditional texture synthesis is analogous to a transposed convolution operation, we propose a novel way of using transposed convolution operation. Specifically, we directly treat the whole encoded feature map of the input texture as transposed convolution filters and the features' self-similarity map, which captures the auto-correlation information, as input to the transposed convolution. Such a design allows our framework, once trained, to be generalizable to perform synthesis of unseen textures with a single forward pass in nearly real-time. Our method achieves state-of-the-art texture synthesis quality based on various metrics. While self-similarity helps preserve the input textures' regular structural patterns, our framework can also take random noise maps for irregular input textures instead of self-similarity maps as transposed convolution inputs. It allows to get more diverse results as well as generate arbitrarily large texture outputs by directly sampling large noise maps in a single pass as well.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">📢 Transposer: Universal Texture Synthesis Using Feature Maps as Transposed Convolution Filter📢<br><br>Video: <a href="https://t.co/cKjbJyeH6f">https://t.co/cKjbJyeH6f</a><br>Paper: <a href="https://t.co/rzwoVWBwB5">https://t.co/rzwoVWBwB5</a><br><br>We propose a generalizable framework that can perform texture synthesis for unseen texture images in nearly real-time.</p>&mdash; Guilin Liu (@GuilinL) <a href="https://twitter.com/GuilinL/status/1283208684047265792?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Transposer: Universal Texture Synthesis Using Feature Maps as Transposed Convolution Filter<br>pdf: <a href="https://t.co/mqfh1CbqPI">https://t.co/mqfh1CbqPI</a><br>abs: <a href="https://t.co/J1ZW0wBSpA">https://t.co/J1ZW0wBSpA</a> <a href="https://t.co/I0MQB1Zhkj">pic.twitter.com/I0MQB1Zhkj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1283213611431874560?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Alleviating Over-segmentation Errors by Detecting Action Boundaries

Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, Hirokatsu Kataoka

- retweets: 23, favorites: 75 (07/16/2020 09:44:52)

- links: [abs](https://arxiv.org/abs/2007.06866) | [pdf](https://arxiv.org/pdf/2007.06866)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose an effective framework for the temporal action segmentation task, namely an Action Segment Refinement Framework (ASRF). Our model architecture consists of a long-term feature extractor and two branches: the Action Segmentation Branch (ASB) and the Boundary Regression Branch (BRB). The long-term feature extractor provides shared features for the two branches with a wide temporal receptive field. The ASB classifies video frames with action classes, while the BRB regresses the action boundary probabilities. The action boundaries predicted by the BRB refine the output from the ASB, which results in a significant performance improvement. Our contributions are three-fold: (i) We propose a framework for temporal action segmentation, the ASRF, which divides temporal action segmentation into frame-wise action classification and action boundary regression. Our framework refines frame-level hypotheses of action classes using predicted action boundaries. (ii) We propose a loss function for smoothing the transition of action probabilities, and analyze combinations of various loss functions for temporal action segmentation. (iii) Our framework outperforms state-of-the-art methods on three challenging datasets, offering an improvement of up to 13.7% in terms of segmental edit distance and up to 16.1% in terms of segmental F1 score. Our code will be publicly available soon.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">arXivで時系列行動セグメンテーションの論文を公開しました！時間軸の人物行動を過剰に分離してしまう問題（Over Segmentation）の解決に、行動間の切れ目を認識するネットワークを導入し、現在の最高精度を達成しました。F1値では従来法と比較し最大+16.1pt 向上しています。<a href="https://t.co/6GMr1yZrTu">https://t.co/6GMr1yZrTu</a> <a href="https://t.co/6NUDjiVm0W">pic.twitter.com/6NUDjiVm0W</a></p>&mdash; cvpaper.challenge (@CVpaperChalleng) <a href="https://twitter.com/CVpaperChalleng/status/1283202024679436288?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">初arXivでした<br>次はもっといい研究できるように頑張ります<a href="https://t.co/gdyeK5g03h">https://t.co/gdyeK5g03h</a></p>&mdash; yuchi (@yciskw_) <a href="https://twitter.com/yciskw_/status/1283197276127559680?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Transformer-XL Based Music Generation with Multiple Sequences of  Time-valued Notes

Xianchao Wu, Chengyuan Wang, Qinying Lei

- retweets: 15, favorites: 74 (07/16/2020 09:44:52)

- links: [abs](https://arxiv.org/abs/2007.07244) | [pdf](https://arxiv.org/pdf/2007.07244)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Current state-of-the-art AI based classical music creation algorithms such as Music Transformer are trained by employing single sequence of notes with time-shifts. The major drawback of absolute time interval expression is the difficulty of similarity computing of notes that share the same note value yet different tempos, in one or among MIDI files. In addition, the usage of single sequence restricts the model to separately and effectively learn music information such as harmony and rhythm. In this paper, we propose a framework with two novel methods to respectively track these two shortages, one is the construction of time-valued note sequences that liberate note values from tempos and the other is the separated usage of four sequences, namely, former note on to current note on, note on to note off, pitch, and velocity, for jointly training of four Transformer-XL networks. Through training on a 23-hour piano MIDI dataset, our framework generates significantly better and hour-level longer music than three state-of-the-art baselines, namely Music Transformer, DeepJ, and single sequence-based Transformer-XL, evaluated automatically and manually.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Transformer-XL Based Music Generation with Multiple Sequences of Time-valued Notes<br>pdf: <a href="https://t.co/xTrQBOTspz">https://t.co/xTrQBOTspz</a><br>abs: <a href="https://t.co/GiCuFyyVOc">https://t.co/GiCuFyyVOc</a> <a href="https://t.co/k8fVWqGmku">pic.twitter.com/k8fVWqGmku</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1283229461689896960?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Causal Inference using Gaussian Processes with Structured Latent  Confounders

Sam Witty, Kenta Takatsu, David Jensen, Vikash Mansinghka

- retweets: 21, favorites: 59 (07/16/2020 09:44:52)

- links: [abs](https://arxiv.org/abs/2007.07127) | [pdf](https://arxiv.org/pdf/2007.07127)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Latent confounders---unobserved variables that influence both treatment and outcome---can bias estimates of causal effects. In some cases, these confounders are shared across observations, e.g. all students taking a course are influenced by the course's difficulty in addition to any educational interventions they receive individually. This paper shows how to semiparametrically model latent confounders that have this structure and thereby improve estimates of causal effects. The key innovations are a hierarchical Bayesian model, Gaussian processes with structured latent confounders (GP-SLC), and a Monte Carlo inference algorithm for this model based on elliptical slice sampling. GP-SLC provides principled Bayesian uncertainty estimates of individual treatment effect with minimal assumptions about the functional forms relating confounders, covariates, treatment, and outcome. Finally, this paper shows GP-SLC is competitive with or more accurate than widely used causal inference techniques on three benchmark datasets, including the Infant Health and Development Program and a dataset showing the effect of changing temperatures on state-wide energy consumption across New England.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper “Causal Inference using Gaussian Processes with Structured Latent Confounders” has been accepted at <a href="https://twitter.com/hashtag/ICML2020?src=hash&amp;ref_src=twsrc%5Etfw">#ICML2020</a>! Joint work with <a href="https://twitter.com/kenta_takatsu?ref_src=twsrc%5Etfw">@kenta_takatsu</a>, David Jensen, and <a href="https://twitter.com/vmansinghka?ref_src=twsrc%5Etfw">@vmansinghka</a>.<br><br>Paper:  <a href="https://t.co/yufXNEloo8">https://t.co/yufXNEloo8</a><br>Poster: <a href="https://t.co/87229HOFRo">https://t.co/87229HOFRo</a> <a href="https://t.co/eI2fLwTzAi">pic.twitter.com/eI2fLwTzAi</a></p>&mdash; Sam Witty (@ICML2020) (@swittbit) <a href="https://twitter.com/swittbit/status/1283378791813873681?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. A Survey of Numerical Methods Utilizing Mixed Precision Arithmetic

Ahmad Abdelfattah, Hartwig Anzt, Erik G. Boman, Erin Carson, Terry Cojean, Jack Dongarra, Mark Gates, Thomas Grützmacher, Nicholas J. Higham, Sherry Li, Neil Lindquist, Yang Liu, Jennifer Loe, Piotr Luszczek, Pratik Nayak, Sri Pranesh, Siva Rajamanickam, Tobias Ribizel, Barry Smith, Kasia Swirydowicz, Stephen Thomas, Stanimire Tomov, Yaohung M. Tsai, Ichitaro Yamazaki, Urike Meier Yang

- retweets: 17, favorites: 52 (07/16/2020 09:44:52)

- links: [abs](https://arxiv.org/abs/2007.06674) | [pdf](https://arxiv.org/pdf/2007.06674)
- [cs.MS](https://arxiv.org/list/cs.MS/recent) | [math.NA](https://arxiv.org/list/math.NA/recent)

Within the past years, hardware vendors have started designing low precision special function units in response to the demand of the Machine Learning community and their demand for high compute power in low precision formats. Also the server-line products are increasingly featuring low-precision special function units, such as the NVIDIA tensor cores in ORNL's Summit supercomputer providing more than an order of magnitude higher performance than what is available in IEEE double precision. At the same time, the gap between the compute power on the one hand and the memory bandwidth on the other hand keeps increasing, making data access and communication prohibitively expensive compared to arithmetic operations. To start the multiprecision focus effort, we survey the numerical linear algebra community and summarize all existing multiprecision knowledge, expertise, and software capabilities in this landscape analysis report. We also include current efforts and preliminary results that may not yet be considered "mature technology," but have the potential to grow into production quality within the multiprecision focus effort. As we expect the reader to be familiar with the basics of numerical linear algebra, we refrain from providing a detailed background on the algorithms themselves but focus on how mixed- and multiprecision technology can help improving the performance of these methods and present highlights of application significantly outperforming the traditional fixed precision methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Exascale Computing Project Team have presented a comprehensive survey on numerical methods utilizing mixed precision arithmetic, highlighting some application which significantly outperform the traditional fixed precision methods. <a href="https://twitter.com/hashtag/HPC?src=hash&amp;ref_src=twsrc%5Etfw">#HPC</a><a href="https://t.co/I0XOmsWu3g">https://t.co/I0XOmsWu3g</a> <a href="https://t.co/eMOAbVViTC">pic.twitter.com/eMOAbVViTC</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1283253870119006210?ref_src=twsrc%5Etfw">July 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Power, Preferment, and Patronage: Catholic Bishops, Social Networks, and  the Affair(s) of Ex-Cardinal McCarrick

Stephen Bullivant, Giovanni Radhitio Putra Sadewo

- retweets: 9, favorites: 46 (07/16/2020 09:44:52)

- links: [abs](https://arxiv.org/abs/2007.06606) | [pdf](https://arxiv.org/pdf/2007.06606)
- [cs.SI](https://arxiv.org/list/cs.SI/recent)

Social Network Analysis (SNA) has shed powerful light on cultures where the influence of patronage, preferment, and reciprocal obligations are traditionally important. Accordingly, we argue here that episcopal appointments, culture, and governance within the Catholic Church are ideal topics for SNA interrogation. We analyse original network data for the Catholic Bishops' Conference of England and Wales, and the United States Conference of Catholic Bishops. Significantly, we show how a network-informed approach may help with the urgent task of understanding the ecclesiastical cultures in which sexual abuse occurs, and/or is enabled, ignored, and covered up. Particular reference is made to Theodore McCarrick, the former DC Archbishop "dismissed from the clerical state" for sexual offences. Commentators naturally use terms like "protege", "clique", "network", and "kingmaker" when discussing both the McCarrick affair and church politics more generally: precisely such folk-descriptions of social and political life that SNA is designed to quantify and explain.




# 9. Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale  Recommendations

Weihao Gao, Xiangjun Fan, Jiankai Sun, Kai Jia, Wenzhi Xiao, Chong Wang, Xiaobing Liu

- retweets: 6, favorites: 46 (07/16/2020 09:44:52)

- links: [abs](https://arxiv.org/abs/2007.07203) | [pdf](https://arxiv.org/pdf/2007.07203)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

One of the core problems in large-scale recommendations is to retrieve top relevant candidates accurately and efficiently, preferably in sub-linear time. Previous approaches are mostly based on a two-step procedure: first learn an inner-product model and then use maximum inner product search (MIPS) algorithms to search top candidates, leading to potential loss of retrieval accuracy. In this paper, we present Deep Retrieval (DR), an end-to-end learnable structure model for large-scale recommendations. DR encodes all candidates into a discrete latent space. Those latent codes for the candidates are model parameters and to be learnt together with other neural network parameters to maximize the same objective function. With the model learnt, a beam search over the latent codes is performed to retrieve the top candidates. Empirically, we showed that DR, with sub-linear computational complexity, can achieve almost the same accuracy as the brute-force baseline.



