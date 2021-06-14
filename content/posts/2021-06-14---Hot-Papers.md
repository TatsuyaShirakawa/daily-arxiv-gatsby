---
title: Hot Papers 2021-06-14
date: 2021-06-15T07:49:25.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-14"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-14"
socialImage: "/media/flying-marine.jpg"

---

# 1. Coordinate Independent Convolutional Networks -- Isometry and Gauge  Equivariant Convolutions on Riemannian Manifolds

Maurice Weiler, Patrick Forré, Erik Verlinde, Max Welling

- retweets: 16856, favorites: 4 (06/15/2021 07:49:25)

- links: [abs](https://arxiv.org/abs/2106.06020) | [pdf](https://arxiv.org/pdf/2106.06020)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CG](https://arxiv.org/list/cs.CG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Motivated by the vast success of deep convolutional networks, there is a great interest in generalizing convolutions to non-Euclidean manifolds. A major complication in comparison to flat spaces is that it is unclear in which alignment a convolution kernel should be applied on a manifold. The underlying reason for this ambiguity is that general manifolds do not come with a canonical choice of reference frames (gauge). Kernels and features therefore have to be expressed relative to arbitrary coordinates. We argue that the particular choice of coordinatization should not affect a network's inference -- it should be coordinate independent. A simultaneous demand for coordinate independence and weight sharing is shown to result in a requirement on the network to be equivariant under local gauge transformations (changes of local reference frames). The ambiguity of reference frames depends thereby on the G-structure of the manifold, such that the necessary level of gauge equivariance is prescribed by the corresponding structure group G. Coordinate independent convolutions are proven to be equivariant w.r.t. those isometries that are symmetries of the G-structure. The resulting theory is formulated in a coordinate free fashion in terms of fiber bundles. To exemplify the design of coordinate independent convolutions, we implement a convolutional network on the M\"obius strip. The generality of our differential geometric formulation of convolutional networks is demonstrated by an extensive literature review which explains a large number of Euclidean CNNs, spherical CNNs and CNNs on general surfaces as specific instances of coordinate independent convolutions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to announce our work on Coordinate Independent Convolutional Networks.<br>It develops a theory of CNNs on Riemannian manifolds and clarifies the interplay of the kernels&#39; local gauge equivariance and the networks&#39; global isometry equivariance.<a href="https://t.co/Vb0KHlTJ7S">https://t.co/Vb0KHlTJ7S</a><br>[1/N] <a href="https://t.co/TU3FusKk7M">pic.twitter.com/TU3FusKk7M</a></p>&mdash; Maurice Weiler (@maurice_weiler) <a href="https://twitter.com/maurice_weiler/status/1404405650151596046?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. LocoProp: Enhancing BackProp via Local Loss Optimization

Ehsan Amid, Rohan Anil, Manfred K. Warmuth

- retweets: 3660, favorites: 300 (06/15/2021 07:49:25)

- links: [abs](https://arxiv.org/abs/2106.06199) | [pdf](https://arxiv.org/pdf/2106.06199)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We study a local loss construction approach for optimizing neural networks. We start by motivating the problem as minimizing a squared loss between the pre-activations of each layer and a local target, plus a regularizer term on the weights. The targets are chosen so that the first gradient descent step on the local objectives recovers vanilla BackProp, while the exact solution to each problem results in a preconditioned gradient update. We improve the local loss construction by forming a Bregman divergence in each layer tailored to the transfer function which keeps the local problem convex w.r.t. the weights. The generalized local problem is again solved iteratively by taking small gradient descent steps on the weights, for which the first step recovers BackProp. We run several ablations and show that our construction consistently improves convergence, reducing the gap between first-order and second-order methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our progress on faster training of deep neural networks!<br><br>With the amazing Ehsan Amid (<a href="https://twitter.com/esiamid?ref_src=twsrc%5Etfw">@esiamid</a>) &amp; Manfred K. Warmuth<br><br>Achieves convergence close to second-order methods e.g. Shampoo &amp; K-FAC while significantly faster in walltime.<br><br>🧵👇<a href="https://t.co/KwZroZIiUU">https://t.co/KwZroZIiUU</a> <a href="https://t.co/CDYSdkRElB">pic.twitter.com/CDYSdkRElB</a></p>&mdash; Rohan Anil 🏠🤯 (@_arohan_) <a href="https://twitter.com/_arohan_/status/1404246293606637571?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Graph Neural Networks for Natural Language Processing: A Survey

Lingfei Wu, Yu Chen, Kai Shen, Xiaojie Guo, Hanning Gao, Shucheng Li, Jian Pei, Bo Long

- retweets: 3135, favorites: 307 (06/15/2021 07:49:25)

- links: [abs](https://arxiv.org/abs/2106.06090) | [pdf](https://arxiv.org/pdf/2106.06090)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep learning has become the dominant approach in coping with various tasks in Natural LanguageProcessing (NLP). Although text inputs are typically represented as a sequence of tokens, there isa rich variety of NLP problems that can be best expressed with a graph structure. As a result, thereis a surge of interests in developing new deep learning techniques on graphs for a large numberof NLP tasks. In this survey, we present a comprehensive overview onGraph Neural Networks(GNNs) for Natural Language Processing. We propose a new taxonomy of GNNs for NLP, whichsystematically organizes existing research of GNNs for NLP along three axes: graph construction,graph representation learning, and graph based encoder-decoder models. We further introducea large number of NLP applications that are exploiting the power of GNNs and summarize thecorresponding benchmark datasets, evaluation metrics, and open-source codes. Finally, we discussvarious outstanding challenges for making the full use of GNNs for NLP as well as future researchdirections. To the best of our knowledge, this is the first comprehensive overview of Graph NeuralNetworks for Natural Language Processing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔥 Graph Neural Networks for Natural Language Processing: A Survey<br><br>Lots of NLP tasks and applications can benefit from deep learning techniques on graphs. <br><br>Learn about these in one of the most comprehensive overviews of GNNs for NLP I&#39;ve seen.<a href="https://t.co/EsjWHxhKsN">https://t.co/EsjWHxhKsN</a> <a href="https://t.co/Ut17E5hlpZ">pic.twitter.com/Ut17E5hlpZ</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1404409987573731328?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Scaling Vision with Sparse Mixture of Experts

Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, Neil Houlsby

- retweets: 1562, favorites: 452 (06/15/2021 07:49:26)

- links: [abs](https://arxiv.org/abs/2106.05974) | [pdf](https://arxiv.org/pdf/2106.05974)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Sparsely-gated Mixture of Experts networks (MoEs) have demonstrated excellent scalability in Natural Language Processing. In Computer Vision, however, almost all performant networks are "dense", that is, every input is processed by every parameter. We present a Vision MoE (V-MoE), a sparse version of the Vision Transformer, that is scalable and competitive with the largest dense networks. When applied to image recognition, V-MoE matches the performance of state-of-the-art networks, while requiring as little as half of the compute at inference time. Further, we propose an extension to the routing algorithm that can prioritize subsets of each input across the entire batch, leading to adaptive per-image compute. This allows V-MoE to trade-off performance and compute smoothly at test-time. Finally, we demonstrate the potential of V-MoE to scale vision models, and train a 15B parameter model that attains 90.35% on ImageNet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out what we’ve been working on for the last months: We decouple the model size and the compute cost in a Vision Transformer backbone by using Sparse MoE layers. These have been popularised in NLP, and they are fantastic for Vision too! <a href="https://t.co/AdDwJKvkBk">https://t.co/AdDwJKvkBk</a> <a href="https://t.co/apg9sLYirY">pic.twitter.com/apg9sLYirY</a></p>&mdash; Joan Puigcerver (@joapuipe) <a href="https://twitter.com/joapuipe/status/1404346212116017153?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Vision with Sparse Mixture of Experts<br><br>Trains a 15B Vision Transformer with MoE that attains 90.35% on ImageNet.<a href="https://t.co/UitlGeymrv">https://t.co/UitlGeymrv</a> <a href="https://t.co/t3B3Mtcu1z">pic.twitter.com/t3B3Mtcu1z</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1404238514888212483?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">[3/3] Towards big vision<br><br>While dense models are still the norm, sparse MoE layers can work well too!<br><br>Large Vision-MoEs (15B params) can be trained to high performance relatively efficiently, and can even prioritize amongst patches (see duck).<a href="https://t.co/0WHyJTlGG5">https://t.co/0WHyJTlGG5</a><br><br>... <a href="https://t.co/INsCIiSmIi">pic.twitter.com/INsCIiSmIi</a></p>&mdash; Neil Houlsby (@neilhoulsby) <a href="https://twitter.com/neilhoulsby/status/1404319329005576192?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Vision with Sparse Mixture of Experts<br>pdf: <a href="https://t.co/GoTcuIGldr">https://t.co/GoTcuIGldr</a><br>abs: <a href="https://t.co/ZurZ2FhZnm">https://t.co/ZurZ2FhZnm</a><br><br>demonstrate the potential of V-MoE to scale vision models, and train a 15B parameter model that attains 90.35% on ImageNet <a href="https://t.co/nOOyWPK9w4">pic.twitter.com/nOOyWPK9w4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404237228075884545?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can we scale deep vision models to billions of parameters? Yes! By only activating the relevant parts of the network for each input. We present the Vision Mixture of Experts &amp; train a 15B-parameter model with 24 routers; transfers to ImageNet w. 90.35% acc <a href="https://t.co/QzgLQfKGZt">https://t.co/QzgLQfKGZt</a></p>&mdash; Carlos Riquelme (@rikelhood) <a href="https://twitter.com/rikelhood/status/1404371919336624137?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. SimSwap: An Efficient Framework For High Fidelity Face Swapping

Renwang Chen, Xuanhong Chen, Bingbing Ni, Yanhao Ge

- retweets: 1192, favorites: 184 (06/15/2021 07:49:26)

- links: [abs](https://arxiv.org/abs/2106.06340) | [pdf](https://arxiv.org/pdf/2106.06340)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose an efficient framework, called Simple Swap (SimSwap), aiming for generalized and high fidelity face swapping. In contrast to previous approaches that either lack the ability to generalize to arbitrary identity or fail to preserve attributes like facial expression and gaze direction, our framework is capable of transferring the identity of an arbitrary source face into an arbitrary target face while preserving the attributes of the target face. We overcome the above defects in the following two ways. First, we present the ID Injection Module (IIM) which transfers the identity information of the source face into the target face at feature level. By using this module, we extend the architecture of an identity-specific face swapping algorithm to a framework for arbitrary face swapping. Second, we propose the Weak Feature Matching Loss which efficiently helps our framework to preserve the facial attributes in an implicit way. Extensive experiments on wild faces demonstrate that our SimSwap is able to achieve competitive identity performance while preserving attributes better than previous state-of-the-art methods. The code is already available on github: https://github.com/neuralchen/SimSwap.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SimSwap: An Efficient Framework For High Fidelity<br>Face Swapping<br>pdf: <a href="https://t.co/l2aWTrM1CP">https://t.co/l2aWTrM1CP</a><br>abs: <a href="https://t.co/ZSuDnRLUuF">https://t.co/ZSuDnRLUuF</a><br>github: <a href="https://t.co/deYKr8rhLY">https://t.co/deYKr8rhLY</a> <a href="https://t.co/cBuaXySkd9">pic.twitter.com/cBuaXySkd9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404255810763509760?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. MlTr: Multi-label Classification with Transformer

Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, Dong Shen, Zhongyuan Wang, Nian Shi, Honglin Liu

- retweets: 324, favorites: 85 (06/15/2021 07:49:27)

- links: [abs](https://arxiv.org/abs/2106.06195) | [pdf](https://arxiv.org/pdf/2106.06195)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The task of multi-label image classification is to recognize all the object labels presented in an image. Though advancing for years, small objects, similar objects and objects with high conditional probability are still the main bottlenecks of previous convolutional neural network(CNN) based models, limited by convolutional kernels' representational capacity. Recent vision transformer networks utilize the self-attention mechanism to extract the feature of pixel granularity, which expresses richer local semantic information, while is insufficient for mining global spatial dependence. In this paper, we point out the three crucial problems that CNN-based methods encounter and explore the possibility of conducting specific transformer modules to settle them. We put forward a Multi-label Transformer architecture(MlTr) constructed with windows partitioning, in-window pixel attention, cross-window attention, particularly improving the performance of multi-label image classification tasks. The proposed MlTr shows state-of-the-art results on various prevalent multi-label datasets such as MS-COCO, Pascal-VOC, and NUS-WIDE with 88.5%, 95.8%, and 65.5% respectively. The code will be available soon at https://github.com/starmemda/MlTr/

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">MlTr: Multi-label Classification with Transformer<br>pdf: <a href="https://t.co/wqvd89AtJq">https://t.co/wqvd89AtJq</a><br>abs: <a href="https://t.co/H2n64N5OGa">https://t.co/H2n64N5OGa</a> <a href="https://t.co/yXFM5caLMK">pic.twitter.com/yXFM5caLMK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404298917651550208?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Dynamic Language Models for Continuously Evolving Content

Spurthi Amba Hombaiah, Tao Chen, Mingyang Zhang, Michael Bendersky, Marc Najork

- retweets: 324, favorites: 69 (06/15/2021 07:49:27)

- links: [abs](https://arxiv.org/abs/2106.06297) | [pdf](https://arxiv.org/pdf/2106.06297)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The content on the web is in a constant state of flux. New entities, issues, and ideas continuously emerge, while the semantics of the existing conversation topics gradually shift. In recent years, pre-trained language models like BERT greatly improved the state-of-the-art for a large spectrum of content understanding tasks. Therefore, in this paper, we aim to study how these language models can be adapted to better handle continuously evolving web content. In our study, we first analyze the evolution of 2013 - 2019 Twitter data, and unequivocally confirm that a BERT model trained on past tweets would heavily deteriorate when directly applied to data from later years. Then, we investigate two possible sources of the deterioration: the semantic shift of existing tokens and the sub-optimal or failed understanding of new tokens. To this end, we both explore two different vocabulary composition methods, as well as propose three sampling methods which help in efficient incremental training for BERT-like models. Compared to a new model trained from scratch offline, our incremental training (a) reduces the training costs, (b) achieves better performance on evolving content, and (c) is suitable for online deployment. The superiority of our methods is validated using two downstream tasks. We demonstrate significant improvements when incrementally evolving the model from a particular base year, on the task of Country Hashtag Prediction, as well as on the OffensEval 2019 task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dynamic Language Models for Continuously Evolving Content<br>pdf: <a href="https://t.co/dlkofC1MZ9">https://t.co/dlkofC1MZ9</a><br>abs: <a href="https://t.co/Ibed9GQSMJ">https://t.co/Ibed9GQSMJ</a> <a href="https://t.co/W6HoCer8ta">pic.twitter.com/W6HoCer8ta</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404260061967433731?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Part-aware Panoptic Segmentation

Daan de Geus, Panagiotis Meletis, Chenyang Lu, Xiaoxiao Wen, Gijs Dubbelman

- retweets: 272, favorites: 79 (06/15/2021 07:49:27)

- links: [abs](https://arxiv.org/abs/2106.06351) | [pdf](https://arxiv.org/pdf/2106.06351)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this work, we introduce the new scene understanding task of Part-aware Panoptic Segmentation (PPS), which aims to understand a scene at multiple levels of abstraction, and unifies the tasks of scene parsing and part parsing. For this novel task, we provide consistent annotations on two commonly used datasets: Cityscapes and Pascal VOC. Moreover, we present a single metric to evaluate PPS, called Part-aware Panoptic Quality (PartPQ). For this new task, using the metric and annotations, we set multiple baselines by merging results of existing state-of-the-art methods for panoptic segmentation and part segmentation. Finally, we conduct several experiments that evaluate the importance of the different levels of abstraction in this single task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Part-aware Panoptic Segmentation<br>pdf: <a href="https://t.co/519l92lKsm">https://t.co/519l92lKsm</a><br>abs: <a href="https://t.co/tm2UsWNnlz">https://t.co/tm2UsWNnlz</a><br>github: <a href="https://t.co/uGYVU0PPmZ">https://t.co/uGYVU0PPmZ</a> <a href="https://t.co/Fu1xRslPPP">pic.twitter.com/Fu1xRslPPP</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404303462221815808?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Conditional Variational Autoencoder with Adversarial Learning for  End-to-End Text-to-Speech

Jaehyeon Kim, Jungil Kong, Juhee Son

- retweets: 163, favorites: 111 (06/15/2021 07:49:27)

- links: [abs](https://arxiv.org/abs/2106.06103) | [pdf](https://arxiv.org/pdf/2106.06103)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech 🗣️<br>pdf: <a href="https://t.co/7YqvteNvDv">https://t.co/7YqvteNvDv</a><br>abs: <a href="https://t.co/uf8lt52BYd">https://t.co/uf8lt52BYd</a><br>github: <a href="https://t.co/GnZOHaS7o5">https://t.co/GnZOHaS7o5</a><br>project page: <a href="https://t.co/ZQPcUhuOZ3">https://t.co/ZQPcUhuOZ3</a> <a href="https://t.co/ViCgjeYXDZ">pic.twitter.com/ViCgjeYXDZ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404247998981361665?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LJ Speech: Game over.<a href="https://t.co/yFGqLEM1ZZ">https://t.co/yFGqLEM1ZZ</a> <a href="https://t.co/DwFLzhhvoy">pic.twitter.com/DwFLzhhvoy</a></p>&mdash; Seung-won Park (@veydpz_public) <a href="https://twitter.com/veydpz_public/status/1404267565656854529?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. View Generalization for Single Image Textured 3D Models

Anand Bhattad, Aysegul Dundar, Guilin Liu, Andrew Tao, Bryan Catanzaro

- retweets: 144, favorites: 82 (06/15/2021 07:49:27)

- links: [abs](https://arxiv.org/abs/2106.06533) | [pdf](https://arxiv.org/pdf/2106.06533)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Humans can easily infer the underlying 3D geometry and texture of an object only from a single 2D image. Current computer vision methods can do this, too, but suffer from view generalization problems - the models inferred tend to make poor predictions of appearance in novel views. As for generalization problems in machine learning, the difficulty is balancing single-view accuracy (cf. training error; bias) with novel view accuracy (cf. test error; variance). We describe a class of models whose geometric rigidity is easily controlled to manage this tradeoff. We describe a cycle consistency loss that improves view generalization (roughly, a model from a generated view should predict the original view well). View generalization of textures requires that models share texture information, so a car seen from the back still has headlights because other cars have headlights. We describe a cycle consistency loss that encourages model textures to be aligned, so as to encourage sharing. We compare our method against the state-of-the-art method and show both qualitative and quantitative improvements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">View Generalization for Single Image Textured 3D Models<br>pdf: <a href="https://t.co/cHbMfNofTH">https://t.co/cHbMfNofTH</a><br>abs: <a href="https://t.co/0a0YVyb8Qt">https://t.co/0a0YVyb8Qt</a><br><br>a new 3D inference pipeline, principally designed to display good view generalization <a href="https://t.co/kzbNdqT1Pz">pic.twitter.com/kzbNdqT1Pz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404244116557516801?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Scalable Variational Gaussian Processes via Harmonic Kernel  Decomposition

Shengyang Sun, Jiaxin Shi, Andrew Gordon Wilson, Roger Grosse

- retweets: 156, favorites: 56 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.05992) | [pdf](https://arxiv.org/pdf/2106.05992)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We introduce a new scalable variational Gaussian process approximation which provides a high fidelity approximation while retaining general applicability. We propose the harmonic kernel decomposition (HKD), which uses Fourier series to decompose a kernel as a sum of orthogonal kernels. Our variational approximation exploits this orthogonality to enable a large number of inducing points at a low computational cost. We demonstrate that, on a range of regression and classification problems, our approach can exploit input space symmetries such as translations and reflections, and it significantly outperforms standard variational methods in scalability and accuracy. Notably, our approach achieves state-of-the-art results on CIFAR-10 among pure GP models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our HVGP <a href="https://twitter.com/ICML2021?ref_src=twsrc%5Etfw">@ICML2021</a><br> : <br>1) discrete Fourier transform meets kernels<br>2) decomposing a kernel into an orthogonal sum of kernels  <br>3) a scalable SVGP to use more inducing points<br>4) being applicable to common kernels: RBF, Matern, poly, ...<a href="https://t.co/3WZElhCOdt">https://t.co/3WZElhCOdt</a> <a href="https://t.co/6v2LHnIPXg">pic.twitter.com/6v2LHnIPXg</a></p>&mdash; Shengyang Sun (@ssydasheng) <a href="https://twitter.com/ssydasheng/status/1404444605110800395?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Preferential Temporal Difference Learning

Nishanth Anand, Doina Precup

- retweets: 156, favorites: 45 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.06508) | [pdf](https://arxiv.org/pdf/2106.06508)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Temporal-Difference (TD) learning is a general and very useful tool for estimating the value function of a given policy, which in turn is required to find good policies. Generally speaking, TD learning updates states whenever they are visited. When the agent lands in a state, its value can be used to compute the TD-error, which is then propagated to other states. However, it may be interesting, when computing updates, to take into account other information than whether a state is visited or not. For example, some states might be more important than others (such as states which are frequently seen in a successful trajectory). Or, some states might have unreliable value estimates (for example, due to partial observability or lack of data), making their values less desirable as targets. We propose an approach to re-weighting states used in TD updates, both when they are the input and when they provide the target for the update. We prove that our approach converges with linear function approximation and illustrate its desirable empirical behaviour compared to other TD-style methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Did you know that Temporal Difference (TD) Learning has a major flaw?<br><br>Happy to announce Preferential Temporal Difference Learning (accepted at the <a href="https://twitter.com/icmlconf?ref_src=twsrc%5Etfw">@icmlconf</a>) that fixes these!<br><br>Work supervised by Doina Precup. Link: <a href="https://t.co/j0UPOUHs5k">https://t.co/j0UPOUHs5k</a><br><br>More in the thread below. (1/6)</p>&mdash; Nishanth Anand (@itsNVA7) <a href="https://twitter.com/itsNVA7/status/1404468661767786501?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Catch-A-Waveform: Learning to Generate Audio from a Single Short Example

Gal Greshler, Tamar Rott Shaham, Tomer Michaeli

- retweets: 112, favorites: 83 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.06426) | [pdf](https://arxiv.org/pdf/2106.06426)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Models for audio generation are typically trained on hours of recordings. Here, we illustrate that capturing the essence of an audio source is typically possible from as little as a few tens of seconds from a single training signal. Specifically, we present a GAN-based generative model that can be trained on one short audio signal from any domain (e.g. speech, music, etc.) and does not require pre-training or any other form of external supervision. Once trained, our model can generate random samples of arbitrary duration that maintain semantic similarity to the training waveform, yet exhibit new compositions of its audio primitives. This enables a long line of interesting applications, including generating new jazz improvisations or new a-cappella rap variants based on a single short example, producing coherent modifications to famous songs (e.g. adding a new verse to a Beatles song based solely on the original recording), filling-in of missing parts (inpainting), extending the bandwidth of a speech signal (super-resolution), and enhancing old recordings without access to any clean training example. We show that in all cases, no more than 20 seconds of training audio commonly suffice for our model to achieve state-of-the-art results. This is despite its complete lack of prior knowledge about the nature of audio signals in general.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Catch-A-Waveform: Learning to Generate Audio<br>from a Single Short Example<br>pdf: <a href="https://t.co/hXZ7oPhBI1">https://t.co/hXZ7oPhBI1</a><br>abs: <a href="https://t.co/TwIkJGvbHi">https://t.co/TwIkJGvbHi</a><br>project page: <a href="https://t.co/KWNr0YIU4o">https://t.co/KWNr0YIU4o</a><br><br>no more than 20 seconds of training audio commonly suffice for model to achieve sota results <a href="https://t.co/N2oZxsz2kh">pic.twitter.com/N2oZxsz2kh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404250822557540353?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Zero-Shot Controlled Generation with Encoder-Decoder Transformers

Devamanyu Hazarika, Mahdi Namazifar, Dilek Hakkani-Tür

- retweets: 67, favorites: 71 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.06411) | [pdf](https://arxiv.org/pdf/2106.06411)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Controlling neural network-based models for natural language generation (NLG) has broad applications in numerous areas such as machine translation, document summarization, and dialog systems. Approaches that enable such control in a zero-shot manner would be of great importance as, among other reasons, they remove the need for additional annotated data and training. In this work, we propose novel approaches for controlling encoder-decoder transformer-based NLG models in a zero-shot manner. This is done by introducing three control knobs; namely, attention biasing, decoder mixing, and context augmentation, that are applied to these models at generation time. These knobs control the generation process by directly manipulating trained NLG models (e.g., biasing cross-attention layers) to realize the desired attributes in the generated outputs. We show that not only are these NLG models robust to such manipulations, but also their behavior could be controlled without an impact on their generation performance. These results, to the best of our knowledge, are the first of their kind. Through these control knobs, we also investigate the role of transformer decoder's self-attention module and show strong evidence that its primary role is maintaining fluency of sentences generated by these models. Based on this hypothesis, we show that alternative architectures for transformer decoders could be viable options. We also study how this hypothesis could lead to more efficient ways for training encoder-decoder transformer models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Zero-Shot Controlled Generation with Encoder-Decoder Transformers<br>pdf: <a href="https://t.co/Ak2pHvh9J5">https://t.co/Ak2pHvh9J5</a><br>abs: <a href="https://t.co/9npeMy0kHa">https://t.co/9npeMy0kHa</a><br><br>can achieve zero-shot controllability for models that are orders of magnitude smaller than GPT-3 (e.g., BART-base) <a href="https://t.co/O3yL5PzJgw">pic.twitter.com/O3yL5PzJgw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404241274987859970?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Recovery of Meteorites Using an Autonomous Drone and Machine Learning

Robert I. Citron, Peter Jenniskens, Christopher Watkins, Sravanthi Sinha, Amar Shah, Chedy Raissi, Hadrien Devillepoix, Jim Albers

- retweets: 73, favorites: 27 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.06523) | [pdf](https://arxiv.org/pdf/2106.06523)
- [astro-ph.EP](https://arxiv.org/list/astro-ph.EP/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The recovery of freshly fallen meteorites from tracked and triangulated meteors is critical to determining their source asteroid families. However, locating meteorite fragments in strewn fields remains a challenge with very few meteorites being recovered from the meteors triangulated in past and ongoing meteor camera networks. We examined if locating meteorites can be automated using machine learning and an autonomous drone. Drones can be programmed to fly a grid search pattern and take systematic pictures of the ground over a large survey area. Those images can be analyzed using a machine learning classifier to identify meteorites in the field among many other features. Here, we describe a proof-of-concept meteorite classifier that deploys off-line a combination of different convolution neural networks to recognize meteorites from images taken by drones in the field. The system was implemented in a conceptual drone setup and tested in the suspected strewn field of a recent meteorite fall near Walker Lake, Nevada.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Researchers have taught a drone to recognize and retrieve meteorites autonomously -  <a href="https://t.co/7sZVEGUOLX">https://t.co/7sZVEGUOLX</a> <a href="https://t.co/ux8UlTBoPG">pic.twitter.com/ux8UlTBoPG</a></p>&mdash; Fraser Cain (@fcain) <a href="https://twitter.com/fcain/status/1404520783314051072?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Where to Encode: A Performance Analysis of x86 and Arm-based Amazon EC2  Instance

Roland Mathá, Dragi Kimovski, Anatoliy Zabrovskiy, Christian Timmerer, Radu Prodan

- retweets: 36, favorites: 33 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.06242) | [pdf](https://arxiv.org/pdf/2106.06242)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.AR](https://arxiv.org/list/cs.AR/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

Video streaming became an undivided part of the Internet. To efficiently utilize the limited network bandwidth it is essential to encode the video content. However, encoding is a computationally intensive task, involving high-performance resources provided by private infrastructures or public clouds. Public clouds, such as Amazon EC2, provide a large portfolio of services and instances optimized for specific purposes and budgets. The majority of Amazon instances use x86 processors, such as Intel Xeon or AMD EPYC. However, following the recent trends in computer architecture, Amazon introduced Arm-based instances that promise up to 40% better cost-performance ratio than comparable x86 instances for specific workloads. We evaluate in this paper the video encoding performance of x86 and Arm instances of four instance families using the latest FFmpeg version and two video codecs. We examine the impact of the encoding parameters, such as different presets and bitrates, on the time and cost for encoding. Our experiments reveal that Arm instances show high time and cost-saving potential of up to 33.63% for specific bitrates and presets, especially for the x264 codec. However, the x86 instances are more general and achieve low encoding times, regardless of the codec.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper, researchers evaluated the video encoding performance of x86 and ARM instances of four Amazon EC2 instance families using the latest FFmpeg version and two video codecs.<a href="https://t.co/GcaqVNLy7Q">https://t.co/GcaqVNLy7Q</a> <a href="https://t.co/gtrVoW8QIC">pic.twitter.com/gtrVoW8QIC</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1404297020685303808?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Going Beyond Linear Transformers with Recurrent Fast Weight Programmers

Kazuki Irie, Imanol Schlag, Róbert Csordás, Jürgen Schmidhuber

- retweets: 30, favorites: 29 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.06295) | [pdf](https://arxiv.org/pdf/2106.06295)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers with linearised attention ("linear Transformers") have demonstrated the practical scalability and effectiveness of outer product-based Fast Weight Programmers (FWPs) from the '90s. However, the original FWP formulation is more general than the one of linear Transformers: a slow neural network (NN) continually reprograms the weights of a fast NN with arbitrary NN architectures. In existing linear Transformers, both NNs are feedforward and consist of a single layer. Here we explore new variations by adding recurrence to the slow and fast nets. We evaluate our novel recurrent FWPs (RFWPs) on two synthetic algorithmic tasks (code execution and sequential ListOps), Wikitext-103 language models, and on the Atari 2600 2D game environment. Our models exhibit properties of Transformers and RNNs. In the reinforcement learning setting, we report large improvements over LSTM in several Atari games. Our code is public.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Going Beyond Linear Transformers with Recurrent Fast Weight Programmers<br>pdf: <a href="https://t.co/oS6D1IQTRG">https://t.co/oS6D1IQTRG</a><br>abs: <a href="https://t.co/LqKRq9kBrM">https://t.co/LqKRq9kBrM</a><br>github: <a href="https://t.co/KGewO2s85A">https://t.co/KGewO2s85A</a><br><br>various new linear Transformer variants with recurrent connections <a href="https://t.co/d4geCESa9z">pic.twitter.com/d4geCESa9z</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404239584507146244?ref_src=twsrc%5Etfw">June 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Probabilistic, Structure-Aware Algorithms for Improved Variety,  Accuracy, and Coverage of AMR Alignments

Austin Blodgett, Nathan Schneider

- retweets: 42, favorites: 13 (06/15/2021 07:49:28)

- links: [abs](https://arxiv.org/abs/2106.06002) | [pdf](https://arxiv.org/pdf/2106.06002)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We present algorithms for aligning components of Abstract Meaning Representation (AMR) graphs to spans in English sentences. We leverage unsupervised learning in combination with heuristics, taking the best of both worlds from previous AMR aligners. Our unsupervised models, however, are more sensitive to graph substructures, without requiring a separate syntactic parse. Our approach covers a wider variety of AMR substructures than previously considered, achieves higher coverage of nodes and edges, and does so with higher accuracy. We will release our LEAMR datasets and aligner for use in research on AMR parsing, generation, and evaluation.



