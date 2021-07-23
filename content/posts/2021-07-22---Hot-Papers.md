---
title: Hot Papers 2021-07-22
date: 2021-07-23T19:39:28.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-22"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-22"
socialImage: "/media/flying-marine.jpg"

---

# 1. CycleMLP: A MLP-like Architecture for Dense Prediction

Shoufa Chen, Enze Xie, Chongjian Ge, Ding Liang, Ping Luo

- retweets: 1602, favorites: 162 (07/23/2021 19:39:28)

- links: [abs](https://arxiv.org/abs/2107.10224) | [pdf](https://arxiv.org/pdf/2107.10224)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper presents a simple MLP-like architecture, CycleMLP, which is a versatile backbone for visual recognition and dense predictions, unlike modern MLP architectures, e.g., MLP-Mixer, ResMLP, and gMLP, whose architectures are correlated to image size and thus are infeasible in object detection and segmentation. CycleMLP has two advantages compared to modern approaches. (1) It can cope with various image sizes. (2) It achieves linear computational complexity to image size by using local windows. In contrast, previous MLPs have quadratic computations because of their fully spatial connections. We build a family of models that surpass existing MLPs and achieve a comparable accuracy (83.2%) on ImageNet-1K classification compared to the state-of-the-art Transformer such as Swin Transformer (83.3%) but using fewer parameters and FLOPs. We expand the MLP-like models' applicability, making them a versatile backbone for dense prediction tasks. CycleMLP aims to provide a competitive baseline on object detection, instance segmentation, and semantic segmentation for MLP models. In particular, CycleMLP achieves 45.1 mIoU on ADE20K val, comparable to Swin (45.2 mIOU). Code is available at \url{https://github.com/ShoufaChen/CycleMLP}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CycleMLP: A MLP-like Architecture for Dense Prediction<br>pdf: <a href="https://t.co/EMFJTNDjP0">https://t.co/EMFJTNDjP0</a><br>abs: <a href="https://t.co/gEobffbGon">https://t.co/gEobffbGon</a><br><br>CycleMLP achieves 45.1 mIoU on ADE20K val, comparable to Swin (45.2 mIOU) <a href="https://t.co/1Hf2uErufD">pic.twitter.com/1Hf2uErufD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1418007982034587650?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Audio Captioning Transformer

Xinhao Mei, Xubo Liu, Qiushi Huang, Mark D. Plumbley, Wenwu Wang

- retweets: 197, favorites: 96 (07/23/2021 19:39:29)

- links: [abs](https://arxiv.org/abs/2107.09817) | [pdf](https://arxiv.org/pdf/2107.09817)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Audio captioning aims to automatically generate a natural language description of an audio clip. Most captioning models follow an encoder-decoder architecture, where the decoder predicts words based on the audio features extracted by the encoder. Convolutional neural networks (CNNs) and recurrent neural networks (RNNs) are often used as the audio encoder. However, CNNs can be limited in modelling temporal relationships among the time frames in an audio signal, while RNNs can be limited in modelling the long-range dependencies among the time frames. In this paper, we propose an Audio Captioning Transformer (ACT), which is a full Transformer network based on an encoder-decoder architecture and is totally convolution-free. The proposed method has a better ability to model the global information within an audio signal as well as capture temporal relationships between audio events. We evaluate our model on AudioCaps, which is the largest audio captioning dataset publicly available. Our model shows competitive performance compared to other state-of-the-art approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Audio Captioning Transformer<br>pdf: <a href="https://t.co/DlByzehFna">https://t.co/DlByzehFna</a><br>abs: <a href="https://t.co/x1eDCuupU8">https://t.co/x1eDCuupU8</a><br><br>a full Transformer network based on an encoder-decoder architecture and is totally convolution-free <a href="https://t.co/5QCWEUYrOW">pic.twitter.com/5QCWEUYrOW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1418008507757047811?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Group Contrastive Self-Supervised Learning on Graphs

Xinyi Xu, Cheng Deng, Yaochen Xie, Shuiwang Ji

- retweets: 130, favorites: 68 (07/23/2021 19:39:29)

- links: [abs](https://arxiv.org/abs/2107.09787) | [pdf](https://arxiv.org/pdf/2107.09787)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We study self-supervised learning on graphs using contrastive methods. A general scheme of prior methods is to optimize two-view representations of input graphs. In many studies, a single graph-level representation is computed as one of the contrastive objectives, capturing limited characteristics of graphs. We argue that contrasting graphs in multiple subspaces enables graph encoders to capture more abundant characteristics. To this end, we propose a group contrastive learning framework in this work. Our framework embeds the given graph into multiple subspaces, of which each representation is prompted to encode specific characteristics of graphs. To learn diverse and informative representations, we develop principled objectives that enable us to capture the relations among both intra-space and inter-space representations in groups. Under the proposed framework, we further develop an attention-based representor function to compute representations that capture different substructures of a given graph. Built upon our framework, we extend two current methods into GroupCL and GroupIG, equipped with the proposed objective. Comprehensive experimental results show our framework achieves a promising boost in performance on a variety of datasets. In addition, our qualitative results show that features generated from our representor successfully capture various specific characteristics of graphs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper on SSL for Graphs:<br>Group Contrastive Self-Supervised Learning on Graphs<br><br>Paper link: <a href="https://t.co/jew3RpCXu1">https://t.co/jew3RpCXu1</a></p>&mdash; Shuiwang Ji (@ShuiwangJi) <a href="https://twitter.com/ShuiwangJi/status/1418038418882416642?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Neural Fixed-Point Acceleration for Convex Optimization

Shobha Venkataraman, Brandon Amos

- retweets: 111, favorites: 81 (07/23/2021 19:39:29)

- links: [abs](https://arxiv.org/abs/2107.10254) | [pdf](https://arxiv.org/pdf/2107.10254)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [math.OC](https://arxiv.org/list/math.OC/recent)

Fixed-point iterations are at the heart of numerical computing and are often a computational bottleneck in real-time applications, which typically instead need a fast solution of moderate accuracy. Classical acceleration methods for fixed-point problems focus on designing algorithms with theoretical guarantees that apply to any fixed-point problem. We present neural fixed-point acceleration, a framework to automatically learn to accelerate convex fixed-point problems that are drawn from a distribution, using ideas from meta-learning and classical acceleration algorithms. We apply our framework to SCS, the state-of-the-art solver for convex cone programming, and design models and loss functions to overcome the challenges of learning over unrolled optimization and acceleration instabilities. Our work brings neural acceleration into any optimization problem expressible with CVXPY. The source code behind this paper is available at https://github.com/facebookresearch/neural-scs

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our <a href="https://twitter.com/hashtag/automl?src=hash&amp;ref_src=twsrc%5Etfw">#automl</a> workshop at <a href="https://twitter.com/icmlconf?ref_src=twsrc%5Etfw">@icmlconf</a>,  <a href="https://twitter.com/trailsofwater?ref_src=twsrc%5Etfw">@trailsofwater</a> and I look into learning better convex optimization solvers.<br><br>Paper: <a href="https://t.co/S8Xz4aO6yU">https://t.co/S8Xz4aO6yU</a><a href="https://twitter.com/PyTorch?ref_src=twsrc%5Etfw">@PyTorch</a> Code: <a href="https://t.co/fY1VKP0OYu">https://t.co/fY1VKP0OYu</a> <a href="https://t.co/Muht8z38sW">pic.twitter.com/Muht8z38sW</a></p>&mdash; Brandon Amos (@brandondamos) <a href="https://twitter.com/brandondamos/status/1418236182463459337?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Fixed-Point Acceleration for Convex Optimization<br>pdf: <a href="https://t.co/rnizBX9SY4">https://t.co/rnizBX9SY4</a><br>github: <a href="https://t.co/gnzg3vBYTN">https://t.co/gnzg3vBYTN</a><br><br>a framework to automatically learn to accelerate convex<br>fixed-point problems that are drawn from a distribution <a href="https://t.co/2jgtSa2FAE">pic.twitter.com/2jgtSa2FAE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1418010708382568448?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. What Do You Get When You Cross Beam Search with Nucleus Sampling?

Uri Shaham, Omer Levy

- retweets: 158, favorites: 31 (07/23/2021 19:39:29)

- links: [abs](https://arxiv.org/abs/2107.09729) | [pdf](https://arxiv.org/pdf/2107.09729)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We combine beam search with the probabilistic pruning technique of nucleus sampling to create two deterministic nucleus search algorithms for natural language generation. The first algorithm, p-exact search, locally prunes the next-token distribution and performs an exact search over the remaining space. The second algorithm, dynamic beam search, shrinks and expands the beam size according to the entropy of the candidate's probability distribution. Despite the probabilistic intuition behind nucleus search, experiments on machine translation and summarization benchmarks show that both algorithms reach the same performance levels as standard beam search.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What do you get when you cross beam search and nucleus sampling?<br><br>We explore two variants of &quot;nucleus search&quot; that integrate top-p pruning into deterministic search.<br>Can they find some gems that beam search misses?👇<a href="https://t.co/ILq9AYbdOo">https://t.co/ILq9AYbdOo</a><br>with <a href="https://twitter.com/omerlevy_?ref_src=twsrc%5Etfw">@omerlevy_</a></p>&mdash; Uri Shaham (@Uri_Shaham) <a href="https://twitter.com/Uri_Shaham/status/1418228961050435584?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. VQMIVC: Vector Quantization and Mutual Information-Based Unsupervised  Speech Representation Disentanglement for One-shot Voice Conversion

Disong Wang, Liqun Deng, Yu Ting Yeung, Xiao Chen, Xunying Liu, Helen Meng

- retweets: 64, favorites: 29 (07/23/2021 19:39:29)

- links: [abs](https://arxiv.org/abs/2106.10132) | [pdf](https://arxiv.org/pdf/2106.10132)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.SP](https://arxiv.org/list/eess.SP/recent)

One-shot voice conversion (VC), which performs conversion across arbitrary speakers with only a single target-speaker utterance for reference, can be effectively achieved by speech representation disentanglement. Existing work generally ignores the correlation between different speech representations during training, which causes leakage of content information into the speaker representation and thus degrades VC performance. To alleviate this issue, we employ vector quantization (VQ) for content encoding and introduce mutual information (MI) as the correlation metric during training, to achieve proper disentanglement of content, speaker and pitch representations, by reducing their inter-dependencies in an unsupervised manner. Experimental results reflect the superiority of the proposed method in learning effective disentangled speech representations for retaining source linguistic content and intonation variations, while capturing target speaker characteristics. In doing so, the proposed approach achieves higher speech naturalness and speaker similarity than current state-of-the-art one-shot VC systems. Our code, pre-trained models and demo are available at https://github.com/Wendison/VQMIVC.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VQMIVC: Vector Quantization and Mutual Information-Based Unsupervised Speech Representation Disentanglement for One-shot Voice Conversion<br>pdf: <a href="https://t.co/AzUXyeFyLZ">https://t.co/AzUXyeFyLZ</a><br>abs: <a href="https://t.co/OATW8a19AB">https://t.co/OATW8a19AB</a><br>github: <a href="https://t.co/jWr2xWhWz0">https://t.co/jWr2xWhWz0</a> <a href="https://t.co/6L86YeGW09">pic.twitter.com/6L86YeGW09</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1418031041793699847?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Discovering Latent Causal Variables via Mechanism Sparsity: A New  Principle for Nonlinear ICA

Sébastien Lachapelle, Pau Rodríguez López, Rémi Le Priol, Alexandre Lacoste, Simon Lacoste-Julien

- retweets: 65, favorites: 18 (07/23/2021 19:39:29)

- links: [abs](https://arxiv.org/abs/2107.10098) | [pdf](https://arxiv.org/pdf/2107.10098)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

It can be argued that finding an interpretable low-dimensional representation of a potentially high-dimensional phenomenon is central to the scientific enterprise. Independent component analysis (ICA) refers to an ensemble of methods which formalize this goal and provide estimation procedure for practical application. This work proposes mechanism sparsity regularization as a new principle to achieve nonlinear ICA when latent factors depend sparsely on observed auxiliary variables and/or past latent factors. We show that the latent variables can be recovered up to a permutation if one regularizes the latent mechanisms to be sparse and if some graphical criterion is satisfied by the data generating process. As a special case, our framework shows how one can leverage unknown-target interventions on the latent factors to disentangle them, thus drawing further connections between ICA and causality. We validate our theoretical results with toy experiments.




# 8. Optimal Rates for Nonparametric Density Estimation under Communication  Constraints

Jayadev Acharya, Clément L. Canonne, Aditya Vikram Singh, Himanshu Tyagi

- retweets: 22, favorites: 39 (07/23/2021 19:39:29)

- links: [abs](https://arxiv.org/abs/2107.10078) | [pdf](https://arxiv.org/pdf/2107.10078)
- [math.ST](https://arxiv.org/list/math.ST/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent)

We consider density estimation for Besov spaces when each sample is quantized to only a limited number of bits. We provide a noninteractive adaptive estimator that exploits the sparsity of wavelet bases, along with a simulate-and-infer technique from parametric estimation under communication constraints. We show that our estimator is nearly rate-optimal by deriving minimax lower bounds that hold even when interactive protocols are allowed. Interestingly, while our wavelet-based estimator is almost rate-optimal for Sobolev spaces as well, it is unclear whether the standard Fourier basis, which arise naturally for those spaces, can be used to achieve the same performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper on nonparametric estimation under communication constraints is up! In which <a href="https://twitter.com/AcharyaJayadev?ref_src=twsrc%5Etfw">@AcharyaJayadev</a>, Aditya Vikram Singh, <a href="https://twitter.com/hstyagi?ref_src=twsrc%5Etfw">@hstyagi</a> and I (but mostly Aditya!) derive optimal* rates for adaptive density estimation over Besov spaces.<br><br>📝<a href="https://t.co/mQ1zZRAit8">https://t.co/mQ1zZRAit8</a><br><br>* up to log factors <a href="https://t.co/h1hVwgPxIL">https://t.co/h1hVwgPxIL</a> <a href="https://t.co/Qnu9A7BeSL">pic.twitter.com/Qnu9A7BeSL</a></p>&mdash; Clément Canonne (@ccanonne_) <a href="https://twitter.com/ccanonne_/status/1418006863350681601?ref_src=twsrc%5Etfw">July 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



