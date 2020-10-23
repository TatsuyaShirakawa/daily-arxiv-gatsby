---
title: Hot Papers 2020-10-22
date: 2020-10-23T09:53:01.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-22"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-22"
socialImage: "/media/flying-marine.jpg"

---

# 1. Logistic $Q$-Learning

Joan Bas-Serrano, Sebastian Curi, Andreas Krause, Gergely Neu

- retweets: 9025, favorites: 498 (10/23/2020 09:53:01)

- links: [abs](https://arxiv.org/abs/2010.11151) | [pdf](https://arxiv.org/pdf/2010.11151)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We propose a new reinforcement learning algorithm derived from a regularized linear-programming formulation of optimal control in MDPs. The method is closely related to the classic Relative Entropy Policy Search (REPS) algorithm of Peters et al. (2010), with the key difference that our method introduces a Q-function that enables efficient exact model-free implementation. The main feature of our algorithm (called QREPS) is a convex loss function for policy evaluation that serves as a theoretically sound alternative to the widely used squared Bellman error. We provide a practical saddle-point optimization method for minimizing this loss function and provide an error-propagation analysis that relates the quality of the individual updates to the performance of the output policy. Finally, we demonstrate the effectiveness of our method on a range of benchmark problems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RL folks, meet your newest friend: THE LOGISTIC BELLMAN ERROR<br><br>A convex loss function derived from first principles of MDP theory that leads to practical RL algorithms that can be implemented without *any* approximation of the theory.<br><br>Preprint: <a href="https://t.co/guQg8VwFRR">https://t.co/guQg8VwFRR</a><br>🧵👇 <br>1/18 <a href="https://t.co/Wo1Sz6BkIo">pic.twitter.com/Wo1Sz6BkIo</a></p>&mdash; Gergely Neu (@neu_rips) <a href="https://twitter.com/neu_rips/status/1319182610728423424?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Runtime Safety Assurance Using Reinforcement Learning

Christopher Lazarus, James G. Lopez, Mykel J. Kochenderfer

- retweets: 662, favorites: 20 (10/23/2020 09:53:02)

- links: [abs](https://arxiv.org/abs/2010.10618) | [pdf](https://arxiv.org/pdf/2010.10618)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent)

The airworthiness and safety of a non-pedigreed autopilot must be verified, but the cost to formally do so can be prohibitive. We can bypass formal verification of non-pedigreed components by incorporating Runtime Safety Assurance (RTSA) as mechanism to ensure safety. RTSA consists of a meta-controller that observes the inputs and outputs of a non-pedigreed component and verifies formally specified behavior as the system operates. When the system is triggered, a verified recovery controller is deployed. Recovery controllers are designed to be safe but very likely disruptive to the operational objective of the system, and thus RTSA systems must balance safety and efficiency. The objective of this paper is to design a meta-controller capable of identifying unsafe situations with high accuracy. High dimensional and non-linear dynamics in which modern controllers are deployed along with the black-box nature of the nominal controllers make this a difficult problem. Current approaches rely heavily on domain expertise and human engineering. We frame the design of RTSA with the Markov decision process (MDP) framework and use reinforcement learning (RL) to solve it. Our learned meta-controller consistently exhibits superior performance in our experiments compared to our baseline, human engineered approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Runtime Safety Assurance Using Reinforcement Learning. <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/Java?src=hash&amp;ref_src=twsrc%5Etfw">#Java</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/TensorFlow?src=hash&amp;ref_src=twsrc%5Etfw">#TensorFlow</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a><a href="https://t.co/WLGQVo1FFJ">https://t.co/WLGQVo1FFJ</a> <a href="https://t.co/tW6M9mc0PH">pic.twitter.com/tW6M9mc0PH</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1319362367503933441?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Contrastive Learning of General-Purpose Audio Representations

Aaqib Saeed, David Grangier, Neil Zeghidour

- retweets: 514, favorites: 158 (10/23/2020 09:53:02)

- links: [abs](https://arxiv.org/abs/2010.10915) | [pdf](https://arxiv.org/pdf/2010.10915)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We introduce COLA, a self-supervised pre-training approach for learning a general-purpose representation of audio. Our approach is based on contrastive learning: it learns a representation which assigns high similarity to audio segments extracted from the same recording while assigning lower similarity to segments from different recordings. We build on top of recent advances in contrastive learning for computer vision and reinforcement learning to design a lightweight, easy-to-implement self-supervised model of audio. We pre-train embeddings on the large-scale Audioset database and transfer these representations to 9 diverse classification tasks, including speech, music, animal sounds, and acoustic scenes. We show that despite its simplicity, our method significantly outperforms previous self-supervised systems. We furthermore conduct ablation studies to identify key design choices and release a library to pre-train and fine-tune COLA models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">With my <a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> intern <a href="https://twitter.com/aaqib_saeed?ref_src=twsrc%5Etfw">@aaqib_saeed</a> and <a href="https://twitter.com/GrangierDavid?ref_src=twsrc%5Etfw">@GrangierDavid</a> we introduce COLA, a simple, yet strong contrastive model for learning general-purpose audio representations.<br>* No augmentation<br>* No momentum<br>* 9 downstream tasks<br>Paper: <a href="https://t.co/ZzLCGi3H84">https://t.co/ZzLCGi3H84</a><br>Code: <a href="https://t.co/F9lvowEiXA">https://t.co/F9lvowEiXA</a> <a href="https://t.co/9qLjHYZ4kn">pic.twitter.com/9qLjHYZ4kn</a></p>&mdash; Neil Zeghidour (@neilzegh) <a href="https://twitter.com/neilzegh/status/1319189254728306689?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contrastive Learning of General-Purpose Audio Representations<br>pdf: <a href="https://t.co/TEDughTOnz">https://t.co/TEDughTOnz</a><br>abs: <a href="https://t.co/03v1AzoTdD">https://t.co/03v1AzoTdD</a><br>github: <a href="https://t.co/zXCgX1YCtO">https://t.co/zXCgX1YCtO</a> <a href="https://t.co/OisJfFiYb4">pic.twitter.com/OisJfFiYb4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1319100222069460992?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. One Model to Reconstruct Them All: A Novel Way to Use the Stochastic  Noise in StyleGAN

Christian Bartz, Joseph Bethge, Haojin Yang, Christoph Meinel

- retweets: 240, favorites: 70 (10/23/2020 09:53:02)

- links: [abs](https://arxiv.org/abs/2010.11113) | [pdf](https://arxiv.org/pdf/2010.11113)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Generative Adversarial Networks (GANs) have achieved state-of-the-art performance for several image generation and manipulation tasks. Different works have improved the limited understanding of the latent space of GANs by embedding images into specific GAN architectures to reconstruct the original images. We present a novel StyleGAN-based autoencoder architecture, which can reconstruct images with very high quality across several data domains. We demonstrate a previously unknown grade of generalizablility by training the encoder and decoder independently and on different datasets. Furthermore, we provide new insights about the significance and capabilities of noise inputs of the well-known StyleGAN architecture. Our proposed architecture can handle up to 40 images per second on a single GPU, which is approximately 28x faster than previous approaches. Finally, our model also shows promising results, when compared to the state-of-the-art on the image denoising task, although it was not explicitly designed for this task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">One Model to Reconstruct Them All: A Novel Way to Use the Stochastic Noise in StyleGAN<br>pdf: <a href="https://t.co/rh5FXOmCoZ">https://t.co/rh5FXOmCoZ</a><br>abs: <a href="https://t.co/LMAsiapQHD">https://t.co/LMAsiapQHD</a><br>github: <a href="https://t.co/EW5nJmBIp7">https://t.co/EW5nJmBIp7</a> <a href="https://t.co/J7itfaDL2m">pic.twitter.com/J7itfaDL2m</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1319092638298050563?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Analyzing the Source and Target Contributions to Predictions in Neural  Machine Translation

Elena Voita, Rico Sennrich, Ivan Titov

- retweets: 225, favorites: 69 (10/23/2020 09:53:02)

- links: [abs](https://arxiv.org/abs/2010.10907) | [pdf](https://arxiv.org/pdf/2010.10907)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

In Neural Machine Translation (and, more generally, conditional language modeling), the generation of a target token is influenced by two types of context: the source and the prefix of the target sequence. While many attempts to understand the internal workings of NMT models have been made, none of them explicitly evaluates relative source and target contributions to a generation decision. We argue that this relative contribution can be evaluated by adopting a variant of Layerwise Relevance Propagation (LRP). Its underlying 'conservation principle' makes relevance propagation unique: differently from other methods, it evaluates not an abstract quantity reflecting token importance, but the proportion of each token's influence. We extend LRP to the Transformer and conduct an analysis of NMT models which explicitly evaluates the source and target relative contributions to the generation process. We analyze changes in these contributions when conditioning on different types of prefixes, when varying the training objective or the amount of training data, and during the training process. We find that models trained with more data tend to rely on source information more and to have more sharp token contributions; the training process is non-monotonic with several stages of different nature.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">[1/4] Analyzing Source and Target Contributions to NMT Predictions - new work with <a href="https://twitter.com/iatitov?ref_src=twsrc%5Etfw">@iatitov</a> and <a href="https://twitter.com/RicoSennrich?ref_src=twsrc%5Etfw">@RicoSennrich</a>!<br><br>What influences the predictions in NMT: the source or the target prefix? We measure and find out!<br><br>Paper: <a href="https://t.co/x07tmYMpBW">https://t.co/x07tmYMpBW</a><br>Blog: <a href="https://t.co/uRCpSuxjKr">https://t.co/uRCpSuxjKr</a> <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> <a href="https://t.co/Out9HtjHY2">pic.twitter.com/Out9HtjHY2</a></p>&mdash; Lena Voita (@lena_voita) <a href="https://twitter.com/lena_voita/status/1319275723807285248?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Riemannian Langevin Algorithm for Solving Semidefinite Programs

Mufan, Murat A. Erdogdu

- retweets: 156, favorites: 135 (10/23/2020 09:53:02)

- links: [abs](https://arxiv.org/abs/2010.11176) | [pdf](https://arxiv.org/pdf/2010.11176)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.OC](https://arxiv.org/list/math.OC/recent)

We propose a Langevin diffusion-based algorithm for non-convex optimization and sampling on a product manifold of spheres. Under a logarithmic Sobolev inequality, we establish a guarantee for finite iteration convergence to the Gibbs distribution in terms of Kullback-Leibler divergence. We show that with an appropriate temperature choice, the suboptimality gap to the global minimum is guaranteed to be arbitrarily small with high probability.   As an application, we analyze the proposed Langevin algorithm for solving the Burer-Monteiro relaxation of a semidefinite program (SDP). In particular, we establish a logarithmic Sobolev inequality for the Burer-Monteiro problem when there are no spurious local minima; hence implying a fast escape from saddle points. Combining the results, we then provide a global optimality guarantee for the SDP and the Max-Cut problem. More precisely, we show the Langevin algorithm achieves $\epsilon$-multiplicative accuracy with high probability in $\widetilde{\Omega}( n^2 \epsilon^{-3} )$ iterations, where $n$ is the size of the cost matrix.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">If only my first arXiv paper was this impressive!<a href="https://t.co/4ljX8XLBEj">https://t.co/4ljX8XLBEj</a><br><br>First author is <a href="https://twitter.com/mufan_li?ref_src=twsrc%5Etfw">@mufan_li</a> who works with <a href="https://twitter.com/MuratAErdogdu?ref_src=twsrc%5Etfw">@MuratAErdogdu</a> and me at <a href="https://twitter.com/UofTStatSci?ref_src=twsrc%5Etfw">@UofTStatSci</a> and <a href="https://twitter.com/VectorInst?ref_src=twsrc%5Etfw">@VectorInst</a>. <a href="https://t.co/FkAKmf0m3N">pic.twitter.com/FkAKmf0m3N</a></p>&mdash; Daniel Roy (@roydanroy) <a href="https://twitter.com/roydanroy/status/1319094604797456385?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Complex data labeling with deep learning methods: Lessons from fisheries  acoustics

J.M.A.Sarr, T. Brochier, P.Brehmer, Y.Perrot, A.Bah, A.Sarré, M.A.Jeyid, M.Sidibeh, S.El Ayoub

- retweets: 90, favorites: 25 (10/23/2020 09:53:02)

- links: [abs](https://arxiv.org/abs/2010.11010) | [pdf](https://arxiv.org/pdf/2010.11010)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Quantitative and qualitative analysis of acoustic backscattered signals from the seabed bottom to the sea surface is used worldwide for fish stocks assessment and marine ecosystem monitoring. Huge amounts of raw data are collected yet require tedious expert labeling. This paper focuses on a case study where the ground truth labels are non-obvious: echograms labeling, which is time-consuming and critical for the quality of fisheries and ecological analysis. We investigate how these tasks can benefit from supervised learning algorithms and demonstrate that convolutional neural networks trained with non-stationary datasets can be used to stress parts of a new dataset needing human expert correction. Further development of this approach paves the way toward a standardization of the labeling process in fisheries acoustics and is a good case study for non-obvious data labeling processes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our work on active acoustics data labeling with supervised learning finally got accepted at ISA Transaction: <a href="https://t.co/vkjXdZmoNR">https://t.co/vkjXdZmoNR</a>. An open access version has also been deposed on arXiv: <a href="https://t.co/iRyNpr5vTv">https://t.co/iRyNpr5vTv</a></p>&mdash; JM Amath Sarr (@jmamathsarr) <a href="https://twitter.com/jmamathsarr/status/1319250839698087936?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. RECONSIDER: Re-Ranking using Span-Focused Cross-Attention for Open  Domain Question Answering

Srinivasan Iyer, Sewon Min, Yashar Mehdad, Wen-tau Yih

- retweets: 74, favorites: 30 (10/23/2020 09:53:03)

- links: [abs](https://arxiv.org/abs/2010.10757) | [pdf](https://arxiv.org/pdf/2010.10757)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

State-of-the-art Machine Reading Comprehension (MRC) models for Open-domain Question Answering (QA) are typically trained for span selection using distantly supervised positive examples and heuristically retrieved negative examples. This training scheme possibly explains empirical observations that these models achieve a high recall amongst their top few predictions, but a low overall accuracy, motivating the need for answer re-ranking. We develop a simple and effective re-ranking approach (RECONSIDER) for span-extraction tasks, that improves upon the performance of large pre-trained MRC models. RECONSIDER is trained on positive and negative examples extracted from high confidence predictions of MRC models, and uses in-passage span annotations to perform span-focused re-ranking over a smaller candidate set. As a result, RECONSIDER learns to eliminate close false positive passages, and achieves a new state of the art on four QA tasks, including 45.5% Exact Match accuracy on Natural Questions with real user questions, and 61.7% on TriviaQA.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">(1/2) New Paper! A re-ranking method to improve the performance of distantly supervised extractive open-domain QA models like DPR. We improve over DPR on 4 open domain QA datasets. — with <a href="https://twitter.com/sewon__min?ref_src=twsrc%5Etfw">@sewon__min</a> <a href="https://twitter.com/scottyih?ref_src=twsrc%5Etfw">@scottyih</a> <a href="https://twitter.com/YasharMehdad?ref_src=twsrc%5Etfw">@YasharMehdad</a><a href="https://t.co/9ylrER0y9W">https://t.co/9ylrER0y9W</a></p>&mdash; Srini Iyer (@sriniiyer88) <a href="https://twitter.com/sriniiyer88/status/1319350226243645441?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Emformer: Efficient Memory Transformer Based Acoustic Model For Low  Latency Streaming Speech Recognition

Yangyang Shi, Yongqiang Wang, Chunyang Wu, Ching-Feng Yeh, Julian Chan, Frank Zhang, Duc Le, Mike Seltzer

- retweets: 51, favorites: 36 (10/23/2020 09:53:03)

- links: [abs](https://arxiv.org/abs/2010.10759) | [pdf](https://arxiv.org/pdf/2010.10759)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper proposes an efficient memory transformer Emformer for low latency streaming speech recognition. In Emformer, the long-range history context is distilled into an augmented memory bank to reduce self-attention's computation complexity. A cache mechanism saves the computation for the key and value in self-attention for the left context. Emformer applies a parallelized block processing in training to support low latency models. We carry out experiments on benchmark LibriSpeech data. Under average latency of 960 ms, Emformer gets WER $2.50\%$ on test-clean and $5.62\%$ on test-other. Comparing with a strong baseline augmented memory transformer (AM-TRF), Emformer gets $4.6$ folds training speedup and $18\%$ relative real-time factor (RTF) reduction in decoding with relative WER reduction $17\%$ on test-clean and $9\%$ on test-other. For a low latency scenario with an average latency of 80 ms, Emformer achieves WER $3.01\%$ on test-clean and $7.09\%$ on test-other. Comparing with the LSTM baseline with the same latency and model size, Emformer gets relative WER reduction $9\%$ and $16\%$ on test-clean and test-other, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition<br>pdf: <a href="https://t.co/sMZJ7OxVIi">https://t.co/sMZJ7OxVIi</a><br>abs: <a href="https://t.co/c9O2KGodiW">https://t.co/c9O2KGodiW</a> <a href="https://t.co/tvOFfZZ1RT">pic.twitter.com/tvOFfZZ1RT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1319078910865399810?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Image-Driven Furniture Style for Interactive 3D Scene Modeling

Tomer Weiss, Ilkay Yildiz, Nitin Agarwal, Esra Ataer-Cansizoglu, Jae-Woo Choi

- retweets: 56, favorites: 29 (10/23/2020 09:53:03)

- links: [abs](https://arxiv.org/abs/2010.10557) | [pdf](https://arxiv.org/pdf/2010.10557)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Creating realistic styled spaces is a complex task, which involves design know-how for what furniture pieces go well together. Interior style follows abstract rules involving color, geometry and other visual elements. Following such rules, users manually select similar-style items from large repositories of 3D furniture models, a process which is both laborious and time-consuming. We propose a method for fast-tracking style-similarity tasks, by learning a furniture's style-compatibility from interior scene images. Such images contain more style information than images depicting single furniture. To understand style, we train a deep learning network on a classification task. Based on image embeddings extracted from our network, we measure stylistic compatibility of furniture. We demonstrate our method with several 3D model style-compatibility results, and with an interactive system for modeling style-consistent scenes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Image-Driven Furniture Style for Interactive 3D Scene Modeling<br>pdf: <a href="https://t.co/Inse7lotHD">https://t.co/Inse7lotHD</a><br>abs: <a href="https://t.co/dtHzynunwD">https://t.co/dtHzynunwD</a> <a href="https://t.co/SEcmgDSdyj">pic.twitter.com/SEcmgDSdyj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1319083790036312069?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. A Survey on Deep Learning and Explainability for Automatic Image-based  Medical Report Generation

Pablo Messina, Pablo Pino, Denis Parra, Alvaro Soto, Cecilia Besa, Sergio Uribe, Marcelo andía, Cristian Tejos, Claudia Prieto, Daniel Capurro

- retweets: 48, favorites: 22 (10/23/2020 09:53:03)

- links: [abs](https://arxiv.org/abs/2010.10563) | [pdf](https://arxiv.org/pdf/2010.10563)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Every year physicians face an increasing demand of image-based diagnosis from patients, a problem that can be addressed with recent artificial intelligence methods. In this context, we survey works in the area of automatic report generation from medical images, with emphasis on methods using deep neural networks, with respect to: (1) Datasets, (2) Architecture Design, (3) Explainability and (4) Evaluation Metrics. Our survey identifies interesting developments, but also remaining challenges. Among them, the current evaluation of generated reports is especially weak, since it mostly relies on traditional Natural Language Processing (NLP) metrics, which do not accurately capture medical correctness.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Survey on Deep Learning and Explainability for Automatic Image-based Medical Report Gen... <a href="https://t.co/Bg5KF8nvPi">https://t.co/Bg5KF8nvPi</a> <a href="https://t.co/orJKgOiHmX">pic.twitter.com/orJKgOiHmX</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1319233526844125185?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Grapheme or phoneme? An Analysis of Tacotron's Embedded Representations

Antoine Perquin, Erica Cooper, Junichi Yamagishi

- retweets: 42, favorites: 14 (10/23/2020 09:53:03)

- links: [abs](https://arxiv.org/abs/2010.10694) | [pdf](https://arxiv.org/pdf/2010.10694)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

End-to-end models, particularly Tacotron-based ones, are currently a popular solution for text-to-speech synthesis. They allow the production of high-quality synthesized speech with little to no text preprocessing. Phoneme inputs are usually preferred over graphemes in order to limit the amount of pronunciation errors. In this work we show that, in the case of a well-curated French dataset, graphemes can be used as input without increasing the amount of pronunciation errors. Furthermore, we perform an analysis of the representation learned by the Tacotron model and show that the contextual grapheme embeddings encode phoneme information, and that they can be used for grapheme-to-phoneme conversion and phoneme control of synthetic speech.




# 13. Learning Disentangled Phone and Speaker Representations in a  Semi-Supervised VQ-VAE Paradigm

Jennifer Williams, Yi Zhao, Erica Cooper, Junichi Yamagishi

- retweets: 42, favorites: 12 (10/23/2020 09:53:03)

- links: [abs](https://arxiv.org/abs/2010.10727) | [pdf](https://arxiv.org/pdf/2010.10727)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

We present a new approach to disentangle speaker voice and phone content by introducing new components to the VQ-VAE architecture for speech synthesis. The original VQ-VAE does not generalize well to unseen speakers or content. To alleviate this problem, we have incorporated a speaker encoder and speaker VQ codebook that learns global speaker characteristics entirely separate from the existing sub-phone codebooks. We also compare two training methods: self-supervised with global conditions and semi-supervised with speaker labels. Adding a speaker VQ component improves objective measures of speech synthesis quality (estimated MOS, speaker similarity, ASR-based intelligibility) and provides learned representations that are meaningful. Our speaker VQ codebook indices can be used in a simple speaker diarization task and perform slightly better than an x-vector baseline. Additionally, phones can be recognized from sub-phone VQ codebook indices in our semi-supervised VQ-VAE better than self-supervised with global conditions.



