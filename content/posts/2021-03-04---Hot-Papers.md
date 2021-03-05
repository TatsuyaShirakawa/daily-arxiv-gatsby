---
title: Hot Papers 2021-03-04
date: 2021-03-05T11:11:18.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-04"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-04"
socialImage: "/media/flying-marine.jpg"

---

# 1. Neural 3D Video Synthesis

Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Zhaoyang Lv

- retweets: 5719, favorites: 3 (03/05/2021 11:11:18)

- links: [abs](https://arxiv.org/abs/2103.02597) | [pdf](https://arxiv.org/pdf/2103.02597)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We propose a novel approach for 3D video synthesis that is able to represent multi-view video recordings of a dynamic real-world scene in a compact, yet expressive representation that enables high-quality view synthesis and motion interpolation. Our approach takes the high quality and compactness of static neural radiance fields in a new direction: to a model-free, dynamic setting. At the core of our approach is a novel time-conditioned neural radiance fields that represents scene dynamics using a set of compact latent codes. To exploit the fact that changes between adjacent frames of a video are typically small and locally consistent, we propose two novel strategies for efficient training of our neural network: 1) An efficient hierarchical training scheme, and 2) an importance sampling strategy that selects the next rays for training based on the temporal variation of the input videos. In combination, these two strategies significantly boost the training speed, lead to fast convergence of the training process, and enable high quality results. Our learned representation is highly compact and able to represent a 10 second 30 FPS multi-view video recording by 18 cameras with a model size of just 28MB. We demonstrate that our method can render high-fidelity wide-angle novel views at over 1K resolution, even for highly complex and dynamic scenes. We perform an extensive qualitative and quantitative evaluation that shows that our approach outperforms the current state of the art. We include additional video and information at: https://neural-3d-video.github.io/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are proud to introduce &quot;Neural 3D Video Synthesis&quot;, our recent team efforts in FRL Research that can enable high-fidelity high resolution wide-angle 3D video synthesis.<br><br>Arxiv: <a href="https://t.co/TvhgYFENjF">https://t.co/TvhgYFENjF</a><br>Page (with full video): <a href="https://t.co/gbsC6R8HEh">https://t.co/gbsC6R8HEh</a> <a href="https://t.co/eeseP7Cxej">pic.twitter.com/eeseP7Cxej</a></p>&mdash; Zhaoyang Lv (@LvZhaoyang) <a href="https://twitter.com/LvZhaoyang/status/1367302037440847878?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Self-supervised Pretraining of Visual Features in the Wild

Priya Goyal, Mathilde Caron, Benjamin Lefaudeux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh, Vitaliy Liptchinsky, Ishan Misra, Armand Joulin, Piotr Bojanowski

- retweets: 1504, favorites: 312 (03/05/2021 11:11:18)

- links: [abs](https://arxiv.org/abs/2103.01988) | [pdf](https://arxiv.org/pdf/2103.01988)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Recently, self-supervised learning methods like MoCo, SimCLR, BYOL and SwAV have reduced the gap with supervised methods. These results have been achieved in a control environment, that is the highly curated ImageNet dataset. However, the premise of self-supervised learning is that it can learn from any random image and from any unbounded dataset. In this work, we explore if self-supervision lives to its expectation by training large models on random, uncurated images with no supervision. Our final SElf-supERvised (SEER) model, a RegNetY with 1.3B parameters trained on 1B random images with 512 GPUs achieves 84.2% top-1 accuracy, surpassing the best self-supervised pretrained model by 1% and confirming that self-supervised learning works in a real world setting. Interestingly, we also observe that self-supervised models are good few-shot learners achieving 77.9% top-1 with access to only 10% of ImageNet. Code: https://github.com/facebookresearch/vissl

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-supervised Pretraining of Visual Features in the Wild<br><br>Their 1.3B model trained on 1B random images achieves 84.2% top-1 acc., confirming that self-supervised learning works in a real world setting. <br><br>Also works well as a few-shot learner.<a href="https://t.co/HNcnBgves7">https://t.co/HNcnBgves7</a> <a href="https://t.co/IkfxFiLg0F">pic.twitter.com/IkfxFiLg0F</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1367293720370941952?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-supervised Pretraining of Visual Features in the Wild<br>pdf: <a href="https://t.co/pcdNYgLXQA">https://t.co/pcdNYgLXQA</a><br>abs: <a href="https://t.co/m3GIDr1eeH">https://t.co/m3GIDr1eeH</a><br>github: <a href="https://t.co/VomSo258DO">https://t.co/VomSo258DO</a> <a href="https://t.co/86kQlV1vhL">pic.twitter.com/86kQlV1vhL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367291729603395585?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Random Feature Attention

Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith, Lingpeng Kong

- retweets: 464, favorites: 106 (03/05/2021 11:11:18)

- links: [abs](https://arxiv.org/abs/2103.02143) | [pdf](https://arxiv.org/pdf/2103.02143)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Transformers are state-of-the-art models for a variety of sequence modeling tasks. At their core is an attention function which models pairwise interactions between the inputs at every timestep. While attention is powerful, it does not scale efficiently to long sequences due to its quadratic time and space complexity in the sequence length. We propose RFA, a linear time and space attention that uses random feature methods to approximate the softmax function, and explore its application in transformers. RFA can be used as a drop-in replacement for conventional softmax attention and offers a straightforward way of learning with recency bias through an optional gating mechanism. Experiments on language modeling and machine translation demonstrate that RFA achieves similar or better performance compared to strong transformer baselines. In the machine translation experiment, RFA decodes twice as fast as a vanilla transformer. Compared to existing efficient transformer variants, RFA is competitive in terms of both accuracy and efficiency on three long text classification datasets. Our analysis shows that RFA's efficiency gains are especially notable on long sequences, suggesting that RFA will be particularly useful in tasks that require working with large inputs, fast decoding speed, or low memory footprints.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Random Feature Attention<br><br>Proposes RFA, a linear-complexity Transformer that random feature methods to approximate the softmax. Performs on par or better than vanilla attention on several standard LM benchmarks. <a href="https://t.co/cf0bXGaqSI">https://t.co/cf0bXGaqSI</a> <a href="https://t.co/m2g37iWRU9">pic.twitter.com/m2g37iWRU9</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1367292773716557826?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. House-GAN++: Generative Adversarial Layout Refinement Networks

Nelson Nauata, Sepidehsadat Hosseini, Kai-Hung Chang, Hang Chu, Chin-Yi Cheng, Yasutaka Furukawa

- retweets: 184, favorites: 90 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02574) | [pdf](https://arxiv.org/pdf/2103.02574)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper proposes a novel generative adversarial layout refinement network for automated floorplan generation. Our architecture is an integration of a graph-constrained relational GAN and a conditional GAN, where a previously generated layout becomes the next input constraint, enabling iterative refinement. A surprising discovery of our research is that a simple non-iterative training process, dubbed component-wise GT-conditioning, is effective in learning such a generator. The iterative generator also creates a new opportunity in further improving a metric of choice via meta-optimization techniques by controlling when to pass which input constraints during iterative layout refinement. Our qualitative and quantitative evaluation based on the three standard metrics demonstrate that the proposed system makes significant improvements over the current state-of-the-art, even competitive against the ground-truth floorplans, designed by professional architects.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">House-GAN++: Generative Adversarial Layout Refinement Networks<br>pdf: <a href="https://t.co/m2JB77BfAG">https://t.co/m2JB77BfAG</a><br>abs: <a href="https://t.co/AiMbkv0PT0">https://t.co/AiMbkv0PT0</a> <a href="https://t.co/SxBZB5vf2C">pic.twitter.com/SxBZB5vf2C</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367296473331879941?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Who Can Find My Devices? Security and Privacy of Apple's Crowd-Sourced  Bluetooth Location Tracking System

Alexander Heinrich, Milan Stute, Tim Kornhuber, Matthias Hollick

- retweets: 210, favorites: 44 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02282) | [pdf](https://arxiv.org/pdf/2103.02282)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.NI](https://arxiv.org/list/cs.NI/recent)

Overnight, Apple has turned its hundreds-of-million-device ecosystem into the world's largest crowd-sourced location tracking network called offline finding (OF). OF leverages online finder devices to detect the presence of missing offline devices using Bluetooth and report an approximate location back to the owner via the Internet. While OF is not the first system of its kind, it is the first to commit to strong privacy goals. In particular, OF aims to ensure finder anonymity, untrackability of owner devices, and confidentiality of location reports. This paper presents the first comprehensive security and privacy analysis of OF. To this end, we recover the specifications of the closed-source OF protocols by means of reverse engineering. We experimentally show that unauthorized access to the location reports allows for accurate device tracking and retrieving a user's top locations with an error in the order of 10 meters in urban areas. While we find that OF's design achieves its privacy goals, we discover two distinct design and implementation flaws that can lead to a location correlation attack and unauthorized access to the location history of the past seven days, which could deanonymize users. Apple has partially addressed the issues following our responsible disclosure. Finally, we make our research artifacts publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This looks really cool: this paper reverses Apple’s FindMy system and looks closely at its security. Still reading but the figures look exciting. <a href="https://t.co/am0ZMBlATq">https://t.co/am0ZMBlATq</a> <a href="https://t.co/HSI7ISLrH0">pic.twitter.com/HSI7ISLrH0</a></p>&mdash; Matthew Green (@matthew_d_green) <a href="https://twitter.com/matthew_d_green/status/1367495937396998147?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Energy-Based Learning for Scene Graph Generation

Mohammed Suhail, Abhay Mittal, Behjat Siddiquie, Chris Broaddus, Jayan Eledath, Gerard Medioni, Leonid Sigal

- retweets: 196, favorites: 53 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02221) | [pdf](https://arxiv.org/pdf/2103.02221)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Traditional scene graph generation methods are trained using cross-entropy losses that treat objects and relationships as independent entities. Such a formulation, however, ignores the structure in the output space, in an inherently structured prediction problem. In this work, we introduce a novel energy-based learning framework for generating scene graphs. The proposed formulation allows for efficiently incorporating the structure of scene graphs in the output space. This additional constraint in the learning framework acts as an inductive bias and allows models to learn efficiently from a small number of labels. We use the proposed energy-based framework to train existing state-of-the-art models and obtain a significant performance improvement, of up to 21% and 27%, on the Visual Genome and GQA benchmark datasets, respectively. Furthermore, we showcase the learning efficiency of the proposed framework by demonstrating superior performance in the zero- and few-shot settings where data is scarce.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Energy-Based Learning for Scene Graph Generation<br>pdf: <a href="https://t.co/y78KXQ5HVz">https://t.co/y78KXQ5HVz</a><br>abs: <a href="https://t.co/f0z273ibWP">https://t.co/f0z273ibWP</a><br>github: <a href="https://t.co/I8j9YiFEuM">https://t.co/I8j9YiFEuM</a> <a href="https://t.co/svWNJLVIEt">pic.twitter.com/svWNJLVIEt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367319815350874118?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Towards Open World Object Detection

K J Joseph, Salman Khan, Fahad Shahbaz Khan, Vineeth N Balasubramanian

- retweets: 89, favorites: 47 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02603) | [pdf](https://arxiv.org/pdf/2103.02603)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Humans have a natural instinct to identify unknown object instances in their environments. The intrinsic curiosity about these unknown instances aids in learning about them, when the corresponding knowledge is eventually available. This motivates us to propose a novel computer vision problem called: `Open World Object Detection', where a model is tasked to: 1) identify objects that have not been introduced to it as `unknown', without explicit supervision to do so, and 2) incrementally learn these identified unknown categories without forgetting previously learned classes, when the corresponding labels are progressively received. We formulate the problem, introduce a strong evaluation protocol and provide a novel solution, which we call ORE: Open World Object Detector, based on contrastive clustering and energy based unknown identification. Our experimental evaluation and ablation studies analyze the efficacy of ORE in achieving Open World objectives. As an interesting by-product, we find that identifying and characterizing unknown instances helps to reduce confusion in an incremental object detection setting, where we achieve state-of-the-art performance, with no extra methodological effort. We hope that our work will attract further research into this newly identified, yet crucial research direction.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Open World Object Detection<br>pdf: <a href="https://t.co/D4UdKDYOiN">https://t.co/D4UdKDYOiN</a><br>abs: <a href="https://t.co/8chdt0gpr9">https://t.co/8chdt0gpr9</a><br>github: <a href="https://t.co/K4S5VThz33">https://t.co/K4S5VThz33</a> <a href="https://t.co/mBUnidU8m9">pic.twitter.com/mBUnidU8m9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367300515613331459?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. On the Just-In-Time Discovery of Profit-Generating Transactions in DeFi  Protocols

Liyi Zhou, Kaihua Qin, Antoine Cully, Benjamin Livshits, Arthur Gervais

- retweets: 90, favorites: 34 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02228) | [pdf](https://arxiv.org/pdf/2103.02228)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

In this paper, we investigate two methods that allow us to automatically create profitable DeFi trades, one well-suited to arbitrage and the other applicable to more complicated settings. We first adopt the Bellman-Ford-Moore algorithm with DEFIPOSER-ARB and then create logical DeFi protocol models for a theorem prover in DEFIPOSER-SMT. While DEFIPOSER-ARB focuses on DeFi transactions that form a cycle and performs very well for arbitrage, DEFIPOSER-SMT can detect more complicated profitable transactions. We estimate that DEFIPOSER-ARB and DEFIPOSER-SMT can generate an average weekly revenue of 191.48ETH (76,592USD) and 72.44ETH (28,976USD) respectively, with the highest transaction revenue being 81.31ETH(32,524USD) and22.40ETH (8,960USD) respectively. We further show that DEFIPOSER-SMT finds the known economic bZx attack from February 2020, which yields 0.48M USD. Our forensic investigations show that this opportunity existed for 69 days and could have yielded more revenue if exploited one day earlier. Our evaluation spans 150 days, given 96 DeFi protocol actions, and 25 assets.   Looking beyond the financial gains mentioned above, forks deteriorate the blockchain consensus security, as they increase the risks of double-spending and selfish mining. We explore the implications of DEFIPOSER-ARB and DEFIPOSER-SMT on blockchain consensus. Specifically, we show that the trades identified by our tools exceed the Ethereum block reward by up to 874x. Given optimal adversarial strategies provided by a Markov Decision Process (MDP), we quantify the value threshold at which a profitable transaction qualifies as Miner ExtractableValue (MEV) and would incentivize MEV-aware miners to fork the blockchain. For instance, we find that on Ethereum, a miner with a hash rate of 10% would fork the blockchain if an MEV opportunity exceeds 4x the block reward.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New <a href="https://twitter.com/IEEESSP?ref_src=twsrc%5Etfw">@IEEESSP</a>&#39;21 paper on how to discover automagically profitable transactions in the intertwined DeFi graph: <a href="https://t.co/Qstj3qPI6D">https://t.co/Qstj3qPI6D</a> with <a href="https://twitter.com/lzhou1110?ref_src=twsrc%5Etfw">@lzhou1110</a> <a href="https://twitter.com/KaihuaQIN?ref_src=twsrc%5Etfw">@KaihuaQIN</a> <a href="https://twitter.com/CULLYAntoine?ref_src=twsrc%5Etfw">@CULLYAntoine</a> <a href="https://twitter.com/convoluted_code?ref_src=twsrc%5Etfw">@convoluted_code</a> ✨🔥🎉 <a href="https://t.co/rlFyzrcJcZ">pic.twitter.com/rlFyzrcJcZ</a></p>&mdash; Arthur Gervais (@HatforceSec) <a href="https://twitter.com/HatforceSec/status/1367421082316263425?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. VELOC: VEry Low Overhead Checkpointing in the Age of Exascale

Bogdan Nicolae, Adam Moody, Gregory Kosinovsky, Kathryn Mohror, Franck Cappello

- retweets: 72, favorites: 24 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02131) | [pdf](https://arxiv.org/pdf/2103.02131)
- [cs.DC](https://arxiv.org/list/cs.DC/recent)

Checkpointing large amounts of related data concurrently to stable storage is a common I/O pattern of many HPC applications. However, such a pattern frequently leads to I/O bottlenecks that lead to poor scalability and performance. As modern HPC infrastructures continue to evolve, there is a growing gap between compute capacity vs. I/O capabilities. Furthermore, the storage hierarchy is becoming increasingly heterogeneous: in addition to parallel file systems, it comprises burst buffers, key-value stores, deep memory hierarchies at node level, etc. In this context, state of art is insufficient to deal with the diversity of vendor APIs, performance and persistency characteristics. This extended abstract presents an overview of VeloC (Very Low Overhead Checkpointing System), a checkpointing runtime specifically design to address these challenges for the next generation Exascale HPC applications and systems. VeloC offers a simple API at user level, while employing an advanced multi-level resilience strategy that transparently optimizes the performance and scalability of checkpointing by leveraging heterogeneous storage.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this extended abstract is presented an overview of VeloC, a checkpointing runtime specifically design to address the challenges for the next gen Exascale <a href="https://twitter.com/hashtag/HPC?src=hash&amp;ref_src=twsrc%5Etfw">#HPC</a> applications and systems.<a href="https://t.co/i43pU5mLKl">https://t.co/i43pU5mLKl</a> <a href="https://t.co/YPFffOkIWx">pic.twitter.com/YPFffOkIWx</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1367339641603555333?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Domain Generalization: A Survey

Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, Chen Change Loy

- retweets: 50, favorites: 32 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02503) | [pdf](https://arxiv.org/pdf/2103.02503)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generalization to out-of-distribution (OOD) data is a capability natural to humans yet challenging for machines to reproduce. This is because most statistical learning algorithms strongly rely on the i.i.d.~assumption while in practice the target data often come from a different distribution than the source data, known as domain shift. Domain generalization (DG) aims to achieve OOD generalization by only using source domain data for model learning. Since first introduced in 2011, research in DG has undergone a decade progress. Ten years of research in this topic have led to a broad spectrum of methodologies, e.g., based on domain alignment, meta-learning, data augmentation, or ensemble learning, just to name a few; and have covered various applications such as object recognition, segmentation, action recognition, and person re-identification. In this paper, for the first time, a comprehensive literature review is provided to summarize the ten-year development in DG. First, we cover the background by giving the problem definitions and discussing how DG is related to other fields like domain adaptation and transfer learning. Second, we conduct a thorough review into existing methods and present a taxonomy based on their methodologies and motivations. Finally, we conclude this survey with potential research directions.




# 11. Style-based Point Generator with Adversarial Rendering for Point Cloud  Completion

Chulin Xie, Chuxin Wang, Bo Zhang, Hao Yang, Dong Chen, Fang Wen

- retweets: 42, favorites: 21 (03/05/2021 11:11:19)

- links: [abs](https://arxiv.org/abs/2103.02535) | [pdf](https://arxiv.org/pdf/2103.02535)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we proposed a novel Style-based Point Generator with Adversarial Rendering (SpareNet) for point cloud completion. Firstly, we present the channel-attentive EdgeConv to fully exploit the local structures as well as the global shape in point features. Secondly, we observe that the concatenation manner used by vanilla foldings limits its potential of generating a complex and faithful shape. Enlightened by the success of StyleGAN, we regard the shape feature as style code that modulates the normalization layers during the folding, which considerably enhances its capability. Thirdly, we realize that existing point supervisions, e.g., Chamfer Distance or Earth Mover's Distance, cannot faithfully reflect the perceptual quality of the reconstructed points. To address this, we propose to project the completed points to depth maps with a differentiable renderer and apply adversarial training to advocate the perceptual realism under different viewpoints. Comprehensive experiments on ShapeNet and KITTI prove the effectiveness of our method, which achieves state-of-the-art quantitative performance while offering superior visual quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Style-based Point Generator with Adversarial Rendering for<br>Point Cloud Completion<br>pdf: <a href="https://t.co/0it0juk2Re">https://t.co/0it0juk2Re</a><br>abs: <a href="https://t.co/CTC1nGVWz2">https://t.co/CTC1nGVWz2</a> <a href="https://t.co/jKljZUcvZh">pic.twitter.com/jKljZUcvZh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367295996787634187?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



