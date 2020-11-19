---
title: Hot Papers 2020-11-18
date: 2020-11-19T10:53:51.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Hierarchical clustering in particle physics through reinforcement  learning

Johann Brehmer, Sebastian Macaluso, Duccio Pappadopulo, Kyle Cranmer

- retweets: 554, favorites: 122 (11/19/2020 10:53:51)

- links: [abs](https://arxiv.org/abs/2011.08191) | [pdf](https://arxiv.org/pdf/2011.08191)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [hep-ph](https://arxiv.org/list/hep-ph/recent)

Particle physics experiments often require the reconstruction of decay patterns through a hierarchical clustering of the observed final-state particles. We show that this task can be phrased as a Markov Decision Process and adapt reinforcement learning algorithms to solve it. In particular, we show that Monte-Carlo Tree Search guided by a neural policy can construct high-quality hierarchical clusterings and outperform established greedy and beam search baselines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper! <br>Hierarchical clustering in particle physics through reinforcement learning<br>with Johann Brehmer, Sebastian Macaluso and Duccio Pappadopulo (<a href="https://twitter.com/ducciolvp?ref_src=twsrc%5Etfw">@ducciolvp</a>)<br><br>One of the few papers using reinforcement learning for particle physics.<a href="https://t.co/WfWvmnLb9v">https://t.co/WfWvmnLb9v</a><br><br>thread 👇 <a href="https://t.co/46O16ccpmm">pic.twitter.com/46O16ccpmm</a></p>&mdash; Kyle Cranmer (@KyleCranmer) <a href="https://twitter.com/KyleCranmer/status/1329039662074236930?ref_src=twsrc%5Etfw">November 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Avoiding Tampering Incentives in Deep RL via Decoupled Approval

Jonathan Uesato, Ramana Kumar, Victoria Krakovna, Tom Everitt, Richard Ngo, Shane Legg

- retweets: 360, favorites: 83 (11/19/2020 10:53:51)

- links: [abs](https://arxiv.org/abs/2011.08827) | [pdf](https://arxiv.org/pdf/2011.08827)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

How can we design agents that pursue a given objective when all feedback mechanisms are influenceable by the agent? Standard RL algorithms assume a secure reward function, and can thus perform poorly in settings where agents can tamper with the reward-generating mechanism. We present a principled solution to the problem of learning from influenceable feedback, which combines approval with a decoupled feedback collection procedure. For a natural class of corruption functions, decoupled approval algorithms have aligned incentives both at convergence and for their local updates. Empirically, they also scale to complex 3D environments where tampering is possible.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Building safe AI requires accounting for the possibility of feedback corruption. The REALab platform provides new insights by studying tampering in simulation: <a href="https://t.co/t2rjxjnpJT">https://t.co/t2rjxjnpJT</a> <br><br>More reading on REALab &amp; Decoupled Approval: <a href="https://t.co/bg2sDKXAFz">https://t.co/bg2sDKXAFz</a> &amp; <a href="https://t.co/2zjtslE63O">https://t.co/2zjtslE63O</a> <a href="https://t.co/cC25m3hOuY">pic.twitter.com/cC25m3hOuY</a></p>&mdash; DeepMind (@DeepMind) <a href="https://twitter.com/DeepMind/status/1329098253153996802?ref_src=twsrc%5Etfw">November 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. REALab: An Embedded Perspective on Tampering

Ramana Kumar, Jonathan Uesato, Richard Ngo, Tom Everitt, Victoria Krakovna, Shane Legg

- retweets: 360, favorites: 79 (11/19/2020 10:53:51)

- links: [abs](https://arxiv.org/abs/2011.08820) | [pdf](https://arxiv.org/pdf/2011.08820)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

This paper describes REALab, a platform for embedded agency research in reinforcement learning (RL). REALab is designed to model the structure of tampering problems that may arise in real-world deployments of RL. Standard Markov Decision Process (MDP) formulations of RL and simulated environments mirroring the MDP structure assume secure access to feedback (e.g., rewards). This may be unrealistic in settings where agents are embedded and can corrupt the processes producing feedback (e.g., human supervisors, or an implemented reward function). We describe an alternative Corrupt Feedback MDP formulation and the REALab environment platform, which both avoid the secure feedback assumption. We hope the design of REALab provides a useful perspective on tampering problems, and that the platform may serve as a unit test for the presence of tampering incentives in RL agent designs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Building safe AI requires accounting for the possibility of feedback corruption. The REALab platform provides new insights by studying tampering in simulation: <a href="https://t.co/t2rjxjnpJT">https://t.co/t2rjxjnpJT</a> <br><br>More reading on REALab &amp; Decoupled Approval: <a href="https://t.co/bg2sDKXAFz">https://t.co/bg2sDKXAFz</a> &amp; <a href="https://t.co/2zjtslE63O">https://t.co/2zjtslE63O</a> <a href="https://t.co/cC25m3hOuY">pic.twitter.com/cC25m3hOuY</a></p>&mdash; DeepMind (@DeepMind) <a href="https://twitter.com/DeepMind/status/1329098253153996802?ref_src=twsrc%5Etfw">November 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Learning Efficient GANs via Differentiable Masks and co-Attention  Distillation

Shaojie Li, Mingbao Lin, Yan Wang, Mingliang Xu, Feiyue Huang, Yongjian Wu, Ling Shao, Rongrong Ji

- retweets: 127, favorites: 65 (11/19/2020 10:53:51)

- links: [abs](https://arxiv.org/abs/2011.08382) | [pdf](https://arxiv.org/pdf/2011.08382)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generative Adversarial Networks (GANs) have been widely-used in image translation, but their high computational and storage costs impede the deployment on mobile devices. Prevalent methods for CNN compression cannot be directly applied to GANs due to the complicated generator architecture and the unstable adversarial training. To solve these, in this paper, we introduce a novel GAN compression method, termed DMAD, by proposing a Differentiable Mask and a co-Attention Distillation. The former searches for a light-weight generator architecture in a training-adaptive manner. To overcome channel inconsistency when pruning the residual connections, an adaptive cross-block group sparsity is further incorporated. The latter simultaneously distills informative attention maps from both the generator and discriminator of a pre-trained model to the searched generator, effectively stabilizing the adversarial training of our light-weight model. Experiments show that DMAD can reduce the Multiply Accumulate Operations (MACs) of CycleGAN by 13x and that of Pix2Pix by 4x while retaining a comparable performance against the full model. Code is available at https://github.com/SJLeo/DMAD.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Efficient GANs using Differentiable Masks and Co-Attention Distillation<br>pdf: <a href="https://t.co/3VTb87um91">https://t.co/3VTb87um91</a><br>abs: <a href="https://t.co/c9AGrkrGn0">https://t.co/c9AGrkrGn0</a><br>github: <a href="https://t.co/b7LFa6Tfz7">https://t.co/b7LFa6Tfz7</a> <a href="https://t.co/gpgUGlEUJw">pic.twitter.com/gpgUGlEUJw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1328881690308186112?ref_src=twsrc%5Etfw">November 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Design Space for Graph Neural Networks

Jiaxuan You, Rex Ying, Jure Leskovec

- retweets: 66, favorites: 18 (11/19/2020 10:53:51)

- links: [abs](https://arxiv.org/abs/2011.08843) | [pdf](https://arxiv.org/pdf/2011.08843)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

The rapid evolution of Graph Neural Networks (GNNs) has led to a growing number of new architectures as well as novel applications. However, current research focuses on proposing and evaluating specific architectural designs of GNNs, as opposed to studying the more general design space of GNNs that consists of a Cartesian product of different design dimensions, such as the number of layers or the type of the aggregation function. Additionally, GNN designs are often specialized to a single task, yet few efforts have been made to understand how to quickly find the best GNN design for a novel task or a novel dataset. Here we define and systematically study the architectural design space for GNNs which consists of 315,000 different designs over 32 different predictive tasks. Our approach features three key innovations: (1) A general GNN design space; (2) a GNN task space with a similarity metric, so that for a given novel task/dataset, we can quickly identify/transfer the best performing architecture; (3) an efficient and effective design space evaluation method which allows insights to be distilled from a huge number of model-task combinations. Our key results include: (1) A comprehensive set of guidelines for designing well-performing GNNs; (2) while best GNN designs for different tasks vary significantly, the GNN task space allows for transferring the best designs across different tasks; (3) models discovered using our design space achieve state-of-the-art performance. Overall, our work offers a principled and scalable approach to transition from studying individual GNN designs for specific tasks, to systematically studying the GNN design space and the task space. Finally, we release GraphGym, a powerful platform for exploring different GNN designs and tasks. GraphGym features modularized GNN implementation, standardized GNN evaluation, and reproducible and scalable experiment management.




# 6. Beyond Static Features for Temporally Consistent 3D Human Pose and Shape  from a Video

Hongsuk Choi, Gyeongsik Moon, Kyoung Mu Lee

- retweets: 27, favorites: 49 (11/19/2020 10:53:51)

- links: [abs](https://arxiv.org/abs/2011.08627) | [pdf](https://arxiv.org/pdf/2011.08627)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Despite the recent success of single image-based 3D human pose and shape estimation methods, recovering temporally consistent and smooth 3D human motion from a video is still challenging. Several video-based methods have been proposed; however, they fail to resolve the single image-based methods' temporal inconsistency issue due to a strong dependency on a static feature of the current frame. In this regard, we present a temporally consistent mesh recovery system (TCMR). It effectively focuses on the past and future frames' temporal information without being dominated by the current static feature. Our TCMR significantly outperforms previous video-based methods in temporal consistency with better per-frame 3D pose and shape accuracy. We will release the codes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video<br>pdf: <a href="https://t.co/v0dpNgNgiG">https://t.co/v0dpNgNgiG</a><br>abs: <a href="https://t.co/qogmmaiiJZ">https://t.co/qogmmaiiJZ</a> <a href="https://t.co/S5HcJDM1cP">pic.twitter.com/S5HcJDM1cP</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1328896506464776205?ref_src=twsrc%5Etfw">November 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Learn2Sing: Target Speaker Singing Voice Synthesis by learning from a  Singing Teacher

Heyang Xue, Shan Yang, Yi Lei, Lei Xie, Xiulin Li

- retweets: 30, favorites: 30 (11/19/2020 10:53:51)

- links: [abs](https://arxiv.org/abs/2011.08467) | [pdf](https://arxiv.org/pdf/2011.08467)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Singing voice synthesis has been paid rising attention with the rapid development of speech synthesis area. In general, a studio-level singing corpus is usually necessary to produce a natural singing voice from lyrics and music-related transcription. However, such a corpus is difficult to collect since it's hard for many of us to sing like a professional singer. In this paper, we propose an approach -- Learn2Sing that only needs a singing teacher to generate the target speakers' singing voice without their singing voice data. In our approach, a teacher's singing corpus and speech from multiple target speakers are trained in a frame-level auto-regressive acoustic model where singing and speaking share the common speaker embedding and style tag embedding. Meanwhile, since there is no music-related transcription for the target speaker, we use log-scale fundamental frequency (LF0) as an auxiliary feature as the inputs of the acoustic model for building a unified input representation. In order to enable the target speaker to sing without singing reference audio in the inference stage, a duration model and an LF0 prediction model are also trained. Particularly, we employ domain adversarial training (DAT) in the acoustic model, which aims to enhance the singing performance of target speakers by disentangling style from acoustic features of singing and speaking data. Our experiments indicate that the proposed approach is capable of synthesizing singing voice for target speaker given only their speech samples.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learn2Sing: Target Speaker Singing Voice Synthesis by learning from a Singing Teacher<br>pdf: <a href="https://t.co/R4UaYQcd7l">https://t.co/R4UaYQcd7l</a><br>abs: <a href="https://t.co/6inNcwKYAd">https://t.co/6inNcwKYAd</a><br>project page: <a href="https://t.co/F076kfq7h2">https://t.co/F076kfq7h2</a> <a href="https://t.co/HPoRY6Sojw">pic.twitter.com/HPoRY6Sojw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1328887436487892992?ref_src=twsrc%5Etfw">November 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



