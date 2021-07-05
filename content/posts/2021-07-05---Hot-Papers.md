---
title: Hot Papers 2021-07-05
date: 2021-07-06T07:09:30.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-05"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-05"
socialImage: "/media/flying-marine.jpg"

---

# 1. A Primer on Pretrained Multilingual Language Models

Sumanth Doddapaneni, Gowtham Ramesh, Anoop Kunchukuttan, Pratyush Kumar, Mitesh M. Khapra

- retweets: 1521, favorites: 183 (07/06/2021 07:09:30)

- links: [abs](https://arxiv.org/abs/2107.00676) | [pdf](https://arxiv.org/pdf/2107.00676)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Multilingual Language Models (MLLMs) such as mBERT, XLM, XLM-R, \textit{etc.} have emerged as a viable option for bringing the power of pretraining to a large number of languages. Given their success in zero shot transfer learning, there has emerged a large body of work in (i) building bigger MLLMs covering a large number of languages (ii) creating exhaustive benchmarks covering a wider variety of tasks and languages for evaluating MLLMs (iii) analysing the performance of MLLMs on monolingual, zero shot crosslingual and bilingual tasks (iv) understanding the universal language patterns (if any) learnt by MLLMs and (v) augmenting the (often) limited capacity of MLLMs to improve their performance on seen or even unseen languages. In this survey, we review the existing literature covering the above broad areas of research pertaining to MLLMs. Based on our survey, we recommend some promising directions of future research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Primer on Pretrained Multilingual Language Models<br><br>This survey paper reviews the existing literature covering research around Multilingual Language Models (MLLMs) such as mBERT, XLM, XLM-R.<br><br>Great progress happening in NLP beyond English. <a href="https://t.co/tCJ7uehh05">https://t.co/tCJ7uehh05</a> <a href="https://t.co/ru7lUniAxo">pic.twitter.com/ru7lUniAxo</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1412003824433258498?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Rapid Neural Architecture Search by Learning to Generate Graphs from  Datasets

Hayeon Lee, Eunyoung Hyung, Sung Ju Hwang

- retweets: 270, favorites: 57 (07/06/2021 07:09:30)

- links: [abs](https://arxiv.org/abs/2107.00860) | [pdf](https://arxiv.org/pdf/2107.00860)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Despite the success of recent Neural Architecture Search (NAS) methods on various tasks which have shown to output networks that largely outperform human-designed networks, conventional NAS methods have mostly tackled the optimization of searching for the network architecture for a single task (dataset), which does not generalize well across multiple tasks (datasets). Moreover, since such task-specific methods search for a neural architecture from scratch for every given task, they incur a large computational cost, which is problematic when the time and monetary budget are limited. In this paper, we propose an efficient NAS framework that is trained once on a database consisting of datasets and pretrained networks and can rapidly search for a neural architecture for a novel dataset. The proposed MetaD2A (Meta Dataset-to-Architecture) model can stochastically generate graphs (architectures) from a given set (dataset) via a cross-modal latent space learned with amortized meta-learning. Moreover, we also propose a meta-performance predictor to estimate and select the best architecture without direct training on target datasets. The experimental results demonstrate that our model meta-learned on subsets of ImageNet-1K and architectures from NAS-Bench 201 search space successfully generalizes to multiple unseen datasets including CIFAR-10 and CIFAR-100, with an average search time of 33 GPU seconds. Even under MobileNetV3 search space, MetaD2A is 5.5K times faster than NSGANetV2, a transferable NAS method, with comparable performance. We believe that the MetaD2A proposes a new research direction for rapid NAS as well as ways to utilize the knowledge from rich databases of datasets and architectures accumulated over the past years. Code is available at https://github.com/HayeonLee/MetaD2A.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets<br>pdf: <a href="https://t.co/tx0OH7nkNg">https://t.co/tx0OH7nkNg</a><br>abs: <a href="https://t.co/bZAuunEksJ">https://t.co/bZAuunEksJ</a><br>github: <a href="https://t.co/8awlEeSWOw">https://t.co/8awlEeSWOw</a> <a href="https://t.co/zdoUzaiIKv">pic.twitter.com/zdoUzaiIKv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412056263085760515?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Systematic Evaluation of Causal Discovery in Visual Model Based  Reinforcement Learning

Nan Rosemary Ke, Aniket Didolkar, Sarthak Mittal, Anirudh Goyal, Guillaume Lajoie, Stefan Bauer, Danilo Rezende, Yoshua Bengio, Michael Mozer, Christopher Pal

- retweets: 138, favorites: 85 (07/06/2021 07:09:30)

- links: [abs](https://arxiv.org/abs/2107.00848) | [pdf](https://arxiv.org/pdf/2107.00848)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Inducing causal relationships from observations is a classic problem in machine learning. Most work in causality starts from the premise that the causal variables themselves are observed. However, for AI agents such as robots trying to make sense of their environment, the only observables are low-level variables like pixels in images. To generalize well, an agent must induce high-level variables, particularly those which are causal or are affected by causal variables. A central goal for AI and causality is thus the joint discovery of abstract representations and causal structure. However, we note that existing environments for studying causal induction are poorly suited for this objective because they have complicated task-specific causal graphs which are impossible to manipulate parametrically (e.g., number of nodes, sparsity, causal chain length, etc.). In this work, our goal is to facilitate research in learning representations of high-level variables as well as causal structures among them. In order to systematically probe the ability of methods to identify these variables and structures, we design a suite of benchmarking RL environments. We evaluate various representation learning algorithms from the literature and find that explicitly incorporating structure and modularity in models can help causal induction in model-based reinforcement learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Model-based RL as a causal induction problem. Our new work on analyzing models for causal induction in MBRL <a href="https://t.co/WPiYREEdnU">https://t.co/WPiYREEdnU</a> which includes 1. An environment for causal induction in MBRL 2. model benchmarks <a href="https://twitter.com/Aniket_d98?ref_src=twsrc%5Etfw">@Aniket_d98</a> <a href="https://twitter.com/anirudhg9119?ref_src=twsrc%5Etfw">@anirudhg9119</a> <a href="https://twitter.com/DaniloJRezende?ref_src=twsrc%5Etfw">@DaniloJRezende</a> <a href="https://twitter.com/chrisjpal?ref_src=twsrc%5Etfw">@chrisjpal</a> <a href="https://t.co/l4ma52Ho5j">pic.twitter.com/l4ma52Ho5j</a></p>&mdash; Nan Rosemary Ke (@rosemary_ke) <a href="https://twitter.com/rosemary_ke/status/1412114970389667841?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Systematic Evaluation of Causal Discovery in Visual Model Based Reinforcement Learning<br>pdf: <a href="https://t.co/kqU053QcQA">https://t.co/kqU053QcQA</a><br>abs: <a href="https://t.co/vfXF4bzVAW">https://t.co/vfXF4bzVAW</a> <a href="https://t.co/3YAWfAgh3n">pic.twitter.com/3YAWfAgh3n</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1411878849646891015?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Flow-based sampling for multimodal distributions in lattice field theory

Daniel C. Hackett, Chung-Chun Hsieh, Michael S. Albergo, Denis Boyda, Jiunn-Wei Chen, Kai-Feng Chen, Kyle Cranmer, Gurtej Kanwar, Phiala E. Shanahan

- retweets: 132, favorites: 72 (07/06/2021 07:09:30)

- links: [abs](https://arxiv.org/abs/2107.00734) | [pdf](https://arxiv.org/pdf/2107.00734)
- [hep-lat](https://arxiv.org/list/hep-lat/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent results have demonstrated that samplers constructed with flow-based generative models are a promising new approach for configuration generation in lattice field theory. In this paper, we present a set of methods to construct flow models for targets with multiple separated modes (i.e. theories with multiple vacua). We demonstrate the application of these methods to modeling two-dimensional real scalar field theory in its symmetry-broken phase. In this context we investigate the performance of different flow-based sampling algorithms, including a composite sampling algorithm where flow-based proposals are occasionally augmented by applying updates using traditional algorithms like HMC.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper, it&#39;s a beast! Huge effort led by Dan Hackett. We investigate different ways of training flows (reverse &amp; forwards KL) for multi-modal distributions (eg. scalar field theory), combining them with MCMC samplers, &amp; pros/cons of performance metrics<a href="https://t.co/CcF7I18tJ5">https://t.co/CcF7I18tJ5</a> <a href="https://t.co/qrUMomTX5G">pic.twitter.com/qrUMomTX5G</a></p>&mdash; Kyle Cranmer (@KyleCranmer) <a href="https://twitter.com/KyleCranmer/status/1412105767545716742?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Learned Token Pruning for Transformers

Sehoon Kim, Sheng Shen, David Thorsley, Amir Gholami, Joseph Hassoun, Kurt Keutzer

- retweets: 81, favorites: 46 (07/06/2021 07:09:31)

- links: [abs](https://arxiv.org/abs/2107.00910) | [pdf](https://arxiv.org/pdf/2107.00910)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

A major challenge in deploying transformer models is their prohibitive inference cost, which quadratically scales with the input sequence length. This makes it especially difficult to use transformers for processing long sequences. To address this, we present a novel Learned Token Pruning (LTP) method that reduces redundant tokens as the data passes through the different layers of the transformer. In particular, LTP prunes tokens with an attention score below a threshold value, which is learned during training. Importantly, our threshold based method avoids algorithmically expensive operations such as top-k token selection which are used in prior token pruning methods, and also leads to structured pruning. We extensively test the performance of our approach on multiple GLUE tasks and show that our learned threshold based method consistently outperforms the prior state-of-the-art top-k token based method by up to ~2% higher accuracy with the same amount of FLOPs. Furthermore, our preliminary results show up to 1.4x and 1.9x throughput improvement on Tesla T4 GPU and Intel Haswell CPU, respectively, with less than 1% of accuracy drop (and up to 2.1x FLOPs reduction). Our code has been developed in PyTorch and has been open-sourced.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learned Token Pruning for Transformers<br>pdf: <a href="https://t.co/skejiZmjiE">https://t.co/skejiZmjiE</a><br>abs: <a href="https://t.co/ObMya09VZH">https://t.co/ObMya09VZH</a><br><br>learned threshold based method consistently outperforms the prior sota top-k token based method by up to ∼2% higher accuracy with the same amount of FLOPs <a href="https://t.co/qNdCwEISEd">pic.twitter.com/qNdCwEISEd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1411850527684575238?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Structural biology in the clouds: The WeNMR-EOSC Ecosystem

Rodrigo Vargas Honorato, Panagiotis I. Koukos, Brian Jiménez-García, Andrei Tsaregorodtsev, Marco Verlato, Andrea Giachetti, Antonio Rosato, Alexandre M.J.J. Bonvin

- retweets: 42, favorites: 19 (07/06/2021 07:09:31)

- links: [abs](https://arxiv.org/abs/2107.01056) | [pdf](https://arxiv.org/pdf/2107.01056)
- [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent) | [cs.CE](https://arxiv.org/list/cs.CE/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Structural biology aims at characterizing the structural and dynamic properties of biological macromolecules at atomic details. Gaining insight into three dimensional structures of biomolecules and their interactions is critical for understanding the vast majority of cellular processes, with direct applications in health and food sciences. Since 2010, the WeNMR project (www.wenmr.eu) has implemented numerous web-based services to facilitate the use of advanced computational tools by researchers in the field, using the high throughput computing infrastructure provided by EGI. These services have been further developed in subsequent initiatives under H2020 projects and are now operating as Thematic Services in the European Open Science Cloud (EOSC) portal (www.eosc-portal.eu), sending >12 millions of jobs and using around 4000 CPU-years per year. Here we review 10 years of successful e-infrastructure solutions serving a large worldwide community of over 23,000 users to date, providing them with user-friendly, web-based solutions that run complex workflows in structural biology. The current set of active WeNMR portals are described, together with the complex backend machinery that allows distributed computing resources to be harvested efficiently.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The preprint of our manuscript describing the <a href="https://twitter.com/hashtag/WeNMR?src=hash&amp;ref_src=twsrc%5Etfw">#WeNMR</a>/EOSC ecosystem for structural biology in the clouds is now available - <a href="https://t.co/zUzGdlPDVA">https://t.co/zUzGdlPDVA</a> <a href="https://t.co/P16TUC1pBk">pic.twitter.com/P16TUC1pBk</a></p>&mdash; Alexandre Bonvin (@amjjbonvin) <a href="https://twitter.com/amjjbonvin/status/1412044392777588743?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Autonomous Navigation for Quadrupedal Robots with Optimized Jumping  through Constrained Obstacles

Scott Gilroy, Derek Lau, Lizhi Yang, Ed Izaguirre, Kristen Biermayer, Anxing Xiao, Mengti Sun, Ayush Agrawal, Jun Zeng, Zhongyu Li, Koushil Sreenath

- retweets: 30, favorites: 30 (07/06/2021 07:09:31)

- links: [abs](https://arxiv.org/abs/2107.00773) | [pdf](https://arxiv.org/pdf/2107.00773)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent)

Quadrupeds are strong candidates for navigating challenging environments because of their agile and dynamic designs. This paper presents a methodology that extends the range of exploration for quadrupedal robots by creating an end-to-end navigation framework that exploits walking and jumping modes. To obtain a dynamic jumping maneuver while avoiding obstacles, dynamically-feasible trajectories are optimized offline through collocation-based optimization where safety constraints are imposed. Such optimization schematic allows the robot to jump through window-shaped obstacles by considering both obstacles in the air and on the ground. The resulted jumping mode is utilized in an autonomous navigation pipeline that leverages a search-based global planner and a local planner to enable the robot to reach the goal location by walking. A state machine together with a decision making strategy allows the system to switch behaviors between walking around obstacles or jumping through them. The proposed framework is experimentally deployed and validated on a quadrupedal robot, a Mini Cheetah, to enable the robot to autonomously navigate through an environment while avoiding obstacles and jumping over a maximum height of 13 cm to pass through a window-shaped opening in order to reach its goal.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Autonomous Navigation for Quadrupedal Robots with Optimized Jumping through Constrained Obstacles<br>pdf: <a href="https://t.co/A465u2fe8s">https://t.co/A465u2fe8s</a><br>abs: <a href="https://t.co/3tIVEIfioo">https://t.co/3tIVEIfioo</a> <a href="https://t.co/GHiz4Qg1Sx">pic.twitter.com/GHiz4Qg1Sx</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1411873327891746817?ref_src=twsrc%5Etfw">July 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Shared Data and Algorithms for Deep Learning in Fundamental Physics

Lisa Benato, Erik Buhmann, Martin Erdmann, Peter Fackeldey, Jonas Glombitza, Nikolai Hartmann, Gregor Kasieczka, William Korcari, Thomas Kuhr, Jan Steinheimer, Horst Stöcker, Tilman Plehn, Kai Zhou

- retweets: 36, favorites: 14 (07/06/2021 07:09:31)

- links: [abs](https://arxiv.org/abs/2107.00656) | [pdf](https://arxiv.org/pdf/2107.00656)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [hep-ph](https://arxiv.org/list/hep-ph/recent) | [nucl-th](https://arxiv.org/list/nucl-th/recent) | [physics.data-an](https://arxiv.org/list/physics.data-an/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We introduce a collection of datasets from fundamental physics research -- including particle physics, astroparticle physics, and hadron- and nuclear physics -- for supervised machine learning studies. These datasets, containing hadronic top quarks, cosmic-ray induced air showers, phase transitions in hadronic matter, and generator-level histories, are made public to simplify future work on cross-disciplinary machine learning and transfer learning in fundamental physics. Based on these data, we present a simple yet flexible graph-based neural network architecture that can easily be applied to a wide range of supervised learning tasks in these domains. We show that our approach reaches performance close to state-of-the-art dedicated methods on all datasets. To simplify adaptation for various problems, we provide easy-to-follow instructions on how graph-based representations of data structures, relevant for fundamental physics, can be constructed and provide code implementations for several of them. Implementations are also provided for our proposed method and all reference algorithms.



