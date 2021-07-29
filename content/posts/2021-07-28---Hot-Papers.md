---
title: Hot Papers 2021-07-28
date: 2021-07-29T10:09:49.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-28"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-28"
socialImage: "/media/flying-marine.jpg"

---

# 1. Geometric Deep Learning on Molecular Representations

Kenneth Atz, Francesca Grisoni, Gisbert Schneider

- retweets: 7316, favorites: 358 (07/29/2021 10:09:49)

- links: [abs](https://arxiv.org/abs/2107.12375) | [pdf](https://arxiv.org/pdf/2107.12375)
- [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent)

Geometric deep learning (GDL), which is based on neural network architectures that incorporate and process symmetry information, has emerged as a recent paradigm in artificial intelligence. GDL bears particular promise in molecular modeling applications, in which various molecular representations with different symmetry properties and levels of abstraction exist. This review provides a structured and harmonized overview of molecular GDL, highlighting its applications in drug discovery, chemical synthesis prediction, and quantum chemistry. Emphasis is placed on the relevance of the learned molecular features and their complementarity to well-established molecular descriptors. This review provides an overview of current challenges and opportunities, and presents a forecast of the future of GDL for molecular sciences.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How can geometric <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> help us tackle modeling challenges in molecular sciences? Check out our latest review &quot;Geometric Deep Learning on Molecular Representations&quot; <a href="https://t.co/nfffFMemME">https://t.co/nfffFMemME</a><a href="https://twitter.com/keennethy?ref_src=twsrc%5Etfw">@keennethy</a> <a href="https://twitter.com/ETH_en?ref_src=twsrc%5Etfw">@ETH_en</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/chemtwitter?src=hash&amp;ref_src=twsrc%5Etfw">#chemtwitter</a> <a href="https://twitter.com/hashtag/compchem?src=hash&amp;ref_src=twsrc%5Etfw">#compchem</a> <a href="https://t.co/SjIv62W0Bf">pic.twitter.com/SjIv62W0Bf</a></p>&mdash; Francesca Grisoni (@fra_grisoni) <a href="https://twitter.com/fra_grisoni/status/1420284463469277196?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Segmentation in Style: Unsupervised Semantic Image Segmentation with  Stylegan and CLIP

Daniil Pakhomov, Sanchit Hira, Narayani Wagle, Kemar E. Green, Nassir Navab

- retweets: 3420, favorites: 318 (07/29/2021 10:09:50)

- links: [abs](https://arxiv.org/abs/2107.12518) | [pdf](https://arxiv.org/pdf/2107.12518)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce a method that allows to automatically segment images into semantically meaningful regions without human supervision. Derived regions are consistent across different images and coincide with human-defined semantic classes on some datasets. In cases where semantic regions might be hard for human to define and consistently label, our method is still able to find meaningful and consistent semantic classes. In our work, we use pretrained StyleGAN2~\cite{karras2020analyzing} generative model: clustering in the feature space of the generative model allows to discover semantic classes. Once classes are discovered, a synthetic dataset with generated images and corresponding segmentation masks can be created. After that a segmentation model is trained on the synthetic dataset and is able to generalize to real images. Additionally, by using CLIP~\cite{radford2021learning} we are able to use prompts defined in a natural language to discover some desired semantic classes. We test our method on publicly available datasets and show state-of-the-art results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Segmentation in Style: Unsupervised Semantic Image Segmentation with Stylegan and CLIP<br>pdf: <a href="https://t.co/KSjm8txURN">https://t.co/KSjm8txURN</a><br>abs: <a href="https://t.co/olELgzAMqI">https://t.co/olELgzAMqI</a><br><br>a method that allows to automatically segment images into semantically meaningful regions without human supervision <a href="https://t.co/or7MmWgL15">pic.twitter.com/or7MmWgL15</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420181693131014146?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction

Eduard Ramon, Gil Triginer, Janna Escur, Albert Pumarola, Jaime Garcia, Xavier Giro-i-Nieto, Francesc Moreno-Noguer

- retweets: 3304, favorites: 268 (07/29/2021 10:09:50)

- links: [abs](https://arxiv.org/abs/2107.12512) | [pdf](https://arxiv.org/pdf/2107.12512)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Recent learning approaches that implicitly represent surface geometry using coordinate-based neural representations have shown impressive results in the problem of multi-view 3D reconstruction. The effectiveness of these techniques is, however, subject to the availability of a large number (several tens) of input views of the scene, and computationally demanding optimizations. In this paper, we tackle these limitations for the specific problem of few-shot full 3D head reconstruction, by endowing coordinate-based representations with a probabilistic shape prior that enables faster convergence and better generalization when using few input images (down to three). First, we learn a shape model of 3D heads from thousands of incomplete raw scans using implicit representations. At test time, we jointly overfit two coordinate-based neural networks to the scene, one modeling the geometry and another estimating the surface radiance, using implicit differentiable rendering. We devise a two-stage optimization strategy in which the learned prior is used to initialize and constrain the geometry during an initial optimization phase. Then, the prior is unfrozen and fine-tuned to the scene. By doing this, we achieve high-fidelity head reconstructions, including hair and shoulders, and with a high level of detail that consistently outperforms both state-of-the-art 3D Morphable Models methods in the few-shot scenario, and non-parametric methods when large sets of views are available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction<br>pdf: <a href="https://t.co/NsZ0nPgunn">https://t.co/NsZ0nPgunn</a><br>abs: <a href="https://t.co/3JH1SgeGJg">https://t.co/3JH1SgeGJg</a><br>project page: <a href="https://t.co/WWwTqCoeGx">https://t.co/WWwTqCoeGx</a> <a href="https://t.co/w1IHGlzA1j">pic.twitter.com/w1IHGlzA1j</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420186993179471872?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. TaikoNation: Patterning-focused Chart Generation for Rhythm Action Games

Emily Halina, Matthew Guzdial

- retweets: 2064, favorites: 256 (07/29/2021 10:09:50)

- links: [abs](https://arxiv.org/abs/2107.12506) | [pdf](https://arxiv.org/pdf/2107.12506)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Generating rhythm game charts from songs via machine learning has been a problem of increasing interest in recent years. However, all existing systems struggle to replicate human-like patterning: the placement of game objects in relation to each other to form congruent patterns based on events in the song. Patterning is a key identifier of high quality rhythm game content, seen as a necessary component in human rankings. We establish a new approach for chart generation that produces charts with more congruent, human-like patterning than seen in prior work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My first academic paper is now public!<br><br>TaikoNation: Patterning-focused Chart Generation for Rhythm Action Games<br><br>&quot;We establish a new approach for chart generation that produces charts with more congruent, human-like patterning than seen in prior work.&quot;<a href="https://t.co/2915W3IuWM">https://t.co/2915W3IuWM</a> <a href="https://t.co/lM5JALsenR">pic.twitter.com/lM5JALsenR</a></p>&mdash; Emily (@livingsuitcase) <a href="https://twitter.com/livingsuitcase/status/1420463898122100737?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. QA Dataset Explosion: A Taxonomy of NLP Resources for Question Answering  and Reading Comprehension

Anna Rogers, Matt Gardner, Isabelle Augenstein

- retweets: 1766, favorites: 175 (07/29/2021 10:09:51)

- links: [abs](https://arxiv.org/abs/2107.12708) | [pdf](https://arxiv.org/pdf/2107.12708)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Alongside huge volumes of research on deep learning models in NLP in the recent years, there has been also much work on benchmark datasets needed to track modeling progress. Question answering and reading comprehension have been particularly prolific in this regard, with over 80 new datasets appearing in the past two years. This study is the largest survey of the field to date. We provide an overview of the various formats and domains of the current resources, highlighting the current lacunae for future work. We further discuss the current classifications of ``reasoning types" in question answering and propose a new taxonomy. We also discuss the implications of over-focusing on English, and survey the current monolingual resources for other languages and multilingual resources. The study is aimed at both practitioners looking for pointers to the wealth of existing data, and at researchers working on new resources.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr"><a href="https://twitter.com/hashtag/NLPaperAlert?src=hash&amp;ref_src=twsrc%5Etfw">#NLPaperAlert</a>: QA Dataset Explosion!🔥<br>A survey of 200+ QA/RC datasets proposing a taxonomy of formats &amp; reasoning skills. Also in the bag: modalities, conversational QA, domains &amp; beyond-English data.<br>Honored to work on this with <a href="https://twitter.com/nlpmattg?ref_src=twsrc%5Etfw">@nlpmattg</a> &amp; <a href="https://twitter.com/IAugenstein?ref_src=twsrc%5Etfw">@IAugenstein</a><a href="https://t.co/xaz9GIXjI4">https://t.co/xaz9GIXjI4</a> <a href="https://t.co/HVfAPvm3OC">pic.twitter.com/HVfAPvm3OC</a></p>&mdash; Anna Rogers (@annargrs) <a href="https://twitter.com/annargrs/status/1420333705726468097?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Human-Level Reinforcement Learning through Theory-Based Modeling,  Exploration, and Planning

Pedro A. Tsividis, Joao Loula, Jake Burga, Nathan Foss, Andres Campero, Thomas Pouncy, Samuel J. Gershman, Joshua B. Tenenbaum

- retweets: 100, favorites: 54 (07/29/2021 10:09:51)

- links: [abs](https://arxiv.org/abs/2107.12544) | [pdf](https://arxiv.org/pdf/2107.12544)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Reinforcement learning (RL) studies how an agent comes to achieve reward in an environment through interactions over time. Recent advances in machine RL have surpassed human expertise at the world's oldest board games and many classic video games, but they require vast quantities of experience to learn successfully -- none of today's algorithms account for the human ability to learn so many different tasks, so quickly. Here we propose a new approach to this challenge based on a particularly strong form of model-based RL which we call Theory-Based Reinforcement Learning, because it uses human-like intuitive theories -- rich, abstract, causal models of physical objects, intentional agents, and their interactions -- to explore and model an environment, and plan effectively to achieve task goals. We instantiate the approach in a video game playing agent called EMPA (the Exploring, Modeling, and Planning Agent), which performs Bayesian inference to learn probabilistic generative models expressed as programs for a game-engine simulator, and runs internal simulations over these models to support efficient object-based, relational exploration and heuristic planning. EMPA closely matches human learning efficiency on a suite of 90 challenging Atari-style video games, learning new games in just minutes of game play and generalizing robustly to new game situations and new levels. The model also captures fine-grained structure in people's exploration trajectories and learning dynamics. Its design and behavior suggest a way forward for building more general human-like AI systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Human-Level Reinforcement Learning through Theory-Based Modeling, Exploration, and Planning<br>pdf: <a href="https://t.co/6nDz0DhsUu">https://t.co/6nDz0DhsUu</a><br>abs: <a href="https://t.co/4x0kAzHoaB">https://t.co/4x0kAzHoaB</a><br><br>closely matches human learning efficiency on a suite of 90 Atari-style video games, learning new games in just minutes <a href="https://t.co/kdapVulrmD">pic.twitter.com/kdapVulrmD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420186090972098569?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Language Grounding with 3D Objects

Jesse Thomason, Mohit Shridhar, Yonatan Bisk, Chris Paxton, Luke Zettlemoyer

- retweets: 76, favorites: 67 (07/29/2021 10:09:51)

- links: [abs](https://arxiv.org/abs/2107.12514) | [pdf](https://arxiv.org/pdf/2107.12514)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Seemingly simple natural language requests to a robot are generally underspecified, for example "Can you bring me the wireless mouse?" When viewing mice on the shelf, the number of buttons or presence of a wire may not be visible from certain angles or positions. Flat images of candidate mice may not provide the discriminative information needed for "wireless". The world, and objects in it, are not flat images but complex 3D shapes. If a human requests an object based on any of its basic properties, such as color, shape, or texture, robots should perform the necessary exploration to accomplish the task. In particular, while substantial effort and progress has been made on understanding explicitly visual attributes like color and category, comparatively little progress has been made on understanding language about shapes and contours. In this work, we introduce a novel reasoning task that targets both visual and non-visual language about 3D objects. Our new benchmark, ShapeNet Annotated with Referring Expressions (SNARE), requires a model to choose which of two objects is being referenced by a natural language description. We introduce several CLIP-based models for distinguishing objects and demonstrate that while recent advances in jointly modeling vision and language are useful for robotic language understanding, it is still the case that these models are weaker at understanding the 3D nature of objects -- properties which play a key role in manipulation. In particular, we find that adding view estimation to language grounding models improves accuracy on both SNARE and when identifying objects referred to in language on a robot platform.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Robots can go beyond image-based grounding by looking at objects from multiple vantage points. <br>To study this ability, &quot;Language Grounding with 3D Objects&quot; presents the ShapeNet Annotated with Referring Expressions (SNARE) benchmark<a href="https://t.co/qgspOJeb4m">https://t.co/qgspOJeb4m</a><a href="https://t.co/IRWvvIujt4">https://t.co/IRWvvIujt4</a> <a href="https://t.co/ebRwSyJeFf">pic.twitter.com/ebRwSyJeFf</a></p>&mdash; Jesse Thomason (@_jessethomason_) <a href="https://twitter.com/_jessethomason_/status/1420458554381594630?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Language Grounding with 3D Objects<br>pdf: <a href="https://t.co/0pnKXWu72R">https://t.co/0pnKXWu72R</a><br>abs: <a href="https://t.co/a5rnWkUnFk">https://t.co/a5rnWkUnFk</a> <a href="https://t.co/5RQFhO7bns">pic.twitter.com/5RQFhO7bns</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420182215875612676?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Ensemble Learning For Mega Man Level Generation

Bowei Li, Ruohan Chen, Yuqing Xue, Ricky Wang, Wenwen Li, Matthew Guzdial

- retweets: 30, favorites: 35 (07/29/2021 10:09:51)

- links: [abs](https://arxiv.org/abs/2107.12524) | [pdf](https://arxiv.org/pdf/2107.12524)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Procedural content generation via machine learning (PCGML) is the process of procedurally generating game content using models trained on existing game content. PCGML methods can struggle to capture the true variance present in underlying data with a single model. In this paper, we investigated the use of ensembles of Markov chains for procedurally generating \emph{Mega Man} levels. We conduct an initial investigation of our approach and evaluate it on measures of playability and stylistic similarity in comparison to a non-ensemble, existing Markov chain approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Ensemble Learning For Mega Man Level Generation<br>pdf: <a href="https://t.co/1VEL3UBQVI">https://t.co/1VEL3UBQVI</a><br>abs: <a href="https://t.co/3IU5E4aq5W">https://t.co/3IU5E4aq5W</a> <a href="https://t.co/mZHhhTQqZe">pic.twitter.com/mZHhhTQqZe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420249874277322753?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Don't Sweep your Learning Rate under the Rug: A Closer Look at  Cross-modal Transfer of Pretrained Transformers

Danielle Rothermel, Margaret Li, Tim Rocktäschel, Jakob Foerster

- retweets: 25, favorites: 36 (07/29/2021 10:09:52)

- links: [abs](https://arxiv.org/abs/2107.12460) | [pdf](https://arxiv.org/pdf/2107.12460)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Self-supervised pre-training of large-scale transformer models on text corpora followed by finetuning has achieved state-of-the-art on a number of natural language processing tasks. Recently, Lu et al. (2021, arXiv:2103.05247) claimed that frozen pretrained transformers (FPTs) match or outperform training from scratch as well as unfrozen (fine-tuned) pretrained transformers in a set of transfer tasks to other modalities. In our work, we find that this result is, in fact, an artifact of not tuning the learning rates. After carefully redesigning the empirical setup, we find that when tuning learning rates properly, pretrained transformers do outperform or match training from scratch in all of our tasks, but only as long as the entire model is finetuned. Thus, while transfer from pretrained language models to other modalities does indeed provide gains and hints at exciting possibilities for future work, properly tuning hyperparameters is important for arriving at robust findings.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Don’t Sweep your Learning Rate under the Rug: A Closer Look at Cross-modal Transfer of Pretrained Transformers<br>pdf: <a href="https://t.co/0OmBiMNeEN">https://t.co/0OmBiMNeEN</a><br><br>show that, across a variety of tasks, the best results are obtained when finetuning all of the weights of a pretrained<br>model <a href="https://t.co/VXWTebdcH5">pic.twitter.com/VXWTebdcH5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420183336488738819?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



