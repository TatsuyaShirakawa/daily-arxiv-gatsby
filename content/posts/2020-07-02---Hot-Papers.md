---
title: Hot Papers 2020-07-02
date: 2020-07-03T15:24:10.Z
template: "post"
draft: false
slug: "hot-papers-2020-07-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-07-02"
socialImage: "/media/42-line-bible.jpg"

---

# 1. Causal Discovery in Physical Systems from Videos

Yunzhu Li, Antonio Torralba, Animashree Anandkumar, Dieter Fox, Animesh Garg

- retweets: 53, favorites: 194 (07/03/2020 15:24:10)

- links: [abs](https://arxiv.org/abs/2007.00631) | [pdf](https://arxiv.org/pdf/2007.00631)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Causal discovery is at the core of human cognition. It enables us to reason about the environment and make counterfactual predictions about unseen scenarios, that can vastly differ from our previous experiences. We consider the task of causal discovery from videos in an end-to-end fashion without supervision on the ground-truth graph structure. In particular, our goal is to discover the structural dependencies among environmental and object variables: inferring the type and strength of interactions that have a causal effect on the behavior of the dynamical system. Our model consists of (a) a perception module that extracts a semantically meaningful and temporally consistent keypoint representation from images, (b) an inference module for determining the graph distribution induced by the detected keypoints, and (c) a dynamics module that can predict the future by conditioning on the inferred graph. We assume access to different configurations and environmental conditions, i.e., data from unknown interventions on the underlying system; thus, we can hope to discover the correct underlying causal graph without explicit interventions. We evaluate our method in a planar multi-body interaction environment and scenarios involving fabrics of different shapes like shirts and pants. Experiments demonstrate that our model can correctly identify the interactions from a short sequence of images and make long-term future predictions. The causal structure assumed by the model also allows it to make counterfactual predictions and extrapolate to systems of unseen interaction graphs or graphs of various sizes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our work on &quot;Causal Discovery in Physical Systems from Videos&quot; from my internship at <a href="https://twitter.com/NVIDIAAI?ref_src=twsrc%5Etfw">@NVIDIAAI</a> <br><br>Paper <a href="https://t.co/9MY2lSKlCS">https://t.co/9MY2lSKlCS</a><br>Website <a href="https://t.co/dFXqbhhPOZ">https://t.co/dFXqbhhPOZ</a><br><br>Thanks to my amazing collaborators!<a href="https://twitter.com/animesh_garg?ref_src=twsrc%5Etfw">@animesh_garg</a>, <a href="https://twitter.com/AnimaAnandkumar?ref_src=twsrc%5Etfw">@AnimaAnandkumar</a>, Dieter Fox, Antonio Torralba<br><br>1/7 <a href="https://t.co/Gc28f2mz6h">pic.twitter.com/Gc28f2mz6h</a></p>&mdash; Yunzhu Li (@YunzhuLiYZ) <a href="https://twitter.com/YunzhuLiYZ/status/1278717373239799808?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Causal Graphs that capture Physical Systems has high potential yet challenging!<br><br>Check out End-to-End Causal Discovery from videos<br>Site: <a href="https://t.co/YriS5oXZXm">https://t.co/YriS5oXZXm</a><br>Paper: <a href="https://t.co/QoC7njUpVa">https://t.co/QoC7njUpVa</a><br><br>w\ <a href="https://twitter.com/YunzhuLiYZ?ref_src=twsrc%5Etfw">@YunzhuLiYZ</a> <a href="https://twitter.com/AnimaAnandkumar?ref_src=twsrc%5Etfw">@AnimaAnandkumar</a>, A.Torralba, D. Fox <a href="https://t.co/n0mIJCOVZU">pic.twitter.com/n0mIJCOVZU</a></p>&mdash; Animesh Garg (@animesh_garg) <a href="https://twitter.com/animesh_garg/status/1278708256656044032?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Causal Discovery in Physical Systems from Videos<br>pdf: <a href="https://t.co/8oKB4DsSKq">https://t.co/8oKB4DsSKq</a><br>abs: <a href="https://t.co/yXnOELnD7s">https://t.co/yXnOELnD7s</a><br>project page: <a href="https://t.co/dKkBKDaiFN">https://t.co/dKkBKDaiFN</a> <a href="https://t.co/SpzZKZlXbr">pic.twitter.com/SpzZKZlXbr</a></p>&mdash; roadrunner01 (@ak92501) <a href="https://twitter.com/ak92501/status/1278496650718216193?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Similarity Search for Efficient Active Learning and Search of Rare  Concepts

Cody Coleman, Edward Chou, Sean Culatana, Peter Bailis, Alexander C. Berg, Roshan Sumbaly, Matei Zaharia, I. Zeki Yalniz

- retweets: 30, favorites: 150 (07/03/2020 15:24:11)

- links: [abs](https://arxiv.org/abs/2007.00077) | [pdf](https://arxiv.org/pdf/2007.00077)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Many active learning and search approaches are intractable for industrial settings with billions of unlabeled examples. Existing approaches, such as uncertainty sampling or information density, search globally for the optimal examples to label, scaling linearly or even quadratically with the unlabeled data. However, in practice, data is often heavily skewed; only a small fraction of collected data will be relevant for a given learning task. For example, when identifying rare classes, detecting malicious content, or debugging model performance, the ratio of positive to negative examples can be 1 to 1,000 or more. In this work, we exploit this skew in large training datasets to reduce the number of unlabeled examples considered in each selection round by only looking at the nearest neighbors to the labeled examples. Empirically, we observe that learned representations effectively cluster unseen concepts, making active learning very effective and substantially reducing the number of viable unlabeled examples. We evaluate several active learning and search techniques in this setting on three large-scale datasets: ImageNet, Goodreads spoiler detection, and OpenImages. For rare classes, active learning methods need as little as 0.31% of the labeled data to match the average precision of full supervision. By limiting active learning methods to only consider the immediate neighbors of the labeled data as candidates for labeling, we need only process as little as 1% of the unlabeled data while achieving similar reductions in labeling costs as the traditional global approach. This process of expanding the candidate pool with the nearest neighbors of the labeled set can be done efficiently and reduces the computational complexity of selection by orders of magnitude.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can active learning scale to millions (potentially billions) of examples? Yes! We propose Similarity search for Efficient Active Learning and Search (SEALS) to restrict the candidates considered in each round and vastly reduce the computational complexity: <a href="https://t.co/wnCHMzXege">https://t.co/wnCHMzXege</a></p>&mdash; Cody Coleman (@codyaustun) <a href="https://twitter.com/codyaustun/status/1278546271595200513?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Debiased Contrastive Learning

Ching-Yao Chuang, Joshua Robinson, Lin Yen-Chen, Antonio Torralba, Stefanie Jegelka

- retweets: 29, favorites: 133 (07/03/2020 15:24:11)

- links: [abs](https://arxiv.org/abs/2007.00224) | [pdf](https://arxiv.org/pdf/2007.00224)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

A prominent technique for self-supervised representation learning has been to contrast semantically similar and dissimilar pairs of samples. Without access to labels, dissimilar (negative) points are typically taken to be randomly sampled datapoints, implicitly accepting that these points may, in reality, actually have the same label. Perhaps unsurprisingly, we observe that sampling negative examples from truly different labels improves performance, in a synthetic setting where labels are available. Motivated by this observation, we develop a debiased contrastive objective that corrects for the sampling of same-label datapoints, even without knowledge of the true labels. Empirically, the proposed objective consistently outperforms the state-of-the-art for representation learning in vision, language, and reinforcement learning benchmarks. Theoretically, we establish generalization bounds for the downstream classification task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our latest preprint on contrastive representation learning!<br><br>Debiased Contrastive Learning<br>paper: <a href="https://t.co/yDh0v64Pp8">https://t.co/yDh0v64Pp8</a><br>code: <a href="https://t.co/Ng7s7Q05xq">https://t.co/Ng7s7Q05xq</a><br><br>w Joshua Robinson, <a href="https://twitter.com/yen_chen_lin?ref_src=twsrc%5Etfw">@yen_chen_lin</a>, Antonio Torralba, &amp; Stefanie Jegelka <a href="https://t.co/wxS3hOTWNX">pic.twitter.com/wxS3hOTWNX</a></p>&mdash; Ching-Yao Chuang (@ChingYaoChuang) <a href="https://twitter.com/ChingYaoChuang/status/1278571799848972289?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Emergence of polarized ideological opinions in multidimensional topic  spaces

Fabian Baumann, Philipp Lorenz-Spreen, Igor M. Sokolov, Michele Starnini

- retweets: 44, favorites: 102 (07/03/2020 15:24:11)

- links: [abs](https://arxiv.org/abs/2007.00601) | [pdf](https://arxiv.org/pdf/2007.00601)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Opinion polarization is on the rise, causing concerns for the openness of public debates. Additionally, extreme opinions on different topics often show significant correlations. The dynamics leading to these polarized ideological opinions pose a challenge: How can such correlations emerge, without assuming them a priori in the individual preferences or in a preexisting social structure? Here we propose a simple model that reproduces ideological opinion states found in survey data, even between rather unrelated, but sufficiently controversial, topics. Inspired by skew coordinate systems recently proposed in natural language processing models, we solidify these intuitions in a formalism where opinions evolve in a multidimensional space where topics form a non-orthogonal basis. The model features a phase transition between consensus, opinion polarization, and ideological states, which we analytically characterize as a function of the controversialness and overlap of the topics. Our findings shed light upon the mechanisms driving the emergence of ideology in the formation of opinions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Last out! <a href="https://twitter.com/electionstudies?ref_src=twsrc%5Etfw">@electionstudies</a> survey data shows that extreme opinions wrt different topics can be correlated. We propose a model where these polarized ideological opinions emerge, without assuming apriori such correlations or preexisting social structures 1/3<a href="https://t.co/vqxupKYhif">https://t.co/vqxupKYhif</a> <a href="https://t.co/SRkh0TNfoS">pic.twitter.com/SRkh0TNfoS</a></p>&mdash; Michele Starnini (@m_starnini) <a href="https://twitter.com/m_starnini/status/1278623624874930176?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Deep Geometric Texture Synthesis

Amir Hertz, Rana Hanocka, Raja Giryes, Daniel Cohen-Or

- retweets: 27, favorites: 111 (07/03/2020 15:24:11)

- links: [abs](https://arxiv.org/abs/2007.00074) | [pdf](https://arxiv.org/pdf/2007.00074)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recently, deep generative adversarial networks for image generation have advanced rapidly; yet, only a small amount of research has focused on generative models for irregular structures, particularly meshes. Nonetheless, mesh generation and synthesis remains a fundamental topic in computer graphics. In this work, we propose a novel framework for synthesizing geometric textures. It learns geometric texture statistics from local neighborhoods (i.e., local triangular patches) of a single reference 3D model. It learns deep features on the faces of the input triangulation, which is used to subdivide and generate offsets across multiple scales, without parameterization of the reference or target mesh. Our network displaces mesh vertices in any direction (i.e., in the normal and tangential direction), enabling synthesis of geometric textures, which cannot be expressed by a simple 2D displacement map. Learning and synthesizing on local geometric patches enables a genus-oblivious framework, facilitating texture transfer between shapes of different genus.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Geometric Texture Synthesis<br>pdf: <a href="https://t.co/oZyaDzxlu3">https://t.co/oZyaDzxlu3</a><br>abs: <a href="https://t.co/w2aIlSg93G">https://t.co/w2aIlSg93G</a> <a href="https://t.co/6TmEFy6unN">pic.twitter.com/6TmEFy6unN</a></p>&mdash; roadrunner01 (@ak92501) <a href="https://twitter.com/ak92501/status/1278497429604745217?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Swapping Autoencoder for Deep Image Manipulation

Taesung Park, Jun-Yan Zhu, Oliver Wang, Jingwan Lu, Eli Shechtman, Alexei A. Efros, Richard Zhang

- retweets: 24, favorites: 114 (07/03/2020 15:24:11)

- links: [abs](https://arxiv.org/abs/2007.00653) | [pdf](https://arxiv.org/pdf/2007.00653)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep generative models have become increasingly effective at producing realistic images from randomly sampled seeds, but using such models for controllable manipulation of existing images remains challenging. We propose the Swapping Autoencoder, a deep model designed specifically for image manipulation, rather than random sampling. The key idea is to encode an image with two independent components and enforce that any swapped combination maps to a realistic image. In particular, we encourage the components to represent structure and texture, by enforcing one component to encode co-occurrent patch statistics across different parts of an image. As our method is trained with an encoder, finding the latent codes for a new input image becomes trivial, rather than cumbersome. As a result, it can be used to manipulate real input images in various ways, including texture swapping, local and global editing, and latent code vector arithmetic. Experiments on multiple datasets show that our model produces better results and is substantially more efficient compared to recent generative models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Swapping Autoencoder for Deep Image Manipulation<br>pdf: <a href="https://t.co/ymWzqolF99">https://t.co/ymWzqolF99</a><br>abs: <a href="https://t.co/8nBm22jOOS">https://t.co/8nBm22jOOS</a><br>project page: <a href="https://t.co/jhfFwb9VyH">https://t.co/jhfFwb9VyH</a><br>video: <a href="https://t.co/278pF01UUI">https://t.co/278pF01UUI</a> <a href="https://t.co/AVqTheajr9">pic.twitter.com/AVqTheajr9</a></p>&mdash; roadrunner01 (@ak92501) <a href="https://twitter.com/ak92501/status/1278498183035924481?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Adaptive Procedural Task Generation for Hard-Exploration Problems

Kuan Fang, Yuke Zhu, Silvio Savarese, Li Fei-Fei

- retweets: 21, favorites: 59 (07/03/2020 15:24:11)

- links: [abs](https://arxiv.org/abs/2007.00350) | [pdf](https://arxiv.org/pdf/2007.00350)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We introduce Adaptive Procedural Task Generation (APT-Gen), an approach for progressively generating a sequence of tasks as curricula to facilitate reinforcement learning in hard-exploration problems. At the heart of our approach, a task generator learns to create tasks via a black-box procedural generation module by adaptively sampling from the parameterized task space. To enable curriculum learning in the absence of a direct indicator of learning progress, the task generator is trained by balancing the agent's expected return in the generated tasks and their similarities to the target task. Through adversarial training, the similarity between the generated tasks and the target task is adaptively estimated by a task discriminator defined on the agent's behaviors. In this way, our approach can efficiently generate tasks of rich variations for target tasks of unknown parameterization or not covered by the predefined task space. Experiments demonstrate the effectiveness of our approach through quantitative and qualitative analysis in various scenarios.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We introduce APT-Gen to procedurally generate tasks of rich variations as curricula for reinforcement learning in hard-exploration problems.<br><br>Webpage: <a href="https://t.co/hRvlVHXStR">https://t.co/hRvlVHXStR</a><br><br>Paper: <a href="https://t.co/24MWtkVxtL">https://t.co/24MWtkVxtL</a><br><br>w/ <a href="https://twitter.com/yukez?ref_src=twsrc%5Etfw">@yukez</a> <a href="https://twitter.com/silviocinguetta?ref_src=twsrc%5Etfw">@silviocinguetta</a> <a href="https://twitter.com/drfeifei?ref_src=twsrc%5Etfw">@drfeifei</a> <a href="https://t.co/vNGDsF87ex">pic.twitter.com/vNGDsF87ex</a></p>&mdash; Kuan Fang (@KuanFang) <a href="https://twitter.com/KuanFang/status/1278542235663925248?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection

Kumar Pratik, Bhaskar D. Rao, Max Welling

- retweets: 8, favorites: 52 (07/03/2020 15:24:11)

- links: [abs](https://arxiv.org/abs/2007.00140) | [pdf](https://arxiv.org/pdf/2007.00140)
- [eess.SP](https://arxiv.org/list/eess.SP/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In this paper, we present a novel neural network for MIMO symbol detection. It is motivated by several important considerations in wireless communication systems; permutation equivariance and a variable number of users. The neural detector learns an iterative decoding algorithm that is implemented as a stack of iterative units. Each iterative unit is a neural computation module comprising of 3 sub-modules: the likelihood module, the encoder module, and the predictor module. The likelihood module injects information about the generative (forward) process into the neural network. The encoder-predictor modules together update the state vector and symbol estimates. The encoder module updates the state vector and employs a transformer based attention network to handle the interactions among the users in a permutation equivariant manner. The predictor module refines the symbol estimates. The modular and permutation equivariant architecture allows for dealing with a varying number of users. The resulting neural detector architecture is unique and exhibits several desirable properties unseen in any of the previously proposed neural detectors. We compare its performance against existing methods and the results show the ability of our network to efficiently handle a variable number of transmitters with high accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This was nice project I did with Pratik Kumar (MSc student UvA [!]) and Bhaskar Rao (UCSD). Pratik combined transformers and Recurrent Inference Machines to do inference in massive MIMO systems in a user-permutation equivariant model. Great work Pratik!  <a href="https://t.co/FUGjuh48Xm">https://t.co/FUGjuh48Xm</a></p>&mdash; Max Welling (@wellingmax) <a href="https://twitter.com/wellingmax/status/1278756525641805829?ref_src=twsrc%5Etfw">July 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



