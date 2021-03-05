---
title: Hot Papers 2021-03-03
date: 2021-03-05T10:30:34.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-03"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-03"
socialImage: "/media/flying-marine.jpg"

---

# 1. WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual  Machine Learning

Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, Marc Najork

- retweets: 2052, favorites: 201 (03/05/2021 10:30:34)

- links: [abs](https://arxiv.org/abs/2103.01913) | [pdf](https://arxiv.org/pdf/2103.01913)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

The milestone improvements brought about by deep representation learning and pre-training techniques have led to large performance gains across downstream NLP, IR and Vision tasks. Multimodal modeling techniques aim to leverage large high-quality visio-linguistic datasets for learning complementary information (across image and text modalities). In this paper, we introduce the Wikipedia-based Image Text (WIT) Dataset (https://github.com/google-research-datasets/wit) to better facilitate multimodal, multilingual learning. WIT is composed of a curated set of 37.6 million entity rich image-text examples with 11.5 million unique images across 108 Wikipedia languages. Its size enables WIT to be used as a pretraining dataset for multimodal models, as we show when applied to downstream tasks such as image-text retrieval. WIT has four main and unique advantages. First, WIT is the largest multimodal dataset by the number of image-text examples by 3x (at the time of writing). Second, WIT is massively multilingual (first of its kind) with coverage over 100+ languages (each of which has at least 12K examples) and provides cross-lingual texts for many images. Third, WIT represents a more diverse set of concepts and real world entities relative to what previous datasets cover. Lastly, WIT provides a very challenging real-world test set, as we empirically illustrate using an image-text retrieval task as an example.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">WIT: Wikipedia-based Image Text Dataset for Multimodal<br>Multilingual Machine Learning<br>pdf: <a href="https://t.co/fblyzH2hGe">https://t.co/fblyzH2hGe</a><br>abs: <a href="https://t.co/tVgBdfOnQ5">https://t.co/tVgBdfOnQ5</a><br>github: <a href="https://t.co/NNkF3oheok">https://t.co/NNkF3oheok</a> <a href="https://t.co/nnFUaPJaYU">pic.twitter.com/nnFUaPJaYU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1366936554749521920?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Contrastive Explanations for Model Interpretability

Alon Jacovi, Swabha Swayamdipta, Shauli Ravfogel, Yanai Elazar, Yejin Choi, Yoav Goldberg

- retweets: 1164, favorites: 178 (03/05/2021 10:30:34)

- links: [abs](https://arxiv.org/abs/2103.01378) | [pdf](https://arxiv.org/pdf/2103.01378)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Contrastive explanations clarify why an event occurred in contrast to another. They are more inherently intuitive to humans to both produce and comprehend. We propose a methodology to produce contrastive explanations for classification models by modifying the representation to disregard non-contrastive information, and modifying model behavior to only be based on contrastive reasoning. Our method is based on projecting model representation to a latent space that captures only the features that are useful (to the model) to differentiate two potential decisions. We demonstrate the value of contrastive explanations by analyzing two different scenarios, using both high-level abstract concept attribution and low-level input token/span attribution, on two widely used text classification tasks. Specifically, we produce explanations for answering: for which label, and against which alternative label, is some aspect of the input useful? And which aspects of the input are useful for and against particular decisions? Overall, our findings shed light on the ability of label-contrastive explanations to provide a more accurate and finer-grained interpretability of a model's decision.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new paper 😊 (w\ <a href="https://twitter.com/swabhz?ref_src=twsrc%5Etfw">@swabhz</a> <a href="https://twitter.com/ravfogel?ref_src=twsrc%5Etfw">@ravfogel</a> <a href="https://twitter.com/yanaiela?ref_src=twsrc%5Etfw">@yanaiela</a> <a href="https://twitter.com/YejinChoinka?ref_src=twsrc%5Etfw">@YejinChoinka</a> <a href="https://twitter.com/yoavgo?ref_src=twsrc%5Etfw">@yoavgo</a> )<br><br>Contrastive Explanations for Model Interpretability<a href="https://t.co/StaGM5q5qN">https://t.co/StaGM5q5qN</a><br><br>This paper is about explaining classifier decisions contrastively against alternative decisions. <a href="https://t.co/ZRa0Z1xUDc">pic.twitter.com/ZRa0Z1xUDc</a></p>&mdash; Alon Jacovi (@alon_jacovi) <a href="https://twitter.com/alon_jacovi/status/1367118232671387648?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Mixture of Volumetric Primitives for Efficient Neural Rendering

Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, Jason Saragih

- retweets: 680, favorites: 144 (03/05/2021 10:30:35)

- links: [abs](https://arxiv.org/abs/2103.01954) | [pdf](https://arxiv.org/pdf/2103.01954)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Real-time rendering and animation of humans is a core function in games, movies, and telepresence applications. Existing methods have a number of drawbacks we aim to address with our work. Triangle meshes have difficulty modeling thin structures like hair, volumetric representations like Neural Volumes are too low-resolution given a reasonable memory budget, and high-resolution implicit representations like Neural Radiance Fields are too slow for use in real-time applications. We present Mixture of Volumetric Primitives (MVP), a representation for rendering dynamic 3D content that combines the completeness of volumetric representations with the efficiency of primitive-based rendering, e.g., point-based or mesh-based methods. Our approach achieves this by leveraging spatially shared computation with a deconvolutional architecture and by minimizing computation in empty regions of space with volumetric primitives that can move to cover only occupied regions. Our parameterization supports the integration of correspondence and tracking constraints, while being robust to areas where classical tracking fails, such as around thin or translucent structures and areas with large topological variability. MVP is a hybrid that generalizes both volumetric and primitive-based representations. Through a series of extensive experiments we demonstrate that it inherits the strengths of each, while avoiding many of their limitations. We also compare our approach to several state-of-the-art methods and demonstrate that MVP produces superior results in terms of quality and runtime performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mixture of Volumetric Primitives for Efficient Neural Rendering<br>pdf: <a href="https://t.co/N3L5PLVWGb">https://t.co/N3L5PLVWGb</a><br>abs: <a href="https://t.co/TpbAOLSJZk">https://t.co/TpbAOLSJZk</a> <a href="https://t.co/PRURQky6Pl">pic.twitter.com/PRURQky6Pl</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1366936919444234242?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Predicting Video with VQVAE

Jacob Walker, Ali Razavi, Aäron van den Oord

- retweets: 324, favorites: 163 (03/05/2021 10:30:35)

- links: [abs](https://arxiv.org/abs/2103.01950) | [pdf](https://arxiv.org/pdf/2103.01950)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In recent years, the task of video prediction-forecasting future video given past video frames-has attracted attention in the research community. In this paper we propose a novel approach to this problem with Vector Quantized Variational AutoEncoders (VQ-VAE). With VQ-VAE we compress high-resolution videos into a hierarchical set of multi-scale discrete latent variables. Compared to pixels, this compressed latent space has dramatically reduced dimensionality, allowing us to apply scalable autoregressive generative models to predict video. In contrast to previous work that has largely emphasized highly constrained datasets, we focus on very diverse, large-scale datasets such as Kinetics-600. We predict video at a higher resolution on unconstrained videos, 256x256, than any other previous method to our knowledge. We further validate our approach against prior work via a crowdsourced human evaluation.

<blockquote class="twitter-tweet"><p lang="pt" dir="ltr">Predicting Video with VQVAE<br>pdf: <a href="https://t.co/E6Ij9rCOlL">https://t.co/E6Ij9rCOlL</a><br>abs: <a href="https://t.co/joGZcCUWlC">https://t.co/joGZcCUWlC</a> <a href="https://t.co/1NQMfrZlxh">pic.twitter.com/1NQMfrZlxh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1366932712511201280?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Categorical Foundations of Gradient-Based Learning

G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, Fabio Zanasi

- retweets: 328, favorites: 101 (03/05/2021 10:30:35)

- links: [abs](https://arxiv.org/abs/2103.01931) | [pdf](https://arxiv.org/pdf/2103.01931)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.CT](https://arxiv.org/list/math.CT/recent)

We propose a categorical foundation of gradient-based machine learning algorithms in terms of lenses, parametrised maps, and reverse derivative categories. This foundation provides a powerful explanatory and unifying framework: it encompasses a variety of gradient descent algorithms such as ADAM, AdaGrad, and Nesterov momentum, as well as a variety of loss functions such as as MSE and Softmax cross-entropy, shedding new light on their similarities and differences. Our approach also generalises beyond neural networks (modelled in categories of smooth maps), accounting for other structures relevant to gradient-based learning such as boolean circuits. Finally, we also develop a novel implementation of gradient-based learning in Python, informed by the principles introduced by our framework.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper on &quot;Categorical Foundations of Gradient-Based Learning&quot; is out! (<a href="https://t.co/aREoHErf0L">https://t.co/aREoHErf0L</a>), It&#39;s accompanied by a very short blog post (<a href="https://t.co/qdEbl63zuH">https://t.co/qdEbl63zuH</a>) describing some of the main ideas, as well as a video with the presentation <a href="https://t.co/gGi6vF5YXk">https://t.co/gGi6vF5YXk</a></p>&mdash; Bruno Gavranović (@bgavran3) <a href="https://twitter.com/bgavran3/status/1367580038783188995?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Categorical foundations of gradient-based learning, by Geoffrey Cruttwell, <a href="https://twitter.com/bgavran3?ref_src=twsrc%5Etfw">@bgavran3</a>, <a href="https://twitter.com/Anarchia45?ref_src=twsrc%5Etfw">@Anarchia45</a>, <a href="https://twitter.com/statusfailed?ref_src=twsrc%5Etfw">@statusfailed</a> and Fabio Zanasi<br><br>I believe this is quite an important paper for applied category theory!<a href="https://t.co/36QAUodnaw">https://t.co/36QAUodnaw</a> <a href="https://t.co/cZ0LWTaDJq">pic.twitter.com/cZ0LWTaDJq</a></p>&mdash; julesh (@_julesh_) <a href="https://twitter.com/_julesh_/status/1367146198755328008?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Cruttwell et al &quot;Categorical Foundations of Gradient-Based Learning&quot;<a href="https://t.co/E9SEOEHFtx">https://t.co/E9SEOEHFtx</a><br><br>should be some juicy goodies in there. Also, an implementation (in Python)!</p>&mdash; theHigherGeometer (@HigherGeometer) <a href="https://twitter.com/HigherGeometer/status/1367057477552902146?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Reinforcement Learning for Adaptive Mesh Refinement

Jiachen Yang, Tarik Dzanic, Brenden Petersen, Jun Kudo, Ketan Mittal, Vladimir Tomov, Jean-Sylvain Camier, Tuo Zhao, Hongyuan Zha, Tzanio Kolev, Robert Anderson, Daniel Faissol

- retweets: 156, favorites: 49 (03/05/2021 10:30:35)

- links: [abs](https://arxiv.org/abs/2103.01342) | [pdf](https://arxiv.org/pdf/2103.01342)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.NA](https://arxiv.org/list/math.NA/recent)

Large-scale finite element simulations of complex physical systems governed by partial differential equations crucially depend on adaptive mesh refinement (AMR) to allocate computational budget to regions where higher resolution is required. Existing scalable AMR methods make heuristic refinement decisions based on instantaneous error estimation and thus do not aim for long-term optimality over an entire simulation. We propose a novel formulation of AMR as a Markov decision process and apply deep reinforcement learning (RL) to train refinement policies directly from simulation. AMR poses a new problem for RL in that both the state dimension and available action set changes at every step, which we solve by proposing new policy architectures with differing generality and inductive bias. The model sizes of these policy architectures are independent of the mesh size and hence scale to arbitrarily large and complex simulations. We demonstrate in comprehensive experiments on static function estimation and the advection of different fields that RL policies can be competitive with a widely-used error estimator and generalize to larger, more complex, and unseen test problems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Reinforcement Learning for Adaptive Mesh Refinement<br>pdf: <a href="https://t.co/6pxJhHmh4I">https://t.co/6pxJhHmh4I</a><br>abs: <a href="https://t.co/wEX7r6G1rQ">https://t.co/wEX7r6G1rQ</a><br>project page: <a href="https://t.co/Xk4NdmqszT">https://t.co/Xk4NdmqszT</a> <a href="https://t.co/VygqoYLwHP">pic.twitter.com/VygqoYLwHP</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1366955863982936064?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. PyQUBO: Python Library for Mapping Combinatorial Optimization Problems  to QUBO Form

Mashiyat Zaman, Kotaro Tanahashi, Shu Tanaka

- retweets: 146, favorites: 28 (03/05/2021 10:30:35)

- links: [abs](https://arxiv.org/abs/2103.01708) | [pdf](https://arxiv.org/pdf/2103.01708)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.ET](https://arxiv.org/list/cs.ET/recent)

We present PyQUBO, an open-source, Python library for constructing quadratic unconstrained binary optimizations (QUBOs) from the objective functions and the constraints of optimization problems. PyQUBO enables users to prepare QUBOs or Ising models for various combinatorial optimization problems with ease thanks to the abstraction of expressions and the extensibility of the program. QUBOs and Ising models formulated using PyQUBO are solvable by Ising machines, including quantum annealing machines. We introduce the features of PyQUBO with applications in the number partitioning problem, knapsack problem, graph coloring problem, and integer factorization using a binary multiplier. Moreover, we demonstrate how PyQUBO can be applied to production-scale problems through integration with quantum annealing machines. Through its flexibility and ease of use, PyQUBO has the potential to make quantum annealing a more practical tool among researchers.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">arXivに論文プレプリントをアップロードしました！<br><br>量子アニーリング等イジングマシンを利用するためのOSS「PYQUBO」（棚橋さんが数年前から開発。2018年9月より公開）に関する論文<br>PyQUBO: Python Library for Mapping Combinatorial Optimization Problems to QUBO Form<a href="https://t.co/Wte56CqAO6">https://t.co/Wte56CqAO6</a></p>&mdash; Shu Tanaka (@tnksh) <a href="https://twitter.com/tnksh/status/1366958659369013249?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Kernel Interpolation for Scalable Online Gaussian Processes

Samuel Stanton, Wesley J. Maddox, Ian Delbridge, Andrew Gordon Wilson

- retweets: 90, favorites: 38 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01454) | [pdf](https://arxiv.org/pdf/2103.01454)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Gaussian processes (GPs) provide a gold standard for performance in online settings, such as sample-efficient control and black box optimization, where we need to update a posterior distribution as we acquire data in a sequential fashion. However, updating a GP posterior to accommodate even a single new observation after having observed $n$ points incurs at least $O(n)$ computations in the exact setting. We show how to use structured kernel interpolation to efficiently recycle computations for constant-time $O(1)$ online updates with respect to the number of points $n$, while retaining exact inference. We demonstrate the promise of our approach in a range of online regression and classification settings, Bayesian optimization, and active sampling to reduce error in malaria incidence forecasting. Code is available at https://github.com/wjmaddox/online_gp.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our new <a href="https://twitter.com/hashtag/AISTATS2021?src=hash&amp;ref_src=twsrc%5Etfw">#AISTATS2021</a> paper, &quot;Kernel Interpolation for Scalable Online Gaussian Processes&quot;, we show how to do O(1) streaming Bayesian updates, while retaining exact inference! <a href="https://t.co/i4TeqoSVz3">https://t.co/i4TeqoSVz3</a><br>with <a href="https://twitter.com/samscub?ref_src=twsrc%5Etfw">@samscub</a>, W. Maddox, <a href="https://twitter.com/DelbridgeIan?ref_src=twsrc%5Etfw">@DelbridgeIan</a>. 1/6 <a href="https://t.co/YyWQdhClJL">pic.twitter.com/YyWQdhClJL</a></p>&mdash; Andrew Gordon Wilson (@andrewgwils) <a href="https://twitter.com/andrewgwils/status/1367614872578580480?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Median Optimal Treatment Regimes

Liu Leqi, Edward H. Kennedy

- retweets: 72, favorites: 50 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01802) | [pdf](https://arxiv.org/pdf/2103.01802)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Optimal treatment regimes are personalized policies for making a treatment decision based on subject characteristics, with the policy chosen to maximize some value. It is common to aim to maximize the mean outcome in the population, via a regime assigning treatment only to those whose mean outcome is higher under treatment versus control. However, the mean can be an unstable measure of centrality, resulting in imprecise statistical procedures, as well as unfair decisions that can be overly influenced by a small fraction of subjects. In this work, we propose a new median optimal treatment regime that instead treats individuals whose conditional median is higher under treatment. This ensures that optimal decisions for individuals from the same group are not overly influenced either by (i) a small fraction of the group (unlike the mean criterion), or (ii) unrelated subjects from different groups (unlike marginal median/quantile criteria). We introduce a new measure of value, the Average Conditional Median Effect (ACME), which summarizes across-group median treatment outcomes of a policy, and which the optimal median treatment regime maximizes. After developing key motivating examples that distinguish median optimal treatment regimes from mean and marginal median optimal treatment regimes, we give a nonparametric efficiency bound for estimating the ACME of a policy, and propose a new doubly robust-style estimator that achieves the efficiency bound under weak conditions. Finite-sample properties of the estimator are explored via numerical simulations and the proposed algorithm is illustrated using data from a randomized clinical trial in patients with HIV.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Cool new paper by <a href="https://twitter.com/leqi_liu?ref_src=twsrc%5Etfw">@leqi_liu</a>!<br><br>Mean optimal trt rules aren&#39;t robust: sensitive to small % w/ extreme outcomes<br><br>Marginal median opt rules are unfair in different way: my treatment depends on *your* outcomes<br><br>We study fair+robust median optimal trt rules:<a href="https://t.co/oXWwRKYnSh">https://t.co/oXWwRKYnSh</a> <a href="https://t.co/ieQoLS8GYX">pic.twitter.com/ieQoLS8GYX</a></p>&mdash; Edward Kennedy (@edwardhkennedy) <a href="https://twitter.com/edwardhkennedy/status/1366945466219388928?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Image-to-image Translation via Hierarchical Style Disentanglement

Xinyang Li, Shengchuan Zhang, Jie Hu, Liujuan Cao, Xiaopeng Hong, Xudong Mao, Feiyue Huang, Yongjian Wu, Rongrong Ji

- retweets: 56, favorites: 44 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01456) | [pdf](https://arxiv.org/pdf/2103.01456)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, image-to-image translation has made significant progress in achieving both multi-label (\ie, translation conditioned on different labels) and multi-style (\ie, generation with diverse styles) tasks. However, due to the unexplored independence and exclusiveness in the labels, existing endeavors are defeated by involving uncontrolled manipulations to the translation results. In this paper, we propose Hierarchical Style Disentanglement (HiSD) to address this issue. Specifically, we organize the labels into a hierarchical tree structure, in which independent tags, exclusive attributes, and disentangled styles are allocated from top to bottom. Correspondingly, a new translation process is designed to adapt the above structure, in which the styles are identified for controllable translations. Both qualitative and quantitative results on the CelebA-HQ dataset verify the ability of the proposed HiSD. We hope our method will serve as a solid baseline and provide fresh insights with the hierarchically organized annotations for future research in image-to-image translation. The code has been released at https://github.com/imlixinyang/HiSD.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Image-to-image Translation via Hierarchical Style Disentanglement<br>pdf: <a href="https://t.co/WgqJRgGnrY">https://t.co/WgqJRgGnrY</a><br>abs: <a href="https://t.co/QLeuF76wlL">https://t.co/QLeuF76wlL</a> <a href="https://t.co/uJx0kUF66w">pic.twitter.com/uJx0kUF66w</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1366936037386248196?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. An Analysis of Distributed Systems Syllabi With a Focus on  Performance-Related Topics

Cristina L. Abad, Alexandru Iosup, Edwin F. Boza, Eduardo Ortiz-Holguin

- retweets: 72, favorites: 25 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01858) | [pdf](https://arxiv.org/pdf/2103.01858)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

We analyze a dataset of 51 current (2019-2020) Distributed Systems syllabi from top Computer Science programs, focusing on finding the prevalence and context in which topics related to performance are being taught in these courses. We also study the scale of the infrastructure mentioned in DS courses, from small client-server systems to cloud-scale, peer-to-peer, global-scale systems. We make eight main findings, covering goals such as performance, and scalability and its variant elasticity; activities such as performance benchmarking and monitoring; eight selected performance-enhancing techniques (replication, caching, sharding, load balancing, scheduling, streaming, migrating, and offloading); and control issues such as trade-offs that include performance and performance variability.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">As members of <a href="https://twitter.com/spec_perf?ref_src=twsrc%5Etfw">@spec_perf</a> Research Group, we also took a look at Distributed Systems syllabi focusing on performance-related topics. Paper to be presented at Workshop on Education &amp; Practice of Performance Eng. at <a href="https://twitter.com/ICPEconf?ref_src=twsrc%5Etfw">@ICPEconf</a>: <a href="https://t.co/hVstqKujWz">https://t.co/hVstqKujWz</a> cc <a href="https://twitter.com/AIosup?ref_src=twsrc%5Etfw">@AIosup</a> <a href="https://twitter.com/edwinboza?ref_src=twsrc%5Etfw">@edwinboza</a> <a href="https://twitter.com/leortyz?ref_src=twsrc%5Etfw">@leortyz</a> <a href="https://t.co/hPmJnjcRL1">https://t.co/hPmJnjcRL1</a> <a href="https://t.co/nV21sub2VH">pic.twitter.com/nV21sub2VH</a></p>&mdash; Cristina L. Abad (@cabad3) <a href="https://twitter.com/cabad3/status/1367095866671136778?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for  Place Recognition

Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford, Tobias Fischer

- retweets: 48, favorites: 48 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01486) | [pdf](https://arxiv.org/pdf/2103.01486)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Visual Place Recognition is a challenging task for robotics and autonomous systems, which must deal with the twin problems of appearance and viewpoint change in an always changing world. This paper introduces Patch-NetVLAD, which provides a novel formulation for combining the advantages of both local and global descriptor methods by deriving patch-level features from NetVLAD residuals. Unlike the fixed spatial neighborhood regime of existing local keypoint features, our method enables aggregation and matching of deep-learned local features defined over the feature-space grid. We further introduce a multi-scale fusion of patch features that have complementary scales (i.e. patch sizes) via an integral feature space and show that the fused features are highly invariant to both condition (season, structure, and illumination) and viewpoint (translation and rotation) changes. Patch-NetVLAD outperforms both global and local feature descriptor-based methods with comparable compute, achieving state-of-the-art visual place recognition results on a range of challenging real-world datasets, including winning the Facebook Mapillary Visual Place Recognition Challenge at ECCV2020. It is also adaptable to user requirements, with a speed-optimised version operating over an order of magnitude faster than the state-of-the-art. By combining superior performance with improved computational efficiency in a configurable framework, Patch-NetVLAD is well suited to enhance both stand-alone place recognition capabilities and the overall performance of SLAM systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Incredibly happy that our <a href="https://twitter.com/CVPR?ref_src=twsrc%5Etfw">@CVPR</a> <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> submission &quot;Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition&quot; has been accepted - preprint: <a href="https://t.co/hifE1u9P31">https://t.co/hifE1u9P31</a><br>Great teamwork by <a href="https://twitter.com/Dalek25?ref_src=twsrc%5Etfw">@Dalek25</a> <a href="https://twitter.com/sourav_garg_?ref_src=twsrc%5Etfw">@sourav_garg_</a> Ming Xu and <a href="https://twitter.com/maththrills?ref_src=twsrc%5Etfw">@maththrills</a> <a href="https://twitter.com/QUTRobotics?ref_src=twsrc%5Etfw">@QUTRobotics</a>. 1/n <a href="https://t.co/NW6DU0xX9K">pic.twitter.com/NW6DU0xX9K</a></p>&mdash; Tobias Fischer (@TobiasRobotics) <a href="https://twitter.com/TobiasRobotics/status/1367268624923119616?ref_src=twsrc%5Etfw">March 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. A practical tutorial on Variational Bayes

Minh-Ngoc Tran, Trong-Nghia Nguyen, Viet-Hung Dao

- retweets: 54, favorites: 38 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01327) | [pdf](https://arxiv.org/pdf/2103.01327)
- [stat.CO](https://arxiv.org/list/stat.CO/recent) | [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

This tutorial gives a quick introduction to Variational Bayes (VB), also called Variational Inference or Variational Approximation, from a practical point of view. The paper covers a range of commonly used VB methods and an attempt is made to keep the materials accessible to the wide community of data analysis practitioners. The aim is that the reader can quickly derive and implement their first VB algorithm for Bayesian inference with their data analysis problem. An end-user software package in Matlab together with the documentation can be found at https://vbayeslab.github.io/VBLabDocs/

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">A practical tutorial on Variational Bayes. (arXiv:2103.01327v1 [<a href="https://t.co/uFsa4flLGi">https://t.co/uFsa4flLGi</a>]) <a href="https://t.co/4OpQ53KQFq">https://t.co/4OpQ53KQFq</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1367033921754566656?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Listen, Read, and Identify: Multimodal Singing Language Identification

Keunwoo Choi, Yuxuan Wang

- retweets: 22, favorites: 49 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01893) | [pdf](https://arxiv.org/pdf/2103.01893)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We propose a multimodal singing language classification model that uses both audio content and textual metadata. LRID-Net, the proposed model, takes an audio signal and a language probability vector estimated from the metadata and outputs the probabilities of the ten target languages. Optionally, LRID-Net is facilitated with modality dropouts to handle a missing modality. In the experiment, we trained several LRID-Nets with varying modality dropout configuration and test them with various combinations of input modalities. The experiment results demonstrate that using multimodal input improves the performance. The results also suggest that adopting modality dropout does not degrade performance of the model when there are full modality inputs while enabling the model to handle missing modality cases to some extent.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr"><a href="https://t.co/frxIYHmuQs">https://t.co/frxIYHmuQs</a> Long time no first-authoring! Listen, Read, and Identify network (LRID-Net) identifies singing language by reading the metadata (title, album, artist) and listening to the audio.</p>&mdash; Keunwoo Choi (@keunwoochoi) <a href="https://twitter.com/keunwoochoi/status/1366988931120525313?ref_src=twsrc%5Etfw">March 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. A Deep Emulator for Secondary Motion of 3D Characters

Mianlun Zheng, Yi Zhou, Duygu Ceylan, Jernej Barbic

- retweets: 30, favorites: 21 (03/05/2021 10:30:36)

- links: [abs](https://arxiv.org/abs/2103.01261) | [pdf](https://arxiv.org/pdf/2103.01261)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Fast and light-weight methods for animating 3D characters are desirable in various applications such as computer games. We present a learning-based approach to enhance skinning-based animations of 3D characters with vivid secondary motion effects. We design a neural network that encodes each local patch of a character simulation mesh where the edges implicitly encode the internal forces between the neighboring vertices. The network emulates the ordinary differential equations of the character dynamics, predicting new vertex positions from the current accelerations, velocities and positions. Being a local method, our network is independent of the mesh topology and generalizes to arbitrarily shaped 3D character meshes at test time. We further represent per-vertex constraints and material properties such as stiffness, enabling us to easily adjust the dynamics in different parts of the mesh. We evaluate our method on various character meshes and complex motion sequences. Our method can be over 30 times more efficient than ground-truth physically based simulation, and outperforms alternative solutions that provide fast approximations.



