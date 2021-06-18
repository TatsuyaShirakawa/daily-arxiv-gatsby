---
title: Hot Papers 2021-06-17
date: 2021-06-18T09:59:33.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-17"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-17"
socialImage: "/media/flying-marine.jpg"

---

# 1. Efficient Deep Learning: A Survey on Making Deep Learning Models  Smaller, Faster, and Better

Gaurav Menghani

- retweets: 6072, favorites: 277 (06/18/2021 09:59:33)

- links: [abs](https://arxiv.org/abs/2106.08962) | [pdf](https://arxiv.org/pdf/2106.08962)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep Learning has revolutionized the fields of computer vision, natural language understanding, speech recognition, information retrieval and more. However, with the progressive improvements in deep learning models, their number of parameters, latency, resources required to train, etc. have all have increased significantly. Consequently, it has become important to pay attention to these footprint metrics of a model as well, not just its quality. We present and motivate the problem of efficiency in deep learning, followed by a thorough survey of the five core areas of model efficiency (spanning modeling techniques, infrastructure, and hardware) and the seminal work there. We also present an experiment-based guide along with code, for practitioners to optimize their model training and deployment. We believe this is the first comprehensive survey in the efficient deep learning space that covers the landscape of model efficiency from modeling techniques to hardware support. Our hope is that this survey would provide the reader with the mental model and the necessary understanding of the field to apply generic efficiency techniques to immediately get significant improvements, and also equip them with ideas for further research and experimentation to achieve additional gains.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient Deep Learning<br><br>This survey provides a comprehensive overview of techniques to make deep learning models smaller, faster, and better. <br><br>A lot of actionable insights in this report.<br><br>A great read for machine learning practitioners.<a href="https://t.co/0Zfghnmwje">https://t.co/0Zfghnmwje</a> <a href="https://t.co/pWfoIaCBVg">pic.twitter.com/pWfoIaCBVg</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1405491104150003714?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Predicting Unreliable Predictions by Shattering a Neural Network

Xu Ji, Razvan Pascanu, Devon Hjelm, Andrea Vedaldi, Balaji Lakshminarayanan, Yoshua Bengio

- retweets: 327, favorites: 110 (06/18/2021 09:59:33)

- links: [abs](https://arxiv.org/abs/2106.08365) | [pdf](https://arxiv.org/pdf/2106.08365)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Piecewise linear neural networks can be split into subfunctions, each with its own activation pattern, domain, and empirical error. Empirical error for the full network can be written as an expectation over empirical error of subfunctions. Constructing a generalization bound on subfunction empirical error indicates that the more densely a subfunction is surrounded by training samples in representation space, the more reliable its predictions are. Further, it suggests that models with fewer activation regions generalize better, and models that abstract knowledge to a greater degree generalize better, all else equal. We propose not only a theoretical framework to reason about subfunction error bounds but also a pragmatic way of approximately evaluating it, which we apply to predicting which samples the network will not successfully generalize to. We test our method on detection of misclassification and out-of-distribution samples, finding that it performs competitively in both cases. In short, some network activation patterns are associated with higher reliability than others, and these can be identified using subfunction error bounds.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My new work on neural network generalization ✨<br>Predicting Unreliable Predictions by Shattering a Neural Network<br>Me, <a href="https://twitter.com/rpascanu?ref_src=twsrc%5Etfw">@rpascanu</a>, <a href="https://twitter.com/devon_hjelm?ref_src=twsrc%5Etfw">@devon_hjelm</a>, Andrea Vedaldi, <a href="https://twitter.com/balajiln?ref_src=twsrc%5Etfw">@balajiln</a>, Yoshua Bengio <a href="https://t.co/GcoXO37zwY">https://t.co/GcoXO37zwY</a></p>&mdash; Xu Ji (@xu__ji) <a href="https://twitter.com/xu__ji/status/1405469489936977922?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Predicting Unreliable Predictions by Shattering a Neural Network<br>pdf: <a href="https://t.co/Oa9C9bXPKP">https://t.co/Oa9C9bXPKP</a><br>abs: <a href="https://t.co/C5Ptb7KS3j">https://t.co/C5Ptb7KS3j</a><br>github: <a href="https://t.co/ah4i8J7KRF">https://t.co/ah4i8J7KRF</a> <a href="https://t.co/xX3uyXFq1W">pic.twitter.com/xX3uyXFq1W</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405382479129755655?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks  Trained from Scratch

Hossein Souri, Micah Goldblum, Liam Fowl, Rama Chellappa, Tom Goldstein

- retweets: 361, favorites: 71 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.08970) | [pdf](https://arxiv.org/pdf/2106.08970)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

As the curation of data for machine learning becomes increasingly automated, dataset tampering is a mounting threat. Backdoor attackers tamper with training data to embed a vulnerability in models that are trained on that data. This vulnerability is then activated at inference time by placing a "trigger" into the model's input. Typical backdoor attacks insert the trigger directly into the training data, although the presence of such an attack may be visible upon inspection. In contrast, the Hidden Trigger Backdoor Attack achieves poisoning without placing a trigger into the training data at all. However, this hidden trigger attack is ineffective at poisoning neural networks trained from scratch. We develop a new hidden trigger attack, Sleeper Agent, which employs gradient matching, data selection, and target model re-training during the crafting process. Sleeper Agent is the first hidden trigger backdoor attack to be effective against neural networks trained from scratch. We demonstrate its effectiveness on ImageNet and in black-box settings. Our implementation code can be found at https://github.com/hsouri/Sleeper-Agent.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch<br>pdf: <a href="https://t.co/rufBvH6bzN">https://t.co/rufBvH6bzN</a><br>abs: <a href="https://t.co/TGq11vRQxv">https://t.co/TGq11vRQxv</a><br><br>employs gradient matching, data selection, and target model re-training during the crafting process <a href="https://t.co/eMUD8Fj6yj">pic.twitter.com/eMUD8Fj6yj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405367567447904259?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Invertible Attention

Jiajun Zha, Yiran Zhong, Jing Zhang, Liang Zheng, Richard Hartley

- retweets: 240, favorites: 97 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.09003) | [pdf](https://arxiv.org/pdf/2106.09003)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Attention has been proved to be an efficient mechanism to capture long-range dependencies. However, so far it has not been deployed in invertible networks. This is due to the fact that in order to make a network invertible, every component within the network needs to be a bijective transformation, but a normal attention block is not. In this paper, we propose invertible attention that can be plugged into existing invertible models. We mathematically and experimentally prove that the invertibility of an attention model can be achieved by carefully constraining its Lipschitz constant. We validate the invertibility of our invertible attention on image reconstruction task with 3 popular datasets: CIFAR-10, SVHN, and CelebA. We also show that our invertible attention achieves similar performance in comparison with normal non-invertible attention on dense prediction tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Invertible Attention<br>pdf: <a href="https://t.co/irw3HXeHMl">https://t.co/irw3HXeHMl</a><br>abs: <a href="https://t.co/ssQ9fom7Eo">https://t.co/ssQ9fom7Eo</a><br><br>invertibility of an attention model can be achieved by carefully constraining its Lipschitz constant <a href="https://t.co/6Xo5cYuQJ4">pic.twitter.com/6Xo5cYuQJ4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405369682023043075?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. KALE Flow: A Relaxed KL Gradient Flow for Probabilities with Disjoint  Support

Pierre Glaser, Michael Arbel, Arthur Gretton

- retweets: 210, favorites: 101 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.08929) | [pdf](https://arxiv.org/pdf/2106.08929)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We study the gradient flow for a relaxed approximation to the Kullback-Leibler (KL) divergence between a moving source and a fixed target distribution. This approximation, termed the KALE (KL approximate lower-bound estimator), solves a regularized version of the Fenchel dual problem defining the KL over a restricted class of functions. When using a Reproducing Kernel Hilbert Space (RKHS) to define the function class, we show that the KALE continuously interpolates between the KL and the Maximum Mean Discrepancy (MMD). Like the MMD and other Integral Probability Metrics, the KALE remains well defined for mutually singular distributions. Nonetheless, the KALE inherits from the limiting KL a greater sensitivity to mismatch in the support of the distributions, compared with the MMD. These two properties make the KALE gradient flow particularly well suited when the target distribution is supported on a low-dimensional manifold. Under an assumption of sufficient smoothness of the trajectories, we show the global convergence of the KALE flow. We propose a particle implementation of the flow given initial samples from the source and the target distribution, which we use to empirically confirm the KALE's properties.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wasserstein gradient flow for KALE 🥬 (KL approximate lower-bound estimator). The 🥬 &quot;interpolates&quot; between KL and MMD.  Unlike KL, the 🥬 gradient flow is defined for source/target distributions with disjoint support. <a href="https://t.co/C0XVHrhwcM">https://t.co/C0XVHrhwcM</a><br>with <a href="https://twitter.com/PierreGlaser?ref_src=twsrc%5Etfw">@PierreGlaser</a> and <a href="https://twitter.com/MichaelArbel?ref_src=twsrc%5Etfw">@MichaelArbel</a></p>&mdash; Arthur Gretton (@ArthurGretton) <a href="https://twitter.com/ArthurGretton/status/1405539445395181578?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Watching Too Much Television is Good: Self-Supervised Audio-Visual  Representation Learning from Movies and TV Shows

Mahdi M. Kalayeh, Nagendra Kamath, Lingyi Liu, Ashok Chandrashekar

- retweets: 158, favorites: 90 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.08513) | [pdf](https://arxiv.org/pdf/2106.08513)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The abundance and ease of utilizing sound, along with the fact that auditory clues reveal so much about what happens in the scene, make the audio-visual space a perfectly intuitive choice for self-supervised representation learning. However, the current literature suggests that training on \textit{uncurated} data yields considerably poorer representations compared to the \textit{curated} alternatives collected in supervised manner, and the gap only narrows when the volume of data significantly increases. Furthermore, the quality of learned representations is known to be heavily influenced by the size and taxonomy of the curated datasets used for self-supervised training. This begs the question of whether we are celebrating too early on catching up with supervised learning when our self-supervised efforts still rely almost exclusively on curated data. In this paper, we study the efficacy of learning from Movies and TV Shows as forms of uncurated data for audio-visual self-supervised learning. We demonstrate that a simple model based on contrastive learning, trained on a collection of movies and TV shows, not only dramatically outperforms more complex methods which are trained on orders of magnitude larger uncurated datasets, but also performs very competitively with the state-of-the-art that learns from large-scale curated data. We identify that audiovisual patterns like the appearance of the main character or prominent scenes and mise-en-sc\`ene which frequently occur through the whole duration of a movie, lead to an overabundance of easy negative instances in the contrastive learning formulation. Capitalizing on such observation, we propose a hierarchical sampling policy, which despite its simplicity, effectively improves the performance, particularly when learning from TV shows which naturally face less semantic diversity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Watching Too Much Television is Good: Self-Supervised Audio-Visual Representation Learning from Movies and TV Shows<br>pdf: <a href="https://t.co/hBN7Um0RMm">https://t.co/hBN7Um0RMm</a><br>abs: <a href="https://t.co/cRXjYZzUhy">https://t.co/cRXjYZzUhy</a> <a href="https://t.co/t56CsoEYPx">pic.twitter.com/t56CsoEYPx</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405334444450459648?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. A learning agent that acquires social norms from public sanctions in  decentralized multi-agent settings

Eugene Vinitsky, Raphael Köster, John P. Agapiou, Edgar Duéñez-Guzmán, Alexander Sasha Vezhnevets, Joel Z. Leibo

- retweets: 112, favorites: 79 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.09012) | [pdf](https://arxiv.org/pdf/2106.09012)
- [cs.MA](https://arxiv.org/list/cs.MA/recent)

Society is characterized by the presence of a variety of social norms: collective patterns of sanctioning that can prevent miscoordination and free-riding. Inspired by this, we aim to construct learning dynamics where potentially beneficial social norms can emerge. Since social norms are underpinned by sanctioning, we introduce a training regime where agents can access all sanctioning events but learning is otherwise decentralized. This setting is technologically interesting because sanctioning events may be the only available public signal in decentralized multi-agent systems where reward or policy-sharing is infeasible or undesirable. To achieve collective action in this setting we construct an agent architecture containing a classifier module that categorizes observed behaviors as approved or disapproved, and a motivation to punish in accord with the group. We show that social norms emerge in multi-agent systems containing this agent and investigate the conditions under which this helps them achieve socially beneficial outcomes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In new work with <a href="https://twitter.com/DeepMind?ref_src=twsrc%5Etfw">@DeepMind</a> and <a href="https://twitter.com/jzl86?ref_src=twsrc%5Etfw">@jzl86</a>, we investigate how decentralized agents can learn social norms to help them overcome social dilemmas. We introduce a setting in which rewards and policies are private but sanctioning behavior is public<a href="https://t.co/AYHPltw01O">https://t.co/AYHPltw01O</a> <a href="https://t.co/LkYVGZ0tTG">pic.twitter.com/LkYVGZ0tTG</a></p>&mdash; Eugene Vinitsky (@EugeneVinitsky) <a href="https://twitter.com/EugeneVinitsky/status/1405577538852864004?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Smoothing the Disentangled Latent Style Space for Unsupervised  Image-to-Image Translation

Yahui Liu, Enver Sangineto, Yajing Chen, Linchao Bao, Haoxian Zhang, Nicu Sebe, Bruno Lepri, Wei Wang, Marco De Nadai

- retweets: 110, favorites: 61 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.09016) | [pdf](https://arxiv.org/pdf/2106.09016)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Image-to-Image (I2I) multi-domain translation models are usually evaluated also using the quality of their semantic interpolation results. However, state-of-the-art models frequently show abrupt changes in the image appearance during interpolation, and usually perform poorly in interpolations across domains. In this paper, we propose a new training protocol based on three specific losses which help a translation network to learn a smooth and disentangled latent style space in which: 1) Both intra- and inter-domain interpolations correspond to gradual changes in the generated images and 2) The content of the source image is better preserved during the translation. Moreover, we propose a novel evaluation metric to properly measure the smoothness of latent style space of I2I translation models. The proposed method can be plugged into existing translation approaches, and our extensive experiments on different datasets show that it can significantly boost the quality of the generated images and the graduality of the interpolations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Smoothing the Disentangled Latent Style Space for<br>Unsupervised Image-to-Image Translation<br>pdf: <a href="https://t.co/MpS0R0fgUU">https://t.co/MpS0R0fgUU</a><br>abs: <a href="https://t.co/IPOhpU3imi">https://t.co/IPOhpU3imi</a> <a href="https://t.co/ykvJ9IWFHM">pic.twitter.com/ykvJ9IWFHM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405333134057615360?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. The Oxford Road Boundaries Dataset

Tarlan Suleymanov, Matthew Gadd, Daniele De Martini, Paul Newman

- retweets: 92, favorites: 51 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.08983) | [pdf](https://arxiv.org/pdf/2106.08983)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

In this paper we present the Oxford Road Boundaries Dataset, designed for training and testing machine-learning-based road-boundary detection and inference approaches. We have hand-annotated two of the 10 km-long forays from the Oxford Robotcar Dataset and generated from other forays several thousand further examples with semi-annotated road-boundary masks. To boost the number of training samples in this way, we used a vision-based localiser to project labels from the annotated datasets to other traversals at different times and weather conditions. As a result, we release 62605 labelled samples, of which 47639 samples are curated. Each of these samples contains both raw and classified masks for left and right lenses. Our data contains images from a diverse set of scenarios such as straight roads, parked cars, junctions, etc. Files for download and tools for manipulating the labelled data are available at: oxford-robotics-institute.github.io/road-boundaries-dataset

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Oxford Road Boundaries Dataset<br>pdf: <a href="https://t.co/AgwpTMo9b5">https://t.co/AgwpTMo9b5</a><br>abs: <a href="https://t.co/pKy2iq4mUm">https://t.co/pKy2iq4mUm</a><br>project page: <a href="https://t.co/L6nLLNDKlm">https://t.co/L6nLLNDKlm</a> <a href="https://t.co/hozvCfPufT">pic.twitter.com/hozvCfPufT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405368362365034496?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. OpenSSLNTRU: Faster post-quantum TLS key exchange

Daniel J. Bernstein, Billy Bob Brumley, Ming-Shing Chen, Nicola Tuveri

- retweets: 112, favorites: 17 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.08759) | [pdf](https://arxiv.org/pdf/2106.08759)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

Google's CECPQ1 experiment in 2016 integrated a post-quantum key-exchange algorithm, newhope1024, into TLS 1.2. The Google-Cloudflare CECPQ2 experiment in 2019 integrated a more efficient key-exchange algorithm, ntruhrss701, into TLS 1.3.   This paper revisits the choices made in CECPQ2, and shows how to achieve higher performance for post-quantum key exchange in TLS 1.3 using a higher-security algorithm, sntrup761. Previous work had indicated that ntruhrss701 key generation was much faster than sntrup761 key generation, but this paper makes sntrup761 key generation much faster by generating a batch of keys at once.   Batch key generation is invisible at the TLS protocol layer, but raises software-engineering questions regarding the difficulty of integrating batch key exchange into existing TLS libraries and applications. This paper shows that careful choices of software layers make it easy to integrate fast post-quantum software, including batch key exchange, into TLS with minor changes to TLS libraries and no changes to applications.   As a demonstration of feasibility, this paper reports successful integration of its fast sntrup761 library, via a lightly patched OpenSSL, into an unmodified web browser and an unmodified TLS terminator. This paper also reports TLS 1.3 handshake benchmarks, achieving more TLS 1.3 handshakes per second than any software included in OpenSSL.




# 11. Differentiable Diffusion for Dense Depth Estimation from Multi-view  Images

Numair Khan, Min H. Kim, James Tompkin

- retweets: 81, favorites: 33 (06/18/2021 09:59:34)

- links: [abs](https://arxiv.org/abs/2106.08917) | [pdf](https://arxiv.org/pdf/2106.08917)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a method to estimate dense depth by optimizing a sparse set of points such that their diffusion into a depth map minimizes a multi-view reprojection error from RGB supervision. We optimize point positions, depths, and weights with respect to the loss by differential splatting that models points as Gaussians with analytic transmittance. Further, we develop an efficient optimization routine that can simultaneously optimize the 50k+ points required for complex scene reconstruction. We validate our routine using ground truth data and show high reconstruction quality. Then, we apply this to light field and wider baseline images via self supervision, and show improvements in both average and outlier error for depth maps diffused from inaccurate sparse points. Finally, we compare qualitative and quantitative results to image processing and deep learning methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Differentiable Diffusion for Dense Depth Estimation from Multi-view Images<br>pdf: <a href="https://t.co/SMrUxwmOAB">https://t.co/SMrUxwmOAB</a><br>abs: <a href="https://t.co/Jr9BwfvVYV">https://t.co/Jr9BwfvVYV</a><br>project page: <a href="https://t.co/SEpXyHgBrI">https://t.co/SEpXyHgBrI</a><br>github: <a href="https://t.co/VaAbJnPzY6">https://t.co/VaAbJnPzY6</a> <a href="https://t.co/OVK9xSA5h1">pic.twitter.com/OVK9xSA5h1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405364716667650055?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Manifolds.jl: An Extensible Julia Framework for Data Analysis on  Manifolds

Seth D. Axen, Mateusz Baran, Ronny Bergmann, Krzysztof Rzecki

- retweets: 50, favorites: 27 (06/18/2021 09:59:35)

- links: [abs](https://arxiv.org/abs/2106.08777) | [pdf](https://arxiv.org/pdf/2106.08777)
- [cs.MS](https://arxiv.org/list/cs.MS/recent)

For data given on a nonlinear space, like angles, symmetric positive matrices, the sphere, or the hyperbolic space, there is often enough structure to form a Riemannian manifold. We present the Julia package Manifolds.jl, providing a fast and easy to use library of Riemannian manifolds and Lie groups. We introduce a common interface, available in ManifoldsBase.jl, with which new manifolds, applications, and algorithms can be implemented. We demonstrate the utility of Manifolds.jl using B\'ezier splines, an optimization task on manifolds, and a principal component analysis on nonlinear data. In a benchmark, Manifolds.jl outperforms existing packages in Matlab or Python by several orders of magnitude and is about twice as fast as a comparable package implemented in C++.




# 13. Scene Transformer: A unified multi-task model for behavior prediction  and planning

Jiquan Ngiam, Benjamin Caine, Vijay Vasudevan, Zhengdong Zhang, Hao-Tien Lewis Chiang, Jeffrey Ling, Rebecca Roelofs, Alex Bewley, Chenxi Liu, Ashish Venugopal, David Weiss, Ben Sapp, Zhifeng Chen, Jonathon Shlens

- retweets: 42, favorites: 30 (06/18/2021 09:59:35)

- links: [abs](https://arxiv.org/abs/2106.08417) | [pdf](https://arxiv.org/pdf/2106.08417)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Predicting the future motion of multiple agents is necessary for planning in dynamic environments. This task is challenging for autonomous driving since agents (e.g., vehicles and pedestrians) and their associated behaviors may be diverse and influence each other. Most prior work has focused on first predicting independent futures for each agent based on all past motion, and then planning against these independent predictions. However, planning against fixed predictions can suffer from the inability to represent the future interaction possibilities between different agents, leading to sub-optimal planning. In this work, we formulate a model for predicting the behavior of all agents jointly in real-world driving environments in a unified manner. Inspired by recent language modeling approaches, we use a masking strategy as the query to our model, enabling one to invoke a single model to predict agent behavior in many ways, such as potentially conditioned on the goal or full future trajectory of the autonomous vehicle or the behavior of other agents in the environment. Our model architecture fuses heterogeneous world state in a unified Transformer architecture by employing attention across road elements, agent interactions and time steps. We evaluate our approach on autonomous driving datasets for behavior prediction, and achieve state-of-the-art performance. Our work demonstrates that formulating the problem of behavior prediction in a unified architecture with a masking strategy may allow us to have a single model that can perform multiple motion prediction and planning related tasks effectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scene Transformer: A unified multi-task model<br>for behavior prediction and planning<br>pdf: <a href="https://t.co/zp2BV1j2sL">https://t.co/zp2BV1j2sL</a><br>abs: <a href="https://t.co/bAb3fJ12le">https://t.co/bAb3fJ12le</a><br><br>a model for predicting the behavior of all agents jointly in real-world driving environments in a unified manner <a href="https://t.co/zqsfvXyZrU">pic.twitter.com/zqsfvXyZrU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405327382307917824?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. WSRGlow: A Glow-based Waveform Generative Model for Audio  Super-Resolution

Kexun Zhang, Yi Ren, Changliang Xu, Zhou Zhao

- retweets: 20, favorites: 32 (06/18/2021 09:59:35)

- links: [abs](https://arxiv.org/abs/2106.08507) | [pdf](https://arxiv.org/pdf/2106.08507)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Audio super-resolution is the task of constructing a high-resolution (HR) audio from a low-resolution (LR) audio by adding the missing band. Previous methods based on convolutional neural networks and mean squared error training objective have relatively low performance, while adversarial generative models are difficult to train and tune. Recently, normalizing flow has attracted a lot of attention for its high performance, simple training and fast inference. In this paper, we propose WSRGlow, a Glow-based waveform generative model to perform audio super-resolution. Specifically, 1) we integrate WaveNet and Glow to directly maximize the exact likelihood of the target HR audio conditioned on LR information; and 2) to exploit the audio information from low-resolution audio, we propose an LR audio encoder and an STFT encoder, which encode the LR information from the time domain and frequency domain respectively. The experimental results show that the proposed model is easier to train and outperforms the previous works in terms of both objective and perceptual quality. WSRGlow is also the first model to produce 48kHz waveforms from 12kHz LR audio.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">WSRGlow: A Glow-based Waveform Generative Model for Audio Super-Resolution<br>pdf: <a href="https://t.co/f1kljovq7v">https://t.co/f1kljovq7v</a><br>abs: <a href="https://t.co/If1QIqSADh">https://t.co/If1QIqSADh</a><br>project page: <a href="https://t.co/6BzBkmsITM">https://t.co/6BzBkmsITM</a> <a href="https://t.co/vAVjem8z9W">pic.twitter.com/vAVjem8z9W</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405342278013624322?ref_src=twsrc%5Etfw">June 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



