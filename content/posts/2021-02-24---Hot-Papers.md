---
title: Hot Papers 2021-02-24
date: 2021-02-25T10:09:08.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-24"
socialImage: "/media/flying-marine.jpg"

---

# 1. STEP: Segmenting and Tracking Every Pixel

Mark Weber, Jun Xie, Maxwell Collins, Yukun Zhu, Paul Voigtlaender, Hartwig Adam, Bradley Green, Andreas Geiger, Bastian Leibe, Daniel Cremers, Aljosa Osep, Laura Leal-Taixe, Liang-Chieh Chen

- retweets: 872, favorites: 161 (02/25/2021 10:09:08)

- links: [abs](https://arxiv.org/abs/2102.11859) | [pdf](https://arxiv.org/pdf/2102.11859)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we tackle video panoptic segmentation, a task that requires assigning semantic classes and track identities to all pixels in a video. To study this important problem in a setting that requires a continuous interpretation of sensory data, we present a new benchmark: Segmenting and Tracking Every Pixel (STEP), encompassing two datasets, KITTI-STEP, and MOTChallenge-STEP together with a new evaluation metric. Our work is the first that targets this task in a real-world setting that requires dense interpretation in both spatial and temporal domains. As the ground-truth for this task is difficult and expensive to obtain, existing datasets are either constructed synthetically or only sparsely annotated within short video clips. By contrast, our datasets contain long video sequences, providing challenging examples and a test-bed for studying long-term pixel-precise segmentation and tracking. For measuring the performance, we propose a novel evaluation metric Segmentation and Tracking Quality (STQ) that fairly balances semantic and tracking aspects of this task and is suitable for evaluating sequences of arbitrary length. We will make our datasets, metric, and baselines publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">STEP: Segmenting and Tracking Every Pixel<br>pdf: <a href="https://t.co/g53OM2BeOv">https://t.co/g53OM2BeOv</a><br>abs: <a href="https://t.co/omK7WlV80C">https://t.co/omK7WlV80C</a> <a href="https://t.co/DJ9EV8aZON">pic.twitter.com/DJ9EV8aZON</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364432386541449217?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Learning with User-Level Privacy

Daniel Levy, Ziteng Sun, Kareem Amin, Satyen Kale, Alex Kulesza, Mehryar Mohri, Ananda Theertha Suresh

- retweets: 242, favorites: 31 (02/25/2021 10:09:08)

- links: [abs](https://arxiv.org/abs/2102.11845) | [pdf](https://arxiv.org/pdf/2102.11845)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [math.OC](https://arxiv.org/list/math.OC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We propose and analyze algorithms to solve a range of learning tasks under user-level differential privacy constraints. Rather than guaranteeing only the privacy of individual samples, user-level DP protects a user's entire contribution ($m \ge 1$ samples), providing more stringent but more realistic protection against information leaks. We show that for high-dimensional mean estimation, empirical risk minimization with smooth losses, stochastic convex optimization, and learning hypothesis class with finite metric entropy, the privacy cost decreases as $O(1/\sqrt{m})$ as users provide more samples. In contrast, when increasing the number of users $n$, the privacy cost decreases at a faster $O(1/n)$ rate. We complement these results with lower bounds showing the worst-case optimality of our algorithm for mean estimation and stochastic convex optimization. Our algorithms rely on novel techniques for private mean estimation in arbitrary dimension with error scaling as the concentration radius $\tau$ of the distribution rather than the entire range. Under uniform convergence, we derive an algorithm that privately answers a sequence of $K$ adaptively chosen queries with privacy cost proportional to $\tau$, and apply it to solve the learning tasks we consider.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper out on learning under *user*-level differential privacy constraints! <a href="https://t.co/46D53iLCOL">https://t.co/46D53iLCOL</a><br><br>In the standard DP setting, we implicitly assume that each user contributes a single sample but it turns out we often contribute many many samples (like all of our texts). 1/</p>&mdash; Daniel Levy (@daniellevy__) <a href="https://twitter.com/daniellevy__/status/1364387994493280257?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Quantum query complexity with matrix-vector products

Andrew M. Childs, Shih-Han Hung, Tongyang Li

- retweets: 196, favorites: 55 (02/25/2021 10:09:08)

- links: [abs](https://arxiv.org/abs/2102.11349) | [pdf](https://arxiv.org/pdf/2102.11349)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.CC](https://arxiv.org/list/cs.CC/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent)

We study quantum algorithms that learn properties of a matrix using queries that return its action on an input vector. We show that for various problems, including computing the trace, determinant, or rank of a matrix or solving a linear system that it specifies, quantum computers do not provide an asymptotic speedup over classical computation. On the other hand, we show that for some problems, such as computing the parities of rows or columns or deciding if there are two identical rows or columns, quantum computers provide exponential speedup. We demonstrate this by showing equivalence between models that provide matrix-vector products, vector-matrix products, and vector-matrix-vector products, whereas the power of these models can vary significantly for classical computation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fun new paper with <a href="https://twitter.com/hungshihhan?ref_src=twsrc%5Etfw">@hungshihhan</a> and <a href="https://twitter.com/tongyang93?ref_src=twsrc%5Etfw">@tongyang93</a> on quantum query complexity with matrix-vector products. How fast can you learn properties of a matrix A with queries giving Ax for any vector x? Some problems have exponential speedup; others have none. <a href="https://t.co/zN6EPcXCQu">https://t.co/zN6EPcXCQu</a></p>&mdash; Andrew Childs (@andrewmchilds) <a href="https://twitter.com/andrewmchilds/status/1364583109132496897?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Three Ways to Solve Partial Differential Equations with Neural Networks  -- A Review

Jan Blechschmidt, Oliver G. Ernst

- retweets: 197, favorites: 48 (02/25/2021 10:09:09)

- links: [abs](https://arxiv.org/abs/2102.11802) | [pdf](https://arxiv.org/pdf/2102.11802)
- [math.NA](https://arxiv.org/list/math.NA/recent)

Neural networks are increasingly used to construct numerical solution methods for partial differential equations. In this expository review, we introduce and contrast three important recent approaches attractive in their simplicity and their suitability for high-dimensional problems: physics-informed neural networks, methods based on the Feynman-Kac formula and the Deep BSDE solver. The article is accompanied by a suite of expository software in the form of Jupyter notebooks in which each basic methodology is explained step by step, allowing for a quick assimilation and experimentation. An extensive bibliography summarizes the state of the art.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Three Ways to Solve Partial Differential Equations with Neural Networks — A Review&quot; (by Jan Blechschmidt, Oliver G. Ernst): <a href="https://t.co/Z9BlZw2uwa">https://t.co/Z9BlZw2uwa</a><br><br>&quot;The article is accompanied by a suite of expository software in the form of Jupyter notebooks&quot;</p>&mdash; DynamicalSystemsSIAM (@DynamicsSIAM) <a href="https://twitter.com/DynamicsSIAM/status/1364404825606090754?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Generative Modelling of BRDF Textures from Flash Images

Philipp Henzler, Valentin Deschaintre, Niloy J. Mitra, Tobias Ritschel

- retweets: 168, favorites: 44 (02/25/2021 10:09:09)

- links: [abs](https://arxiv.org/abs/2102.11861) | [pdf](https://arxiv.org/pdf/2102.11861)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We learn a latent space for easy capture, semantic editing, consistent interpolation, and efficient reproduction of visual material appearance. When users provide a photo of a stationary natural material captured under flash light illumination, it is converted in milliseconds into a latent material code. In a second step, conditioned on the material code, our method, again in milliseconds, produces an infinite and diverse spatial field of BRDF model parameters (diffuse albedo, specular albedo, roughness, normals) that allows rendering in complex scenes and illuminations, matching the appearance of the input picture. Technically, we jointly embed all flash images into a latent space using a convolutional encoder, and -- conditioned on these latent codes -- convert random spatial fields into fields of BRDF parameters using a convolutional neural network (CNN). We condition these BRDF parameters to match the visual characteristics (statistics and spectra of visual features) of the input under matching light. A user study confirms that the semantics of the latent material space agree with user expectations and compares our approach favorably to previous work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generative Modelling of BRDF Textures from Flash Images<br>pdf: <a href="https://t.co/M2CcdOfhj2">https://t.co/M2CcdOfhj2</a><br>abs: <a href="https://t.co/mlmJ9oba7H">https://t.co/mlmJ9oba7H</a><br>project page: <a href="https://t.co/m4xjzjj91R">https://t.co/m4xjzjj91R</a> <a href="https://t.co/4gJjlBItvR">pic.twitter.com/4gJjlBItvR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364396031534891009?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. UnsupervisedR&R: Unsupervised Point Cloud Registration via  Differentiable Rendering

Mohamed El Banani, Luya Gao, Justin Johnson

- retweets: 138, favorites: 60 (02/25/2021 10:09:09)

- links: [abs](https://arxiv.org/abs/2102.11870) | [pdf](https://arxiv.org/pdf/2102.11870)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Aligning partial views of a scene into a single whole is essential to understanding one's environment and is a key component of numerous robotics tasks such as SLAM and SfM. Recent approaches have proposed end-to-end systems that can outperform traditional methods by leveraging pose supervision. However, with the rising prevalence of cameras with depth sensors, we can expect a new stream of raw RGB-D data without the annotations needed for supervision. We propose UnsupervisedR&R: an end-to-end unsupervised approach to learning point cloud registration from raw RGB-D video. The key idea is to leverage differentiable alignment and rendering to enforce photometric and geometric consistency between frames. We evaluate our approach on indoor scene datasets and find that we outperform existing traditional approaches with classic and learned descriptors while being competitive with supervised geometric point cloud registration approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper (w/ Luya Gao and <a href="https://twitter.com/jcjohnss?ref_src=twsrc%5Etfw">@jcjohnss</a>) proposes an unsupervised approach to point cloud registration using RGB-D video. <br><br>project: <a href="https://t.co/sZTyrgvy22">https://t.co/sZTyrgvy22</a><br>paper: <a href="https://t.co/mP7SOHkYsn">https://t.co/mP7SOHkYsn</a><br>code: <a href="https://t.co/u28624kFBW">https://t.co/u28624kFBW</a> <a href="https://t.co/yoH5jihGGh">pic.twitter.com/yoH5jihGGh</a></p>&mdash; Mohamed El Banani (@_mbanani) <a href="https://twitter.com/_mbanani/status/1364649133110398978?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">UnsupervisedR&amp;R: Unsupervised Point Cloud Registration via Differentiable Rendering<br>pdf: <a href="https://t.co/xnNp7KN7Kt">https://t.co/xnNp7KN7Kt</a><br>abs: <a href="https://t.co/9SCWRQgitF">https://t.co/9SCWRQgitF</a><br>github: <a href="https://t.co/BEiakseW27">https://t.co/BEiakseW27</a> <a href="https://t.co/I7aNve6hsr">pic.twitter.com/I7aNve6hsr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364433486980018177?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Dynamic Neural Garments

Meng Zhang, Duygu Ceylan, Tuanfeng Wang, Niloy J. Mitra

- retweets: 121, favorites: 67 (02/25/2021 10:09:09)

- links: [abs](https://arxiv.org/abs/2102.11811) | [pdf](https://arxiv.org/pdf/2102.11811)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

A vital task of the wider digital human effort is the creation of realistic garments on digital avatars, both in the form of characteristic fold patterns and wrinkles in static frames as well as richness of garment dynamics under avatars' motion. Existing workflow of modeling, simulation, and rendering closely replicates the physics behind real garments, but is tedious and requires repeating most of the workflow under changes to characters' motion, camera angle, or garment resizing. Although data-driven solutions exist, they either focus on static scenarios or only handle dynamics of tight garments. We present a solution that, at test time, takes in body joint motion to directly produce realistic dynamic garment image sequences. Specifically, given the target joint motion sequence of an avatar, we propose dynamic neural garments to jointly simulate and render plausible dynamic garment appearance from an unseen viewpoint. Technically, our solution generates a coarse garment proxy sequence, learns deep dynamic features attached to this template, and neurally renders the features to produce appearance changes such as folds, wrinkles, and silhouettes. We demonstrate generalization behavior to both unseen motion and unseen camera views. Further, our network can be fine-tuned to adopt to new body shape and/or background images. We also provide comparisons against existing neural rendering and image sequence translation approaches, and report clear quantitative improvements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dynamic Neural Garments<br>pdf: <a href="https://t.co/eePMgvZKJs">https://t.co/eePMgvZKJs</a><br>abs: <a href="https://t.co/vnXq46d310">https://t.co/vnXq46d310</a> <a href="https://t.co/wZg14YhtB2">pic.twitter.com/wZg14YhtB2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364431826677338113?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Artificial Intelligence as an Anti-Corruption Tool (AI-ACT) --  Potentials and Pitfalls for Top-down and Bottom-up Approaches

Nils Köbis, Christopher Starke, Iyad Rahwan

- retweets: 130, favorites: 44 (02/25/2021 10:09:09)

- links: [abs](https://arxiv.org/abs/2102.11567) | [pdf](https://arxiv.org/pdf/2102.11567)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Corruption continues to be one of the biggest societal challenges of our time. New hope is placed in Artificial Intelligence (AI) to serve as an unbiased anti-corruption agent. Ever more available (open) government data paired with unprecedented performance of such algorithms render AI the next frontier in anti-corruption. Summarizing existing efforts to use AI-based anti-corruption tools (AI-ACT), we introduce a conceptual framework to advance research and policy. It outlines why AI presents a unique tool for top-down and bottom-up anti-corruption approaches. For both approaches, we outline in detail how AI-ACT present different potentials and pitfalls for (a) input data, (b) algorithmic design, and (c) institutional implementation. Finally, we venture a look into the future and flesh out key questions that need to be addressed to develop AI-ACT while considering citizens' views, hence putting "society in the loop".

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In a new preprint <a href="https://twitter.com/ch_starke?ref_src=twsrc%5Etfw">@ch_starke</a>, <a href="https://twitter.com/iyadrahwan?ref_src=twsrc%5Etfw">@iyadrahwan</a>, and I cover:<br>How to use <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> to tackle <a href="https://twitter.com/hashtag/corruption?src=hash&amp;ref_src=twsrc%5Etfw">#corruption</a>?<br><br>New potentials and pitfalls arise along: <br>- input data<br>- algorithmic design<br>- institutional implementation<br><br>Success requires &quot;society-in-the-loop&quot;<br> <a href="https://t.co/LUiliJPbMC">https://t.co/LUiliJPbMC</a><a href="https://twitter.com/hashtag/KIcamp2021?src=hash&amp;ref_src=twsrc%5Etfw">#KIcamp2021</a></p>&mdash; Nils Kobis (@NCKobis) <a href="https://twitter.com/NCKobis/status/1364548836073603072?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚨New pre-print on <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> for Anti-Corruption🚨<br><br>We outline a conceptual framework along 3 dimensions: <br><br>1) input data<br>2) algorithmic design<br>3) institutional implementation<br><br>It was much fun working with <a href="https://twitter.com/NCKobis?ref_src=twsrc%5Etfw">@NCKobis</a> &amp; <a href="https://twitter.com/iyadrahwan?ref_src=twsrc%5Etfw">@iyadrahwan</a> on this project!<a href="https://t.co/YWUmaSkLS0">https://t.co/YWUmaSkLS0</a> <a href="https://t.co/JeT9a3d3ud">pic.twitter.com/JeT9a3d3ud</a></p>&mdash; Christopher Starke (@ch_starke) <a href="https://twitter.com/ch_starke/status/1364557768867405827?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. The SmartSHARK Repository Mining Data

Alexander Trautsch, Steffen Herbold

- retweets: 64, favorites: 38 (02/25/2021 10:09:10)

- links: [abs](https://arxiv.org/abs/2102.11540) | [pdf](https://arxiv.org/pdf/2102.11540)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

The SmartSHARK repository mining data is a collection of rich and detailed information about the evolution of software projects. The data is unique in its diversity and contains detailed information about each change, issue tracking data, continuous integration data, as well as pull request and code review data. Moreover, the data does not contain only raw data scraped from repositories, but also annotations in form of labels determined through a combination of manual analysis and heuristics, as well as links between the different parts of the data set. The SmartSHARK data set provides a rich source of data that enables us to explore research questions that require data from different sources and/or longitudinal data over time.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are happy to share our rejected <a href="https://twitter.com/msrconf?ref_src=twsrc%5Etfw">@msrconf</a> data showcase.<br><br>Our data combines the commit and code history, issue tracking (Jira+GH Issues), pull requests, and travis logs (if used) of 69 Apache projects.<br><br>A thread for details (1/n)<a href="https://t.co/1mRR5LVHyI">https://t.co/1mRR5LVHyI</a></p>&mdash; Steffen Herbold (@HerboldSteffen) <a href="https://twitter.com/HerboldSteffen/status/1364476257770037249?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Assigning Confidence to Molecular Property Prediction

AkshatKumar Nigam, Robert Pollice, Matthew F. D. Hurley, Riley J. Hickman, Matteo Aldeghi, Naruki Yoshikawa, Seyone Chithrananda, Vincent A. Voelz, Alán Aspuru-Guzik

- retweets: 64, favorites: 35 (02/25/2021 10:09:10)

- links: [abs](https://arxiv.org/abs/2102.11439) | [pdf](https://arxiv.org/pdf/2102.11439)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Introduction: Computational modeling has rapidly advanced over the last decades, especially to predict molecular properties for chemistry, material science and drug design. Recently, machine learning techniques have emerged as a powerful and cost-effective strategy to learn from existing datasets and perform predictions on unseen molecules. Accordingly, the explosive rise of data-driven techniques raises an important question: What confidence can be assigned to molecular property predictions and what techniques can be used for that purpose?   Areas covered: In this work, we discuss popular strategies for predicting molecular properties relevant to drug design, their corresponding uncertainty sources and methods to quantify uncertainty and confidence. First, our considerations for assessing confidence begin with dataset bias and size, data-driven property prediction and feature design. Next, we discuss property simulation via molecular docking, and free-energy simulations of binding affinity in detail. Lastly, we investigate how these uncertainties propagate to generative models, as they are usually coupled with property predictors.   Expert opinion: Computational techniques are paramount to reduce the prohibitive cost and timing of brute-force experimentation when exploring the enormous chemical space. We believe that assessing uncertainty in property prediction models is essential whenever closed-loop drug design campaigns relying on high-throughput virtual screening are deployed. Accordingly, considering sources of uncertainty leads to better-informed experimental validations, more reliable predictions and to more realistic expectations of the entire workflow. Overall, this increases confidence in the predictions and designs and, ultimately, accelerates drug design.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our perspective on &quot;Assigning Confidence to Molecular Property Prediction&quot; 😀<a href="https://t.co/ANWySFewQL">https://t.co/ANWySFewQL</a><a href="https://twitter.com/hashtag/matterlab?src=hash&amp;ref_src=twsrc%5Etfw">#matterlab</a> <a href="https://twitter.com/UofTCompSci?ref_src=twsrc%5Etfw">@UofTCompSci</a> <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@UofT</a> <a href="https://twitter.com/voelzlab?ref_src=twsrc%5Etfw">@voelzlab</a> <a href="https://t.co/sWnGZkGWSs">pic.twitter.com/sWnGZkGWSs</a></p>&mdash; Akshat Nigam (@akshat_ai) <a href="https://twitter.com/akshat_ai/status/1364413310364504075?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Memory-efficient Speech Recognition on Smart Devices

Ganesh Venkatesh, Alagappan Valliappan, Jay Mahadeokar, Yuan Shangguan, Christian Fuegen, Michael L. Seltzer, Vikas Chandra

- retweets: 42, favorites: 35 (02/25/2021 10:09:10)

- links: [abs](https://arxiv.org/abs/2102.11531) | [pdf](https://arxiv.org/pdf/2102.11531)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Recurrent transducer models have emerged as a promising solution for speech recognition on the current and next generation smart devices. The transducer models provide competitive accuracy within a reasonable memory footprint alleviating the memory capacity constraints in these devices. However, these models access parameters from off-chip memory for every input time step which adversely effects device battery life and limits their usability on low-power devices.   We address transducer model's memory access concerns by optimizing their model architecture and designing novel recurrent cell designs. We demonstrate that i) model's energy cost is dominated by accessing model weights from off-chip memory, ii) transducer model architecture is pivotal in determining the number of accesses to off-chip memory and just model size is not a good proxy, iii) our transducer model optimizations and novel recurrent cell reduces off-chip memory accesses by 4.5x and model size by 2x with minimal accuracy impact.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Memory-efficient Speech Recognition on Smart Devices<br>pdf: <a href="https://t.co/deTTSTLIPc">https://t.co/deTTSTLIPc</a><br>abs: <a href="https://t.co/f3tMD4Y52i">https://t.co/f3tMD4Y52i</a> <a href="https://t.co/CO8xsZhSS6">pic.twitter.com/CO8xsZhSS6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364402584082124801?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Multi-Knowledge Fusion for New Feature Generation in Generalized  Zero-Shot Learning

Hongxin Xiang, Cheng Xie, Ting Zeng, Yun Yang

- retweets: 42, favorites: 20 (02/25/2021 10:09:10)

- links: [abs](https://arxiv.org/abs/2102.11566) | [pdf](https://arxiv.org/pdf/2102.11566)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Suffering from the semantic insufficiency and domain-shift problems, most of existing state-of-the-art methods fail to achieve satisfactory results for Zero-Shot Learning (ZSL). In order to alleviate these problems, we propose a novel generative ZSL method to learn more generalized features from multi-knowledge with continuously generated new semantics in semantic-to-visual embedding. In our approach, the proposed Multi-Knowledge Fusion Network (MKFNet) takes different semantic features from multi-knowledge as input, which enables more relevant semantic features to be trained for semantic-to-visual embedding, and finally generates more generalized visual features by adaptively fusing visual features from different knowledge domain. The proposed New Feature Generator (NFG) with adaptive genetic strategy is used to enrich semantic information on the one hand, and on the other hand it greatly improves the intersection of visual feature generated by MKFNet and unseen visual faetures. Empirically, we show that our approach can achieve significantly better performance compared to existing state-of-the-art methods on a large number of benchmarks for several ZSL tasks, including traditional ZSL, generalized ZSL and zero-shot retrieval.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Knowledge Fusion for New Feature Generation in Generalized Zero-Shot Learning<br>pdf: <a href="https://t.co/I6WrP4iL37">https://t.co/I6WrP4iL37</a><br>abs: <a href="https://t.co/roKVRkcPp9">https://t.co/roKVRkcPp9</a> <a href="https://t.co/qnjRRNmI9n">pic.twitter.com/qnjRRNmI9n</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364392312864989187?ref_src=twsrc%5Etfw">February 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



