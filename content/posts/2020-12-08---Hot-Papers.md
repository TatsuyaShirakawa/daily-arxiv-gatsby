---
title: Hot Papers 2020-12-08
date: 2020-12-09T11:29:21.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-08"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-08"
socialImage: "/media/flying-marine.jpg"

---

# 1. A bounded-noise mechanism for differential privacy

Yuval Dagan, Gil Kur

- retweets: 2147, favorites: 530 (12/09/2020 11:29:21)

- links: [abs](https://arxiv.org/abs/2012.03817) | [pdf](https://arxiv.org/pdf/2012.03817)
- [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Answering multiple counting queries is one of the best-studied problems in differential privacy. Its goal is to output an approximation of the average $\frac{1}{n}\sum_{i=1}^n \vec{x}^{(i)}$ of vectors $\vec{x}^{(i)} \in [0,1]^k$, while preserving the privacy with respect to any $\vec{x}^{(i)}$. We present an $(\epsilon,\delta)$-private mechanism with optimal $\ell_\infty$ error for most values of $\delta$. This result settles the conjecture of Steinke and Ullman [2020] for the these values of $\delta$. Our algorithm adds independent noise of bounded magnitude to each of the $k$ coordinates, while prior solutions relied on unbounded noise such as the Laplace and Gaussian mechanisms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">OH MY GOD!<br>It has been solved. <br>🎉🎉🎉🎉🎉🎉🎉<a href="https://t.co/HJmA70ykxF">https://t.co/HJmA70ykxF</a> <a href="https://t.co/cd9xkydd3V">pic.twitter.com/cd9xkydd3V</a></p>&mdash; Thomas Steinke (@shortstein) <a href="https://twitter.com/shortstein/status/1336167223308607489?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Amazing! Preprint by Yuval Dagan (<a href="https://twitter.com/YuvalDagan3?ref_src=twsrc%5Etfw">@YuvalDagan3</a>) &amp; Gil Kur (<a href="https://twitter.com/GilKur1?ref_src=twsrc%5Etfw">@GilKur1</a>) solves an open problem of Steinke (<a href="https://twitter.com/shortstein?ref_src=twsrc%5Etfw">@shortstein</a>) &amp; Ullman (<a href="https://twitter.com/thejonullman?ref_src=twsrc%5Etfw">@thejonullman</a>)! They shaved the last sqrt(log log log k) factor for answering k queries, winning themselves a sushi dinner! 🍣🍣🍣<a href="https://t.co/9edkfWrbJb">https://t.co/9edkfWrbJb</a> <a href="https://t.co/euzFGlMTae">pic.twitter.com/euzFGlMTae</a></p>&mdash; Gautam Kamath ✈️ NeurIPS 2020 (@thegautamkamath) <a href="https://twitter.com/thegautamkamath/status/1336166826888183810?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Selective Inference for Hierarchical Clustering

Lucy L. Gao, Jacob Bien, Daniela Witten

- retweets: 2254, favorites: 218 (12/09/2020 11:29:21)

- links: [abs](https://arxiv.org/abs/2012.02936) | [pdf](https://arxiv.org/pdf/2012.02936)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Testing for a difference in means between two groups is fundamental to answering research questions across virtually every scientific area. Classical tests control the Type I error rate when the groups are defined a priori. However, when the groups are instead defined via a clustering algorithm, then applying a classical test for a difference in means between the groups yields an extremely inflated Type I error rate. Notably, this problem persists even if two separate and independent data sets are used to define the groups and to test for a difference in their means. To address this problem, in this paper, we propose a selective inference approach to test for a difference in means between two clusters obtained from any clustering method. Our procedure controls the selective Type I error rate by accounting for the fact that the null hypothesis was generated from the data. We describe how to efficiently compute exact p-values for clusters obtained using agglomerative hierarchical clustering with many commonly used linkages. We apply our method to simulated data and to single-cell RNA-seq data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our preprint on fixing double-dipping in the clustering setting is now on arxiv! <a href="https://t.co/Vrnpii0X13">https://t.co/Vrnpii0X13</a> Joint work with Jacob Bien (USC) and <a href="https://twitter.com/daniela_witten?ref_src=twsrc%5Etfw">@daniela_witten</a> 1/2 <a href="https://t.co/7jGow89RDu">https://t.co/7jGow89RDu</a></p>&mdash; Lucy L. Gao (@lucylgao) <a href="https://twitter.com/lucylgao/status/1336134312727519232?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. MPG: A Multi-ingredient Pizza Image Generator with Conditional StyleGANs

Fangda Han, Guoyao Hao, Ricardo Guerrero, Vladimir Pavlovic

- retweets: 1983, favorites: 393 (12/09/2020 11:29:21)

- links: [abs](https://arxiv.org/abs/2012.02821) | [pdf](https://arxiv.org/pdf/2012.02821)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Multilabel conditional image generation is a challenging problem in computer vision. In this work we propose Multi-ingredient Pizza Generator (MPG), a conditional Generative Neural Network (GAN) framework for synthesizing multilabel images. We design MPG based on a state-of-the-art GAN structure called StyleGAN2, in which we develop a new conditioning technique by enforcing intermediate feature maps to learn scalewise label information. Because of the complex nature of the multilabel image generation problem, we also regularize synthetic image by predicting the corresponding ingredients as well as encourage the discriminator to distinguish between matched image and mismatched image. To verify the efficacy of MPG, we test it on Pizza10, which is a carefully annotated multi-ingredient pizza image dataset. MPG can successfully generate photo-realist pizza images with desired ingredients. The framework can be easily extend to other multilabel image generation scenarios.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Love Figure 8. of <a href="https://t.co/sJc3fcqHQe">https://t.co/sJc3fcqHQe</a><br><br>“Images generated from different combinations of ingredient list and style noise. Images in the same row are generated with identical style noise.” <a href="https://t.co/Svx46eNmas">https://t.co/Svx46eNmas</a> <a href="https://t.co/AnOjtuD3Sx">pic.twitter.com/AnOjtuD3Sx</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1336219583863091200?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MPG: A Multi-ingredient Pizza Image Generator with Conditional StyleGANs. <a href="https://t.co/KbxsKVDxjJ">https://t.co/KbxsKVDxjJ</a> <a href="https://t.co/ilvONRS3MR">pic.twitter.com/ilvONRS3MR</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1336214080747499522?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MPG: A Multi-ingredient Pizza Image Generator with Conditional StyleGANs<br>pdf: <a href="https://t.co/2Hs0Xth3ug">https://t.co/2Hs0Xth3ug</a><br>abs: <a href="https://t.co/vkuUsASiOx">https://t.co/vkuUsASiOx</a> <a href="https://t.co/wwKtnleHn2">pic.twitter.com/wwKtnleHn2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336138171495706630?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. MFST: A Python OpenFST Wrapper With Support for Custom Semirings and  Jupyter Notebooks

Matthew Francis-Landau

- retweets: 1443, favorites: 331 (12/09/2020 11:29:22)

- links: [abs](https://arxiv.org/abs/2012.03437) | [pdf](https://arxiv.org/pdf/2012.03437)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.MS](https://arxiv.org/list/cs.MS/recent)

This paper introduces mFST, a new Python library for working with Finite-State Machines based on OpenFST. mFST is a thin wrapper for OpenFST and exposes all of OpenFST's methods for manipulating FSTs. Additionally, mFST is the only Python wrapper for OpenFST that exposes OpenFST's ability to define a custom semirings. This makes mFST ideal for developing models that involve learning the weights on a FST or creating neuralized FSTs. mFST has been designed to be easy to get started with and has been previously used in homework assignments for a NLP class as well in projects for integrating FSTs and neural networks. In this paper, we exhibit mFST API and how to use mFST to build a simple neuralized FST with PyTorch.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper on arXiv about mFST, a Python library for working with Finite-State Machines with Custom Semirings<br>Paper: <a href="https://t.co/aBbHZvyx5R">https://t.co/aBbHZvyx5R</a><br>Code: <a href="https://t.co/XBuKV5OBwZ">https://t.co/XBuKV5OBwZ</a><br>In the paper, I demonstate how quickly get started with FSTs and how one could mix PyTorch+FSTs <a href="https://t.co/qIUx7diGlp">pic.twitter.com/qIUx7diGlp</a></p>&mdash; Matthew FL (@matthewfl) <a href="https://twitter.com/matthewfl/status/1336127022154706944?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar  Reconstruction

Guy Gafni, Justus Thies, Michael Zollhöfer, Matthias Nießner

- retweets: 881, favorites: 232 (12/09/2020 11:29:22)

- links: [abs](https://arxiv.org/abs/2012.03065) | [pdf](https://arxiv.org/pdf/2012.03065)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present dynamic neural radiance fields for modeling the appearance and dynamics of a human face. Digitally modeling and reconstructing a talking human is a key building-block for a variety of applications. Especially, for telepresence applications in AR or VR, a faithful reproduction of the appearance including novel viewpoints or head-poses is required. In contrast to state-of-the-art approaches that model the geometry and material properties explicitly, or are purely image-based, we introduce an implicit representation of the head based on scene representation networks. To handle the dynamics of the face, we combine our scene representation network with a low-dimensional morphable model which provides explicit control over pose and expressions. We use volumetric rendering to generate images from this hybrid representation and demonstrate that such a dynamic neural scene representation can be learned from monocular input data only, without the need of a specialized capture setup. In our experiments, we show that this learned volumetric representation allows for photo-realistic image generation that surpasses the quality of state-of-the-art video-based reenactment methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction<br>pdf: <a href="https://t.co/o70GUFyOWf">https://t.co/o70GUFyOWf</a><br>abs: <a href="https://t.co/lMIqim3DNA">https://t.co/lMIqim3DNA</a><br>project page: <a href="https://t.co/SkKhrqWW7E">https://t.co/SkKhrqWW7E</a> <a href="https://t.co/4e97uOr0kj">pic.twitter.com/4e97uOr0kj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336145862393815040?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our Dynamic Neural Radiance Fields approach for 4D Facial Avatars<br><br>Video: <a href="https://t.co/oU7VsBzhSE">https://t.co/oU7VsBzhSE</a><br>Paper: <a href="https://t.co/gVWak521kR">https://t.co/gVWak521kR</a><br>Project Page: <a href="https://t.co/OzP4kOCCR3">https://t.co/OzP4kOCCR3</a><br><br>Kudos to <a href="https://twitter.com/GafniGuy?ref_src=twsrc%5Etfw">@GafniGuy</a> <a href="https://twitter.com/hashtag/NerFACE?src=hash&amp;ref_src=twsrc%5Etfw">#NerFACE</a> <a href="https://t.co/lJ9eh7LQDX">pic.twitter.com/lJ9eh7LQDX</a></p>&mdash; Justus Thies (@JustusThies) <a href="https://twitter.com/JustusThies/status/1336266524680740864?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Machine learning for public policy: Do we need to sacrifice accuracy to  make models fair?

Kit T. Rodolfa, Hemank Lamba, Rayid Ghani

- retweets: 529, favorites: 78 (12/09/2020 11:29:23)

- links: [abs](https://arxiv.org/abs/2012.02972) | [pdf](https://arxiv.org/pdf/2012.02972)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Growing applications of machine learning in policy settings have raised concern for fairness implications, especially for racial minorities, but little work has studied the practical trade-offs between fairness and accuracy in real-world settings. This empirical study fills this gap by investigating the accuracy cost of mitigating disparities across several policy settings, focusing on the common context of using machine learning to inform benefit allocation in resource-constrained programs across education, mental health, criminal justice, and housing safety. In each setting, explicitly focusing on achieving equity and using our proposed post-hoc disparity mitigation methods, fairness was substantially improved without sacrificing accuracy, challenging the commonly held assumption that reducing disparities either requires accepting an appreciable drop in accuracy or the development of novel, complex methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">There is not always a tradeoff between &quot;accuracy&quot; and &quot;fairness&quot; in ML/AI. Our recent empirical work investigating the accuracy cost of mitigating disparities across policy problems - we find that we can achieve fairness without sacrificing accuracy <a href="https://t.co/K739cB094M">https://t.co/K739cB094M</a></p>&mdash; Rayid Ghani (@rayidghani) <a href="https://twitter.com/rayidghani/status/1336349078276464641?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Parallel Training of Deep Networks with Local Updates

Michael Laskin, Luke Metz, Seth Nabarrao, Mark Saroufim, Badreddine Noune, Carlo Luschi, Jascha Sohl-Dickstein, Pieter Abbeel

- retweets: 335, favorites: 111 (12/09/2020 11:29:23)

- links: [abs](https://arxiv.org/abs/2012.03837) | [pdf](https://arxiv.org/pdf/2012.03837)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Deep learning models trained on large data sets have been widely successful in both vision and language domains. As state-of-the-art deep learning architectures have continued to grow in parameter count so have the compute budgets and times required to train them, increasing the need for compute-efficient methods that parallelize training. Two common approaches to parallelize the training of deep networks have been data and model parallelism. While useful, data and model parallelism suffer from diminishing returns in terms of compute efficiency for large batch sizes. In this paper, we investigate how to continue scaling compute efficiently beyond the point of diminishing returns for large batches through local parallelism, a framework which parallelizes training of individual layers in deep networks by replacing global backpropagation with truncated layer-wise backpropagation. Local parallelism enables fully asynchronous layer-wise parallelism with a low memory footprint, and requires little communication overhead compared with model parallelism. We show results in both vision and language domains across a diverse set of architectures, and find that local parallelism is particularly effective in the high-compute regime.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share a paper on local updates as an alternative to global backprop, co-led with <a href="https://twitter.com/Luke_Metz?ref_src=twsrc%5Etfw">@Luke_Metz</a> + <a href="https://twitter.com/graphcoreai?ref_src=twsrc%5Etfw">@graphcoreai</a> <a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> &amp; <a href="https://twitter.com/berkeley_ai?ref_src=twsrc%5Etfw">@berkeley_ai</a>.<br><br>tl;dr - Local updates can improve the efficiency of training deep nets in the high-compute regime. <br>👉  <a href="https://t.co/viEiwBx6Wf">https://t.co/viEiwBx6Wf</a><br><br>1/N</p>&mdash; Michael (Misha) Laskin (@MishaLaskin) <a href="https://twitter.com/MishaLaskin/status/1336401401350598659?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m very excited to finally share our work on Training Deep Networks with Local Updates<br><br>Model Parallelism suffers from high communication costs and poor utilization <br>Data and Pipeline Parallelism introduce a tradeoff between consistency and utilization<a href="https://t.co/WurG9vQD6g">https://t.co/WurG9vQD6g</a> <a href="https://t.co/U5H3Uc7SzD">pic.twitter.com/U5H3Uc7SzD</a></p>&mdash; Mark Saroufim (@marksaroufim) <a href="https://twitter.com/marksaroufim/status/1336359731569389568?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. NeRV: Neural Reflectance and Visibility Fields for Relighting and View  Synthesis

Pratul P. Srinivasan, Boyang Deng, Xiuming Zhang, Matthew Tancik, Ben Mildenhall, Jonathan T. Barron

- retweets: 288, favorites: 117 (12/09/2020 11:29:23)

- links: [abs](https://arxiv.org/abs/2012.03927) | [pdf](https://arxiv.org/pdf/2012.03927)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a method that takes as input a set of images of a scene illuminated by unconstrained known lighting, and produces as output a 3D representation that can be rendered from novel viewpoints under arbitrary lighting conditions. Our method represents the scene as a continuous volumetric function parameterized as MLPs whose inputs are a 3D location and whose outputs are the following scene properties at that input location: volume density, surface normal, material parameters, distance to the first surface intersection in any direction, and visibility of the external environment in any direction. Together, these allow us to render novel views of the object under arbitrary lighting, including indirect illumination effects. The predicted visibility and surface intersection fields are critical to our model's ability to simulate direct and indirect illumination during training, because the brute-force techniques used by prior work are intractable for lighting conditions outside of controlled setups with a single light. Our method outperforms alternative approaches for recovering relightable 3D scene representations, and performs well in complex lighting settings that have posed a significant challenge to prior work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis<br>pdf: <a href="https://t.co/vAIcTSDXvd">https://t.co/vAIcTSDXvd</a><br>abs: <a href="https://t.co/z5xZ0cgZeT">https://t.co/z5xZ0cgZeT</a><br>project page: <a href="https://t.co/SO4W7P01oX">https://t.co/SO4W7P01oX</a> <a href="https://t.co/3v7IHNxXgB">pic.twitter.com/3v7IHNxXgB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336152373186744322?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Deep Learning for Human Mobility: a Survey on Data and Models

Massimiliano Luca, Gianni Barlacchi, Bruno Lepri, Luca Pappalardo

- retweets: 274, favorites: 50 (12/09/2020 11:29:23)

- links: [abs](https://arxiv.org/abs/2012.02825) | [pdf](https://arxiv.org/pdf/2012.02825)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GL](https://arxiv.org/list/cs.GL/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

The study of human mobility is crucial due to its impact on several aspects of our society, such as disease spreading, urban planning, well-being, pollution, and more. The proliferation of digital mobility data, such as phone records, GPS traces, and social media posts, combined with the outstanding predictive power of artificial intelligence, triggered the application of deep learning to human mobility. In particular, the literature is focusing on three tasks: next-location prediction, i.e., predicting an individual's future locations; crowd flow prediction, i.e., forecasting flows on a geographic region; and trajectory generation, i.e., generating realistic individual trajectories. Existing surveys focus on single tasks, data sources, mechanistic or traditional machine learning approaches, while a comprehensive description of deep learning solutions is missing. This survey provides: (i) basic notions on mobility and deep learning; (ii) a review of data sources and public datasets; (iii) a description of deep learning models and (iv) a discussion about relevant open challenges. Our survey is a guide to the leading deep learning solutions to next-location prediction, crowd flow prediction, and trajectory generation. At the same time, it helps deep learning scientists and practitioners understand the fundamental concepts and the open challenges of the study of human mobility.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">📢🔥 The latest survey on <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> approaches to <br>1⃣Next-Location prediction2⃣Crowd-Flow prediction and3⃣Trajectory Generation! (+ list of open datasets)<br><br>Arxiv 👉 <a href="https://t.co/q4wCmjxytJ">https://t.co/q4wCmjxytJ</a><br>GitHub 👉 <a href="https://t.co/NJUf9WuHmT">https://t.co/NJUf9WuHmT</a><br><br>with <a href="https://twitter.com/luca_msl?ref_src=twsrc%5Etfw">@luca_msl</a> <a href="https://twitter.com/GianniBarlacchi?ref_src=twsrc%5Etfw">@GianniBarlacchi</a> <a href="https://twitter.com/brulepri?ref_src=twsrc%5Etfw">@brulepri</a> <a href="https://t.co/Uk31F85d18">pic.twitter.com/Uk31F85d18</a></p>&mdash; Luca Pappalardo (@lucpappalard) <a href="https://twitter.com/lucpappalard/status/1336262154400509953?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Perspectives on Sim2Real Transfer for Robotics: A Summary of the R:SS  2020 Workshop

Sebastian Höfer, Kostas Bekris, Ankur Handa, Juan Camilo Gamboa, Florian Golemo, Melissa Mozifian, Chris Atkeson, Dieter Fox, Ken Goldberg, John Leonard, C. Karen Liu, Jan Peters, Shuran Song, Peter Welinder, Martha White

- retweets: 68, favorites: 102 (12/09/2020 11:29:23)

- links: [abs](https://arxiv.org/abs/2012.03806) | [pdf](https://arxiv.org/pdf/2012.03806)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This report presents the debates, posters, and discussions of the Sim2Real workshop held in conjunction with the 2020 edition of the "Robotics: Science and System" conference. Twelve leaders of the field took competing debate positions on the definition, viability, and importance of transferring skills from simulation to the real world in the context of robotics problems. The debaters also joined a large panel discussion, answering audience questions and outlining the future of Sim2Real in robotics. Furthermore, we invited extended abstracts to this workshop which are summarized in this report. Based on the workshop, this report concludes with directions for practitioners exploiting this technology and for researchers further exploring open problems in this area.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Perspectives on Sim2Real Transfer for Robotics: A Summary of the R:SS 2020 Workshop<a href="https://t.co/myhkkNWLzE">https://t.co/myhkkNWLzE</a> <a href="https://t.co/DvWGvir7c1">pic.twitter.com/DvWGvir7c1</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1336364134510678017?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Perspectives on Sim2Real Transfer for Robotics: A Summary of the R:SS 2020 Workshop <a href="https://t.co/EubRlfLRZ6">https://t.co/EubRlfLRZ6</a> <a href="https://twitter.com/hashtag/robotics?src=hash&amp;ref_src=twsrc%5Etfw">#robotics</a> <a href="https://t.co/EBDIoPWrym">pic.twitter.com/EBDIoPWrym</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1336351160249315330?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Multi-Instrumentalist Net: Unsupervised Generation of Music from Body  Movements

Kun Su, Xiulong Liu, Eli Shlizerman

- retweets: 132, favorites: 28 (12/09/2020 11:29:24)

- links: [abs](https://arxiv.org/abs/2012.03478) | [pdf](https://arxiv.org/pdf/2012.03478)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We propose a novel system that takes as an input body movements of a musician playing a musical instrument and generates music in an unsupervised setting. Learning to generate multi-instrumental music from videos without labeling the instruments is a challenging problem. To achieve the transformation, we built a pipeline named 'Multi-instrumentalistNet' (MI Net). At its base, the pipeline learns a discrete latent representation of various instruments music from log-spectrogram using a Vector Quantized Variational Autoencoder (VQ-VAE) with multi-band residual blocks. The pipeline is then trained along with an autoregressive prior conditioned on the musician's body keypoints movements encoded by a recurrent neural network. Joint training of the prior with the body movements encoder succeeds in the disentanglement of the music into latent features indicating the musical components and the instrumental features. The latent space results in distributions that are clustered into distinct instruments from which new music can be generated. Furthermore, the VQ-VAE architecture supports detailed music generation with additional conditioning. We show that a Midi can further condition the latent space such that the pipeline will generate the exact content of the music being played by the instrument in the video. We evaluate MI Net on two datasets containing videos of 13 instruments and obtain generated music of reasonable audio quality, easily associated with the corresponding instrument, and consistent with the music audio content.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Instrumentalist Net: Unsupervised Generation of Music from Body Movements<br>pdf: <a href="https://t.co/EDKPysqtO0">https://t.co/EDKPysqtO0</a><br>abs: <a href="https://t.co/AoE4p4rG70">https://t.co/AoE4p4rG70</a> <a href="https://t.co/WjvlNcr681">pic.twitter.com/WjvlNcr681</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336156399055855616?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Spatially-Adaptive Pixelwise Networks for Fast Image Translation

Tamar Rott Shaham, Michael Gharbi, Richard Zhang, Eli Shechtman, Tomer Michaeli

- retweets: 90, favorites: 59 (12/09/2020 11:29:24)

- links: [abs](https://arxiv.org/abs/2012.02992) | [pdf](https://arxiv.org/pdf/2012.02992)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce a new generator architecture, aimed at fast and efficient high-resolution image-to-image translation. We design the generator to be an extremely lightweight function of the full-resolution image. In fact, we use pixel-wise networks; that is, each pixel is processed independently of others, through a composition of simple affine transformations and nonlinearities. We take three important steps to equip such a seemingly simple function with adequate expressivity. First, the parameters of the pixel-wise networks are spatially varying so they can represent a broader function class than simple 1x1 convolutions. Second, these parameters are predicted by a fast convolutional network that processes an aggressively low-resolution representation of the input; Third, we augment the input image with a sinusoidal encoding of spatial coordinates, which provides an effective inductive bias for generating realistic novel high-frequency image content. As a result, our model is up to 18x faster than state-of-the-art baselines. We achieve this speedup while generating comparable visual quality across different image resolutions and translation domains.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Spatially-Adaptive Pixelwise Networks for Fast Image Translation<br>pdf: <a href="https://t.co/PSU0oKUIph">https://t.co/PSU0oKUIph</a><br>abs: <a href="https://t.co/oJJizjm26y">https://t.co/oJJizjm26y</a><br>project page: <a href="https://t.co/5AgvTv28n7">https://t.co/5AgvTv28n7</a> <a href="https://t.co/d5Cji0JNxj">pic.twitter.com/d5Cji0JNxj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336193892694241280?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture

Chenfeng Miao, Shuang Liang, Zhencheng Liu, Minchuan Chen, Jun Ma, Shaojun Wang, Jing Xiao

- retweets: 81, favorites: 61 (12/09/2020 11:29:24)

- links: [abs](https://arxiv.org/abs/2012.03500) | [pdf](https://arxiv.org/pdf/2012.03500)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

In this work, we address the Text-to-Speech (TTS) task by proposing a non-autoregressive architecture called EfficientTTS. Unlike the dominant non-autoregressive TTS models, which are trained with the need of external aligners, EfficientTTS optimizes all its parameters with a stable, end-to-end training procedure, while allowing for synthesizing high quality speech in a fast and efficient manner. EfficientTTS is motivated by a new monotonic alignment modeling approach (also introduced in this work), which specifies monotonic constraints to the sequence alignment with almost no increase of computation. By combining EfficientTTS with different feed-forward network structures, we develop a family of TTS models, including both text-to-melspectrogram and text-to-waveform networks. We experimentally show that the proposed models significantly outperform counterpart models such as Tacotron 2 and Glow-TTS in terms of speech quality, training efficiency and synthesis speed, while still producing the speeches of strong robustness and great diversity. In addition, we demonstrate that proposed approach can be easily extended to autoregressive models such as Tacotron 2.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture<br>pdf: <a href="https://t.co/HadDs7Bv8Y">https://t.co/HadDs7Bv8Y</a><br>abs: <a href="https://t.co/dgKUFLzNvX">https://t.co/dgKUFLzNvX</a><br>project page: <a href="https://t.co/59vF5oygIh">https://t.co/59vF5oygIh</a> <a href="https://t.co/vUpG3zDr0K">pic.twitter.com/vUpG3zDr0K</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336158616894771206?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Grammar-Aware Question-Answering on Quantum Computers

Konstantinos Meichanetzidis, Alexis Toumi, Giovanni de Felice, Bob Coecke

- retweets: 58, favorites: 40 (12/09/2020 11:29:24)

- links: [abs](https://arxiv.org/abs/2012.03756) | [pdf](https://arxiv.org/pdf/2012.03756)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Natural language processing (NLP) is at the forefront of great advances in contemporary AI, and it is arguably one of the most challenging areas of the field. At the same time, with the steady growth of quantum hardware and notable improvements towards implementations of quantum algorithms, we are approaching an era when quantum computers perform tasks that cannot be done on classical computers with a reasonable amount of resources. This provides a new range of opportunities for AI, and for NLP specifically. Earlier work has already demonstrated a potential quantum advantage for NLP in a number of manners: (i) algorithmic speedups for search-related or classification tasks, which are the most dominant tasks within NLP, (ii) exponentially large quantum state spaces allow for accommodating complex linguistic structures, (iii) novel models of meaning employing density matrices naturally model linguistic phenomena such as hyponymy and linguistic ambiguity, among others. In this work, we perform the first implementation of an NLP task on noisy intermediate-scale quantum (NISQ) hardware. Sentences are instantiated as parameterised quantum circuits. We encode word-meanings in quantum states and we explicitly account for grammatical structure, which even in mainstream NLP is not commonplace, by faithfully hard-wiring it as entangling operations. This makes our approach to quantum natural language processing (QNLP) particularly NISQ-friendly. Our novel QNLP model shows concrete promise for scalability as the quality of the quantum hardware improves in the near future.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Yay, our latest paper on QNLP experiments is out on the arXiv! We tried to explain Shakespeare&#39;s Romeo and Juliet to a small and noisy quantum computer, and it did get some of it: &quot;Romeo who loves Juliet dies&quot; <a href="https://twitter.com/coecke?ref_src=twsrc%5Etfw">@coecke</a> <a href="https://twitter.com/konstantinosmei?ref_src=twsrc%5Etfw">@konstantinosmei</a> <a href="https://twitter.com/gio_defel?ref_src=twsrc%5Etfw">@gio_defel</a> <a href="https://t.co/sjW0RH2sQ1">https://t.co/sjW0RH2sQ1</a> <a href="https://t.co/rjfingbnvh">pic.twitter.com/rjfingbnvh</a></p>&mdash; alexis.toumi (@AlexisToumi) <a href="https://twitter.com/AlexisToumi/status/1336227603091480578?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Sample-efficient proper PAC learning with approximate differential  privacy

Badih Ghazi, Noah Golowich, Ravi Kumar, Pasin Manurangsi

- retweets: 20, favorites: 51 (12/09/2020 11:29:24)

- links: [abs](https://arxiv.org/abs/2012.03893) | [pdf](https://arxiv.org/pdf/2012.03893)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent)

In this paper we prove that the sample complexity of properly learning a class of Littlestone dimension $d$ with approximate differential privacy is $\tilde O(d^6)$, ignoring privacy and accuracy parameters. This result answers a question of Bun et al. (FOCS 2020) by improving upon their upper bound of $2^{O(d)}$ on the sample complexity. Prior to our work, finiteness of the sample complexity for privately learning a class of finite Littlestone dimension was only known for improper private learners, and the fact that our learner is proper answers another question of Bun et al., which was also asked by Bousquet et al. (NeurIPS 2020). Using machinery developed by Bousquet et al., we then show that the sample complexity of sanitizing a binary hypothesis class is at most polynomial in its Littlestone dimension and dual Littlestone dimension. This implies that a class is sanitizable if and only if it has finite Littlestone dimension. An important ingredient of our proofs is a new property of binary hypothesis classes that we call irreducibility, which may be of independent interest.




# 16. iGibson, a Simulation Environment for Interactive Tasks in Large  Realistic Scenes

Bokui Shen, Fei Xia, Chengshu Li, Roberto Martín-Martín, Linxi Fan, Guanzhi Wang, Shyamal Buch, Claudia D'Arpino, Sanjana Srivastava, Lyne P. Tchapmi, Micael E. Tchapmi, Kent Vainio, Li Fei-Fei, Silvio Savarese

- retweets: 18, favorites: 46 (12/09/2020 11:29:24)

- links: [abs](https://arxiv.org/abs/2012.02924) | [pdf](https://arxiv.org/pdf/2012.02924)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We present iGibson, a novel simulation environment to develop robotic solutions for interactive tasks in large-scale realistic scenes. Our environment contains fifteen fully interactive home-sized scenes populated with rigid and articulated objects. The scenes are replicas of 3D scanned real-world homes, aligning the distribution of objects and layout to that of the real world. iGibson integrates several key features to facilitate the study of interactive tasks: i) generation of high-quality visual virtual sensor signals (RGB, depth, segmentation, LiDAR, flow, among others), ii) domain randomization to change the materials of the objects (both visual texture and dynamics) and/or their shapes, iii) integrated sampling-based motion planners to generate collision-free trajectories for robot bases and arms, and iv) intuitive human-iGibson interface that enables efficient collection of human demonstrations. Through experiments, we show that the full interactivity of the scenes enables agents to learn useful visual representations that accelerate the training of downstream manipulation tasks. We also show that iGibson features enable the generalization of navigation agents, and that the human-iGibson interface and integrated motion planners facilitate efficient imitation learning of simple human demonstrated behaviors. iGibson is open-sourced with comprehensive examples and documentation. For more information, visit our project website: http://svl.stanford.edu/igibson/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">iGibson, a Simulation Environment for Interactive Tasks in Large RealisticScenes<br>pdf: <a href="https://t.co/y2dcMTkRym">https://t.co/y2dcMTkRym</a><br>abs: <a href="https://t.co/zHkZDkB2qt">https://t.co/zHkZDkB2qt</a><br>project page: <a href="https://t.co/lizGdNZyFp">https://t.co/lizGdNZyFp</a> <a href="https://t.co/o6UiE2HSDy">pic.twitter.com/o6UiE2HSDy</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336179646015164417?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. MemPool: A Shared-L1 Memory Many-Core Cluster with a Low-Latency  Interconnect

Matheus Cavalcante, Samuel Riedel, Antonio Pullini, Luca Benini

- retweets: 36, favorites: 23 (12/09/2020 11:29:24)

- links: [abs](https://arxiv.org/abs/2012.02973) | [pdf](https://arxiv.org/pdf/2012.02973)
- [cs.AR](https://arxiv.org/list/cs.AR/recent)

A key challenge in scaling shared-L1 multi-core clusters towards many-core (more than 16 cores) configurations is to ensure low-latency and efficient access to the L1 memory. In this work we demonstrate that it is possible to scale up the shared-L1 architecture: We present MemPool, a 32 bit many-core system with 256 fast RV32IMA "Snitch" cores featuring application-tunable execution units, running at 700 MHz in typical conditions (TT/0.80 V/25{\deg}C). MemPool is easy to program, with all the cores sharing a global view of a large L1 scratchpad memory pool, accessible within at most 5 cycles. In MemPool's physical-aware design, we emphasized the exploration, design, and optimization of the low-latency processor-to-L1-memory interconnect. We compare three candidate topologies, analyzing them in terms of latency, throughput, and back-end feasibility. The chosen topology keeps the average latency at fewer than 6 cycles, even for a heavy injected load of 0.33 request/core/cycle. We also propose a lightweight addressing scheme that maps each core private data to a memory bank accessible within one cycle, which leads to performance gains of up to 20% in real-world signal processing benchmarks. The addressing scheme is also highly efficient in terms of energy consumption since requests to local banks consume only half of the energy required to access remote banks. Our design achieves competitive performance with respect to an ideal, non-implementable full-crossbar baseline.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our latest large-scale <a href="https://twitter.com/pulp_platform?ref_src=twsrc%5Etfw">@pulp_platform</a> embodiment is out (to appear at DATE21). Mempool is a giant cluster with 256 snitch cores clocked at 700MHz and with max zero-load latency to L1 memory of 5cycles. Quite a lot of new stuff <a href="https://t.co/03H8sg45af">https://t.co/03H8sg45af</a></p>&mdash; Luca Benini (@LucaBeniniZhFe) <a href="https://twitter.com/LucaBeniniZhFe/status/1336221763429142528?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Data Boost: Text Data Augmentation Through Reinforcement Learning Guided  Conditional Generation

Ruibo Liu, Guangxuan Xu, Chenyan Jia, Weicheng Ma, Lili Wang, Soroush Vosoughi

- retweets: 34, favorites: 24 (12/09/2020 11:29:25)

- links: [abs](https://arxiv.org/abs/2012.02952) | [pdf](https://arxiv.org/pdf/2012.02952)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Data augmentation is proven to be effective in many NLU tasks, especially for those suffering from data scarcity. In this paper, we present a powerful and easy to deploy text augmentation framework, Data Boost, which augments data through reinforcement learning guided conditional generation. We evaluate Data Boost on three diverse text classification tasks under five different classifier architectures. The result shows that Data Boost can boost the performance of classifiers especially in low-resource data scenarios. For instance, Data Boost improves F1 for the three tasks by 8.7% on average when given only 10% of the whole data for training. We also compare Data Boost with six prior text augmentation methods. Through human evaluations (N=178), we confirm that Data Boost augmentation has comparable quality as the original data with respect to readability and class consistency.




# 19. Rethinking FUN: Frequency-Domain Utilization Networks

Kfir Goldberg, Stav Shapiro, Elad Richardson, Shai Avidan

- retweets: 16, favorites: 42 (12/09/2020 11:29:25)

- links: [abs](https://arxiv.org/abs/2012.03357) | [pdf](https://arxiv.org/pdf/2012.03357)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The search for efficient neural network architectures has gained much focus in recent years, where modern architectures focus not only on accuracy but also on inference time and model size. Here, we present FUN, a family of novel Frequency-domain Utilization Networks. These networks utilize the inherent efficiency of the frequency-domain by working directly in that domain, represented with the Discrete Cosine Transform. Using modern techniques and building blocks such as compound-scaling and inverted-residual layers we generate a set of such networks allowing one to balance between size, latency and accuracy while outperforming competing RGB-based models. Extensive evaluations verifies that our networks present strong alternatives to previous approaches. Moreover, we show that working in frequency domain allows for dynamic compression of the input at inference time without any explicit change to the architecture.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking FUN: Frequency-Domain Utilization Networks<br>pdf: <a href="https://t.co/ultpzPoXrn">https://t.co/ultpzPoXrn</a><br>abs: <a href="https://t.co/8YxaTFXshQ">https://t.co/8YxaTFXshQ</a><br>github: <a href="https://t.co/Mca7SeZbyL">https://t.co/Mca7SeZbyL</a> <a href="https://t.co/SVffoh8kiI">pic.twitter.com/SVffoh8kiI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1336202553768468482?ref_src=twsrc%5Etfw">December 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



