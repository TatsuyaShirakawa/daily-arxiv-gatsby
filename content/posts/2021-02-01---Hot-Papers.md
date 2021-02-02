---
title: Hot Papers 2021-02-01
date: 2021-02-02T10:20:28.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-01"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-01"
socialImage: "/media/flying-marine.jpg"

---

# 1. Efficient-CapsNet: Capsule Network with Self-Attention Routing

Vittorio Mazzia, Francesco Salvetti, Marcello Chiaberge

- retweets: 1894, favorites: 213 (02/02/2021 10:20:28)

- links: [abs](https://arxiv.org/abs/2101.12491) | [pdf](https://arxiv.org/pdf/2101.12491)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Deep convolutional neural networks, assisted by architectural design strategies, make extensive use of data augmentation techniques and layers with a high number of feature maps to embed object transformations. That is highly inefficient and for large datasets implies a massive redundancy of features detectors. Even though capsules networks are still in their infancy, they constitute a promising solution to extend current convolutional networks and endow artificial visual perception with a process to encode more efficiently all feature affine transformations. Indeed, a properly working capsule network should theoretically achieve higher results with a considerably lower number of parameters count due to intrinsic capability to generalize to novel viewpoints. Nevertheless, little attention has been given to this relevant aspect. In this paper, we investigate the efficiency of capsule networks and, pushing their capacity to the limits with an extreme architecture with barely 160K parameters, we prove that the proposed architecture is still able to achieve state-of-the-art results on three different datasets with only 2% of the original CapsNet parameters. Moreover, we replace dynamic routing with a novel non-iterative, highly parallelizable routing algorithm that can easily cope with a reduced number of capsules. Extensive experimentation with other capsule implementations has proved the effectiveness of our methodology and the capability of capsule networks to efficiently embed visual representations more prone to generalization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient-CapsNet: Capsule Network with Self-Attention Routing<br>pdf: <a href="https://t.co/dWLFcRppDo">https://t.co/dWLFcRppDo</a><br>abs: <a href="https://t.co/Ww5IKYVpiJ">https://t.co/Ww5IKYVpiJ</a> <a href="https://t.co/4cDxKVcOlM">pic.twitter.com/4cDxKVcOlM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356069183528849411?ref_src=twsrc%5Etfw">February 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose  Estimation

Angtian Wang, Adam Kortylewski, Alan Yuille

- retweets: 274, favorites: 109 (02/02/2021 10:20:28)

- links: [abs](https://arxiv.org/abs/2101.12378) | [pdf](https://arxiv.org/pdf/2101.12378)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

3D pose estimation is a challenging but important task in computer vision. In this work, we show that standard deep learning approaches to 3D pose estimation are not robust when objects are partially occluded or viewed from a previously unseen pose. Inspired by the robustness of generative vision models to partial occlusion, we propose to integrate deep neural networks with 3D generative representations of objects into a unified neural architecture that we term NeMo. In particular, NeMo learns a generative model of neural feature activations at each vertex on a dense 3D mesh. Using differentiable rendering we estimate the 3D object pose by minimizing the reconstruction error between NeMo and the feature representation of the target image. To avoid local optima in the reconstruction loss, we train the feature extractor to maximize the distance between the individual feature representations on the mesh using contrastive learning. Our extensive experiments on PASCAL3D+, occluded-PASCAL3D+ and ObjectNet3D show that NeMo is much more robust to partial occlusion and unseen pose compared to standard deep networks, while retaining competitive performance on regular data. Interestingly, our experiments also show that NeMo performs reasonably well even when the mesh representation only crudely approximates the true object geometry with a cuboid, hence revealing that the detailed 3D geometry is not needed for accurate 3D pose estimation. The code is publicly available at https://github.com/Angtian/NeMo.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation<br>pdf: <a href="https://t.co/1hrrqjxmFd">https://t.co/1hrrqjxmFd</a><br>abs: <a href="https://t.co/YnblkEY0Pw">https://t.co/YnblkEY0Pw</a><br>github: <a href="https://t.co/s5qalReocz">https://t.co/s5qalReocz</a> <a href="https://t.co/WYThH6GNF6">pic.twitter.com/WYThH6GNF6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356075772772016128?ref_src=twsrc%5Etfw">February 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Does injecting linguistic structure into language models lead to better  alignment with brain recordings?

Mostafa Abdou, Ana Valeria Gonzalez, Mariya Toneva, Daniel Hershcovich, Anders Søgaard

- retweets: 46, favorites: 60 (02/02/2021 10:20:29)

- links: [abs](https://arxiv.org/abs/2101.12608) | [pdf](https://arxiv.org/pdf/2101.12608)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Neuroscientists evaluate deep neural networks for natural language processing as possible candidate models for how language is processed in the brain. These models are often trained without explicit linguistic supervision, but have been shown to learn some linguistic structure in the absence of such supervision (Manning et al., 2020), potentially questioning the relevance of symbolic linguistic theories in modeling such cognitive processes (Warstadt and Bowman, 2020). We evaluate across two fMRI datasets whether language models align better with brain recordings, if their attention is biased by annotations from syntactic or semantic formalisms. Using structure from dependency or minimal recursion semantic annotations, we find alignments improve significantly for one of the datasets. For another dataset, we see more mixed results. We present an extensive analysis of these results. Our proposed approach enables the evaluation of more targeted hypotheses about the composition of meaning in the brain, expanding the range of possible scientific inferences a neuroscientist could make, and opens up new opportunities for cross-pollination between computational neuroscience and linguistics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/4 New preprint! <a href="https://t.co/6lHYf4mwut">https://t.co/6lHYf4mwut</a><br>Does injecting structural bias into LMs lead to better alignment with brain recordings? Yes, it does!<br><br>work done in collaboration with Mostafa Abdou, <a href="https://twitter.com/mtoneva1?ref_src=twsrc%5Etfw">@mtoneva1</a> , <a href="https://twitter.com/daniel_hers?ref_src=twsrc%5Etfw">@daniel_hers</a> and Anders Søgaard<br><br>More details in thread ! <a href="https://t.co/5LUXHTnkaA">pic.twitter.com/5LUXHTnkaA</a></p>&mdash; Ana Valeria Gonzalez (@AnaValeriaGlez) <a href="https://twitter.com/AnaValeriaGlez/status/1356161111951872003?ref_src=twsrc%5Etfw">February 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Applying Bayesian Analysis Guidelines to Empirical Software Engineering  Data: The Case of Programming Languages and Code Quality

Carlo A. Furia, Richard Torkar, Robert Feldt

- retweets: 56, favorites: 36 (02/02/2021 10:20:29)

- links: [abs](https://arxiv.org/abs/2101.12591) | [pdf](https://arxiv.org/pdf/2101.12591)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

Statistical analysis is the tool of choice to turn data into information, and then information into empirical knowledge. To be valid, the process that goes from data to knowledge should be supported by detailed, rigorous guidelines, which help ferret out issues with the data or model, and lead to qualified results that strike a reasonable balance between generality and practical relevance. Such guidelines are being developed by statisticians to support the latest techniques for Bayesian data analysis. In this article, we frame these guidelines in a way that is apt to empirical research in software engineering.   To demonstrate the guidelines in practice, we apply them to reanalyze a GitHub dataset about code quality in different programming languages. The dataset's original analysis (Ray et al., 2014) and a critical reanalysis (Berger at al., 2019) have attracted considerable attention -- in no small part because they target a topic (the impact of different programming languages) on which strong opinions abound. The goals of our reanalysis are largely orthogonal to this previous work, as we are concerned with demonstrating, on data in an interesting domain, how to build a principled Bayesian data analysis and to showcase some of its benefits. In the process, we will also shed light on some critical aspects of the analyzed data and of the relationship between programming languages and code quality.   The high-level conclusions of our exercise will be that Bayesian statistical techniques can be applied to analyze software engineering data in a way that is principled, flexible, and leads to convincing results that inform the state of the art while highlighting the boundaries of its validity. The guidelines can support building solid statistical analyses and connecting their results, and hence help buttress continued progress in empirical software engineering research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">3rd paper of our BaySE (Bayesian Analysis in SE ;) ) &quot;trifecta&quot; now on arxiv, this one focused on concrete *guidelines* and how to *apply it*:<a href="https://t.co/1E42OYhJWZ">https://t.co/1E42OYhJWZ</a><br><br>with <a href="https://twitter.com/bugcounting?ref_src=twsrc%5Etfw">@bugcounting</a> and <a href="https://twitter.com/rtorkar?ref_src=twsrc%5Etfw">@rtorkar</a> , as usual :)</p>&mdash; Robert Feldt (@drfeldt) <a href="https://twitter.com/drfeldt/status/1356167484097835008?ref_src=twsrc%5Etfw">February 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Position, Padding and Predictions: A Deeper Look at Position Information  in CNNs

Md Amirul Islam, Matthew Kowal, Sen Jia, Konstantinos G. Derpanis, Neil D. B. Bruce

- retweets: 18, favorites: 48 (02/02/2021 10:20:29)

- links: [abs](https://arxiv.org/abs/2101.12322) | [pdf](https://arxiv.org/pdf/2101.12322)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In contrast to fully connected networks, Convolutional Neural Networks (CNNs) achieve efficiency by learning weights associated with local filters with a finite spatial extent. An implication of this is that a filter may know what it is looking at, but not where it is positioned in the image. In this paper, we first test this hypothesis and reveal that a surprising degree of absolute position information is encoded in commonly used CNNs. We show that zero padding drives CNNs to encode position information in their internal representations, while a lack of padding precludes position encoding. This gives rise to deeper questions about the role of position information in CNNs: (i) What boundary heuristics enable optimal position encoding for downstream tasks?; (ii) Does position encoding affect the learning of semantic representations?; (iii) Does position encoding always improve performance? To provide answers, we perform the largest case study to date on the role that padding and border heuristics play in CNNs. We design novel tasks which allow us to quantify boundary effects as a function of the distance to the border. Numerous semantic objectives reveal the effect of the border on semantic representations. Finally, we demonstrate the implications of these findings on multiple real-world tasks to show that position information can both help or hurt performance.




# 6. A Survey of Complex-Valued Neural Networks

Joshua Bassey, Lijun Qian, Xianfang Li

- retweets: 32, favorites: 25 (02/02/2021 10:20:29)

- links: [abs](https://arxiv.org/abs/2101.12249) | [pdf](https://arxiv.org/pdf/2101.12249)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Artificial neural networks (ANNs) based machine learning models and especially deep learning models have been widely applied in computer vision, signal processing, wireless communications, and many other domains, where complex numbers occur either naturally or by design. However, most of the current implementations of ANNs and machine learning frameworks are using real numbers rather than complex numbers. There are growing interests in building ANNs using complex numbers, and exploring the potential advantages of the so-called complex-valued neural networks (CVNNs) over their real-valued counterparts. In this paper, we discuss the recent development of CVNNs by performing a survey of the works on CVNNs in the literature. Specifically, a detailed review of various CVNNs in terms of activation function, learning and optimization, input and output representations, and their applications in tasks such as signal processing and computer vision are provided, followed by a discussion on some pertinent challenges and future research directions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Survey of Complex-Valued Neural Networks. <a href="https://t.co/xAfkJHhzK3">https://t.co/xAfkJHhzK3</a> <a href="https://t.co/clQXOvg5GI">pic.twitter.com/clQXOvg5GI</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1356099621299265537?ref_src=twsrc%5Etfw">February 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



