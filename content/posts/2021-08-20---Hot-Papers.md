---
title: Hot Papers 2021-08-20
date: 2021-08-21T09:06:00.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-20"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-20"
socialImage: "/media/flying-marine.jpg"

---

# 1. Do Vision Transformers See Like Convolutional Neural Networks?

Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, Alexey Dosovitskiy

- retweets: 9128, favorites: 5 (08/21/2021 09:06:00)

- links: [abs](https://arxiv.org/abs/2108.08810) | [pdf](https://arxiv.org/pdf/2108.08810)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Convolutional neural networks (CNNs) have so far been the de-facto model for visual data. Recent work has shown that (Vision) Transformer models (ViT) can achieve comparable or even superior performance on image classification tasks. This raises a central question: how are Vision Transformers solving these tasks? Are they acting like convolutional networks, or learning entirely different visual representations? Analyzing the internal representation structure of ViTs and CNNs on image classification benchmarks, we find striking differences between the two architectures, such as ViT having more uniform representations across all layers. We explore how these differences arise, finding crucial roles played by self-attention, which enables early aggregation of global information, and ViT residual connections, which strongly propagate features from lower to higher layers. We study the ramifications for spatial localization, demonstrating ViTs successfully preserve input spatial information, with noticeable effects from different classification methods. Finally, we study the effect of (pretraining) dataset scale on intermediate features and transfer learning, and conclude with a discussion on connections to new architectures such as the MLP-Mixer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do Vision Transformers See Like Convolutional Neural Networks?<br><br>New paper <a href="https://t.co/mxLCIRBRLy">https://t.co/mxLCIRBRLy</a><br><br>The successes of Transformers in computer vision prompts a fundamental question: how are they solving these tasks? Do Transformers act like CNNs, or learn very different features? <a href="https://t.co/3gJSZ3rArt">pic.twitter.com/3gJSZ3rArt</a></p>&mdash; Maithra Raghu (@maithra_raghu) <a href="https://twitter.com/maithra_raghu/status/1428740724074291208?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Image2Lego: Customized LEGO Set Generation from Images

Kyle Lennon, Katharina Fransen, Alexander O'Brien, Yumeng Cao, Matthew Beveridge, Yamin Arefeen, Nikhil Singh, Iddo Drori

- retweets: 1558, favorites: 163 (08/21/2021 09:06:00)

- links: [abs](https://arxiv.org/abs/2108.08477) | [pdf](https://arxiv.org/pdf/2108.08477)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Although LEGO sets have entertained generations of children and adults, the challenge of designing customized builds matching the complexity of real-world or imagined scenes remains too great for the average enthusiast. In order to make this feat possible, we implement a system that generates a LEGO brick model from 2D images. We design a novel solution to this problem that uses an octree-structured autoencoder trained on 3D voxelized models to obtain a feasible latent representation for model reconstruction, and a separate network trained to predict this latent representation from 2D images. LEGO models are obtained by algorithmic conversion of the 3D voxelized model to bricks. We demonstrate first-of-its-kind conversion of photographs to 3D LEGO models. An octree architecture enables the flexibility to produce multiple resolutions to best fit a user's creative vision or design needs. In order to demonstrate the broad applicability of our system, we generate step-by-step building instructions and animations for LEGO models of objects and human faces. Finally, we test these automatically generated LEGO sets by constructing physical builds using real LEGO bricks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Image2Lego: Customized LEGO® Set Generation from Images <br>pdf: <a href="https://t.co/yHBU4o5qSt">https://t.co/yHBU4o5qSt</a><br>abs: <a href="https://t.co/3pBPqI1dFz">https://t.co/3pBPqI1dFz</a><br>project page: <a href="https://t.co/96SPLGgO06">https://t.co/96SPLGgO06</a><br><br>a pipeline for producing 3D LEGO® models from 2D images <a href="https://t.co/EUs4AQiFnM">pic.twitter.com/EUs4AQiFnM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428556160790147078?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. ImageBART: Bidirectional Context with Multinomial Diffusion for  Autoregressive Image Synthesis

Patrick Esser, Robin Rombach, Andreas Blattmann, Björn Ommer

- retweets: 272, favorites: 81 (08/21/2021 09:06:00)

- links: [abs](https://arxiv.org/abs/2108.08827) | [pdf](https://arxiv.org/pdf/2108.08827)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Autoregressive models and their sequential factorization of the data likelihood have recently demonstrated great potential for image representation and synthesis. Nevertheless, they incorporate image context in a linear 1D order by attending only to previously synthesized image patches above or to the left. Not only is this unidirectional, sequential bias of attention unnatural for images as it disregards large parts of a scene until synthesis is almost complete. It also processes the entire image on a single scale, thus ignoring more global contextual information up to the gist of the entire scene. As a remedy we incorporate a coarse-to-fine hierarchy of context by combining the autoregressive formulation with a multinomial diffusion process: Whereas a multistage diffusion process successively removes information to coarsen an image, we train a (short) Markov chain to invert this process. In each stage, the resulting autoregressive ImageBART model progressively incorporates context from previous stages in a coarse-to-fine manner. Experiments show greatly improved image modification capabilities over autoregressive models while also providing high-fidelity image generation, both of which are enabled through efficient training in a compressed latent space. Specifically, our approach can take unrestricted, user-provided masks into account to perform local image editing. Thus, in contrast to pure autoregressive models, it can solve free-form image inpainting and, in the case of conditional models, local, text-guided image modification without requiring mask-specific training.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ImageBART: Bidirectional Context with Multinomial<br>Diffusion for Autoregressive Image Synthesis<br>abs: <a href="https://t.co/dTknlBbANY">https://t.co/dTknlBbANY</a><br><br>a hierarchical approach to introduce bidirectional context into autoregressive transformer models for high-fidelity controllable image synthesis <a href="https://t.co/sjxiCFTd6G">pic.twitter.com/sjxiCFTd6G</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428517648237801480?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Neural-GIF: Neural Generalized Implicit Functions for Animating People  in Clothing

Garvita Tiwari, Nikolaos Sarafianos, Tony Tung, Gerard Pons-Moll1

- retweets: 169, favorites: 69 (08/21/2021 09:06:01)

- links: [abs](https://arxiv.org/abs/2108.08807) | [pdf](https://arxiv.org/pdf/2108.08807)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present Neural Generalized Implicit Functions(Neural-GIF), to animate people in clothing as a function of the body pose. Given a sequence of scans of a subject in various poses, we learn to animate the character for new poses. Existing methods have relied on template-based representations of the human body (or clothing). However such models usually have fixed and limited resolutions, require difficult data pre-processing steps and cannot be used with complex clothing. We draw inspiration from template-based methods, which factorize motion into articulation and non-rigid deformation, but generalize this concept for implicit shape learning to obtain a more flexible model. We learn to map every point in the space to a canonical space, where a learned deformation field is applied to model non-rigid effects, before evaluating the signed distance field. Our formulation allows the learning of complex and non-rigid deformations of clothing and soft tissue, without computing a template registration as it is common with current approaches. Neural-GIF can be trained on raw 3D scans and reconstructs detailed complex surface geometry and deformations. Moreover, the model can generalize to new poses. We evaluate our method on a variety of characters from different public datasets in diverse clothing styles and show significant improvements over baseline methods, quantitatively and qualitatively. We also extend our model to multiple shape setting. To stimulate further research, we will make the model, code and data publicly available at: https://virtualhumans.mpi-inf.mpg.de/neuralgif/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural-GIF: Neural Generalized Implicit Functions<br>for Animating People in Clothing<br>pdf: <a href="https://t.co/JEnRjpK03Z">https://t.co/JEnRjpK03Z</a><br>abs: <a href="https://t.co/Ykt8cfJyBr">https://t.co/Ykt8cfJyBr</a><br><br>model to learn articulation and pose-dependent deformation for humans in complex clothing using an implicit 3D surface representation <a href="https://t.co/YDTkvxdSBK">pic.twitter.com/YDTkvxdSBK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428518956525051907?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Successive cohorts of Twitter users show increasing activity and  shrinking content horizons

Frederik Wolf, Philipp Lorenz-Spreen, Sune Lehmann

- retweets: 166, favorites: 50 (08/21/2021 09:06:01)

- links: [abs](https://arxiv.org/abs/2108.08641) | [pdf](https://arxiv.org/pdf/2108.08641)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

The global public sphere has changed dramatically over the past decades: a significant part of public discourse now takes place on algorithmically driven platforms owned by a handful of private companies. Despite its growing importance, there is scant large-scale academic research on the long-term evolution of user behaviour on these platforms, because the data are often proprietary to the platforms. Here, we evaluate the individual behaviour of 600,000 Twitter users between 2012 and 2019 and find empirical evidence for an acceleration of the way Twitter is used on an individual level. This manifests itself in the fact that cohorts of Twitter users behave differently depending on when they joined the platform. Behaviour within a cohort is relatively consistent over time and characterised by strong internal interactions, but over time behaviour from cohort to cohort shifts towards increased activity. Specifically, we measure this in terms of more tweets per user over time, denser interactions with others via retweets, and shorter content horizons, expressed as an individual's decaying autocorrelation of topics over time. Our observations are explained by a growing proportion of active users who not only tweet more actively but also elicit more retweets. These behaviours suggest a collective contribution to an increased flow of information through each cohort's news feed -- an increase that potentially depletes available collective attention over time. Our findings complement recent, empirical work on social acceleration, which has been largely agnostic about individual user activity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Successive cohorts of Twitter users show increasing activity and shrinking content horizons&quot;<br><br>Looking forward to reading this preprint from <a href="https://twitter.com/suneman?ref_src=twsrc%5Etfw">@suneman</a> &amp; team. <br><br>Findings look consistent w/accelerated engagement and broad language churn we&#39;ve observed.<a href="https://t.co/e7mtgLGvA7">https://t.co/e7mtgLGvA7</a> <a href="https://t.co/hggvsj1NBV">pic.twitter.com/hggvsj1NBV</a></p>&mdash; Chris Danforth (@ChrisDanforth) <a href="https://twitter.com/ChrisDanforth/status/1428690024455938052?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Gravity-Aware Monocular 3D Human-Object Reconstruction

Rishabh Dabral, Soshi Shimada, Arjun Jain, Christian Theobalt, Vladislav Golyanik

- retweets: 110, favorites: 31 (08/21/2021 09:06:01)

- links: [abs](https://arxiv.org/abs/2108.08844) | [pdf](https://arxiv.org/pdf/2108.08844)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper proposes GraviCap, i.e., a new approach for joint markerless 3D human motion capture and object trajectory estimation from monocular RGB videos. We focus on scenes with objects partially observed during a free flight. In contrast to existing monocular methods, we can recover scale, object trajectories as well as human bone lengths in meters and the ground plane's orientation, thanks to the awareness of the gravity constraining object motions. Our objective function is parametrised by the object's initial velocity and position, gravity direction and focal length, and jointly optimised for one or several free flight episodes. The proposed human-object interaction constraints ensure geometric consistency of the 3D reconstructions and improved physical plausibility of human poses compared to the unconstrained case. We evaluate GraviCap on a new dataset with ground-truth annotations for persons and different objects undergoing free flights. In the experiments, our approach achieves state-of-the-art accuracy in 3D human motion capture on various metrics. We urge the reader to watch our supplementary video. Both the source code and the dataset are released; see http://4dqv.mpi-inf.mpg.de/GraviCap/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Gravity-Aware Monocular 3D Human-Object Reconstruction<br>pdf: <a href="https://t.co/e4w4RrbbZw">https://t.co/e4w4RrbbZw</a><br>abs: <a href="https://t.co/yUyEdvK3wP">https://t.co/yUyEdvK3wP</a><br>project page: <a href="https://t.co/ubIrqJQAxv">https://t.co/ubIrqJQAxv</a> <a href="https://t.co/7oeirKsTTm">pic.twitter.com/7oeirKsTTm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428528411333603331?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Learning to Match Features with Seeded Graph Matching Network

Hongkai Chen, Zixin Luo, Jiahui Zhang, Lei Zhou, Xuyang Bai, Zeyu Hu, Chiew-Lan Tai, Long Quan

- retweets: 49, favorites: 22 (08/21/2021 09:06:01)

- links: [abs](https://arxiv.org/abs/2108.08771) | [pdf](https://arxiv.org/pdf/2108.08771)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Matching local features across images is a fundamental problem in computer vision. Targeting towards high accuracy and efficiency, we propose Seeded Graph Matching Network, a graph neural network with sparse structure to reduce redundant connectivity and learn compact representation. The network consists of 1) Seeding Module, which initializes the matching by generating a small set of reliable matches as seeds. 2) Seeded Graph Neural Network, which utilizes seed matches to pass messages within/across images and predicts assignment costs. Three novel operations are proposed as basic elements for message passing: 1) Attentional Pooling, which aggregates keypoint features within the image to seed matches. 2) Seed Filtering, which enhances seed features and exchanges messages across images. 3) Attentional Unpooling, which propagates seed features back to original keypoints. Experiments show that our method reduces computational and memory complexity significantly compared with typical attention-based networks while competitive or higher performance is achieved.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning to Match Features with Seeded Graph Matching Network<br>Hongkai Chen, Zixin Luo, Jiahui Zhang, Lei Zhou, Xuyang Bai, Zeyu Hu, Chiew-Lan Tai, Long Quan<br><br>tl;dr:  More weight on high-confidence matches, but fails to reach original SuperGlue performance.<a href="https://t.co/KQ8Uuk9mz8">https://t.co/KQ8Uuk9mz8</a> <a href="https://t.co/RRSwalpdY0">pic.twitter.com/RRSwalpdY0</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1428711795464753152?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers

Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, Jie Zhou

- retweets: 42, favorites: 26 (08/21/2021 09:06:01)

- links: [abs](https://arxiv.org/abs/2108.08839) | [pdf](https://arxiv.org/pdf/2108.08839)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Point clouds captured in real-world applications are often incomplete due to the limited sensor resolution, single viewpoint, and occlusion. Therefore, recovering the complete point clouds from partial ones becomes an indispensable task in many practical applications. In this paper, we present a new method that reformulates point cloud completion as a set-to-set translation problem and design a new model, called PoinTr that adopts a transformer encoder-decoder architecture for point cloud completion. By representing the point cloud as a set of unordered groups of points with position embeddings, we convert the point cloud to a sequence of point proxies and employ the transformers for point cloud generation. To facilitate transformers to better leverage the inductive bias about 3D geometric structures of point clouds, we further devise a geometry-aware block that models the local geometric relationships explicitly. The migration of transformers enables our model to better learn structural knowledge and preserve detailed information for point cloud completion. Furthermore, we propose two more challenging benchmarks with more diverse incomplete point clouds that can better reflect the real-world scenarios to promote future research. Experimental results show that our method outperforms state-of-the-art methods by a large margin on both the new benchmarks and the existing ones. Code is available at https://github.com/yuxumin/PoinTr

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers<br>pdf: <a href="https://t.co/5LnPNERwB5">https://t.co/5LnPNERwB5</a><br>abs: <a href="https://t.co/7ejrxRXGEa">https://t.co/7ejrxRXGEa</a><br>github: <a href="https://t.co/fqeEjoBC8s">https://t.co/fqeEjoBC8s</a> <a href="https://t.co/Z3hJbipXXY">pic.twitter.com/Z3hJbipXXY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428526965141745672?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Estimating distinguishability measures on quantum computers

Rochisha Agarwal, Soorya Rethinasamy, Kunal Sharma, Mark M. Wilde

- retweets: 30, favorites: 29 (08/21/2021 09:06:01)

- links: [abs](https://arxiv.org/abs/2108.08406) | [pdf](https://arxiv.org/pdf/2108.08406)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent)

The performance of a quantum information processing protocol is ultimately judged by distinguishability measures that quantify how distinguishable the actual result of the protocol is from the ideal case. The most prominent distinguishability measures are those based on the fidelity and trace distance, due to their physical interpretations. In this paper, we propose and review several algorithms for estimating distinguishability measures based on trace distance and fidelity, and we evaluate their performance using simulators of quantum computers. The algorithms can be used for distinguishing quantum states, channels, and strategies (the last also known in the literature as "quantum combs"). The fidelity-based algorithms offer novel physical interpretations of these distinguishability measures in terms of the maximum probability with which a single prover (or competing provers) can convince a verifier to accept the outcome of an associated computation. We simulate these algorithms by using a variational approach with parameterized quantum circuits and find that they converge well for the examples that we consider.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">preprint &quot;Estimating distinguishability measures on quantum computers&quot; now available on the arXiv<a href="https://t.co/SPBRjwoxxg">https://t.co/SPBRjwoxxg</a><br><br>In collaboration with <a href="https://twitter.com/AgarwalRochisha?ref_src=twsrc%5Etfw">@AgarwalRochisha</a>, <a href="https://twitter.com/SooryaRethin?ref_src=twsrc%5Etfw">@SooryaRethin</a>, and <a href="https://twitter.com/kunal_phy?ref_src=twsrc%5Etfw">@kunal_phy</a></p>&mdash; Mark M. Wilde (@markwilde) <a href="https://twitter.com/markwilde/status/1428519237862273034?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Towards Controllable and Photorealistic Region-wise Image Manipulation

Ansheng You, Chenglin Zhou, Qixuan Zhang, Lan Xu

- retweets: 30, favorites: 26 (08/21/2021 09:06:02)

- links: [abs](https://arxiv.org/abs/2108.08674) | [pdf](https://arxiv.org/pdf/2108.08674)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Adaptive and flexible image editing is a desirable function of modern generative models. In this work, we present a generative model with auto-encoder architecture for per-region style manipulation. We apply a code consistency loss to enforce an explicit disentanglement between content and style latent representations, making the content and style of generated samples consistent with their corresponding content and style references. The model is also constrained by a content alignment loss to ensure the foreground editing will not interfere background contents. As a result, given interested region masks provided by users, our model supports foreground region-wise style transfer. Specially, our model receives no extra annotations such as semantic labels except for self-supervision. Extensive experiments show the effectiveness of the proposed method and exhibit the flexibility of the proposed model for various applications, including region-wise style editing, latent space interpolation, cross-domain style transfer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Controllable and Photorealistic Region-wise Image Manipulation<br>pdf: <a href="https://t.co/vG3LJQGkoT">https://t.co/vG3LJQGkoT</a><br>abs: <a href="https://t.co/ShjpROJVVm">https://t.co/ShjpROJVVm</a> <a href="https://t.co/rYe85j00Yp">pic.twitter.com/rYe85j00Yp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1428558884671791111?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval

Xinyu Zhang, Xueguang Ma, Peng Shi, Jimmy Lin

- retweets: 25, favorites: 30 (08/21/2021 09:06:02)

- links: [abs](https://arxiv.org/abs/2108.08787) | [pdf](https://arxiv.org/pdf/2108.08787)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

We present Mr. TyDi, a multi-lingual benchmark dataset for mono-lingual retrieval in eleven typologically diverse languages, designed to evaluate ranking with learned dense representations. The goal of this resource is to spur research in dense retrieval techniques in non-English languages, motivated by recent observations that existing techniques for representation learning perform poorly when applied to out-of-distribution data. As a starting point, we provide zero-shot baselines for this new dataset based on a multi-lingual adaptation of DPR that we call "mDPR". Experiments show that although the effectiveness of mDPR is much lower than BM25, dense representations nevertheless appear to provide valuable relevance signals, improving BM25 results in sparse-dense hybrids. In addition to analyses of our results, we also discuss future challenges and present a research agenda in multi-lingual dense retrieval. Mr. TyDi can be downloaded at https://github.com/castorini/mr.tydi.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share Mr. TyDi, a multi-lingual benchmark dataset for mono-lingual retrieval in 11 languages by <a href="https://twitter.com/crystina_z?ref_src=twsrc%5Etfw">@crystina_z</a> <a href="https://twitter.com/xueguang_ma?ref_src=twsrc%5Etfw">@xueguang_ma</a> <a href="https://twitter.com/ShiPeng16?ref_src=twsrc%5Etfw">@ShiPeng16</a> tl;dr - think of this as the open-retrieval condition of TyDi.<br><br>Paper: <a href="https://t.co/9qMoT1oYxd">https://t.co/9qMoT1oYxd</a><br>Data: <a href="https://t.co/rjsxwNA2r6">https://t.co/rjsxwNA2r6</a></p>&mdash; Jimmy Lin (@lintool) <a href="https://twitter.com/lintool/status/1428723443906850820?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



