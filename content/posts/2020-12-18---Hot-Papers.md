---
title: Hot Papers 2020-12-18
date: 2020-12-21T18:49:38.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Taming Transformers for High-Resolution Image Synthesis

Patrick Esser, Robin Rombach, Björn Ommer

- retweets: 5545, favorites: 5 (12/21/2020 18:49:38)

- links: [abs](https://arxiv.org/abs/2012.09841) | [pdf](https://arxiv.org/pdf/2012.09841)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Designed to learn long-range interactions on sequential data, transformers continue to show state-of-the-art results on a wide variety of tasks. In contrast to CNNs, they contain no inductive bias that prioritizes local interactions. This makes them expressive, but also computationally infeasible for long sequences, such as high-resolution images. We demonstrate how combining the effectiveness of the inductive bias of CNNs with the expressivity of transformers enables them to model and thereby synthesize high-resolution images. We show how to (i) use CNNs to learn a context-rich vocabulary of image constituents, and in turn (ii) utilize transformers to efficiently model their composition within high-resolution images. Our approach is readily applied to conditional synthesis tasks, where both non-spatial information, such as object classes, and spatial information, such as segmentations, can control the generated image. In particular, we present the first results on semantically-guided synthesis of megapixel images with transformers. Project page at https://compvis.github.io/taming-transformers/ .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Taming Transformers for High-Resolution Image Synthesis<br>pdf: <a href="https://t.co/fRwnXjKahS">https://t.co/fRwnXjKahS</a><br>abs: <a href="https://t.co/s9e42zZrrV">https://t.co/s9e42zZrrV</a><br>project page: <a href="https://t.co/aiA2PlSODq">https://t.co/aiA2PlSODq</a> <a href="https://t.co/emVvlP2vcg">pic.twitter.com/emVvlP2vcg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339754735658799106?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Variational Quantum Algorithms

M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, Patrick J. Coles

- retweets: 1907, favorites: 285 (12/21/2020 18:49:38)

- links: [abs](https://arxiv.org/abs/2012.09265) | [pdf](https://arxiv.org/pdf/2012.09265)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Applications such as simulating large quantum systems or solving large-scale linear algebra problems are immensely challenging for classical computers due their extremely high computational cost. Quantum computers promise to unlock these applications, although fault-tolerant quantum computers will likely not be available for several years. Currently available quantum devices have serious constraints, including limited qubit numbers and noise processes that limit circuit depth. Variational Quantum Algorithms (VQAs), which employ a classical optimizer to train a parametrized quantum circuit, have emerged as a leading strategy to address these constraints. VQAs have now been proposed for essentially all applications that researchers have envisioned for quantum computers, and they appear to the best hope for obtaining quantum advantage. Nevertheless, challenges remain including the trainability, accuracy, and efficiency of VQAs. In this review article we present an overview of the field of VQAs. Furthermore, we discuss strategies to overcome their challenges as well as the exciting prospects for using them as a means to obtain quantum advantage.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔥Today on arXiv we bring you a Review on Variational Quantum Algorithms (VQAs) 🔥<a href="https://t.co/9ojsa3Cwlt">https://t.co/9ojsa3Cwlt</a><br><br>We present the framework of VQAs, their applications, challenges and potential solutions, and how they could bring quantum advantage in the near term.</p>&mdash; Marco Cerezo (@MvsCerezo) <a href="https://twitter.com/MvsCerezo/status/1339775692355997697?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We hope our review article on Variational Quantum Algorithms is a helpful resource:<a href="https://t.co/DJ7tbBKUke">https://t.co/DJ7tbBKUke</a><br><br>This was a worldwide multi-institutional collaboration (see thread for details). <a href="https://t.co/3UwBW7FvXx">https://t.co/3UwBW7FvXx</a> <a href="https://t.co/NlElsEmtVC">pic.twitter.com/NlElsEmtVC</a></p>&mdash; Patrick Coles (@ColesQuantum) <a href="https://twitter.com/ColesQuantum/status/1340339759294664712?ref_src=twsrc%5Etfw">December 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. SceneFormer: Indoor Scene Generation with Transformers

Xinpeng Wang, Chandan Yeshwanth, Matthias Nießner

- retweets: 1520, favorites: 229 (12/21/2020 18:49:38)

- links: [abs](https://arxiv.org/abs/2012.09793) | [pdf](https://arxiv.org/pdf/2012.09793)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The task of indoor scene generation is to generate a sequence of objects, their locations and orientations conditioned on the shape and size of a room. Large scale indoor scene datasets allow us to extract patterns from user-designed indoor scenes and then generate new scenes based on these patterns. Existing methods rely on the 2D or 3D appearance of these scenes in addition to object positions, and make assumptions about the possible relations between objects. In contrast, we do not use any appearance information, and learn relations between objects using the self attention mechanism of transformers. We show that this leads to faster scene generation compared to existing methods with the same or better levels of realism. We build simple and effective generative models conditioned on the room shape, and on text descriptions of the room using only the cross-attention mechanism of transformers. We carried out a user study showing that our generated scenes are preferred over DeepSynth scenes 57.7% of the time for bedroom scenes, and 63.3% for living room scenes. In addition, we generate a scene in 1.48 seconds on average, 20% faster than the state of the art method Fast & Flexible, allowing interactive scene generation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating synthetic scenes using Transformers. <a href="https://t.co/ciMeffp2Gq">https://t.co/ciMeffp2Gq</a><br><br>Given an empty room, it figures out where to place an object (x, y, z, theta) and its size (l, w, h). All in an autoregressive manner (new object placement conditioned on the objects added already). <a href="https://t.co/Li57VV7e2y">pic.twitter.com/Li57VV7e2y</a></p>&mdash; Ankur Handa (@ankurhandos) <a href="https://twitter.com/ankurhandos/status/1339977949273944065?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SceneFormer: Indoor Scene Generation with Transformers<br>pdf: <a href="https://t.co/wQkgR5de3n">https://t.co/wQkgR5de3n</a><br>abs: <a href="https://t.co/UbmVRHsZ1U">https://t.co/UbmVRHsZ1U</a> <a href="https://t.co/trTUm0367s">pic.twitter.com/trTUm0367s</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339752800289435652?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a  Single Image

Ronghang Hu, Deepak Pathak

- retweets: 1482, favorites: 240 (12/21/2020 18:49:39)

- links: [abs](https://arxiv.org/abs/2012.09854) | [pdf](https://arxiv.org/pdf/2012.09854)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We present Worldsheet, a method for novel view synthesis using just a single RGB image as input. This is a challenging problem as it requires an understanding of the 3D geometry of the scene as well as texture mapping to generate both visible and occluded regions from new view-points. Our main insight is that simply shrink-wrapping a planar mesh sheet onto the input image, consistent with the learned intermediate depth, captures underlying geometry sufficient enough to generate photorealistic unseen views with arbitrarily large view-point changes. To operationalize this, we propose a novel differentiable texture sampler that allows our wrapped mesh sheet to be textured; which is then transformed into a target image via differentiable rendering. Our approach is category-agnostic, end-to-end trainable without using any 3D supervision and requires a single image at test time. Worldsheet consistently outperforms prior state-of-the-art methods on single-image view synthesis across several datasets. Furthermore, this simple idea captures novel views surprisingly well on a wide range of high resolution in-the-wild images in converting them into a navigable 3D pop-up. Video results and code at https://worldsheet.github.io

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image<br>pdf: <a href="https://t.co/XKbL6ra3fV">https://t.co/XKbL6ra3fV</a><br>abs: <a href="https://t.co/7UkNme2Dbq">https://t.co/7UkNme2Dbq</a><br>project page: <a href="https://t.co/QXSWr7Zdqp">https://t.co/QXSWr7Zdqp</a> <a href="https://t.co/SQIs3IV8U1">pic.twitter.com/SQIs3IV8U1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339770345214119937?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. ViNG: Learning Open-World Navigation with Visual Goals

Dhruv Shah, Benjamin Eysenbach, Gregory Kahn, Nicholas Rhinehart, Sergey Levine

- retweets: 1258, favorites: 181 (12/21/2020 18:49:39)

- links: [abs](https://arxiv.org/abs/2012.09812) | [pdf](https://arxiv.org/pdf/2012.09812)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a learning-based navigation system for reaching visually indicated goals and demonstrate this system on a real mobile robot platform. Learning provides an appealing alternative to conventional methods for robotic navigation: instead of reasoning about environments in terms of geometry and maps, learning can enable a robot to learn about navigational affordances, understand what types of obstacles are traversable (e.g., tall grass) or not (e.g., walls), and generalize over patterns in the environment. However, unlike conventional planning algorithms, it is harder to change the goal for a learned policy during deployment. We propose a method for learning to navigate towards a goal image of the desired destination. By combining a learned policy with a topological graph constructed out of previously observed data, our system can determine how to reach this visually indicated goal even in the presence of variable appearance and lighting. Three key insights, waypoint proposal, graph pruning and negative mining, enable our method to learn to navigate in real-world environments using only offline data, a setting where prior methods struggle. We instantiate our method on a real outdoor ground robot and show that our system, which we call ViNG, outperforms previously-proposed methods for goal-conditioned reinforcement learning, including other methods that incorporate reinforcement learning and search. We also study how ViNG generalizes to unseen environments and evaluate its ability to adapt to such an environment with growing experience. Finally, we demonstrate ViNG on a number of real-world applications, such as last-mile delivery and warehouse inspection. We encourage the reader to check out the videos of our experiments and demonstrations at our project website https://sites.google.com/view/ving-robot

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RL enables robots to navigate real-world environments, with diverse visually indicated goals: <a href="https://t.co/r6m5yJYrQW">https://t.co/r6m5yJYrQW</a><br><br>w/ <a href="https://twitter.com/_prieuredesion?ref_src=twsrc%5Etfw">@_prieuredesion</a>, B. Eysenbach, G. Kahn, <a href="https://twitter.com/nick_rhinehart?ref_src=twsrc%5Etfw">@nick_rhinehart</a> <br><br>paper: <a href="https://t.co/MRKmGStx6Y">https://t.co/MRKmGStx6Y</a><br>video: <a href="https://t.co/RZVVD2pku7">https://t.co/RZVVD2pku7</a><br><br>Thread below -&gt; <a href="https://t.co/mXD8N89bYc">pic.twitter.com/mXD8N89bYc</a></p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1339757657352142849?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Sparse Signal Models for Data Augmentation in Deep Learning ATR

Tushar Agarwal, Nithin Sugavanam, Emre Ertin

- retweets: 959, favorites: 11 (12/21/2020 18:49:39)

- links: [abs](https://arxiv.org/abs/2012.09284) | [pdf](https://arxiv.org/pdf/2012.09284)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent) | [eess.SP](https://arxiv.org/list/eess.SP/recent)

Automatic Target Recognition (ATR) algorithms classify a given Synthetic Aperture Radar (SAR) image into one of the known target classes using a set of training images available for each class. Recently, learning methods have shown to achieve state-of-the-art classification accuracy if abundant training data is available, sampled uniformly over the classes, and their poses. In this paper, we consider the task of ATR with a limited set of training images. We propose a data augmentation approach to incorporate domain knowledge and improve the generalization power of a data-intensive learning algorithm, such as a Convolutional neural network (CNN). The proposed data augmentation method employs a limited persistence sparse modeling approach, capitalizing on commonly observed characteristics of wide-angle synthetic aperture radar (SAR) imagery. Specifically, we exploit the sparsity of the scattering centers in the spatial domain and the smoothly-varying structure of the scattering coefficients in the azimuthal domain to solve the ill-posed problem of over-parametrized model fitting. Using this estimated model, we synthesize new images at poses and sub-pixel translations not available in the given data to augment CNN's training data. The experimental results show that for the training data starved region, the proposed method provides a significant gain in the resulting ATR algorithm's generalization performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sparse Signal Models for Data Augmentation in Deep Learning ATR. <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/Coding?src=hash&amp;ref_src=twsrc%5Etfw">#Coding</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a><a href="https://t.co/qfZpe6oY2M">https://t.co/qfZpe6oY2M</a> <a href="https://t.co/45CIE8xrkM">pic.twitter.com/45CIE8xrkM</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1340077114507313156?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Decentralized Finance, Centralized Ownership? An Iterative Mapping  Process to Measure Protocol Token Distribution

Matthias Nadler, Fabian Schär

- retweets: 663, favorites: 138 (12/21/2020 18:49:40)

- links: [abs](https://arxiv.org/abs/2012.09306) | [pdf](https://arxiv.org/pdf/2012.09306)
- [econ.GN](https://arxiv.org/list/econ.GN/recent) | [cs.CE](https://arxiv.org/list/cs.CE/recent)

In this paper, we analyze various Decentralized Finance (DeFi) protocols in terms of their token distributions. We propose an iterative mapping process that allows us to split aggregate token holdings from custodial and escrow contracts and assign them to their economic beneficiaries. This method accounts for liquidity-, lending-, and staking-pools, as well as token wrappers, and can be used to break down token holdings, even for high nesting levels. We compute individual address balances for several snapshots and analyze intertemporal distribution changes. In addition, we study reallocation and protocol usage data, and propose a proxy for measuring token dependencies and ecosystem integration. The paper offers new insights on DeFi interoperability as well as token ownership distribution and may serve as a foundation for further research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Decentralized Finance, Centralized Ownership? <br><br>Read our new working paper on ownership concentration &amp; wrapping complexity in the <a href="https://twitter.com/hashtag/DeFi?src=hash&amp;ref_src=twsrc%5Etfw">#DeFi</a> space. This is joint-work w/ Matthias Nadler <a href="https://twitter.com/mat_nadler?ref_src=twsrc%5Etfw">@mat_nadler</a>!<a href="https://t.co/RFWz3URMTd">https://t.co/RFWz3URMTd</a><br><br>cc: <a href="https://twitter.com/MAMA_global?ref_src=twsrc%5Etfw">@MAMA_global</a> <a href="https://twitter.com/defiprime?ref_src=twsrc%5Etfw">@defiprime</a> <a href="https://twitter.com/defipulse?ref_src=twsrc%5Etfw">@defipulse</a> <a href="https://twitter.com/DeFi_Dad?ref_src=twsrc%5Etfw">@DeFi_Dad</a> <a href="https://twitter.com/CamiRusso?ref_src=twsrc%5Etfw">@CamiRusso</a> <a href="https://t.co/Ah80BoWfbD">pic.twitter.com/Ah80BoWfbD</a></p>&mdash; Fabian Schär (@chainomics) <a href="https://twitter.com/chainomics/status/1339855484002693121?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DeFI Unicorn Token Ownership Structure 🔥<br><br>Nice research paper: &quot;Decentralized Finance, Centralized Ownership? An Iterative Mapping Process to Measure Protocol Token Distribution&quot;<a href="https://t.co/L40zR2Sx0q">https://t.co/L40zR2Sx0q</a> <a href="https://t.co/FY3B7VaRMp">https://t.co/FY3B7VaRMp</a> <a href="https://t.co/CYVMV1UjqS">pic.twitter.com/CYVMV1UjqS</a></p>&mdash; Julien Bouteloup (@bneiluj) <a href="https://twitter.com/bneiluj/status/1339863396116803585?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Unsupervised Learning of Local Discriminative Representation for Medical  Images

Huai Chen, Jieyu Li, Renzhen Wang, Yijie Huang, Fanrui Meng, Deyu Meng, Qing Peng, Lisheng Wang

- retweets: 780, favorites: 11 (12/21/2020 18:49:40)

- links: [abs](https://arxiv.org/abs/2012.09333) | [pdf](https://arxiv.org/pdf/2012.09333)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Local discriminative representation is needed in many medical image analysis tasks such as identifying sub-types of lesion or segmenting detailed components of anatomical structures by measuring similarity of local image regions. However, the commonly applied supervised representation learning methods require a large amount of annotated data, and unsupervised discriminative representation learning distinguishes different images by learning a global feature. In order to avoid the limitations of these two methods and be suitable for localized medical image analysis tasks, we introduce local discrimination into unsupervised representation learning in this work. The model contains two branches: one is an embedding branch which learns an embedding function to disperse dissimilar pixels over a low-dimensional hypersphere; and the other is a clustering branch which learns a clustering function to classify similar pixels into the same cluster. These two branches are trained simultaneously in a mutually beneficial pattern, and the learnt local discriminative representations are able to well measure the similarity of local image regions. These representations can be transferred to enhance various downstream tasks. Meanwhile, they can also be applied to cluster anatomical structures from unlabeled medical images under the guidance of topological priors from simulation or other structures with similar topological characteristics. The effectiveness and usefulness of the proposed method are demonstrated by enhancing various downstream tasks and clustering anatomical structures in retinal images and chest X-ray images. The corresponding code is available at https://github.com/HuaiChen-1994/LDLearning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Learning of Local Discriminative Representation for Medical Images.<a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/ML?src=hash&amp;ref_src=twsrc%5Etfw">#ML</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/NeuralNetworks?src=hash&amp;ref_src=twsrc%5Etfw">#NeuralNetworks</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a><a href="https://t.co/3Kb9gYqSxt">https://t.co/3Kb9gYqSxt</a> <a href="https://t.co/1fj1kTs38d">pic.twitter.com/1fj1kTs38d</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1340422453823234050?ref_src=twsrc%5Etfw">December 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Projected Distribution Loss for Image Enhancement

Mauricio Delbracio, Hossein Talebi, Peyman Milanfar

- retweets: 324, favorites: 161 (12/21/2020 18:49:40)

- links: [abs](https://arxiv.org/abs/2012.09289) | [pdf](https://arxiv.org/pdf/2012.09289)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Features obtained from object recognition CNNs have been widely used for measuring perceptual similarities between images. Such differentiable metrics can be used as perceptual learning losses to train image enhancement models. However, the choice of the distance function between input and target features may have a consequential impact on the performance of the trained model. While using the norm of the difference between extracted features leads to limited hallucination of details, measuring the distance between distributions of features may generate more textures; yet also more unrealistic details and artifacts. In this paper, we demonstrate that aggregating 1D-Wasserstein distances between CNN activations is more reliable than the existing approaches, and it can significantly improve the perceptual performance of enhancement models. More explicitly, we show that in imaging applications such as denoising, super-resolution, demosaicing, deblurring and JPEG artifact removal, the proposed learning loss outperforms the current state-of-the-art on reference-based perceptual losses. This means that the proposed learning loss can be plugged into different imaging frameworks and produce perceptually realistic results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We propose a loss function based on aggregate 1D Wasserstein distance on projected feature distributions. More stable with far fewer artifacts; improves on state-of-the-art perceptual quality in denoising, super-res, demosaic, deblur &amp; JPG artifact removal<a href="https://t.co/RokYHfjdNK">https://t.co/RokYHfjdNK</a> <a href="https://t.co/Fs5EZH0h0T">pic.twitter.com/Fs5EZH0h0T</a></p>&mdash; Peyman Milanfar (@docmilanfar) <a href="https://twitter.com/docmilanfar/status/1340167309227257856?ref_src=twsrc%5Etfw">December 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Deep Molecular Dreaming: Inverse machine learning for de-novo molecular  design and interpretability with surjective representations

Cynthia Shen, Mario Krenn, Sagi Eppel, Alan Aspuru-Guzik

- retweets: 302, favorites: 134 (12/21/2020 18:49:40)

- links: [abs](https://arxiv.org/abs/2012.09712) | [pdf](https://arxiv.org/pdf/2012.09712)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent)

Computer-based de-novo design of functional molecules is one of the most prominent challenges in cheminformatics today. As a result, generative and evolutionary inverse designs from the field of artificial intelligence have emerged at a rapid pace, with aims to optimize molecules for a particular chemical property. These models 'indirectly' explore the chemical space; by learning latent spaces, policies, distributions or by applying mutations on populations of molecules. However, the recent development of the SELFIES string representation of molecules, a surjective alternative to SMILES, have made possible other potential techniques. Based on SELFIES, we therefore propose PASITHEA, a direct gradient-based molecule optimization that applies inceptionism techniques from computer vision. PASITHEA exploits the use of gradients by directly reversing the learning process of a neural network, which is trained to predict real-valued chemical properties. Effectively, this forms an inverse regression model, which is capable of generating molecular variants optimized for a certain property. Although our results are preliminary, we observe a shift in distribution of a chosen property during inverse-training, a clear indication of PASITHEA's viability. A striking property of inceptionism is that we can directly probe the model's understanding of the chemical space it was trained on. We expect that extending PASITHEA to larger datasets, molecules and more complex properties will lead to advances in the design of new functional molecules as well as the interpretation and explanation of machine learning models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Ever heard of DeepDreaming? It&#39;s a method to inspect the inner working of <a href="https://twitter.com/hashtag/NeuralNets?src=hash&amp;ref_src=twsrc%5Etfw">#NeuralNets</a> &amp;to create amazing dreamlike images.<br><br>We adapt this great idea for molecular design: <a href="https://t.co/ezAcv6Zdfc">https://t.co/ezAcv6Zdfc</a><br><br>spearheaded by <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@UofT</a> undergrad Cynthia Shen, w/ S.Eppel, <a href="https://twitter.com/A_Aspuru_Guzik?ref_src=twsrc%5Etfw">@A_Aspuru_Guzik</a> <a href="https://twitter.com/hashtag/matterlab?src=hash&amp;ref_src=twsrc%5Etfw">#matterlab</a> <a href="https://t.co/BIC4F8faln">pic.twitter.com/BIC4F8faln</a></p>&mdash; Mario Krenn (@MarioKrenn6240) <a href="https://twitter.com/MarioKrenn6240/status/1339766024841867264?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this work with Cynthia Shen, <a href="https://twitter.com/EppelSagi?ref_src=twsrc%5Etfw">@EppelSagi</a> and <a href="https://twitter.com/MarioKrenn6240?ref_src=twsrc%5Etfw">@MarioKrenn6240</a> we  develop a deep dreaming algorithm for molecular design using SELFIES and we call it PASITHEA. Learn more about it here: <a href="https://t.co/Y9YKzbOpIp">https://t.co/Y9YKzbOpIp</a> <a href="https://twitter.com/hashtag/matterlab?src=hash&amp;ref_src=twsrc%5Etfw">#matterlab</a> <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@UofT</a> <a href="https://twitter.com/UofTCompSci?ref_src=twsrc%5Etfw">@UofTCompSci</a> <a href="https://twitter.com/chemuoft?ref_src=twsrc%5Etfw">@chemuoft</a> <a href="https://twitter.com/VectorInst?ref_src=twsrc%5Etfw">@VectorInst</a> <a href="https://twitter.com/hashtag/compchem?src=hash&amp;ref_src=twsrc%5Etfw">#compchem</a> <a href="https://t.co/w65VIQ40Iz">https://t.co/w65VIQ40Iz</a></p>&mdash; Alan Aspuru-Guzik (@A_Aspuru_Guzik) <a href="https://twitter.com/A_Aspuru_Guzik/status/1339767953403854848?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. A Generalization of Transformer Networks to Graphs

Vijay Prakash Dwivedi, Xavier Bresson

- retweets: 272, favorites: 131 (12/21/2020 18:49:41)

- links: [abs](https://arxiv.org/abs/2012.09699) | [pdf](https://arxiv.org/pdf/2012.09699)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a generalization of transformer neural network architecture for arbitrary graphs. The original transformer was designed for Natural Language Processing (NLP), which operates on fully connected graphs representing all connections between the words in a sequence. Such architecture does not leverage the graph connectivity inductive bias, and can perform poorly when the graph topology is important and has not been encoded into the node features. We introduce a graph transformer with four new properties compared to the standard model. First, the attention mechanism is a function of the neighborhood connectivity for each node in the graph. Second, the positional encoding is represented by the Laplacian eigenvectors, which naturally generalize the sinusoidal positional encodings often used in NLP. Third, the layer normalization is replaced by a batch normalization layer, which provides faster training and better generalization performance. Finally, the architecture is extended to edge feature representation, which can be critical to tasks s.a. chemistry (bond type) or link prediction (entity relationship in knowledge graphs). Numerical experiments on a graph benchmark demonstrate the performance of the proposed graph transformer architecture. This work closes the gap between the original transformer, which was designed for the limited case of line graphs, and graph neural networks, that can work with arbitrary graphs. As our architecture is simple and generic, we believe it can be used as a black box for future applications that wish to consider transformer and graphs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Generalization of Transformer Networks to Graphs<br>pdf: <a href="https://t.co/NSemyGeKIG">https://t.co/NSemyGeKIG</a><br>abs: <a href="https://t.co/zjQBkDtqZ4">https://t.co/zjQBkDtqZ4</a><br>github: <a href="https://t.co/42kkBMvPKE">https://t.co/42kkBMvPKE</a> <a href="https://t.co/IC44D5jx7v">pic.twitter.com/IC44D5jx7v</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339814087824433152?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Generalization of Transformer Networks to Graphs - A generalization of transformer neural network architecture for arbitrary graphs<br><br>Paper <a href="https://t.co/LbSywLsozM">https://t.co/LbSywLsozM</a><br><br>GitHub <a href="https://t.co/erz4dHrDaW">https://t.co/erz4dHrDaW</a><a href="https://twitter.com/hashtag/nlproc?src=hash&amp;ref_src=twsrc%5Etfw">#nlproc</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://t.co/gOkR0byhKq">pic.twitter.com/gOkR0byhKq</a></p>&mdash; Philip Vollet (@philipvollet) <a href="https://twitter.com/philipvollet/status/1339820319901687808?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Transformer Interpretability Beyond Attention Visualization

Hila Chefer, Shir Gur, Lior Wolf

- retweets: 198, favorites: 115 (12/21/2020 18:49:41)

- links: [abs](https://arxiv.org/abs/2012.09838) | [pdf](https://arxiv.org/pdf/2012.09838)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Self-attention techniques, and specifically Transformers, are dominating the field of text processing and are becoming increasingly popular in computer vision classification tasks. In order to visualize the parts of the image that led to a certain classification, existing methods either rely on the obtained attention maps, or employ heuristic propagation along the attention graph. In this work, we propose a novel way to compute relevancy for Transformer networks. The method assigns local relevance based on the deep Taylor decomposition principle and then propagates these relevancy scores through the layers. This propagation involves attention layers and skip connections, which challenge existing methods. Our solution is based on a specific formulation that is shown to maintain the total relevancy across layers. We benchmark our method on very recent visual Transformer networks, as well as on a text classification problem, and demonstrate a clear advantage over the existing explainability methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Transformer Interpretability Beyond Attention Visualization<br>pdf: <a href="https://t.co/2wAO7aOmJS">https://t.co/2wAO7aOmJS</a><br>abs: <a href="https://t.co/Lzcnnu8316">https://t.co/Lzcnnu8316</a><br>github: <a href="https://t.co/Ed2iSf8L2h">https://t.co/Ed2iSf8L2h</a> <a href="https://t.co/fVLaaNQjXc">pic.twitter.com/fVLaaNQjXc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339753654673432582?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Polyblur: Removing mild blur by polynomial reblurring

Mauricio Delbracio, Ignacio Garcia-Dorado, Sungjoon Choi, Damien Kelly, Peyman Milanfar

- retweets: 225, favorites: 80 (12/21/2020 18:49:41)

- links: [abs](https://arxiv.org/abs/2012.09322) | [pdf](https://arxiv.org/pdf/2012.09322)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

We present a highly efficient blind restoration method to remove mild blur in natural images. Contrary to the mainstream, we focus on removing slight blur that is often present, damaging image quality and commonly generated by small out-of-focus, lens blur, or slight camera motion. The proposed algorithm first estimates image blur and then compensates for it by combining multiple applications of the estimated blur in a principled way. To estimate blur we introduce a simple yet robust algorithm based on empirical observations about the distribution of the gradient in sharp natural images. Our experiments show that, in the context of mild blur, the proposed method outperforms traditional and modern blind deblurring methods and runs in a fraction of the time. Our method can be used to blindly correct blur before applying off-the-shelf deep super-resolution methods leading to superior results than other highly complex and computationally demanding techniques. The proposed method estimates and removes mild blur from a 12MP image on a modern mobile phone in a fraction of a second.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deblurring is unstable. Undoing &quot;mild&quot; blur is far more stable. We 1st estimate the blur &amp; then deblur by aggregating repeated applications of the *same* estimated blur. For mild blur, we outperform even deep methods, yet run in a fraction of the time. 1/2<a href="https://t.co/dfbcPvYpk9">https://t.co/dfbcPvYpk9</a> <a href="https://t.co/feJw6nDQUO">pic.twitter.com/feJw6nDQUO</a></p>&mdash; Peyman Milanfar (@docmilanfar) <a href="https://twitter.com/docmilanfar/status/1340837150284152832?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Neural Radiance Flow for 4D View Synthesis and Video Processing

Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B. Tenenbaum, Jiajun Wu

- retweets: 83, favorites: 75 (12/21/2020 18:49:41)

- links: [abs](https://arxiv.org/abs/2012.09790) | [pdf](https://arxiv.org/pdf/2012.09790)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We present a method, Neural Radiance Flow (NeRFlow),to learn a 4D spatial-temporal representation of a dynamic scene from a set of RGB images. Key to our approach is the use of a neural implicit representation that learns to capture the 3D occupancy, radiance, and dynamics of the scene. By enforcing consistency across different modalities, our representation enables multi-view rendering in diverse dynamic scenes, including water pouring, robotic interaction, and real images, outperforming state-of-the-art methods for spatial-temporal view synthesis. Our approach works even when inputs images are captured with only one camera. We further demonstrate that the learned representation can serve as an implicit scene prior, enabling video processing tasks such as image super-resolution and de-noising without any additional supervision.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Radiance Flow for 4D View Synthesis and Video Processing<br>pdf: <a href="https://t.co/u3MxBKXLLz">https://t.co/u3MxBKXLLz</a><br>abs: <a href="https://t.co/5VyWAZn2ol">https://t.co/5VyWAZn2ol</a><br>project page: <a href="https://t.co/JIIqXAUa7H">https://t.co/JIIqXAUa7H</a> <a href="https://t.co/YTvR7T7c0q">pic.twitter.com/YTvR7T7c0q</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339748763255123968?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Infinite Nature: Perpetual View Generation of Natural Scenes from a  Single Image

Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, Angjoo Kanazawa

- retweets: 74, favorites: 81 (12/21/2020 18:49:41)

- links: [abs](https://arxiv.org/abs/2012.09855) | [pdf](https://arxiv.org/pdf/2012.09855)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We introduce the problem of perpetual view generation -- long-range generation of novel views corresponding to an arbitrarily long camera trajectory given a single image. This is a challenging problem that goes far beyond the capabilities of current view synthesis methods, which work for a limited range of viewpoints and quickly degenerate when presented with a large camera motion. Methods designed for video generation also have limited ability to produce long video sequences and are often agnostic to scene geometry. We take a hybrid approach that integrates both geometry and image synthesis in an iterative render, refine, and repeat framework, allowing for long-range generation that cover large distances after hundreds of frames. Our approach can be trained from a set of monocular video sequences without any manual annotation. We propose a dataset of aerial footage of natural coastal scenes, and compare our method with recent view synthesis and conditional video generation baselines, showing that it can generate plausible scenes for much longer time horizons over large camera trajectories compared to existing methods. Please visit our project page at https://infinite-nature.github.io/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image<br>pdf: <a href="https://t.co/3IcEn1jEyr">https://t.co/3IcEn1jEyr</a><br>abs: <a href="https://t.co/MAwmKLO2XO">https://t.co/MAwmKLO2XO</a> <a href="https://t.co/D2WTRsB1Zm">pic.twitter.com/D2WTRsB1Zm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339771876030214145?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. BERT Goes Shopping: Comparing Distributional Models for Product  Representations

Federico Bianchi, Bingqing Yu, Jacopo Tagliabue

- retweets: 90, favorites: 45 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09807) | [pdf](https://arxiv.org/pdf/2012.09807)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

Word embeddings (e.g., word2vec) have been applied successfully to eCommerce products through prod2vec. Inspired by the recent performance improvements on several NLP tasks brought by contextualized embeddings, we propose to transfer BERT-like architectures to eCommerce: our model -- ProdBERT -- is trained to generate representations of products through masked session modeling. Through extensive experiments over multiple shops, different tasks, and a range of design choices, we systematically compare the accuracy of ProdBERT and prod2vec embeddings: while ProdBERT is found to be superior to traditional methods in several scenarios, we highlight the importance of resources and hyperparameters in the best performing models. Finally, we conclude by providing guidelines for training embeddings under a variety of computational and data constraints.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share our new pre-print: &quot;BERT Goes Shopping: Comparing Distributional Models for Product Representations&quot; with <a href="https://twitter.com/christineyyuu?ref_src=twsrc%5Etfw">@christineyyuu</a> and <a href="https://twitter.com/jacopotagliabue?ref_src=twsrc%5Etfw">@jacopotagliabue</a> <br><br>We introduce ProdBERT for eCommerce product representations.<br><br>pre-print: <a href="https://t.co/B1tCURnJqa">https://t.co/B1tCURnJqa</a><a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> <a href="https://twitter.com/hashtag/ecommerce?src=hash&amp;ref_src=twsrc%5Etfw">#ecommerce</a> <a href="https://twitter.com/hashtag/ai?src=hash&amp;ref_src=twsrc%5Etfw">#ai</a> <a href="https://t.co/YFDOpwyWl9">pic.twitter.com/YFDOpwyWl9</a></p>&mdash; Federico Bianchi (@fb_vinid) <a href="https://twitter.com/fb_vinid/status/1339749525540499457?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Classifying Sequences of Extreme Length with Constant Memory Applied to  Malware Detection

Edward Raff, William Fleshman, Richard Zak, Hyrum S. Anderson, Bobby Filar, Mark McLean

- retweets: 93, favorites: 38 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09390) | [pdf](https://arxiv.org/pdf/2012.09390)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent works within machine learning have been tackling inputs of ever-increasing size, with cybersecurity presenting sequence classification problems of particularly extreme lengths. In the case of Windows executable malware detection, inputs may exceed $100$ MB, which corresponds to a time series with $T=100,000,000$ steps. To date, the closest approach to handling such a task is MalConv, a convolutional neural network capable of processing up to $T=2,000,000$ steps. The $\mathcal{O}(T)$ memory of CNNs has prevented further application of CNNs to malware. In this work, we develop a new approach to temporal max pooling that makes the required memory invariant to the sequence length $T$. This makes MalConv $116\times$ more memory efficient, and up to $25.8\times$ faster to train on its original dataset, while removing the input length restrictions to MalConv. We re-invest these gains into improving the MalConv architecture by developing a new Global Channel Gating design, giving us an attention mechanism capable of learning feature interactions across 100 million time steps in an efficient manner, a capability lacked by the original MalConv CNN. Our implementation can be found at https://github.com/NeuromorphicComputationResearchProgram/MalConv2

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection<br><br>&quot;an attention mechanism capable of learning feature interactions across 100 million time steps in an efficient manner&quot;<a href="https://t.co/9kQfVp6kms">https://t.co/9kQfVp6kms</a><a href="https://t.co/QevfvdGbCm">https://t.co/QevfvdGbCm</a> <a href="https://t.co/B3TSXG21op">pic.twitter.com/B3TSXG21op</a></p>&mdash; Thomas (@evolvingstuff) <a href="https://twitter.com/evolvingstuff/status/1339787426932416513?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Learning to Recover 3D Scene Shape from a Single Image

Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, Chunhua Shen

- retweets: 78, favorites: 44 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09365) | [pdf](https://arxiv.org/pdf/2012.09365)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Despite significant progress in monocular depth estimation in the wild, recent state-of-the-art methods cannot be used to recover accurate 3D scene shape due to an unknown depth shift induced by shift-invariant reconstruction losses used in mixed-data depth prediction training, and possible unknown camera focal length. We investigate this problem in detail, and propose a two-stage framework that first predicts depth up to an unknown scale and shift from a single monocular image, and then use 3D point cloud encoders to predict the missing depth shift and focal length that allow us to recover a realistic 3D scene shape. In addition, we propose an image-level normalized regression loss and a normal-based geometry loss to enhance depth prediction models trained on mixed datasets. We test our depth model on nine unseen datasets and achieve state-of-the-art performance on zero-shot dataset generalization. Code is available at: https://git.io/Depth

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning to Recover 3D Scene Shape from a Single Image<br>pdf: <a href="https://t.co/EDVxkr8U9g">https://t.co/EDVxkr8U9g</a><br>abs: <a href="https://t.co/jaOohdPu4C">https://t.co/jaOohdPu4C</a> <a href="https://t.co/SzYN0NyQKI">pic.twitter.com/SzYN0NyQKI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339779317535006720?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. Parallel WaveNet conditioned on VAE latent vectors

Jonas Rohnke, Tom Merritt, Jaime Lorenzo-Trueba, Adam Gabrys, Vatsal Aggarwal, Alexis Moinet, Roberto Barra-Chicote

- retweets: 56, favorites: 49 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09703) | [pdf](https://arxiv.org/pdf/2012.09703)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Recently the state-of-the-art text-to-speech synthesis systems have shifted to a two-model approach: a sequence-to-sequence model to predict a representation of speech (typically mel-spectrograms), followed by a 'neural vocoder' model which produces the time-domain speech waveform from this intermediate speech representation. This approach is capable of synthesizing speech that is confusable with natural speech recordings. However, the inference speed of neural vocoder approaches represents a major obstacle for deploying this technology for commercial applications. Parallel WaveNet is one approach which has been developed to address this issue, trading off some synthesis quality for significantly faster inference speed. In this paper we investigate the use of a sentence-level conditioning vector to improve the signal quality of a Parallel WaveNet neural vocoder. We condition the neural vocoder with the latent vector from a pre-trained VAE component of a Tacotron 2-style sequence-to-sequence model. With this, we are able to significantly improve the quality of vocoded speech.

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">Parallel WaveNet conditioned on VAE latent vectors<br>pdf: <a href="https://t.co/12qVuv0Oqa">https://t.co/12qVuv0Oqa</a><br>abs: <a href="https://t.co/8u4VYpKlw6">https://t.co/8u4VYpKlw6</a> <a href="https://t.co/Zi7vCg5QbN">pic.twitter.com/Zi7vCg5QbN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339797271265861632?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. Task Uncertainty Loss Reduce Negative Transfer in Asymmetric Multi-task  Feature Learning

Rafael Peres da Silva, Chayaporn Suphavilai, Niranjan Nagarajan

- retweets: 100, favorites: 5 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09575) | [pdf](https://arxiv.org/pdf/2012.09575)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [q-bio.GN](https://arxiv.org/list/q-bio.GN/recent)

Multi-task learning (MTL) is frequently used in settings where a target task has to be learnt based on limited training data, but knowledge can be leveraged from related auxiliary tasks. While MTL can improve task performance overall relative to single-task learning (STL), these improvements can hide negative transfer (NT), where STL may deliver better performance for many individual tasks. Asymmetric multitask feature learning (AMTFL) is an approach that tries to address this by allowing tasks with higher loss values to have smaller influence on feature representations for learning other tasks. Task loss values do not necessarily indicate reliability of models for a specific task. We present examples of NT in two orthogonal datasets (image recognition and pharmacogenomics) and tackle this challenge by using aleatoric homoscedastic uncertainty to capture the relative confidence between tasks, and set weights for task loss. Our results show that this approach reduces NT providing a new approach to enable robust MTL.




# 21. Self-Supervised Sketch-to-Image Synthesis

Bingchen Liu, Yizhe Zhu, Kunpeng Song, Ahmed Elgammal

- retweets: 56, favorites: 41 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09290) | [pdf](https://arxiv.org/pdf/2012.09290)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Imagining a colored realistic image from an arbitrarily drawn sketch is one of the human capabilities that we eager machines to mimic. Unlike previous methods that either requires the sketch-image pairs or utilize low-quantity detected edges as sketches, we study the exemplar-based sketch-to-image (s2i) synthesis task in a self-supervised learning manner, eliminating the necessity of the paired sketch data. To this end, we first propose an unsupervised method to efficiently synthesize line-sketches for general RGB-only datasets. With the synthetic paired-data, we then present a self-supervised Auto-Encoder (AE) to decouple the content/style features from sketches and RGB-images, and synthesize images that are both content-faithful to the sketches and style-consistent to the RGB-images. While prior works employ either the cycle-consistence loss or dedicated attentional modules to enforce the content/style fidelity, we show AE's superior performance with pure self-supervisions. To further improve the synthesis quality in high resolution, we also leverage an adversarial network to refine the details of synthetic images. Extensive experiments on 1024*1024 resolution demonstrate a new state-of-art-art performance of the proposed model on CelebA-HQ and Wiki-Art datasets. Moreover, with the proposed sketch generator, the model shows a promising performance on style mixing and style transfer, which require synthesized images to be both style-consistent and semantically meaningful. Our code is available on https://github.com/odegeasslbc/Self-Supervised-Sketch-to-Image-Synthesis-PyTorch, and please visit https://create.playform.io/my-projects?mode=sketch for an online demo of our model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Supervised Sketch-to-Image Synthesis<br>pdf: <a href="https://t.co/HthysJ8HzS">https://t.co/HthysJ8HzS</a><br>abs: <a href="https://t.co/acVBlkImCx">https://t.co/acVBlkImCx</a><br>github: <a href="https://t.co/OFQdsw6Haa">https://t.co/OFQdsw6Haa</a> <a href="https://t.co/58nSLPakKg">pic.twitter.com/58nSLPakKg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339766588770283520?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 22. Computation-Efficient Knowledge Distillation via Uncertainty-Aware Mixup

Guodong Xu, Ziwei Liu, Chen Change Loy

- retweets: 32, favorites: 52 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09413) | [pdf](https://arxiv.org/pdf/2012.09413)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Knowledge distillation, which involves extracting the "dark knowledge" from a teacher network to guide the learning of a student network, has emerged as an essential technique for model compression and transfer learning. Unlike previous works that focus on the accuracy of student network, here we study a little-explored but important question, i.e., knowledge distillation efficiency. Our goal is to achieve a performance comparable to conventional knowledge distillation with a lower computation cost during training. We show that the UNcertainty-aware mIXup (UNIX) can serve as a clean yet effective solution. The uncertainty sampling strategy is used to evaluate the informativeness of each training sample. Adaptive mixup is applied to uncertain samples to compact knowledge. We further show that the redundancy of conventional knowledge distillation lies in the excessive learning of easy samples. By combining uncertainty and mixup, our approach reduces the redundancy and makes better use of each query to the teacher network. We validate our approach on CIFAR100 and ImageNet. Notably, with only 79% computation cost, we outperform conventional knowledge distillation on CIFAR100 and achieve a comparable result on ImageNet.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Computation-Efficient Knowledge Distillation via Uncertainty-Aware Mixup<a href="https://t.co/IvLQNeb5Hc">https://t.co/IvLQNeb5Hc</a><br> 生徒モデルにとって難しいサンプルのみを使用して教師モデルの計算コストを削減。mixupはaugmentationではなくkdの観点から使用（詳しくは論文）。何かのコンペで使いたい。 <a href="https://t.co/OqSk6VhFhQ">pic.twitter.com/OqSk6VhFhQ</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1339995342612254720?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 23. End-to-End Human Pose and Mesh Reconstruction with Transformers

Kevin Lin, Lijuan Wang, Zicheng Liu

- retweets: 44, favorites: 34 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09760) | [pdf](https://arxiv.org/pdf/2012.09760)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a new method, called MEsh TRansfOrmer (METRO), to reconstruct 3D human pose and mesh vertices from a single image. Our method uses a transformer encoder to jointly model vertex-vertex and vertex-joint interactions, and outputs 3D joint coordinates and mesh vertices simultaneously. Compared to existing techniques that regress pose and shape parameters, METRO does not rely on any parametric mesh models like SMPL, thus it can be easily extended to other objects such as hands. We further relax the mesh topology and allow the transformer self-attention mechanism to freely attend between any two vertices, making it possible to learn non-local relationships among mesh vertices and joints. With the proposed masked vertex modeling, our method is more robust and effective in handling challenging situations like partial occlusions. METRO generates new state-of-the-art results for human mesh reconstruction on the public Human3.6M and 3DPW datasets. Moreover, we demonstrate the generalizability of METRO to 3D hand reconstruction in the wild, outperforming existing state-of-the-art methods on FreiHAND dataset.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">End-to-End Human Pose and Mesh Reconstruction with Transformers<br>pdf: <a href="https://t.co/lbv4euzZ07">https://t.co/lbv4euzZ07</a><br>abs: <a href="https://t.co/PjYbmYhGC8">https://t.co/PjYbmYhGC8</a> <a href="https://t.co/fQHcdzFY1B">pic.twitter.com/fQHcdzFY1B</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339752158879690752?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 24. On the experimental feasibility of quantum state reconstruction via  machine learning

Sanjaya Lohani, Thomas A. Searles, Brian T. Kirby, Ryan T. Glasser

- retweets: 55, favorites: 15 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09432) | [pdf](https://arxiv.org/pdf/2012.09432)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We determine the resource scaling of machine learning-based quantum state reconstruction methods, in terms of both inference and training, for systems of up to four qubits. Further, we examine system performance in the low-count regime, likely to be encountered in the tomography of high-dimensional systems. Finally, we implement our quantum state reconstruction method on a IBM Q quantum computer and confirm our results.




# 25. Towards Resolving the Implicit Bias of Gradient Descent for Matrix  Factorization: Greedy Low-Rank Learning

Zhiyuan Li, Yuping Luo, Kaifeng Lyu

- retweets: 20, favorites: 45 (12/21/2020 18:49:42)

- links: [abs](https://arxiv.org/abs/2012.09839) | [pdf](https://arxiv.org/pdf/2012.09839)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Matrix factorization is a simple and natural test-bed to investigate the implicit regularization of gradient descent. Gunasekar et al. (2018) conjectured that Gradient Flow with infinitesimal initialization converges to the solution that minimizes the nuclear norm, but a series of recent papers argued that the language of norm minimization is not sufficient to give a full characterization for the implicit regularization. In this work, we provide theoretical and empirical evidence that for depth-2 matrix factorization, gradient flow with infinitesimal initialization is mathematically equivalent to a simple heuristic rank minimization algorithm, Greedy Low-Rank Learning, under some reasonable assumptions. This generalizes the rank minimization view from previous works to a much broader setting and enables us to construct counter-examples to refute the conjecture from Gunasekar et al. (2018). We also extend the results to the case where depth $\ge 3$, and we show that the benefit of being deeper is that the above convergence has a much weaker dependence over initialization magnitude so that this rank minimization is more likely to take effect for initialization with practical scale.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">For matrix factorization, GD + tiny init is equivalent to a heuristic rank-minimization algorithm, GLRL.<br><br>This negatively resolves the conjecture by Gunasekar et al., 2017, GD + tiny init minimizes nuclear norm.<br><br>paper: <a href="https://t.co/cWrq6OfERd">https://t.co/cWrq6OfERd</a><br>w/ <a href="https://twitter.com/luo_yuping?ref_src=twsrc%5Etfw">@luo_yuping</a>, <a href="https://twitter.com/vfleaking?ref_src=twsrc%5Etfw">@vfleaking</a><br>(1/4) <a href="https://t.co/aPlx5w67Bs">pic.twitter.com/aPlx5w67Bs</a></p>&mdash; Zhiyuan Li (@zhiyuanli_) <a href="https://twitter.com/zhiyuanli_/status/1339977649884700674?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 26. PCT: Point Cloud Transformer

Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu

- retweets: 16, favorites: 48 (12/21/2020 18:49:43)

- links: [abs](https://arxiv.org/abs/2012.09688) | [pdf](https://arxiv.org/pdf/2012.09688)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The irregular domain and lack of ordering make it challenging to design deep neural networks for point cloud processing. This paper presents a novel framework named Point Cloud Transformer(PCT) for point cloud learning. PCT is based on Transformer, which achieves huge success in natural language processing and displays great potential in image processing. It is inherently permutation invariant for processing a sequence of points, making it well-suited for point cloud learning. To better capture local context within the point cloud, we enhance input embedding with the support of farthest point sampling and nearest neighbor search. Extensive experiments demonstrate that the PCT achieves the state-of-the-art performance on shape classification, part segmentation and normal estimation tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PCT: Point Cloud Transformer<br>pdf: <a href="https://t.co/ol61Z3dKZb">https://t.co/ol61Z3dKZb</a><br>abs: <a href="https://t.co/HtUBIfaM3X">https://t.co/HtUBIfaM3X</a> <a href="https://t.co/XAWugJ8DeS">pic.twitter.com/XAWugJ8DeS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339751139370266630?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 27. Learning Cross-Domain Correspondence for Control with Dynamics  Cycle-Consistency

Qiang Zhang, Tete Xiao, Alexei A. Efros, Lerrel Pinto, Xiaolong Wang

- retweets: 20, favorites: 36 (12/21/2020 18:49:43)

- links: [abs](https://arxiv.org/abs/2012.09811) | [pdf](https://arxiv.org/pdf/2012.09811)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

At the heart of many robotics problems is the challenge of learning correspondences across domains. For instance, imitation learning requires obtaining correspondence between humans and robots; sim-to-real requires correspondence between physics simulators and the real world; transfer learning requires correspondences between different robotics environments. This paper aims to learn correspondence across domains differing in representation (vision vs. internal state), physics parameters (mass and friction), and morphology (number of limbs). Importantly, correspondences are learned using unpaired and randomly collected data from the two domains. We propose \textit{dynamics cycles} that align dynamic robot behavior across two domains using a cycle-consistency constraint. Once this correspondence is found, we can directly transfer the policy trained on one domain to the other, without needing any additional fine-tuning on the second domain. We perform experiments across a variety of problem domains, both in simulation and on real robot. Our framework is able to align uncalibrated monocular video of a real robot arm to dynamic state-action trajectories of a simulated arm without paired data. Video demonstrations of our results are available at: https://sjtuzq.github.io/cycle_dynamics.html .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">After adding time in cycles, it is time to add dynamics in cycles (<a href="https://t.co/dcNojvuVVM">https://t.co/dcNojvuVVM</a>).  <br><br>We add a forward dynamics model in CycleGAN to learn correspondence and align dynamic robot behavior across two domains differing in observed representation, physics, and morphology. <a href="https://t.co/2WLwFY0TXa">pic.twitter.com/2WLwFY0TXa</a></p>&mdash; Xiaolong Wang (@xiaolonw) <a href="https://twitter.com/xiaolonw/status/1340038933543960576?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 28. Roof-GAN: Learning to Generate Roof Geometry and Relations for  Residential Houses

Yiming Qian, Hao Zhang, Yasutaka Furukawa

- retweets: 12, favorites: 40 (12/21/2020 18:49:43)

- links: [abs](https://arxiv.org/abs/2012.09340) | [pdf](https://arxiv.org/pdf/2012.09340)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper presents Roof-GAN, a novel generative adversarial network that generates structured geometry of residential roof structures as a set of roof primitives and their relationships. Given the number of primitives, the generator produces a structured roof model as a graph, which consists of 1) primitive geometry as raster images at each node, encoding facet segmentation and angles; 2) inter-primitive colinear/coplanar relationships at each edge; and 3) primitive geometry in a vector format at each node, generated by a novel differentiable vectorizer while enforcing the relationships. The discriminator is trained to assess the primitive raster geometry, the primitive relationships, and the primitive vector geometry in a fully end-to-end architecture. Qualitative and quantitative evaluations demonstrate the effectiveness of our approach in generating diverse and realistic roof models over the competing methods with a novel metric proposed in this paper for the task of structured geometry generation. We will share our code and data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Roof-GAN: Learning to Generate Roof Geometry and Relations for Residential Houses<br>pdf: <a href="https://t.co/7ZjS7RkM5L">https://t.co/7ZjS7RkM5L</a><br>abs: <a href="https://t.co/C0hX3br5nU">https://t.co/C0hX3br5nU</a> <a href="https://t.co/32eyiJa4VN">pic.twitter.com/32eyiJa4VN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1339761343608610816?ref_src=twsrc%5Etfw">December 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



