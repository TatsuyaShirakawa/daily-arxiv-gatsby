---
title: Hot Papers 2021-04-09
date: 2021-04-10T09:00:11.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-09"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-09"
socialImage: "/media/flying-marine.jpg"

---

# 1. SiT: Self-supervised vIsion Transformer

Sara Atito, Muhammad Awais, Josef Kittler

- retweets: 5120, favorites: 356 (04/10/2021 09:00:11)

- links: [abs](https://arxiv.org/abs/2104.03602) | [pdf](https://arxiv.org/pdf/2104.03602)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Self-supervised learning methods are gaining increasing traction in computer vision due to their recent success in reducing the gap with supervised learning. In natural language processing (NLP) self-supervised learning and transformers are already the methods of choice. The recent literature suggests that the transformers are becoming increasingly popular also in computer vision. So far, the vision transformers have been shown to work well when pretrained either using a large scale supervised data or with some kind of co-supervision, e.g. in terms of teacher network. These supervised pretrained vision transformers achieve very good results in downstream tasks with minimal changes. In this work we investigate the merits of self-supervised learning for pretraining image/vision transformers and then using them for downstream classification tasks. We propose Self-supervised vIsion Transformers (SiT) and discuss several self-supervised training mechanisms to obtain a pretext model. The architectural flexibility of SiT allows us to use it as an autoencoder and work with multiple self-supervised tasks seamlessly. We show that a pretrained SiT can be finetuned for a downstream classification task on small scale datasets, consisting of a few thousand images rather than several millions. The proposed approach is evaluated on standard datasets using common protocols. The results demonstrate the strength of the transformers and their suitability for self-supervised learning. We outperformed existing self-supervised learning methods by large margin. We also observed that SiT is good for few shot learning and also showed that it is learning useful representation by simply training a linear classifier on top of the learned features from SiT. Pretraining, finetuning, and evaluation codes will be available under: https://github.com/Sara-Ahmed/SiT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SiT: Self-supervised vIsion Transformer<br>pdf: <a href="https://t.co/LUmX5fyCbn">https://t.co/LUmX5fyCbn</a><br>abs: <a href="https://t.co/ms4ksWdHnD">https://t.co/ms4ksWdHnD</a> <a href="https://t.co/0ALi1YPLTF">pic.twitter.com/0ALi1YPLTF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1380322070073118722?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Revisiting Simple Neural Probabilistic Language Models

Simeng Sun, Mohit Iyyer

- retweets: 2707, favorites: 412 (04/10/2021 09:00:12)

- links: [abs](https://arxiv.org/abs/2104.03474) | [pdf](https://arxiv.org/pdf/2104.03474)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Recent progress in language modeling has been driven not only by advances in neural architectures, but also through hardware and optimization improvements. In this paper, we revisit the neural probabilistic language model (NPLM) of~\citet{Bengio2003ANP}, which simply concatenates word embeddings within a fixed window and passes the result through a feed-forward network to predict the next word. When scaled up to modern hardware, this model (despite its many limitations) performs much better than expected on word-level language model benchmarks. Our analysis reveals that the NPLM achieves lower perplexity than a baseline Transformer with short input contexts but struggles to handle long-term dependencies. Inspired by this result, we modify the Transformer by replacing its first self-attention layer with the NPLM's local concatenation layer, which results in small but consistent perplexity decreases across three word-level language modeling datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Revisiting Simple Neural Probabilistic Language Models<br><br>Neural probabilistic language model of Bengio et al. (2003), which simply concatenates word embeddings within a fixed window, performs much better than expected when scaled up to modern hardware.<a href="https://t.co/HzLtTjpThG">https://t.co/HzLtTjpThG</a> <a href="https://t.co/AVqgbQcHM7">pic.twitter.com/AVqgbQcHM7</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1380325057910644736?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. InfinityGAN: Towards Infinite-Resolution Image Synthesis

Chieh Hubert Lin, Hsin-Ying Lee, Yen-Chi Cheng, Sergey Tulyakov, Ming-Hsuan Yang

- retweets: 1708, favorites: 356 (04/10/2021 09:00:12)

- links: [abs](https://arxiv.org/abs/2104.03963) | [pdf](https://arxiv.org/pdf/2104.03963)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present InfinityGAN, a method to generate arbitrary-resolution images. The problem is associated with several key challenges. First, scaling existing models to a high resolution is resource-constrained, both in terms of computation and availability of high-resolution training data. Infinity-GAN trains and infers patch-by-patch seamlessly with low computational resources. Second, large images should be locally and globally consistent, avoid repetitive patterns, and look realistic. To address these, InfinityGAN takes global appearance, local structure and texture into account.With this formulation, we can generate images with resolution and level of detail not attainable before. Experimental evaluation supports that InfinityGAN generates imageswith superior global structure compared to baselines at the same time featuring parallelizable inference. Finally, we how several applications unlocked by our approach, such as fusing styles spatially, multi-modal outpainting and image inbetweening at arbitrary input and output resolutions

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">InfinityGAN: Towards Infinite-Resolution Image Synthesis<br><br>Generates arbitrary resolution images by training and infering patch-by-patch seamlessly with low computational resources and superior global structure. <br><br>abs: <a href="https://t.co/ktGdisrMit">https://t.co/ktGdisrMit</a><br>site: <a href="https://t.co/NeqLcX1tyw">https://t.co/NeqLcX1tyw</a> <a href="https://t.co/k3zUaEhcIq">pic.twitter.com/k3zUaEhcIq</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1380319815680729095?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">InfinityGAN: Towards Infinite-Resolution Image Synthesis<br>pdf: <a href="https://t.co/HrNqpbxnhA">https://t.co/HrNqpbxnhA</a><br>abs: <a href="https://t.co/aZX57FbWS5">https://t.co/aZX57FbWS5</a><br>project page: <a href="https://t.co/KhKCzlpC8k">https://t.co/KhKCzlpC8k</a> <a href="https://t.co/tE7UMTsCyE">pic.twitter.com/tE7UMTsCyE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1380325067733745665?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">To infinity and beyond!<br>Is it possible to learn from limited-size images and generate images of infinite resolution? Check out &quot;InfinityGAN: Towards Infinite-Resolution Image Synthesis&quot;. (1/4)<br>Site: <a href="https://t.co/UH9e4GxMgE">https://t.co/UH9e4GxMgE</a><br>Abs: <a href="https://t.co/g5tptubQit">https://t.co/g5tptubQit</a><br>Complete video on site <a href="https://t.co/5ItF7mbfsi">pic.twitter.com/5ItF7mbfsi</a></p>&mdash; Hsin-Ying James Lee (@hyjameslee) <a href="https://twitter.com/hyjameslee/status/1380558819860697090?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Does Your Dermatology Classifier Know What It Doesn't Know? Detecting  the Long-Tail of Unseen Conditions

Abhijit Guha Roy, Jie Ren, Shekoofeh Azizi, Aaron Loh, Vivek Natarajan, Basil Mustafa, Nick Pawlowski, Jan Freyberg, Yuan Liu, Zach Beaver, Nam Vo, Peggy Bui, Samantha Winter, Patricia MacWilliams, Greg S. Corrado, Umesh Telang, Yun Liu, Taylan Cemgil, Alan Karthikesalingam, Balaji Lakshminarayanan, Jim Winkens

- retweets: 262, favorites: 188 (04/10/2021 09:00:12)

- links: [abs](https://arxiv.org/abs/2104.03829) | [pdf](https://arxiv.org/pdf/2104.03829)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We develop and rigorously evaluate a deep learning based system that can accurately classify skin conditions while detecting rare conditions for which there is not enough data available for training a confident classifier. We frame this task as an out-of-distribution (OOD) detection problem. Our novel approach, hierarchical outlier detection (HOD) assigns multiple abstention classes for each training outlier class and jointly performs a coarse classification of inliers vs. outliers, along with fine-grained classification of the individual classes. We demonstrate the effectiveness of the HOD loss in conjunction with modern representation learning approaches (BiT, SimCLR, MICLe) and explore different ensembling strategies for further improving the results. We perform an extensive subgroup analysis over conditions of varying risk levels and different skin types to investigate how the OOD detection performance changes over each subgroup and demonstrate the gains of our framework in comparison to baselines. Finally, we introduce a cost metric to approximate downstream clinical impact. We use this cost metric to compare the proposed method against a baseline system, thereby making a stronger case for the overall system effectiveness in a real-world deployment scenario.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new paper tackles an important safety hurdle for ML from code to clinic- “how does your dermatology classifier know what it doesn’t know?” In clinical practice patients may present with conditions unseen by ML systems in training, causing errors <a href="https://t.co/EgT3jOm6a5">https://t.co/EgT3jOm6a5</a> 1/2</p>&mdash; Alan Karthikesalingam (@alan_karthi) <a href="https://twitter.com/alan_karthi/status/1380459343586455553?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to announce our new paper &quot;Does Your Dermatology Classifier Know What It Doesn&#39;t Know?&quot;, led by <a href="https://twitter.com/abzz4ssj?ref_src=twsrc%5Etfw">@abzz4ssj</a> <a href="https://twitter.com/jessierenjie?ref_src=twsrc%5Etfw">@jessierenjie</a> <a href="https://twitter.com/jimwinkens?ref_src=twsrc%5Etfw">@jimwinkens</a> along with an awesome set of collaborators <a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> <a href="https://twitter.com/GoogleHealth?ref_src=twsrc%5Etfw">@GoogleHealth</a> <a href="https://twitter.com/DeepMind?ref_src=twsrc%5Etfw">@DeepMind</a>.<br><br>Paper: <a href="https://t.co/I237izZ1K5">https://t.co/I237izZ1K5</a><br><br>Thread 1/n: <a href="https://t.co/mZ9gG8tI99">pic.twitter.com/mZ9gG8tI99</a></p>&mdash; Balaji Lakshminarayanan (@balajiln) <a href="https://twitter.com/balajiln/status/1380582300971196418?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Does Your Dermatology Classifier Know What It Doesn&#39;t Know? Detecting the Long-Tail of Unseen Conditions<br>pdf: <a href="https://t.co/xZWC6FI5qA">https://t.co/xZWC6FI5qA</a><br>abs: <a href="https://t.co/jverk8mk3Y">https://t.co/jverk8mk3Y</a> <a href="https://t.co/UX76l6WXbR">pic.twitter.com/UX76l6WXbR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1380329935533920256?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new work on how to safely handle the long tail of conditions you didn&#39;t observe during training: <a href="https://t.co/5YbTFWQIGw">https://t.co/5YbTFWQIGw</a><br><br>Often training data does not capture all conditions you might see during deployment...<br><br>Led by <a href="https://twitter.com/abzz4ssj?ref_src=twsrc%5Etfw">@abzz4ssj</a> <a href="https://twitter.com/jessierenjie?ref_src=twsrc%5Etfw">@jessierenjie</a> <a href="https://twitter.com/balajiln?ref_src=twsrc%5Etfw">@balajiln</a> <a href="https://twitter.com/jimwinkens?ref_src=twsrc%5Etfw">@jimwinkens</a> <a href="https://t.co/Dg1IUudP41">https://t.co/Dg1IUudP41</a> <a href="https://t.co/01j7dxMbmy">pic.twitter.com/01j7dxMbmy</a></p>&mdash; Nick Pawlowski (@pwnic) <a href="https://twitter.com/pwnic/status/1380494171375566849?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. De-rendering the World's Revolutionary Artefacts

Shangzhe Wu, Ameesh Makadia, Jiajun Wu, Noah Snavely, Richard Tucker, Angjoo Kanazawa

- retweets: 275, favorites: 59 (04/10/2021 09:00:13)

- links: [abs](https://arxiv.org/abs/2104.03954) | [pdf](https://arxiv.org/pdf/2104.03954)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Recent works have shown exciting results in unsupervised image de-rendering -- learning to decompose 3D shape, appearance, and lighting from single-image collections without explicit supervision. However, many of these assume simplistic material and lighting models. We propose a method, termed RADAR, that can recover environment illumination and surface materials from real single-image collections, relying neither on explicit 3D supervision, nor on multi-view or multi-light images. Specifically, we focus on rotationally symmetric artefacts that exhibit challenging surface properties including specular reflections, such as vases. We introduce a novel self-supervised albedo discriminator, which allows the model to recover plausible albedo without requiring any ground-truth during training. In conjunction with a shape reconstruction module exploiting rotational symmetry, we present an end-to-end learning framework that is able to de-render the world's revolutionary artefacts. We conduct experiments on a real vase dataset and demonstrate compelling decomposition results, allowing for applications including free-viewpoint rendering and relighting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">De-rendering the World&#39;s Revolutionary Artefacts<br>pdf: <a href="https://t.co/DTQnEWgDKQ">https://t.co/DTQnEWgDKQ</a><br>abs: <a href="https://t.co/UzsiDPz8Ba">https://t.co/UzsiDPz8Ba</a><br>project page: <a href="https://t.co/XAvU8ZpBCV">https://t.co/XAvU8ZpBCV</a> <a href="https://t.co/OM3w2ACAdb">pic.twitter.com/OM3w2ACAdb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1380326615427788800?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. SOLD2: Self-supervised Occlusion-aware Line Description and Detection

Rémi Pautrat, Juan-Ting Lin, Viktor Larsson, Martin R. Oswald, Marc Pollefeys

- retweets: 225, favorites: 47 (04/10/2021 09:00:13)

- links: [abs](https://arxiv.org/abs/2104.03362) | [pdf](https://arxiv.org/pdf/2104.03362)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Compared to feature point detection and description, detecting and matching line segments offer additional challenges. Yet, line features represent a promising complement to points for multi-view tasks. Lines are indeed well-defined by the image gradient, frequently appear even in poorly textured areas and offer robust structural cues. We thus hereby introduce the first joint detection and description of line segments in a single deep network. Thanks to a self-supervised training, our method does not require any annotated line labels and can therefore generalize to any dataset. Our detector offers repeatable and accurate localization of line segments in images, departing from the wireframe parsing approach. Leveraging the recent progresses in descriptor learning, our proposed line descriptor is highly discriminative, while remaining robust to viewpoint changes and occlusions. We evaluate our approach against previous line detection and description methods on several multi-view datasets created with homographic warps as well as real-world viewpoint changes. Our full pipeline yields higher repeatability, localization accuracy and matching metrics, and thus represents a first step to bridge the gap with learned feature points methods. Code and trained weights are available at https://github.com/cvg/SOLD2.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SOLD2: Self-supervised Occlusion-aware Line Description and Detection<br><br>Rémi Pautrat, Juan-Ting Lin, Viktor Larsson, Martin R. Oswald, <a href="https://twitter.com/mapo1?ref_src=twsrc%5Etfw">@mapo1</a> <br><br>Tl;dr: SuperPoint-&gt;SuperLine +dynamic programming matching.<a href="https://t.co/kFMPPDKKR4">https://t.co/kFMPPDKKR4</a> <a href="https://t.co/x33HKpp0qm">pic.twitter.com/x33HKpp0qm</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1380403407425572868?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. An Information-Theoretic Proof of a Finite de Finetti Theorem

Lampros Gavalakis, Ioannis Kontoyiannis

- retweets: 165, favorites: 101 (04/10/2021 09:00:13)

- links: [abs](https://arxiv.org/abs/2104.03882) | [pdf](https://arxiv.org/pdf/2104.03882)
- [cs.IT](https://arxiv.org/list/cs.IT/recent) | [math.PR](https://arxiv.org/list/math.PR/recent)

A finite form of de Finetti's representation theorem is established using elementary information-theoretic tools: The distribution of the first $k$ random variables in an exchangeable binary vector of length $n\geq k$ is close to a mixture of product distributions. Closeness is measured in terms of the relative entropy and an explicit bound is provided.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A fun little result for those of you with a taste for relative entropy :-)<a href="https://t.co/qY93c5IYHk">https://t.co/qY93c5IYHk</a> <a href="https://t.co/sCz4NXt2f1">pic.twitter.com/sCz4NXt2f1</a></p>&mdash; Ioannis Kontoyiannis (@yiannis_entropy) <a href="https://twitter.com/yiannis_entropy/status/1380562640334901250?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A single gradient step finds adversarial examples on random two-layers  neural networks

Sébastien Bubeck, Yeshwanth Cherapanamjeri, Gauthier Gidel, Rémi Tachet des Combes

- retweets: 182, favorites: 47 (04/10/2021 09:00:13)

- links: [abs](https://arxiv.org/abs/2104.03863) | [pdf](https://arxiv.org/pdf/2104.03863)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Daniely and Schacham recently showed that gradient descent finds adversarial examples on random undercomplete two-layers ReLU neural networks. The term "undercomplete" refers to the fact that their proof only holds when the number of neurons is a vanishing fraction of the ambient dimension. We extend their result to the overcomplete case, where the number of neurons is larger than the dimension (yet also subexponential in the dimension). In fact we prove that a single step of gradient descent suffices. We also show this result for any subexponential width random neural network with smooth activation function.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New video: what can we say about adversarial examples at (random) initialization? <br><br>Based on joint work <a href="https://t.co/8msy9NuYiT">https://t.co/8msy9NuYiT</a> with Y. Cherapanamjeri, <a href="https://twitter.com/gauthier_gidel?ref_src=twsrc%5Etfw">@gauthier_gidel</a> and <a href="https://twitter.com/RemiTachet?ref_src=twsrc%5Etfw">@RemiTachet</a> .<a href="https://t.co/2eTt3LKaRH">https://t.co/2eTt3LKaRH</a></p>&mdash; Sebastien Bubeck (@SebastienBubeck) <a href="https://twitter.com/SebastienBubeck/status/1380378797652910082?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. CoCoNets: Continuous Contrastive 3D Scene Representations

Shamit Lal, Mihir Prabhudesai, Ishita Mediratta, Adam W. Harley, Katerina Fragkiadaki

- retweets: 144, favorites: 48 (04/10/2021 09:00:14)

- links: [abs](https://arxiv.org/abs/2104.03851) | [pdf](https://arxiv.org/pdf/2104.03851)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper explores self-supervised learning of amodal 3D feature representations from RGB and RGB-D posed images and videos, agnostic to object and scene semantic content, and evaluates the resulting scene representations in the downstream tasks of visual correspondence, object tracking, and object detection. The model infers a latent3D representation of the scene in the form of 3D feature points, where each continuous world 3D point is mapped to its corresponding feature vector. The model is trained for contrastive view prediction by rendering 3D feature clouds in queried viewpoints and matching against the 3D feature point cloud predicted from the query view. Notably, the representation can be queried for any 3D location, even if it is not visible from the input view. Our model brings together three powerful ideas of recent exciting research work: 3D feature grids as a neural bottleneck for view prediction, implicit functions for handling resolution limitations of 3D grids, and contrastive learning for unsupervised training of feature representations. We show the resulting 3D visual feature representations effectively scale across objects and scenes, imagine information occluded or missing from the input viewpoints, track objects over time, align semantically related objects in 3D, and improve 3D object detection. We outperform many existing state-of-the-art methods for 3D feature learning and view prediction, which are either limited by 3D grid spatial resolution, do not attempt to build amodal 3D representations, or do not handle combinatorial scene variability due to their non-convolutional bottlenecks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CoCoNets: Continuous Contrastive 3D Scene Representations<br>pdf: <a href="https://t.co/5pwIBahGfA">https://t.co/5pwIBahGfA</a><br>abs: <a href="https://t.co/alb3fMIXHS">https://t.co/alb3fMIXHS</a><br>project page: <a href="https://t.co/N19CjpA3rN">https://t.co/N19CjpA3rN</a> <a href="https://t.co/2xQ2V3weK9">pic.twitter.com/2xQ2V3weK9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1380327850801258501?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. How Transferable are Reasoning Patterns in VQA?

Corentin Kervadec, Theo Jaunet, Grigory Antipov, Moez Baccouche, Romain Vuillemot, Christian Wolf

- retweets: 62, favorites: 50 (04/10/2021 09:00:14)

- links: [abs](https://arxiv.org/abs/2104.03656) | [pdf](https://arxiv.org/pdf/2104.03656)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Since its inception, Visual Question Answering (VQA) is notoriously known as a task, where models are prone to exploit biases in datasets to find shortcuts instead of performing high-level reasoning. Classical methods address this by removing biases from training data, or adding branches to models to detect and remove biases. In this paper, we argue that uncertainty in vision is a dominating factor preventing the successful learning of reasoning in vision and language problems. We train a visual oracle and in a large scale study provide experimental evidence that it is much less prone to exploiting spurious dataset biases compared to standard models. We propose to study the attention mechanisms at work in the visual oracle and compare them with a SOTA Transformer-based model. We provide an in-depth analysis and visualizations of reasoning patterns obtained with an online visualization tool which we make publicly available (https://reasoningpatterns.github.io). We exploit these insights by transferring reasoning patterns from the oracle to a SOTA Transformer-based VQA model taking standard noisy visual inputs via fine-tuning. In experiments we report higher overall accuracy, as well as accuracy on infrequent answers for each question type, which provides evidence for improved generalization and a decrease of the dependency on dataset biases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> paper is on arxiv: &quot;How Transferrable are Reasoning Patterns in VQA?&quot;, by <a href="https://twitter.com/CorentK?ref_src=twsrc%5Etfw">@CorentK</a>, <a href="https://twitter.com/jaunet_theo?ref_src=twsrc%5Etfw">@jaunet_theo</a>, <a href="https://twitter.com/antigregory?ref_src=twsrc%5Etfw">@antigregory</a> , <a href="https://twitter.com/moezbac?ref_src=twsrc%5Etfw">@moezbac</a>, <a href="https://twitter.com/romsson?ref_src=twsrc%5Etfw">@romsson</a>. A deep analysis of attention in visual reasoning.<br><br>arXiv: <a href="https://t.co/XHRtAlSzzh">https://t.co/XHRtAlSzzh</a><br>interactive visualization: <a href="https://t.co/WJiJvJe0pG">https://t.co/WJiJvJe0pG</a> <a href="https://t.co/dcPXO1O3Tk">pic.twitter.com/dcPXO1O3Tk</a></p>&mdash; Christian Wolf (@chriswolfvision) <a href="https://twitter.com/chriswolfvision/status/1380442418705747970?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Pushing the Limits of Non-Autoregressive Speech Recognition

Edwin G. Ng, Chung-Cheng Chiu, Yu Zhang, William Chan

- retweets: 62, favorites: 42 (04/10/2021 09:00:14)

- links: [abs](https://arxiv.org/abs/2104.03416) | [pdf](https://arxiv.org/pdf/2104.03416)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

We combine recent advancements in end-to-end speech recognition to non-autoregressive automatic speech recognition. We push the limits of non-autoregressive state-of-the-art results for multiple datasets: LibriSpeech, Fisher+Switchboard and Wall Street Journal. Key to our recipe, we leverage CTC on giant Conformer neural network architectures with SpecAugment and wav2vec2 pre-training. We achieve 1.8%/3.6% WER on LibriSpeech test/test-other sets, 5.1%/9.8% WER on Switchboard, and 3.4% on the Wall Street Journal, all without a language model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pushing the Limits of Non-Autoregressive Speech Recognition<br><br>Achieves SotA non-AR performance and even outperforms many AR models on speech recognition without using a language model.<a href="https://t.co/rStqUfz95U">https://t.co/rStqUfz95U</a> <a href="https://t.co/AnXPQ6pYGx">pic.twitter.com/AnXPQ6pYGx</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1380335058410606593?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. 3D Shape Generation and Completion through Point-Voxel Diffusion

Linqi Zhou, Yilun Du, Jiajun Wu

- retweets: 42, favorites: 54 (04/10/2021 09:00:14)

- links: [abs](https://arxiv.org/abs/2104.03670) | [pdf](https://arxiv.org/pdf/2104.03670)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a novel approach for probabilistic generative modeling of 3D shapes. Unlike most existing models that learn to deterministically translate a latent vector to a shape, our model, Point-Voxel Diffusion (PVD), is a unified, probabilistic formulation for unconditional shape generation and conditional, multi-modal shape completion. PVD marries denoising diffusion models with the hybrid, point-voxel representation of 3D shapes. It can be viewed as a series of denoising steps, reversing the diffusion process from observed point cloud data to Gaussian noise, and is trained by optimizing a variational lower bound to the (conditional) likelihood function. Experiments demonstrate that PVD is capable of synthesizing high-fidelity shapes, completing partial point clouds, and generating multiple completion results from single-view depth scans of real objects.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">3D Shape Generation and Completion through Point-Voxel Diffusion<br>pdf: <a href="https://t.co/IHpYXiY4Z2">https://t.co/IHpYXiY4Z2</a><br>abs: <a href="https://t.co/S3kgGg6yMK">https://t.co/S3kgGg6yMK</a><br>project page: <a href="https://t.co/TDvHvEkdNQ">https://t.co/TDvHvEkdNQ</a> <a href="https://t.co/kvJHvHVwDh">pic.twitter.com/kvJHvHVwDh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1380328878476910594?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. On Biasing Transformer Attention Towards Monotonicity

Annette Rios, Chantal Amrhein, Noëmi Aepli, Rico Sennrich

- retweets: 58, favorites: 36 (04/10/2021 09:00:14)

- links: [abs](https://arxiv.org/abs/2104.03945) | [pdf](https://arxiv.org/pdf/2104.03945)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Many sequence-to-sequence tasks in natural language processing are roughly monotonic in the alignment between source and target sequence, and previous work has facilitated or enforced learning of monotonic attention behavior via specialized attention functions or pretraining. In this work, we introduce a monotonicity loss function that is compatible with standard attention mechanisms and test it on several sequence-to-sequence tasks: grapheme-to-phoneme conversion, morphological inflection, transliteration, and dialect normalization. Experiments show that we can achieve largely monotonic behavior. Performance is mixed, with larger gains on top of RNN baselines. General monotonicity does not benefit transformer multihead attention, however, we see isolated improvements when only a subset of heads is biased towards monotonic behavior.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Monotonic attention is popular for roughly monotonic seq2seq tasks.<br><br>We show that standard attention can be biased towards monotonicity via new loss, but RNNs more likely to benefit than Transformers.<a href="https://twitter.com/hashtag/NAACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#NAACL2021</a> by <a href="https://twitter.com/arios272?ref_src=twsrc%5Etfw">@arios272</a> <a href="https://twitter.com/chantalamrhein?ref_src=twsrc%5Etfw">@chantalamrhein</a> <a href="https://twitter.com/noeminaepli?ref_src=twsrc%5Etfw">@noeminaepli</a> <a href="https://t.co/qtRHPLxIPh">https://t.co/qtRHPLxIPh</a></p>&mdash; Rico Sennrich (@RicoSennrich) <a href="https://twitter.com/RicoSennrich/status/1380456732644761607?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Neural Temporal Point Processes: A Review

Oleksandr Shchur, Ali Caner Türkmen, Tim Januschowski, Stephan Günnemann

- retweets: 42, favorites: 48 (04/10/2021 09:00:14)

- links: [abs](https://arxiv.org/abs/2104.03528) | [pdf](https://arxiv.org/pdf/2104.03528)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Temporal point processes (TPP) are probabilistic generative models for continuous-time event sequences. Neural TPPs combine the fundamental ideas from point process literature with deep learning approaches, thus enabling construction of flexible and efficient models. The topic of neural TPPs has attracted significant attention in the recent years, leading to the development of numerous new architectures and applications for this class of models. In this review paper we aim to consolidate the existing body of knowledge on neural TPPs. Specifically, we focus on important design choices and general principles for defining neural TPP models. Next, we provide an overview of application areas commonly considered in the literature. We conclude this survey with the list of open challenges and important directions for future work in the field of neural TPPs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our review of Neural Temporal Point Processes <a href="https://t.co/KxuglTRw4V">https://t.co/KxuglTRw4V</a>. We analyze the important design choices for neural TPPs, talk about applications of these models, and discuss some of the main challenges that the field currently faces. <a href="https://t.co/mTiShVVFCS">pic.twitter.com/mTiShVVFCS</a></p>&mdash; Oleksandr Shchur (@shchur_) <a href="https://twitter.com/shchur_/status/1380435447516659713?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. EXPATS: A Toolkit for Explainable Automated Text Scoring

Hitoshi Manabe, Masato Hagiwara

- retweets: 36, favorites: 29 (04/10/2021 09:00:14)

- links: [abs](https://arxiv.org/abs/2104.03364) | [pdf](https://arxiv.org/pdf/2104.03364)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Automated text scoring (ATS) tasks, such as automated essay scoring and readability assessment, are important educational applications of natural language processing. Due to their interpretability of models and predictions, traditional machine learning (ML) algorithms based on handcrafted features are still in wide use for ATS tasks. Practitioners often need to experiment with a variety of models (including deep and traditional ML ones), features, and training objectives (regression and classification), although modern deep learning frameworks such as PyTorch require deep ML expertise to fully utilize. In this paper, we present EXPATS, an open-source framework to allow its users to develop and experiment with different ATS models quickly by offering flexible components, an easy-to-use configuration system, and the command-line interface. The toolkit also provides seamless integration with the Language Interpretability Tool (LIT) so that one can interpret and visualize models and their predictions. We also describe two case studies where we build ATS models quickly with minimal engineering efforts. The toolkit is available at \url{https://github.com/octanove/expats}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Launching EXPATS—an open-source toolkit for explainable automated text scoring! You can build &amp; interpret traditional+deep scoring models just by writing config files and CLI commands<br><br>Paper: <a href="https://t.co/8LjxyWtInA">https://t.co/8LjxyWtInA</a><br>Code: <a href="https://t.co/f8ficzvLYA">https://t.co/f8ficzvLYA</a><br><br>Joint work w/ Hitoshi <a href="https://twitter.com/ManaYsh13?ref_src=twsrc%5Etfw">@ManaYsh13</a></p>&mdash; Masato Hagiwara (@mhagiwara) <a href="https://twitter.com/mhagiwara/status/1380315130337112064?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. On tuning consistent annealed sampling for denoising score matching

Joan Serrà, Santiago Pascual, Jordi Pons

- retweets: 30, favorites: 25 (04/10/2021 09:00:15)

- links: [abs](https://arxiv.org/abs/2104.03725) | [pdf](https://arxiv.org/pdf/2104.03725)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Score-based generative models provide state-of-the-art quality for image and audio synthesis. Sampling from these models is performed iteratively, typically employing a discretized series of noise levels and a predefined scheme. In this note, we first overview three common sampling schemes for models trained with denoising score matching. Next, we focus on one of them, consistent annealed sampling, and study its hyper-parameter boundaries. We then highlight a possible formulation of such hyper-parameter that explicitly considers those boundaries and facilitates tuning when using few or a variable number of steps. Finally, we highlight some connections of the formulation with other sampling schemes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We&#39;ve lately been digging into tuning consistent annealed sampling for denoising score matching, and wrote a short note about it: <a href="https://t.co/eLzkXmPIsU">https://t.co/eLzkXmPIsU</a><br><br>w/ <a href="https://twitter.com/santty128?ref_src=twsrc%5Etfw">@santty128</a> and <a href="https://twitter.com/jordiponsdotme?ref_src=twsrc%5Etfw">@jordiponsdotme</a> at <a href="https://twitter.com/Dolby?ref_src=twsrc%5Etfw">@Dolby</a> AI <a href="https://t.co/H4jLCriR3H">pic.twitter.com/H4jLCriR3H</a></p>&mdash; Joan Serrà (@serrjoa) <a href="https://twitter.com/serrjoa/status/1380487671630430209?ref_src=twsrc%5Etfw">April 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



