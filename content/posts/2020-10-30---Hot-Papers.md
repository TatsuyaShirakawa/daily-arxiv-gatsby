---
title: Hot Papers 2020-10-30
date: 2020-10-31T09:40:17.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-30"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-30"
socialImage: "/media/flying-marine.jpg"

---

# 1. The De-democratization of AI: Deep Learning and the Compute Divide in  Artificial Intelligence Research

Nur Ahmed, Muntasir Wahed

- retweets: 4696, favorites: 218 (10/31/2020 09:40:17)

- links: [abs](https://arxiv.org/abs/2010.15581) | [pdf](https://arxiv.org/pdf/2010.15581)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Increasingly, modern Artificial Intelligence (AI) research has become more computationally intensive. However, a growing concern is that due to unequal access to computing power, only certain firms and elite universities have advantages in modern AI research. Using a novel dataset of 171394 papers from 57 prestigious computer science conferences, we document that firms, in particular, large technology firms and elite universities have increased participation in major AI conferences since deep learning's unanticipated rise in 2012. The effect is concentrated among elite universities, which are ranked 1-50 in the QS World University Rankings. Further, we find two strategies through which firms increased their presence in AI research: first, they have increased firm-only publications; and second, firms are collaborating primarily with elite universities. Consequently, this increased presence of firms and elite universities in AI research has crowded out mid-tier (QS ranked 201-300) and lower-tier (QS ranked 301-500) universities. To provide causal evidence that deep learning's unanticipated rise resulted in this divergence, we leverage the generalized synthetic control method, a data-driven counterfactual estimator. Using machine learning based text analysis methods, we provide additional evidence that the divergence between these two groups - large firms and non-elite universities - is driven by access to computing power or compute, which we term as the "compute divide". This compute divide between large firms and non-elite universities increases concerns around bias and fairness within AI technology, and presents an obstacle towards "democratizing" AI. These results suggest that a lack of access to specialized equipment such as compute can de-democratize knowledge production.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">There is a significant inequality in AI research according to a new paper <a href="https://t.co/Ab6cSOSwp6">https://t.co/Ab6cSOSwp6</a><br><br>Core findings:<br><br>1.Big Tech &amp; elite universities have increased presence in top AI conf&#39;s since the rise of deep learning<br><br>2.Their increased presence has crowded out non-elite unis <a href="https://t.co/Zp4xtwdObF">pic.twitter.com/Zp4xtwdObF</a></p>&mdash; Abeba Birhane (@Abebab) <a href="https://twitter.com/Abebab/status/1322229024387727360?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Understanding the Failure Modes of Out-of-Distribution Generalization

Vaishnavh Nagarajan, Anders Andreassen, Behnam Neyshabur

- retweets: 3026, favorites: 271 (10/31/2020 09:40:18)

- links: [abs](https://arxiv.org/abs/2010.15775) | [pdf](https://arxiv.org/pdf/2010.15775)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Empirical studies suggest that machine learning models often rely on features, such as the background, that may be spuriously correlated with the label only during training time, resulting in poor accuracy during test-time. In this work, we identify the fundamental factors that give rise to this behavior, by explaining why models fail this way {\em even} in easy-to-learn tasks where one would expect these models to succeed. In particular, through a theoretical study of gradient-descent-trained linear classifiers on some easy-to-learn tasks, we uncover two complementary failure modes. These modes arise from how spurious correlations induce two kinds of skews in the data: one geometric in nature, and another, statistical in nature. Finally, we construct natural modifications of image classification datasets to understand when these failure modes can arise in practice. We also design experiments to isolate the two failure modes when training modern neural networks on these datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">“Understanding the failure modes of out-of-distribution generalization”, new paper w/ <a href="https://twitter.com/bneyshabur?ref_src=twsrc%5Etfw">@bneyshabur</a> and <a href="https://twitter.com/AJAndreassen?ref_src=twsrc%5Etfw">@AJAndreassen</a> at Google <a href="https://t.co/RcHFLtdXGQ">https://t.co/RcHFLtdXGQ</a><br><br>We explain why classifiers rely on spurious correlations (e.g. bkgd.) that hold only in training. 1/ <a href="https://t.co/K48ujnu5h7">pic.twitter.com/K48ujnu5h7</a></p>&mdash; Vaishnavh Nagarajan (@_vaishnavh) <a href="https://twitter.com/_vaishnavh/status/1321990856644190208?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. RelationNet++: Bridging Visual Representations for Object Detection via  Transformer Decoder

Cheng Chi, Fangyun Wei, Han Hu

- retweets: 961, favorites: 135 (10/31/2020 09:40:18)

- links: [abs](https://arxiv.org/abs/2010.15831) | [pdf](https://arxiv.org/pdf/2010.15831)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Existing object detection frameworks are usually built on a single format of object/part representation, i.e., anchor/proposal rectangle boxes in RetinaNet and Faster R-CNN, center points in FCOS and RepPoints, and corner points in CornerNet. While these different representations usually drive the frameworks to perform well in different aspects, e.g., better classification or finer localization, it is in general difficult to combine these representations in a single framework to make good use of each strength, due to the heterogeneous or non-grid feature extraction by different representations. This paper presents an attention-based decoder module similar as that in Transformer~\cite{vaswani2017attention} to bridge other representations into a typical object detector built on a single representation format, in an end-to-end fashion. The other representations act as a set of \emph{key} instances to strengthen the main \emph{query} representation features in the vanilla detectors. Novel techniques are proposed towards efficient computation of the decoder module, including a \emph{key sampling} approach and a \emph{shared location embedding} approach. The proposed module is named \emph{bridging visual representations} (BVR). It can perform in-place and we demonstrate its broad effectiveness in bridging other representations into prevalent object detection frameworks, including RetinaNet, Faster R-CNN, FCOS and ATSS, where about $1.5\sim3.0$ AP improvements are achieved. In particular, we improve a state-of-the-art framework with a strong backbone by about $2.0$ AP, reaching $52.7$ AP on COCO test-dev. The resulting network is named RelationNet++. The code will be available at https://github.com/microsoft/RelationNet2.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RelationNet++: Bridging Visual Representations for<br>Object Detection via Transformer Decoder<br>pdf: <a href="https://t.co/Ed7eg3RfnY">https://t.co/Ed7eg3RfnY</a><br>abs: <a href="https://t.co/5FNWqpmSka">https://t.co/5FNWqpmSka</a><br>github: <a href="https://t.co/w2wcpnrxlq">https://t.co/w2wcpnrxlq</a> <a href="https://t.co/oejSz7rFME">pic.twitter.com/oejSz7rFME</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1321996852938551296?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural  Network Representations Vary with Width and Depth

Thao Nguyen, Maithra Raghu, Simon Kornblith

- retweets: 266, favorites: 135 (10/31/2020 09:40:18)

- links: [abs](https://arxiv.org/abs/2010.15327) | [pdf](https://arxiv.org/pdf/2010.15327)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

A key factor in the success of deep neural networks is the ability to scale models to improve performance by varying the architecture depth and width. This simple property of neural network design has resulted in highly effective architectures for a variety of tasks. Nevertheless, there is limited understanding of effects of depth and width on the learned representations. In this paper, we study this fundamental question. We begin by investigating how varying depth and width affects model hidden representations, finding a characteristic block structure in the hidden representations of larger capacity (wider or deeper) models. We demonstrate that this block structure arises when model capacity is large relative to the size of the training set, and is indicative of the underlying layers preserving and propagating the dominant principal component of their representations. This discovery has important ramifications for features learned by different models, namely, representations outside the block structure are often similar across architectures with varying widths and depths, but the block structure is unique to each model. We analyze the output predictions of different model architectures, finding that even when the overall accuracy is similar, wide and deep models exhibit distinctive error patterns and variations across classes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do Wide and Deep neural networks Learn the Same Things? <br>Paper: <a href="https://t.co/tnLUsKrNo5">https://t.co/tnLUsKrNo5</a><br><br>We study representational properties of neural networks with different depths and widths on CIFAR/ImageNet, with insights on model capacity effects, feature similarity &amp; characteristic errors <a href="https://t.co/YdsuRy6SBt">https://t.co/YdsuRy6SBt</a></p>&mdash; Maithra Raghu (@maithra_raghu) <a href="https://twitter.com/maithra_raghu/status/1322301181687857152?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do wide and deep neural networks learn the same thing? In a new paper (<a href="https://t.co/2p8OM2jjmQ">https://t.co/2p8OM2jjmQ</a>) with <a href="https://twitter.com/maithra_raghu?ref_src=twsrc%5Etfw">@maithra_raghu</a> and <a href="https://twitter.com/skornblith?ref_src=twsrc%5Etfw">@skornblith</a> we study how width and depth affect learned representations within and across models trained on CIFAR and ImageNet. 1/6</p>&mdash; Thao Nguyen (@thao_nguyen26) <a href="https://twitter.com/thao_nguyen26/status/1322285065322196992?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Ray-marching Thurston geometries

Rémi Coulon, Elisabetta A. Matsumoto, Henry Segerman, Steve J. Trettel

- retweets: 156, favorites: 117 (10/31/2020 09:40:19)

- links: [abs](https://arxiv.org/abs/2010.15801) | [pdf](https://arxiv.org/pdf/2010.15801)
- [math.GT](https://arxiv.org/list/math.GT/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [math.DG](https://arxiv.org/list/math.DG/recent)

We describe algorithms that produce accurate real-time interactive in-space views of the eight Thurston geometries using ray-marching. We give a theoretical framework for our algorithms, independent of the geometry involved. In addition to scenes within a geometry $X$, we also consider scenes within quotient manifolds and orbifolds $X / \Gamma$. We adapt the Phong lighting model to non-euclidean geometries. The most difficult part of this is the calculation of light intensity, which relates to the area density of geodesic spheres. We also give extensive practical details for each geometry.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Another postcard from our &#39;Raymarching Thurston Geometries&#39; project <a href="https://t.co/VlVNwXFLAL">https://t.co/VlVNwXFLAL</a>. This time, we are in S²⨉E. With <a href="https://twitter.com/LaMiReMiMath?ref_src=twsrc%5Etfw">@LaMiReMiMath</a>, <a href="https://twitter.com/Sabetta_?ref_src=twsrc%5Etfw">@Sabetta_</a>, and <a href="https://twitter.com/stevejtrettel?ref_src=twsrc%5Etfw">@stevejtrettel</a>. <a href="https://t.co/Yi8NQqdcXw">pic.twitter.com/Yi8NQqdcXw</a></p>&mdash; Henry Segerman (@henryseg) <a href="https://twitter.com/henryseg/status/1322265760723591169?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper &#39;Raymarching Thurston Geometries&#39; is now available on the ArXiv! <a href="https://t.co/6P04ANRPw1">https://t.co/6P04ANRPw1</a><br>Here&#39;s a postcard from Sol geometry 😀<br>With <a href="https://twitter.com/LaMiReMiMath?ref_src=twsrc%5Etfw">@LaMiReMiMath</a> <a href="https://twitter.com/Sabetta_?ref_src=twsrc%5Etfw">@Sabetta_</a> <a href="https://twitter.com/henryseg?ref_src=twsrc%5Etfw">@henryseg</a> <a href="https://t.co/EjLU1dJ2sT">pic.twitter.com/EjLU1dJ2sT</a></p>&mdash; Steve Trettel (@stevejtrettel) <a href="https://twitter.com/stevejtrettel/status/1322194903800033280?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Probabilistic Transformers

Javier R. Movellan

- retweets: 83, favorites: 37 (10/31/2020 09:40:19)

- links: [abs](https://arxiv.org/abs/2010.15583) | [pdf](https://arxiv.org/pdf/2010.15583)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We show that Transformers are Maximum Posterior Probability estimators for Mixtures of Gaussian Models. This brings a probabilistic point of view to Transformers and suggests extensions to other probabilistic cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">nice little note (2 + ε pages) which might be fun to play around with a bit<a href="https://t.co/9JIGcKCoRr">https://t.co/9JIGcKCoRr</a><br>`Probabilistic Transformers&#39;<br>- Javier R. Movellan</p>&mdash; Non-Markovian Sam Power (@sam_power_825) <a href="https://twitter.com/sam_power_825/status/1322110149172092928?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Matern Gaussian Processes on Graphs

Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth, Nicolas Durrande

- retweets: 56, favorites: 42 (10/31/2020 09:40:19)

- links: [abs](https://arxiv.org/abs/2010.15538) | [pdf](https://arxiv.org/pdf/2010.15538)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Gaussian processes are a versatile framework for learning unknown functions in a manner that permits one to utilize prior information about their properties. Although many different Gaussian process models are readily available when the input space is Euclidean, the choice is much more limited for Gaussian processes whose input space is an undirected graph. In this work, we leverage the stochastic partial differential equation characterization of Mat\'{e}rn Gaussian processes - a widely-used model class in the Euclidean setting - to study their analog for undirected graphs. We show that the resulting Gaussian processes inherit various attractive properties of their Euclidean and Riemannian analogs and provide techniques that allow them to be trained using standard methods, such as inducing points. This enables graph Mat\'{e}rn Gaussian processes to be employed in mini-batch and non-conjugate settings, thereby making them more accessible to practitioners and easier to deploy within larger learning frameworks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Matérn Gaussian Processes on Graphs<br><br>We study extensions of the Matérn family of kernels for weighted undirected graphs. Now on arXiv - blog post and tweets soon - stay tuned!<a href="https://t.co/Cf26tnZRpp">https://t.co/Cf26tnZRpp</a><a href="https://twitter.com/mpd37?ref_src=twsrc%5Etfw">@mpd37</a> <a href="https://twitter.com/NicolasDurrande?ref_src=twsrc%5Etfw">@NicolasDurrande</a> <a href="https://t.co/JF7nRoNCMg">pic.twitter.com/JF7nRoNCMg</a></p>&mdash; Alexander Terenin (@avt_im) <a href="https://twitter.com/avt_im/status/1322241317519396867?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Generalized eigen, singular value, and partial least squares  decompositions: The GSVD package

Derek Beaton

- retweets: 30, favorites: 20 (10/31/2020 09:40:19)

- links: [abs](https://arxiv.org/abs/2010.14734) | [pdf](https://arxiv.org/pdf/2010.14734)
- [cs.MS](https://arxiv.org/list/cs.MS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.CO](https://arxiv.org/list/stat.CO/recent) | [stat.ME](https://arxiv.org/list/stat.ME/recent)

The generalized singular value decomposition (GSVD, a.k.a. "SVD triplet", "duality diagram" approach) provides a unified strategy and basis to perform nearly all of the most common multivariate analyses (e.g., principal components, correspondence analysis, multidimensional scaling, canonical correlation, partial least squares). Though the GSVD is ubiquitous, powerful, and flexible, it has very few implementations. Here I introduce the GSVD package for R. The general goal of GSVD is to provide a small set of accessible functions to perform the GSVD and two other related decompositions (generalized eigenvalue decomposition, generalized partial least squares-singular value decomposition). Furthermore, GSVD helps provide a more unified conceptual approach and nomenclature to many techniques. I first introduce the concept of the GSVD, followed by a formal definition of the generalized decompositions. Next I provide some key decisions made during development, and then a number of examples of how to use GSVD to implement various statistical techniques. These examples also illustrate one of the goals of GSVD: how others can (or should) build analysis packages that depend on GSVD. Finally, I discuss the possible future of GSVD.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/ New preprint up: &quot;Generalized eigen, singular value, and partial least squares decompositions: The GSVD package&quot;<a href="https://t.co/nQH6PvR5H9">https://t.co/nQH6PvR5H9</a><br><br>The generalized SVD can do almost anything you&#39;d like, which is why I developed the GSVD <a href="https://twitter.com/hashtag/rstats?src=hash&amp;ref_src=twsrc%5Etfw">#rstats</a> package. Short thread follows <a href="https://t.co/21WUwlYIGQ">pic.twitter.com/21WUwlYIGQ</a></p>&mdash; Derek Beaton (wears a mask and you should too) (@derek__beaton) <a href="https://twitter.com/derek__beaton/status/1321813022294319104?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. A Helmholtz equation solver using unsupervised learning: Application to  transcranial ultrasound

Antonio Stanziola, Simon R. Arridge, Ben T. Cox, Bradley E. Treeby

- retweets: 25, favorites: 25 (10/31/2020 09:40:20)

- links: [abs](https://arxiv.org/abs/2010.15761) | [pdf](https://arxiv.org/pdf/2010.15761)
- [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent) | [physics.med-ph](https://arxiv.org/list/physics.med-ph/recent)

Transcranial ultrasound therapy is increasingly used for the non-invasive treatment of brain disorders. However, conventional numerical wave solvers are currently too computationally expensive to be used online during treatments to predict the acoustic field passing through the skull (e.g., to account for subject-specific dose and targeting variations). As a step towards real-time predictions, in the current work, a fast iterative solver for the heterogeneous Helmholtz equation in 2D is developed using a fully-learned optimizer. The lightweight network architecture is based on a modified UNet that includes a learned hidden state. The network is trained using a physics-based loss function and a set of idealized sound speed distributions with fully unsupervised training (no knowledge of the true solution is required). The learned optimizer shows excellent performance on the test set, and is capable of generalization well outside the training examples, including to much larger computational domains, and more complex source and sound speed distributions, for example, those derived from x-ray computed tomography images of the skull.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to publish our first pre-print on solving the Helmholtz equation using unsupervised learning, with applications to transcranial ultrasound simulation <a href="https://t.co/lyKpURcPb7">https://t.co/lyKpURcPb7</a>. [1/n] <a href="https://t.co/VFlukb7UvL">pic.twitter.com/VFlukb7UvL</a></p>&mdash; UCL Biomedical Ultrasound Group (@UCL_Ultrasound) <a href="https://twitter.com/UCL_Ultrasound/status/1322112660058624000?ref_src=twsrc%5Etfw">October 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



