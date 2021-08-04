---
title: Hot Papers 2021-08-03
date: 2021-08-04T09:49:00.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-03"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-03"
socialImage: "/media/flying-marine.jpg"

---

# 1. StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators

Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, Daniel Cohen-Or

- retweets: 6396, favorites: 301 (08/04/2021 09:49:00)

- links: [abs](https://arxiv.org/abs/2108.00946) | [pdf](https://arxiv.org/pdf/2108.00946)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Can a generative model be trained to produce images from a specific domain, guided by a text prompt only, without seeing any image? In other words: can an image generator be trained blindly? Leveraging the semantic power of large scale Contrastive-Language-Image-Pre-training (CLIP) models, we present a text-driven method that allows shifting a generative model to new domains, without having to collect even a single image from those domains. We show that through natural language prompts and a few minutes of training, our method can adapt a generator across a multitude of domains characterized by diverse styles and shapes. Notably, many of these modifications would be difficult or outright impossible to reach with existing methods. We conduct an extensive set of experiments and comparisons across a wide range of domains. These demonstrate the effectiveness of our approach and show that our shifted models maintain the latent-space properties that make generative models appealing for downstream tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators<br>pdf: <a href="https://t.co/yeg2wLWNCw">https://t.co/yeg2wLWNCw</a><br>abs: <a href="https://t.co/CLyjRs0oWE">https://t.co/CLyjRs0oWE</a><br>project page: <a href="https://t.co/1biePT4Bgp">https://t.co/1biePT4Bgp</a> <a href="https://t.co/WBeVIKJGTQ">pic.twitter.com/WBeVIKJGTQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422383627384594434?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Sequoia: A Software Framework to Unify Continual Learning Research

Fabrice Normandin, Florian Golemo, Oleksiy Ostapenko, Pau Rodriguez, Matthew D Riemer, Julio Hurtado, Khimya Khetarpal1, Dominic Zhao, Ryan Lindeborg, Thimothée Lesort, Laurent Charlin, Irina Rish, Massimo Caccia

- retweets: 420, favorites: 72 (08/04/2021 09:49:01)

- links: [abs](https://arxiv.org/abs/2108.01005) | [pdf](https://arxiv.org/pdf/2108.01005)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

The field of Continual Learning (CL) seeks to develop algorithms that accumulate knowledge and skills over time through interaction with non-stationary environments and data distributions. Measuring progress in CL can be difficult because a plethora of evaluation procedures (\emph{settings}) and algorithmic solutions (\emph{methods}) have emerged, each with their own potentially disjoint set of assumptions about the CL problem. In this work, we view each setting as a set of \emph{assumptions}. We then create a tree-shaped hierarchy of the research settings in CL, in which more general settings become the parents of those with more restrictive assumptions. This makes it possible to use inheritance to share and reuse research, as developing a method for a given setting also makes it directly applicable onto any of its children. We instantiate this idea as a publicly available software framework called \emph{Sequoia}, which features a variety of settings from both the Continual Supervised Learning (CSL) and Continual Reinforcement Learning (CRL) domains. Sequoia also includes a growing suite of methods which are easy to extend and customize, in addition to more specialized methods from third-party libraries. We hope that this new paradigm and its first implementation can serve as a foundation for the unification and acceleration of research in CL. You can help us grow the tree by visiting \url{www.github.com/lebrice/Sequoia}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sequoia: A Software Framework to Unify Continual Learning Research<br>pdf: <a href="https://t.co/CNhuTaEdAn">https://t.co/CNhuTaEdAn</a><br>abs: <a href="https://t.co/Myiep9F13h">https://t.co/Myiep9F13h</a><br>github: <a href="https://t.co/9S6plpPKw8">https://t.co/9S6plpPKw8</a> <a href="https://t.co/is7Y83nj77">pic.twitter.com/is7Y83nj77</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422372005186314241?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Large-scale quantum machine learning

Tobias Haug, Chris N. Self, M. S. Kim

- retweets: 291, favorites: 55 (08/04/2021 09:49:01)

- links: [abs](https://arxiv.org/abs/2108.01039) | [pdf](https://arxiv.org/pdf/2108.01039)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Quantum computers promise to enhance machine learning for practical applications. Quantum machine learning for real-world data has to handle extensive amounts of high-dimensional data. However, conventional methods for measuring quantum kernels are impractical for large datasets as they scale with the square of the dataset size. Here, we measure quantum kernels using randomized measurements to gain a quadratic speedup in computation time and quickly process large datasets. Further, we efficiently encode high-dimensional data into quantum computers with the number of features scaling linearly with the circuit depth. The encoding is characterized by the quantum Fisher information metric and is related to the radial basis function kernel. We demonstrate the advantages and speedups of our methods by classifying images with the IBM quantum computer. Our approach is exceptionally robust to noise via a complementary error mitigation scheme. Using currently available quantum computers, the MNIST database can be processed within 220 hours instead of 10 years which opens up industrial applications of quantum machine learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">On arXiv: Large-scale quantum machine learning with randomized measurements.<br>The IBM quantum computer classifies images resilient to noise.<br>The MNIST database could be processed within 220 hours instead of 10 years. (1/5)<a href="https://t.co/8O6ZFWSB6Z">https://t.co/8O6ZFWSB6Z</a><a href="https://t.co/SyclzUqfm2">https://t.co/SyclzUqfm2</a> <a href="https://t.co/eDIR7M218x">pic.twitter.com/eDIR7M218x</a></p>&mdash; Tobias Haug (@TobiasHaug_Q) <a href="https://twitter.com/TobiasHaug_Q/status/1422517128717127681?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. On The State of Data In Computer Vision: Human Annotations Remain  Indispensable for Developing Deep Learning Models

Zeyad Emam, Andrew Kondrich, Sasha Harrison, Felix Lau, Yushi Wang, Aerin Kim, Elliot Branson

- retweets: 257, favorites: 75 (08/04/2021 09:49:01)

- links: [abs](https://arxiv.org/abs/2108.00114) | [pdf](https://arxiv.org/pdf/2108.00114)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

High-quality labeled datasets play a crucial role in fueling the development of machine learning (ML), and in particular the development of deep learning (DL). However, since the emergence of the ImageNet dataset and the AlexNet model in 2012, the size of new open-source labeled vision datasets has remained roughly constant. Consequently, only a minority of publications in the computer vision community tackle supervised learning on datasets that are orders of magnitude larger than Imagenet. In this paper, we survey computer vision research domains that study the effects of such large datasets on model performance across different vision tasks. We summarize the community's current understanding of those effects, and highlight some open questions related to training with massive datasets. In particular, we tackle: (a) The largest datasets currently used in computer vision research and the interesting takeaways from training on such datasets; (b) The effectiveness of pre-training on large datasets; (c) Recent advancements and hurdles facing synthetic datasets; (d) An overview of double descent and sample non-monotonicity phenomena; and finally, (e) A brief discussion of lifelong/continual learning and how it fares compared to learning from huge labeled datasets in an offline setting. Overall, our findings are that research on optimization for deep learning focuses on perfecting the training routine and thus making DL models less data hungry, while research on synthetic datasets aims to offset the cost of data labeling. However, for the time being, acquiring non-synthetic labeled data remains indispensable to boost performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">On The State of Data In Computer Vision: Human Annotations Remain Indispensable for Developing Deep Learning Models<br>pdf: <a href="https://t.co/HTnj3AITcY">https://t.co/HTnj3AITcY</a><br>abs: <a href="https://t.co/n825wZar6Y">https://t.co/n825wZar6Y</a> <a href="https://t.co/p8on9bv8Qc">pic.twitter.com/p8on9bv8Qc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422409479207936000?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Cross-cultural Mood Perception in Pop Songs and its Alignment with Mood  Detection Algorithms

Harin Lee, Frank Hoeger, Marc Schoenwiesner, Minsu Park, Nori Jacoby

- retweets: 156, favorites: 53 (08/04/2021 09:49:01)

- links: [abs](https://arxiv.org/abs/2108.00768) | [pdf](https://arxiv.org/pdf/2108.00768)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Do people from different cultural backgrounds perceive the mood in music the same way? How closely do human ratings across different cultures approximate automatic mood detection algorithms that are often trained on corpora of predominantly Western popular music? Analyzing 166 participants responses from Brazil, South Korea, and the US, we examined the similarity between the ratings of nine categories of perceived moods in music and estimated their alignment with four popular mood detection algorithms. We created a dataset of 360 recent pop songs drawn from major music charts of the countries and constructed semantically identical mood descriptors across English, Korean, and Portuguese languages. Multiple participants from the three countries rated their familiarity, preference, and perceived moods for a given song. Ratings were highly similar within and across cultures for basic mood attributes such as sad, cheerful, and energetic. However, we found significant cross-cultural differences for more complex characteristics such as dreamy and love. To our surprise, the results of mood detection algorithms were uniformly correlated across human ratings from all three countries and did not show a detectable bias towards any particular culture. Our study thus suggests that the mood detection algorithms can be considered as an objective measure at least within the popular music context.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Delighted to share the first chunk of my PhD work on cross-cultural mood perception in pop songs and its alignment with current mood detection algorithms. <br><br>Presentation: Nov, 2021 <a href="https://twitter.com/ismir2021?ref_src=twsrc%5Etfw">@ismir2021</a> <br>Preprint: <a href="https://t.co/70bTM6QtFT">https://t.co/70bTM6QtFT</a><br>Open data: <a href="https://t.co/DGrQyF9ewj">https://t.co/DGrQyF9ewj</a><br>Thread ↓ <a href="https://t.co/KhJJ6Zvl9P">pic.twitter.com/KhJJ6Zvl9P</a></p>&mdash; HARIN (@TweetHarin) <a href="https://twitter.com/TweetHarin/status/1422464756900900901?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. CrossFormer: A Versatile Vision Transformer Based on Cross-scale  Attention

Wenxiao Wang, Lu Yao, Long Chen, Deng Cai, Xiaofei He, Wei Liu

- retweets: 144, favorites: 51 (08/04/2021 09:49:01)

- links: [abs](https://arxiv.org/abs/2108.00154) | [pdf](https://arxiv.org/pdf/2108.00154)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers have made much progress in dealing with visual tasks. However, existing vision transformers still do not possess an ability that is important to visual input: building the attention among features of different scales. The reasons for this problem are two-fold: (1) Input embeddings of each layer are equal-scale without cross-scale features; (2) Some vision transformers sacrifice the small-scale features of embeddings to lower the cost of the self-attention module. To make up this defect, we propose Cross-scale Embedding Layer (CEL) and Long Short Distance Attention (LSDA). In particular, CEL blends each embedding with multiple patches of different scales, providing the model with cross-scale embeddings. LSDA splits the self-attention module into a short-distance and long-distance one, also lowering the cost but keeping both small-scale and large-scale features in embeddings. Through these two designs, we achieve cross-scale attention. Besides, we propose dynamic position bias for vision transformers to make the popular relative position bias apply to variable-sized images. Based on these proposed modules, we construct our vision architecture called CrossFormer. Experiments show that CrossFormer outperforms other transformers on several representative visual tasks, especially object detection and segmentation. The code has been released: https://github.com/cheerss/CrossFormer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention<br>paper: <a href="https://t.co/39cUeteOPt">https://t.co/39cUeteOPt</a><br><br>improvements in detection and segmentation, which indicates that cross-scale embedding and LSDA are particularly essential for dense prediction vision tasks <a href="https://t.co/nht53utDK3">pic.twitter.com/nht53utDK3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422364388913209345?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. SDEdit: Image Synthesis and Editing with Stochastic Differential  Equations

Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon

- retweets: 112, favorites: 78 (08/04/2021 09:49:01)

- links: [abs](https://arxiv.org/abs/2108.01073) | [pdf](https://arxiv.org/pdf/2108.01073)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We introduce a new image editing and synthesis framework, Stochastic Differential Editing (SDEdit), based on a recent generative model using stochastic differential equations (SDEs). Given an input image with user edits (e.g., hand-drawn color strokes), we first add noise to the input according to an SDE, and subsequently denoise it by simulating the reverse SDE to gradually increase its likelihood under the prior. Our method does not require task-specific loss function designs, which are critical components for recent image editing methods based on GAN inversion. Compared to conditional GANs, we do not need to collect new datasets of original and edited images for new applications. Therefore, our method can quickly adapt to various editing tasks at test time without re-training models. Our approach achieves strong performance on a wide range of applications, including image synthesis and editing guided by stroke paintings and image compositing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Image Synthesis and Editing with Stochastic Differential Equations<br>paper: <a href="https://t.co/vXrkU8fe0f">https://t.co/vXrkU8fe0f</a><br><br>does not require task-specific optimization algorithms for reconstructing inputs, suitable for datasets or tasks where GAN inversion losses are hard to design or optimize <a href="https://t.co/YuAoAJ7Gey">pic.twitter.com/YuAoAJ7Gey</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422370391104819206?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. S$^2$-MLPv2: Improved Spatial-Shift MLP Architecture for Vision

Tan Yu, Xu Li, Yunfeng Cai, Mingming Sun, Ping Li

- retweets: 110, favorites: 63 (08/04/2021 09:49:02)

- links: [abs](https://arxiv.org/abs/2108.01072) | [pdf](https://arxiv.org/pdf/2108.01072)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, MLP-based vision backbones emerge. MLP-based vision architectures with less inductive bias achieve competitive performance in image recognition compared with CNNs and vision Transformers. Among them, spatial-shift MLP (S$^2$-MLP), adopting the straightforward spatial-shift operation, achieves better performance than the pioneering works including MLP-mixer and ResMLP. More recently, using smaller patches with a pyramid structure, Vision Permutator (ViP) and Global Filter Network (GFNet) achieve better performance than S$^2$-MLP.   In this paper, we improve the S$^2$-MLP vision backbone. We expand the feature map along the channel dimension and split the expanded feature map into several parts. We conduct different spatial-shift operations on split parts.   Meanwhile, we exploit the split-attention operation to fuse these split parts. Moreover, like the counterparts, we adopt smaller-scale patches and use a pyramid structure for boosting the image recognition accuracy. We term the improved spatial-shift MLP vision backbone as S$^2$-MLPv2. Using 55M parameters, our medium-scale model, S$^2$-MLPv2-Medium achieves an $83.6\%$ top-1 accuracy on the ImageNet-1K benchmark using $224\times 224$ images without self-attention and external training data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">S2-MLPv2: Improved Spatial-Shift MLP Architecture for Vision<br>paper: <a href="https://t.co/SXOcXSwt5n">https://t.co/SXOcXSwt5n</a><br><br>Using 55M parameters, S2-MLPv2-Medium achieves an 83.6% top-1 accuracy on the ImageNet-1K benchmark using 224 × 224 images without self-attention and external training data <a href="https://t.co/cqTNTvcsfg">pic.twitter.com/cqTNTvcsfg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422361548434165765?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Object-aware Contrastive Learning for Debiased Scene Representation

Sangwoo Mo, Hyunwoo Kang, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin

- retweets: 110, favorites: 54 (08/04/2021 09:49:02)

- links: [abs](https://arxiv.org/abs/2108.00049) | [pdf](https://arxiv.org/pdf/2108.00049)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Contrastive self-supervised learning has shown impressive results in learning visual representations from unlabeled images by enforcing invariance against different data augmentations. However, the learned representations are often contextually biased to the spurious scene correlations of different objects or object and background, which may harm their generalization on the downstream tasks. To tackle the issue, we develop a novel object-aware contrastive learning framework that first (a) localizes objects in a self-supervised manner and then (b) debias scene correlations via appropriate data augmentations considering the inferred object locations. For (a), we propose the contrastive class activation map (ContraCAM), which finds the most discriminative regions (e.g., objects) in the image compared to the other images using the contrastively trained models. We further improve the ContraCAM to detect multiple objects and entire shapes via an iterative refinement procedure. For (b), we introduce two data augmentations based on ContraCAM, object-aware random crop and background mixup, which reduce contextual and background biases during contrastive self-supervised learning, respectively. Our experiments demonstrate the effectiveness of our representation learning framework, particularly when trained under multi-object images or evaluated under the background (and distribution) shifted images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Object-aware Contrastive Learning for Debiased Scene Representation<br>pdf: <a href="https://t.co/pqzHtLpLg4">https://t.co/pqzHtLpLg4</a><br>abs: <a href="https://t.co/rJTM4gjR51">https://t.co/rJTM4gjR51</a><br>github: <a href="https://t.co/57GjY4C3Fo">https://t.co/57GjY4C3Fo</a> <a href="https://t.co/Us8LGzzUmw">pic.twitter.com/Us8LGzzUmw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422410185356742658?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Multi-Head Self-Attention via Vision Transformer for Zero-Shot Learning

Faisal Alamri, Anjan Dutta

- retweets: 56, favorites: 45 (08/04/2021 09:49:02)

- links: [abs](https://arxiv.org/abs/2108.00045) | [pdf](https://arxiv.org/pdf/2108.00045)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Zero-Shot Learning (ZSL) aims to recognise unseen object classes, which are not observed during the training phase. The existing body of works on ZSL mostly relies on pretrained visual features and lacks the explicit attribute localisation mechanism on images. In this work, we propose an attention-based model in the problem settings of ZSL to learn attributes useful for unseen class recognition. Our method uses an attention mechanism adapted from Vision Transformer to capture and learn discriminative attributes by splitting images into small patches. We conduct experiments on three popular ZSL benchmarks (i.e., AWA2, CUB and SUN) and set new state-of-the-art harmonic mean results {on all the three datasets}, which illustrate the effectiveness of our proposed method.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Head Self-Attention via Vision Transformer for Zero-Shot Learning<br>pdf: <a href="https://t.co/jYTRIHc4K1">https://t.co/jYTRIHc4K1</a><br>abs: <a href="https://t.co/YOdzP0H4FS">https://t.co/YOdzP0H4FS</a><br><br>experiments on three popular ZSL benchmarks (i.e., AWA2, CUB and SUN) and set new sota harmonic mean results on all the three datasets <a href="https://t.co/1MQVyekjYL">pic.twitter.com/1MQVyekjYL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422365654750007296?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Soft Calibration Objectives for Neural Networks

Archit Karandikar, Nicholas Cain, Dustin Tran, Balaji Lakshminarayanan, Jonathon Shlens, Michael C. Mozer, Becca Roelofs

- retweets: 56, favorites: 34 (08/04/2021 09:49:02)

- links: [abs](https://arxiv.org/abs/2108.00106) | [pdf](https://arxiv.org/pdf/2108.00106)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Optimal decision making requires that classifiers produce uncertainty estimates consistent with their empirical accuracy. However, deep neural networks are often under- or over-confident in their predictions. Consequently, methods have been developed to improve the calibration of their predictive uncertainty both during training and post-hoc. In this work, we propose differentiable losses to improve calibration based on a soft (continuous) version of the binning operation underlying popular calibration-error estimators. When incorporated into training, these soft calibration losses achieve state-of-the-art single-model ECE across multiple datasets with less than 1% decrease in accuracy. For instance, we observe an 82% reduction in ECE (70% relative to the post-hoc rescaled ECE) in exchange for a 0.7% relative decrease in accuracy relative to the cross entropy baseline on CIFAR-100. When incorporated post-training, the soft-binning-based calibration error objective improves upon temperature scaling, a popular recalibration method. Overall, experiments across losses and datasets demonstrate that using calibration-sensitive procedures yield better uncertainty estimates under dataset shift than the standard practice of using a cross entropy loss and post-hoc recalibration methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Soft Calibration Objectives for Neural Networks<br>pdf: <a href="https://t.co/gWbR1WvlrV">https://t.co/gWbR1WvlrV</a><br>abs: <a href="https://t.co/jW1FiSio3M">https://t.co/jW1FiSio3M</a><br><br>When incorporated into training, these soft calibration losses achieve sota single-model ECE across multiple datasets with less than 1% decrease in accuracy <a href="https://t.co/pe4xg3czUZ">pic.twitter.com/pe4xg3czUZ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1422362211046023168?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Elements of Differential Geometry in Lean: A Report for Mathematicians

Anthony Bordg, Nicolò Cavalleri

- retweets: 49, favorites: 36 (08/04/2021 09:49:02)

- links: [abs](https://arxiv.org/abs/2108.00484) | [pdf](https://arxiv.org/pdf/2108.00484)
- [cs.LO](https://arxiv.org/list/cs.LO/recent) | [math.DG](https://arxiv.org/list/math.DG/recent)

We report on our experience formalizing differential geometry with mathlib, the Lean mathematical library. Our account is geared towards geometers with no knowledge of type theory, but eager to learn more about the formalization of mathematics and maybe curious enough to give Lean a try in the future. To this effect, we stress the possibly surprising difference between the formalization and its pen-and-paper counterpart arising from Lean's treatment of equality. Our three case studies are Lie groups, vector bundles and the Lie algebra of a Lie group.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The long march towards de Rham cohomology in Lean goes on: <a href="https://t.co/uvgfCqtDiC">https://t.co/uvgfCqtDiC</a></p>&mdash; The Xena Project (@XenaProject) <a href="https://twitter.com/XenaProject/status/1422625366577975300?ref_src=twsrc%5Etfw">August 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



