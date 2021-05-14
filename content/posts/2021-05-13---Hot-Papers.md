---
title: Hot Papers 2021-05-13
date: 2021-05-14T10:14:57.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-13"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-13"
socialImage: "/media/flying-marine.jpg"

---

# 1. Segmenter: Transformer for Semantic Segmentation

Robin Strudel, Ricardo Garcia, Ivan Laptev, Cordelia Schmid

- retweets: 2497, favorites: 248 (05/14/2021 10:14:57)

- links: [abs](https://arxiv.org/abs/2105.05633) | [pdf](https://arxiv.org/pdf/2105.05633)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Image segmentation is often ambiguous at the level of individual image patches and requires contextual information to reach label consensus. In this paper we introduce Segmenter, a transformer model for semantic segmentation. In contrast to convolution based approaches, our approach allows to model global context already at the first layer and throughout the network. We build on the recent Vision Transformer (ViT) and extend it to semantic segmentation. To do so, we rely on the output embeddings corresponding to image patches and obtain class labels from these embeddings with a point-wise linear decoder or a mask transformer decoder. We leverage models pre-trained for image classification and show that we can fine-tune them on moderate sized datasets available for semantic segmentation. The linear decoder allows to obtain excellent results already, but the performance can be further improved by a mask transformer generating class masks. We conduct an extensive ablation study to show the impact of the different parameters, in particular the performance is better for large models and small patch sizes. Segmenter attains excellent results for semantic segmentation. It outperforms the state of the art on the challenging ADE20K dataset and performs on-par on Pascal Context and Cityscapes.

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">Segmenter: Transformer for Semantic Segmentation<br>pdf: <a href="https://t.co/HGQVRNuMLC">https://t.co/HGQVRNuMLC</a><br>abs: <a href="https://t.co/xUjh7rjBKT">https://t.co/xUjh7rjBKT</a> <a href="https://t.co/O6vpNtixz7">pic.twitter.com/O6vpNtixz7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392647735023706117?ref_src=twsrc%5Etfw">May 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. When Does Contrastive Visual Representation Learning Work?

Elijah Cole, Xuan Yang, Kimberly Wilber, Oisin Mac Aodha, Serge Belongie

- retweets: 487, favorites: 142 (05/14/2021 10:14:57)

- links: [abs](https://arxiv.org/abs/2105.05837) | [pdf](https://arxiv.org/pdf/2105.05837)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent self-supervised representation learning techniques have largely closed the gap between supervised and unsupervised learning on ImageNet classification. While the particulars of pretraining on ImageNet are now relatively well understood, the field still lacks widely accepted best practices for replicating this success on other datasets. As a first step in this direction, we study contrastive self-supervised learning on four diverse large-scale datasets. By looking through the lenses of data quantity, data domain, data quality, and task granularity, we provide new insights into the necessary conditions for successful self-supervised learning. Our key findings include observations such as: (i) the benefit of additional pretraining data beyond 500k images is modest, (ii) adding pretraining images from another domain does not lead to more general representations, (iii) corrupted pretraining images have a disparate impact on supervised and self-supervised pretraining, and (iv) contrastive learning lags far behind supervised learning on fine-grained visual classification tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">When Does Contrastive Visual Representation Learning Work?<br>pdf: <a href="https://t.co/pw83mXu7Sn">https://t.co/pw83mXu7Sn</a><br>abs: <a href="https://t.co/tSHW6h8hVs">https://t.co/tSHW6h8hVs</a><br><br>benefit of additional pretraining data beyond<br>500k images is modest, adding pretraining images from<br>another domain does not lead to more general representations <a href="https://t.co/XVHBNvI1bv">pic.twitter.com/XVHBNvI1bv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392649447306268674?ref_src=twsrc%5Etfw">May 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped  scene text

Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, Tal Hassner

- retweets: 182, favorites: 65 (05/14/2021 10:14:57)

- links: [abs](https://arxiv.org/abs/2105.05486) | [pdf](https://arxiv.org/pdf/2105.05486)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

A crucial component for the scene text based reasoning required for TextVQA and TextCaps datasets involve detecting and recognizing text present in the images using an optical character recognition (OCR) system. The current systems are crippled by the unavailability of ground truth text annotations for these datasets as well as lack of scene text detection and recognition datasets on real images disallowing the progress in the field of OCR and evaluation of scene text based reasoning in isolation from OCR systems. In this work, we propose TextOCR, an arbitrary-shaped scene text detection and recognition with 900k annotated words collected on real images from TextVQA dataset. We show that current state-of-the-art text-recognition (OCR) models fail to perform well on TextOCR and that training on TextOCR helps achieve state-of-the-art performance on multiple other OCR datasets as well. We use a TextOCR trained OCR model to create PixelM4C model which can do scene text based reasoning on an image in an end-to-end fashion, allowing us to revisit several design choices to achieve new state-of-the-art performance on TextVQA dataset.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text<br>pdf: <a href="https://t.co/TDw1QSlSzw">https://t.co/TDw1QSlSzw</a><br>abs: <a href="https://t.co/rWRCRSlLKd">https://t.co/rWRCRSlLKd</a><br>project page: <a href="https://t.co/zX0Wf1ucH6">https://t.co/zX0Wf1ucH6</a><br><br>arbitrary-shaped scene text detection and recognition with 900k annotated words <a href="https://t.co/Vn8ad0EKSx">pic.twitter.com/Vn8ad0EKSx</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392651137858293765?ref_src=twsrc%5Etfw">May 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. An Introduction to Algorithmic Fairness

Hilde J.P. Weerts

- retweets: 196, favorites: 50 (05/14/2021 10:14:58)

- links: [abs](https://arxiv.org/abs/2105.05595) | [pdf](https://arxiv.org/pdf/2105.05595)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

In recent years, there has been an increasing awareness of both the public and scientific community that algorithmic systems can reproduce, amplify, or even introduce unfairness in our societies. These lecture notes provide an introduction to some of the core concepts in algorithmic fairness research. We list different types of fairness-related harms, explain two main notions of algorithmic fairness, and map the biases that underlie these harms upon the machine learning development process.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">After some initial nice responses, I&#39;ve decided to put the introduction chapter of my lecture notes on algorithmic fairness on arXiv: <a href="https://t.co/3UTqNOWUcw">https://t.co/3UTqNOWUcw</a>. I hope these will make it easier for folks who are interested to learn more about fairness but don&#39;t know where to start!</p>&mdash; Hilde Weerts (@hildeweerts) <a href="https://twitter.com/hildeweerts/status/1392768528873365505?ref_src=twsrc%5Etfw">May 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Fairness and Discrimination in Information Access Systems

Michael D. Ekstrand, Anubrata Das, Robin Burke, Fernando Diaz

- retweets: 156, favorites: 46 (05/14/2021 10:14:58)

- links: [abs](https://arxiv.org/abs/2105.05779) | [pdf](https://arxiv.org/pdf/2105.05779)
- [cs.IR](https://arxiv.org/list/cs.IR/recent)

Recommendation, information retrieval, and other information access systems pose unique challenges for investigating and applying the fairness and non-discrimination concepts that have been developed for studying other machine learning systems. While fair information access shares many commonalities with fair classification, the multistakeholder nature of information access applications, the rank-based problem setting, the centrality of personalization in many cases, and the role of user response complicate the problem of identifying precisely what types and operationalizations of fairness may be relevant, let alone measuring or promoting them.   In this monograph, we present a taxonomy of the various dimensions of fair information access and survey the literature to date on this new and rapidly-growing topic. We preface this with brief introductions to information access and algorithmic fairness, to facilitate use of this work by scholars with experience in one (or neither) of these fields who wish to learn about their intersection. We conclude with several open problems in fair information access, along with some suggestions for how to approach research in this space.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚨 new preprint alert 🚨<br><br>A survey and systematization of fairness in information retrieval and <a href="https://twitter.com/hashtag/recsys?src=hash&amp;ref_src=twsrc%5Etfw">#recsys</a>, with <a href="https://twitter.com/d_anubrata?ref_src=twsrc%5Etfw">@d_anubrata</a>, <a href="https://twitter.com/rburke2233?ref_src=twsrc%5Etfw">@rburke2233</a>, and <a href="https://twitter.com/841io?ref_src=twsrc%5Etfw">@841io</a>. Share and enjoy, and please send feedback! <a href="https://t.co/B0at5cTIMl">https://t.co/B0at5cTIMl</a></p>&mdash; Michael Ekstrand (@mdekstrand) <a href="https://twitter.com/mdekstrand/status/1392650256127455235?ref_src=twsrc%5Etfw">May 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Operation-wise Attention Network for Tampering Localization Fusion

Polychronis Charitidis, Giorgos Kordopatis-Zilos, Symeon Papadopoulos, Ioannis Kompatsiaris

- retweets: 56, favorites: 5 (05/14/2021 10:14:58)

- links: [abs](https://arxiv.org/abs/2105.05515) | [pdf](https://arxiv.org/pdf/2105.05515)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this work, we present a deep learning-based approach for image tampering localization fusion. This approach is designed to combine the outcomes of multiple image forensics algorithms and provides a fused tampering localization map, which requires no expert knowledge and is easier to interpret by end users. Our fusion framework includes a set of five individual tampering localization methods for splicing localization on JPEG images. The proposed deep learning fusion model is an adapted architecture, initially proposed for the image restoration task, that performs multiple operations in parallel, weighted by an attention mechanism to enable the selection of proper operations depending on the input signals. This weighting process can be very beneficial for cases where the input signal is very diverse, as in our case where the output signals of multiple image forensics algorithms are combined. Evaluation in three publicly available forensics datasets demonstrates that the performance of the proposed approach is competitive, outperforming the individual forensics techniques as well as another recently proposed fusion framework in the majority of cases.




# 7. GANs for Medical Image Synthesis: An Empirical Study

Youssef Skandarani, Pierre-Marc Jodoin, Alain Lalande

- retweets: 30, favorites: 29 (05/14/2021 10:14:58)

- links: [abs](https://arxiv.org/abs/2105.05318) | [pdf](https://arxiv.org/pdf/2105.05318)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Generative Adversarial Networks (GANs) have become increasingly powerful, generating mind-blowing photorealistic images that mimic the content of datasets they were trained to replicate. One recurrent theme in medical imaging is whether GANs can also be effective at generating workable medical data as they are for generating realistic RGB images. In this paper, we perform a multi-GAN and multi-application study to gauge the benefits of GANs in medical imaging. We tested various GAN architectures from basic DCGAN to more sophisticated style-based GANs on three medical imaging modalities and organs namely : cardiac cine-MRI, liver CT and RGB retina images. GANs were trained on well-known and widely utilized datasets from which their FID score were computed to measure the visual acuity of their generated images. We further tested their usefulness by measuring the segmentation accuracy of a U-Net trained on these generated images.   Results reveal that GANs are far from being equal as some are ill-suited for medical imaging applications while others are much better off. The top-performing GANs are capable of generating realistic-looking medical images by FID standards that can fool trained experts in a visual Turing test and comply to some metrics. However, segmentation results suggests that no GAN is capable of reproducing the full richness of a medical datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GANs for Medical Image Synthesis: An Empirical Study<br>Youssef Skandarani, <a href="https://twitter.com/PMJodoin?ref_src=twsrc%5Etfw">@PMJodoin</a>, Alain Lalande<a href="https://t.co/47eV12lpQ2">https://t.co/47eV12lpQ2</a><br><br>tl;dr: <br>- FID score != downstream metrics in the medical domain<br>- adding GAN-generated data to the train set helps only a little if any. <a href="https://t.co/Fi3SVEmgG1">pic.twitter.com/Fi3SVEmgG1</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1392748065145114626?ref_src=twsrc%5Etfw">May 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Collaborative Regression of Expressive Bodies using Moderation

Yao Feng, Vasileios Choutas, Timo Bolkart, Dimitrios Tzionas, Michael J. Black

- retweets: 12, favorites: 43 (05/14/2021 10:14:58)

- links: [abs](https://arxiv.org/abs/2105.05301) | [pdf](https://arxiv.org/pdf/2105.05301)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recovering expressive humans from images is essential for understanding human behavior. Methods that estimate 3D bodies, faces, or hands have progressed significantly, yet separately. Face methods recover accurate 3D shape and geometric details, but need a tight crop and struggle with extreme views and low resolution. Whole-body methods are robust to a wide range of poses and resolutions, but provide only a rough 3D face shape without details like wrinkles. To get the best of both worlds, we introduce PIXIE, which produces animatable, whole-body 3D avatars from a single image, with realistic facial detail. To get accurate whole bodies, PIXIE uses two key observations. First, body parts are correlated, but existing work combines independent estimates from body, face, and hand experts, by trusting them equally. PIXIE introduces a novel moderator that merges the features of the experts, weighted by their confidence. Uniquely, part experts can contribute to the whole, using SMPL-X's shared shape space across all body parts. Second, human shape is highly correlated with gender, but existing work ignores this. We label training images as male, female, or non-binary, and train PIXIE to infer "gendered" 3D body shapes with a novel shape loss. In addition to 3D body pose and shape parameters, PIXIE estimates expression, illumination, albedo and 3D surface displacements for the face. Quantitative and qualitative evaluation shows that PIXIE estimates 3D humans with a more accurate whole-body shape and detailed face shape than the state of the art. Our models and code are available for research at https://pixie.is.tue.mpg.de.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Collaborative Regression of Expressive Bodies using Moderation<br>pdf: <a href="https://t.co/rriYtSrXC2">https://t.co/rriYtSrXC2</a><br>abs: <a href="https://t.co/XtqaXJloGo">https://t.co/XtqaXJloGo</a><br>project page: <a href="https://t.co/u86fe8Hcy9">https://t.co/u86fe8Hcy9</a><br><br>whole-body reconstruction method that recovers an animatable 3D avatar with a detailed face from a single RGB image <a href="https://t.co/UXf1oJRoRF">pic.twitter.com/UXf1oJRoRF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392653839061430272?ref_src=twsrc%5Etfw">May 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



