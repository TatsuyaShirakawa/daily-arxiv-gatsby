---
title: Hot Papers 2021-01-15
date: 2021-01-17T04:10:03.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-15"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-15"
socialImage: "/media/flying-marine.jpg"

---

# 1. Eating Garlic Prevents COVID-19 Infection: Detecting Misinformation on  the Arabic Content of Twitter

Sarah Alqurashi, Btool Hamoui, Abdulaziz Alashaikh, Ahmad Alhindi, Eisa Alanazi

- retweets: 110, favorites: 57 (01/17/2021 04:10:03)

- links: [abs](https://arxiv.org/abs/2101.05626) | [pdf](https://arxiv.org/pdf/2101.05626)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

The rapid growth of social media content during the current pandemic provides useful tools for disseminating information which has also become a root for misinformation. Therefore, there is an urgent need for fact-checking and effective techniques for detecting misinformation in social media. In this work, we study the misinformation in the Arabic content of Twitter. We construct a large Arabic dataset related to COVID-19 misinformation and gold-annotate the tweets into two categories: misinformation or not. Then, we apply eight different traditional and deep machine learning models, with different features including word embeddings and word frequency. The word embedding models (\textsc{FastText} and word2vec) exploit more than two million Arabic tweets related to COVID-19. Experiments show that optimizing the area under the curve (AUC) improves the models' performance and the Extreme Gradient Boosting (XGBoost) presents the highest accuracy in detecting COVID-19 misinformation online.

<blockquote class="twitter-tweet"><p lang="ar" dir="rtl">ورقتنا الخاصة بإكتشاف معلومات كوڤيد-١٩ المغلوطة في تويتر العربي بإستخدام خوارزميات تعلم الآله (العميق).<br><br>في بداية الأزمة، صرح المدير العام لWHO أننا لانواجه فقط Pandemic ولكن  أيضاً Infodemic في إشارة لمعلومات كثيرة مغلوطة تنتشر بين المجتمعات (العربية). <a href="https://t.co/chRfzR1CCL">https://t.co/chRfzR1CCL</a> <a href="https://t.co/TGRuUD2e3Z">pic.twitter.com/TGRuUD2e3Z</a></p>&mdash; عيسى العنزي (@eisa_ayed) <a href="https://twitter.com/eisa_ayed/status/1350154722351730688?ref_src=twsrc%5Etfw">January 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Practical Face Reconstruction via Differentiable Ray Tracing

Abdallah Dib, Gaurav Bharaj, Junghyun Ahn, Cédric Thébault, Philippe-Henri Gosselin, Marco Romeo, Louis Chevallier

- retweets: 66, favorites: 98 (01/17/2021 04:10:04)

- links: [abs](https://arxiv.org/abs/2101.05356) | [pdf](https://arxiv.org/pdf/2101.05356)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a differentiable ray-tracing based novel face reconstruction approach where scene attributes - 3D geometry, reflectance (diffuse, specular and roughness), pose, camera parameters, and scene illumination - are estimated from unconstrained monocular images. The proposed method models scene illumination via a novel, parameterized virtual light stage, which in-conjunction with differentiable ray-tracing, introduces a coarse-to-fine optimization formulation for face reconstruction. Our method can not only handle unconstrained illumination and self-shadows conditions, but also estimates diffuse and specular albedos. To estimate the face attributes consistently and with practical semantics, a two-stage optimization strategy systematically uses a subset of parametric attributes, where subsequent attribute estimations factor those previously estimated. For example, self-shadows estimated during the first stage, later prevent its baking into the personalized diffuse and specular albedos in the second stage. We show the efficacy of our approach in several real-world scenarios, where face attributes can be estimated even under extreme illumination conditions. Ablation studies, analyses and comparisons against several recent state-of-the-art methods show improved accuracy and versatility of our approach. With consistent face attributes reconstruction, our method leads to several style -- illumination, albedo, self-shadow -- edit and transfer applications, as discussed in the paper.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Practical Face Reconstruction via Differentiable Ray Tracing<br>pdf: <a href="https://t.co/zfFMIFilDb">https://t.co/zfFMIFilDb</a><br>abs: <a href="https://t.co/J4MDXTC3Hy">https://t.co/J4MDXTC3Hy</a> <a href="https://t.co/x9byAg41ej">pic.twitter.com/x9byAg41ej</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1349934657375264770?ref_src=twsrc%5Etfw">January 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Whispered and Lombard Neural Speech Synthesis

Qiong Hu, Tobias Bleisch, Petko Petkov, Tuomo Raitio, Erik Marchi, Varun Lakshminarasimhan

- retweets: 20, favorites: 47 (01/17/2021 04:10:04)

- links: [abs](https://arxiv.org/abs/2101.05313) | [pdf](https://arxiv.org/pdf/2101.05313)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

It is desirable for a text-to-speech system to take into account the environment where synthetic speech is presented, and provide appropriate context-dependent output to the user. In this paper, we present and compare various approaches for generating different speaking styles, namely, normal, Lombard, and whisper speech, using only limited data. The following systems are proposed and assessed: 1) Pre-training and fine-tuning a model for each style. 2) Lombard and whisper speech conversion through a signal processing based approach. 3) Multi-style generation using a single model based on a speaker verification model. Our mean opinion score and AB preference listening tests show that 1) we can generate high quality speech through the pre-training/fine-tuning approach for all speaking styles. 2) Although our speaker verification (SV) model is not explicitly trained to discriminate different speaking styles, and no Lombard and whisper voice is used for pre-training this system, the SV model can be used as a style encoder for generating different style embeddings as input for the Tacotron system. We also show that the resulting synthetic Lombard speech has a significant positive impact on intelligibility gain.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Whispered and Lombard Neural Speech Synthesis<br>pdf: <a href="https://t.co/Qdvb3f21gi">https://t.co/Qdvb3f21gi</a><br>abs: <a href="https://t.co/841FPh9Dto">https://t.co/841FPh9Dto</a><br>project page: <a href="https://t.co/u1sRMrljAW">https://t.co/u1sRMrljAW</a> <a href="https://t.co/Lga4P6MNk7">pic.twitter.com/Lga4P6MNk7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1349926195580776450?ref_src=twsrc%5Etfw">January 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. White-Box Analysis over Machine Learning: Modeling Performance of  Configurable Systems

Miguel Velez, Pooyan Jamshidi, Norbert Siegmund, Sven Apel, Christian Kästner

- retweets: 42, favorites: 24 (01/17/2021 04:10:04)

- links: [abs](https://arxiv.org/abs/2101.05362) | [pdf](https://arxiv.org/pdf/2101.05362)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

Performance-influence models can help stakeholders understand how and where configuration options and their interactions influence the performance of a system. With this understanding, stakeholders can debug performance behavior and make deliberate configuration decisions. Current black-box techniques to build such models combine various sampling and learning strategies, resulting in tradeoffs between measurement effort, accuracy, and interpretability. We present Comprex, a white-box approach to build performance-influence models for configurable systems, combining insights of local measurements, dynamic taint analysis to track options in the implementation, compositionality, and compression of the configuration space, without relying on machine learning to extrapolate incomplete samples. Our evaluation on 4 widely-used, open-source projects demonstrates that Comprex builds similarly accurate performance-influence models to the most accurate and expensive black-box approach, but at a reduced cost and with additional benefits from interpretable and local models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper &quot;White-Box Analysis over Machine Learning: Modeling Performance of Configurable Systems&quot; was accepted to ICSE. In collab with <a href="https://twitter.com/PooyanJamshidi?ref_src=twsrc%5Etfw">@PooyanJamshidi</a>, <a href="https://twitter.com/Norbsen?ref_src=twsrc%5Etfw">@Norbsen</a>, <a href="https://twitter.com/SvenApel?ref_src=twsrc%5Etfw">@SvenApel</a>, <a href="https://twitter.com/p0nk?ref_src=twsrc%5Etfw">@p0nk</a>. Pre-print: <a href="https://t.co/rKj1iLcarT">https://t.co/rKj1iLcarT</a> <a href="https://twitter.com/hashtag/icse21?src=hash&amp;ref_src=twsrc%5Etfw">#icse21</a></p>&mdash; Miguel Velez (@mvelezce) <a href="https://twitter.com/mvelezce/status/1350072048777834496?ref_src=twsrc%5Etfw">January 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. GAN Inversion: A Survey

Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, Ming-Hsuan Yang

- retweets: 15, favorites: 40 (01/17/2021 04:10:04)

- links: [abs](https://arxiv.org/abs/2101.05278) | [pdf](https://arxiv.org/pdf/2101.05278)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

GAN inversion aims to invert a given image back into the latent space of a pretrained GAN model, for the image to be faithfully reconstructed from the inverted code by the generator. As an emerging technique to bridge the real and fake image domains, GAN inversion plays an essential role in enabling the pretrained GAN models such as StyleGAN and BigGAN to be used for real image editing applications. Meanwhile, GAN inversion also provides insights on the interpretation of GAN's latent space and how the realistic images can be generated. In this paper, we provide an overview of GAN inversion with a focus on its recent algorithms and applications. We cover important techniques of GAN inversion and their applications to image restoration and image manipulation. We further elaborate on some trends and challenges for future directions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GAN Inversion: A Survey. <a href="https://t.co/GYhJtownR4">https://t.co/GYhJtownR4</a> <a href="https://t.co/3xQb4ztWoy">pic.twitter.com/3xQb4ztWoy</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1350165276982083584?ref_src=twsrc%5Etfw">January 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. No-go Theorem for Acceleration in the Hyperbolic Plane

Linus Hamilton, Ankur Moitra

- retweets: 30, favorites: 24 (01/17/2021 04:10:04)

- links: [abs](https://arxiv.org/abs/2101.05657) | [pdf](https://arxiv.org/pdf/2101.05657)
- [math.OC](https://arxiv.org/list/math.OC/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In recent years there has been significant effort to adapt the key tools and ideas in convex optimization to the Riemannian setting. One key challenge has remained: Is there a Nesterov-like accelerated gradient method for geodesically convex functions on a Riemannian manifold? Recent work has given partial answers and the hope was that this ought to be possible.   Here we dash these hopes. We prove that in a noisy setting, there is no analogue of accelerated gradient descent for geodesically convex functions on the hyperbolic plane. Our results apply even when the noise is exponentially small. The key intuition behind our proof is short and simple: In negatively curved spaces, the volume of a ball grows so fast that information about the past gradients is not useful in the future.




# 7. Signal Processing on Higher-Order Networks: Livin' on the Edge ... and  Beyond

Michael T. Schaub, Yu Zhu, Jean-Baptiste Seby, T. Mitchell Roddenberry, Santiago Segarra

- retweets: 26, favorites: 25 (01/17/2021 04:10:04)

- links: [abs](https://arxiv.org/abs/2101.05510) | [pdf](https://arxiv.org/pdf/2101.05510)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

This tutorial paper presents a didactic treatment of the emerging topic of signal processing on higher-order networks. Drawing analogies from discrete and graph signal processing, we introduce the building blocks for processing data on simplicial complexes and hypergraphs, two common abstractions of higher-order networks that can incorporate polyadic relationships.We provide basic introductions to simplicial complexes and hypergraphs, making special emphasis on the concepts needed for processing signals on them. Leveraging these concepts, we discuss Fourier analysis, signal denoising, signal interpolation, node embeddings, and non-linear processing through neural networks in these two representations of polyadic relational structures. In the context of simplicial complexes, we specifically focus on signal processing using the Hodge Laplacian matrix, a multi-relational operator that leverages the special structure of simplicial complexes and generalizes desirable properties of the Laplacian matrix in graph signal processing. For hypergraphs, we present both matrix and tensor representations, and discuss the trade-offs in adopting one or the other. We also highlight limitations and potential research avenues, both to inform practitioners and to motivate the contribution of new researchers to the area.



