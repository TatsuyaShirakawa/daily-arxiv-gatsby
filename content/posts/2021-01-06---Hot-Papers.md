---
title: Hot Papers 2021-01-06
date: 2021-01-07T10:01:53.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-06"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-06"
socialImage: "/media/flying-marine.jpg"

---

# 1. STaR: Self-supervised Tracking and Reconstruction of Rigid Objects in  Motion with Neural Rendering

Wentao Yuan, Zhaoyang Lv, Tanner Schmidt, Steven Lovegrove

- retweets: 78, favorites: 49 (01/07/2021 10:01:53)

- links: [abs](https://arxiv.org/abs/2101.01602) | [pdf](https://arxiv.org/pdf/2101.01602)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present STaR, a novel method that performs Self-supervised Tracking and Reconstruction of dynamic scenes with rigid motion from multi-view RGB videos without any manual annotation. Recent work has shown that neural networks are surprisingly effective at the task of compressing many views of a scene into a learned function which maps from a viewing ray to an observed radiance value via volume rendering. Unfortunately, these methods lose all their predictive power once any object in the scene has moved. In this work, we explicitly model rigid motion of objects in the context of neural representations of radiance fields. We show that without any additional human specified supervision, we can reconstruct a dynamic scene with a single rigid object in motion by simultaneously decomposing it into its two constituent parts and encoding each with its own neural representation. We achieve this by jointly optimizing the parameters of two neural radiance fields and a set of rigid poses which align the two fields at each frame. On both synthetic and real world datasets, we demonstrate that our method can render photorealistic novel views, where novelty is measured on both spatial and temporal axes. Our factored representation furthermore enables animation of unseen object motion.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">STaR: Self-supervised Tracking and Reconstruction of Rigid Objects in Motion with Neural Rendering<br>pdf: <a href="https://t.co/ihHgsH0zQS">https://t.co/ihHgsH0zQS</a><br>abs: <a href="https://t.co/0VspkFQ1l2">https://t.co/0VspkFQ1l2</a> <a href="https://t.co/t7bs22sD9j">pic.twitter.com/t7bs22sD9j</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1346634293427236864?ref_src=twsrc%5Etfw">January 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. A Survey of Community Detection Approaches: From Statistical Modeling to  Deep Learning

Di Jin, Zhizhi Yu, Pengfei Jiao, Shirui Pan, Philip S. Yu, Weixiong Zhang

- retweets: 24, favorites: 30 (01/07/2021 10:01:54)

- links: [abs](https://arxiv.org/abs/2101.01669) | [pdf](https://arxiv.org/pdf/2101.01669)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

Community detection, a fundamental task for network analysis, aims to partition a network into multiple sub-structures to help reveal their latent functions. Community detection has been extensively studied in and broadly applied to many real-world network problems. Classical approaches to community detection typically utilize probabilistic graphical models and adopt a variety of prior knowledge to infer community structures. As the problems that network methods try to solve and the network data to be analyzed become increasingly more sophisticated, new approaches have also been proposed and developed, particularly those that utilize deep learning and convert networked data into low dimensional representation. Despite all the recent advancement, there is still a lack of insightful understanding of the theoretical and methodological underpinning of community detection, which will be critically important for future development of the area of network analysis. In this paper, we develop and present a unified architecture of network community-finding methods to characterize the state-of-the-art of the field of community detection. Specifically, we provide a comprehensive review of the existing community detection methods and introduce a new taxonomy that divides the existing methods into two categories, namely probabilistic graphical model and deep learning. We then discuss in detail the main idea behind each method in the two categories. Furthermore, to promote future development of community detection, we release several benchmark datasets from several problem domains and highlight their applications to various network analysis tasks. We conclude with discussions of the challenges of the field and suggestions of possible directions for future research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Survey of Community Detection Approaches: From Statistical Modeling to Deep Learning. (arXiv:2101.01669v1 [<a href="https://t.co/lwVVolmoyC">https://t.co/lwVVolmoyC</a>]) <a href="https://t.co/QyfnqM4Qct">https://t.co/QyfnqM4Qct</a></p>&mdash; NetScience (@net_science) <a href="https://twitter.com/net_science/status/1346679731530338305?ref_src=twsrc%5Etfw">January 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. SpotPatch: Parameter-Efficient Transfer Learning for Mobile Object  Detection

Keren Ye, Adriana Kovashka, Mark Sandler, Menglong Zhu, Andrew Howard, Marco Fornoni

- retweets: 36, favorites: 17 (01/07/2021 10:01:54)

- links: [abs](https://arxiv.org/abs/2101.01260) | [pdf](https://arxiv.org/pdf/2101.01260)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Deep learning based object detectors are commonly deployed on mobile devices to solve a variety of tasks. For maximum accuracy, each detector is usually trained to solve one single specific task, and comes with a completely independent set of parameters. While this guarantees high performance, it is also highly inefficient, as each model has to be separately downloaded and stored. In this paper we address the question: can task-specific detectors be trained and represented as a shared set of weights, plus a very small set of additional weights for each task? The main contributions of this paper are the following: 1) we perform the first systematic study of parameter-efficient transfer learning techniques for object detection problems; 2) we propose a technique to learn a model patch with a size that is dependent on the difficulty of the task to be learned, and validate our approach on 10 different object detection tasks. Our approach achieves similar accuracy as previously proposed approaches, while being significantly more compact.



