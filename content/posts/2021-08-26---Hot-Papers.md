---
title: Hot Papers 2021-08-26
date: 2021-08-27T08:12:59.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-26"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-26"
socialImage: "/media/flying-marine.jpg"

---

# 1. Multi-Task Self-Training for Learning General Representations

Golnaz Ghiasi, Barret Zoph, Ekin D. Cubuk, Quoc V. Le, Tsung-Yi Lin

- retweets: 1406, favorites: 134 (08/27/2021 08:12:59)

- links: [abs](https://arxiv.org/abs/2108.11353) | [pdf](https://arxiv.org/pdf/2108.11353)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Despite the fast progress in training specialized models for various tasks, learning a single general model that works well for many tasks is still challenging for computer vision. Here we introduce multi-task self-training (MuST), which harnesses the knowledge in independent specialized teacher models (e.g., ImageNet model on classification) to train a single general student model. Our approach has three steps. First, we train specialized teachers independently on labeled datasets. We then use the specialized teachers to label an unlabeled dataset to create a multi-task pseudo labeled dataset. Finally, the dataset, which now contains pseudo labels from teacher models trained on different datasets/tasks, is then used to train a student model with multi-task learning. We evaluate the feature representations of the student model on 6 vision tasks including image recognition (classification, detection, segmentation)and 3D geometry estimation (depth and surface normal estimation). MuST is scalable with unlabeled or partially labeled datasets and outperforms both specialized supervised models and self-supervised models when training on large scale datasets. Lastly, we show MuST can improve upon already strong checkpoints trained with billions of examples. The results suggest self-training is a promising direction to aggregate labeled and unlabeled training data for learning general feature representations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Task Self-Training for Learning General Representations<br>pdf: <a href="https://t.co/c6IlbxEymv">https://t.co/c6IlbxEymv</a><br>abs: <a href="https://t.co/t1Hf7o4vVb">https://t.co/t1Hf7o4vVb</a><br><br>a scalable multi-task self-training method for learning general representations <a href="https://t.co/gWaMsmPTuW">pic.twitter.com/gWaMsmPTuW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1430691351948038147?ref_src=twsrc%5Etfw">August 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. YOLOP: You Only Look Once for Panoptic Driving Perception

Dong Wu, Manwen Liao, Weitian Zhang, Xinggang Wang

- retweets: 837, favorites: 117 (08/27/2021 08:13:00)

- links: [abs](https://arxiv.org/abs/2108.11250) | [pdf](https://arxiv.org/pdf/2108.11250)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

A panoptic driving perception system is an essential part of autonomous driving. A high-precision and real-time perception system can assist the vehicle in making the reasonable decision while driving. We present a panoptic driving perception network (YOLOP) to perform traffic object detection, drivable area segmentation and lane detection simultaneously. It is composed of one encoder for feature extraction and three decoders to handle the specific tasks. Our model performs extremely well on the challenging BDD100K dataset, achieving state-of-the-art on all three tasks in terms of accuracy and speed. Besides, we verify the effectiveness of our multi-task learning model for joint training via ablative studies. To our best knowledge, this is the first work that can process these three visual perception tasks simultaneously in real-time on an embedded device Jetson TX2(23 FPS) and maintain excellent accuracy. To facilitate further research, the source codes and pre-trained models will be released at https://github.com/hustvl/YOLOP.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">YOLOP: You Only Look Once for Panoptic Driving Perception<br>pdf: <a href="https://t.co/8rlg7943Qw">https://t.co/8rlg7943Qw</a><br>abs: <a href="https://t.co/56n9NmzYzF">https://t.co/56n9NmzYzF</a><br>github: <a href="https://t.co/wt6uMXYz4o">https://t.co/wt6uMXYz4o</a> <a href="https://t.co/WP97KoM115">pic.twitter.com/WP97KoM115</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1430703868871122944?ref_src=twsrc%5Etfw">August 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral

Lucio M. Dery, Yann Dauphin, David Grangier

- retweets: 56, favorites: 27 (08/27/2021 08:13:00)

- links: [abs](https://arxiv.org/abs/2108.11346) | [pdf](https://arxiv.org/pdf/2108.11346)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

While deep learning has been very beneficial in data-rich settings, tasks with smaller training set often resort to pre-training or multitask learning to leverage data from other tasks. In this case, careful consideration is needed to select tasks and model parameterizations such that updates from the auxiliary tasks actually help the primary task. We seek to alleviate this burden by formulating a model-agnostic framework that performs fine-grained manipulation of the auxiliary task gradients. We propose to decompose auxiliary updates into directions which help, damage or leave the primary task loss unchanged. This allows weighting the update directions differently depending on their impact on the problem of interest. We present a novel and efficient algorithm for that purpose and show its advantage in practice. Our method leverages efficient automatic differentiation procedures and randomized singular value decomposition for scalability. We show that our framework is generic and encompasses some prior work as particular cases. Our approach consistently outperforms strong and widely used baselines when leveraging out-of-distribution data for Text and Image classification tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral<br>pdf: <a href="https://t.co/4usxlO7k1U">https://t.co/4usxlO7k1U</a><br>abs: <a href="https://t.co/o4C2A0cwir">https://t.co/o4C2A0cwir</a><br>github: <a href="https://t.co/ehIqytSHRJ">https://t.co/ehIqytSHRJ</a> <a href="https://t.co/cLNQHVsfm1">pic.twitter.com/cLNQHVsfm1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1430709844902821899?ref_src=twsrc%5Etfw">August 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



