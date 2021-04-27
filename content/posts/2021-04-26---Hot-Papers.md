---
title: Hot Papers 2021-04-26
date: 2021-04-27T09:47:36.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-26"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-26"
socialImage: "/media/flying-marine.jpg"

---

# 1. A field guide to cultivating computational biology

Anne E Carpenter, Casey S Greene, Piero Carnici, Benilton S Carvalho, Michiel de Hoon, Stacey Finley, Kim-Anh Le Cao, Jerry SH Lee, Luigi Marchionni, Suzanne Sindi, Fabian J Theis, Gregory P Way, Jean YH Yang, Elana J Fertig

- retweets: 11085, favorites: 20 (04/27/2021 09:47:36)

- links: [abs](https://arxiv.org/abs/2104.11364) | [pdf](https://arxiv.org/pdf/2104.11364)
- [q-bio.OT](https://arxiv.org/list/q-bio.OT/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Biomedical research centers can empower basic discovery and novel therapeutic strategies by leveraging their large-scale datasets from experiments and patients. This data, together with new technologies to create and analyze it, has ushered in an era of data-driven discovery which requires moving beyond the traditional individual, single-discipline investigator research model. This interdisciplinary niche is where computational biology thrives. It has matured over the past three decades and made major contributions to scientific knowledge and human health, yet researchers in the field often languish in career advancement, publication, and grant review. We propose solutions for individual scientists, institutions, journal publishers, funding agencies, and educators.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do you want to attract computational biologists to your project?<br><br>Do you want to attract computational biologists to your department?<br><br>With a dozen colleagues around the globe, we present &quot;A field guide to cultivating computational biology&quot;!<br><br>Read on: <a href="https://t.co/IQU33e2TF3">https://t.co/IQU33e2TF3</a></p>&mdash; Dr. Elana J Fertig (@FertigLab) <a href="https://twitter.com/FertigLab/status/1386663542405935105?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. VidTr: Video Transformer Without Convolutions

Xinyu Li, Yanyi Zhang, Chunhui Liu, Bing Shuai, Yi Zhu, Biagio Brattoli, Hao Chen, Ivan Marsic, Joseph Tighe

- retweets: 3428, favorites: 415 (04/27/2021 09:47:36)

- links: [abs](https://arxiv.org/abs/2104.11746) | [pdf](https://arxiv.org/pdf/2104.11746)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce Video Transformer (VidTr) with separable-attention for video classification. Comparing with commonly used 3D networks, VidTr is able to aggregate spatio-temporal information via stacked attentions and provide better performance with higher efficiency. We first introduce the vanilla video transformer and show that transformer module is able to perform spatio-temporal modeling from raw pixels, but with heavy memory usage. We then present VidTr which reduces the memory cost by 3.3$\times$ while keeping the same performance. To further compact the model, we propose the standard deviation based topK pooling attention, which reduces the computation by dropping non-informative features. VidTr achieves state-of-the-art performance on five commonly used dataset with lower computational requirement, showing both the efficiency and effectiveness of our design. Finally, error analysis and visualization show that VidTr is especially good at predicting actions that require long-term temporal reasoning. The code and pre-trained weights will be released.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VidTr: Video Transformer Without Convolutions<br><br>By efficiently aggregating spatio-temporal info with stacked attentions, VidTr achieves SotA performance on video classification with lower computes.<a href="https://t.co/dtEmGRkbvx">https://t.co/dtEmGRkbvx</a> <a href="https://t.co/Kb6FzXiA18">pic.twitter.com/Kb6FzXiA18</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1386479739628523525?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VidTr: Video Transformer Without Convolutions<br>pdf: <a href="https://t.co/P2DkK9eSAI">https://t.co/P2DkK9eSAI</a><br>abs: <a href="https://t.co/FQMNHPY1LH">https://t.co/FQMNHPY1LH</a> <a href="https://t.co/mIpPpiSHXQ">pic.twitter.com/mIpPpiSHXQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386484075423313928?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Skip-Convolutions for Efficient Video Processing

Amirhossein Habibian, Davide Abati, Taco S. Cohen, Babak Ehteshami Bejnordi

- retweets: 1417, favorites: 332 (04/27/2021 09:47:37)

- links: [abs](https://arxiv.org/abs/2104.11487) | [pdf](https://arxiv.org/pdf/2104.11487)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose Skip-Convolutions to leverage the large amount of redundancies in video streams and save computations. Each video is represented as a series of changes across frames and network activations, denoted as residuals. We reformulate standard convolution to be efficiently computed on residual frames: each layer is coupled with a binary gate deciding whether a residual is important to the model prediction,~\eg foreground regions, or it can be safely skipped, e.g. background regions. These gates can either be implemented as an efficient network trained jointly with convolution kernels, or can simply skip the residuals based on their magnitude. Gating functions can also incorporate block-wise sparsity structures, as required for efficient implementation on hardware platforms. By replacing all convolutions with Skip-Convolutions in two state-of-the-art architectures, namely EfficientDet and HRNet, we reduce their computational cost consistently by a factor of 3~4x for two different tasks, without any accuracy drop. Extensive comparisons with existing model compression, as well as image and video efficiency methods demonstrate that Skip-Convolutions set a new state-of-the-art by effectively exploiting the temporal redundancies in videos.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do we need to process every single pixel in a video?<br><br>TLDR: Compute the features only at the pixels that convey new information.<br><br>Result: 4~5x less compute without any performance drop.<br><br>Manuscript: <a href="https://t.co/FRDaKOJV3g">https://t.co/FRDaKOJV3g</a> <a href="https://t.co/U7coyk6h9O">pic.twitter.com/U7coyk6h9O</a></p>&mdash; Amir Habibian (@amir_habibian) <a href="https://twitter.com/amir_habibian/status/1386596439510462465?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Skip-Convolutions for Efficient Video Processing<br>pdf: <a href="https://t.co/2cQR9vkEE7">https://t.co/2cQR9vkEE7</a><br>abs: <a href="https://t.co/czwSP4e6Gp">https://t.co/czwSP4e6Gp</a> <a href="https://t.co/m9pAGmzrfR">pic.twitter.com/m9pAGmzrfR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386493316615770114?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Motion Representations for Articulated Animation

Aliaksandr Siarohin, Oliver J. Woodford, Jian Ren, Menglei Chai, Sergey Tulyakov

- retweets: 324, favorites: 103 (04/27/2021 09:47:37)

- links: [abs](https://arxiv.org/abs/2104.11280) | [pdf](https://arxiv.org/pdf/2104.11280)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose novel motion representations for animating articulated objects consisting of distinct parts. In a completely unsupervised manner, our method identifies object parts, tracks them in a driving video, and infers their motions by considering their principal axes. In contrast to the previous keypoint-based works, our method extracts meaningful and consistent regions, describing locations, shape, and pose. The regions correspond to semantically relevant and distinct object parts, that are more easily detected in frames of the driving video. To force decoupling of foreground from background, we model non-object related global motion with an additional affine transformation. To facilitate animation and prevent the leakage of the shape of the driving object, we disentangle shape and pose of objects in the region space. Our model can animate a variety of objects, surpassing previous methods by a large margin on existing benchmarks. We present a challenging new benchmark with high-resolution videos and show that the improvement is particularly pronounced when articulated objects are considered, reaching 96.6% user preference vs. the state of the art.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Motion Representations for Articulated Animation<br>pdf: <a href="https://t.co/U9MKzPUVx7">https://t.co/U9MKzPUVx7</a><br>abs: <a href="https://t.co/oKryzN27bN">https://t.co/oKryzN27bN</a><br>project page: <a href="https://t.co/ddnhj6oCst">https://t.co/ddnhj6oCst</a> <a href="https://t.co/DfvzlpYOpJ">pic.twitter.com/DfvzlpYOpJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386495686284296193?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. What are higher-order networks?

Christian Bick, Elizabeth Gross, Heather A. Harrington, Michael T. Schaub

- retweets: 289, favorites: 45 (04/27/2021 09:47:37)

- links: [abs](https://arxiv.org/abs/2104.11329) | [pdf](https://arxiv.org/pdf/2104.11329)
- [cs.SI](https://arxiv.org/list/cs.SI/recent)

Modeling complex systems and data using the language of graphs and networks has become an essential topic across a range of different disciplines. Arguably, this network-based perspective derives is success from the relative simplicity of graphs: A graph consists of nothing more than a set of vertices and a set of edges, describing relationships between pairs of such vertices. This simple combinatorial structure makes graphs interpretable and flexible modeling tools. The simplicity of graphs as system models, however, has been scrutinized in the literature recently. Specifically, it has been argued from a variety of different angles that there is a need for higher-order networks, which go beyond the paradigm of modeling pairwise relationships, as encapsulated by graphs. In this survey article we take stock of these recent developments. Our goals are to clarify (i) what higher-order networks are, (ii) why these are interesting objects of study, and (iii) how they can be used in applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What are higher-order networks? If you have ever asked yourself this question, have a look at our new preprint <a href="https://t.co/t0QLGBEUG3">https://t.co/t0QLGBEUG3</a> (w/ E Gross, <a href="https://twitter.com/haharrington?ref_src=twsrc%5Etfw">@haharrington</a>, and M Schaub). Feedback is welcome!</p>&mdash; Christian Bick (@BickMath) <a href="https://twitter.com/BickMath/status/1386555232851738626?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Sketch-based Normal Map Generation with Geometric Sampling

Yi He, Haoran Xie, Chao Zhang, Xi Yang, Kazunori Miyata

- retweets: 90, favorites: 38 (04/27/2021 09:47:37)

- links: [abs](https://arxiv.org/abs/2104.11554) | [pdf](https://arxiv.org/pdf/2104.11554)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Normal map is an important and efficient way to represent complex 3D models. A designer may benefit from the auto-generation of high quality and accurate normal maps from freehand sketches in 3D content creation. This paper proposes a deep generative model for generating normal maps from users sketch with geometric sampling. Our generative model is based on Conditional Generative Adversarial Network with the curvature-sensitive points sampling of conditional masks. This sampling process can help eliminate the ambiguity of generation results as network input. In addition, we adopted a U-Net structure discriminator to help the generator be better trained. It is verified that the proposed framework can generate more accurate normal maps.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sketch-based Normal Map Generation with Geometric Sampling<br>pdf: <a href="https://t.co/QzeKy0r3XN">https://t.co/QzeKy0r3XN</a><br>abs: <a href="https://t.co/3oifoT6JS6">https://t.co/3oifoT6JS6</a> <a href="https://t.co/VnpTn5r1TM">pic.twitter.com/VnpTn5r1TM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386487450336440320?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Skeletor: Skeletal Transformers for Robust Body-Pose Estimation

Tao Jiang, Necati Cihan Camgoz, Richard Bowden

- retweets: 72, favorites: 56 (04/27/2021 09:47:37)

- links: [abs](https://arxiv.org/abs/2104.11712) | [pdf](https://arxiv.org/pdf/2104.11712)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Predicting 3D human pose from a single monoscopic video can be highly challenging due to factors such as low resolution, motion blur and occlusion, in addition to the fundamental ambiguity in estimating 3D from 2D. Approaches that directly regress the 3D pose from independent images can be particularly susceptible to these factors and result in jitter, noise and/or inconsistencies in skeletal estimation. Much of which can be overcome if the temporal evolution of the scene and skeleton are taken into account. However, rather than tracking body parts and trying to temporally smooth them, we propose a novel transformer based network that can learn a distribution over both pose and motion in an unsupervised fashion. We call our approach Skeletor. Skeletor overcomes inaccuracies in detection and corrects partial or entire skeleton corruption. Skeletor uses strong priors learn from on 25 million frames to correct skeleton sequences smoothly and consistently. Skeletor can achieve this as it implicitly learns the spatio-temporal context of human motion via a transformer based neural network. Extensive experiments show that Skeletor achieves improved performance on 3D human pose estimation and further provides benefits for downstream tasks such as sign language translation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Skeletor: Skeletal Transformers for Robust Body-Pose Estimation<br>pdf: <a href="https://t.co/wIRNkHgRY8">https://t.co/wIRNkHgRY8</a><br>abs: <a href="https://t.co/sCU99VAPCJ">https://t.co/sCU99VAPCJ</a> <a href="https://t.co/h3QNsUZLRD">pic.twitter.com/h3QNsUZLRD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386483596727361536?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. DisCo RL: Distribution-Conditioned Reinforcement Learning for  General-Purpose Policies

Soroush Nasiriany, Vitchyr H. Pong, Ashvin Nair, Alexander Khazatsky, Glen Berseth, Sergey Levine

- retweets: 42, favorites: 23 (04/27/2021 09:47:38)

- links: [abs](https://arxiv.org/abs/2104.11707) | [pdf](https://arxiv.org/pdf/2104.11707)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Can we use reinforcement learning to learn general-purpose policies that can perform a wide range of different tasks, resulting in flexible and reusable skills? Contextual policies provide this capability in principle, but the representation of the context determines the degree of generalization and expressivity. Categorical contexts preclude generalization to entirely new tasks. Goal-conditioned policies may enable some generalization, but cannot capture all tasks that might be desired. In this paper, we propose goal distributions as a general and broadly applicable task representation suitable for contextual policies. Goal distributions are general in the sense that they can represent any state-based reward function when equipped with an appropriate distribution class, while the particular choice of distribution class allows us to trade off expressivity and learnability. We develop an off-policy algorithm called distribution-conditioned reinforcement learning (DisCo RL) to efficiently learn these policies. We evaluate DisCo RL on a variety of robot manipulation tasks and find that it significantly outperforms prior methods on tasks that require generalization to new goal distributions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DisCo RL: Distribution-Conditioned Reinforcement Learning for General-Purpose Policies<br>pdf: <a href="https://t.co/aiFHTZl2kF">https://t.co/aiFHTZl2kF</a><br>abs: <a href="https://t.co/ZFTgZNrY7S">https://t.co/ZFTgZNrY7S</a><br>project page: <a href="https://t.co/at9D3buabj">https://t.co/at9D3buabj</a> <a href="https://t.co/5OMrIeJBIa">pic.twitter.com/5OMrIeJBIa</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386486592550981637?ref_src=twsrc%5Etfw">April 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Exact priors of finite neural networks

Jacob A. Zavatone-Veth, Cengiz Pehlevan

- retweets: 42, favorites: 17 (04/27/2021 09:47:38)

- links: [abs](https://arxiv.org/abs/2104.11734) | [pdf](https://arxiv.org/pdf/2104.11734)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Bayesian neural networks are theoretically well-understood only in the infinite-width limit, where Gaussian priors over network weights yield Gaussian priors over network outputs. Recent work has suggested that finite Bayesian networks may outperform their infinite counterparts, but their non-Gaussian output priors have been characterized only though perturbative approaches. Here, we derive exact solutions for the output priors for individual input examples of a class of finite fully-connected feedforward Bayesian neural networks. For deep linear networks, the prior has a simple expression in terms of the Meijer $G$-function. The prior of a finite ReLU network is a mixture of the priors of linear networks of smaller widths, corresponding to different numbers of active units in each layer. Our results unify previous descriptions of finite network priors in terms of their tail decay and large-width behavior.



