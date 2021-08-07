---
title: Hot Papers 2021-08-06
date: 2021-08-07T17:48:26.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-06"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-06"
socialImage: "/media/flying-marine.jpg"

---

# 1. Sketch Your Own GAN

Sheng-Yu Wang, David Bau, Jun-Yan Zhu

- retweets: 2597, favorites: 245 (08/07/2021 17:48:26)

- links: [abs](https://arxiv.org/abs/2108.02774) | [pdf](https://arxiv.org/pdf/2108.02774)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Can a user create a deep generative model by sketching a single example? Traditionally, creating a GAN model has required the collection of a large-scale dataset of exemplars and specialized knowledge in deep learning. In contrast, sketching is possibly the most universally accessible way to convey a visual concept. In this work, we present a method, GAN Sketching, for rewriting GANs with one or more sketches, to make GANs training easier for novice users. In particular, we change the weights of an original GAN model according to user sketches. We encourage the model's output to match the user sketches through a cross-domain adversarial loss. Furthermore, we explore different regularization methods to preserve the original model's diversity and image quality. Experiments have shown that our method can mold GANs to match shapes and poses specified by sketches while maintaining realism and diversity. Finally, we demonstrate a few applications of the resulting GAN, including latent space interpolation and image editing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sketch Your Own GAN<br>pdf: <a href="https://t.co/RkxmDGnAN4">https://t.co/RkxmDGnAN4</a><br>abs: <a href="https://t.co/B10d2OQnfr">https://t.co/B10d2OQnfr</a><br>project page: <a href="https://t.co/lRfBvHyWFR">https://t.co/lRfBvHyWFR</a><br>method can mold GANs to match shapes and poses specified by sketches while maintaining realism and diversity <a href="https://t.co/kCyMnxtIuL">pic.twitter.com/kCyMnxtIuL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423445312602099712?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. The AI Economist: Optimal Economic Policy Design via Two-level Deep  Reinforcement Learning

Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C. Parkes, Richard Socher

- retweets: 428, favorites: 70 (08/07/2021 17:48:26)

- links: [abs](https://arxiv.org/abs/2108.02755) | [pdf](https://arxiv.org/pdf/2108.02755)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [econ.GN](https://arxiv.org/list/econ.GN/recent)

AI and reinforcement learning (RL) have improved many areas, but are not yet widely adopted in economic policy design, mechanism design, or economics at large. At the same time, current economic methodology is limited by a lack of counterfactual data, simplistic behavioral models, and limited opportunities to experiment with policies and evaluate behavioral responses. Here we show that machine-learning-based economic simulation is a powerful policy and mechanism design framework to overcome these limitations. The AI Economist is a two-level, deep RL framework that trains both agents and a social planner who co-adapt, providing a tractable solution to the highly unstable and novel two-level RL challenge. From a simple specification of an economy, we learn rational agent behaviors that adapt to learned planner policies and vice versa. We demonstrate the efficacy of the AI Economist on the problem of optimal taxation. In simple one-step economies, the AI Economist recovers the optimal tax policy of economic theory. In complex, dynamic economies, the AI Economist substantially improves both utilitarian social welfare and the trade-off between equality and productivity over baselines. It does so despite emergent tax-gaming strategies, while accounting for agent interactions and behavioral change more accurately than economic theory. These results demonstrate for the first time that two-level, deep RL can be used for understanding and as a complement to theory for economic design, unlocking a new computational learning-based approach to understanding economic policy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The AI Economist: Optimal Economic Policy Design via Two-level Deep Reinforcement Learning<br>pdf: <a href="https://t.co/us7gsDrgkO">https://t.co/us7gsDrgkO</a><br>abs: <a href="https://t.co/GfBZqG0B9x">https://t.co/GfBZqG0B9x</a> <a href="https://t.co/u2gWPRDbvF">pic.twitter.com/u2gWPRDbvF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423451048644157442?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Token Shift Transformer for Video Classification

Hao Zhang, Yanbin Hao, Chong-Wah Ngo

- retweets: 266, favorites: 98 (08/07/2021 17:48:26)

- links: [abs](https://arxiv.org/abs/2108.02432) | [pdf](https://arxiv.org/pdf/2108.02432)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Transformer achieves remarkable successes in understanding 1 and 2-dimensional signals (e.g., NLP and Image Content Understanding). As a potential alternative to convolutional neural networks, it shares merits of strong interpretability, high discriminative power on hyper-scale data, and flexibility in processing varying length inputs. However, its encoders naturally contain computational intensive operations such as pair-wise self-attention, incurring heavy computational burden when being applied on the complex 3-dimensional video signals.   This paper presents Token Shift Module (i.e., TokShift), a novel, zero-parameter, zero-FLOPs operator, for modeling temporal relations within each transformer encoder. Specifically, the TokShift barely temporally shifts partial [Class] token features back-and-forth across adjacent frames. Then, we densely plug the module into each encoder of a plain 2D vision transformer for learning 3D video representation. It is worth noticing that our TokShift transformer is a pure convolutional-free video transformer pilot with computational efficiency for video understanding. Experiments on standard benchmarks verify its robustness, effectiveness, and efficiency. Particularly, with input clips of 8/12 frames, the TokShift transformer achieves SOTA precision: 79.83%/80.40% on the Kinetics-400, 66.56% on EGTEA-Gaze+, and 96.80% on UCF-101 datasets, comparable or better than existing SOTA convolutional counterparts. Our code is open-sourced in: https://github.com/VideoNetworks/TokShift-Transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Token Shift Transformer for Video Classification<br>pdf: <a href="https://t.co/sdbS5P5RpD">https://t.co/sdbS5P5RpD</a><br>abs: <a href="https://t.co/w5UpOnjHjl">https://t.co/w5UpOnjHjl</a><br>github: <a href="https://t.co/4KQ0rdfCHN">https://t.co/4KQ0rdfCHN</a> <a href="https://t.co/A2RA717L84">pic.twitter.com/A2RA717L84</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423455367367901186?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Video Contrastive Learning with Global Context

Haofei Kuang, Yi Zhu, Zhi Zhang, Xinyu Li, Joseph Tighe, Sören Schwertfeger, Cyrill Stachniss, Mu Li

- retweets: 180, favorites: 72 (08/07/2021 17:48:26)

- links: [abs](https://arxiv.org/abs/2108.02722) | [pdf](https://arxiv.org/pdf/2108.02722)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Contrastive learning has revolutionized self-supervised image representation learning field, and recently been adapted to video domain. One of the greatest advantages of contrastive learning is that it allows us to flexibly define powerful loss objectives as long as we can find a reasonable way to formulate positive and negative samples to contrast. However, existing approaches rely heavily on the short-range spatiotemporal salience to form clip-level contrastive signals, thus limit themselves from using global context. In this paper, we propose a new video-level contrastive learning method based on segments to formulate positive pairs. Our formulation is able to capture global context in a video, thus robust to temporal content change. We also incorporate a temporal order regularization term to enforce the inherent sequential structure of videos. Extensive experiments show that our video-level contrastive learning framework (VCLR) is able to outperform previous state-of-the-arts on five video datasets for downstream action classification, action localization and video retrieval. Code is available at https://github.com/amazon-research/video-contrastive-learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Video Contrastive Learning with Global Context<br>pdf: <a href="https://t.co/0kkXi2hu3X">https://t.co/0kkXi2hu3X</a><br>abs: <a href="https://t.co/se2YGoaoo6">https://t.co/se2YGoaoo6</a><br>github: <a href="https://t.co/Rhn4WJjquM">https://t.co/Rhn4WJjquM</a> <a href="https://t.co/HQlA0zw2O2">pic.twitter.com/HQlA0zw2O2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423464499571568642?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. FMMformer: Efficient and Flexible Transformer via Decomposed Near-field  and Far-field Attention

Tan M. Nguyen, Vai Suliafu, Stanley J. Osher, Long Chen, Bao Wang

- retweets: 184, favorites: 40 (08/07/2021 17:48:27)

- links: [abs](https://arxiv.org/abs/2108.02347) | [pdf](https://arxiv.org/pdf/2108.02347)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [math.NA](https://arxiv.org/list/math.NA/recent)

We propose FMMformers, a class of efficient and flexible transformers inspired by the celebrated fast multipole method (FMM) for accelerating interacting particle simulation. FMM decomposes particle-particle interaction into near-field and far-field components and then performs direct and coarse-grained computation, respectively. Similarly, FMMformers decompose the attention into near-field and far-field attention, modeling the near-field attention by a banded matrix and the far-field attention by a low-rank matrix. Computing the attention matrix for FMMformers requires linear complexity in computational time and memory footprint with respect to the sequence length. In contrast, standard transformers suffer from quadratic complexity. We analyze and validate the advantage of FMMformers over the standard transformer on the Long Range Arena and language modeling benchmarks. FMMformers can even outperform the standard transformer in terms of accuracy by a significant margin. For instance, FMMformers achieve an average classification accuracy of $60.74\%$ over the five Long Range Arena tasks, which is significantly better than the standard transformer's average accuracy of $58.70\%$.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FMMformer: Efficient and Flexible Transformer via Decomposed Near-field and Far-field Attention<br>pdf: <a href="https://t.co/UaMK9kpATm">https://t.co/UaMK9kpATm</a><br>abs: <a href="https://t.co/i95PMoPn47">https://t.co/i95PMoPn47</a><br><br>achieves an average classification accuracy of 60.74% over the five Long Range Arena tasks <a href="https://t.co/YLsNilUkn2">pic.twitter.com/YLsNilUkn2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423487452149334017?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Accelerating XOR-based Erasure Coding using Program Optimization  Techniques

Yuya Uezato

- retweets: 132, favorites: 33 (08/07/2021 17:48:27)

- links: [abs](https://arxiv.org/abs/2108.02692) | [pdf](https://arxiv.org/pdf/2108.02692)
- [cs.PL](https://arxiv.org/list/cs.PL/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

Erasure coding (EC) affords data redundancy for large-scale systems. XOR-based EC is an easy-to-implement method for optimizing EC. This paper addresses a significant performance gap between the state-of-the-art XOR-based EC approach (with 4.9 GB/s coding throughput) and Intel's high-performance EC library based on another approach (with 6.7 GB/s). We propose a novel approach based on our observation that XOR-based EC virtually generates programs of a Domain Specific Language for XORing byte arrays. We formalize such programs as straight-line programs (SLPs) of compiler construction and optimize SLPs using various optimization techniques. Our optimization flow is three-fold: 1) reducing operations using grammar compression algorithms; 2) reducing memory accesses using deforestation, a functional program optimization method; and 3) reducing cache misses using the (red-blue) pebble game of program analysis. We provide an experimental library, which outperforms Intel's library with 8.92 GB/s throughput.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">SCというHPC分野の最高峰の国際会議に単著論文が採録されました!! 著者版をarXivで公開してます<a href="https://t.co/5VvA6an5ej">https://t.co/5VvA6an5ej</a><br>SCはスパコンTop500やゴードンベル賞の発表等も行われる由緒ある会議で、スパコンなど巨大スケール上の計算に関する最前線の応用や大きな理論的成果を挙げた論文が発表されます。</p>&mdash; Ü+1F980🦀 (@ranha) <a href="https://twitter.com/ranha/status/1423624176330416135?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Nonperturbative renormalization for the neural network-QFT  correspondence

Harold Erbin, Vincent Lahoche, Dine Ousmane Samary

- retweets: 64, favorites: 29 (08/07/2021 17:48:27)

- links: [abs](https://arxiv.org/abs/2108.01403) | [pdf](https://arxiv.org/pdf/2108.01403)
- [hep-th](https://arxiv.org/list/hep-th/recent) | [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In a recent work arXiv:2008.08601, Halverson, Maiti and Stoner proposed a description of neural networks in terms of a Wilsonian effective field theory. The infinite-width limit is mapped to a free field theory, while finite $N$ corrections are taken into account by interactions (non-Gaussian terms in the action). In this paper, we study two related aspects of this correspondence. First, we comment on the concepts of locality and power-counting in this context. Indeed, these usual space-time notions may not hold for neural networks (since inputs can be arbitrary), however, the renormalization group provides natural notions of locality and scaling. Moreover, we comment on several subtleties, for example, that data components may not have a permutation symmetry: in that case, we argue that random tensor field theories could provide a natural generalization. Second, we improve the perturbative Wilsonian renormalization from arXiv:2008.08601 by providing an analysis in terms of the nonperturbative renormalization group using the Wetterich-Morris equation. An important difference with usual nonperturbative RG analysis is that only the effective (IR) 2-point function is known, which requires setting the problem with care. Our aim is to provide a useful formalism to investigate neural networks behavior beyond the large-width limit (i.e.~far from Gaussian limit) in a nonperturbative fashion. A major result of our analysis is that changing the standard deviation of the neural network weight distribution can be interpreted as a renormalization flow in the space of networks. We focus on translations invariant kernels and provide preliminary numerical results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper on <a href="https://twitter.com/hashtag/QFT?src=hash&amp;ref_src=twsrc%5Etfw">#QFT</a> and <a href="https://twitter.com/hashtag/renormalization?src=hash&amp;ref_src=twsrc%5Etfw">#renormalization</a> for <a href="https://twitter.com/hashtag/NeuralNetworks?src=hash&amp;ref_src=twsrc%5Etfw">#NeuralNetworks</a> is online! Inspired by great work from <a href="https://twitter.com/jhhalverson?ref_src=twsrc%5Etfw">@jhhalverson</a>.<br>Main practical result: networks with weights initialized with different std are related by a renormalization flow.<a href="https://t.co/vPe0blq35X">https://t.co/vPe0blq35X</a></p>&mdash; Harold Erbin (@HaroldErbin) <a href="https://twitter.com/HaroldErbin/status/1422908502297780225?ref_src=twsrc%5Etfw">August 4, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A FAIR and AI-ready Higgs Boson Decay Dataset

Yifan Chen, E. A. Huerta, Javier Duarte, Philip Harris, Daniel S. Katz, Mark S. Neubauer, Daniel Diaz, Farouk Mokhtar, Raghav Kansal, Sang Eon Park, Volodymyr V. Kindratenko, Zhizhen Zhao, Roger Rusack

- retweets: 55, favorites: 26 (08/07/2021 17:48:27)

- links: [abs](https://arxiv.org/abs/2108.02214) | [pdf](https://arxiv.org/pdf/2108.02214)
- [hep-ex](https://arxiv.org/list/hep-ex/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent) | [hep-ph](https://arxiv.org/list/hep-ph/recent)

To enable the reusability of massive scientific datasets by humans and machines, researchers aim to create scientific datasets that adhere to the principles of findability, accessibility, interoperability, and reusability (FAIR) for data and artificial intelligence (AI) models. This article provides a domain-agnostic, step-by-step assessment guide to evaluate whether or not a given dataset meets each FAIR principle. We then demonstrate how to use this guide to evaluate the FAIRness of an open simulated dataset produced by the CMS Collaboration at the CERN Large Hadron Collider. This dataset consists of Higgs boson decays and quark and gluon background, and is available through the CERN Open Data Portal. We also use other available tools to assess the FAIRness of this dataset, and incorporate feedback from members of the FAIR community to validate our results. This article is accompanied by a Jupyter notebook to facilitate an understanding and exploration of the dataset, including visualization of its elements. This study marks the first in a planned series of articles that will guide scientists in the creation and quantification of FAIRness in high energy particle physics datasets and AI models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">First paper from FAIR4HEP posted! <a href="https://t.co/4wbGognuln">https://t.co/4wbGognuln</a>. We provide an assessment guide to evaluate the degree to which a given data product meets the FAIR standards and apply it to an open, simulated <a href="https://twitter.com/hashtag/Higgs?src=hash&amp;ref_src=twsrc%5Etfw">#Higgs</a> dataset produced by <a href="https://twitter.com/CMSExperiment?ref_src=twsrc%5Etfw">@CMSExperiment</a>.  With <a href="https://twitter.com/danielskatz?ref_src=twsrc%5Etfw">@danielskatz</a> <a href="https://twitter.com/jmgduarte?ref_src=twsrc%5Etfw">@jmgduarte</a> ++</p>&mdash; Mark Neubauer (@MarkSNeubauer) <a href="https://twitter.com/MarkSNeubauer/status/1423494914286014469?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. SLAMP: Stochastic Latent Appearance and Motion Prediction

Adil Kaan Akan, Erkut Erdem, Aykut Erdem, Fatma Güney

- retweets: 42, favorites: 35 (08/07/2021 17:48:27)

- links: [abs](https://arxiv.org/abs/2108.02760) | [pdf](https://arxiv.org/pdf/2108.02760)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Motion is an important cue for video prediction and often utilized by separating video content into static and dynamic components. Most of the previous work utilizing motion is deterministic but there are stochastic methods that can model the inherent uncertainty of the future. Existing stochastic models either do not reason about motion explicitly or make limiting assumptions about the static part. In this paper, we reason about appearance and motion in the video stochastically by predicting the future based on the motion history. Explicit reasoning about motion without history already reaches the performance of current stochastic models. The motion history further improves the results by allowing to predict consistent dynamics several frames into the future. Our model performs comparably to the state-of-the-art models on the generic video prediction datasets, however, significantly outperforms them on two challenging real-world autonomous driving datasets with complex motion and dynamic background.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/ICCV2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICCV2021</a> paper &quot;SLAMP: Stochastic Latent Appearance and Motion Prediction&quot; is now public! joint work with <a href="https://twitter.com/ftmguney?ref_src=twsrc%5Etfw">@ftmguney</a>, <a href="https://twitter.com/aykuterdemml?ref_src=twsrc%5Etfw">@aykuterdemml</a>, <a href="https://twitter.com/erkuterdem?ref_src=twsrc%5Etfw">@erkuterdem</a>.<br>Paper: <a href="https://t.co/CjWxbuUCWX">https://t.co/CjWxbuUCWX</a><br>Project website: <a href="https://t.co/78p88SfLkl">https://t.co/78p88SfLkl</a> <a href="https://t.co/lgbomzAbDw">pic.twitter.com/lgbomzAbDw</a></p>&mdash; Kaan Akan (@akaan_akan) <a href="https://twitter.com/akaan_akan/status/1423688530124025864?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Object Wake-up: 3-D Object Reconstruction, Animation, and in-situ  Rendering from a Single Image

Xinxin Zuo, Ji Yang, Sen Wang, Zhenbo Yu, Xinyu Li, Bingbing Ni, Minglun Gong, Li Cheng

- retweets: 22, favorites: 41 (08/07/2021 17:48:27)

- links: [abs](https://arxiv.org/abs/2108.02708) | [pdf](https://arxiv.org/pdf/2108.02708)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Given a picture of a chair, could we extract the 3-D shape of the chair, animate its plausible articulations and motions, and render in-situ in its original image space? The above question prompts us to devise an automated approach to extract and manipulate articulated objects in single images. Comparing with previous efforts on object manipulation, our work goes beyond 2-D manipulation and focuses on articulable objects, thus introduces greater flexibility for possible object deformations. The pipeline of our approach starts by reconstructing and refining a 3-D mesh representation of the object of interest from an input image; its control joints are predicted by exploiting the semantic part segmentation information; the obtained object 3-D mesh is then rigged \& animated by non-rigid deformation, and rendered to perform in-situ motions in its original image space. Quantitative evaluations are carried out on 3-D reconstruction from single images, an established task that is related to our pipeline, where our results surpass those of the SOTAs by a noticeable margin. Extensive visual results also demonstrate the applicability of our approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Object Wake-up: 3-D Object Reconstruction, Animation, and in-situ Rendering from a Single Image<br>pdf: <a href="https://t.co/B37FwZ7PJp">https://t.co/B37FwZ7PJp</a><br>abs: <a href="https://t.co/IvQNPzuOYV">https://t.co/IvQNPzuOYV</a> <a href="https://t.co/oy4GDP8biL">pic.twitter.com/oy4GDP8biL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423456374495846405?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Fast Convergence of DETR with Spatially Modulated Co-Attention

Peng Gao, Minghang Zheng, Xiaogang Wang, Jifeng Dai, Hongsheng Li

- retweets: 35, favorites: 22 (08/07/2021 17:48:27)

- links: [abs](https://arxiv.org/abs/2108.02404) | [pdf](https://arxiv.org/pdf/2108.02404)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The recently proposed Detection Transformer (DETR) model successfully applies Transformer to objects detection and achieves comparable performance with two-stage object detection frameworks, such as Faster-RCNN. However, DETR suffers from its slow convergence. Training DETR from scratch needs 500 epochs to achieve a high accuracy. To accelerate its convergence, we propose a simple yet effective scheme for improving the DETR framework, namely Spatially Modulated Co-Attention (SMCA) mechanism. The core idea of SMCA is to conduct location-aware co-attention in DETR by constraining co-attention responses to be high near initially estimated bounding box locations. Our proposed SMCA increases DETR's convergence speed by replacing the original co-attention mechanism in the decoder while keeping other operations in DETR unchanged. Furthermore, by integrating multi-head and scale-selection attention designs into SMCA, our fully-fledged SMCA can achieve better performance compared to DETR with a dilated convolution-based backbone (45.6 mAP at 108 epochs vs. 43.3 mAP at 500 epochs). We perform extensive ablation studies on COCO dataset to validate SMCA. Code is released at https://github.com/gaopengcuhk/SMCA-DETR .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fast Convergence of DETR with Spatially Modulated Co-Attention<br>pdf: <a href="https://t.co/6LvvePylaZ">https://t.co/6LvvePylaZ</a><br>abs: <a href="https://t.co/OIWmU9CsRA">https://t.co/OIWmU9CsRA</a><br><br>fully fledged SMCA achieves better performance compared to DETR with a dilated convolution-based backbone (45.6 mAP at 108 epochs vs. 43.3 mAP at 500 epochs) <a href="https://t.co/K3Xj2eqtms">pic.twitter.com/K3Xj2eqtms</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423453069505859586?ref_src=twsrc%5Etfw">August 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



