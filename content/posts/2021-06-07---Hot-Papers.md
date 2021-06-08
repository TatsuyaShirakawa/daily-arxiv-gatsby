---
title: Hot Papers 2021-06-07
date: 2021-06-08T09:10:32.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-07"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-07"
socialImage: "/media/flying-marine.jpg"

---

# 1. How Great is the Great Firewall? Measuring China's DNS Censorship

Nguyen Phong Hoang, Arian Akhavan Niaki, Jakub Dalek, Jeffrey Knockel, Pellaeon Lin, Bill Marczak, Masashi Crete-Nishihata, Phillipa Gill, Michalis Polychronakis

- retweets: 9496, favorites: 5 (06/08/2021 09:10:32)

- links: [abs](https://arxiv.org/abs/2106.02167) | [pdf](https://arxiv.org/pdf/2106.02167)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.NI](https://arxiv.org/list/cs.NI/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

The DNS filtering apparatus of China's Great Firewall (GFW) has evolved considerably over the past two decades. However, most prior studies of China's DNS filtering were performed over short time periods, leading to unnoticed changes in the GFW's behavior. In this study, we introduce GFWatch, a large-scale, longitudinal measurement platform capable of testing hundreds of millions of domains daily, enabling continuous monitoring of the GFW's DNS filtering behavior.   We present the results of running GFWatch over a nine-month period, during which we tested an average of 411M domains per day and detected a total of 311K domains censored by GFW's DNS filter. To the best of our knowledge, this is the largest number of domains tested and censored domains discovered in the literature. We further reverse engineer regular expressions used by the GFW and find 41K innocuous domains that match these filters, resulting in overblocking of their content. We also observe bogus IPv6 and globally routable IPv4 addresses injected by the GFW, including addresses owned by US companies, such as Facebook, Dropbox, and Twitter.   Using data from GFWatch, we studied the impact of GFW blocking on the global DNS system. We found 77K censored domains with DNS resource records polluted in popular public DNS resolvers, such as Google and Cloudflare. Finally, we propose strategies to detect poisoned responses that can (1) sanitize poisoned DNS records from the cache of public DNS resolvers, and (2) assist in the development of circumvention tools to bypass the GFW's DNS censorship.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">For the last several months, you might have noticed many tweets of mine reporting new domains censored by the Great Firewall. Today, I’m happy to share the research paper behind these tweets, which will be presented at the 30th <a href="https://twitter.com/USENIXSecurity?ref_src=twsrc%5Etfw">@USENIXSecurity</a> Symposium.<a href="https://t.co/iRGr4VcJSi">https://t.co/iRGr4VcJSi</a> <a href="https://t.co/IB8xnnSmd7">pic.twitter.com/IB8xnnSmd7</a></p>&mdash; Phong (@NP_tokumei) <a href="https://twitter.com/NP_tokumei/status/1401716316394844161?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Self-Attention Between Datapoints: Going Beyond Individual Input-Output  Pairs in Deep Learning

Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, Yarin Gal

- retweets: 7578, favorites: 4 (06/08/2021 09:10:32)

- links: [abs](https://arxiv.org/abs/2106.02584) | [pdf](https://arxiv.org/pdf/2106.02584)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We challenge a common assumption underlying most supervised deep learning: that a model makes a prediction depending only on its parameters and the features of a single input. To this end, we introduce a general-purpose deep learning architecture that takes as input the entire dataset instead of processing one datapoint at a time. Our approach uses self-attention to reason about relationships between datapoints explicitly, which can be seen as realizing non-parametric models using parametric attention mechanisms. However, unlike conventional non-parametric models, we let the model learn end-to-end from the data how to make use of other datapoints for prediction. Empirically, our models solve cross-datapoint lookup and complex reasoning tasks unsolvable by traditional deep learning models. We show highly competitive results on tabular data, early results on CIFAR-10, and give insight into how the model makes use of the interactions between points.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning<br><br>Introduces a general-purpose deep learning architecture that takes as input the entire dataset instead of processing one datapoint at a time.<a href="https://t.co/ARIsdy3nUg">https://t.co/ARIsdy3nUg</a> <a href="https://t.co/UfSeJRSDwX">pic.twitter.com/UfSeJRSDwX</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1401701981790474240?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. X-volution: On the unification of convolution and self-attention

Xuanhong Chen, Hang Wang, Bingbing Ni

- retweets: 5955, favorites: 585 (06/08/2021 09:10:32)

- links: [abs](https://arxiv.org/abs/2106.02253) | [pdf](https://arxiv.org/pdf/2106.02253)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Convolution and self-attention are acting as two fundamental building blocks in deep neural networks, where the former extracts local image features in a linear way while the latter non-locally encodes high-order contextual relationships. Though essentially complementary to each other, i.e., first-/high-order, stat-of-the-art architectures, i.e., CNNs or transformers lack a principled way to simultaneously apply both operations in a single computational module, due to their heterogeneous computing pattern and excessive burden of global dot-product for visual tasks. In this work, we theoretically derive a global self-attention approximation scheme, which approximates a self-attention via the convolution operation on transformed features. Based on the approximated scheme, we establish a multi-branch elementary module composed of both convolution and self-attention operation, capable of unifying both local and non-local feature interaction. Importantly, once trained, this multi-branch module could be conditionally converted into a single standard convolution operation via structural re-parameterization, rendering a pure convolution styled operator named X-volution, ready to be plugged into any modern networks as an atomic operation. Extensive experiments demonstrate that the proposed X-volution, achieves highly competitive visual understanding improvements (+1.2% top-1 accuracy on ImageNet classification, +1.7 box AP and +1.5 mask AP on COCO detection and segmentation).

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">「Attention is all you need」「Convolutionの方がいいかも?」「原点回帰でMLPが最強」とカオスを極めたアーキテクチャ論争,ついに「ならAttentionとConvolutionでいいとこ取り合体しよう」の発想が登場<br>X-volution: On the unification of convolution and self-attention<a href="https://t.co/D0pgRgciCB">https://t.co/D0pgRgciCB</a> <a href="https://t.co/Ry58IbPWyE">pic.twitter.com/Ry58IbPWyE</a></p>&mdash; えるエル (@ImAI_Eruel) <a href="https://twitter.com/ImAI_Eruel/status/1401745084765245443?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">X-volution: On the Unification of Convolution and<br>Self-attention<br>pdf: <a href="https://t.co/QhTwIybRvx">https://t.co/QhTwIybRvx</a><br>abs: <a href="https://t.co/NyFh07kpGE">https://t.co/NyFh07kpGE</a><br><br>+1.2% top-1 accuracy on ImageNet classification, +1.7 box AP and +1.5 mask AP on COCO detection and segmentation <a href="https://t.co/wxiRb0KY0l">pic.twitter.com/wxiRb0KY0l</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401701719785021441?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. MERLOT: Multimodal Neural Script Knowledge Models

Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, Yejin Choi

- retweets: 1755, favorites: 322 (06/08/2021 09:10:32)

- links: [abs](https://arxiv.org/abs/2106.02636) | [pdf](https://arxiv.org/pdf/2106.02636)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

As humans, we understand events in the visual world contextually, performing multimodal reasoning across time to make inferences about the past, present, and future. We introduce MERLOT, a model that learns multimodal script knowledge by watching millions of YouTube videos with transcribed speech -- in an entirely label-free, self-supervised manner. By pretraining with a mix of both frame-level (spatial) and video-level (temporal) objectives, our model not only learns to match images to temporally corresponding words, but also to contextualize what is happening globally over time. As a result, MERLOT exhibits strong out-of-the-box representations of temporal commonsense, and achieves state-of-the-art performance on 12 different video QA datasets when finetuned. It also transfers well to the world of static images, allowing models to reason about the dynamic context behind visual scenes. On Visual Commonsense Reasoning, MERLOT answers questions correctly with 80.6% accuracy, outperforming state-of-the-art models of similar size by over 3%, even those that make heavy use of auxiliary supervised data (like object bounding boxes).   Ablation analyses demonstrate the complementary importance of: 1) training on videos versus static images; 2) scaling the magnitude and diversity of the pretraining video corpus; and 3) using diverse objectives that encourage full-stack multimodal reasoning, from the recognition to cognition level.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing MERLOT: a new model that learns about language, vision, &amp; the world from 6M YouTube videos.<br><br>Out-of-the-box, MERLOT has intrinsic notions of multimodal temporal commonsense. When finetuned, we get SOTA performance on 12 video tasks + VCR.<a href="https://t.co/2H6ng2Yfxt">https://t.co/2H6ng2Yfxt</a> <a href="https://t.co/oMPPtOjLBm">pic.twitter.com/oMPPtOjLBm</a></p>&mdash; Rowan Zellers (@rown) <a href="https://twitter.com/rown/status/1401980209876783109?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🍷Super excited about our new preprint!🍷<br><br>𝓜𝓔𝓡𝓛𝓞𝓣: Multimodal Script Knowledge Models!<a href="https://t.co/qkUVY8Im5B">https://t.co/qkUVY8Im5B</a><a href="https://t.co/msAKGFzewv">https://t.co/msAKGFzewv</a><br><br>TL;DR: By pretraining on 6M youtube videos, we transfer with SoTA performance on 10+ tasks (e.g. Video QA) that require temporal reasoning <a href="https://t.co/sqbf7qo0hr">pic.twitter.com/sqbf7qo0hr</a></p>&mdash; Jack Hessel (@jmhessel) <a href="https://twitter.com/jmhessel/status/1401983972272345088?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MERLOT: Multimodal Neural Script Knowledge Models<br>pdf: <a href="https://t.co/vzmHC42rI4">https://t.co/vzmHC42rI4</a><br>abs: <a href="https://t.co/3ADDscKw8i">https://t.co/3ADDscKw8i</a><br>project page: <a href="https://t.co/LhPfzluxqd">https://t.co/LhPfzluxqd</a><br><br>learns multimodal script knowledge, watching millions of YT videos with transcribed speech in entirely<br>label-free, ss manner <a href="https://t.co/W7B9OVOF9c">pic.twitter.com/W7B9OVOF9c</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401700556926763018?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Ukiyo-e Analysis and Creativity with Attribute and Geometry Annotation

Yingtao Tian, Tarin Clanuwat, Chikahiko Suzuki, Asanobu Kitamoto

- retweets: 1784, favorites: 231 (06/08/2021 09:10:33)

- links: [abs](https://arxiv.org/abs/2106.02267) | [pdf](https://arxiv.org/pdf/2106.02267)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The study of Ukiyo-e, an important genre of pre-modern Japanese art, focuses on the object and style like other artwork researches. Such study has benefited from the renewed interest by the machine learning community in culturally important topics, leading to interdisciplinary works including collections of images, quantitative approaches, and machine learning-based creativities. They, however, have several drawbacks, and it remains challenging to integrate these works into a comprehensive view. To bridge this gap, we propose a holistic approach We first present a large-scale Ukiyo-e dataset with coherent semantic labels and geometric annotations, then show its value in a quantitative study of Ukiyo-e paintings' object using these labels and annotations. We further demonstrate the machine learning methods could help style study through soft color decomposition of Ukiyo-e, and finally provides joint insights into object and style by composing sketches and colors using colorization. Dataset available at https://github.com/rois-codh/arc-ukiyoe-faces

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our work &quot;Ukiyo-e Analysis and Creativity with Attribute and Geometry Annotation&quot; has been accepted <a href="https://twitter.com/hashtag/iccc21?src=hash&amp;ref_src=twsrc%5Etfw">#iccc21</a>!<br><br>Ukiyo-e paintings with labelled attributes and automatically extracted face landmarks, allowing quantitative analysis and fun ML experiments. <a href="https://t.co/dDOL216N5Z">https://t.co/dDOL216N5Z</a> <a href="https://t.co/CCuiK44OOZ">pic.twitter.com/CCuiK44OOZ</a></p>&mdash; Yingtao Tian (@alanyttian) <a href="https://twitter.com/alanyttian/status/1401720872713420801?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Light Field Networks: Neural Scene Representations with  Single-Evaluation Rendering

Vincent Sitzmann, Semon Rezchikov, William T. Freeman, Joshua B. Tenenbaum, Fredo Durand

- retweets: 1084, favorites: 230 (06/08/2021 09:10:33)

- links: [abs](https://arxiv.org/abs/2106.02634) | [pdf](https://arxiv.org/pdf/2106.02634)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Inferring representations of 3D scenes from 2D observations is a fundamental problem of computer graphics, computer vision, and artificial intelligence. Emerging 3D-structured neural scene representations are a promising approach to 3D scene understanding. In this work, we propose a novel neural scene representation, Light Field Networks or LFNs, which represent both geometry and appearance of the underlying 3D scene in a 360-degree, four-dimensional light field parameterized via a neural implicit representation. Rendering a ray from an LFN requires only a *single* network evaluation, as opposed to hundreds of evaluations per ray for ray-marching or volumetric based renderers in 3D-structured neural scene representations. In the setting of simple scenes, we leverage meta-learning to learn a prior over LFNs that enables multi-view consistent light field reconstruction from as little as a single image observation. This results in dramatic reductions in time and memory complexity, and enables real-time rendering. The cost of storing a 360-degree light field via an LFN is two orders of magnitude lower than conventional methods such as the Lumigraph. Utilizing the analytical differentiability of neural implicit representations and a novel parameterization of light space, we further demonstrate the extraction of sparse depth maps from LFNs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering <a href="https://t.co/inQAm1tbd5">https://t.co/inQAm1tbd5</a> <br><br>New work based on neural 3D representations that shows how to perform really fast “single-evaluation” rendering. <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://twitter.com/hashtag/deeplearning?src=hash&amp;ref_src=twsrc%5Etfw">#deeplearning</a> <a href="https://twitter.com/hashtag/3D?src=hash&amp;ref_src=twsrc%5Etfw">#3D</a> <a href="https://t.co/IQ6zQJzSmh">pic.twitter.com/IQ6zQJzSmh</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1401719299430141960?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering<br>pdf: <a href="https://t.co/WqXVBY51yA">https://t.co/WqXVBY51yA</a><br>abs: <a href="https://t.co/q45R1v06TH">https://t.co/q45R1v06TH</a><br><br>neural scene representation directly parameterizes the full 360-degree, 4D light field of a 3D scene via a neural implicit representation <a href="https://t.co/fENMzAlPWq">pic.twitter.com/fENMzAlPWq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401723108759773187?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Solving Schrödinger Bridges via Maximum Likelihood

Francisco Vargas, Pierre Thodoroff, Neil D. Lawrence, Austen Lamacraft

- retweets: 964, favorites: 175 (06/08/2021 09:10:34)

- links: [abs](https://arxiv.org/abs/2106.02081) | [pdf](https://arxiv.org/pdf/2106.02081)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The Schr\"odinger bridge problem (SBP) finds the most likely stochastic evolution between two probability distributions given a prior stochastic evolution. As well as applications in the natural sciences, problems of this kind have important applications in machine learning such as dataset alignment and hypothesis testing. Whilst the theory behind this problem is relatively mature, scalable numerical recipes to estimate the Schr\"odinger bridge remain an active area of research. We prove an equivalence between the SBP and maximum likelihood estimation enabling direct application of successful machine learning techniques. We propose a numerical procedure to estimate SBPs using Gaussian process and demonstrate the practical usage of our approach in numerical simulations and experiments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Solving Schrödinger bridges via maximum likelihood arxiv: <a href="https://t.co/AhcnXhFOvL">https://t.co/AhcnXhFOvL</a> <br><br>We propose an approximate IPFP/Sinkhorn variant based on the time reveral of diffusions with the goal of learning meaningful interpolating dynamics between two distributions. <a href="https://t.co/47uekAkQLg">pic.twitter.com/47uekAkQLg</a></p>&mdash; Neil Lawrence (@lawrennd) <a href="https://twitter.com/lawrennd/status/1401945316979511296?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Few-Shot Segmentation via Cycle-Consistent Transformer

Gengwei Zhang, Guoliang Kang, Yunchao Wei, Yi Yang

- retweets: 483, favorites: 85 (06/08/2021 09:10:34)

- links: [abs](https://arxiv.org/abs/2106.02320) | [pdf](https://arxiv.org/pdf/2106.02320)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Few-shot segmentation aims to train a segmentation model that can fast adapt to novel classes with few exemplars. The conventional training paradigm is to learn to make predictions on query images conditioned on the features from support images. Previous methods only utilized the semantic-level prototypes of support images as the conditional information. These methods cannot utilize all pixel-wise support information for the query predictions, which is however critical for the segmentation task. In this paper, we focus on utilizing pixel-wise relationships between support and target images to facilitate the few-shot semantic segmentation task. We design a novel Cycle-Consistent Transformer (CyCTR) module to aggregate pixel-wise support features into query ones. CyCTR performs cross-attention between features from different images, i.e. support and query images. We observe that there may exist unexpected irrelevant pixel-level support features. Directly performing cross-attention may aggregate these features from support to query and bias the query features. Thus, we propose using a novel cycle-consistent attention mechanism to filter out possible harmful support features and encourage query features to attend to the most informative pixels from support images. Experiments on all few-shot segmentation benchmarks demonstrate that our proposed CyCTR leads to remarkable improvement compared to previous state-of-the-art methods. Specifically, on Pascal-$5^i$ and COCO-$20^i$ datasets, we achieve 66.6% and 45.6% mIoU for 5-shot segmentation, outperforming previous state-of-the-art by 4.6% and 7.1% respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Few-Shot Segmentation via Cycle-Consistent Transformer<br>pdf: <a href="https://t.co/2O4fUUFuSY">https://t.co/2O4fUUFuSY</a><br>abs: <a href="https://t.co/f6lirYQG1d">https://t.co/f6lirYQG1d</a><br><br>on Pascal-5i and COCO-20i datasets, achieve 66.6% and 45.6% mIoU for 5-shot segmentation, outperforming previous sota by 4.6% and 7.1% respectively <a href="https://t.co/b8zxCrlktm">pic.twitter.com/b8zxCrlktm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401761732088025093?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Detecting and Adapting to Novelty in Games

Xiangyu Peng, Jonathan C. Balloch, Mark O. Riedl

- retweets: 231, favorites: 77 (06/08/2021 09:10:34)

- links: [abs](https://arxiv.org/abs/2106.02204) | [pdf](https://arxiv.org/pdf/2106.02204)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Open-world novelty occurs when the rules of an environment can change abruptly, such as when a game player encounters "house rules". To address open-world novelty, game playing agents must be able to detect when novelty is injected, and to quickly adapt to the new rules. We propose a model-based reinforcement learning approach where game state and rules are represented as knowledge graphs. The knowledge graph representation of the state and rules allows novelty to be detected as changes in the knowledge graph, assists with the training of deep reinforcement learners, and enables imagination-based re-training where the agent uses the knowledge graph to perform look-ahead.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Detecting and Adapting to Novelty in Games<br>pdf: <a href="https://t.co/xqLavTbywA">https://t.co/xqLavTbywA</a><br>abs: <a href="https://t.co/rAaHfc0H4W">https://t.co/rAaHfc0H4W</a><br><br>model-based reinforcement learning approach where game state and rules are represented as knowledge graphs <a href="https://t.co/92N1grQIzn">pic.twitter.com/92N1grQIzn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401732103415402500?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. RL-DARTS: Differentiable Architecture Search for Reinforcement Learning

Yingjie Miao, Xingyou Song, Daiyi Peng, Summer Yue, Eugene Brevdo, Aleksandra Faust

- retweets: 164, favorites: 54 (06/08/2021 09:10:34)

- links: [abs](https://arxiv.org/abs/2106.02229) | [pdf](https://arxiv.org/pdf/2106.02229)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce RL-DARTS, one of the first applications of Differentiable Architecture Search (DARTS) in reinforcement learning (RL) to search for convolutional cells, applied to the Procgen benchmark. We outline the initial difficulties of applying neural architecture search techniques in RL, and demonstrate that by simply replacing the image encoder with a DARTS supernet, our search method is sample-efficient, requires minimal extra compute resources, and is also compatible with off-policy and on-policy RL algorithms, needing only minor changes in preexisting code. Surprisingly, we find that the supernet can be used as an actor for inference to generate replay data in standard RL training loops, and thus train end-to-end. Throughout this training process, we show that the supernet gradually learns better cells, leading to alternative architectures which can be highly competitive against manually designed policies, but also verify previous design choices for RL policies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RL-DARTS: Differentiable Architecture Search for Reinforcement Learning<br>pdf: <a href="https://t.co/53XSc0o0lR">https://t.co/53XSc0o0lR</a><br>abs: <a href="https://t.co/hkb1YqpLka">https://t.co/hkb1YqpLka</a><br><br>one of the first applications of Differentiable Architecture Search in RL to search for convolutional cells, applied to the Procgen benchmark <a href="https://t.co/OLvGXDSEX6">pic.twitter.com/OLvGXDSEX6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401758807865364480?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Glance-and-Gaze Vision Transformer

Qihang Yu, Yingda Xia, Yutong Bai, Yongyi Lu, Alan Yuille, Wei Shen

- retweets: 156, favorites: 47 (06/08/2021 09:10:34)

- links: [abs](https://arxiv.org/abs/2106.02277) | [pdf](https://arxiv.org/pdf/2106.02277)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, there emerges a series of vision Transformers, which show superior performance with a more compact model size than conventional convolutional neural networks, thanks to the strong ability of Transformers to model long-range dependencies. However, the advantages of vision Transformers also come with a price: Self-attention, the core part of Transformer, has a quadratic complexity to the input sequence length. This leads to a dramatic increase of computation and memory cost with the increase of sequence length, thus introducing difficulties when applying Transformers to the vision tasks that require dense predictions based on high-resolution feature maps. In this paper, we propose a new vision Transformer, named Glance-and-Gaze Transformer (GG-Transformer), to address the aforementioned issues. It is motivated by the Glance and Gaze behavior of human beings when recognizing objects in natural scenes, with the ability to efficiently model both long-range dependencies and local context. In GG-Transformer, the Glance and Gaze behavior is realized by two parallel branches: The Glance branch is achieved by performing self-attention on the adaptively-dilated partitions of the input, which leads to a linear complexity while still enjoying a global receptive field; The Gaze branch is implemented by a simple depth-wise convolutional layer, which compensates local image context to the features obtained by the Glance mechanism. We empirically demonstrate our method achieves consistently superior performance over previous state-of-the-art Transformers on various vision tasks and benchmarks. The codes and models will be made available at https://github.com/yucornetto/GG-Transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Glance-and-Gaze Vision Transformer<br>pdf: <a href="https://t.co/GGcirv36Fz">https://t.co/GGcirv36Fz</a><br>abs: <a href="https://t.co/WsvVAJt6vS">https://t.co/WsvVAJt6vS</a><br><br>parallel and complementary Glance branch and Gaze branch, which offer long-range relationship and short-range modeling <a href="https://t.co/ewkYVocgIh">pic.twitter.com/ewkYVocgIh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401702871536705538?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Semantic Correspondence with Transformers

Seokju Cho, Sunghwan Hong, Sangryul Jeon, Yunsung Lee, Kwanghoon Sohn, Seungryong Kim

- retweets: 86, favorites: 58 (06/08/2021 09:10:34)

- links: [abs](https://arxiv.org/abs/2106.02520) | [pdf](https://arxiv.org/pdf/2106.02520)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a novel cost aggregation network, called Cost Aggregation with Transformers (CATs), to find dense correspondences between semantically similar images with additional challenges posed by large intra-class appearance and geometric variations. Compared to previous hand-crafted or CNN-based methods addressing the cost aggregation stage, which either lack robustness to severe deformations or inherit the limitation of CNNs that fail to discriminate incorrect matches due to limited receptive fields, CATs explore global consensus among initial correlation map with the help of some architectural designs that allow us to exploit full potential of self-attention mechanism. Specifically, we include appearance affinity modelling to disambiguate the initial correlation maps and multi-level aggregation to benefit from hierarchical feature representations within Transformer-based aggregator, and combine with swapping self-attention and residual connections not only to enforce consistent matching, but also to ease the learning process. We conduct experiments to demonstrate the effectiveness of the proposed model over the latest methods and provide extensive ablation studies. Code and trained models will be made available at https://github.com/SunghwanHong/CATs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Semantic Correspondence with Transformers<br>pdf: <a href="https://t.co/EMx0X1jxF0">https://t.co/EMx0X1jxF0</a><br>abs: <a href="https://t.co/7gQUC3Cd7d">https://t.co/7gQUC3Cd7d</a><br><br>cost aggregation network, find dense correspondences between semantically similar images with additional challenges posed by large intra-class appearance and geometric variations <a href="https://t.co/8b6sSPEpfX">pic.twitter.com/8b6sSPEpfX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401765027972108288?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. nmT5 -- Is parallel data still relevant for pre-training massively  multilingual language models?

Mihir Kale, Aditya Siddhant, Noah Constant, Melvin Johnson, Rami Al-Rfou, Linting Xue

- retweets: 56, favorites: 38 (06/08/2021 09:10:35)

- links: [abs](https://arxiv.org/abs/2106.02171) | [pdf](https://arxiv.org/pdf/2106.02171)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Recently, mT5 - a massively multilingual version of T5 - leveraged a unified text-to-text format to attain state-of-the-art results on a wide variety of multilingual NLP tasks. In this paper, we investigate the impact of incorporating parallel data into mT5 pre-training. We find that multi-tasking language modeling with objectives such as machine translation during pre-training is a straightforward way to improve performance on downstream multilingual and cross-lingual tasks. However, the gains start to diminish as the model capacity increases, suggesting that parallel data might not be as essential for larger models. At the same time, even at larger model sizes, we find that pre-training with parallel data still provides benefits in the limited labelled data regime.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">nmT5 - Is parallel data still relevant for pre-training massively multilingual language models?<br>pdf: <a href="https://t.co/KKoNuLTkPy">https://t.co/KKoNuLTkPy</a><br>abs: <a href="https://t.co/VNawWupnse">https://t.co/VNawWupnse</a><br><br>larger model sizes, pre-training with parallel data still provides benefits in the limited labelled data regime <a href="https://t.co/IHzTlm9UwY">pic.twitter.com/IHzTlm9UwY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401710383572922369?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. SOLQ: Segmenting Objects by Learning Queries

Bin Dong, Fangao Zeng, Tiancai Wang, Xiangyu Zhang, Yichen Wei

- retweets: 58, favorites: 31 (06/08/2021 09:10:35)

- links: [abs](https://arxiv.org/abs/2106.02351) | [pdf](https://arxiv.org/pdf/2106.02351)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we propose an end-to-end framework for instance segmentation. Based on the recently introduced DETR [1], our method, termed SOLQ, segments objects by learning unified queries. In SOLQ, each query represents one object and has multiple representations: class, location and mask. The object queries learned perform classification, box regression and mask encoding simultaneously in an unified vector form. During training phase, the mask vectors encoded are supervised by the compression coding of raw spatial masks. In inference time, mask vectors produced can be directly transformed to spatial masks by the inverse process of compression coding. Experimental results show that SOLQ can achieve state-of-the-art performance, surpassing most of existing approaches. Moreover, the joint learning of unified query representation can greatly improve the detection performance of original DETR. We hope our SOLQ can serve as a strong baseline for the Transformer-based instance segmentation. Code is available at https://github.com/megvii-research/SOLQ.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SOLQ: Segmenting Objects by Learning Queries<br>pdf: <a href="https://t.co/W6Y4sJvEeO">https://t.co/W6Y4sJvEeO</a><br>abs: <a href="https://t.co/4LYTbWBYi4">https://t.co/4LYTbWBYi4</a><br>github: <a href="https://t.co/ROiTSNDcho">https://t.co/ROiTSNDcho</a> <a href="https://t.co/V1GMtpbKQn">pic.twitter.com/V1GMtpbKQn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401975478383165455?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Do Syntactic Probes Probe Syntax? Experiments with Jabberwocky Probing

Rowan Hall Maudslay, Ryan Cotterell

- retweets: 44, favorites: 38 (06/08/2021 09:10:35)

- links: [abs](https://arxiv.org/abs/2106.02559) | [pdf](https://arxiv.org/pdf/2106.02559)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Analysing whether neural language models encode linguistic information has become popular in NLP. One method of doing so, which is frequently cited to support the claim that models like BERT encode syntax, is called probing; probes are small supervised models trained to extract linguistic information from another model's output. If a probe is able to predict a particular structure, it is argued that the model whose output it is trained on must have implicitly learnt to encode it. However, drawing a generalisation about a model's linguistic knowledge about a specific phenomena based on what a probe is able to learn may be problematic: in this work, we show that semantic cues in training data means that syntactic probes do not properly isolate syntax. We generate a new corpus of semantically nonsensical but syntactically well-formed Jabberwocky sentences, which we use to evaluate two probes trained on normal data. We train the probes on several popular language models (BERT, GPT, and RoBERTa), and find that in all settings they perform worse when evaluated on these data, for one probe by an average of 15.4 UUAS points absolute. Although in most cases they still outperform the baselines, their lead is reduced substantially, e.g. by 53% in the case of BERT for one probe. This begs the question: what empirical scores constitute knowing syntax?

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Some (<a href="https://twitter.com/ryandcotterell?ref_src=twsrc%5Etfw">@ryandcotterell</a>) would say that posting an arXiv link &lt;1hr before the conference started was leaving things late....BUT what the hell: <br><br>Do Syntactic Probes Probe Syntax? Experiments with Jabberwocky Probing<a href="https://t.co/p2hMeycos5">https://t.co/p2hMeycos5</a> <br>Done at <a href="https://twitter.com/CSatETH?ref_src=twsrc%5Etfw">@CSatETH</a> &amp; <a href="https://twitter.com/cambridgenlp?ref_src=twsrc%5Etfw">@cambridgenlp</a> [1/6] <a href="https://t.co/uaUmyyM1RF">pic.twitter.com/uaUmyyM1RF</a></p>&mdash; Rowan Hall Maudslay (@rowhallmauds) <a href="https://twitter.com/rowhallmauds/status/1401909289489416198?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. A Consciousness-Inspired Planning Agent for Model-Based Reinforcement  Learning

Mingde Zhao, Zhen Liu, Sitao Luan, Shuyuan Zhang, Doina Precup, Yoshua Bengio

- retweets: 44, favorites: 26 (06/08/2021 09:10:35)

- links: [abs](https://arxiv.org/abs/2106.02097) | [pdf](https://arxiv.org/pdf/2106.02097)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present an end-to-end, model-based deep reinforcement learning agent which dynamically attends to relevant parts of its state, in order to plan and to generalize better out-of-distribution. The agent's architecture uses a set representation and a bottleneck mechanism, forcing the number of entities to which the agent attends at each planning step to be small. In experiments with customized MiniGrid environments with different dynamics, we observe that the design allows agents to learn to plan effectively, by attending to the relevant objects, leading to better out-of-distribution generalization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Consciousness-Inspired Planning Agent for Model-Based Reinforcement Learning<br>pdf: <a href="https://t.co/aW21hIkEW7">https://t.co/aW21hIkEW7</a><br>abs: <a href="https://t.co/7OP9ctAHRs">https://t.co/7OP9ctAHRs</a><br><br>end-to-end, model-based DRL agent which dynamically attends to relevant parts of its state, in order to plan and<br>to generalize better ood <a href="https://t.co/ZN7jHIS38Z">pic.twitter.com/ZN7jHIS38Z</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401707049029804039?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Fundamental tradeoffs between memorization and robustness in random  features and neural tangent regimes

Elvis Dohmatob

- retweets: 30, favorites: 34 (06/08/2021 09:10:35)

- links: [abs](https://arxiv.org/abs/2106.02630) | [pdf](https://arxiv.org/pdf/2106.02630)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This work studies the (non)robustness of two-layer neural networks in various high-dimensional linearized regimes. We establish fundamental trade-offs between memorization and robustness, as measured by the Sobolev-seminorm of the model w.r.t the data distribution, i.e the square root of the average squared $L_2$-norm of the gradients of the model w.r.t the its input. More precisely, if $n$ is the number of training examples, $d$ is the input dimension, and $k$ is the number of hidden neurons in a two-layer neural network, we prove for a large class of activation functions that, if the model memorizes even a fraction of the training, then its Sobolev-seminorm is lower-bounded by (i) $\sqrt{n}$ in case of infinite-width random features (RF) or neural tangent kernel (NTK) with $d \gtrsim n$; (ii) $\sqrt{n}$ in case of finite-width RF with proportionate scaling of $d$ and $k$; and (iii) $\sqrt{n/k}$ in case of finite-width NTK with proportionate scaling of $d$ and $k$. Moreover, all of these lower-bounds are tight: they are attained by the min-norm / least-squares interpolator (when $n$, $d$, and $k$ are in the appropriate interpolating regime). All our results hold as soon as data is log-concave isotropic, and there is label-noise, i.e the target variable is not a deterministic function of the data / features. We empirically validate our theoretical results with experiments. Accidentally, these experiments also reveal for the first time, (iv) a multiple-descent phenomenon in the robustness of the min-norm interpolator.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/ NEW preprint <a href="https://t.co/aIMxhTPoDf">https://t.co/aIMxhTPoDf</a> wherein we uncover a fundamental tradeoff between memorization and robustness for NNs in linearized regimes (RF, NTK, ...). Also, we accidentally observe, for the first time (it seems), a multiple-descent phenomenon in robustness <a href="https://t.co/mpGVKoyi4e">pic.twitter.com/mpGVKoyi4e</a></p>&mdash; Elvis Dohmatob (@dohmatobelvis) <a href="https://twitter.com/dohmatobelvis/status/1401858375902707713?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Eliciting Spoken Interruptions to Inform Proactive Speech Agent Design

Justin Edwards, Christian Janssen, Sandy Gould, Benjamin R Cowan

- retweets: 42, favorites: 21 (06/08/2021 09:10:35)

- links: [abs](https://arxiv.org/abs/2106.02077) | [pdf](https://arxiv.org/pdf/2106.02077)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

Current speech agent interactions are typically user-initiated, limiting the interactions they can deliver. Future functionality will require agents to be proactive, sometimes interrupting users. Little is known about how these spoken interruptions should be designed, especially in urgent interruption contexts. We look to inform design of proactive agent interruptions through investigating how people interrupt others engaged in complex tasks. We therefore developed a new technique to elicit human spoken interruptions of people engaged in other tasks. We found that people interrupted sooner when interruptions were urgent. Some participants used access rituals to forewarn interruptions, but most rarely used them. People balanced speed and accuracy in timing interruptions, often using cues from the task they interrupted. People also varied phrasing and delivery of interruptions to reflect urgency. We discuss how our findings can inform speech agent design and how our paradigm can help gain insight into human interruptions in new contexts.




# 19. The Image Local Autoregressive Transformer

Chenjie Cao, Yuxin Hong, Xiang Li, Chengrong Wang, Chengming Xu, XiangYang Xue, Yanwei Fu

- retweets: 20, favorites: 33 (06/08/2021 09:10:35)

- links: [abs](https://arxiv.org/abs/2106.02514) | [pdf](https://arxiv.org/pdf/2106.02514)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Recently, AutoRegressive (AR) models for the whole image generation empowered by transformers have achieved comparable or even better performance to Generative Adversarial Networks (GANs). Unfortunately, directly applying such AR models to edit/change local image regions, may suffer from the problems of missing global information, slow inference speed, and information leakage of local guidance. To address these limitations, we propose a novel model -- image Local Autoregressive Transformer (iLAT), to better facilitate the locally guided image synthesis. Our iLAT learns the novel local discrete representations, by the newly proposed local autoregressive (LA) transformer of the attention mask and convolution mechanism. Thus iLAT can efficiently synthesize the local image regions by key guidance information. Our iLAT is evaluated on various locally guided image syntheses, such as pose-guided person image synthesis and face editing. Both the quantitative and qualitative results show the efficacy of our model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Image Local Autoregressive Transformer<br>pdf: <a href="https://t.co/Ldk53mswBh">https://t.co/Ldk53mswBh</a><br>abs: <a href="https://t.co/frzJ3ZaNgR">https://t.co/frzJ3ZaNgR</a><br><br>learns the novel local discrete representations, by the newly proposed local autoregressive transformer<br>of the attention mask and convolution mechanism <a href="https://t.co/uOCch4f6qF">pic.twitter.com/uOCch4f6qF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1401705331189768196?ref_src=twsrc%5Etfw">June 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



