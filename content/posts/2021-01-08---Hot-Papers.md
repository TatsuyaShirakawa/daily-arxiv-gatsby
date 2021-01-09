---
title: Hot Papers 2021-01-08
date: 2021-01-09T23:42:13.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-08"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-08"
socialImage: "/media/flying-marine.jpg"

---

# 1. VOGUE: Try-On by StyleGAN Interpolation Optimization

Kathleen M Lewis, Srivatsan Varadharajan, Ira Kemelmacher-Shlizerman

- retweets: 6770, favorites: 227 (01/09/2021 23:42:13)

- links: [abs](https://arxiv.org/abs/2101.02285) | [pdf](https://arxiv.org/pdf/2101.02285)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Given an image of a target person and an image of another person wearing a garment, we automatically generate the target person in the given garment. At the core of our method is a pose-conditioned StyleGAN2 latent space interpolation, which seamlessly combines the areas of interest from each image, i.e., body shape, hair, and skin color are derived from the target person, while the garment with its folds, material properties, and shape comes from the garment image. By automatically optimizing for interpolation coefficients per layer in the latent space, we can perform a seamless, yet true to source, merging of the garment and target person. Our algorithm allows for garments to deform according to the given body shape, while preserving pattern and material details. Experiments demonstrate state-of-the-art photo-realistic results at high resolution ($512\times 512$).

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">ほんとヴァーチャル試着のクオリティ高くなったなぁ。こういうのが発展していくと、試着の手間がかなり省けるな<br><br>「試着させたい人物の画像」と「その服を着てる他人の画像」を与えると、試着したときの画像を生成するニューラルネット<br><br>VOGUE<a href="https://t.co/MdiKKBuSM7">https://t.co/MdiKKBuSM7</a><a href="https://t.co/ofqwZzNQGD">pic.twitter.com/ofqwZzNQGD</a></p>&mdash; 小猫遊りょう（たかにゃし・りょう） (@jaguring1) <a href="https://twitter.com/jaguring1/status/1347711248134139905?ref_src=twsrc%5Etfw">January 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Information-theoretic bounds on quantum advantage in machine learning

Hsin-Yuan Huang, Richard Kueng, John Preskill

- retweets: 4224, favorites: 261 (01/09/2021 23:43:29)

- links: [abs](https://arxiv.org/abs/2101.02464) | [pdf](https://arxiv.org/pdf/2101.02464)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We study the complexity of training classical and quantum machine learning (ML) models for predicting outcomes of physical experiments. The experiments depend on an input parameter $x$ and involve the execution of a (possibly unknown) quantum process $\mathcal{E}$. Our figure of merit is the number of runs of $\mathcal{E}$ during training, disregarding other measures of runtime. A classical ML model performs a measurement and records the classical outcome after each run of $\mathcal{E}$, while a quantum ML model can access $\mathcal{E}$ coherently to acquire quantum data; the classical or quantum data is then used to predict outcomes of future experiments. We prove that, for any input distribution $\mathcal{D}(x)$, a classical ML model can provide accurate predictions on average by accessing $\mathcal{E}$ a number of times comparable to the optimal quantum ML model. In contrast, for achieving accurate prediction on all inputs, we show that exponential quantum advantage is possible for certain tasks. For example, to predict expectation values of all Pauli observables in an $n$-qubit system $\rho$, we present a quantum ML model using only $\mathcal{O}(n)$ copies of $\rho$ and prove that classical ML models require $2^{\Omega(n)}$ copies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper with <a href="https://twitter.com/RobertHuangHY?ref_src=twsrc%5Etfw">@RobertHuangHY</a> and Richard Kueng, we compare quantum and classical machine learning for predicting outcomes of quantum experiments.  /1<a href="https://t.co/J1nny03bW5">https://t.co/J1nny03bW5</a> <a href="https://t.co/baDCWBkE4R">pic.twitter.com/baDCWBkE4R</a></p>&mdash; John Preskill (@preskill) <a href="https://twitter.com/preskill/status/1347363829525536768?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Disentangling homophily, community structure and triadic closure in  networks

Tiago P. Peixoto

- retweets: 1784, favorites: 201 (01/09/2021 23:43:29)

- links: [abs](https://arxiv.org/abs/2101.02510) | [pdf](https://arxiv.org/pdf/2101.02510)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.data-an](https://arxiv.org/list/physics.data-an/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Network homophily, the tendency of similar nodes to be connected, and transitivity, the tendency of two nodes being connected if they share a common neighbor, are conflated properties in network analysis, since one mechanism can drive the other. Here we present a generative model and corresponding inference procedure that is capable of distinguishing between both mechanisms. Our approach is based on a variation of the stochastic block model (SBM) with the addition of triadic closure edges, and its inference can identify the most plausible mechanism responsible for the existence of every edge in the network, in addition to the underlying community structure itself. We show how the method can evade the detection of spurious communities caused solely by the formation of triangles in the network, and how it can improve the performance of link prediction when compared to the pure version of the SBM without triadic closure.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New on the <a href="https://twitter.com/arxiv?ref_src=twsrc%5Etfw">@arxiv</a>! &quot;Disentangling homophily, community structure and triadic closure in networks&quot;, <a href="https://t.co/lnBlOQ4EX1">https://t.co/lnBlOQ4EX1</a><br><br>Homophily/communities and triadic closure (triangles) are conflated properties in network analysis, and this method tells them apart. An explainer: 1/5 <a href="https://t.co/tnY0svm0Ug">pic.twitter.com/tnY0svm0Ug</a></p>&mdash; Tiago Peixoto (@tiagopeixoto) <a href="https://twitter.com/tiagopeixoto/status/1347513720424034305?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. GAN-Control: Explicitly Controllable GANs

Alon Shoshan, Nadav Bhonker, Igor Kviatkovsky, Gerard Medioni

- retweets: 1726, favorites: 211 (01/09/2021 23:43:29)

- links: [abs](https://arxiv.org/abs/2101.02477) | [pdf](https://arxiv.org/pdf/2101.02477)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a framework for training GANs with explicit control over generated images. We are able to control the generated image by settings exact attributes such as age, pose, expression, etc. Most approaches for editing GAN-generated images achieve partial control by leveraging the latent space disentanglement properties, obtained implicitly after standard GAN training. Such methods are able to change the relative intensity of certain attributes, but not explicitly set their values. Recently proposed methods, designed for explicit control over human faces, harness morphable 3D face models to allow fine-grained control capabilities in GANs. Unlike these methods, our control is not constrained to morphable 3D face model parameters and is extendable beyond the domain of human faces. Using contrastive learning, we obtain GANs with an explicitly disentangled latent space. This disentanglement is utilized to train control-encoders mapping human-interpretable inputs to suitable latent vectors, thus allowing explicit control. In the domain of human faces we demonstrate control over identity, age, pose, expression, hair color and illumination. We also demonstrate control capabilities of our framework in the domains of painted portraits and dog image generation. We demonstrate that our approach achieves state-of-the-art performance both qualitatively and quantitatively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GAN-Control: Explicitly Controllable GANs<br>pdf: <a href="https://t.co/ZtZrcGLB3L">https://t.co/ZtZrcGLB3L</a><br>abs: <a href="https://t.co/C6mMyiLAQ3">https://t.co/C6mMyiLAQ3</a><br>project page: <a href="https://t.co/nvf0IMzzey">https://t.co/nvf0IMzzey</a> <a href="https://t.co/gup78d64k0">pic.twitter.com/gup78d64k0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347368572557914112?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Coding for Distributed Multi-Agent Reinforcement Learning

Baoqian Wang, Junfei Xie, Nikolay Atanasov

- retweets: 1381, favorites: 36 (01/09/2021 23:43:30)

- links: [abs](https://arxiv.org/abs/2101.02308) | [pdf](https://arxiv.org/pdf/2101.02308)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper aims to mitigate straggler effects in synchronous distributed learning for multi-agent reinforcement learning (MARL) problems. Stragglers arise frequently in a distributed learning system, due to the existence of various system disturbances such as slow-downs or failures of compute nodes and communication bottlenecks. To resolve this issue, we propose a coded distributed learning framework, which speeds up the training of MARL algorithms in the presence of stragglers, while maintaining the same accuracy as the centralized approach. As an illustration, a coded distributed version of the multi-agent deep deterministic policy gradient(MADDPG) algorithm is developed and evaluated. Different coding schemes, including maximum distance separable (MDS)code, random sparse code, replication-based code, and regular low density parity check (LDPC) code are also investigated. Simulations in several multi-robot problems demonstrate the promising performance of the proposed framework.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Coding for Distributed Multi-Agent Reinforcement Learning.<a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/DevCommunity?src=hash&amp;ref_src=twsrc%5Etfw">#DevCommunity</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Cloud?src=hash&amp;ref_src=twsrc%5Etfw">#Cloud</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/IIoT?src=hash&amp;ref_src=twsrc%5Etfw">#IIoT</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/javascript?src=hash&amp;ref_src=twsrc%5Etfw">#javascript</a> <a href="https://twitter.com/hashtag/womenwhocode?src=hash&amp;ref_src=twsrc%5Etfw">#womenwhocode</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/Robotics?src=hash&amp;ref_src=twsrc%5Etfw">#Robotics</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a><a href="https://t.co/boKiGpo2dz">https://t.co/boKiGpo2dz</a> <a href="https://t.co/P6WJLQ0KEq">pic.twitter.com/P6WJLQ0KEq</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1347735560056668165?ref_src=twsrc%5Etfw">January 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. TrackFormer: Multi-Object Tracking with Transformers

Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, Christoph Feichtenhofer

- retweets: 1191, favorites: 171 (01/09/2021 23:43:30)

- links: [abs](https://arxiv.org/abs/2101.02702) | [pdf](https://arxiv.org/pdf/2101.02702)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present TrackFormer, an end-to-end multi-object tracking and segmentation model based on an encoder-decoder Transformer architecture. Our approach introduces track query embeddings which follow objects through a video sequence in an autoregressive fashion. New track queries are spawned by the DETR object detector and embed the position of their corresponding object over time. The Transformer decoder adjusts track query embeddings from frame to frame, thereby following the changing object positions. TrackFormer achieves a seamless data association between frames in a new tracking-by-attention paradigm by self- and encoder-decoder attention mechanisms which simultaneously reason about location, occlusion, and object identity. TrackFormer yields state-of-the-art performance on the tasks of multi-object tracking (MOT17) and segmentation (MOTS20). We hope our unified way of performing detection and tracking will foster future research in multi-object tracking and video understanding. Code will be made publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TrackFormer: Multi-Object Tracking with Transformers<br>pdf: <a href="https://t.co/83rx5tbkig">https://t.co/83rx5tbkig</a><br>abs: <a href="https://t.co/BCJPohxZ6i">https://t.co/BCJPohxZ6i</a> <a href="https://t.co/Ixg0wlTI1U">pic.twitter.com/Ixg0wlTI1U</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347360802395795456?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit  Reasoning Strategies

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, Jonathan Berant

- retweets: 934, favorites: 142 (01/09/2021 23:43:30)

- links: [abs](https://arxiv.org/abs/2101.02235) | [pdf](https://arxiv.org/pdf/2101.02235)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

A key limitation in current datasets for multi-hop reasoning is that the required steps for answering the question are mentioned in it explicitly. In this work, we introduce StrategyQA, a question answering (QA) benchmark where the required reasoning steps are implicit in the question, and should be inferred using a strategy. A fundamental challenge in this setup is how to elicit such creative questions from crowdsourcing workers, while covering a broad range of potential strategies. We propose a data collection procedure that combines term-based priming to inspire annotators, careful control over the annotator population, and adversarial filtering for eliminating reasoning shortcuts. Moreover, we annotate each question with (1) a decomposition into reasoning steps for answering it, and (2) Wikipedia paragraphs that contain the answers to each step. Overall, StrategyQA includes 2,780 examples, each consisting of a strategy question, its decomposition, and evidence paragraphs. Analysis shows that questions in StrategyQA are short, topic-diverse, and cover a wide range of strategies. Empirically, we show that humans perform well (87%) on this task, while our best baseline reaches an accuracy of $\sim$66%.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We present StrategyQA, a question answering benchmark with *implicit* reasoning strategies, accepted to TACL, 2021.<br>Dataset --&gt; <a href="https://t.co/xr1erD7mYS">https://t.co/xr1erD7mYS</a><br>Paper --&gt; <a href="https://t.co/TqLeoAIL7i">https://t.co/TqLeoAIL7i</a><br><br>With <a href="https://twitter.com/DanielKhashabi?ref_src=twsrc%5Etfw">@DanielKhashabi</a> <a href="https://twitter.com/EladSegal?ref_src=twsrc%5Etfw">@EladSegal</a> <a href="https://twitter.com/tusharkhot?ref_src=twsrc%5Etfw">@tusharkhot</a>  <a href="https://twitter.com/dannydanr?ref_src=twsrc%5Etfw">@dannydanr</a> <a href="https://twitter.com/JonathanBerant?ref_src=twsrc%5Etfw">@JonathanBerant</a> <a href="https://t.co/VI2Gix3PHu">pic.twitter.com/VI2Gix3PHu</a></p>&mdash; Mor Geva (@megamor2) <a href="https://twitter.com/megamor2/status/1347525449883275266?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Distribution-Free, Risk-Controlling Prediction Sets

Stephen Bates, Anastasios Angelopoulos, Lihua Lei, Jitendra Malik, Michael I. Jordan

- retweets: 528, favorites: 86 (01/09/2021 23:43:30)

- links: [abs](https://arxiv.org/abs/2101.02703) | [pdf](https://arxiv.org/pdf/2101.02703)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

To communicate instance-wise uncertainty for prediction tasks, we show how to generate set-valued predictions for black-box predictors that control the expected loss on future test points at a user-specified level. Our approach provides explicit finite-sample guarantees for any dataset by using a holdout set to calibrate the size of the prediction sets. This framework enables simple, distribution-free, rigorous error control for many tasks, and we demonstrate it in five large-scale machine learning problems: (1) classification problems where some mistakes are more costly than others; (2) multi-label classification, where each observation has multiple associated labels; (3) classification problems where the labels have a hierarchical structure; (4) image segmentation, where we wish to predict a set of pixels containing an object of interest; and (5) protein structure prediction. Lastly, we discuss extensions to uncertainty quantification for ranking, metric learning and distributionally robust learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new work <a href="https://t.co/WdSowP9Iwt">https://t.co/WdSowP9Iwt</a>! We propose a framework, inspired by conformal inference, that is able to control risk for any <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> algorithms in finite samples for iid data w/o distributional assumptions! <a href="https://twitter.com/hashtag/statstwitter?src=hash&amp;ref_src=twsrc%5Etfw">#statstwitter</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://twitter.com/hashtag/deeplearning?src=hash&amp;ref_src=twsrc%5Etfw">#deeplearning</a> <a href="https://twitter.com/hashtag/ai?src=hash&amp;ref_src=twsrc%5Etfw">#ai</a> <a href="https://t.co/LuUHZctkOQ">https://t.co/LuUHZctkOQ</a></p>&mdash; Lihua Lei (@lihua_lei_stat) <a href="https://twitter.com/lihua_lei_stat/status/1347624066971099136?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Who's a Good Boy? Reinforcing Canine Behavior using Machine Learning in  Real-Time

Jason Stock, Tom Cavey

- retweets: 390, favorites: 99 (01/09/2021 23:43:31)

- links: [abs](https://arxiv.org/abs/2101.02380) | [pdf](https://arxiv.org/pdf/2101.02380)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper we outline the development methodology for an automatic dog treat dispenser which combines machine learning and embedded hardware to identify and reward dog behaviors in real-time. Using machine learning techniques for training an image classification model we identify three behaviors of our canine companions: "sit", "stand", and "lie down" with up to 92% test accuracy and 39 frames per second. We evaluate a variety of neural network architectures, interpretability methods, model quantization and optimization techniques to develop a model specifically for an NVIDIA Jetson Nano. We detect the aforementioned behaviors in real-time and reinforce positive actions by making inference on the Jetson Nano and transmitting a signal to a servo motor to release rewards from a treat delivery apparatus.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Who&#39;s a Good Boy? Reinforcing Canine Behavior using Machine Learning in Real-Time,&quot; <a href="https://t.co/MSjkFjs5aF">https://t.co/MSjkFjs5aF</a><br><br>👀</p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1347388900310847493?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Compound Word Transformer: Learning to Compose Full-Song Music over  Dynamic Directed Hypergraphs

Wen-Yi Hsiao, Jen-Yu Liu, Yin-Cheng Yeh, Yi-Hsuan Yang

- retweets: 210, favorites: 76 (01/09/2021 23:43:31)

- links: [abs](https://arxiv.org/abs/2101.02402) | [pdf](https://arxiv.org/pdf/2101.02402)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

To apply neural sequence models such as the Transformers to music generation tasks, one has to represent a piece of music by a sequence of tokens drawn from a finite set of pre-defined vocabulary. Such a vocabulary usually involves tokens of various types. For example, to describe a musical note, one needs separate tokens to indicate the note's pitch, duration, velocity (dynamics), and placement (onset time) along the time grid. While different types of tokens may possess different properties, existing models usually treat them equally, in the same way as modeling words in natural languages. In this paper, we present a conceptually different approach that explicitly takes into account the type of the tokens, such as note types and metric types. And, we propose a new Transformer decoder architecture that uses different feed-forward heads to model tokens of different types. With an expansion-compression trick, we convert a piece of music to a sequence of compound words by grouping neighboring tokens, greatly reducing the length of the token sequences. We show that the resulting model can be viewed as a learner over dynamic directed hypergraphs. And, we employ it to learn to compose expressive Pop piano music of full-song length (involving up to 10K individual tokens per song), both conditionally and unconditionally. Our experiment shows that, compared to state-of-the-art models, the proposed model converges 5--10 times faster at training (i.e., within a day on a single GPU with 11 GB memory), and with comparable quality in the generated music.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Our AAAI&#39;21 paper, &quot;Compound Word Transformer&quot;, is finally on arxiv (+ pytorch code + data)!<br>- paper: <a href="https://t.co/10friglKlH">https://t.co/10friglKlH</a><br>- code: <a href="https://t.co/CGCzh7ebp8">https://t.co/CGCzh7ebp8</a><br>- blog: <a href="https://t.co/VYFUCCbJjv">https://t.co/VYFUCCbJjv</a> <a href="https://t.co/kJLKa5nGp9">https://t.co/kJLKa5nGp9</a></p>&mdash; Yi-Hsuan Yang (@affige_yang) <a href="https://twitter.com/affige_yang/status/1347406553700212736?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs<br>pdf: <a href="https://t.co/CJEKuNyv4U">https://t.co/CJEKuNyv4U</a><br>abs: <a href="https://t.co/bTVWgLHCxf">https://t.co/bTVWgLHCxf</a> <a href="https://t.co/4dF5RcnAjk">pic.twitter.com/4dF5RcnAjk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347359650249891840?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Towards Meaningful Statements in IR Evaluation. Mapping Evaluation  Measures to Interval Scales

Marco Ferrante, Nicola Ferro, Norbert Fuhr

- retweets: 158, favorites: 52 (01/09/2021 23:43:31)

- links: [abs](https://arxiv.org/abs/2101.02668) | [pdf](https://arxiv.org/pdf/2101.02668)
- [cs.IR](https://arxiv.org/list/cs.IR/recent)

Recently, it was shown that most popular IR measures are not interval-scaled, implying that decades of experimental IR research used potentially improper methods, which may have produced questionable results. However, it was unclear if and to what extent these findings apply to actual evaluations and this opened a debate in the community with researchers standing on opposite positions about whether this should be considered an issue (or not) and to what extent.   In this paper, we first give an introduction to the representational measurement theory explaining why certain operations and significance tests are permissible only with scales of a certain level. For that, we introduce the notion of meaningfulness specifying the conditions under which the truth (or falsity) of a statement is invariant under permissible transformations of a scale. Furthermore, we show how the recall base and the length of the run may make comparison and aggregation across topics problematic.   Then we propose a straightforward and powerful approach for turning an evaluation measure into an interval scale, and describe an experimental evaluation of the differences between using the original measures and the interval-scaled ones.   For all the regarded measures - namely Precision, Recall, Average Precision, (Normalized) Discounted Cumulative Gain, Rank-Biased Precision and Reciprocal Rank - we observe substantial effects, both on the order of average values and on the outcome of significance tests. For the latter, previously significant differences turn out to be insignificant, while insignificant ones become significant. The effect varies remarkably between the tests considered but overall, on average, we observed a 25% change in the decision about which systems are significantly different and which are not.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our comprehensive answer to the IR measures problem: <a href="https://t.co/NiUNzA82vg">https://t.co/NiUNzA82vg</a>. 1. Everything but P@k is flawed. 2. Avoid Recall-based measures (R@k, AP, nDCG). 3. For all others, apply our transformation method.</p>&mdash; Norbert Fuhr (@NorbertFuhr) <a href="https://twitter.com/NorbertFuhr/status/1347472192242667520?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Machine learning dismantling and early-warning signals of disintegration  in complex systems

Marco Grassia, Manlio De Domenico, Giuseppe Mangioni

- retweets: 142, favorites: 37 (01/09/2021 23:43:31)

- links: [abs](https://arxiv.org/abs/2101.02453) | [pdf](https://arxiv.org/pdf/2101.02453)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

From physics to engineering, biology and social science, natural and artificial systems are characterized by interconnected topologies whose features - e.g., heterogeneous connectivity, mesoscale organization, hierarchy - affect their robustness to external perturbations, such as targeted attacks to their units. Identifying the minimal set of units to attack to disintegrate a complex network, i.e. network dismantling, is a computationally challenging (NP-hard) problem which is usually attacked with heuristics. Here, we show that a machine trained to dismantle relatively small systems is able to identify higher-order topological patterns, allowing to disintegrate large-scale social, infrastructural and technological networks more efficiently than human-based heuristics. Remarkably, the machine assesses the probability that next attacks will disintegrate the system, providing a quantitative method to quantify systemic risk and detect early-warning signals of system's collapse. This demonstrates that machine-assisted analysis can be effectively used for policy and decision making to better quantify the fragility of complex systems and their response to shocks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share our last work &quot;Machine learning dismantling and early-warning signals of disintegration in complex systems&quot;, with <a href="https://twitter.com/marco__grassia?ref_src=twsrc%5Etfw">@marco__grassia</a> and <a href="https://twitter.com/manlius84?ref_src=twsrc%5Etfw">@manlius84</a>, out today on arXiv (<a href="https://t.co/LeRVPVULEx">https://t.co/LeRVPVULEx</a>). <a href="https://t.co/IMFNAXtbnQ">pic.twitter.com/IMFNAXtbnQ</a></p>&mdash; Giuseppe Mangioni (@MangioniG) <a href="https://twitter.com/MangioniG/status/1347468767282999296?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. PVA: Pixel-aligned Volumetric Avatars

Amit Raj, Michael Zollhoefer, Tomas Simon, Jason Saragih, Shunsuke Saito, James Hays, Stephen Lombardi

- retweets: 110, favorites: 51 (01/09/2021 23:43:31)

- links: [abs](https://arxiv.org/abs/2101.02697) | [pdf](https://arxiv.org/pdf/2101.02697)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Acquisition and rendering of photo-realistic human heads is a highly challenging research problem of particular importance for virtual telepresence. Currently, the highest quality is achieved by volumetric approaches trained in a person specific manner on multi-view data. These models better represent fine structure, such as hair, compared to simpler mesh-based models. Volumetric models typically employ a global code to represent facial expressions, such that they can be driven by a small set of animation parameters. While such architectures achieve impressive rendering quality, they can not easily be extended to the multi-identity setting. In this paper, we devise a novel approach for predicting volumetric avatars of the human head given just a small number of inputs. We enable generalization across identities by a novel parameterization that combines neural radiance fields with local, pixel-aligned features extracted directly from the inputs, thus sidestepping the need for very deep or complex networks. Our approach is trained in an end-to-end manner solely based on a photometric re-rendering loss without requiring explicit 3D supervision.We demonstrate that our approach outperforms the existing state of the art in terms of quality and is able to generate faithful facial expressions in a multi-identity setting.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">PVA: Pixel-aligned Volumetric Avatars<br>pdf: <a href="https://t.co/lUv97MH1uf">https://t.co/lUv97MH1uf</a><br>abs: <a href="https://t.co/GDhUCTsDeL">https://t.co/GDhUCTsDeL</a><br>project page: <a href="https://t.co/1aYb890e1e">https://t.co/1aYb890e1e</a> <a href="https://t.co/qLt4PsVqy0">pic.twitter.com/qLt4PsVqy0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347356997461618689?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Self-Attention Based Context-Aware 3D Object Detection

Prarthana Bhattacharyya, Chengjie Huang, Krzysztof Czarnecki

- retweets: 80, favorites: 59 (01/09/2021 23:43:32)

- links: [abs](https://arxiv.org/abs/2101.02672) | [pdf](https://arxiv.org/pdf/2101.02672)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Most existing point-cloud based 3D object detectors use convolution-like operators to process information in a local neighbourhood with fixed-weight kernels and aggregate global context hierarchically. However, recent work on non-local neural networks and self-attention for 2D vision has shown that explicitly modeling global context and long-range interactions between positions can lead to more robust and competitive models. In this paper, we explore two variants of self-attention for contextual modeling in 3D object detection by augmenting convolutional features with self-attention features. We first incorporate the pairwise self-attention mechanism into the current state-of-the-art BEV, voxel and point-based detectors and show consistent improvement over strong baseline models while simultaneously significantly reducing their parameter footprint and computational cost. We also propose a self-attention variant that samples a subset of the most representative features by learning deformations over randomly sampled locations. This not only allows us to scale explicit global contextual modeling to larger point-clouds, but also leads to more discriminative and informative feature descriptors. Our method can be flexibly applied to most state-of-the-art detectors with increased accuracy and parameter and compute efficiency. We achieve new state-of-the-art detection performance on KITTI and nuScenes datasets. Code is available at \url{https://github.com/AutoVision-cloud/SA-Det3D}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Attention Based Context-Aware 3D Object Detection<br>pdf: <a href="https://t.co/fCV3bMcsno">https://t.co/fCV3bMcsno</a><br>abs: <a href="https://t.co/MkXTAterRo">https://t.co/MkXTAterRo</a><br>github: <a href="https://t.co/C2dFFguPsb">https://t.co/C2dFFguPsb</a> <a href="https://t.co/Mn9aRZ2pfS">pic.twitter.com/Mn9aRZ2pfS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347398389059772418?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Mesh Total Generalized Variation for Denoising

Zheng Liu, YanLei Li, Weina Wang, Ligang Liu, Renjie Chen

- retweets: 90, favorites: 45 (01/09/2021 23:43:32)

- links: [abs](https://arxiv.org/abs/2101.02322) | [pdf](https://arxiv.org/pdf/2101.02322)
- [cs.CG](https://arxiv.org/list/cs.CG/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Total Generalized Variation (TGV) has recently been proven certainly successful in image processing for preserving sharp features as well as smooth transition variations. However, none of the existing works aims at numerically calculating TGV over triangular meshes. In this paper, we develop a novel numerical framework to discretize the second-order TGV over triangular meshes. Further, we propose a TGV-based variational model to restore the face normal field for mesh denoising. The TGV regularization in the proposed model is represented by a combination of a first- and second-order term, which can be automatically balanced. This TGV regularization is able to locate sharp features and preserve them via the first-order term, while recognize smoothly curved regions and recover them via the second-order term. To solve the optimization problem, we introduce an efficient iterative algorithm based on variable-splitting and augmented Lagrangian method. Extensive results and comparisons on synthetic and real scanning data validate that the proposed method outperforms the state-of-the-art methods visually and numerically.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mesh Total Generalized Variation for Denoising<br>pdf: <a href="https://t.co/vOmE0buiCH">https://t.co/vOmE0buiCH</a><br>abs: <a href="https://t.co/zHrOxdkbE8">https://t.co/zHrOxdkbE8</a> <a href="https://t.co/RPIy6EU981">pic.twitter.com/RPIy6EU981</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1347373179237502976?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Does Crowdfunding Really Foster Innovation? Evidence from the Board Game  Industry

Johannes Wachs, Balazs Vedres

- retweets: 64, favorites: 30 (01/09/2021 23:43:32)

- links: [abs](https://arxiv.org/abs/2101.02683) | [pdf](https://arxiv.org/pdf/2101.02683)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Crowdfunding offers inventors and entrepreneurs alternative access to resources with which they can develop and realize their ideas. Besides helping to secure capital, crowdfunding also connects creators with engaged early supporters who provide public feedback. But does this process foster truly innovative outcomes? Does the proliferation of crowdfunding in an industry make it more innovative overall? Prior studies investigating the link between crowdfunding and innovation do not compare traditional and crowdfunded products and so while claims that crowdfunding supports innovation are theoretically sound, they lack empirical backing. We address this gap using a unique dataset of board games, an industry with significant crowdfunding activity in recent years. Each game is described by how it combines fundamental mechanisms such as dice-rolling, negotiation, and resource-management, from which we develop quantitative measures of innovation in game design. Using these measures to compare games, we find that crowdfunded games tend to be more distinctive from previous games than their traditionally published counterparts. They are also significantly more likely to implement novel combinations of mechanisms. Crowdfunded games are not just transient experiments: subsequent games imitate their novel ideas. These results hold in regression models controlling for game and designer-level confounders. Our findings demonstrate that the innovative potential of crowdfunding goes beyond individual products to entire industries, as new ideas spill over to traditionally funded products.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint on crowdfunding and innovation w/ <a href="https://twitter.com/balazsvedres?ref_src=twsrc%5Etfw">@balazsvedres</a>. We ask: does crowdfunding create more novel products? We use data from <a href="https://twitter.com/BoardGameGeek?ref_src=twsrc%5Etfw">@BoardGameGeek</a>, embedding board games into a space of mechanisms (ie dice-rolling, pattern recognition, modular boards).<a href="https://t.co/DXUXcyheG1">https://t.co/DXUXcyheG1</a> <a href="https://t.co/ANPvGYswOE">pic.twitter.com/ANPvGYswOE</a></p>&mdash; Johannes Wachs (@johannes_wachs) <a href="https://twitter.com/johannes_wachs/status/1347501713129955329?ref_src=twsrc%5Etfw">January 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. VHS to HDTV Video Translation using Multi-task Adversarial Learning

Hongming Luo, Guangsen Liao, Xianxu Hou, Bozhi Liu, Fei Zhou, Guoping Qiu

- retweets: 48, favorites: 16 (01/09/2021 23:43:32)

- links: [abs](https://arxiv.org/abs/2101.02384) | [pdf](https://arxiv.org/pdf/2101.02384)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

There are large amount of valuable video archives in Video Home System (VHS) format. However, due to the analog nature, their quality is often poor. Compared to High-definition television (HDTV), VHS video not only has a dull color appearance but also has a lower resolution and often appears blurry. In this paper, we focus on the problem of translating VHS video to HDTV video and have developed a solution based on a novel unsupervised multi-task adversarial learning model. Inspired by the success of generative adversarial network (GAN) and CycleGAN, we employ cycle consistency loss, adversarial loss and perceptual loss together to learn a translation model. An important innovation of our work is the incorporation of super-resolution model and color transfer model that can solve unsupervised multi-task problem. To our knowledge, this is the first work that dedicated to the study of the relation between VHS and HDTV and the first computational solution to translate VHS to HDTV. We present experimental results to demonstrate the effectiveness of our solution qualitatively and quantitatively.




# 18. Gender Imbalance and Spatiotemporal Patterns of Contributions to Citizen  Science Projects: the case of Zooniverse

Khairunnisa Ibrahim, Samuel Khodursky, Taha Yasseri

- retweets: 26, favorites: 30 (01/09/2021 23:43:32)

- links: [abs](https://arxiv.org/abs/2101.02695) | [pdf](https://arxiv.org/pdf/2101.02695)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [astro-ph.GA](https://arxiv.org/list/astro-ph.GA/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Citizen Science is research undertaken by professional scientists and members of the public collaboratively. Despite numerous benefits of citizen science for both the advancement of science and the community of the citizen scientists, there is still no comprehensive knowledge of patterns of contributions, and the demography of contributors to citizen science projects. In this paper we provide a first overview of spatiotemporal and gender distribution of citizen science workforce by analyzing 54 million classifications contributed by more than 340 thousand citizen science volunteers from 198 countries to one of the largest citizen science platforms, Zooniverse. First we report on the uneven geographical distribution of the citizen scientist and model the variations among countries based on the socio-economic conditions as well as the level of research investment in each country. Analyzing the temporal features of contributions, we report on high "burstiness" of participation instances as well as the leisurely nature of participation suggested by the time of the day that the citizen scientists were the most active. Finally, we discuss the gender imbalance among citizen scientists (about 30% female) and compare it with other collaborative projects as well as the gender distribution in more formal scientific activities. Citizen science projects need further attention from outside of the academic community, and our findings can help attract the attention of public and private stakeholders, as well as to inform the design of the platforms and science policy making processes.



