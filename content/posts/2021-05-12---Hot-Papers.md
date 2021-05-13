---
title: Hot Papers 2021-05-12
date: 2021-05-13T09:14:27.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. Enhancing Photorealism Enhancement

Stephan R. Richter, Hassan Abu AlHaija, Vladlen Koltun

- retweets: 5043, favorites: 356 (05/13/2021 09:14:27)

- links: [abs](https://arxiv.org/abs/2105.04619) | [pdf](https://arxiv.org/pdf/2105.04619)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present an approach to enhancing the realism of synthetic images. The images are enhanced by a convolutional network that leverages intermediate representations produced by conventional rendering pipelines. The network is trained via a novel adversarial objective, which provides strong supervision at multiple perceptual levels. We analyze scene layout distributions in commonly used datasets and find that they differ in important ways. We hypothesize that this is one of the causes of strong artifacts that can be observed in the results of many prior methods. To address this we propose a new strategy for sampling image patches during training. We also introduce multiple architectural improvements in the deep network modules used for photorealism enhancement. We confirm the benefits of our contributions in controlled experiments and report substantial gains in stability and realism in comparison to recent image-to-image translation methods and a variety of other baselines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Enhancing Photorealism Enhancement<br>pdf: <a href="https://t.co/hRrN03fcRZ">https://t.co/hRrN03fcRZ</a><br>abs: <a href="https://t.co/kIfxCJM56r">https://t.co/kIfxCJM56r</a><br><br>an approach to enhancing the realism of synthetic images <a href="https://t.co/ZT1U9H6Az0">pic.twitter.com/ZT1U9H6Az0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392302341559357440?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Diffusion Models Beat GANs on Image Synthesis

Prafulla Dhariwal, Alex Nichol

- retweets: 5179, favorites: 193 (05/13/2021 09:14:28)

- links: [abs](https://arxiv.org/abs/2105.05233) | [pdf](https://arxiv.org/pdf/2105.05233)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We achieve this on unconditional image synthesis by finding a better architecture through a series of ablations. For conditional image synthesis, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for sample quality using gradients from a classifier. We achieve an FID of 2.97 on ImageNet $128 \times 128$, 4.59 on ImageNet $256 \times 256$, and $7.72$ on ImageNet $512 \times 512$, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution. Finally, we find that classifier guidance combines well with upsampling diffusion models, further improving FID to 3.85 on ImageNet $512 \times 512$. We release our code at https://github.com/openai/guided-diffusion

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/n Excited to release what <a href="https://twitter.com/unixpickle?ref_src=twsrc%5Etfw">@unixpickle</a> and I have been working on for the past few months <a href="https://twitter.com/OpenAI?ref_src=twsrc%5Etfw">@OpenAI</a>! We show diffusion models can beat GANs on generating natural images, using an improved architecture and by guiding the generative model with a classifier.<a href="https://t.co/7wnLjSmAm8">https://t.co/7wnLjSmAm8</a> <a href="https://t.co/zxCVjSI66H">pic.twitter.com/zxCVjSI66H</a></p>&mdash; Prafulla Dhariwal (@prafdhar) <a href="https://twitter.com/prafdhar/status/1392575638263926784?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Diffusion Models Beat GANs on Image Synthesis<br><br>Achieves 3.85 FID on ImageNet 512×512 and matches BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution.<a href="https://t.co/egFfH0r0tl">https://t.co/egFfH0r0tl</a> <a href="https://t.co/GARIw40bYK">pic.twitter.com/GARIw40bYK</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1392280377784369152?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. VICReg: Variance-Invariance-Covariance Regularization for  Self-Supervised Learning

Adrien Bardes, Jean Ponce, Yann LeCun

- retweets: 2131, favorites: 318 (05/13/2021 09:14:28)

- links: [abs](https://arxiv.org/abs/2105.04906) | [pdf](https://arxiv.org/pdf/2105.04906)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent self-supervised methods for image representation learning are based on maximizing the agreement between embedding vectors from different views of the same image. A trivial solution is obtained when the encoder outputs constant vectors. This collapse problem is often avoided through implicit biases in the learning architecture, that often lack a clear justification or interpretation. In this paper, we introduce VICReg (Variance-Invariance-Covariance Regularization), a method that explicitly avoids the collapse problem with a simple regularization term on the variance of the embeddings along each dimension individually. VICReg combines the variance term with a decorrelation mechanism based on redundancy reduction and covariance regularization, and achieves results on par with the state of the art on several downstream tasks. In addition, we show that incorporating our new variance term into other methods helps stabilize the training and leads to performance improvements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.<br>By Adrien Bardes, Jean Ponce, and yours truly.<a href="https://t.co/Ih4nRoMZYv">https://t.co/Ih4nRoMZYv</a><br>Insanely simple and effective method for self-supervised training of joint-embedding architectures (e.g. Siamese nets).<br>1/N</p>&mdash; Yann LeCun (@ylecun) <a href="https://twitter.com/ylecun/status/1392493077999325191?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning<br>pdf: <a href="https://t.co/mSLDe7ivNY">https://t.co/mSLDe7ivNY</a><br>abs: <a href="https://t.co/LbBS8y713T">https://t.co/LbBS8y713T</a><br><br>method explicitly avoids collapse problem with simple regularization term on variance of the embeddings along each dimension individually <a href="https://t.co/BWItR0wUw7">pic.twitter.com/BWItR0wUw7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392293600067788801?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Including Signed Languages in Natural Language Processing

Kayo Yin, Amit Moryossef, Julie Hochgesang, Yoav Goldberg, Malihe Alikhani

- retweets: 1521, favorites: 170 (05/13/2021 09:14:28)

- links: [abs](https://arxiv.org/abs/2105.05222) | [pdf](https://arxiv.org/pdf/2105.05222)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Signed languages are the primary means of communication for many deaf and hard of hearing individuals. Since signed languages exhibit all the fundamental linguistic properties of natural language, we believe that tools and theories of Natural Language Processing (NLP) are crucial towards its modeling. However, existing research in Sign Language Processing (SLP) seldom attempt to explore and leverage the linguistic organization of signed languages. This position paper calls on the NLP community to include signed languages as a research area with high social and scientific impact. We first discuss the linguistic properties of signed languages to consider during their modeling. Then, we review the limitations of current SLP models and identify the open challenges to extend NLP to signed languages. Finally, we urge (1) the adoption of an efficient tokenization method; (2) the development of linguistically-informed models; (3) the collection of real-world signed language data; (4) the inclusion of local signed language communities as an active and leading voice in the direction of research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Why should you, as an NLP researcher, work on signed languages?<br><br>Our upcoming <a href="https://twitter.com/hashtag/ACL2021NLP?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021NLP</a> paper (w/ <a href="https://twitter.com/amitmoryossef?ref_src=twsrc%5Etfw">@amitmoryossef</a> <a href="https://twitter.com/jahochcam?ref_src=twsrc%5Etfw">@jahochcam</a> <a href="https://twitter.com/yoavgo?ref_src=twsrc%5Etfw">@yoavgo</a> <a href="https://twitter.com/malihealikhani?ref_src=twsrc%5Etfw">@malihealikhani</a>) is a call-to-action for the NLP community to include signed languages, and explains how to do so 🤟[1/7]<a href="https://t.co/1dqbWZrP9i">https://t.co/1dqbWZrP9i</a> <a href="https://t.co/Z1lrr73uZN">pic.twitter.com/Z1lrr73uZN</a></p>&mdash; Kayo Yin (@kayo_yin) <a href="https://twitter.com/kayo_yin/status/1392340087308967936?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. GSPMD: General and Scalable Parallelization for ML Computation Graphs

Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, Blake Hechtman, Yanping Huang, Rahul Joshi, Maxim Krikun, Dmitry Lepikhin, Andy Ly, Marcello Maggioni, Ruoming Pang, Noam Shazeer, Shibo Wang, Tao Wang, Yonghui Wu, Zhifeng Chen

- retweets: 647, favorites: 171 (05/13/2021 09:14:29)

- links: [abs](https://arxiv.org/abs/2105.04663) | [pdf](https://arxiv.org/pdf/2105.04663)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present GSPMD, an automatic, compiler-based parallelization system for common machine learning computation graphs. It allows users to write programs in the same way as for a single device, then give hints through a few annotations on how to distribute tensors, based on which GSPMD will parallelize the computation. Its representation of partitioning is simple yet general, allowing it to express different or mixed paradigms of parallelism on a wide variety of models.   GSPMD infers the partitioning for every operator in the graph based on limited user annotations, making it convenient to scale up existing single-device programs. It solves several technical challenges for production usage, such as static shape constraints, uneven partitioning, exchange of halo data, and nested operator partitioning. These techniques allow GSPMD to achieve 50% to 62% compute utilization on 128 to 2048 Cloud TPUv3 cores for models with up to one trillion parameters.   GSPMD produces a single program for all devices, which adjusts its behavior based on a run-time partition ID, and uses collective operators for cross-device communication. This property allows the system itself to be scalable: the compilation time stays constant with increasing number of devices.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The paper describing the XLA SPMD automatic partitioning infrastructure (what’s behind JAX model parallelism APIs like sharded_jit and pjit) is out: <a href="https://t.co/vV2FfKCTpL">https://t.co/vV2FfKCTpL</a></p>&mdash; James Bradbury (@jekbradbury) <a href="https://twitter.com/jekbradbury/status/1392320100200456193?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. COVID-19 Vaccine Hesitancy on Social Media: Building a Public Twitter  Dataset of Anti-vaccine Content, Vaccine Misinformation and Conspiracies

Goran Muric, Yusong Wu, Emilio Ferrara

- retweets: 484, favorites: 48 (05/13/2021 09:14:29)

- links: [abs](https://arxiv.org/abs/2105.05134) | [pdf](https://arxiv.org/pdf/2105.05134)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

False claims about COVID-19 vaccines can undermine public trust in ongoing vaccination campaigns, thus posing a threat to global public health. Misinformation originating from various sources has been spreading online since the beginning of the COVID-19 pandemic. In this paper, we present a dataset of Twitter posts that exhibit a strong anti-vaccine stance. The dataset consists of two parts: a) a streaming keyword-centered data collection with more than 1.8 million tweets, and b) a historical account-level collection with more than 135 million tweets. The former leverages the Twitter streaming API to follow a set of specific vaccine-related keywords starting from mid-October 2020. The latter consists of all historical tweets of 70K accounts that were engaged in the active spreading of anti-vaccine narratives. We present descriptive analyses showing the volume of activity over time, geographical distributions, topics, news sources, and inferred account political leaning. This dataset can be used in studying anti-vaccine misinformation on social media and enable a better understanding of vaccine hesitancy. In compliance with Twitter's Terms of Service, our anonymized dataset is publicly available at: https://github.com/gmuric/avax-tweets-dataset

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔥 Dataset Release🔥 w/ <a href="https://twitter.com/goranmuric?ref_src=twsrc%5Etfw">@goranmuric</a> <br>135M tweets on <a href="https://twitter.com/hashtag/vaccine?src=hash&amp;ref_src=twsrc%5Etfw">#vaccine</a> <a href="https://twitter.com/hashtag/misinformation?src=hash&amp;ref_src=twsrc%5Etfw">#misinformation</a><br><br>COVID-19 Vaccine Hesitancy on Social Media: Building a Public Twitter Dataset of Anti-vaccine Content, Vaccine Misinformation and Conspiracies<br><br>Paper <a href="https://t.co/kG06d7cE1Y">https://t.co/kG06d7cE1Y</a><br>Data <a href="https://t.co/8DlIWCLqVo">https://t.co/8DlIWCLqVo</a></p>&mdash; Emilio Ferrara (@emilio__ferrara) <a href="https://twitter.com/emilio__ferrara/status/1392535641880031232?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Differentiable Signal Processing With Black-Box Audio Effects

Marco A. Martínez Ramírez, Oliver Wang, Paris Smaragdis, Nicholas J. Bryan

- retweets: 441, favorites: 81 (05/13/2021 09:14:29)

- links: [abs](https://arxiv.org/abs/2105.04752) | [pdf](https://arxiv.org/pdf/2105.04752)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.SP](https://arxiv.org/list/eess.SP/recent)

We present a data-driven approach to automate audio signal processing by incorporating stateful third-party, audio effects as layers within a deep neural network. We then train a deep encoder to analyze input audio and control effect parameters to perform the desired signal manipulation, requiring only input-target paired audio data as supervision. To train our network with non-differentiable black-box effects layers, we use a fast, parallel stochastic gradient approximation scheme within a standard auto differentiation graph, yielding efficient end-to-end backpropagation. We demonstrate the power of our approach with three separate automatic audio production applications: tube amplifier emulation, automatic removal of breaths and pops from voice recordings, and automatic music mastering. We validate our results with a subjective listening test, showing our approach not only can enable new automatic audio effects tasks, but can yield results comparable to a specialized, state-of-the-art commercial solution for music mastering.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Differentiable Signal Processing With Black-Box Audio Effects<br>pdf: <a href="https://t.co/ewKJMuhfXw">https://t.co/ewKJMuhfXw</a><br>abs: <a href="https://t.co/9PVWT0ed7P">https://t.co/9PVWT0ed7P</a><br>project page: <a href="https://t.co/rCZQk0SkRb">https://t.co/rCZQk0SkRb</a><br>github: <a href="https://t.co/hSgYcox1z7">https://t.co/hSgYcox1z7</a> <a href="https://t.co/AyrpQqlLEv">pic.twitter.com/AyrpQqlLEv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392320863542947840?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. EL-Attention: Memory Efficient Lossless Attention for Generation

Yu Yan, Jiusheng Chen, Weizhen Qi, Nikhil Bhendawade, Yeyun Gong, Nan Duan, Ruofei Zhang

- retweets: 310, favorites: 111 (05/13/2021 09:14:29)

- links: [abs](https://arxiv.org/abs/2105.04779) | [pdf](https://arxiv.org/pdf/2105.04779)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformer model with multi-head attention requires caching intermediate results for efficient inference in generation tasks. However, cache brings new memory-related costs and prevents leveraging larger batch size for faster speed. We propose memory-efficient lossless attention (called EL-attention) to address this issue. It avoids heavy operations for building multi-head keys and values, with no requirements of using cache. EL-attention constructs an ensemble of attention results by expanding query while keeping key and value shared. It produces the same result as multi-head attention with less GPU memory and faster inference speed. We conduct extensive experiments on Transformer, BART, and GPT-2 for summarization and question generation tasks. The results show EL-attention speeds up existing models by 1.6x to 5.3x without accuracy loss.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">EL-Attention: Memory Efficient Lossless Attention for Generation<br><br>Speeds up the inference of various Transformer models by 1.6x to 5.3x without accuracy loss and also saves GPU memory. <a href="https://t.co/Oq1D4BiCIv">https://t.co/Oq1D4BiCIv</a> <a href="https://t.co/9ETraNCiX5">pic.twitter.com/9ETraNCiX5</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1392292220955287555?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Leveraging Sparse Linear Layers for Debuggable Deep Networks

Eric Wong, Shibani Santurkar, Aleksander Mądry

- retweets: 319, favorites: 94 (05/13/2021 09:14:29)

- links: [abs](https://arxiv.org/abs/2105.04857) | [pdf](https://arxiv.org/pdf/2105.04857)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We show how fitting sparse linear models over learned deep feature representations can lead to more debuggable neural networks. These networks remain highly accurate while also being more amenable to human interpretation, as we demonstrate quantiatively via numerical and human experiments. We further illustrate how the resulting sparse explanations can help to identify spurious correlations, explain misclassifications, and diagnose model biases in vision and language tasks. The code for our toolkit can be found at https://github.com/madrylab/debuggabledeepnetworks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How can we build deep networks that are easier to debug? With <a href="https://twitter.com/RICEric22?ref_src=twsrc%5Etfw">@RICEric22</a> and <a href="https://twitter.com/ShibaniSan?ref_src=twsrc%5Etfw">@ShibaniSan</a> we find that fitting a sparse linear decision layer on top of model features gets you surprisingly far. Blogs: <a href="https://t.co/MsmHSKLwz7">https://t.co/MsmHSKLwz7</a> &amp; <a href="https://t.co/Gvms31wwUG">https://t.co/Gvms31wwUG</a> Paper: <a href="https://t.co/f8uoxYuPIm">https://t.co/f8uoxYuPIm</a> <a href="https://t.co/BHwkV4XKwQ">pic.twitter.com/BHwkV4XKwQ</a></p>&mdash; Aleksander Madry (@aleks_madry) <a href="https://twitter.com/aleks_madry/status/1392511253877297157?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Leveraging Sparse Linear Layers for Debuggable Deep Networks<br>pdf: <a href="https://t.co/FwT0nbedyc">https://t.co/FwT0nbedyc</a><br>abs: <a href="https://t.co/d0ErmeyBCU">https://t.co/d0ErmeyBCU</a><br>github: <a href="https://t.co/omAMNKedKD">https://t.co/omAMNKedKD</a><br><br>fitting sparse linear models over learned deep feature representations can lead to more debuggable neural networks <a href="https://t.co/r8xJcbsw8K">pic.twitter.com/r8xJcbsw8K</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392319121883205632?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Deja-Vu: A Glimpse on Radioactive Soft-Error Consequences on Classical  and Quantum Computations

Antonio Nappa, Christopher Hobbs, Andrea Lanzi

- retweets: 332, favorites: 50 (05/13/2021 09:14:30)

- links: [abs](https://arxiv.org/abs/2105.05103) | [pdf](https://arxiv.org/pdf/2105.05103)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [quant-ph](https://arxiv.org/list/quant-ph/recent)

What do Apple, the FBI and a Belgian politician have in common? In 2003, in Belgium there was an election using electronic voting machines. Mysteriously one candidate summed an excess of 4096 votes. An accurate analysis led to the official explanation that a spontaneous creation of a bit in position 13 of the memory of the computer attributed 4096 extra votes to one candidate. One of the most credited answers to this event is attributed to cosmic rays i.e.(gamma), which can filter through the atmosphere. There are cases though, with classical computers, like forensic investigations, or system recovery where such soft-errors may be helpful to gain root privileges and recover data. In this paper we show preliminary results of using radioactive sources as a mean to generate bit-flips and exploit classical electronic computation devices. We used low radioactive emissions generated by Cobalt and Cesium and obtained bit-flips which made the program under attack crash. We also provide the first overview of the consequences of SEUs in quantum computers which are today used in production for protein folding optimization, showing potential impactful consequences. To the best of our knowledge we are the first to leverage SEUs for exploitation purposes which could be of great impact on classical and quantum computers.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr"><a href="https://twitter.com/hashtag/%E3%82%AD%E3%83%A3%E3%83%AB%E3%81%A1%E3%82%83%E3%82%93%E3%81%AEquantph%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF?src=hash&amp;ref_src=twsrc%5Etfw">#キャルちゃんのquantphチェック</a><br>2003年のベルギーでの電子投票不正が高エネルギー宇宙線によるものであった可能性を受けて、放射性物質により古典計算機のビットを反転して悪用する方法を示した論文。量子コンピュータへの影響も指摘。<a href="https://t.co/K0g5I34vJK">https://t.co/K0g5I34vJK</a> <a href="https://t.co/0YFnAu1egN">pic.twitter.com/0YFnAu1egN</a></p>&mdash; キャルちゃん、🇺🇸移住10ヶ月目。 (@tweet_nakasho) <a href="https://twitter.com/tweet_nakasho/status/1392503312868511748?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Representation Learning via Global Temporal Alignment and  Cycle-Consistency

Isma Hadji, Konstantinos G. Derpanis, Allan D. Jepson

- retweets: 144, favorites: 69 (05/13/2021 09:14:30)

- links: [abs](https://arxiv.org/abs/2105.05217) | [pdf](https://arxiv.org/pdf/2105.05217)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce a weakly supervised method for representation learning based on aligning temporal sequences (e.g., videos) of the same process (e.g., human action). The main idea is to use the global temporal ordering of latent correspondences across sequence pairs as a supervisory signal. In particular, we propose a loss based on scoring the optimal sequence alignment to train an embedding network. Our loss is based on a novel probabilistic path finding view of dynamic time warping (DTW) that contains the following three key features: (i) the local path routing decisions are contrastive and differentiable, (ii) pairwise distances are cast as probabilities that are contrastive as well, and (iii) our formulation naturally admits a global cycle consistency loss that verifies correspondences. For evaluation, we consider the tasks of fine-grained action classification, few shot learning, and video synchronization. We report significant performance increases over previous methods. In addition, we report two applications of our temporal alignment framework, namely 3D pose reconstruction and fine-grained audio/visual retrieval.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Representation Learning via Global Temporal Alignment and Cycle-Consistency<br>pdf: <a href="https://t.co/d1c0abH0mw">https://t.co/d1c0abH0mw</a><br>abs: <a href="https://t.co/db0Fd6Spxy">https://t.co/db0Fd6Spxy</a><br><br>weakly supervised method for representation learning relying on sequence alignment as a supervisory signal <a href="https://t.co/2ICly0NvV3">pic.twitter.com/2ICly0NvV3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392299596844515330?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> paper &quot;Representation Learning via Global Temporal Alignment and Cycle-Consistency&quot;!<br><br>Joint work with Isma Hadji (my academic sister 🤓) &amp; Allan Jepson at the Samsung <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> Centre <a href="https://twitter.com/hashtag/Toronto?src=hash&amp;ref_src=twsrc%5Etfw">#Toronto</a><br><br>Paper: <a href="https://t.co/ArZJ1QdaLK">https://t.co/ArZJ1QdaLK</a><br>Project page: <a href="https://t.co/7gTFFKAQeX">https://t.co/7gTFFKAQeX</a> <a href="https://t.co/Nf12WmheOd">https://t.co/Nf12WmheOd</a> <a href="https://t.co/4vMjKOqN6W">pic.twitter.com/4vMjKOqN6W</a></p>&mdash; Kosta Derpanis (@CSProfKGD) <a href="https://twitter.com/CSProfKGD/status/1392509885347213316?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. TransPose: Real-time 3D Human Translation and Pose Estimation with Six  Inertial Sensors

Xinyu Yi, Yuxiao Zhou, Feng Xu

- retweets: 30, favorites: 25 (05/13/2021 09:14:30)

- links: [abs](https://arxiv.org/abs/2105.04605) | [pdf](https://arxiv.org/pdf/2105.04605)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Motion capture is facing some new possibilities brought by the inertial sensing technologies which do not suffer from occlusion or wide-range recordings as vision-based solutions do. However, as the recorded signals are sparse and quite noisy, online performance and global translation estimation turn out to be two key difficulties. In this paper, we present TransPose, a DNN-based approach to perform full motion capture (with both global translations and body poses) from only 6 Inertial Measurement Units (IMUs) at over 90 fps. For body pose estimation, we propose a multi-stage network that estimates leaf-to-full joint positions as intermediate results. This design makes the pose estimation much easier, and thus achieves both better accuracy and lower computation cost. For global translation estimation, we propose a supporting-foot-based method and an RNN-based method to robustly solve for the global translations with a confidence-based fusion technique. Quantitative and qualitative comparisons show that our method outperforms the state-of-the-art learning- and optimization-based methods with a large margin in both accuracy and efficiency. As a purely inertial sensor-based approach, our method is not limited by environmental settings (e.g., fixed cameras), making the capture free from common difficulties such as wide-range motion space and strong occlusion.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors<br>pdf: <a href="https://t.co/QbrHfyBgID">https://t.co/QbrHfyBgID</a><br>abs: <a href="https://t.co/pMwn7pBeuC">https://t.co/pMwn7pBeuC</a><br>project page: <a href="https://t.co/6LybQmocPz">https://t.co/6LybQmocPz</a> <a href="https://t.co/UGtRwnclva">pic.twitter.com/UGtRwnclva</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392296561544155140?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. You Are How (and Where) You Search? Comparative Analysis of Web Search  Behaviour Using Web Tracking Data

Aleksandra Urman, Mykola Makhortykh

- retweets: 30, favorites: 24 (05/13/2021 09:14:30)

- links: [abs](https://arxiv.org/abs/2105.04961) | [pdf](https://arxiv.org/pdf/2105.04961)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

We conduct a comparative analysis of desktop web search behaviour of users from Germany (n=558) and Switzerland (n=563) based on a combination of web tracking and survey data. We find that web search accounts for 13% of all desktop browsing, with the share being higher in Switzerland than in Germany. We find that in over 50% of cases users clicked on the first search result, with over 97% of all clicks being made on the first page of search outputs. Most users rely on Google when conducting searches, and users preferences for other engines are related to their demographics. We also test relationships between user demographics and daily number of searches, average share of search activities among tracked events by user as well as the tendency to click on higher- or lower-ranked results. We find differences in such relationships between the two countries that highlights the importance of comparative research in this domain. Further, we observe differences in the temporal patterns of web search use between women and men, marking the necessity of disaggregating data by gender in observational studies regarding online information behaviour.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We find that people click on the very first (!) web search result over 50% of the time; the first result page gets over 97% of user clicks. This and more about web search behaviour based on web tracking data from 🇩🇪 and🇨🇭 in our new preprint <a href="https://t.co/hhcpxPRNQx">https://t.co/hhcpxPRNQx</a></p>&mdash; Aleksandra Urman (@AUrman21) <a href="https://twitter.com/AUrman21/status/1392520079234019334?ref_src=twsrc%5Etfw">May 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



