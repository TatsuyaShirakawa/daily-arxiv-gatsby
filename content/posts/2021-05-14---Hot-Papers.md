---
title: Hot Papers 2021-05-14
date: 2021-05-15T16:03:59.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-14"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-14"
socialImage: "/media/flying-marine.jpg"

---

# 1. GAN Prior Embedded Network for Blind Face Restoration in the Wild

Tao Yang, Peiran Ren, Xuansong Xie, Lei Zhang

- retweets: 5016, favorites: 358 (05/15/2021 16:03:59)

- links: [abs](https://arxiv.org/abs/2105.06070) | [pdf](https://arxiv.org/pdf/2105.06070)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Blind face restoration (BFR) from severely degraded face images in the wild is a very challenging problem. Due to the high illness of the problem and the complex unknown degradation, directly training a deep neural network (DNN) usually cannot lead to acceptable results. Existing generative adversarial network (GAN) based methods can produce better results but tend to generate over-smoothed restorations. In this work, we propose a new method by first learning a GAN for high-quality face image generation and embedding it into a U-shaped DNN as a prior decoder, then fine-tuning the GAN prior embedded DNN with a set of synthesized low-quality face images. The GAN blocks are designed to ensure that the latent code and noise input to the GAN can be respectively generated from the deep and shallow features of the DNN, controlling the global face structure, local face details and background of the reconstructed image. The proposed GAN prior embedded network (GPEN) is easy-to-implement, and it can generate visually photo-realistic results. Our experiments demonstrated that the proposed GPEN achieves significantly superior results to state-of-the-art BFR methods both quantitatively and qualitatively, especially for the restoration of severely degraded face images in the wild. The source code and models can be found at https://github.com/yangxy/GPEN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GAN Prior Embedded Network for Blind Face Restoration in the Wild<br>pdf: <a href="https://t.co/zOamBeN85A">https://t.co/zOamBeN85A</a><br>abs: <a href="https://t.co/JW5XPISg8r">https://t.co/JW5XPISg8r</a> <a href="https://t.co/LVDkF0Yb7u">pic.twitter.com/LVDkF0Yb7u</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393067933438447616?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Monetizing Propaganda: How Far-right Extremists Earn Money by Video  Streaming

Megan Squire

- retweets: 2478, favorites: 161 (05/15/2021 16:03:59)

- links: [abs](https://arxiv.org/abs/2105.05929) | [pdf](https://arxiv.org/pdf/2105.05929)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Video streaming platforms such as Youtube, Twitch, and DLive allow users to live-stream video content for viewers who can optionally express their appreciation through monetary donations. DLive is one of the smaller and lesser-known streaming platforms, and historically has had fewer content moderation practices. It has thus become a popular place for violent extremists and other clandestine groups to earn money and propagandize. What is the financial structure of the DLive streaming ecosystem and how much money is changing hands? In the past it has been difficult to understand how far-right extremists fundraise via podcasts and video streams because of the secretive nature of the activity and because of the difficulty of getting data from social media platforms. This paper describes a novel experiment to collect and analyze data from DLive's publicly available ledgers of transactions in order to understand the financial structure of the clandestine, extreme far-right video streaming community. The main findings of this paper are, first, that the majority of donors are using micropayments in varying frequencies, but a small handful of donors spend large amounts of money to finance their favorite streamers. Next, the timing of donations to high-profile far-right streamers follows a fairly predictable pattern that is closely tied to a broadcast schedule. Finally, the far-right video streaming financial landscape is divided into separate cliques which exhibit very little crossover in terms of sizable donations. This work will be important to technology companies, policymakers, and researchers who are trying to understand how niche social media services, including video platforms, are being exploited by extremists to propagandize and fundraise.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Finally - here&#39;s my accepted paper on the DLive video streaming service and how it&#39;s being used by far-right propagandists to earn money (Apr 2020-Jan 2021). Lots of data! Here are some of the largest cash-outs including some post-insurrection refunds <a href="https://t.co/FzgR0UqEZq">https://t.co/FzgR0UqEZq</a> <a href="https://t.co/MsE6qH39dN">pic.twitter.com/MsE6qH39dN</a></p>&mdash; Megan Squire (@MeganSquire0) <a href="https://twitter.com/MeganSquire0/status/1393213679651508224?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. High-Resolution Complex Scene Synthesis with Transformers

Manuel Jahn, Robin Rombach, Björn Ommer

- retweets: 940, favorites: 165 (05/15/2021 16:03:59)

- links: [abs](https://arxiv.org/abs/2105.06458) | [pdf](https://arxiv.org/pdf/2105.06458)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The use of coarse-grained layouts for controllable synthesis of complex scene images via deep generative models has recently gained popularity. However, results of current approaches still fall short of their promise of high-resolution synthesis. We hypothesize that this is mostly due to the highly engineered nature of these approaches which often rely on auxiliary losses and intermediate steps such as mask generators. In this note, we present an orthogonal approach to this task, where the generative model is based on pure likelihood training without additional objectives. To do so, we first optimize a powerful compression model with adversarial training which learns to reconstruct its inputs via a discrete latent bottleneck and thereby effectively strips the latent representation of high-frequency details such as texture. Subsequently, we train an autoregressive transformer model to learn the distribution of the discrete image representations conditioned on a tokenized version of the layouts. Our experiments show that the resulting system is able to synthesize high-quality images consistent with the given layouts. In particular, we improve the state-of-the-art FID score on COCO-Stuff and on Visual Genome by up to 19% and 53% and demonstrate the synthesis of images up to 512 x 512 px on COCO and Open Images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High-Resolution Complex Scene Synthesis with Transformers<br>pdf: <a href="https://t.co/ERfWD3zaS4">https://t.co/ERfWD3zaS4</a><br>abs: <a href="https://t.co/y5gF3PaK7C">https://t.co/y5gF3PaK7C</a><br><br>state-of-the-art FID score on COCO-Stuff and on Visual Genome by up to 19% and 53% and demonstrate the synthesis of images up to 512×512 px on COCO and Open Images <a href="https://t.co/HyhifO7hOm">pic.twitter.com/HyhifO7hOm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393008174211510272?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High-Resolution Complex Scene Synthesis with Transformers<br><br>Improves the SotA FID score on COCO-Stuff and on Visual Genome by up to 19% and 53% and demonstrates the synthesis of images up to 512×512 px on COCO and Open Images.<a href="https://t.co/OsMQDhgtzy">https://t.co/OsMQDhgtzy</a> <a href="https://t.co/IugiKBxEud">pic.twitter.com/IugiKBxEud</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1393008690106552321?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech

Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov

- retweets: 634, favorites: 207 (05/15/2021 16:04:00)

- links: [abs](https://arxiv.org/abs/2105.06337) | [pdf](https://arxiv.org/pdf/2105.06337)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score. We will make the code publicly available shortly.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech<br><br>Proposes Grad-TTS, a TTS model with score-based decoder producing mel-spectrograms, which performs competitively with SotA TTS approaches in terms of MOS.<br><br>abs: <a href="https://t.co/bs0ZWEQnm7">https://t.co/bs0ZWEQnm7</a><br>project: <a href="https://t.co/C6xQme546l">https://t.co/C6xQme546l</a> <a href="https://t.co/wBN0FgolhT">pic.twitter.com/wBN0FgolhT</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1393006940918214656?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech<br>pdf: <a href="https://t.co/wtCmJsIT4H">https://t.co/wtCmJsIT4H</a><br>abs: <a href="https://t.co/NkpqPvNW2R">https://t.co/NkpqPvNW2R</a><br>project page: <a href="https://t.co/wKM8wtmuUI">https://t.co/wKM8wtmuUI</a><br><br>acoustic feature generator utilizing the concept of diffusion probabilistic modelling <a href="https://t.co/iSsUugYzUv">pic.twitter.com/iSsUugYzUv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393005166434131977?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/n. Happy to announce that my team presents our new paper “Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech” which has been accepted to ICML 2021! Check it out: <a href="https://t.co/jNN5nNknuQ">https://t.co/jNN5nNknuQ</a>. DEMO: <a href="https://t.co/V6I8L9vLKI">https://t.co/V6I8L9vLKI</a>. The code will also be released shortly.</p>&mdash; Ivan Vovk (@Kartexxx) <a href="https://twitter.com/Kartexxx/status/1393111641248632835?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Editing Conditional Radiance Fields

Steven Liu, Xiuming Zhang, Zhoutong Zhang, Richard Zhang, Jun-Yan Zhu, Bryan Russell

- retweets: 196, favorites: 47 (05/15/2021 16:04:00)

- links: [abs](https://arxiv.org/abs/2105.06466) | [pdf](https://arxiv.org/pdf/2105.06466)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A neural radiance field (NeRF) is a scene model supporting high-quality view synthesis, optimized per scene. In this paper, we explore enabling user editing of a category-level NeRF - also known as a conditional radiance field - trained on a shape category. Specifically, we introduce a method for propagating coarse 2D user scribbles to the 3D space, to modify the color or shape of a local region. First, we propose a conditional radiance field that incorporates new modular network components, including a shape branch that is shared across object instances. Observing multiple instances of the same category, our model learns underlying part semantics without any supervision, thereby allowing the propagation of coarse 2D user scribbles to the entire 3D region (e.g., chair seat). Next, we propose a hybrid network update strategy that targets specific network components, which balances efficiency and accuracy. During user interaction, we formulate an optimization problem that both satisfies the user's constraints and preserves the original object structure. We demonstrate our approach on various editing tasks over three shape datasets and show that it outperforms prior neural editing approaches. Finally, we edit the appearance and shape of a real photograph and show that the edit propagates to extrapolated novel views.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Editing Conditional Radiance Fields<br>pdf: <a href="https://t.co/rZVPqahBGi">https://t.co/rZVPqahBGi</a><br>abs: <a href="https://t.co/LcHRTGig4A">https://t.co/LcHRTGig4A</a><br>project page: <a href="https://t.co/hs1QAjN9yi">https://t.co/hs1QAjN9yi</a><br>github: <a href="https://t.co/JZN7SwvXdE">https://t.co/JZN7SwvXdE</a><br>colab: <a href="https://t.co/d1HhP9WRdM">https://t.co/d1HhP9WRdM</a> <a href="https://t.co/bg7ROR655i">pic.twitter.com/bg7ROR655i</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393032226271547392?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Connecting What to Say With Where to Look by Modeling Human Attention  Traces

Zihang Meng, Licheng Yu, Ning Zhang, Tamara Berg, Babak Damavandi, Vikas Singh, Amy Bearman

- retweets: 182, favorites: 47 (05/15/2021 16:04:00)

- links: [abs](https://arxiv.org/abs/2105.05964) | [pdf](https://arxiv.org/pdf/2105.05964)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce a unified framework to jointly model images, text, and human attention traces. Our work is built on top of the recent Localized Narratives annotation framework [30], where each word of a given caption is paired with a mouse trace segment. We propose two novel tasks: (1) predict a trace given an image and caption (i.e., visual grounding), and (2) predict a caption and a trace given only an image. Learning the grounding of each word is challenging, due to noise in the human-provided traces and the presence of words that cannot be meaningfully visually grounded. We present a novel model architecture that is jointly trained on dual tasks (controlled trace generation and controlled caption generation). To evaluate the quality of the generated traces, we propose a local bipartite matching (LBM) distance metric which allows the comparison of two traces of different lengths. Extensive experiments show our model is robust to the imperfect training data and outperforms the baselines by a clear margin. Moreover, we demonstrate that our model pre-trained on the proposed tasks can be also beneficial to the downstream task of COCO's guided image captioning. Our code and project page are publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Connecting What to Say With Where to Look by Modeling Human Attention Traces<br>pdf: <a href="https://t.co/nmP9yMtcbK">https://t.co/nmP9yMtcbK</a><br>abs: <a href="https://t.co/IExz1ptmxw">https://t.co/IExz1ptmxw</a><br>github: <a href="https://t.co/grRS0Wcqb2">https://t.co/grRS0Wcqb2</a><br><br>unified framework for modeling vision, language, and human attention traces <a href="https://t.co/PU3j7XjXMX">pic.twitter.com/PU3j7XjXMX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393071743166783488?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. 2021 Roadmap on Neuromorphic Computing and Engineering

Dennis V. Christensen, Regina Dittmann, Bernabé Linares-Barranco, Abu Sebastian, Manuel Le Gallo, Andrea Redaelli, Stefan Slesazeck, Thomas Mikolajick, Sabina Spiga, Stephan Menzel, Ilia Valov, Gianluca Milano, Carlo Ricciardi, Shi-Jun Liang, Feng Miao, Mario Lanza, Tyler J. Quill, Scott T. Keene, Alberto Salleo, Julie Grollier, Danijela Marković, Alice Mizrahi, Peng Yao, J. Joshua Yang, Giacomo Indiveri, John Paul Strachan, Suman Datta, Elisa Vianello, Alexandre Valentian, Johannes Feldmann, Xuan Li, Wolfram H.P. Pernice, Harish Bhaskaran, Emre Neftci, Srikanth Ramaswamy, Jonathan Tapson, Franz Scherr, Wolfgang Maass, Priyadarshini Panda, Youngeun Kim, Gouhei Tanaka, Simon Thorpe, Chiara Bartolozzi, Thomas A. Cleland, Christoph Posch, Shih-Chii Liu, Arnab Neelim Mazumder, Morteza Hosseini

- retweets: 131, favorites: 52 (05/15/2021 16:04:01)

- links: [abs](https://arxiv.org/abs/2105.05956) | [pdf](https://arxiv.org/pdf/2105.05956)
- [cs.ET](https://arxiv.org/list/cs.ET/recent) | [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [cond-mat.mtrl-sci](https://arxiv.org/list/cond-mat.mtrl-sci/recent)

Modern computation based on the von Neumann architecture is today a mature cutting-edge science. In this architecture, processing and memory units are implemented as separate blocks interchanging data intensively and continuously. This data transfer is responsible for a large part of the power consumption. The next generation computer technology is expected to solve problems at the exascale. Even though these future computers will be incredibly powerful, if they are based on von Neumann type architectures, they will consume between 20 and 30 megawatts of power and will not have intrinsic physically built-in capabilities to learn or deal with complex and unstructured data as our brain does. Neuromorphic computing systems are aimed at addressing these needs. The human brain performs about 10^15 calculations per second using 20W and a 1.2L volume. By taking inspiration from biology, new generation computers could have much lower power consumption than conventional processors, could exploit integrated non-volatile memory and logic, and could be explicitly designed to support dynamic learning in the context of complex and unstructured data. Among their potential future applications, business, health care, social security, disease and viruses spreading control might be the most impactful at societal level. This roadmap envisages the potential applications of neuromorphic materials in cutting edge technologies and focuses on the design and fabrication of artificial neural systems. The contents of this roadmap will highlight the interdisciplinary nature of this activity which takes inspiration from biology, physics, mathematics, computer science and engineering. This will provide a roadmap to explore and consolidate new technology behind both present and future applications in many technologically relevant areas.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Too many top authors in the 2021 Roadmap on Neuromorphic Computing and Engineering to tag! 2/2 <a href="https://twitter.com/AliceMizrahi?ref_src=twsrc%5Etfw">@AliceMizrahi</a>  <a href="https://twitter.com/giacomoi?ref_src=twsrc%5Etfw">@giacomoi</a>, <a href="https://twitter.com/virtualmind?ref_src=twsrc%5Etfw">@virtualmind</a>, <a href="https://twitter.com/srikipedia?ref_src=twsrc%5Etfw">@srikipedia</a>, <a href="https://twitter.com/jontapson?ref_src=twsrc%5Etfw">@jontapson</a>, <a href="https://twitter.com/franz_scherr?ref_src=twsrc%5Etfw">@franz_scherr</a>, <a href="https://twitter.com/priyapanda12?ref_src=twsrc%5Etfw">@priyapanda12</a>, <a href="https://twitter.com/nyalki?ref_src=twsrc%5Etfw">@nyalki</a>, <a href="https://twitter.com/ElDonati?ref_src=twsrc%5Etfw">@ElDonati</a>, <a href="https://twitter.com/slytolu?ref_src=twsrc%5Etfw">@slytolu</a>, thanks for contributing: <a href="https://t.co/sueEJOH9Nu">https://t.co/sueEJOH9Nu</a> <a href="https://t.co/YOWRYooDEj">pic.twitter.com/YOWRYooDEj</a></p>&mdash; Neuromorphic Computing and Engineering (@IOPneuromorphic) <a href="https://twitter.com/IOPneuromorphic/status/1393208243976085504?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Dynamic View Synthesis from Dynamic Monocular Video

Chen Gao, Ayush Saraf, Johannes Kopf, Jia-Bin Huang

- retweets: 110, favorites: 66 (05/15/2021 16:04:01)

- links: [abs](https://arxiv.org/abs/2105.06468) | [pdf](https://arxiv.org/pdf/2105.06468)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present an algorithm for generating novel views at arbitrary viewpoints and any input time step given a monocular video of a dynamic scene. Our work builds upon recent advances in neural implicit representation and uses continuous and differentiable functions for modeling the time-varying structure and the appearance of the scene. We jointly train a time-invariant static NeRF and a time-varying dynamic NeRF, and learn how to blend the results in an unsupervised manner. However, learning this implicit function from a single video is highly ill-posed (with infinitely many solutions that match the input video). To resolve the ambiguity, we introduce regularization losses to encourage a more physically plausible solution. We show extensive quantitative and qualitative results of dynamic view synthesis from casually captured videos.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dynamic View Synthesis from Dynamic Monocular Video<br>pdf: <a href="https://t.co/ScrGSD3jU7">https://t.co/ScrGSD3jU7</a><br>abs: <a href="https://t.co/QPNL8mWZVn">https://t.co/QPNL8mWZVn</a><br>project page: <a href="https://t.co/w5QCp3ighh">https://t.co/w5QCp3ighh</a><br><br>generating novel views at arbitrary viewpoints and any input time step given a monocular video of a dynamic scene <a href="https://t.co/GIZ4a9cKZc">pic.twitter.com/GIZ4a9cKZc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393034942637125636?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. 3D Spatial Recognition without Spatially Labeled 3D

Zhongzheng Ren, Ishan Misra, Alexander G. Schwing, Rohit Girdhar

- retweets: 121, favorites: 52 (05/15/2021 16:04:01)

- links: [abs](https://arxiv.org/abs/2105.06461) | [pdf](https://arxiv.org/pdf/2105.06461)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

We introduce WyPR, a Weakly-supervised framework for Point cloud Recognition, requiring only scene-level class tags as supervision. WyPR jointly addresses three core 3D recognition tasks: point-level semantic segmentation, 3D proposal generation, and 3D object detection, coupling their predictions through self and cross-task consistency losses. We show that in conjunction with standard multiple-instance learning objectives, WyPR can detect and segment objects in point cloud data without access to any spatial labels at training time. We demonstrate its efficacy using the ScanNet and S3DIS datasets, outperforming prior state of the art on weakly-supervised segmentation by more than 6% mIoU. In addition, we set up the first benchmark for weakly-supervised 3D object detection on both datasets, where WyPR outperforms standard approaches and establishes strong baselines for future work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">3D Spatial Recognition without Spatially Labeled 3D<br>pdf: <a href="https://t.co/uhSbqJhXwY">https://t.co/uhSbqJhXwY</a><br>abs: <a href="https://t.co/X8P0u6oStH">https://t.co/X8P0u6oStH</a><br>project page: <a href="https://t.co/6mN8AXvc4F">https://t.co/6mN8AXvc4F</a><br><br>a novel framework for joint 3D semantic segmentation and object detection, trained using only scene-level class tags as supervision <a href="https://t.co/KtuFlKlmoo">pic.twitter.com/KtuFlKlmoo</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393039832314679296?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. SyntheticFur dataset for neural rendering

Trung Le, Ryan Poplin, Fred Bertsch, Andeep Singh Toor, Margaret L. Oh

- retweets: 90, favorites: 42 (05/15/2021 16:04:01)

- links: [abs](https://arxiv.org/abs/2105.06409) | [pdf](https://arxiv.org/pdf/2105.06409)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce a new dataset called SyntheticFur built specifically for machine learning training. The dataset consists of ray traced synthetic fur renders with corresponding rasterized input buffers and simulation data files. We procedurally generated approximately 140,000 images and 15 simulations with Houdini. The images consist of fur groomed with different skin primitives and move with various motions in a predefined set of lighting environments. We also demonstrated how the dataset could be used with neural rendering to significantly improve fur graphics using inexpensive input buffers by training a conditional generative adversarial network with perceptual loss. We hope the availability of such high fidelity fur renders will encourage new advances with neural rendering for a variety of applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SyntheticFur dataset for neural rendering<br>pdf: <a href="https://t.co/ugiJKHmwvz">https://t.co/ugiJKHmwvz</a><br>abs: <a href="https://t.co/ylqc7EGVqY">https://t.co/ylqc7EGVqY</a><br>github: <a href="https://t.co/XvBIt5sFte">https://t.co/XvBIt5sFte</a><br><br>dataset consists of ray traced synthetic fur renders with corresponding rasterized input buffers and simulation data files <a href="https://t.co/vY9zi3xEfc">pic.twitter.com/vY9zi3xEfc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393080402366115840?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Orienting, Framing, Bridging, Magic, and Counseling: How Data Scientists  Navigate the Outer Loop of Client Collaborations in Industry and Academia

Sean Kross, Philip J. Guo

- retweets: 56, favorites: 34 (05/15/2021 16:04:01)

- links: [abs](https://arxiv.org/abs/2105.05849) | [pdf](https://arxiv.org/pdf/2105.05849)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

Data scientists often collaborate with clients to analyze data to meet a client's needs. What does the end-to-end workflow of a data scientist's collaboration with clients look like throughout the lifetime of a project? To investigate this question, we interviewed ten data scientists (5 female, 4 male, 1 non-binary) in diverse roles across industry and academia. We discovered that they work with clients in a six-stage outer-loop workflow, which involves 1) laying groundwork by building trust before a project begins, 2) orienting to the constraints of the client's environment, 3) collaboratively framing the problem, 4) bridging the gap between data science and domain expertise, 5) the inner loop of technical data analysis work, 6) counseling to help clients emotionally cope with analysis results. This novel outer-loop workflow contributes to CSCW by expanding the notion of what collaboration means in data science beyond the widely-known inner-loop technical workflow stages of acquiring, cleaning, analyzing, modeling, and visualizing data. We conclude by discussing the implications of our findings for data science education, parallels to design work, and unmet needs for tool development.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am so excited to announce that my new paper “Orienting, Framing, Bridging, Magic, and Counseling: How Data Scientists Navigate the Outer Loop of Client Collaborations in Industry and Academia” has been accepted to <a href="https://twitter.com/hashtag/CSCW2021?src=hash&amp;ref_src=twsrc%5Etfw">#CSCW2021</a>! You can read it here: <a href="https://t.co/BqNKcn7Kxf">https://t.co/BqNKcn7Kxf</a> <a href="https://t.co/qzirhQzf3a">pic.twitter.com/qzirhQzf3a</a></p>&mdash; Sean Kross (@seankross) <a href="https://twitter.com/seankross/status/1393295327680110595?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. The Power of the Weisfeiler-Leman Algorithm for Machine Learning with  Graphs

Christopher Morris, Matthias Fey, Nils M. Kriege

- retweets: 32, favorites: 52 (05/15/2021 16:04:01)

- links: [abs](https://arxiv.org/abs/2105.05911) | [pdf](https://arxiv.org/pdf/2105.05911)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

In recent years, algorithms and neural architectures based on the Weisfeiler-Leman algorithm, a well-known heuristic for the graph isomorphism problem, emerged as a powerful tool for (supervised) machine learning with graphs and relational data. Here, we give a comprehensive overview of the algorithm's use in a machine learning setting. We discuss the theoretical background, show how to use it for supervised graph- and node classification, discuss recent extensions, and its connection to neural architectures. Moreover, we give an overview of current applications and future directions to stimulate research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Want to get a high-level overview of the Weisfeiler-Leman algorithm&#39;s use in ML and its connection to GNNs? Check out our IJCAI survey track paper: <a href="https://t.co/7WCYtXoE7b">https://t.co/7WCYtXoE7b</a>.<br><br>Joint work with <a href="https://twitter.com/rusty1s?ref_src=twsrc%5Etfw">@rusty1s</a> (<a href="https://twitter.com/sfb876?ref_src=twsrc%5Etfw">@sfb876</a>) and Nils M. Kriege (<a href="https://twitter.com/univienna?ref_src=twsrc%5Etfw">@univienna</a>).</p>&mdash; Christopher Morris (@chrsmrrs) <a href="https://twitter.com/chrsmrrs/status/1393047213941592064?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Neural Trajectory Fields for Dynamic Novel View Synthesis

Chaoyang Wang, Ben Eckart, Simon Lucey, Orazio Gallo

- retweets: 42, favorites: 22 (05/15/2021 16:04:01)

- links: [abs](https://arxiv.org/abs/2105.05994) | [pdf](https://arxiv.org/pdf/2105.05994)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent approaches to render photorealistic views from a limited set of photographs have pushed the boundaries of our interactions with pictures of static scenes. The ability to recreate moments, that is, time-varying sequences, is perhaps an even more interesting scenario, but it remains largely unsolved. We introduce DCT-NeRF, a coordinatebased neural representation for dynamic scenes. DCTNeRF learns smooth and stable trajectories over the input sequence for each point in space. This allows us to enforce consistency between any two frames in the sequence, which results in high quality reconstruction, particularly in dynamic regions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Trajectory Fields for Dynamic Novel View Synthesis<br>pdf: <a href="https://t.co/3VAaIzMhEX">https://t.co/3VAaIzMhEX</a><br>abs: <a href="https://t.co/LAiOgvEG15">https://t.co/LAiOgvEG15</a><br><br>coordinate-based neural representation that can render photorealistic novel views of dynamic scenes <a href="https://t.co/BJeRZ8uP35">pic.twitter.com/BJeRZ8uP35</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1393030638580731907?ref_src=twsrc%5Etfw">May 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



