---
title: Hot Papers 2021-05-24
date: 2021-05-25T07:20:27.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-24"
socialImage: "/media/flying-marine.jpg"

---

# 1. Analysis of Boolean Functions

Ryan O'Donnell

- retweets: 5922, favorites: 432 (05/25/2021 07:20:27)

- links: [abs](https://arxiv.org/abs/2105.10386) | [pdf](https://arxiv.org/pdf/2105.10386)
- [cs.DM](https://arxiv.org/list/cs.DM/recent) | [math.PR](https://arxiv.org/list/math.PR/recent)

The subject of this textbook is the analysis of Boolean functions. Roughly speaking, this refers to studying Boolean functions $f : \{0,1\}^n \to \{0,1\}$ via their Fourier expansion and other analytic means. Boolean functions are perhaps the most basic object of study in theoretical computer science, and Fourier analysis has become an indispensable tool in the field. The topic has also played a key role in several other areas of mathematics, from combinatorics, random graph theory, and statistical physics, to Gaussian geometry, metric/Banach spaces, and social choice theory.   The intent of this book is both to develop the foundations of the field and to give a wide (though far from exhaustive) overview of its applications. Each chapter ends with a "highlight" showing the power of analysis of Boolean functions in different subject areas: property testing, social choice, cryptography, circuit complexity, learning theory, pseudorandomness, hardness of approximation, concrete complexity, and random graph theory.   The book can be used as a reference for working researchers or as the basis of a one-semester graduate-level course. The author has twice taught such a course at Carnegie Mellon University, attended mainly by graduate students in computer science and mathematics but also by advanced undergraduates, postdocs, and researchers in adjacent fields. In both years most of Chapters 1-5 and 7 were covered, along with parts of Chapters 6, 8, 9, and 11, and some additional material on additive combinatorics. Nearly 500 exercises are provided at the ends of the book's chapters.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My book 𝑨𝒏𝒂𝒍𝒚𝒔𝒊𝒔 𝒐𝒇 𝑩𝒐𝒐𝒍𝒆𝒂𝒏 𝑭𝒖𝒏𝒄𝒕𝒊𝒐𝒏𝒔 is now on arXiv, in a slightly updated version with 100+ typos/bugs fixed:<a href="https://t.co/tXBDItgyd9">https://t.co/tXBDItgyd9</a><br><br>(No new mathematical content added, so it remains a snapshot of A.O.B.F. circa 2014.)</p>&mdash; Ryan O&#39;Donnell (@BooleanAnalysis) <a href="https://twitter.com/BooleanAnalysis/status/1396628094010679296?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Pretrained Language Models for Text Generation: A Survey

Junyi Li, Tianyi Tang, Wayne Xin Zhao, Ji-Rong Wen

- retweets: 4166, favorites: 350 (05/25/2021 07:20:28)

- links: [abs](https://arxiv.org/abs/2105.10311) | [pdf](https://arxiv.org/pdf/2105.10311)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Text generation has become one of the most important yet challenging tasks in natural language processing (NLP). The resurgence of deep learning has greatly advanced this field by neural generation models, especially the paradigm of pretrained language models (PLMs). In this paper, we present an overview of the major advances achieved in the topic of PLMs for text generation. As the preliminaries, we present the general task definition and briefly describe the mainstream architectures of PLMs for text generation. As the core content, we discuss how to adapt existing PLMs to model different input data and satisfy special properties in the generated text. We further summarize several important fine-tuning strategies for text generation. Finally, we present several future directions and conclude this paper. Our survey aims to provide text generation researchers a synthesis and pointer to related research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Text generation is one of the most exciting NLP topics.<br><br>It will power some of the most creative machine learning based applications and products.<br><br>This new survey paper discusses how pretrained language models have improved text generation capabilities.<a href="https://t.co/G2DgdJTyGr">https://t.co/G2DgdJTyGr</a> <a href="https://t.co/6sN6O8j73U">pic.twitter.com/6sN6O8j73U</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1396786741701484549?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Intriguing Properties of Vision Transformers

Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang

- retweets: 1256, favorites: 220 (05/25/2021 07:20:28)

- links: [abs](https://arxiv.org/abs/2105.10497) | [pdf](https://arxiv.org/pdf/2105.10497)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Vision transformers (ViT) have demonstrated impressive performance across various machine vision problems. These models are based on multi-head self-attention mechanisms that can flexibly attend to a sequence of image patches to encode contextual cues. An important question is how such flexibility in attending image-wide context conditioned on a given patch can facilitate handling nuisances in natural images e.g., severe occlusions, domain shifts, spatial permutations, adversarial and natural perturbations. We systematically study this question via an extensive set of experiments encompassing three ViT families and comparisons with a high-performing convolutional neural network (CNN). We show and analyze the following intriguing properties of ViT: (a) Transformers are highly robust to severe occlusions, perturbations and domain shifts, e.g., retain as high as 60% top-1 accuracy on ImageNet even after randomly occluding 80% of the image content. (b) The robust performance to occlusions is not due to a bias towards local textures, and ViTs are significantly less biased towards textures compared to CNNs. When properly trained to encode shape-based features, ViTs demonstrate shape recognition capability comparable to that of human visual system, previously unmatched in the literature. (c) Using ViTs to encode shape representation leads to an interesting consequence of accurate semantic segmentation without pixel-level supervision. (d) Off-the-shelf features from a single ViT model can be combined to create a feature ensemble, leading to high accuracy rates across a range of classification datasets in both traditional and few-shot learning paradigms. We show effective features of ViTs are due to flexible and dynamic receptive fields possible via the self-attention mechanism.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Intriguing Properties of Vision Transformers<br>pdf: <a href="https://t.co/mMujDE3PQF">https://t.co/mMujDE3PQF</a><br>abs: <a href="https://t.co/3crBQpq9Qd">https://t.co/3crBQpq9Qd</a><br><br>advantages of ViTs over CNNs for occlusion handling, robustness to distributional shifts and patch permutations, automatic segmentation with pixel supervision <a href="https://t.co/3u4X0rl033">pic.twitter.com/3u4X0rl033</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1396630379918921728?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. A GAN-Like Approach for Physics-Based Imitation Learning and Interactive  Character Control

Pei Xu, Ioannis Karamouzas

- retweets: 443, favorites: 111 (05/25/2021 07:20:28)

- links: [abs](https://arxiv.org/abs/2105.10066) | [pdf](https://arxiv.org/pdf/2105.10066)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a simple and intuitive approach for interactive control of physically simulated characters. Our work builds upon generative adversarial networks (GAN) and reinforcement learning, and introduces an imitation learning framework where an ensemble of classifiers and an imitation policy are trained in tandem given pre-processed reference clips. The classifiers are trained to discriminate the reference motion from the motion generated by the imitation policy, while the policy is rewarded for fooling the discriminators. Using our GAN-based approach, multiple motor control policies can be trained separately to imitate different behaviors. In runtime, our system can respond to external control signal provided by the user and interactively switch between different policies. Compared to existing methods, our proposed approach has the following attractive properties: 1) achieves state-of-the-art imitation performance without manually designing and fine tuning a reward function; 2) directly controls the character without having to track any target reference pose explicitly or implicitly through a phase state; and 3) supports interactive policy switching without requiring any motion generation or motion matching mechanism. We highlight the applicability of our approach in a range of imitation and interactive control tasks, while also demonstrating its ability to withstand external perturbations as well as to recover balance. Overall, our approach generates high-fidelity motion, has low runtime cost, and can be easily integrated into interactive applications and games.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A GAN-Like Approach for Physics-Based Imitation Learning and Interactive Character Control<br>pdf: <a href="https://t.co/4iIh3PTHvc">https://t.co/4iIh3PTHvc</a><br>abs: <a href="https://t.co/spZXcutrBA">https://t.co/spZXcutrBA</a> <a href="https://t.co/VFHkNhfxpQ">pic.twitter.com/VFHkNhfxpQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1396641554941480960?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Driving-Signal Aware Full-Body Avatars

Timur Bagautdinov, Chenglei Wu, Tomas Simon, Fabian Prada, Takaaki Shiratori, Shih-En Wei, Weipeng Xu, Yaser Sheikh, Jason Saragih

- retweets: 144, favorites: 71 (05/25/2021 07:20:28)

- links: [abs](https://arxiv.org/abs/2105.10441) | [pdf](https://arxiv.org/pdf/2105.10441)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a learning-based method for building driving-signal aware full-body avatars. Our model is a conditional variational autoencoder that can be animated with incomplete driving signals, such as human pose and facial keypoints, and produces a high-quality representation of human geometry and view-dependent appearance. The core intuition behind our method is that better drivability and generalization can be achieved by disentangling the driving signals and remaining generative factors, which are not available during animation. To this end, we explicitly account for information deficiency in the driving signal by introducing a latent space that exclusively captures the remaining information, thus enabling the imputation of the missing factors required during full-body animation, while remaining faithful to the driving signal. We also propose a learnable localized compression for the driving signal which promotes better generalization, and helps minimize the influence of global chance-correlations often found in real datasets. For a given driving signal, the resulting variational model produces a compact space of uncertainty for missing factors that allows for an imputation strategy best suited to a particular application. We demonstrate the efficacy of our approach on the challenging problem of full-body animation for virtual telepresence with driving signals acquired from minimal sensors placed in the environment and mounted on a VR-headset.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Driving-Signal Aware Full-Body Avatars<br>pdf: <a href="https://t.co/di2UewZPpc">https://t.co/di2UewZPpc</a><br>abs: <a href="https://t.co/OXufMHyvvy">https://t.co/OXufMHyvvy</a><br><br>method for building high-quality photorealistic full-body avatars, integrates in its construction, the specific<br>modality of driving signal that is available during the model’s use <a href="https://t.co/8zepa9XHZo">pic.twitter.com/8zepa9XHZo</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1396644098585542659?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Towards Realization of Augmented Intelligence in Dermatology: Advances  and Future Directions

Roxana Daneshjou, Carrie Kovarik, Justin M Ko

- retweets: 74, favorites: 44 (05/25/2021 07:20:28)

- links: [abs](https://arxiv.org/abs/2105.10477) | [pdf](https://arxiv.org/pdf/2105.10477)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

Artificial intelligence (AI) algorithms using deep learning have advanced the classification of skin disease images; however these algorithms have been mostly applied "in silico" and not validated clinically. Most dermatology AI algorithms perform binary classification tasks (e.g. malignancy versus benign lesions), but this task is not representative of dermatologists' diagnostic range. The American Academy of Dermatology Task Force on Augmented Intelligence published a position statement emphasizing the importance of clinical validation to create human-computer synergy, termed augmented intelligence (AuI). Liu et al's recent paper, "A deep learning system for differential diagnosis of skin diseases" represents a significant advancement of AI in dermatology, bringing it closer to clinical impact. However, significant issues must be addressed before this algorithm can be integrated into clinical workflow. These issues include accurate and equitable model development, defining and assessing appropriate clinical outcomes, and real-world integration.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">When a version of Google&#39;s Derm Assist was published in Nature Medicine, Justin Ko, <a href="https://twitter.com/carriekovarik?ref_src=twsrc%5Etfw">@carriekovarik</a>, and I wrote a response praising the advances but also expressing concern about clinical applicability (due to lack of representation &amp; biopsied lesions). <a href="https://t.co/KZ4WCDe1m8">https://t.co/KZ4WCDe1m8</a></p>&mdash; Roxana Daneshjou MD/PhD (@RoxanaDaneshjou) <a href="https://twitter.com/RoxanaDaneshjou/status/1396630221688766472?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Photonic single perceptron at Giga-OP/s speeds with Kerr microcombs for  scalable optical neural networks

Mengxi Tan, Xingyuan Xu, David J. Moss

- retweets: 64, favorites: 24 (05/25/2021 07:20:29)

- links: [abs](https://arxiv.org/abs/2105.10407) | [pdf](https://arxiv.org/pdf/2105.10407)
- [eess.SP](https://arxiv.org/list/eess.SP/recent) | [cs.ET](https://arxiv.org/list/cs.ET/recent) | [physics.app-ph](https://arxiv.org/list/physics.app-ph/recent) | [physics.optics](https://arxiv.org/list/physics.optics/recent)

Optical artificial neural networks (ONNs) have significant potential for ultra-high computing speed and energy efficiency. We report a novel approach to ONNs that uses integrated Kerr optical microcombs. This approach is programmable and scalable and is capable of reaching ultrahigh speeds. We demonstrate the basic building block ONNs, a single neuron perceptron, by mapping synapses onto 49 wavelengths to achieve an operating speed of 11.9 x 109 operations per second, or GigaOPS, at 8 bits per operation, which equates to 95.2 gigabits/s (Gbps). We test the perceptron on handwritten digit recognition and cancer cell detection, achieving over 90% and 85% accuracy, respectively. By scaling the perceptron to a deep learning network using off the shelf telecom technology we can achieve high throughput operation for matrix multiplication for real-time massive data processing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Researchers have developed an optical neural network consisting of a single perceptron that operates with an integrated optical Kerr micro-comb source, which achieves a single processor throughput speed of 95.2 Gigabits/s.<a href="https://t.co/8f8t3nHabL">https://t.co/8f8t3nHabL</a> <a href="https://t.co/Zj6leYt1kS">pic.twitter.com/Zj6leYt1kS</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1396672592489156615?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A Non-Linear Structural Probe

Jennifer C. White, Tiago Pimentel, Naomi Saphra, Ryan Cotterell

- retweets: 38, favorites: 22 (05/25/2021 07:20:29)

- links: [abs](https://arxiv.org/abs/2105.10185) | [pdf](https://arxiv.org/pdf/2105.10185)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Probes are models devised to investigate the encoding of knowledge -- e.g. syntactic structure -- in contextual representations. Probes are often designed for simplicity, which has led to restrictions on probe design that may not allow for the full exploitation of the structure of encoded information; one such restriction is linearity. We examine the case of a structural probe (Hewitt and Manning, 2019), which aims to investigate the encoding of syntactic structure in contextual representations through learning only linear transformations. By observing that the structural probe learns a metric, we are able to kernelize it and develop a novel non-linear variant with an identical number of parameters. We test on 6 languages and find that the radial-basis function (RBF) kernel, in conjunction with regularization, achieves a statistically significant improvement over the baseline in all languages -- implying that at least part of the syntactic knowledge is encoded non-linearly. We conclude by discussing how the RBF kernel resembles BERT's self-attention layers and speculate that this resemblance leads to the RBF-based probe's stronger performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out <a href="https://twitter.com/JenniferCWhite?ref_src=twsrc%5Etfw">@JenniferCWhite</a>&#39;s new <a href="https://twitter.com/NAACLHLT?ref_src=twsrc%5Etfw">@NAACLHLT</a> paper where she derives a kernelization of the influential structural probing paper by <a href="https://twitter.com/johnhewtt?ref_src=twsrc%5Etfw">@johnhewtt</a> and <a href="https://twitter.com/chrmanning?ref_src=twsrc%5Etfw">@chrmanning</a>. <br><br>Paper: <a href="https://t.co/KBVhkHFmzM">https://t.co/KBVhkHFmzM</a></p>&mdash; Ryan D. Cotterell (@ryandcotterell) <a href="https://twitter.com/ryandcotterell/status/1396825425339367424?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. VLM: Task-agnostic Video-Language Model Pre-training for Video  Understanding

Hu Xu, Gargi Ghosh, Po-Yao Huang, Prahal Arora, Masoumeh Aminzadeh, Christoph Feichtenhofer, Florian Metze, Luke Zettlemoyer

- retweets: 12, favorites: 42 (05/25/2021 07:20:29)

- links: [abs](https://arxiv.org/abs/2105.09996) | [pdf](https://arxiv.org/pdf/2105.09996)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

We present a simplified, task-agnostic multi-modal pre-training approach that can accept either video or text input, or both for a variety of end tasks. Existing pre-training are task-specific by adopting either a single cross-modal encoder that requires both modalities, limiting their use for retrieval-style end tasks or more complex multitask learning with two unimodal encoders, limiting early cross-modal fusion. We instead introduce new pretraining masking schemes that better mix across modalities (e.g. by forcing masks for text to predict the closest video embeddings) while also maintaining separability (e.g. unimodal predictions are sometimes required, without using all the input). Experimental results show strong performance across a wider range of tasks than any previous methods, often outperforming task-specific pre-training.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VLM: Task-agnostic Video-Language Model Pre-training<br>for Video Understanding<br>pdf: <a href="https://t.co/y3ooMmNRh2">https://t.co/y3ooMmNRh2</a><br>abs: <a href="https://t.co/e9B8LZK36u">https://t.co/e9B8LZK36u</a><br><br>task-agnostic pre-training with new masking schemes enable training of single masked language model that accepts either video and/or text input <a href="https://t.co/sIV8MV0el6">pic.twitter.com/sIV8MV0el6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1396631507893096448?ref_src=twsrc%5Etfw">May 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



