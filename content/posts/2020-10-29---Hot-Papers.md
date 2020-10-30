---
title: Hot Papers 2020-10-29
date: 2020-10-30T09:57:41.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-29"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-29"
socialImage: "/media/flying-marine.jpg"

---

# 1. Language ID in the Wild: Unexpected Challenges on the Path to a  Thousand-Language Web Text Corpus

Isaac Caswell, Theresa Breiner, Daan van Esch, Ankur Bapna

- retweets: 1058, favorites: 88 (10/30/2020 09:57:41)

- links: [abs](https://arxiv.org/abs/2010.14571) | [pdf](https://arxiv.org/pdf/2010.14571)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Large text corpora are increasingly important for a wide variety of Natural Language Processing (NLP) tasks, and automatic language identification (LangID) is a core technology needed to collect such datasets in a multilingual context. LangID is largely treated as solved in the literature, with models reported that achieve over 90% average F1 on as many as 1,366 languages. We train LangID models on up to 1,629 languages with comparable quality on held-out test sets, but find that human-judged LangID accuracy for web-crawl text corpora created using these models is only around 5% for many lower-resource languages, suggesting a need for more robust evaluation. Further analysis revealed a variety of error modes, arising from domain mismatch, class imbalance, language similarity, and insufficiently expressive models. We propose two classes of techniques to mitigate these errors: wordlist-based tunable-precision filters (for which we release curated lists in about 500 languages) and transformer-based semi-supervised LangID models, which increase median dataset precision from 5.5% to 71.2%. These techniques enable us to create an initial data set covering 100K or more relatively clean sentences in each of 500+ languages, paving the way towards a 1,000-language web text corpus.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What do we need to scale NLP research to 1000 languages? We started off with a goal to build a monolingual corpus in 1000 languages by mining data from the web. Here’s our work documenting our struggles with Language Identification (LangID): <a href="https://t.co/mNVkQf5alt">https://t.co/mNVkQf5alt</a><br>1/8 <a href="https://t.co/s5e97ui7mr">pic.twitter.com/s5e97ui7mr</a></p>&mdash; Isaac R Caswell (@iseeaswell) <a href="https://twitter.com/iseeaswell/status/1321849209633468418?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Unsupervised Domain Adaptation for Visual Navigation

Shangda Li, Devendra Singh Chaplot, Yao-Hung Hubert Tsai, Yue Wu, Louis-Philippe Morency, Ruslan Salakhutdinov

- retweets: 529, favorites: 92 (10/30/2020 09:57:41)

- links: [abs](https://arxiv.org/abs/2010.14543) | [pdf](https://arxiv.org/pdf/2010.14543)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Advances in visual navigation methods have led to intelligent embodied navigation agents capable of learning meaningful representations from raw RGB images and perform a wide variety of tasks involving structural and semantic reasoning. However, most learning-based navigation policies are trained and tested in simulation environments. In order for these policies to be practically useful, they need to be transferred to the real-world. In this paper, we propose an unsupervised domain adaptation method for visual navigation. Our method translates the images in the target domain to the source domain such that the translation is consistent with the representations learned by the navigation policy. The proposed method outperforms several baselines across two different navigation tasks in simulation. We further show that our method can be used to transfer the navigation policies learned in simulation to the real world.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper on Unsupervised Domain Adaptation for Visual Navigation! We transfer navigation policies from simulation to the real-world using policy-based image translation.<br><br>Arxiv:<a href="https://t.co/Sw9x5doNR0">https://t.co/Sw9x5doNR0</a><br><br>with S. Li, Y.H. Tsai, Y. Wu, L.P. Morency, R. Salakhutdinov <a href="https://twitter.com/rsalakhu?ref_src=twsrc%5Etfw">@rsalakhu</a> <a href="https://t.co/wZIy14JH2a">pic.twitter.com/wZIy14JH2a</a></p>&mdash; Devendra Chaplot (@dchaplot) <a href="https://twitter.com/dchaplot/status/1321640168785403905?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer  through Optimal Transport

Eric Risser

- retweets: 306, favorites: 76 (10/30/2020 09:57:42)

- links: [abs](https://arxiv.org/abs/2010.14702) | [pdf](https://arxiv.org/pdf/2010.14702)
- [cs.GR](https://arxiv.org/list/cs.GR/recent)

This paper presents a light-weight, high-quality texture synthesis algorithm that easily generalizes to other applications such as style transfer and texture mixing. We represent texture features through the deep neural activation vectors within the bottleneck layer of an auto-encoder and frame the texture synthesis problem as optimal transport between the activation values of the image being synthesized and those of an exemplar texture. To find this optimal transport mapping, we utilize an N-dimensional probability density function (PDF) transfer process that iterates over multiple random rotations of the PDF basis and matches the 1D marginal distributions across each dimension. This achieves quality and flexibility on par with expensive back-propagation based neural texture synthesis methods, but with the potential of achieving interactive rates. We demonstrate that first order statistics offer a more robust representation for texture than the second order statistics that are used today. We propose an extension of this algorithm that reduces the dimensionality of the neural feature space. We utilize a multi-scale coarse-to-fine synthesis pyramid to capture and preserve larger image features; unify color and style transfer under one framework; and further augment this system with a novel masking scheme that re-samples and re-weights the feature distribution for user-guided texture painting and targeted style transfer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport<br>pdf: <a href="https://t.co/UrttUgStML">https://t.co/UrttUgStML</a><br>abs: <a href="https://t.co/7pO3hrYni8">https://t.co/7pO3hrYni8</a> <a href="https://t.co/CQK8gZ1yVg">pic.twitter.com/CQK8gZ1yVg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1321626995755802625?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Training Generative Adversarial Networks by Solving Ordinary  Differential Equations

Chongli Qin, Yan Wu, Jost Tobias Springenberg, Andrew Brock, Jeff Donahue, Timothy P. Lillicrap, Pushmeet Kohli

- retweets: 240, favorites: 72 (10/30/2020 09:57:42)

- links: [abs](https://arxiv.org/abs/2010.15040) | [pdf](https://arxiv.org/pdf/2010.15040)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The instability of Generative Adversarial Network (GAN) training has frequently been attributed to gradient descent. Consequently, recent methods have aimed to tailor the models and training procedures to stabilise the discrete updates. In contrast, we study the continuous-time dynamics induced by GAN training. Both theory and toy experiments suggest that these dynamics are in fact surprisingly stable. From this perspective, we hypothesise that instabilities in training GANs arise from the integration error in discretising the continuous dynamics. We experimentally verify that well-known ODE solvers (such as Runge-Kutta) can stabilise training - when combined with a regulariser that controls the integration error. Our approach represents a radical departure from previous methods which typically use adaptive optimisation and stabilisation techniques that constrain the functional space (e.g. Spectral Normalisation). Evaluation on CIFAR-10 and ImageNet shows that our method outperforms several strong baselines, demonstrating its efficacy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Training Generative Adversarial Networks by Solving Ordinary Differential Equations<br>pdf: <a href="https://t.co/I2tftGsCPJ">https://t.co/I2tftGsCPJ</a><br>abs: <a href="https://t.co/f9KB9SgObb">https://t.co/f9KB9SgObb</a><br>github: <a href="https://t.co/I9BtFl2DD5">https://t.co/I9BtFl2DD5</a> <a href="https://t.co/y4Qzao8Y33">pic.twitter.com/y4Qzao8Y33</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1321624779506540544?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Scaling Laws for Autoregressive Generative Modeling

Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, Chris Hallacy, Benjamin Mann, Alec Radford, Aditya Ramesh, Nick Ryder, Daniel M. Ziegler, John Schulman, Dario Amodei, Sam McCandlish

- retweets: 214, favorites: 96 (10/30/2020 09:57:42)

- links: [abs](https://arxiv.org/abs/2010.14701) | [pdf](https://arxiv.org/pdf/2010.14701)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We identify empirical scaling laws for the cross-entropy loss in four domains: generative image modeling, video modeling, multimodal image$\leftrightarrow$text models, and mathematical problem solving. In all cases autoregressive Transformers smoothly improve in performance as model size and compute budgets increase, following a power-law plus constant scaling law. The optimal model size also depends on the compute budget through a power-law, with exponents that are nearly universal across all data domains.   The cross-entropy loss has an information theoretic interpretation as $S($True$) + D_{\mathrm{KL}}($True$||$Model$)$, and the empirical scaling laws suggest a prediction for both the true data distribution's entropy and the KL divergence between the true and model distributions. With this interpretation, billion-parameter Transformers are nearly perfect models of the YFCC100M image distribution downsampled to an $8\times 8$ resolution, and we can forecast the model size needed to achieve any given reducible loss (ie $D_{\mathrm{KL}}$) in nats/image for other resolutions.   We find a number of additional scaling laws in specific domains: (a) we identify a scaling relation for the mutual information between captions and images in multimodal models, and show how to answer the question "Is a picture worth a thousand words?"; (b) in the case of mathematical problem solving, we identify scaling laws for model performance when extrapolating beyond the training distribution; (c) we finetune generative image models for ImageNet classification and find smooth scaling of the classification loss and error rate, even as the generative loss levels off. Taken together, these results strengthen the case that scaling laws have important implications for neural network performance, including on downstream tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Laws for Autoregressive Generative Modeling<br><br>- Autoregressive Transformer follows robust power law w/ the same model size scaling exponent over various modalities.<br><br>- Fine-tuning a pretrained model follows a similar power-law. <a href="https://t.co/WkUE55TOy2">https://t.co/WkUE55TOy2</a> <a href="https://t.co/BK6cruQCx6">pic.twitter.com/BK6cruQCx6</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1321629013136171010?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Laws for Autoregressive Generative Modeling<br>pdf: <a href="https://t.co/HKxBK2xmlr">https://t.co/HKxBK2xmlr</a><br>abs: <a href="https://t.co/HSaPdhKtp1">https://t.co/HSaPdhKtp1</a> <a href="https://t.co/mYVnprCfqI">pic.twitter.com/mYVnprCfqI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1321618878171107328?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. The geometry of integration in text classification RNNs

Kyle Aitken, Vinay V. Ramasesh, Ankush Garg, Yuan Cao, David Sussillo, Niru Maheswaranathan

- retweets: 210, favorites: 95 (10/30/2020 09:57:42)

- links: [abs](https://arxiv.org/abs/2010.15114) | [pdf](https://arxiv.org/pdf/2010.15114)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Despite the widespread application of recurrent neural networks (RNNs) across a variety of tasks, a unified understanding of how RNNs solve these tasks remains elusive. In particular, it is unclear what dynamical patterns arise in trained RNNs, and how those patterns depend on the training dataset or task. This work addresses these questions in the context of a specific natural language processing task: text classification. Using tools from dynamical systems analysis, we study recurrent networks trained on a battery of both natural and synthetic text classification tasks. We find the dynamics of these trained RNNs to be both interpretable and low-dimensional. Specifically, across architectures and datasets, RNNs accumulate evidence for each class as they process the text, using a low-dimensional attractor manifold as the underlying mechanism. Moreover, the dimensionality and geometry of the attractor manifold are determined by the structure of the training dataset; in particular, we describe how simple word-count statistics computed on the training dataset can be used to predict these properties. Our observations span multiple architectures and datasets, reflecting a common mechanism RNNs employ to perform text classification. To the degree that integration of evidence towards a decision is a common computational primitive, this work lays the foundation for using dynamical systems techniques to study the inner workings of RNNs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work out on arXiv (<a href="https://t.co/sMBuTbs3qI">https://t.co/sMBuTbs3qI</a>) about the geometry of integration in RNNs. This was led by Vinay Ramasesh (<a href="https://twitter.com/vinayramasesh?ref_src=twsrc%5Etfw">@vinayramasesh</a>) and Kyle Aitken (<a href="https://twitter.com/kyle__aitken?ref_src=twsrc%5Etfw">@kyle__aitken</a>), with additional co-authors Ankush Garg, Yuan Cao, and David Sussillo (<a href="https://twitter.com/sussillo?ref_src=twsrc%5Etfw">@sussillo</a>). <a href="https://twitter.com/hashtag/tweeprint?src=hash&amp;ref_src=twsrc%5Etfw">#tweeprint</a> below! 🧵 <a href="https://t.co/gUTFoH77jE">pic.twitter.com/gUTFoH77jE</a></p>&mdash; Niru Maheswaranathan (@niru_m) <a href="https://twitter.com/niru_m/status/1321890792097550336?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. How to Not Get Caught When You Launder Money on Blockchain?

Cuneyt G. Akcora, Sudhanva Purusotham, Yulia R. Gel, Mitchell Krawiec-Thayer, Murat Kantarcioglu

- retweets: 157, favorites: 66 (10/30/2020 09:57:42)

- links: [abs](https://arxiv.org/abs/2010.15082) | [pdf](https://arxiv.org/pdf/2010.15082)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

The number of blockchain users has tremendously grown in recent years. As an unintended consequence, e-crime transactions on blockchains has been on the rise. Consequently, public blockchains have become a hotbed of research for developing AI tools to detect and trace users and transactions that are related to e-crime.   We argue that following a few select strategies can make money laundering on blockchain virtually undetectable with most of the existing tools and algorithms. As a result, the effective combating of e-crime activities involving cryptocurrencies requires the development of novel analytic methodology in AI.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">ペーパーのタイトルからすごいなこれ<br>逮捕されることなく仮想通貨を用いて資金洗浄する方法<br>How to Not Get Caught When You Launder Money on Blockchain?<a href="https://t.co/YHf6N0LArA">https://t.co/YHf6N0LArA</a></p>&mdash; SttyK (してぃーきっず) (@SttyK) <a href="https://twitter.com/SttyK/status/1321714561985081345?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Nice OPSEC+Common sense lessons when you want to Launder money on blockchains =)  - <a href="https://t.co/yTFMn1DCcS">https://t.co/yTFMn1DCcS</a> How to Not Get Caught When You Launder Money on Blockchain?  cc <a href="https://twitter.com/thegrugq?ref_src=twsrc%5Etfw">@thegrugq</a></p>&mdash; ak1010 (@ak1010) <a href="https://twitter.com/ak1010/status/1321707731884138496?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Character Entropy in Modern and Historical Texts: Comparison Metrics for  an Undeciphered Manuscript

Luke Lindemann, Claire Bowern

- retweets: 102, favorites: 39 (10/30/2020 09:57:43)

- links: [abs](https://arxiv.org/abs/2010.14697) | [pdf](https://arxiv.org/pdf/2010.14697)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

This paper outlines the creation of three corpora for multilingual comparison and analysis of the Voynich manuscript: a corpus of Voynich texts partitioned by Currier language, scribal hand, and transcription system, a corpus of 294 language samples compiled from Wikipedia, and a corpus of eighteen transcribed historical texts in eight languages. These corpora will be utilized in subsequent work by the Voynich Working Group at Yale University.   We demonstrate the utility of these corpora for studying characteristics of the Voynich script and language, with an analysis of conditional character entropy in Voynichese. We discuss the interaction between character entropy and language, script size and type, glyph compositionality, scribal conventions and abbreviations, positional character variants, and bigram frequency.   This analysis characterizes the interaction between script compositionality, character size, and predictability. We show that substantial manipulations of glyph composition are not sufficient to align conditional entropy levels with natural languages. The unusually predictable nature of the Voynichese script is not attributable to a particular script or transcription system, underlying language, or substitution cipher. Voynichese is distinct from every comparison text in our corpora because character placement is highly constrained within the word, and this may indicate the loss of phonemic distinctions from the underlying language.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">very pleased to announce that Luke Lindemann and my latest Voynich paper is ready. See it at <a href="https://t.co/E0OQOP8efI">https://t.co/E0OQOP8efI</a>; deep dive into character entropy</p>&mdash; Claire Bowern (@anggarrgoon) <a href="https://twitter.com/anggarrgoon/status/1321832514353270785?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Tree-structured Ising models can be learned efficiently

Constantinos Daskalakis, Qinxuan Pan

- retweets: 72, favorites: 26 (10/30/2020 09:57:43)

- links: [abs](https://arxiv.org/abs/2010.14864) | [pdf](https://arxiv.org/pdf/2010.14864)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We provide the first polynomial-sample and polynomial-time algorithm for learning tree-structured Ising models. In particular, we show that $n$-variable tree-structured Ising models can be learned computationally-efficiently to within total variation distance~$\epsilon$ from an optimal $O(n \log n/\epsilon^2)$ samples, where $O(.)$ hides an absolute constant which does not depend on the model being learned -- neither its tree nor the magnitude of its edge strengths, on which we place no assumptions. Our guarantees hold, in fact, for the celebrated Chow-Liu [1968] algorithm, using the plug-in estimator for mutual information. While this (or any other) algorithm may fail to identify the structure of the underlying model correctly from a finite sample, we show that it will still learn a tree-structured model that is close to the true one in TV distance, a guarantee called "proper learning."   Prior to our work there were no known sample- and time-efficient algorithms for learning (properly or non-properly) arbitrary tree-structured graphical models. In particular, our guarantees cannot be derived from known results for the Chow-Liu algorithm and the ensuing literature on learning graphical models, including a recent renaissance of algorithms on this learning challenge, which only yield asymptotic consistency results, or sample-inefficient and/or time-inefficient algorithms, unless further assumptions are placed on the graphical model, such as bounds on the "strengths" of the model's edges. While we establish guarantees for a widely known and simple algorithm, the analysis that this algorithm succeeds is quite complex, requiring a hierarchical classification of the edges into layers with different reconstruction guarantees, depending on their strength, combined with delicate uses of the subadditivity of the squared Hellinger distance over graphical models to control the error accumulation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">There are many algorithms for learning Ising models. Yet there is no sample-efficient and time-efficient one that works without assumptions, even for tree-structured models. We obtain the first such result in fresh work w/ <a href="https://twitter.com/PanQinxuan?ref_src=twsrc%5Etfw">@PanQinxuan</a>:<a href="https://t.co/qKBLrlEBHv">https://t.co/qKBLrlEBHv</a></p>&mdash; Constantinos Daskalakis (@KonstDaskalakis) <a href="https://twitter.com/KonstDaskalakis/status/1321963003906961408?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Seen and Unseen emotional style transfer for voice conversion with a new  emotional speech dataset

Kun Zhou, Berrak Sisman, Rui Liu, Haizhou Li

- retweets: 56, favorites: 34 (10/30/2020 09:57:43)

- links: [abs](https://arxiv.org/abs/2010.14794) | [pdf](https://arxiv.org/pdf/2010.14794)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Emotional voice conversion aims to transform emotional prosody in speech while preserving the linguistic content and speaker identity. Prior studies show that it is possible to disentangle emotional prosody using an encoder-decoder network conditioned on discrete representation, such as one-hot emotion labels. Such networks learn to remember a fixed set of emotional styles. In this paper, we propose a novel framework based on variational auto-encoding Wasserstein generative adversarial network (VAW-GAN), which makes use of a pre-trained speech emotion recognition (SER) model to transfer emotional style during training and at run-time inference. In this way, the network is able to transfer both seen and unseen emotional style to a new utterance. We show that the proposed framework achieves remarkable performance by consistently outperforming the baseline framework. This paper also marks the release of an emotional speech dataset (ESD) for voice conversion, which has multiple speakers and languages.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We release an emotional speech dataset (ESD) for voice conversion, which has multiple speakers and languages. ESD has parallel utterances from 10 Mandarin speakers, and 10 English speakers with 5 emotional states (angry, happy, sad, surprise, natural).<br>📎<a href="https://t.co/flqVUvxuLI">https://t.co/flqVUvxuLI</a></p>&mdash; Berrak Sisman (@berraksismann) <a href="https://twitter.com/berraksismann/status/1321617239338934273?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Learning to Represent Action Values as a Hypergraph on the Action  Vertices

Arash Tavakoli, Mehdi Fatemi, Petar Kormushev

- retweets: 30, favorites: 37 (10/30/2020 09:57:43)

- links: [abs](https://arxiv.org/abs/2010.14680) | [pdf](https://arxiv.org/pdf/2010.14680)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Action-value estimation is a critical component of many reinforcement learning (RL) methods whereby sample complexity relies heavily on how fast a good estimator for action value can be learned. By viewing this problem through the lens of representation learning, good representations of both state and action can facilitate action-value estimation. While advances in deep learning have seamlessly driven progress in learning state representations, given the specificity of the notion of agency to RL, little attention has been paid to learning action representations. We conjecture that leveraging the combinatorial structure of multi-dimensional action spaces is a key ingredient for learning good representations of action. To test this, we set forth the action hypergraph networks framework---a class of functions for learning action representations with a relational inductive bias. Using this framework we realise an agent class based on a combination with deep Q-networks, which we dub hypergraph Q-networks. We show the effectiveness of our approach on a myriad of domains: illustrative prediction problems under minimal confounding effects, Atari 2600 games, and physical control benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am thrilled to share Action Hypergraph Networks, a class of models for learning action representations! 🐙 🎉<br><br>Combine in succession with any model for learning state representations (e.g. CNN, RNN, GNN) &amp; train without any change to the RL loss.<br><br>Paper: <a href="https://t.co/U1BVEYPetc">https://t.co/U1BVEYPetc</a> <a href="https://t.co/WDOVgNUCGM">pic.twitter.com/WDOVgNUCGM</a></p>&mdash; Arash Tavakoli (@arshtvk) <a href="https://twitter.com/arshtvk/status/1321826716159979520?ref_src=twsrc%5Etfw">October 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Learning to be Safe: Deep RL with a Safety Critic

Krishnan Srinivasan, Benjamin Eysenbach, Sehoon Ha, Jie Tan, Chelsea Finn

- retweets: 44, favorites: 22 (10/30/2020 09:57:43)

- links: [abs](https://arxiv.org/abs/2010.14603) | [pdf](https://arxiv.org/pdf/2010.14603)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Safety is an essential component for deploying reinforcement learning (RL) algorithms in real-world scenarios, and is critical during the learning process itself. A natural first approach toward safe RL is to manually specify constraints on the policy's behavior. However, just as learning has enabled progress in large-scale development of AI systems, learning safety specifications may also be necessary to ensure safety in messy open-world environments where manual safety specifications cannot scale. Akin to how humans learn incrementally starting in child-safe environments, we propose to learn how to be safe in one set of tasks and environments, and then use that learned intuition to constrain future behaviors when learning new, modified tasks. We empirically study this form of safety-constrained transfer learning in three challenging domains: simulated navigation, quadruped locomotion, and dexterous in-hand manipulation. In comparison to standard deep RL techniques and prior approaches to safe RL, we find that our method enables the learning of new tasks and in new environments with both substantially fewer safety incidents, such as falling or dropping an object, and faster, more stable learning. This suggests a path forward not only for safer RL systems, but also for more effective RL systems.



