---
title: Hot Papers 2020-10-14
date: 2020-10-15T09:38:34.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-14"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-14"
socialImage: "/media/flying-marine.jpg"

---

# 1. The Cone of Silence: Speech Separation by Localization

Teerapat Jenrungrot, Vivek Jayaram, Steve Seitz, Ira Kemelmacher-Shlizerman

- retweets: 1936, favorites: 178 (10/15/2020 09:38:34)

- links: [abs](https://arxiv.org/abs/2010.06007) | [pdf](https://arxiv.org/pdf/2010.06007)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Given a multi-microphone recording of an unknown number of speakers talking concurrently, we simultaneously localize the sources and separate the individual speakers. At the core of our method is a deep network, in the waveform domain, which isolates sources within an angular region $\theta \pm w/2$, given an angle of interest $\theta$ and angular window size $w$. By exponentially decreasing $w$, we can perform a binary search to localize and separate all sources in logarithmic time. Our algorithm allows for an arbitrary number of potentially moving speakers at test time, including more speakers than seen during training. Experiments demonstrate state-of-the-art performance for both source separation and source localization, particularly in high levels of background noise.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Cone of Silence: Speech Separation by Localization<br>pdf: <a href="https://t.co/a7cPK82XTx">https://t.co/a7cPK82XTx</a><br>abs: <a href="https://t.co/yBGPOtsjDr">https://t.co/yBGPOtsjDr</a><br>project page: <a href="https://t.co/5mmhkbfizC">https://t.co/5mmhkbfizC</a> <a href="https://t.co/X8ZlLMbv8T">pic.twitter.com/X8ZlLMbv8T</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1316189706430943233?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Pretrained Transformers for Text Ranking: BERT and Beyond

Jimmy Lin, Rodrigo Nogueira, Andrew Yates

- retweets: 902, favorites: 134 (10/15/2020 09:38:35)

- links: [abs](https://arxiv.org/abs/2010.06467) | [pdf](https://arxiv.org/pdf/2010.06467)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

The goal of text ranking is to generate an ordered list of texts retrieved from a corpus in response to a query. Although the most common formulation of text ranking is search, instances of the task can also be found in many natural language processing applications. This survey provides an overview of text ranking with neural network architectures known as transformers, of which BERT is the best-known example. The combination of transformers and self-supervised pretraining has, without exaggeration, revolutionized the fields of natural language processing (NLP), information retrieval (IR), and beyond. In this survey, we provide a synthesis of existing work as a single point of entry for practitioners who wish to gain a better understanding of how to apply transformers to text ranking problems and researchers who wish to pursue work in this area. We cover a wide range of modern techniques, grouped into two high-level categories: transformer models that perform reranking in multi-stage ranking architectures and learned dense representations that attempt to perform ranking directly. There are two themes that pervade our survey: techniques for handling long documents, beyond the typical sentence-by-sentence processing approaches used in NLP, and techniques for addressing the tradeoff between effectiveness (result quality) and efficiency (query latency). Although transformer architectures and pretraining techniques are recent innovations, many aspects of how they are applied to text ranking are relatively well understood and represent mature techniques. However, there remain many open research questions, and thus in addition to laying out the foundations of pretrained transformers for text ranking, this survey also attempts to prognosticate where the field is heading.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share an early draft of &quot;Pretrained Transformers for Text Ranking: BERT and Beyond&quot;, our forthcoming book (tentatively, early 2021) by <a href="https://twitter.com/lintool?ref_src=twsrc%5Etfw">@lintool</a> <a href="https://twitter.com/rodrigfnogueira?ref_src=twsrc%5Etfw">@rodrigfnogueira</a> <a href="https://twitter.com/andrewyates?ref_src=twsrc%5Etfw">@andrewyates</a> <a href="https://t.co/EnZEOVp2rE">https://t.co/EnZEOVp2rE</a></p>&mdash; Jimmy Lin (@lintool) <a href="https://twitter.com/lintool/status/1316371240211361792?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. LM-Reloc: Levenberg-Marquardt Based Direct Visual Relocalization

Lukas von Stumberg, Patrick Wenzel, Nan Yang, Daniel Cremers

- retweets: 380, favorites: 75 (10/15/2020 09:38:35)

- links: [abs](https://arxiv.org/abs/2010.06323) | [pdf](https://arxiv.org/pdf/2010.06323)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present LM-Reloc -- a novel approach for visual relocalization based on direct image alignment. In contrast to prior works that tackle the problem with a feature-based formulation, the proposed method does not rely on feature matching and RANSAC. Hence, the method can utilize not only corners but any region of the image with gradients. In particular, we propose a loss formulation inspired by the classical Levenberg-Marquardt algorithm to train LM-Net. The learned features significantly improve the robustness of direct image alignment, especially for relocalization across different conditions. To further improve the robustness of LM-Net against large image baselines, we propose a pose estimation network, CorrPoseNet, which regresses the relative pose to bootstrap the direct image alignment. Evaluations on the CARLA and Oxford RobotCar relocalization tracking benchmark show that our approach delivers more accurate results than previous state-of-the-art methods while being comparable in terms of robustness.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LM-Reloc: Levenberg-Marquardt Based Direct Visual Relocalization <a href="https://t.co/lI6o1IaTcq">https://t.co/lI6o1IaTcq</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://t.co/wT0wjXVmhI">pic.twitter.com/wT0wjXVmhI</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1316227607894151168?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. TextHide: Tackling Data Privacy in Language Understanding Tasks

Yangsibo Huang, Zhao Song, Danqi Chen, Kai Li, Sanjeev Arora

- retweets: 256, favorites: 32 (10/15/2020 09:38:35)

- links: [abs](https://arxiv.org/abs/2010.06053) | [pdf](https://arxiv.org/pdf/2010.06053)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

An unsolved challenge in distributed or federated learning is to effectively mitigate privacy risks without slowing down training or reducing accuracy. In this paper, we propose TextHide aiming at addressing this challenge for natural language understanding tasks. It requires all participants to add a simple encryption step to prevent an eavesdropping attacker from recovering private text data. Such an encryption step is efficient and only affects the task performance slightly. In addition, TextHide fits well with the popular framework of fine-tuning pre-trained language models (e.g., BERT) for any sentence or sentence-pair task. We evaluate TextHide on the GLUE benchmark, and our experiments show that TextHide can effectively defend attacks on shared gradients or representations and the averaged accuracy reduction is only $1.9\%$. We also present an analysis of the security of TextHide using a conjecture about the computational intractability of a mathematical problem.   Our code is available at https://github.com/Hazelsuko07/TextHide

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How to tackle data privacy for language understanding tasks in distributed learning (without slowing down training or reducing accuracy)? Happy to share our new <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a> findings paper<br><br>w/ <a href="https://twitter.com/realZhaoSong?ref_src=twsrc%5Etfw">@realZhaoSong</a>, <a href="https://twitter.com/danqi_chen?ref_src=twsrc%5Etfw">@danqi_chen</a>, Prof. Kai Li, <a href="https://twitter.com/prfsanjeevarora?ref_src=twsrc%5Etfw">@prfsanjeevarora</a><br>paper: <a href="https://t.co/z9wZ5d2Yda">https://t.co/z9wZ5d2Yda</a> <a href="https://t.co/hspbz0hl8s">pic.twitter.com/hspbz0hl8s</a></p>&mdash; Yangsibo Huang (@YangsiboHuang) <a href="https://twitter.com/YangsiboHuang/status/1316448818867699714?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Does my multimodal model learn cross-modal interactions? It's harder to  tell than you might think!

Jack Hessel, Lillian Lee

- retweets: 134, favorites: 83 (10/15/2020 09:38:35)

- links: [abs](https://arxiv.org/abs/2010.06572) | [pdf](https://arxiv.org/pdf/2010.06572)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Modeling expressive cross-modal interactions seems crucial in multimodal tasks, such as visual question answering. However, sometimes high-performing black-box algorithms turn out to be mostly exploiting unimodal signals in the data. We propose a new diagnostic tool, empirical multimodally-additive function projection (EMAP), for isolating whether or not cross-modal interactions improve performance for a given model on a given task. This function projection modifies model predictions so that cross-modal interactions are eliminated, isolating the additive, unimodal structure. For seven image+text classification tasks (on each of which we set new state-of-the-art benchmarks), we find that, in many cases, removing cross-modal interactions results in little to no performance degradation. Surprisingly, this holds even when expressive models, with capacity to consider interactions, otherwise outperform less expressive models; thus, performance improvements, even when present, often cannot be attributed to consideration of cross-modal feature interactions. We hence recommend that researchers in multimodal machine learning report the performance not only of unimodal baselines, but also the EMAP of their best-performing model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/EMNLP2020?src=hash&amp;ref_src=twsrc%5Etfw">#EMNLP2020</a> paper is out!<a href="https://t.co/hPQPBVu07H">https://t.co/hPQPBVu07H</a><br><br>TL;DR If you compare two models where A is more expressive than B, if A outperforms B, it&#39;s often not /because/ of the increased expressivity. Our method diagnoses this for multimodal classifiers.<br><br>w/ Lillian Lee<br><br>Thread 👇 <a href="https://t.co/RyZmfpjgYz">pic.twitter.com/RyZmfpjgYz</a></p>&mdash; Jack Hessel (@jmhessel) <a href="https://twitter.com/jmhessel/status/1316411530162782208?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A variational autoencoder for music generation controlled by tonal  tension

Rui Guo, Ivor Simpson, Thor Magnusson, Chris Kiefer, Dorien Herremans

- retweets: 144, favorites: 56 (10/15/2020 09:38:35)

- links: [abs](https://arxiv.org/abs/2010.06230) | [pdf](https://arxiv.org/pdf/2010.06230)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.SC](https://arxiv.org/list/cs.SC/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Many of the music generation systems based on neural networks are fully autonomous and do not offer control over the generation process. In this research, we present a controllable music generation system in terms of tonal tension. We incorporate two tonal tension measures based on the Spiral Array Tension theory into a variational autoencoder model. This allows us to control the direction of the tonal tension throughout the generated piece, as well as the overall level of tonal tension. Given a seed musical fragment, stemming from either the user input or from directly sampling from the latent space, the model can generate variations of this original seed fragment with altered tonal tension. This altered music still resembles the seed music rhythmically, but the pitch of the notes are changed to match the desired tonal tension as conditioned by the user.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A variational autoencoder for music generation controlled by tonal tension<br>pdf: <a href="https://t.co/BtvYOW40z7">https://t.co/BtvYOW40z7</a><br>abs: <a href="https://t.co/HIa5XEpxPs">https://t.co/HIa5XEpxPs</a><br>project page: <a href="https://t.co/pgg819SHmB">https://t.co/pgg819SHmB</a><br>github: <a href="https://t.co/JX6v2dhiTD">https://t.co/JX6v2dhiTD</a> <a href="https://t.co/o8zA5joqT8">pic.twitter.com/o8zA5joqT8</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1316187960396709888?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. TM-NET: Deep Generative Networks for Textured Meshes

Lin Gao, Tong Wu, Yu-Jie Yuan, Ming-Xian Lin, Yu-Kun Lai, Hao Zhang

- retweets: 132, favorites: 59 (10/15/2020 09:38:36)

- links: [abs](https://arxiv.org/abs/2010.06217) | [pdf](https://arxiv.org/pdf/2010.06217)
- [cs.GR](https://arxiv.org/list/cs.GR/recent)

We introduce TM-NET, a novel deep generative model capable of generating meshes with detailed textures, as well as synthesizing plausible textures for a given shape. To cope with complex geometry and structure, inspired by the recently proposed SDM-NET, our method produces texture maps for individual parts, each as a deformed box, which further leads to a natural UV map with minimum distortions. To provide a generic framework for different application scenarios, we encode geometry and texture separately and learn the texture probability distribution conditioned on the geometry. We address challenges for textured mesh generation by sampling textures on the conditional probability distribution. Textures also often contain high-frequency details (e.g. wooden texture), and we encode them effectively with a variational autoencoder (VAE) using dictionary-based vector quantization. We also exploit the transparency in the texture as an effective approach to modeling highly complicated topology and geometry. This work is the first to synthesize high-quality textured meshes for shapes with complex structures. Extensive experiments show that our method produces high-quality textures, and avoids the inconsistency issue common for novel view synthesis methods where textured shapes from different views are generated separately.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TM-NET: Deep Generative Networks for Textured Meshes<br>pdf: <a href="https://t.co/cL4qvT5LWN">https://t.co/cL4qvT5LWN</a><br>abs: <a href="https://t.co/n99c3ZuK7k">https://t.co/n99c3ZuK7k</a> <a href="https://t.co/9mZrjF8xo4">pic.twitter.com/9mZrjF8xo4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1316185249546674178?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Cross-Domain Few-Shot Learning by Representation Fusion

Thomas Adler, Johannes Brandstetter, Michael Widrich, Andreas Mayr, David Kreil, Michael Kopp, Günter Klambauer, Sepp Hochreiter

- retweets: 132, favorites: 14 (10/15/2020 09:38:36)

- links: [abs](https://arxiv.org/abs/2010.06498) | [pdf](https://arxiv.org/pdf/2010.06498)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

In order to quickly adapt to new data, few-shot learning aims at learning from few examples, often by using already acquired knowledge. The new data often differs from the previously seen data due to a domain shift, that is, a change of the input-target distribution. While several methods perform well on small domain shifts like new target classes with similar inputs, larger domain shifts are still challenging. Large domain shifts may result in high-level concepts that are not shared between the original and the new domain. However, low-level concepts like edges in images might still be shared and useful. For cross-domain few-shot learning, we suggest representation fusion to unify different abstraction levels of a deep neural network into one representation. We propose Cross-domain Hebbian Ensemble Few-shot learning (CHEF), which achieves representation fusion by an ensemble of Hebbian learners acting on different layers of a deep neural network that was trained on the original domain. On the few-shot datasets miniImagenet and tieredImagenet, where the domain shift is small, CHEF is competitive with state-of-the-art methods. On cross-domain few-shot benchmark challenges with larger domain shifts, CHEF establishes novel state-of-the-art results in all categories. We further apply CHEF on a real-world cross-domain application in drug discovery. We consider a domain shift from bioactive molecules to environmental chemicals and drugs with twelve associated toxicity prediction tasks. On these tasks, that are highly relevant for computational drug discovery, CHEF significantly outperforms all its competitors. Github: https://github.com/ml-jku/chef

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">📢 Representation fusion to boost few-shot learning<br>paper link: <a href="https://t.co/emVRowltFM">https://t.co/emVRowltFM</a><br>blog post: <a href="https://t.co/XirtbZMarZ">https://t.co/XirtbZMarZ</a></p>&mdash; Johannes Brandstetter (@jbrandi6) <a href="https://twitter.com/jbrandi6/status/1316272281509933056?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Towards Machine Translation for the Kurdish Language

Sina Ahmadi, Mariam Masoud

- retweets: 72, favorites: 22 (10/15/2020 09:38:36)

- links: [abs](https://arxiv.org/abs/2010.06041) | [pdf](https://arxiv.org/pdf/2010.06041)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Machine translation is the task of translating texts from one language to another using computers. It has been one of the major tasks in natural language processing and computational linguistics and has been motivating to facilitate human communication. Kurdish, an Indo-European language, has received little attention in this realm due to the language being less-resourced. Therefore, in this paper, we are addressing the main issues in creating a machine translation system for the Kurdish language, with a focus on the Sorani dialect. We describe the available scarce parallel data suitable for training a neural machine translation model for Sorani Kurdish-English translation. We also discuss some of the major challenges in Kurdish language translation and demonstrate how fundamental text processing tasks, such as tokenization, can improve translation performance.




# 10. XL-WiC: A Multilingual Benchmark for Evaluating Semantic  Contextualization

Alessandro Raganato, Tommaso Pasini, Jose Camacho-Collados, Mohammad Taher Pilehvar

- retweets: 74, favorites: 16 (10/15/2020 09:38:36)

- links: [abs](https://arxiv.org/abs/2010.06478) | [pdf](https://arxiv.org/pdf/2010.06478)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The ability to correctly model distinct meanings of a word is crucial for the effectiveness of semantic representation techniques. However, most existing evaluation benchmarks for assessing this criterion are tied to sense inventories (usually WordNet), restricting their usage to a small subset of knowledge-based representation techniques. The Word-in-Context dataset (WiC) addresses the dependence on sense inventories by reformulating the standard disambiguation task as a binary classification problem; but, it is limited to the English language. We put forward a large multilingual benchmark, XL-WiC, featuring gold standards in 12 new languages from varied language families and with different degrees of resource availability, opening room for evaluation scenarios such as zero-shot cross-lingual transfer. We perform a series of experiments to determine the reliability of the datasets and to set performance baselines for several recent contextualized multilingual models. Experimental results show that even when no tagged instances are available for a target language, models trained solely on the English data can attain competitive performance in the task of distinguishing different meanings of a word, even for distant languages. XL-WiC is available at https://pilehvar.github.io/xlwic/.




# 11. Behavior Trees in Action: A Study of Robotics Applications

Razan Ghzouli, Thorsten Berger, Einar Broch Johnsen, Swaib Dragule, Andrzej Wąsowski

- retweets: 56, favorites: 24 (10/15/2020 09:38:36)

- links: [abs](https://arxiv.org/abs/2010.06256) | [pdf](https://arxiv.org/pdf/2010.06256)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.PL](https://arxiv.org/list/cs.PL/recent) | [cs.SE](https://arxiv.org/list/cs.SE/recent)

Autonomous robots combine a variety of skills to form increasingly complex behaviors called missions. While the skills are often programmed at a relatively low level of abstraction, their coordination is architecturally separated and often expressed in higher-level languages or frameworks. Recently, the language of Behavior Trees gained attention among roboticists for this reason. Originally designed for computer games to model autonomous actors, Behavior Trees offer an extensible tree-based representation of missions. However, even though, several implementations of the language are in use, little is known about its usage and scope in the real world. How do behavior trees relate to traditional languages for describing behavior? How are behavior tree concepts used in applications? What are the benefits of using them?   We present a study of the key language concepts in Behavior Trees and their use in real-world robotic applications. We identify behavior tree languages and compare their semantics to the most well-known behavior modeling languages: state and activity diagrams. We mine open source repositories for robotics applications that use the language and analyze this usage. We find that Behavior Trees are a pragmatic language, not fully specified, allowing projects to extend it even for just one model. Behavior trees clearly resemble the models-at-runtime paradigm. We contribute a dataset of real-world behavior models, hoping to inspire the community to use and further develop this language, associated tools, and analysis techniques.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am happy to share the pre-print of my first paper 😎&quot;Behavior Trees in Action: A Study of Robotics Applications&quot; accepted at <a href="https://twitter.com/sleconf?ref_src=twsrc%5Etfw">@sleconf</a>  and really grateful for the nice collaboration with <a href="https://twitter.com/thorsten_berger?ref_src=twsrc%5Etfw">@thorsten_berger</a> <a href="https://twitter.com/AndrzejWasowski?ref_src=twsrc%5Etfw">@AndrzejWasowski</a> <a href="https://twitter.com/ebjohnsen?ref_src=twsrc%5Etfw">@ebjohnsen</a> <a href="https://twitter.com/dragule?ref_src=twsrc%5Etfw">@dragule</a>.swaib1 <a href="https://t.co/BgKrPftcnj">https://t.co/BgKrPftcnj</a></p>&mdash; Razan Ghzouli (@RGhzouli) <a href="https://twitter.com/RGhzouli/status/1316341985905709057?ref_src=twsrc%5Etfw">October 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



