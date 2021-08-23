---
title: Hot Papers 2021-08-23
date: 2021-08-24T07:55:14.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-23"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-23"
socialImage: "/media/flying-marine.jpg"

---

# 1. Fastformer: Additive Attention is All You Need

Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang

- retweets: 4276, favorites: 393 (08/24/2021 07:55:14)

- links: [abs](https://arxiv.org/abs/2108.09084) | [pdf](https://arxiv.org/pdf/2108.09084)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Transformer is a powerful model for text understanding. However, it is inefficient due to its quadratic complexity to input sequence length. Although there are many methods on Transformer acceleration, they are still either inefficient on long sequences or not effective enough. In this paper, we propose Fastformer, which is an efficient Transformer model based on additive attention. In Fastformer, instead of modeling the pair-wise interactions between tokens, we first use additive attention mechanism to model global contexts, and then further transform each token representation based on its interaction with global context representations. In this way, Fastformer can achieve effective context modeling with linear complexity. Extensive experiments on five datasets show that Fastformer is much more efficient than many existing Transformer models and can meanwhile achieve comparable or even better long text modeling performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fastformer: Additive Attention is All You Need<br>pdf: <a href="https://t.co/HelF2hT4Te">https://t.co/HelF2hT4Te</a><br>abs: <a href="https://t.co/ch8O4kG6oA">https://t.co/ch8O4kG6oA</a><br><br>a Transformer variant based on additive attention<br>that can handle long sequences efficiently with linear complexity <a href="https://t.co/GJULdoMd0L">pic.twitter.com/GJULdoMd0L</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1429612933928062980?ref_src=twsrc%5Etfw">August 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Smart Bird: Learnable Sparse Attention for Efficient and Effective  Transformer

Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang

- retweets: 156, favorites: 90 (08/24/2021 07:55:15)

- links: [abs](https://arxiv.org/abs/2108.09193) | [pdf](https://arxiv.org/pdf/2108.09193)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Transformer has achieved great success in NLP. However, the quadratic complexity of the self-attention mechanism in Transformer makes it inefficient in handling long sequences. Many existing works explore to accelerate Transformers by computing sparse self-attention instead of a dense one, which usually attends to tokens at certain positions or randomly selected tokens. However, manually selected or random tokens may be uninformative for context modeling. In this paper, we propose Smart Bird, which is an efficient and effective Transformer with learnable sparse attention. In Smart Bird, we first compute a sketched attention matrix with a single-head low-dimensional Transformer, which aims to find potential important interactions between tokens. We then sample token pairs based on their probability scores derived from the sketched attention matrix to generate different sparse attention index matrices for different attention heads. Finally, we select token embeddings according to the index matrices to form the input of sparse attention networks. Extensive experiments on six benchmark datasets for different tasks validate the efficiency and effectiveness of Smart Bird in text modeling.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Smart Bird: Learnable Sparse Attention for Efficient and Effective Transformer<br>abs: <a href="https://t.co/bu8NDj4Buc">https://t.co/bu8NDj4Buc</a><br><br>propose an efficient and effective Transformer variant named Smart Bird, which can smartly attend to important token pairs based on a learnable sparse attention mechanism <a href="https://t.co/QfdkE2Vyv6">pic.twitter.com/QfdkE2Vyv6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1429674952261832713?ref_src=twsrc%5Etfw">August 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. An Empirical Cybersecurity Evaluation of GitHub Copilot's Code  Contributions

Hammond Pearce, Baleegh Ahmad, Benjamin Tan, Brendan Dolan-Gavitt, Ramesh Karri

- retweets: 169, favorites: 60 (08/24/2021 07:55:15)

- links: [abs](https://arxiv.org/abs/2108.09293) | [pdf](https://arxiv.org/pdf/2108.09293)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

There is burgeoning interest in designing AI-based systems to assist humans in designing computing systems, including tools that automatically generate computer code. The most notable of these comes in the form of the first self-described `AI pair programmer', GitHub Copilot, a language model trained over open-source GitHub code. However, code often contains bugs - and so, given the vast quantity of unvetted code that Copilot has processed, it is certain that the language model will have learned from exploitable, buggy code. This raises concerns on the security of Copilot's code contributions. In this work, we systematically investigate the prevalence and conditions that can cause GitHub Copilot to recommend insecure code. To perform this analysis we prompt Copilot to generate code in scenarios relevant to high-risk CWEs (e.g. those from MITRE's "Top 25" list). We explore Copilot's performance on three distinct code generation axes -- examining how it performs given diversity of weaknesses, diversity of prompts, and diversity of domains. In total, we produce 89 different scenarios for Copilot to complete, producing 1,692 programs. Of these, we found approximately 40% to be vulnerable.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">An Empirical Cybersecurity Evaluation of GitHub Copilot’s Code Contributions<br>pdf: <a href="https://t.co/HQpCOxkkqX">https://t.co/HQpCOxkkqX</a><br>abs: <a href="https://t.co/k8TSAB6FLi">https://t.co/k8TSAB6FLi</a><br><br>produce 89 different scenarios for Copilot to complete, producing 1,692 programs. Of these, found approximately 40 % to be vulnerable <a href="https://t.co/BhQVzyVF5U">pic.twitter.com/BhQVzyVF5U</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1429808943224795138?ref_src=twsrc%5Etfw">August 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Towards Photorealistic Colorization by Imagination

Chenyang Lei, Yue Wu, Qifeng Chen

- retweets: 140, favorites: 71 (08/24/2021 07:55:15)

- links: [abs](https://arxiv.org/abs/2108.09195) | [pdf](https://arxiv.org/pdf/2108.09195)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a novel approach to automatic image colorization by imitating the imagination process of human experts. Our imagination module is designed to generate color images that are context-correlated with black-and-white photos. Given a black-and-white image, our imagination module firstly extracts the context information, which is then used to synthesize colorful and diverse images using a conditional image synthesis network (e.g., semantic image synthesis model). We then design a colorization module to colorize the black-and-white images with the guidance of imagination for photorealistic colorization. Experimental results show that our work produces more colorful and diverse results than state-of-the-art image colorization methods. Our source codes will be publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Photorealistic Colorization by Imagination<br>pdf: <a href="https://t.co/4NPefmh6FA">https://t.co/4NPefmh6FA</a><br>abs: <a href="https://t.co/06Qig0CqZr">https://t.co/06Qig0CqZr</a> <a href="https://t.co/tPleHnctp9">pic.twitter.com/tPleHnctp9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1429674164978388993?ref_src=twsrc%5Etfw">August 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text  Models

Jianmo Ni, Gustavo Hernández {Á}brego, Noah Constant, Ji Ma, Keith B. Hall, Daniel Cer, Yinfei Yang

- retweets: 132, favorites: 76 (08/24/2021 07:55:15)

- links: [abs](https://arxiv.org/abs/2108.08877) | [pdf](https://arxiv.org/pdf/2108.08877)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We provide the first exploration of text-to-text transformers (T5) sentence embeddings. Sentence embeddings are broadly useful for language processing tasks. While T5 achieves impressive performance on language tasks cast as sequence-to-sequence mapping problems, it is unclear how to produce sentence embeddings from encoder-decoder models. We investigate three methods for extracting T5 sentence embeddings: two utilize only the T5 encoder and one uses the full T5 encoder-decoder model. Our encoder-only models outperforms BERT-based sentence embeddings on both transfer tasks and semantic textual similarity (STS). Our encoder-decoder method achieves further improvement on STS. Scaling up T5 from millions to billions of parameters is found to produce consistent improvements on downstream tasks. Finally, we introduce a two-stage contrastive learning approach that achieves a new state-of-art on STS using sentence embeddings, outperforming both Sentence BERT and SimCSE.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models<br>pdf: <a href="https://t.co/NbGb2W1tfk">https://t.co/NbGb2W1tfk</a><br>abs: <a href="https://t.co/q4eJor0Zpd">https://t.co/q4eJor0Zpd</a><br><br>Scaling up T5 from millions to billions of parameters is found to produce consistent improvements on downstream tasks <a href="https://t.co/ajZuvk1U3q">pic.twitter.com/ajZuvk1U3q</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1429611132046749700?ref_src=twsrc%5Etfw">August 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. GAN Inversion for Out-of-Range Images with Geometric Transformations

Kyoungkook Kang, Seongtae Kim, Sunghyun Cho

- retweets: 72, favorites: 64 (08/24/2021 07:55:15)

- links: [abs](https://arxiv.org/abs/2108.08998) | [pdf](https://arxiv.org/pdf/2108.08998)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

For successful semantic editing of real images, it is critical for a GAN inversion method to find an in-domain latent code that aligns with the domain of a pre-trained GAN model. Unfortunately, such in-domain latent codes can be found only for in-range images that align with the training images of a GAN model. In this paper, we propose BDInvert, a novel GAN inversion approach to semantic editing of out-of-range images that are geometrically unaligned with the training images of a GAN model. To find a latent code that is semantically editable, BDInvert inverts an input out-of-range image into an alternative latent space than the original latent space. We also propose a regularized inversion method to find a solution that supports semantic editing in the alternative space. Our experiments show that BDInvert effectively supports semantic editing of out-of-range images with geometric transformations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GAN Inversion for Out-of-Range Images with Geometric Transformations<br>pdf: <a href="https://t.co/2X9R9Cd9O1">https://t.co/2X9R9Cd9O1</a><br>abs: <a href="https://t.co/PHAPmxHlee">https://t.co/PHAPmxHlee</a> <a href="https://t.co/raa8XmCOws">pic.twitter.com/raa8XmCOws</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1429679561256710150?ref_src=twsrc%5Etfw">August 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Uniformity Testing in the Shuffle Model: Simpler, Better, Faster

Clément L. Canonne, Hongyi Lyu

- retweets: 12, favorites: 38 (08/24/2021 07:55:15)

- links: [abs](https://arxiv.org/abs/2108.08987) | [pdf](https://arxiv.org/pdf/2108.08987)
- [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.DM](https://arxiv.org/list/cs.DM/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Uniformity testing, or testing whether independent observations are uniformly distributed, is the prototypical question in distribution testing. Over the past years, a line of work has been focusing on uniformity testing under privacy constraints on the data, and obtained private and data-efficient algorithms under various privacy models such as central differential privacy (DP), local privacy (LDP), pan-privacy, and, very recently, the shuffle model of differential privacy.   In this work, we considerably simplify the analysis of the known uniformity testing algorithm in the shuffle model, and, using a recent result on "privacy amplification via shuffling," provide an alternative algorithm attaining the same guarantees with an elementary and streamlined argument.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My first foray into shuffle <a href="https://twitter.com/hashtag/privacy?src=hash&amp;ref_src=twsrc%5Etfw">#privacy</a>, spearheaded by my impressive winter research intern Hongyi Lyu (maths undergrad <a href="https://twitter.com/UniMelb?ref_src=twsrc%5Etfw">@UniMelb</a>), who managed to learn about DP+shuffle DP+distribution testing, all this in ~6 weeks!<br><br>Comments welcome! 📝 <a href="https://t.co/MuhjHQA7f8">https://t.co/MuhjHQA7f8</a><br>1/4 <a href="https://t.co/nkFxir991J">pic.twitter.com/nkFxir991J</a></p>&mdash; Clément Canonne (@ccanonne_) <a href="https://twitter.com/ccanonne_/status/1429615738252234757?ref_src=twsrc%5Etfw">August 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



