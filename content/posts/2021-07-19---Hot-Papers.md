---
title: Hot Papers 2021-07-19
date: 2021-07-20T07:00:57.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-19"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-19"
socialImage: "/media/flying-marine.jpg"

---

# 1. Graph Kernel Attention Transformers

Krzysztof Choromanski, Han Lin, Haoxian Chen, Jack Parker-Holder

- retweets: 725, favorites: 99 (07/20/2021 07:00:57)

- links: [abs](https://arxiv.org/abs/2107.07999) | [pdf](https://arxiv.org/pdf/2107.07999)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We introduce a new class of graph neural networks (GNNs), by combining several concepts that were so far studied independently - graph kernels, attention-based networks with structural priors and more recently, efficient Transformers architectures applying small memory footprint implicit attention methods via low rank decomposition techniques. The goal of the paper is twofold. Proposed by us Graph Kernel Attention Transformers (or GKATs) are much more expressive than SOTA GNNs as capable of modeling longer-range dependencies within a single layer. Consequently, they can use more shallow architecture design. Furthermore, GKAT attention layers scale linearly rather than quadratically in the number of nodes of the input graphs, even when those graphs are dense, requiring less compute than their regular graph attention counterparts. They achieve it by applying new classes of graph kernels admitting random feature map decomposition via random walks on graphs. As a byproduct of the introduced techniques, we obtain a new class of learnable graph sketches, called graphots, compactly encoding topological graph properties as well as nodes' features. We conducted exhaustive empirical comparison of our method with nine different GNN classes on tasks ranging from motif detection through social network classification to bioinformatics challenges, showing consistent gains coming from GKATs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Graph Kernel Attention Transformers<br>pdf: <a href="https://t.co/Uyy5ZcTD1I">https://t.co/Uyy5ZcTD1I</a><br>abs: <a href="https://t.co/KTxHRuYvVV">https://t.co/KTxHRuYvVV</a><br><br>comparison of method with 9 different GNN classes on tasks, showing consistent gains coming from GKATs <a href="https://t.co/acoHALTKjv">pic.twitter.com/acoHALTKjv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1416921359255576577?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Unsupervised Discovery of Object Radiance Fields

Hong-Xing Yu, Leonidas J. Guibas, Jiajun Wu

- retweets: 627, favorites: 161 (07/20/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.07905) | [pdf](https://arxiv.org/pdf/2107.07905)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We study the problem of inferring an object-centric scene representation from a single image, aiming to derive a representation that explains the image formation process, captures the scene's 3D nature, and is learned without supervision. Most existing methods on scene decomposition lack one or more of these characteristics, due to the fundamental challenge in integrating the complex 3D-to-2D image formation process into powerful inference schemes like deep networks. In this paper, we propose unsupervised discovery of Object Radiance Fields (uORF), integrating recent progresses in neural 3D scene representations and rendering with deep inference networks for unsupervised 3D scene decomposition. Trained on multi-view RGB images without annotations, uORF learns to decompose complex scenes with diverse, textured background from a single image. We show that uORF performs well on unsupervised 3D scene segmentation, novel view synthesis, and scene editing on three datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Discovery of Object Radiance Fields<br>pdf: <a href="https://t.co/e73XGnwXo0">https://t.co/e73XGnwXo0</a><br>abs: <a href="https://t.co/7Die0miMhr">https://t.co/7Die0miMhr</a><br>project page: <a href="https://t.co/9jwyXfoNEq">https://t.co/9jwyXfoNEq</a><br>github: <a href="https://t.co/yCZ6PmsqYC">https://t.co/yCZ6PmsqYC</a> <a href="https://t.co/XEY8CF7d3r">pic.twitter.com/XEY8CF7d3r</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1416978431724724225?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Painting Style-Aware Manga Colorization Based on Generative Adversarial  Networks

Yugo Shimizu, Ryosuke Furuta, Delong Ouyang, Yukinobu Taniguchi, Ryota Hinami, Shonosuke Ishiwatari

- retweets: 344, favorites: 52 (07/20/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.07943) | [pdf](https://arxiv.org/pdf/2107.07943)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Japanese comics (called manga) are traditionally created in monochrome format. In recent years, in addition to monochrome comics, full color comics, a more attractive medium, have appeared. Unfortunately, color comics require manual colorization, which incurs high labor costs. Although automatic colorization methods have been recently proposed, most of them are designed for illustrations, not for comics. Unlike illustrations, since comics are composed of many consecutive images, the painting style must be consistent. To realize consistent colorization, we propose here a semi-automatic colorization method based on generative adversarial networks (GAN); the method learns the painting style of a specific comic from small amount of training data. The proposed method takes a pair of a screen tone image and a flat colored image as input, and outputs a colorized image. Experiments show that the proposed method achieves better performance than the existing alternatives.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">漫画の半自動着色 <a href="https://t.co/Wir73QH7Wn">https://t.co/Wir73QH7Wn</a><br>スクリーントーン画像と下塗り画像を入力し着色画像を出力。<br>少数のデータで訓練でき、特定の漫画の画風に特化させやすい。<br>下塗り画像は必要だが、着色時間が半減。 <a href="https://t.co/AW3DzIQPmL">pic.twitter.com/AW3DzIQPmL</a></p>&mdash; Yosuke Shinya (@shinya7y) <a href="https://twitter.com/shinya7y/status/1417104456970215424?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Internet-Augmented Dialogue Generation

Mojtaba Komeili, Kurt Shuster, Jason Weston

- retweets: 116, favorites: 131 (07/20/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.07566) | [pdf](https://arxiv.org/pdf/2107.07566)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

The largest store of continually updating knowledge on our planet can be accessed via internet search. In this work we study giving access to this information to conversational agents. Large language models, even though they store an impressive amount of knowledge within their weights, are known to hallucinate facts when generating dialogue (Shuster et al., 2021); moreover, those facts are frozen in time at the point of model training. In contrast, we propose an approach that learns to generate an internet search query based on the context, and then conditions on the search results to finally generate a response, a method that can employ up-to-the-minute relevant information. We train and evaluate such models on a newly collected dataset of human-human conversations whereby one of the speakers is given access to internet search during knowledgedriven discussions in order to ground their responses. We find that search-query based access of the internet in conversation provides superior performance compared to existing approaches that either use no augmentation or FAISS-based retrieval (Lewis et al., 2020).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Internet-Augmented Dialogue Generation<br><br>Search-query based access of the internet in conversation provides superior performance compared to existing approaches that either use no augmentation or FAISS-based retrieval.<a href="https://t.co/cD5lAm2DLk">https://t.co/cD5lAm2DLk</a> <a href="https://t.co/unpZYR5OhN">pic.twitter.com/unpZYR5OhN</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1416923866761371650?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Internet-Augmented Dialogue Generation<br>pdf: <a href="https://t.co/qcQ5ZmAA47">https://t.co/qcQ5ZmAA47</a><br>abs: <a href="https://t.co/u4mnlnjP6o">https://t.co/u4mnlnjP6o</a> <a href="https://t.co/nm2NQXxhNb">pic.twitter.com/nm2NQXxhNb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1416922759159033860?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. The COVID-19 infodemic does not affect vaccine acceptance

Carlo M. Valensise, Matteo Cinelli, Matthieu Nadini, Alessandro Galeazzi, Antonio Peruzzi, Gabriele Etta, Fabiana Zollo, Andrea Baronchelli, Walter Quattrociocchi

- retweets: 128, favorites: 64 (07/20/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.07946) | [pdf](https://arxiv.org/pdf/2107.07946)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

How does information consumption affect behaviour in the context of the COVID-19 pandemic? A popular hypothesis states that the so-called infodemics has substantial impact on orienting individual decisions. A competing hypothesis stresses that exposure to vast amounts of even contradictory information has little effect on personal choices. We analyse the vaccine infodemics on Twitter and Facebook by looking at 146M contents produced by 20M accounts between 1 January 2020 and 30 April 2021. We find that vaccine-related news triggered huge interest through social media, affecting attention patterns and the modality in which information was spreading. However, we find that such a tumultuous information landscape translated in only minimal variations in vaccine acceptance as measured by Facebook's daily COVID-19 World Symptoms Survey on a sample of 1.6M users. This finding supports the hypothesis that altered information consumption patterns are not a reliable predictor of behavioural change. Instead, wider attention on social media seems to resolve in polarisation, with the vaccine-prone and the vaccine-hesitant maintaining their positions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New: &quot;The COVID-19 infodemic does not affect vaccine acceptance&quot;. <a href="https://t.co/LTWmgSvhrT">https://t.co/LTWmgSvhrT</a><br><br>Six European countries, 2021. We found that social media responded vigorously to vaccine news (e.g., EMA suspending AZ). Yet overall vaccine acceptance was largely flat.<br><br>See also below. <a href="https://t.co/92wmK5XOqI">https://t.co/92wmK5XOqI</a></p>&mdash; Andrea Baronchelli (@a_baronca) <a href="https://twitter.com/a_baronca/status/1417089077015465990?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our last work about the interplay between content consumation and behavior. We find overabundance of information about Vaccines and a flat Vaccine hesitancy trend. <a href="https://twitter.com/a_baronca?ref_src=twsrc%5Etfw">@a_baronca</a> <a href="https://twitter.com/zollofab?ref_src=twsrc%5Etfw">@zollofab</a> <a href="https://twitter.com/matteo_cinelli?ref_src=twsrc%5Etfw">@matteo_cinelli</a> <a href="https://twitter.com/valensic_?ref_src=twsrc%5Etfw">@valensic_</a> <a href="https://twitter.com/DeveloperGale?ref_src=twsrc%5Etfw">@DeveloperGale</a> <a href="https://twitter.com/gbrtte_?ref_src=twsrc%5Etfw">@gbrtte_</a> <a href="https://twitter.com/nadin?ref_src=twsrc%5Etfw">@nadin</a> <a href="https://twitter.com/matt55nado?ref_src=twsrc%5Etfw">@matt55nado</a><a href="https://t.co/0ho8eJ9ECK">https://t.co/0ho8eJ9ECK</a> <a href="https://t.co/FHkAWTSuqV">pic.twitter.com/FHkAWTSuqV</a></p>&mdash; W. Quattrociocchi (@Walter4C) <a href="https://twitter.com/Walter4C/status/1417013842417274881?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Recognizing bird species in diverse soundscapes under weak supervision

Christof Henkel, Pascal Pfeiffer, Philipp Singer

- retweets: 121, favorites: 44 (07/20/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.07728) | [pdf](https://arxiv.org/pdf/2107.07728)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We present a robust classification approach for avian vocalization in complex and diverse soundscapes, achieving second place in the BirdCLEF2021 challenge. We illustrate how to make full use of pre-trained convolutional neural networks, by using an efficient modeling and training routine supplemented by novel augmentation methods. Thereby, we improve the generalization of weakly labeled crowd-sourced data to productive data collected by autonomous recording units. As such, we illustrate how to progress towards an accurate automated assessment of avian population which would enable global biodiversity monitoring at scale, impossible by manual annotation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Recognizing bird species in diverse soundscapes<br>under weak supervision&quot; - Paper describing our recent 2nd place Birdclef competition solution (\w <a href="https://twitter.com/kagglingdieter?ref_src=twsrc%5Etfw">@kagglingdieter</a>) <a href="https://t.co/gkZCSFmJuq">https://t.co/gkZCSFmJuq</a></p>&mdash; Philipp Singer (@ph_singer) <a href="https://twitter.com/ph_singer/status/1417017876435853320?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Align before Fuse: Vision and Language Representation Learning with  Momentum Distillation

Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, Steven Hoi

- retweets: 100, favorites: 56 (07/20/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.07651) | [pdf](https://arxiv.org/pdf/2107.07651)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Large-scale vision and language representation learning has shown promising improvements on various vision-language tasks. Most existing methods employ a transformer-based multimodal encoder to jointly model visual tokens (region-based image features) and word tokens. Because the visual tokens and word tokens are unaligned, it is challenging for the multimodal encoder to learn image-text interactions. In this paper, we introduce a contrastive loss to ALign the image and text representations BEfore Fusing (ALBEF) them through cross-modal attention, which enables more grounded vision and language representation learning. Unlike most existing methods, our method does not require bounding box annotations nor high-resolution images. In order to improve learning from noisy web data, we propose momentum distillation, a self-training method which learns from pseudo-targets produced by a momentum model. We provide a theoretical analysis of ALBEF from a mutual information maximization perspective, showing that different training tasks can be interpreted as different ways to generate views for an image-text pair. ALBEF achieves state-of-the-art performance on multiple downstream vision-language tasks. On image-text retrieval, ALBEF outperforms methods that are pre-trained on orders of magnitude larger datasets. On VQA and NLVR$^2$, ALBEF achieves absolute improvements of 2.37% and 3.84% compared to the state-of-the-art, while enjoying faster inference speed. Code and pre-trained models are available at https://github.com/salesforce/ALBEF/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Align before Fuse: Vision and Language Representation Learning with Momentum Distillation<br>pdf: <a href="https://t.co/6I33WgfJXT">https://t.co/6I33WgfJXT</a><br>abs: <a href="https://t.co/MktAVZLdYd">https://t.co/MktAVZLdYd</a><br>GitHub: <a href="https://t.co/9j4QBGkPWO">https://t.co/9j4QBGkPWO</a><br><br>On VQA and NLVR2, ALBEF achieves absolute improvements of 2.37% and 3.84% <a href="https://t.co/2O3WpOrH9Z">pic.twitter.com/2O3WpOrH9Z</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1416928800244445184?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to release Align before Fuse (ALBEF), a new vision-language pre-training method with SoTA performance on multiple V+L tasks! We provide code and models for pre-training and finetuning: <a href="https://t.co/cjCGSLGzxZ">https://t.co/cjCGSLGzxZ</a><br>Blog: <a href="https://t.co/vkfnmX5nhO">https://t.co/vkfnmX5nhO</a><br>Paper: <a href="https://t.co/YNwem7Ni4k">https://t.co/YNwem7Ni4k</a></p>&mdash; Li Junnan (@LiJunnan0409) <a href="https://twitter.com/LiJunnan0409/status/1417024456040411143?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A method for decompilation of AMD GCN kernels to OpenCL

K. I. Mihajlenko, M. A. Lukin, A. S. Stankevich

- retweets: 80, favorites: 38 (07/20/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.07809) | [pdf](https://arxiv.org/pdf/2107.07809)
- [cs.PL](https://arxiv.org/list/cs.PL/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Introduction: Decompilers are useful tools for software analysis and support in the absence of source code. They are available for many hardware architectures and programming languages. However, none of the existing decompilers support modern AMD GPU architectures such as AMD GCN and RDNA. Purpose: We aim at developing the first assembly decompiler tool for a modern AMD GPU architecture that generates code in the OpenCL language, which is widely used for programming GPGPUs. Results: We developed the algorithms for the following operations: preprocessing assembly code, searching data accesses, extracting system values, decompiling arithmetic operations and recovering data types. We also developed templates for decompilation of branching operations. Practical relevance: We implemented the presented algorithms in Python as a tool called OpenCLDecompiler, which supports a large subset of AMD GCN instructions. This tool automatically converts disassembled GPGPU code into the equivalent OpenCL code, which reduces the effort required to analyze assembly code.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper, researchers have developed the first assembly decompiler tool for AMD GCN architecture that automatically converts disassembled GPGPU code into the equivalent OpenCL code, thus reducing the effort required to analyze assembly code.<a href="https://t.co/r5yLSAthko">https://t.co/r5yLSAthko</a> <a href="https://t.co/2mQcFvlpSS">pic.twitter.com/2mQcFvlpSS</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1416987169596612610?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Beyond In-Place Corruption: Insertion and Deletion In Denoising  Probabilistic Models

Daniel D. Johnson, Jacob Austin, Rianne van den Berg, Daniel Tarlow

- retweets: 72, favorites: 40 (07/20/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.07675) | [pdf](https://arxiv.org/pdf/2107.07675)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Denoising diffusion probabilistic models (DDPMs) have shown impressive results on sequence generation by iteratively corrupting each example and then learning to map corrupted versions back to the original. However, previous work has largely focused on in-place corruption, adding noise to each pixel or token individually while keeping their locations the same. In this work, we consider a broader class of corruption processes and denoising models over sequence data that can insert and delete elements, while still being efficient to train and sample from. We demonstrate that these models outperform standard in-place models on an arithmetic sequence task, and that when trained on the text8 dataset they can be used to fix spelling errors without any fine-tuning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Beyond In-Place Corruption: Insertion and Deletion in Denoising Probabilistic Models<br>pdf: <a href="https://t.co/0ZUEPjW9AV">https://t.co/0ZUEPjW9AV</a><br>abs: <a href="https://t.co/pkRs07yHkr">https://t.co/pkRs07yHkr</a> <a href="https://t.co/vObkV7Uc3k">pic.twitter.com/vObkV7Uc3k</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1416919518409072640?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Beyond Goldfish Memory: Long-Term Open-Domain Conversation

Jing Xu, Arthur Szlam, Jason Weston

- retweets: 32, favorites: 72 (07/20/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.07567) | [pdf](https://arxiv.org/pdf/2107.07567)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Despite recent improvements in open-domain dialogue models, state of the art models are trained and evaluated on short conversations with little context. In contrast, the long-term conversation setting has hardly been studied. In this work we collect and release a human-human dataset consisting of multiple chat sessions whereby the speaking partners learn about each other's interests and discuss the things they have learnt from past sessions. We show how existing models trained on existing datasets perform poorly in this long-term conversation setting in both automatic and human evaluations, and we study long-context models that can perform much better. In particular, we find retrieval-augmented methods and methods with an ability to summarize and recall previous conversations outperform the standard encoder-decoder architectures currently considered state of the art.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Beyond Goldfish Memory∗: Long-Term Open-Domain Conversation<br>pdf: <a href="https://t.co/j9cCsIU5V5">https://t.co/j9cCsIU5V5</a><br><br>find retrieval-augmented methods and methods with an ability to summarize and recall previous conversations outperform the standard encoder-decoder architectures currently considered sota <a href="https://t.co/wzubf9Wg7O">pic.twitter.com/wzubf9Wg7O</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1416933821954641921?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. CCVS: Context-aware Controllable Video Synthesis

Guillaume Le Moing, Jean Ponce, Cordelia Schmid

- retweets: 64, favorites: 35 (07/20/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.08037) | [pdf](https://arxiv.org/pdf/2107.08037)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This presentation introduces a self-supervised learning approach to the synthesis of new video clips from old ones, with several new key elements for improved spatial resolution and realism: It conditions the synthesis process on contextual information for temporal continuity and ancillary information for fine control. The prediction model is doubly autoregressive, in the latent space of an autoencoder for forecasting, and in image space for updating contextual information, which is also used to enforce spatio-temporal consistency through a learnable optical flow module. Adversarial training of the autoencoder in the appearance and temporal domains is used to further improve the realism of its output. A quantizer inserted between the encoder and the transformer in charge of forecasting future frames in latent space (and its inverse inserted between the transformer and the decoder) adds even more flexibility by affording simple mechanisms for handling multimodal ancillary information for controlling the synthesis process (eg, a few sample frames, an audio track, a trajectory in image space) and taking into account the intrinsically uncertain nature of the future by allowing multiple predictions. Experiments with an implementation of the proposed approach give very good qualitative and quantitative results on multiple tasks and standard benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CCVS: Context-aware Controllable Video Synthesis<br>pdf: <a href="https://t.co/MI7vHBGkbL">https://t.co/MI7vHBGkbL</a><br>abs: <a href="https://t.co/zO3S1zeZBF">https://t.co/zO3S1zeZBF</a><br>project page: <a href="https://t.co/318ZKXHJv0">https://t.co/318ZKXHJv0</a> <a href="https://t.co/Ukpb3Iv0Aa">pic.twitter.com/Ukpb3Iv0Aa</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1416984253041201154?ref_src=twsrc%5Etfw">July 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



