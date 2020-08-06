---
title: Hot Papers 2020-08-05
date: 2020-08-06T11:01:39.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-05"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-05"
socialImage: "/media/flying-marine.jpg"

---

# 1. A Spectral Energy Distance for Parallel Speech Synthesis

Alexey A. Gritsenko, Tim Salimans, Rianne van den Berg, Jasper Snoek, Nal Kalchbrenner

- retweets: 45, favorites: 199 (08/06/2020 11:01:39)

- links: [abs](https://arxiv.org/abs/2008.01160) | [pdf](https://arxiv.org/pdf/2008.01160)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Speech synthesis is an important practical generative modeling problem that has seen great progress over the last few years, with likelihood-based autoregressive neural models now outperforming traditional concatenative systems. A downside of such autoregressive models is that they require executing tens of thousands of sequential operations per second of generated audio, making them ill-suited for deployment on specialized deep learning hardware. Here, we propose a new learning method that allows us to train highly parallel models of speech, without requiring access to an analytical likelihood function. Our approach is based on a generalized energy distance between the distributions of the generated and real audio. This spectral energy distance is a proper scoring rule with respect to the distribution over magnitude-spectrograms of the generated waveform audio and offers statistical consistency guarantees. The distance can be calculated from minibatches without bias, and does not involve adversarial learning, yielding a stable and consistent method for training implicit generative models. Empirically, we achieve state-of-the-art generation quality among implicit generative models, as judged by the recently-proposed cFDSD metric. When combining our method with adversarial techniques, we also improve upon the recently-proposed GAN-TTS model in terms of Mean Opinion Score as judged by trained human evaluators.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper on implicit generative models for speech synthesis is out! We use a spectrogram-based loss like DDSP and OpenAI Jukebox, but add a repulsive term that offers statistical guarantees:<a href="https://t.co/lGwxLVpbI8">https://t.co/lGwxLVpbI8</a><br>w/ <a href="https://twitter.com/agritsenko?ref_src=twsrc%5Etfw">@agritsenko</a> <a href="https://twitter.com/vdbergrianne?ref_src=twsrc%5Etfw">@vdbergrianne</a> <a href="https://twitter.com/NalKalchbrenner?ref_src=twsrc%5Etfw">@NalKalchbrenner</a>  <a href="https://twitter.com/latentjasper?ref_src=twsrc%5Etfw">@latentjasper</a><br>1/3 <a href="https://t.co/3IgW2gkeEt">pic.twitter.com/3IgW2gkeEt</a></p>&mdash; Tim Salimans (@TimSalimans) <a href="https://twitter.com/TimSalimans/status/1291034889664462853?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Automatic Composition of Guitar Tabs by Transformers and Groove Modeling

Yu-Hua Chen, Yu-Hsiang Huang, Wen-Yi Hsiao, Yi-Hsuan Yang

- retweets: 18, favorites: 82 (08/06/2020 11:01:39)

- links: [abs](https://arxiv.org/abs/2008.01431) | [pdf](https://arxiv.org/pdf/2008.01431)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Deep learning algorithms are increasingly developed for learning to compose music in the form of MIDI files. However, whether such algorithms work well for composing guitar tabs, which are quite different from MIDIs, remain relatively unexplored. To address this, we build a model for composing fingerstyle guitar tabs with Transformer-XL, a neural sequence model architecture. With this model, we investigate the following research questions. First, whether the neural net generates note sequences with meaningful note-string combinations, which is important for the guitar but not other instruments such as the piano. Second, whether it generates compositions with coherent rhythmic groove, crucial for fingerstyle guitar music. And, finally, how pleasant the composed music is in comparison to real, human-made compositions. Our work provides preliminary empirical evidence of the promise of deep learning for tab composition, and suggests areas for future study.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our guitar transformer paper (ISMIR&#39;20) is also on arxiv!<br>* use Transformer-XL to compose guitar TABs (not MIDIs)<br>* represent `groove` by the occurrence of note onsets in a bar and add such `grooving` tokens (after VQ) to the model<a href="https://t.co/B3cHsDJPEu">https://t.co/B3cHsDJPEu</a><a href="https://t.co/KWJgT8L3A2">https://t.co/KWJgT8L3A2</a> <a href="https://t.co/hVWdg469nl">https://t.co/hVWdg469nl</a></p>&mdash; Yi-Hsuan Yang (@affige_yang) <a href="https://twitter.com/affige_yang/status/1290829365031313408?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Automatic Composition of Guitar Tabs by Transformers and Groove Modeling<br>pdf: <a href="https://t.co/4oSLr8NIsi">https://t.co/4oSLr8NIsi</a><br>abs: <a href="https://t.co/OOpgvja6yn">https://t.co/OOpgvja6yn</a><br>project page: <a href="https://t.co/8b8xjrSMAq">https://t.co/8b8xjrSMAq</a> <a href="https://t.co/fnA6xivc9S">pic.twitter.com/fnA6xivc9S</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1290827196093870080?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Music SketchNet: Controllable Music Generation via Factorized  Representations of Pitch and Rhythm

Ke Chen, Cheng-i Wang, Taylor Berg-Kirkpatrick, Shlomo Dubnov

- retweets: 20, favorites: 78 (08/06/2020 11:01:40)

- links: [abs](https://arxiv.org/abs/2008.01291) | [pdf](https://arxiv.org/pdf/2008.01291)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Drawing an analogy with automatic image completion systems, we propose Music SketchNet, a neural network framework that allows users to specify partial musical ideas guiding automatic music generation. We focus on generating the missing measures in incomplete monophonic musical pieces, conditioned on surrounding context, and optionally guided by user-specified pitch and rhythm snippets. First, we introduce SketchVAE, a novel variational autoencoder that explicitly factorizes rhythm and pitch contour to form the basis of our proposed model. Then we introduce two discriminative architectures, SketchInpainter and SketchConnector, that in conjunction perform the guided music completion, filling in representations for the missing measures conditioned on surrounding context and user-specified snippets. We evaluate SketchNet on a standard dataset of Irish folk music and compare with models from recent works. When used for music completion, our approach outperforms the state-of-the-art both in terms of objective metrics and subjective listening tests. Finally, we demonstrate that our model can successfully incorporate user-specified snippets during the generation process.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Music SketchNet: Controllable Music Generation via Factorized Representations of Pitch and Rhythm<br>pdf: <a href="https://t.co/nNCG8ksLik">https://t.co/nNCG8ksLik</a><br>abs: <a href="https://t.co/wIGeOfSO5D">https://t.co/wIGeOfSO5D</a><br>github: <a href="https://t.co/AOxNdM09ea">https://t.co/AOxNdM09ea</a> <a href="https://t.co/NqdK8Gxcd6">pic.twitter.com/NqdK8Gxcd6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1290817592395997186?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. The Jazz Transformer on the Front Line: Exploring the Shortcomings of  AI-composed Music through Quantitative Measures

Shih-Lun Wu, Yi-Hsuan Yang

- retweets: 18, favorites: 64 (08/06/2020 11:01:40)

- links: [abs](https://arxiv.org/abs/2008.01307) | [pdf](https://arxiv.org/pdf/2008.01307)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper presents the Jazz Transformer, a generative model that utilizes a neural sequence model called the Transformer-XL for modeling lead sheets of Jazz music. Moreover, the model endeavors to incorporate structural events present in the Weimar Jazz Database (WJazzD) for inducing structures in the generated music. While we are able to reduce the training loss to a low value, our listening test suggests however a clear gap between the average ratings of the generated and real compositions. We therefore go one step further and conduct a series of computational analysis of the generated compositions from different perspectives. This includes analyzing the statistics of the pitch class, grooving, and chord progression, assessing the structureness of the music with the help of the fitness scape plot, and evaluating the model's understanding of Jazz music through a MIREX-like continuation prediction task. Our work presents in an analytical manner why machine-generated music to date still falls short of the artwork of humanity, and sets some goals for future work on automatic composition to further pursue.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures<br>pdf: <a href="https://t.co/XVgvWqUM1F">https://t.co/XVgvWqUM1F</a><br>abs: <a href="https://t.co/YyQpKJCki0">https://t.co/YyQpKJCki0</a><br>github: <a href="https://t.co/ULgRy4cPSS">https://t.co/ULgRy4cPSS</a> <a href="https://t.co/qWJqH14mDC">pic.twitter.com/qWJqH14mDC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1290823906446770181?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Learning Stereo from Single Images

Jamie Watson, Oisin Mac Aodha, Daniyar Turmukhambetov, Gabriel J. Brostow, Michael Firman

- retweets: 11, favorites: 68 (08/06/2020 11:01:40)

- links: [abs](https://arxiv.org/abs/2008.01484) | [pdf](https://arxiv.org/pdf/2008.01484)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Supervised deep networks are among the best methods for finding correspondences in stereo image pairs. Like all supervised approaches, these networks require ground truth data during training. However, collecting large quantities of accurate dense correspondence data is very challenging. We propose that it is unnecessary to have such a high reliance on ground truth depths or even corresponding stereo pairs. Inspired by recent progress in monocular depth estimation, we generate plausible disparity maps from single images. In turn, we use those flawed disparity maps in a carefully designed pipeline to generate stereo training pairs. Training in this manner makes it possible to convert any collection of single RGB images into stereo training data. This results in a significant reduction in human effort, with no need to collect real depths or to hand-design synthetic data. We can consequently train a stereo matching network from scratch on datasets like COCO, which were previously hard to exploit for stereo. Through extensive experiments we show that our approach outperforms stereo networks trained with standard synthetic datasets, when evaluated on KITTI, ETH3D, and Middlebury.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Niantic&#39;s <a href="https://twitter.com/hashtag/ECCV?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV</a> <a href="https://twitter.com/hashtag/ECCV2020?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV2020</a> Oral:<br>Learning Stereo from Single Images.<br>Project page: <a href="https://t.co/yeknZPE74X">https://t.co/yeknZPE74X</a><br>Arxiv: <a href="https://t.co/kHtZsyRXzO">https://t.co/kHtZsyRXzO</a><br>Video: <a href="https://t.co/ExY2u5rGyV">https://t.co/ExY2u5rGyV</a><br><br>TLDR: We use single image depth prediction to generate stereo data that is better than Sceneflow. <a href="https://t.co/fjzJOFEAsK">pic.twitter.com/fjzJOFEAsK</a></p>&mdash; Daniyar Turmukhambetov (@dantkz) <a href="https://twitter.com/dantkz/status/1291027870177988612?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. The COVID-19 online shadow economy

Alberto Bracci, Matthieu Nadini, Maxwell Aliapoulios, Damon McCoy, Ian Gray, Alexander Teytelboym, Angela Gallo, Andrea Baronchelli

- retweets: 25, favorites: 54 (08/06/2020 11:01:40)

- links: [abs](https://arxiv.org/abs/2008.01585) | [pdf](https://arxiv.org/pdf/2008.01585)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

The COVID-19 pandemic has reshaped the demand for goods and services worldwide. A combination of a public health emergency, economic distress, and disinformation-driven panic have pushed customers and vendors towards the shadow economy. In particular dark web marketplaces (DWMs), commercial websites easily accessible via free software, have gained significant popularity. Here, we analyse 472,372 listings extracted from 23 DWMs between January 1, 2020 and July 7, 2020. We identify 518 listings directly related to COVID-19 products and monitor the temporal evolution of product categories including PPE, medicines (e.g., hydroxyclorochine), and medical frauds (e.g., vaccines). Finally, we compare trends in their temporal evolution with variations in public attention, as measured by Twitter posts and Wikipedia page visits. We reveal how the online shadow economy has evolved during the COVID-19 pandemic and highlight the importance of a continuous monitoring of DWMs, especially when real vaccines or cures become available and potentially in short supply. We anticipate our analysis will be of interest both to researchers and public agencies focused on the protection of public health.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Preprint of &quot;The COVID-19 online shadow economy&quot; is out <a href="https://t.co/1mBUmMuhvQ">https://t.co/1mBUmMuhvQ</a>.<br><br>We tracked 23 dark web marketplaces since Jan. Sold goods include PPE, medicines, and medical frauds (vaccines). DWMs activity correlates with public attention (Twitter, Wiki) and misinformation. <a href="https://t.co/7jAtKOBmY3">pic.twitter.com/7jAtKOBmY3</a></p>&mdash; Andrea Baronchelli (@a_baronca) <a href="https://twitter.com/a_baronca/status/1290942421023457281?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Des chercheurs ont étudié les ventes liées à la pandémie de <a href="https://twitter.com/hashtag/COVID19?src=hash&amp;ref_src=twsrc%5Etfw">#COVID19</a> sur le dark web. Essentiellement des masques... et de la chloroquine/hydroxychloroquine/azithromycine. <a href="https://t.co/Uyal5qNxZL">https://t.co/Uyal5qNxZL</a> <a href="https://t.co/BLDEJl8EDK">pic.twitter.com/BLDEJl8EDK</a></p>&mdash; Vincent Glad (@vincentglad) <a href="https://twitter.com/vincentglad/status/1290970008194093056?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Learning Visual Representations with Caption Annotations

Mert Bulent Sariyildiz, Julien Perez, Diane Larlus

- retweets: 13, favorites: 56 (08/06/2020 11:01:40)

- links: [abs](https://arxiv.org/abs/2008.01392) | [pdf](https://arxiv.org/pdf/2008.01392)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Pretraining general-purpose visual features has become a crucial part of tackling many computer vision tasks. While one can learn such features on the extensively-annotated ImageNet dataset, recent approaches have looked at ways to allow for noisy, fewer, or even no annotations to perform such pretraining. Starting from the observation that captioned images are easily crawlable, we argue that this overlooked source of information can be exploited to supervise the training of visual representations. To do so, motivated by the recent progresses in language models, we introduce {\em image-conditioned masked language modeling} (ICMLM) -- a proxy task to learn visual representations over image-caption pairs. ICMLM consists in predicting masked words in captions by relying on visual cues. To tackle this task, we propose hybrid models, with dedicated visual and textual encoders, and we show that the visual representations learned as a by-product of solving this task transfer well to a variety of target tasks. Our experiments confirm that image captions can be leveraged to inject global and localized semantic information into visual representations. Project website: https://europe.naverlabs.com/icmlm.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">📢 Delighted to share our ECCV2020 paper: <a href="https://t.co/nFkVTzEBfp">https://t.co/nFkVTzEBfp</a>.<br>We propose image-conditioned masked language modeling (ICMLM) to learn visual representations using text (no bounding boxes, just image-caption pairs!)<a href="https://twitter.com/dlarlus?ref_src=twsrc%5Etfw">@dlarlus</a> <a href="https://twitter.com/perezjln?ref_src=twsrc%5Etfw">@perezjln</a> <a href="https://twitter.com/naverlabseurope?ref_src=twsrc%5Etfw">@naverlabseurope</a> <a href="https://t.co/C1qrKy2FoH">pic.twitter.com/C1qrKy2FoH</a></p>&mdash; M.Bülent Sarıyıldız (@mbsariyildiz) <a href="https://twitter.com/mbsariyildiz/status/1290890767523811329?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Deep Knowledge Tracing with Convolutions

Shanghui Yang, Mengxia Zhu, Jingyang Hou, Xuesong Lu

- retweets: 18, favorites: 46 (08/06/2020 11:01:41)

- links: [abs](https://arxiv.org/abs/2008.01169) | [pdf](https://arxiv.org/pdf/2008.01169)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Knowledge tracing (KT) has recently been an active research area of computational pedagogy. The task is to model students mastery level of knowledge based on their responses to the questions in the past, as well as predict the probabilities that they correctly answer subsequent questions in the future. A good KT model can not only make students timely aware of their knowledge states, but also help teachers develop better personalized teaching plans for students. KT tasks were historically solved using statistical modeling methods such as Bayesian inference and factor analysis, but recent advances in deep learning have led to the successive proposals that leverage deep neural networks, including long short-term memory networks, memory-augmented networks and self-attention networks. While those deep models demonstrate superior performance over the traditional approaches, they all neglect more or less the impact on knowledge states of the most recent questions answered by students. The forgetting curve theory states that human memory retention declines over time, therefore knowledge states should be mostly affected by the recent questions. Based on this observation, we propose a Convolutional Knowledge Tracing (CKT) model in this paper. In addition to modeling the long-term effect of the entire question-answer sequence, CKT also strengthens the short-term effect of recent questions using 3D convolutions, thereby effectively modeling the forgetting curve in the learning process. Extensive experiments show that CKT achieves the new state-of-the-art in predicting students performance compared with existing models. Using CKT, we gain 1.55 and 2.03 improvements in terms of AUC over DKT and DKVMN respectively, on the ASSISTments2009 dataset. And on the ASSISTments2015 dataset, the corresponding improvements are 1.01 and 1.96 respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Knowledge Tracing with Convolutions. <a href="https://t.co/LTEjuPU3tK">https://t.co/LTEjuPU3tK</a> <a href="https://t.co/M7ZpIkDAoL">pic.twitter.com/M7ZpIkDAoL</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1291032419944943616?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. TOAD-GAN: Coherent Style Level Generation from a Single Example

Maren Awiszus, Frederik Schubert, Bodo Rosenhahn

- retweets: 12, favorites: 45 (08/06/2020 11:01:41)

- links: [abs](https://arxiv.org/abs/2008.01531) | [pdf](https://arxiv.org/pdf/2008.01531)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In this work, we present TOAD-GAN (Token-based One-shot Arbitrary Dimension Generative Adversarial Network), a novel Procedural Content Generation (PCG) algorithm that generates token-based video game levels. TOAD-GAN follows the SinGAN architecture and can be trained using only one example. We demonstrate its application for Super Mario Bros. levels and are able to generate new levels of similar style in arbitrary sizes. We achieve state-of-the-art results in modeling the patterns of the training level and provide a comparison with different baselines under several metrics. Additionally, we present an extension of the method that allows the user to control the generation process of certain token structures to ensure a coherent global level layout. We provide this tool to the community to spur further research by publishing our source code.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TOAD-GAN: Coherent Style Level Generation from a Single Example<br>pdf: <a href="https://t.co/r5msa3bheA">https://t.co/r5msa3bheA</a><br>abs: <a href="https://t.co/E6dwfIaYB5">https://t.co/E6dwfIaYB5</a> <a href="https://t.co/tRFQZaSY8c">pic.twitter.com/tRFQZaSY8c</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1290812008993095680?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. The Exact Asymptotic Form of Bayesian Generalization Error in Latent  Dirichlet Allocation

Naoki Hayashi

- retweets: 12, favorites: 42 (08/06/2020 11:01:41)

- links: [abs](https://arxiv.org/abs/2008.01304) | [pdf](https://arxiv.org/pdf/2008.01304)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.ST](https://arxiv.org/list/math.ST/recent)

Latent Dirichlet allocation (LDA) obtains essential information from data by using Bayesian inference. It is applied to knowledge discovery via dimension reducing and clustering in many fields. However, its generalization error had not been yet clarified since it is a singular statistical model where there is no one to one map from parameters to probability distributions. In this paper, we give the exact asymptotic form of its generalization error and marginal likelihood, by theoretical analysis of its learning coefficient using algebraic geometry. The theoretical result shows that the Bayesian generalization error in LDA is expressed in terms of that in matrix factorization and a penalty from the simplex restriction of LDA's parameter region.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My new gear...<a href="https://t.co/PmFkClWskX">https://t.co/PmFkClWskX</a></p>&mdash; nhayashi (@nhayashi1994) <a href="https://twitter.com/nhayashi1994/status/1290815301093715968?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Multimodal Image-to-Image Translation via a Single Generative  Adversarial Network

Shihua Huang, Cheng He, Ran Cheng

- retweets: 10, favorites: 43 (08/06/2020 11:01:41)

- links: [abs](https://arxiv.org/abs/2008.01681) | [pdf](https://arxiv.org/pdf/2008.01681)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Despite significant advances in image-to-image (I2I) translation with Generative Adversarial Networks (GANs) have been made, it remains challenging to effectively translate an image to a set of diverse images in multiple target domains using a pair of generator and discriminator. Existing multimodal I2I translation methods adopt multiple domain-specific content encoders for different domains, where each domain-specific content encoder is trained with images from the same domain only. Nevertheless, we argue that the content (domain-invariant) features should be learned from images among all the domains. Consequently, each domain-specific content encoder of existing schemes fails to extract the domain-invariant features efficiently. To address this issue, we present a flexible and general SoloGAN model for efficient multimodal I2I translation among multiple domains with unpaired data. In contrast to existing methods, the SoloGAN algorithm uses a single projection discriminator with an additional auxiliary classifier, and shares the encoder and generator for all domains. As such, the SoloGAN model can be trained effectively with images from all domains such that the domain-invariant content representation can be efficiently extracted. Qualitative and quantitative results over a wide range of datasets against several counterparts and variants of the SoloGAN model demonstrate the merits of the method, especially for the challenging I2I translation tasks, i.e., tasks that involve extreme shape variations or need to keep the complex backgrounds unchanged after translations. Furthermore, we demonstrate the contribution of each component using ablation studies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multimodal Image-to-Image Translation via a Single Generative Adversarial Network<br>pdf: <a href="https://t.co/F7dTkPrBmD">https://t.co/F7dTkPrBmD</a><br>abs: <a href="https://t.co/bK1e1lfPbl">https://t.co/bK1e1lfPbl</a> <a href="https://t.co/epjPP3N5bO">pic.twitter.com/epjPP3N5bO</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1290812916871172097?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Tracking Emerges by Looking Around Static Scenes, with Neural 3D Mapping

Adam W. Harley, Shrinidhi K. Lakshmikanth, Paul Schydlo, Katerina Fragkiadaki

- retweets: 13, favorites: 38 (08/06/2020 11:01:41)

- links: [abs](https://arxiv.org/abs/2008.01295) | [pdf](https://arxiv.org/pdf/2008.01295)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We hypothesize that an agent that can look around in static scenes can learn rich visual representations applicable to 3D object tracking in complex dynamic scenes. We are motivated in this pursuit by the fact that the physical world itself is mostly static, and multiview correspondence labels are relatively cheap to collect in static scenes, e.g., by triangulation. We propose to leverage multiview data of \textit{static points} in arbitrary scenes (static or dynamic), to learn a neural 3D mapping module which produces features that are correspondable across time. The neural 3D mapper consumes RGB-D data as input, and produces a 3D voxel grid of deep features as output. We train the voxel features to be correspondable across viewpoints, using a contrastive loss, and correspondability across time emerges automatically. At test time, given an RGB-D video with approximate camera poses, and given the 3D box of an object to track, we track the target object by generating a map of each timestep and locating the object's features within each map. In contrast to models that represent video streams in 2D or 2.5D, our model's 3D scene representation is disentangled from projection artifacts, is stable under camera motion, and is robust to partial occlusions. We test the proposed architectures in challenging simulated and real data, and show that our unsupervised 3D object trackers outperform prior unsupervised 2D and 2.5D trackers, and approach the accuracy of supervised trackers. This work demonstrates that 3D object trackers can emerge without tracking labels, through multiview self-supervision on static data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/ECCV2020?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV2020</a> paper is now on arXiv. We show that 3D object tracking emerges automatically when you train for multi-view correspondence. No object labels necessary!<a href="https://t.co/zQTVSX8nAv">https://t.co/zQTVSX8nAv</a><br>Video: results from KITTI. Bottom right shows a bird&#39;s eye view of the learned 3D features. <a href="https://t.co/8J0bRHpMJ8">pic.twitter.com/8J0bRHpMJ8</a></p>&mdash; Adam W. Harley (@AdamWHarley) <a href="https://twitter.com/AdamWHarley/status/1291087383283343364?ref_src=twsrc%5Etfw">August 5, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



