---
title: Hot Papers 2021-06-18
date: 2021-06-19T07:55:41.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. LoRA: Low-Rank Adaptation of Large Language Models

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Weizhu Chen

- retweets: 2059, favorites: 229 (06/19/2021 07:55:41)

- links: [abs](https://arxiv.org/abs/2106.09685) | [pdf](https://arxiv.org/pdf/2106.09685)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The dominant paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, conventional fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example, deploying many independent instances of fine-tuned models, each with 175B parameters, is extremely expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. For GPT-3, LoRA can reduce the number of trainable parameters by 10,000 times and the computation hardware requirement by 3 times compared to full fine-tuning. LoRA performs on-par or better than fine-tuning in model quality on both GPT-3 and GPT-2, despite having fewer trainable parameters, a higher training throughput, and no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptations, which sheds light on the efficacy of LoRA. We release our implementation in GPT-2 at https://github.com/microsoft/LoRA .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GPT-3 175B is powerful but too expensive to serve many finetuned copies. We use low-rank adaptation (LoRA) to learn task modules that are 10,000x smaller and can be swapped while the main model is frozen. No extra inference latency or quality drop! Paper: <a href="https://t.co/Nz7aMrgRDj">https://t.co/Nz7aMrgRDj</a> <a href="https://t.co/JeACpcpsTV">pic.twitter.com/JeACpcpsTV</a></p>&mdash; Edward Hu (@edwardjhu) <a href="https://twitter.com/edwardjhu/status/1405891519249207298?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Deep Learning Through the Lens of Example Difficulty

Robert J. N. Baldock, Hartmut Maennel, Behnam Neyshabur

- retweets: 717, favorites: 169 (06/19/2021 07:55:41)

- links: [abs](https://arxiv.org/abs/2106.09647) | [pdf](https://arxiv.org/pdf/2106.09647)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Existing work on understanding deep learning often employs measures that compress all data-dependent information into a few numbers. In this work, we adopt a perspective based on the role of individual examples. We introduce a measure of the computational difficulty of making a prediction for a given input: the (effective) prediction depth. Our extensive investigation reveals surprising yet simple relationships between the prediction depth of a given input and the model's uncertainty, confidence, accuracy and speed of learning for that data point. We further categorize difficult examples into three interpretable groups, demonstrate how these groups are processed differently inside deep models and showcase how this understanding allows us to improve prediction accuracy. Insights from our study lead to a coherent view of a number of separately reported phenomena in the literature: early layers generalize while later layers memorize; early layers converge faster and networks learn easy data and simple functions first.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🆕 📰: Deep Learning Through the Lens of Example Difficulty<br><br>We introduce a measure of computational difficulty and show its surprising relationships with different deep learning phenomena.<br><br>Paper: <a href="https://t.co/XDAL98jHFA">https://t.co/XDAL98jHFA</a><br><br>with <a href="https://twitter.com/Robert_Baldock?ref_src=twsrc%5Etfw">@Robert_Baldock</a> &amp; Hartmut Maennel<br> <br>1/ <a href="https://t.co/9pne9WTgTa">pic.twitter.com/9pne9WTgTa</a></p>&mdash; Behnam Neyshabur (@bneyshabur) <a href="https://twitter.com/bneyshabur/status/1405704947770052611?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Modeling Worlds in Text

Prithviraj Ammanabrolu, Mark O. Riedl

- retweets: 576, favorites: 53 (06/19/2021 07:55:41)

- links: [abs](https://arxiv.org/abs/2106.09578) | [pdf](https://arxiv.org/pdf/2106.09578)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We provide a dataset that enables the creation of learning agents that can build knowledge graph-based world models of interactive narratives. Interactive narratives -- or text-adventure games -- are partially observable environments structured as long puzzles or quests in which an agent perceives and interacts with the world purely through textual natural language. Each individual game typically contains hundreds of locations, characters, and objects -- each with their own unique descriptions -- providing an opportunity to study the problem of giving language-based agents the structured memory necessary to operate in such worlds. Our dataset provides 24198 mappings between rich natural language observations and: (1) knowledge graphs that reflect the world state in the form of a map; (2) natural language actions that are guaranteed to cause a change in that particular world state. The training data is collected across 27 games in multiple genres and contains a further 7836 heldout instances over 9 additional games in the test set. We further provide baseline models using rules-based, question-answering, and sequence learning approaches in addition to an analysis of the data and corresponding learning tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing the JerichoWorld Dataset! Designed to measure textual world modeling agents&#39; situated knowledge representation and commonsense reasoning skills. Thousands of autoannotated (text→knowledge graph+actions) pairs across dozens of text games.<a href="https://t.co/FnsVKYk0o6">https://t.co/FnsVKYk0o6</a> <a href="https://t.co/Iu19Tvmk3N">pic.twitter.com/Iu19Tvmk3N</a></p>&mdash; Prithviraj Ammanabrolu (@rajammanabrolu) <a href="https://twitter.com/rajammanabrolu/status/1405915368368050178?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. JOKR: Joint Keypoint Representation for Unsupervised Cross-Domain Motion  Retargeting

Ron Mokady, Rotem Tzaban, Sagie Benaim, Amit H. Bermano, Daniel Cohen-Or

- retweets: 484, favorites: 95 (06/19/2021 07:55:42)

- links: [abs](https://arxiv.org/abs/2106.09679) | [pdf](https://arxiv.org/pdf/2106.09679)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The task of unsupervised motion retargeting in videos has seen substantial advancements through the use of deep neural networks. While early works concentrated on specific object priors such as a human face or body, recent work considered the unsupervised case. When the source and target videos, however, are of different shapes, current methods fail. To alleviate this problem, we introduce JOKR - a JOint Keypoint Representation that captures the motion common to both the source and target videos, without requiring any object prior or data collection. By employing a domain confusion term, we enforce the unsupervised keypoint representations of both videos to be indistinguishable. This encourages disentanglement between the parts of the motion that are common to the two domains, and their distinctive appearance and motion, enabling the generation of videos that capture the motion of the one while depicting the style of the other. To enable cases where the objects are of different proportions or orientations, we apply a learned affine transformation between the JOKRs. This augments the representation to be affine invariant, and in practice broadens the variety of possible retargeting pairs. This geometry-driven representation enables further intuitive control, such as temporal coherence and manual editing. Through comprehensive experimentation, we demonstrate the applicability of our method to different challenging cross-domain video pairs. We evaluate our method both qualitatively and quantitatively, and demonstrate that our method handles various cross-domain scenarios, such as different animals, different flowers, and humans. We also demonstrate superior temporal coherency and visual quality compared to state-of-the-art alternatives, through statistical metrics and a user study. Source code and videos can be found at https://rmokady.github.io/JOKR/ .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">JOKR: Joint Keypoint Representation for Unsupervised Cross-Domain Motion Retargeting<br>pdf: <a href="https://t.co/CXLhNhrwHC">https://t.co/CXLhNhrwHC</a><br>abs: <a href="https://t.co/aIfp4vH6ej">https://t.co/aIfp4vH6ej</a><br>project page: <a href="https://t.co/sPQq7iccFc">https://t.co/sPQq7iccFc</a> <a href="https://t.co/nI3CyRGS1N">pic.twitter.com/nI3CyRGS1N</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405698966390444037?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Algorithmic Bias and Data Bias: Understanding the Relation between  Distributionally Robust Optimization and Data Curation

Agnieszka Słowik, Léon Bottou

- retweets: 443, favorites: 83 (06/19/2021 07:55:42)

- links: [abs](https://arxiv.org/abs/2106.09467) | [pdf](https://arxiv.org/pdf/2106.09467)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Machine learning systems based on minimizing average error have been shown to perform inconsistently across notable subsets of the data, which is not exposed by a low average error for the entire dataset. In consequential social and economic applications, where data represent people, this can lead to discrimination of underrepresented gender and ethnic groups. Given the importance of bias mitigation in machine learning, the topic leads to contentious debates on how to ensure fairness in practice (data bias versus algorithmic bias). Distributionally Robust Optimization (DRO) seemingly addresses this problem by minimizing the worst expected risk across subpopulations. We establish theoretical results that clarify the relation between DRO and the optimization of the same loss averaged on an adequately weighted training dataset. The results cover finite and infinite number of training distributions, as well as convex and non-convex loss functions. We show that neither DRO nor curating the training set should be construed as a complete solution for bias mitigation: in the same way that there is no universally robust training set, there is no universal way to setup a DRO problem and ensure a socially acceptable set of results. We then leverage these insights to provide a mininal set of practical recommendations for addressing bias with DRO. Finally, we discuss ramifications of our results in other related applications of DRO, using an example of adversarial robustness. Our results show that there is merit to both the algorithm-focused and the data-focused side of the bias debate, as long as arguments in favor of these positions are precisely qualified and backed by relevant mathematics known today.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am thrilled to finally see it out! 🥳<br><br>New results on the relation between optimising for the most adverse distribution and reweighting the training set.<br><br>Many thanks to Léon Bottou, <a href="https://twitter.com/fhuszar?ref_src=twsrc%5Etfw">@fhuszar</a> <a href="https://t.co/whcZ691nY7">https://t.co/whcZ691nY7</a></p>&mdash; Aga Słowik (@slowiika) <a href="https://twitter.com/slowiika/status/1405806071839592461?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A Fait Accompli? An Empirical Study into the Absence of Consent to  Third-Party Tracking in Android~Apps

Konrad Kollnig, Reuben Binns, Pierre Dewitte, Max Van Kleek, Ge Wang, Daniel Omeiza, Helena Webb, Nigel Shadbolt

- retweets: 332, favorites: 55 (06/19/2021 07:55:42)

- links: [abs](https://arxiv.org/abs/2106.09407) | [pdf](https://arxiv.org/pdf/2106.09407)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

Third-party tracking allows companies to collect users' behavioural data and track their activity across digital devices. This can put deep insights into users' private lives into the hands of strangers, and often happens without users' awareness or explicit consent. EU and UK data protection law, however, requires consent, both 1) to access and store information on users' devices and 2) to legitimate the processing of personal data as part of third-party tracking, as we analyse in this paper.   This paper further investigates whether and to what extent consent is implemented in mobile apps. First, we analyse a representative sample of apps from the Google Play Store. We find that most apps engage in third-party tracking, but few obtained consent before doing so, indicating potentially widespread violations of EU and UK privacy law. Second, we examine the most common third-party tracking libraries in detail. While most acknowledge that they rely on app developers to obtain consent on their behalf, they typically fail to put in place robust measures to ensure this: disclosure of consent requirements is limited; default consent implementations are lacking; and compliance guidance is difficult to find, hard to read, and poorly maintained.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My first academic article, together with <a href="https://twitter.com/RDBinns?ref_src=twsrc%5Etfw">@RDBinns</a> <a href="https://twitter.com/PiDewitte?ref_src=twsrc%5Etfw">@PiDewitte</a> <a href="https://twitter.com/emax?ref_src=twsrc%5Etfw">@emax</a> Ge Wang <a href="https://twitter.com/SteadyBits?ref_src=twsrc%5Etfw">@SteadyBits</a> Helena Webb <a href="https://twitter.com/Nigel_Shadbolt?ref_src=twsrc%5Etfw">@Nigel_Shadbolt</a>, will appear at the <a href="https://twitter.com/SOUPSConference?ref_src=twsrc%5Etfw">@SOUPSConference</a>.<br><br>We find that most mobile apps track users, need consent from EU+UK users, but do not seek consent.<a href="https://t.co/brJL6YRpRS">https://t.co/brJL6YRpRS</a></p>&mdash; Konrad 😷 (@KKollnig) <a href="https://twitter.com/KKollnig/status/1405800862325395460?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Multi-head or Single-head? An Empirical Comparison for Transformer  Training

Liyuan Liu, Jialu Liu, Jiawei Han

- retweets: 266, favorites: 108 (06/19/2021 07:55:42)

- links: [abs](https://arxiv.org/abs/2106.09650) | [pdf](https://arxiv.org/pdf/2106.09650)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Multi-head attention plays a crucial role in the recent success of Transformer models, which leads to consistent performance improvements over conventional attention in various applications. The popular belief is that this effectiveness stems from the ability of jointly attending multiple positions. In this paper, we first demonstrate that jointly attending multiple positions is not a unique feature of multi-head attention, as multi-layer single-head attention also attends multiple positions and is more effective. Then, we suggest the main advantage of the multi-head attention is the training stability, since it has less number of layers than the single-head attention, when attending the same number of positions. For example, 24-layer 16-head Transformer (BERT-large) and 384-layer single-head Transformer has the same total attention head number and roughly the same model size, while the multi-head one is significantly shallower. Meanwhile, we show that, with recent advances in deep learning, we can successfully stabilize the training of the 384-layer Transformer. As the training difficulty is no longer a bottleneck, substantially deeper single-head Transformer achieves consistent performance improvements without tuning hyper-parameters.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-head or Single-head? An Empirical Comparison for Transformer Training<br>pdf: <a href="https://t.co/89iVdeAWcY">https://t.co/89iVdeAWcY</a><br>abs: <a href="https://t.co/gXjRZVKr5m">https://t.co/gXjRZVKr5m</a> <a href="https://t.co/p4oi7RgnIn">pic.twitter.com/p4oi7RgnIn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405695274740224000?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis

Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, Najim Dehak, William Chan

- retweets: 220, favorites: 141 (06/19/2021 07:55:42)

- links: [abs](https://arxiv.org/abs/2106.09660) | [pdf](https://arxiv.org/pdf/2106.09660)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

This paper introduces WaveGrad 2, a non-autoregressive generative model for text-to-speech synthesis. WaveGrad 2 is trained to estimate the gradient of the log conditional density of the waveform given a phoneme sequence. The model takes an input phoneme sequence, and through an iterative refinement process, generates an audio waveform. This contrasts to the original WaveGrad vocoder which conditions on mel-spectrogram features, generated by a separate model. The iterative refinement process starts from Gaussian noise, and through a series of refinement steps (e.g., 50 steps), progressively recovers the audio sequence. WaveGrad 2 offers a natural way to trade-off between inference speed and sample quality, through adjusting the number of refinement steps. Experiments show that the model can generate high fidelity audio, approaching the performance of a state-of-the-art neural TTS system. We also report various ablation studies over different model configurations. Audio samples are available at https://wavegrad.github.io/v2.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis<br>pdf: <a href="https://t.co/ku5NUqprfJ">https://t.co/ku5NUqprfJ</a><br>abs: <a href="https://t.co/zxiGIkViQ2">https://t.co/zxiGIkViQ2</a><br><br>end-to-end nonautoregressive TTS model, takes a phoneme sequence as input, synthesizes the waveform directly without using hand-designed intermediate features <a href="https://t.co/6HFWG3FZIy">pic.twitter.com/6HFWG3FZIy</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405708875911806980?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">WaveGrad 2 -- Iterative Refinement for Text-to-Speech Synthesis<br><br>&quot;WaveGrad 2 is trained to estimate the gradient of the log conditional density of the waveform given a phoneme sequence. &quot;<a href="https://t.co/02e1V4MqMF">https://t.co/02e1V4MqMF</a></p>&mdash; Heiga Zen (全 炳河) (@heiga_zen) <a href="https://twitter.com/heiga_zen/status/1405719438054215682?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">``WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis. (arXiv:2106.09660v1 [<a href="https://t.co/3pcQCkeyAA">https://t.co/3pcQCkeyAA</a>]),&#39;&#39; Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, Najim Dehak, William Chan, <a href="https://t.co/MA5BBRH2O4">https://t.co/MA5BBRH2O4</a></p>&mdash; arXiv Sound (@ArxivSound) <a href="https://twitter.com/ArxivSound/status/1405823467577131016?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Cat-like Jumping and Landing of Legged Robots in Low-gravity Using Deep  Reinforcement Learning

Nikita Rudin, Hendrik Kolvenbach, Vassilios Tsounis, Marco Hutter

- retweets: 272, favorites: 45 (06/19/2021 07:55:42)

- links: [abs](https://arxiv.org/abs/2106.09357) | [pdf](https://arxiv.org/pdf/2106.09357)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

In this article, we show that learned policies can be applied to solve legged locomotion control tasks with extensive flight phases, such as those encountered in space exploration. Using an off-the-shelf deep reinforcement learning algorithm, we trained a neural network to control a jumping quadruped robot while solely using its limbs for attitude control. We present tasks of increasing complexity leading to a combination of three-dimensional (re-)orientation and landing locomotion behaviors of a quadruped robot traversing simulated low-gravity celestial bodies. We show that our approach easily generalizes across these tasks and successfully trains policies for each case. Using sim-to-real transfer, we deploy trained policies in the real world on the SpaceBok robot placed on an experimental testbed designed for two-dimensional micro-gravity experiments. The experimental results demonstrate that repetitive, controlled jumping and landing with natural agility is possible.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Cat-like Jumping and Landing of Legged Robots in<br>Low-gravity Using Deep Reinforcement Learning<br>pdf: <a href="https://t.co/DYmfvBZo3t">https://t.co/DYmfvBZo3t</a><br>abs: <a href="https://t.co/nlOnHmq8iY">https://t.co/nlOnHmq8iY</a> <a href="https://t.co/OYQKfk5zNG">pic.twitter.com/OYQKfk5zNG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405717792469495811?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. NeuroMorph: Unsupervised Shape Interpolation and Correspondence in One  Go

Marvin Eisenberger, David Novotny, Gael Kerchenbaum, Patrick Labatut, Natalia Neverova, Daniel Cremers, Andrea Vedaldi

- retweets: 225, favorites: 70 (06/19/2021 07:55:43)

- links: [abs](https://arxiv.org/abs/2106.09431) | [pdf](https://arxiv.org/pdf/2106.09431)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present NeuroMorph, a new neural network architecture that takes as input two 3D shapes and produces in one go, i.e. in a single feed forward pass, a smooth interpolation and point-to-point correspondences between them. The interpolation, expressed as a deformation field, changes the pose of the source shape to resemble the target, but leaves the object identity unchanged. NeuroMorph uses an elegant architecture combining graph convolutions with global feature pooling to extract local features. During training, the model is incentivized to create realistic deformations by approximating geodesics on the underlying shape space manifold. This strong geometric prior allows to train our model end-to-end and in a fully unsupervised manner without requiring any manual correspondence annotations. NeuroMorph works well for a large variety of input shapes, including non-isometric pairs from different object categories. It obtains state-of-the-art results for both shape correspondence and interpolation tasks, matching or surpassing the performance of recent unsupervised and supervised methods on multiple benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeuroMorph: Unsupervised Shape Interpolation and Correspondence in One Go<br>pdf: <a href="https://t.co/Ed2bkowE17">https://t.co/Ed2bkowE17</a><br>abs: <a href="https://t.co/duALfAAVvA">https://t.co/duALfAAVvA</a><br><br>can be trained in a fully unsupervised manner and generates correspondence and interpolation in a single pass <a href="https://t.co/dSdat7xoQC">pic.twitter.com/dSdat7xoQC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405697960407666693?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. XCiT: Cross-Covariance Image Transformers

Alaaeldin El-Nouby, Hugo Touvron, Mathilde Caron, Piotr Bojanowski, Matthijs Douze, Armand Joulin, Ivan Laptev, Natalia Neverova, Gabriel Synnaeve, Jakob Verbeek, Hervé Jegou

- retweets: 81, favorites: 79 (06/19/2021 07:55:43)

- links: [abs](https://arxiv.org/abs/2106.09681) | [pdf](https://arxiv.org/pdf/2106.09681)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Following their success in natural language processing, transformers have recently shown much promise for computer vision. The self-attention operation underlying transformers yields global interactions between all tokens ,i.e. words or image patches, and enables flexible modelling of image data beyond the local interactions of convolutions. This flexibility, however, comes with a quadratic complexity in time and memory, hindering application to long sequences and high-resolution images. We propose a "transposed" version of self-attention that operates across feature channels rather than tokens, where the interactions are based on the cross-covariance matrix between keys and queries. The resulting cross-covariance attention (XCA) has linear complexity in the number of tokens, and allows efficient processing of high-resolution images. Our cross-covariance image transformer (XCiT) is built upon XCA. It combines the accuracy of conventional transformers with the scalability of convolutional architectures. We validate the effectiveness and generality of XCiT by reporting excellent results on multiple vision benchmarks, including image classification and self-supervised feature learning on ImageNet-1k, object detection and instance segmentation on COCO, and semantic segmentation on ADE20k.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">XCiT: Cross-Covariance Image Transformers<br>pdf: <a href="https://t.co/7bx45TodYQ">https://t.co/7bx45TodYQ</a><br>abs: <a href="https://t.co/iWCe3fTP0S">https://t.co/iWCe3fTP0S</a><br>github: <a href="https://t.co/3ohiyazMiW">https://t.co/3ohiyazMiW</a><br><br>alternative to token self-attention, operates on the<br>feature dimension, eliminating the need for expensive computation of quadratic attention<br>maps <a href="https://t.co/jDfvPiJx79">pic.twitter.com/jDfvPiJx79</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405689985764839427?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Paper is available here: <a href="https://t.co/qmkVGbZUGR">https://t.co/qmkVGbZUGR</a></p>&mdash; Facebook AI (@facebookai) <a href="https://twitter.com/facebookai/status/1405912076535558145?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Long-Short Temporal Contrastive Learning of Video Transformers

Jue Wang, Gedas Bertasius, Du Tran, Lorenzo Torresani

- retweets: 100, favorites: 49 (06/19/2021 07:55:43)

- links: [abs](https://arxiv.org/abs/2106.09212) | [pdf](https://arxiv.org/pdf/2106.09212)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Video transformers have recently emerged as a competitive alternative to 3D CNNs for video understanding. However, due to their large number of parameters and reduced inductive biases, these models require supervised pretraining on large-scale image datasets to achieve top performance. In this paper, we empirically demonstrate that self-supervised pretraining of video transformers on video-only datasets can lead to action recognition results that are on par or better than those obtained with supervised pretraining on large-scale image datasets, even massive ones such as ImageNet-21K. Since transformer-based models are effective at capturing dependencies over extended temporal spans, we propose a simple learning procedure that forces the model to match a long-term view to a short-term view of the same video. Our approach, named Long-Short Temporal Contrastive Learning (LSTCL), enables video transformers to learn an effective clip-level representation by predicting temporal context captured from a longer temporal extent. To demonstrate the generality of our findings, we implement and validate our approach under three different self-supervised contrastive learning frameworks (MoCo v3, BYOL, SimSiam) using two distinct video-transformer architectures, including an improved variant of the Swin Transformer augmented with space-time attention. We conduct a thorough ablation study and show that LSTCL achieves competitive performance on multiple video benchmarks and represents a convincing alternative to supervised image-based pretraining.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Long-Short Temporal Contrastive Learning of Video Transformers<br>pdf: <a href="https://t.co/dYlGGUtjIF">https://t.co/dYlGGUtjIF</a><br>abs: <a href="https://t.co/WpwdGHCIFg">https://t.co/WpwdGHCIFg</a><br><br>unsupervised pretraining with LSTCL leads to similar or better video classification accuracy compared to pretraining with full supervision on ImageNet-21K <a href="https://t.co/RPK9bsV8gH">pic.twitter.com/RPK9bsV8gH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405687594831208448?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Learning to Predict Visual Attributes in the Wild

Khoi Pham, Kushal Kafle, Zhe Lin, Zhihong Ding, Scott Cohen, Quan Tran, Abhinav Shrivastava

- retweets: 90, favorites: 32 (06/19/2021 07:55:43)

- links: [abs](https://arxiv.org/abs/2106.09707) | [pdf](https://arxiv.org/pdf/2106.09707)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Visual attributes constitute a large portion of information contained in a scene. Objects can be described using a wide variety of attributes which portray their visual appearance (color, texture), geometry (shape, size, posture), and other intrinsic properties (state, action). Existing work is mostly limited to study of attribute prediction in specific domains. In this paper, we introduce a large-scale in-the-wild visual attribute prediction dataset consisting of over 927K attribute annotations for over 260K object instances. Formally, object attribute prediction is a multi-label classification problem where all attributes that apply to an object must be predicted. Our dataset poses significant challenges to existing methods due to large number of attributes, label sparsity, data imbalance, and object occlusion. To this end, we propose several techniques that systematically tackle these challenges, including a base model that utilizes both low- and high-level CNN features with multi-hop attention, reweighting and resampling techniques, a novel negative label expansion scheme, and a novel supervised attribute-aware contrastive learning algorithm. Using these techniques, we achieve near 3.7 mAP and 5.7 overall F1 points improvement over the current state of the art. Further details about the VAW dataset can be found at http://vawdataset.com/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning to Predict Visual Attributes in the Wild<br>pdf: <a href="https://t.co/y0oDMW8GCu">https://t.co/y0oDMW8GCu</a><br>abs: <a href="https://t.co/lw0IBDEffy">https://t.co/lw0IBDEffy</a><br>dataset: <a href="https://t.co/QRxAgaxRhm">https://t.co/QRxAgaxRhm</a><br><br>large-scale in-the wild visual attribute prediction dataset consisting of over 927K attribute annotations for over 260K object instances <a href="https://t.co/YgIWktqif0">pic.twitter.com/YgIWktqif0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405723239746359310?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Scaling Laws for Acoustic Models

Jasha Droppo, Oguz Elibol

- retweets: 36, favorites: 49 (06/19/2021 07:55:43)

- links: [abs](https://arxiv.org/abs/2106.09488) | [pdf](https://arxiv.org/pdf/2106.09488)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

There is a recent trend in machine learning to increase model quality by growing models to sizes previously thought to be unreasonable. Recent work has shown that autoregressive generative models with cross-entropy objective functions exhibit smooth power-law relationships, or scaling laws, that predict model quality from model size, training set size, and the available compute budget. These scaling laws allow one to choose nearly optimal hyper-parameters given constraints on available training data, model parameter count, or training computation budget. In this paper, we demonstrate that acoustic models trained with an auto-predictive coding loss behave as if they are subject to similar scaling laws. We extend previous work to jointly predict loss due to model size, to training set size, and to the inherent "irreducible loss" of the task. We find that the scaling laws accurately match model performance over two orders of magnitude in both model size and training set size, and make predictions about the limits of model performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Laws for Acoustic Models<br>pdf: <a href="https://t.co/VUk1ufSuya">https://t.co/VUk1ufSuya</a><br>abs: <a href="https://t.co/1Cx2nsTJzD">https://t.co/1Cx2nsTJzD</a><br><br>demonstrate that acoustic models trained with an auto-predictive coding loss behave as if they are subject to similar scaling laws <a href="https://t.co/w36CmGIdun">pic.twitter.com/w36CmGIdun</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405690909644279808?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Learning Knowledge Graph-based World Models of Textual Environments

Prithviraj Ammanabrolu, Mark O. Riedl

- retweets: 51, favorites: 28 (06/19/2021 07:55:43)

- links: [abs](https://arxiv.org/abs/2106.09608) | [pdf](https://arxiv.org/pdf/2106.09608)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

World models improve a learning agent's ability to efficiently operate in interactive and situated environments. This work focuses on the task of building world models of text-based game environments. Text-based games, or interactive narratives, are reinforcement learning environments in which agents perceive and interact with the world using textual natural language. These environments contain long, multi-step puzzles or quests woven through a world that is filled with hundreds of characters, locations, and objects. Our world model learns to simultaneously: (1) predict changes in the world caused by an agent's actions when representing the world as a knowledge graph; and (2) generate the set of contextually relevant natural language actions required to operate in the world. We frame this task as a Set of Sequences generation problem by exploiting the inherent structure of knowledge graphs and actions and introduce both a transformer-based multi-task architecture and a loss function to train it. A zero-shot ablation study on never-before-seen textual worlds shows that our methodology significantly outperforms existing textual world modeling techniques as well as the importance of each of our contributions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Knowledge Graph-based World Models of Textual Environments<br>pdf: <a href="https://t.co/P5oVsTJpCj">https://t.co/P5oVsTJpCj</a><br>abs: <a href="https://t.co/4pxVRAJ07V">https://t.co/4pxVRAJ07V</a><br><br>a sota world model for text games <a href="https://t.co/78niKJoPFL">pic.twitter.com/78niKJoPFL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405694371467599882?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Do Large Scale Molecular Language Representations Capture Important  Structural Information?

Jerret Ross, Brian Belgodere, Vijil Chenthamarakshan, Inkit Padhi, Youssef Mroueh, Payel Das

- retweets: 36, favorites: 31 (06/19/2021 07:55:44)

- links: [abs](https://arxiv.org/abs/2106.09553) | [pdf](https://arxiv.org/pdf/2106.09553)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent)

Predicting chemical properties from the structure of a molecule is of great importance in many applications including drug discovery and material design. Machine learning based molecular property prediction holds the promise of enabling accurate predictions at much less complexity, when compared to, for example Density Functional Theory (DFT) calculations. Features extracted from molecular graphs, using graph neural nets in a supervised manner, have emerged as strong baselines for such tasks. However, the vast chemical space together with the limited availability of labels makes supervised learning challenging, calling for learning a general-purpose molecular representation. Recently, pre-trained transformer-based language models (PTLMs) on large unlabeled corpus have produced state-of-the-art results in many downstream natural language processing tasks. Inspired by this development, here we present molecular embeddings obtained by training an efficient transformer encoder model, referred to as MoLFormer. This model was employed with a linear attention mechanism and highly paralleized training on 1D SMILES sequences of 1.1 billion unlabeled molecules from the PubChem and ZINC datasets. Experiments show that the learned molecular representation performs competitively, when compared to existing graph-based and fingerprint-based supervised learning baselines, on the challenging tasks of predicting properties of QM8 and QM9 molecules. Further task-specific fine-tuning of the MoLFormerr representation improves performance on several of those property prediction benchmarks. These results provide encouraging evidence that large-scale molecular language models can capture sufficient structural information to be able to accurately predict quantum chemical properties and beyond.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do Large Scale Molecular Language Representations<br>Capture Important Structural Information?<br>pdf: <a href="https://t.co/vrKHPinBUI">https://t.co/vrKHPinBUI</a><br>abs: <a href="https://t.co/UAfv3Jp5Xv">https://t.co/UAfv3Jp5Xv</a><br><br>validates the power of large scale self-supervised pre-trained molecular language models on electronic property prediction tasks <a href="https://t.co/AV3DiirWlH">pic.twitter.com/AV3DiirWlH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405693214363992065?ref_src=twsrc%5Etfw">June 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



