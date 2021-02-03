---
title: Hot Papers 2021-02-02
date: 2021-02-03T10:20:09.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-02"
socialImage: "/media/flying-marine.jpg"

---

# 1. Machine learning accelerated computational fluid dynamics

Dmitrii Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, Stephan Hoyer

- retweets: 9526, favorites: 14 (02/03/2021 10:20:09)

- links: [abs](https://arxiv.org/abs/2102.01010) | [pdf](https://arxiv.org/pdf/2102.01010)
- [physics.flu-dyn](https://arxiv.org/list/physics.flu-dyn/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Numerical simulation of fluids plays an essential role in modeling many physical phenomena, such as weather, climate, aerodynamics and plasma physics. Fluids are well described by the Navier-Stokes equations, but solving these equations at scale remains daunting, limited by the computational cost of resolving the smallest spatiotemporal features. This leads to unfavorable trade-offs between accuracy and tractability. Here we use end-to-end deep learning to improve approximations inside computational fluid dynamics for modeling two-dimensional turbulent flows. For both direct numerical simulation of turbulence and large eddy simulation, our results are as accurate as baseline solvers with 8-10x finer resolution in each spatial dimension, resulting in 40-80x fold computational speedups. Our method remains stable during long simulations, and generalizes to forcing functions and Reynolds numbers outside of the flows where it is trained, in contrast to black box machine learning approaches. Our approach exemplifies how scientific computing can leverage machine learning and hardware accelerators to improve simulations without sacrificing accuracy or generalization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/2<br>Excited to share &quot;Machine learning accelerated computational fluid dynamics&quot;<a href="https://t.co/8rXhLGTVZC">https://t.co/8rXhLGTVZC</a><br><br>We use ML inside a CFD simulator to advance the accuracy/speed Pareto frontier<br><br>with/<br>Jamie A. Smith <br>Ayya Alieva<br>Qing Wang<br>Michael P. Brenner<a href="https://twitter.com/shoyer?ref_src=twsrc%5Etfw">@shoyer</a> <a href="https://t.co/RcQDfEAqkH">pic.twitter.com/RcQDfEAqkH</a></p>&mdash; Dmitrii Kochkov (@dkochkov1) <a href="https://twitter.com/dkochkov1/status/1356440834921627650?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Can We Automate Scientific Reviewing?

Weizhe Yuan, Pengfei Liu, Graham Neubig

- retweets: 6741, favorites: 8 (02/03/2021 10:20:09)

- links: [abs](https://arxiv.org/abs/2102.00176) | [pdf](https://arxiv.org/pdf/2102.00176)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The rapid development of science and technology has been accompanied by an exponential growth in peer-reviewed scientific publications. At the same time, the review of each paper is a laborious process that must be carried out by subject matter experts. Thus, providing high-quality reviews of this growing number of papers is a significant challenge. In this work, we ask the question "can we automate scientific reviewing?", discussing the possibility of using state-of-the-art natural language processing (NLP) models to generate first-pass peer reviews for scientific papers. Arguably the most difficult part of this is defining what a "good" review is in the first place, so we first discuss possible evaluation measures for such reviews. We then collect a dataset of papers in the machine learning domain, annotate them with different aspects of content covered in each review, and train targeted summarization models that take in papers to generate reviews. Comprehensive experimental results show that system-generated reviews tend to touch upon more aspects of the paper than human-written reviews, but the generated text can suffer from lower constructiveness for all aspects except the explanation of the core ideas of the papers, which are largely factually correct. We finally summarize eight challenges in the pursuit of a good review generation system together with potential solutions, which, hopefully, will inspire more future research on this subject. We make all code, and the dataset publicly available: https://github.com/neulab/ReviewAdvisor, as well as a ReviewAdvisor system: http://review.nlpedia.ai/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Here&#39;s a scary thought: <br><br>&quot;Can We Automate Scientific Reviewing?&quot;<a href="https://t.co/TAr9Dpwklg">https://t.co/TAr9Dpwklg</a><br><br>Try it for yourself:<a href="https://t.co/OWVzZY3G8g">https://t.co/OWVzZY3G8g</a><br><br>For the love of all that is good, please don&#39;t use this hastily in your own reviewing work. <a href="https://t.co/FGtP8OzIcX">pic.twitter.com/FGtP8OzIcX</a></p>&mdash; Peyman Milanfar (@docmilanfar) <a href="https://twitter.com/docmilanfar/status/1356476249548353537?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Can Small and Synthetic Benchmarks Drive Modeling Innovation? A  Retrospective Study of Question Answering Modeling Approaches

Nelson F. Liu, Tony Lee, Robin Jia, Percy Liang

- retweets: 2864, favorites: 257 (02/03/2021 10:20:09)

- links: [abs](https://arxiv.org/abs/2102.01065) | [pdf](https://arxiv.org/pdf/2102.01065)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Datasets are not only resources for training accurate, deployable systems, but are also benchmarks for developing new modeling approaches. While large, natural datasets are necessary for training accurate systems, are they necessary for driving modeling innovation? For example, while the popular SQuAD question answering benchmark has driven the development of new modeling approaches, could synthetic or smaller benchmarks have led to similar innovations?   This counterfactual question is impossible to answer, but we can study a necessary condition: the ability for a benchmark to recapitulate findings made on SQuAD. We conduct a retrospective study of 20 SQuAD modeling approaches, investigating how well 32 existing and synthesized benchmarks concur with SQuAD -- i.e., do they rank the approaches similarly? We carefully construct small, targeted synthetic benchmarks that do not resemble natural language, yet have high concurrence with SQuAD, demonstrating that naturalness and size are not necessary for reflecting historical modeling improvements on SQuAD. Our results raise the intriguing possibility that small and carefully designed synthetic benchmarks may be useful for driving the development of new modeling approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large, natural datasets are invaluable for training accurate, deployable systems, but are they required for driving modeling innovation? Can we use small, synthetic benchmarks instead? Our new paper asks this: <a href="https://t.co/b9WYYonQxi">https://t.co/b9WYYonQxi</a><br><br>w/ Tony Lee, <a href="https://twitter.com/robinomial?ref_src=twsrc%5Etfw">@robinomial</a>, <a href="https://twitter.com/percyliang?ref_src=twsrc%5Etfw">@percyliang</a><br><br>(1/8) <a href="https://t.co/sdhRk5UT5L">pic.twitter.com/sdhRk5UT5L</a></p>&mdash; Nelson Liu (@nelsonfliu) <a href="https://twitter.com/nelsonfliu/status/1356459339280642048?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Speech Recognition by Simply Fine-tuning BERT

Wen-Chin Huang, Chia-Hua Wu, Shang-Bao Luo, Kuan-Yu Chen, Hsin-Min Wang, Tomoki Toda

- retweets: 1332, favorites: 230 (02/03/2021 10:20:09)

- links: [abs](https://arxiv.org/abs/2102.00291) | [pdf](https://arxiv.org/pdf/2102.00291)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We propose a simple method for automatic speech recognition (ASR) by fine-tuning BERT, which is a language model (LM) trained on large-scale unlabeled text data and can generate rich contextual representations. Our assumption is that given a history context sequence, a powerful LM can narrow the range of possible choices and the speech signal can be used as a simple clue. Hence, comparing to conventional ASR systems that train a powerful acoustic model (AM) from scratch, we believe that speech recognition is possible by simply fine-tuning a BERT model. As an initial study, we demonstrate the effectiveness of the proposed idea on the AISHELL dataset and show that stacking a very simple AM on top of BERT can yield reasonable performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Speech Recognition by Simply Fine-tuning BERT<br>pdf: <a href="https://t.co/2kit83mnj9">https://t.co/2kit83mnj9</a><br>abs: <a href="https://t.co/qyOosTp8Ey">https://t.co/qyOosTp8Ey</a> <a href="https://t.co/brIQfVxWim">pic.twitter.com/brIQfVxWim</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356430939119902721?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. About Face: A Survey of Facial Recognition Evaluation

Inioluwa Deborah Raji, Genevieve Fried

- retweets: 756, favorites: 89 (02/03/2021 10:20:09)

- links: [abs](https://arxiv.org/abs/2102.00813) | [pdf](https://arxiv.org/pdf/2102.00813)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We survey over 100 face datasets constructed between 1976 to 2019 of 145 million images of over 17 million subjects from a range of sources, demographics and conditions. Our historical survey reveals that these datasets are contextually informed, shaped by changes in political motivations, technological capability and current norms. We discuss how such influences mask specific practices (some of which may actually be harmful or otherwise problematic) and make a case for the explicit communication of such details in order to establish a more grounded understanding of the technology's function in the real world.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A long overdue pre-print is finally out today!📣📣<br><br>Me &amp; <a href="https://twitter.com/genmaicha____?ref_src=twsrc%5Etfw">@genmaicha____</a> wrote of the horrors we find looking through over 100 face datasets with  145  million  images  of  over  17  million  subjects (clues: lots of children &amp; Mexican VISAs). <a href="https://t.co/WPQ9F1F9tI">https://t.co/WPQ9F1F9tI</a> <a href="https://t.co/cugBg6DeIT">pic.twitter.com/cugBg6DeIT</a></p>&mdash; Deb Raji (@rajiinio) <a href="https://twitter.com/rajiinio/status/1356715699939508231?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Sparsity in Deep Learning: Pruning and growth for efficient inference  and training in neural networks

Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, Alexandra Peste

- retweets: 730, favorites: 92 (02/03/2021 10:20:10)

- links: [abs](https://arxiv.org/abs/2102.00554) | [pdf](https://arxiv.org/pdf/2102.00554)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.AR](https://arxiv.org/list/cs.AR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

The growing energy and performance costs of deep learning have driven the community to reduce the size of neural networks by selectively pruning components. Similarly to their biological counterparts, sparse networks generalize just as well, if not better than, the original dense networks. Sparsity can reduce the memory footprint of regular networks to fit mobile devices, as well as shorten training time for ever growing networks. In this paper, we survey prior work on sparsity in deep learning and provide an extensive tutorial of sparsification for both inference and training. We describe approaches to remove and add elements of neural networks, different training strategies to achieve model sparsity, and mechanisms to exploit sparsity in practice. Our work distills ideas from more than 300 research papers and provides guidance to practitioners who wish to utilize sparsity today, as well as to researchers whose goal is to push the frontier forward. We include the necessary background on mathematical methods in sparsification, describe phenomena such as early structure adaptation, the intricate relations between sparsity and the training process, and show techniques for achieving acceleration on real hardware. We also define a metric of pruned parameter efficiency that could serve as a baseline for comparison of different sparse networks. We close by speculating on how sparsity can improve future workloads and outline major open problems in the field.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The future of <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> is sparse! See our overview of the field and upcoming opportunities for how to gain 10-100x performance to fuel the next <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> revolution. <a href="https://twitter.com/hashtag/HPC?src=hash&amp;ref_src=twsrc%5Etfw">#HPC</a> techniques will be key as large-scale training is <a href="https://twitter.com/hashtag/supercomputing?src=hash&amp;ref_src=twsrc%5Etfw">#supercomputing</a>.<a href="https://t.co/Pji3zVk2kc">https://t.co/Pji3zVk2kc</a><a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://t.co/jDaiiGzoTt">pic.twitter.com/jDaiiGzoTt</a></p>&mdash; Torsten Hoefler (@thoefler) <a href="https://twitter.com/thoefler/status/1356531629406253061?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Measuring and Improving Consistency in Pretrained Language Models

Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, Yoav Goldberg

- retweets: 558, favorites: 116 (02/03/2021 10:20:10)

- links: [abs](https://arxiv.org/abs/2102.01017) | [pdf](https://arxiv.org/pdf/2102.01017)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Consistency of a model -- that is, the invariance of its behavior under meaning-preserving alternations in its input -- is a highly desirable property in natural language processing. In this paper we study the question: Are Pretrained Language Models (PLMs) consistent with respect to factual knowledge? To this end, we create ParaRel, a high-quality resource of cloze-style query English paraphrases. It contains a total of 328 paraphrases for thirty-eight relations. Using ParaRel, we show that the consistency of all PLMs we experiment with is poor -- though with high variance between relations. Our analysis of the representational spaces of PLMs suggests that they have a poor structure and are currently not suitable for representing knowledge in a robust way. Finally, we propose a method for improving model consistency and experimentally demonstrate its effectiveness.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Are our Language Models consistent? Apparently not!<br>Our new paper quantifies that: <a href="https://t.co/yaOLpbreFW">https://t.co/yaOLpbreFW</a><br><br>w/ <a href="https://twitter.com/KassnerNora?ref_src=twsrc%5Etfw">@KassnerNora</a>, <a href="https://twitter.com/ravfogel?ref_src=twsrc%5Etfw">@ravfogel</a>, <a href="https://twitter.com/Lasha1608?ref_src=twsrc%5Etfw">@Lasha1608</a>, Ed Hovy, <a href="https://twitter.com/HinrichSchuetze?ref_src=twsrc%5Etfw">@HinrichSchuetze</a>, and <a href="https://twitter.com/yoavgo?ref_src=twsrc%5Etfw">@yoavgo</a> <a href="https://t.co/7mb2KhZxFE">pic.twitter.com/7mb2KhZxFE</a></p>&mdash; lazary (@yanaiela) <a href="https://twitter.com/yanaiela/status/1356541380412211215?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Video Transformer Network

Daniel Neimark, Omri Bar, Maya Zohar, Dotan Asselmann

- retweets: 454, favorites: 141 (02/03/2021 10:20:10)

- links: [abs](https://arxiv.org/abs/2102.00719) | [pdf](https://arxiv.org/pdf/2102.00719)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper presents VTN, a transformer-based framework for video recognition. Inspired by recent developments in vision transformers, we ditch the standard approach in video action recognition that relies on 3D ConvNets and introduce a method that classifies actions by attending to the entire video sequence information. Our approach is generic and builds on top of any given 2D spatial network. In terms of wall runtime, it trains $16.1\times$ faster and runs $5.1\times$ faster during inference while maintaining competitive accuracy compared to other state-of-the-art methods. It enables whole video analysis, via a single end-to-end pass, while requiring $1.5\times$ fewer GFLOPs. We report competitive results on Kinetics-400 and present an ablation study of VTN properties and the trade-off between accuracy and inference speed. We hope our approach will serve as a new baseline and start a fresh line of research in the video recognition domain. Code and models will be available soon.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Video Transformer Network<br>pdf: <a href="https://t.co/LPEKgWBHSu">https://t.co/LPEKgWBHSu</a><br>abs: <a href="https://t.co/yN319VZu9J">https://t.co/yN319VZu9J</a> <a href="https://t.co/fdn91cAUu9">pic.twitter.com/fdn91cAUu9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356449447161585666?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Melon Playlist Dataset: a public dataset for audio-based playlist  generation and music tagging

Andres Ferraro, Yuntae Kim, Soohyeon Lee, Biho Kim, Namjun Jo, Semi Lim, Suyon Lim, Jungtaek Jang, Sehwan Kim, Xavier Serra, Dmitry Bogdanov

- retweets: 262, favorites: 61 (02/03/2021 10:20:10)

- links: [abs](https://arxiv.org/abs/2102.00201) | [pdf](https://arxiv.org/pdf/2102.00201)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

One of the main limitations in the field of audio signal processing is the lack of large public datasets with audio representations and high-quality annotations due to restrictions of copyrighted commercial music. We present Melon Playlist Dataset, a public dataset of mel-spectrograms for 649,091tracks and 148,826 associated playlists annotated by 30,652 different tags. All the data is gathered from Melon, a popular Korean streaming service. The dataset is suitable for music information retrieval tasks, in particular, auto-tagging and automatic playlist continuation. Even though the latter can be addressed by collaborative filtering approaches, audio provides opportunities for research on track suggestions and building systems resistant to the cold-start problem, for which we provide a baseline. Moreover, the playlists and the annotations included in the Melon Playlist Dataset make it suitable for metric learning and representation learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to announce the release for <a href="https://twitter.com/hashtag/ISMIR?src=hash&amp;ref_src=twsrc%5Etfw">#ISMIR</a> and <a href="https://twitter.com/hashtag/Recsys?src=hash&amp;ref_src=twsrc%5Etfw">#Recsys</a> of Melon Playlist Dataset, including mel-spectrograms for 649,091 tracks and 148,826 playlists. A collaboration between <a href="https://twitter.com/mtg_upf?ref_src=twsrc%5Etfw">@mtg_upf</a> and <a href="https://twitter.com/Team_Kakao?ref_src=twsrc%5Etfw">@Team_Kakao</a>  <br><br>web: <a href="https://t.co/1L0OlZxwlc">https://t.co/1L0OlZxwlc</a><br>ICASSP paper: <a href="https://t.co/1DKDyeRfME">https://t.co/1DKDyeRfME</a></p>&mdash; andres ferraro (@andrebola_) <a href="https://twitter.com/andrebola_/status/1356652686268071937?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. ObjectAug: Object-level Data Augmentation for Semantic Image  Segmentation

Jiawei Zhang, Yanchun Zhang, Xiaowei Xu

- retweets: 225, favorites: 35 (02/03/2021 10:20:10)

- links: [abs](https://arxiv.org/abs/2102.00221) | [pdf](https://arxiv.org/pdf/2102.00221)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Semantic image segmentation aims to obtain object labels with precise boundaries, which usually suffers from overfitting. Recently, various data augmentation strategies like regional dropout and mix strategies have been proposed to address the problem. These strategies have proved to be effective for guiding the model to attend on less discriminative parts. However, current strategies operate at the image level, and objects and the background are coupled. Thus, the boundaries are not well augmented due to the fixed semantic scenario. In this paper, we propose ObjectAug to perform object-level augmentation for semantic image segmentation. ObjectAug first decouples the image into individual objects and the background using the semantic labels. Next, each object is augmented individually with commonly used augmentation methods (e.g., scaling, shifting, and rotation). Then, the black area brought by object augmentation is further restored using image inpainting. Finally, the augmented objects and background are assembled as an augmented image. In this way, the boundaries can be fully explored in the various semantic scenarios. In addition, ObjectAug can support category-aware augmentation that gives various possibilities to objects in each category, and can be easily combined with existing image-level augmentation methods to further boost performance. Comprehensive experiments are conducted on both natural image and medical image datasets. Experiment results demonstrate that our ObjectAug can evidently improve segmentation performance.

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">ObjectAug: Object-level Data Augmentation for Semantic Image Segmentation<a href="https://t.co/anmv1PSvcG">https://t.co/anmv1PSvcG</a> <a href="https://t.co/GYt4agRspz">pic.twitter.com/GYt4agRspz</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1356431244796469250?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Expressive Neural Voice Cloning

Paarth Neekhara, Shehzeen Hussain, Shlomo Dubnov, Farinaz Koushanfar, Julian McAuley

- retweets: 156, favorites: 83 (02/03/2021 10:20:10)

- links: [abs](https://arxiv.org/abs/2102.00151) | [pdf](https://arxiv.org/pdf/2102.00151)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Voice cloning is the task of learning to synthesize the voice of an unseen speaker from a few samples. While current voice cloning methods achieve promising results in Text-to-Speech (TTS) synthesis for a new voice, these approaches lack the ability to control the expressiveness of synthesized audio. In this work, we propose a controllable voice cloning method that allows fine-grained control over various style aspects of the synthesized speech for an unseen speaker. We achieve this by explicitly conditioning the speech synthesis model on a speaker encoding, pitch contour and latent style tokens during training. Through both quantitative and qualitative evaluations, we show that our framework can be used for various expressive voice cloning tasks using only a few transcribed or untranscribed speech samples for a new speaker. These cloning tasks include style transfer from a reference speech, synthesizing speech directly from text, and fine-grained style control by manipulating the style conditioning variables during inference.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Expressive Neural Voice Cloning<br>pdf: <a href="https://t.co/c62JV3T1tM">https://t.co/c62JV3T1tM</a><br>abs: <a href="https://t.co/2xEc0QdtSO">https://t.co/2xEc0QdtSO</a><br>project page: <a href="https://t.co/b5neKCOy2W">https://t.co/b5neKCOy2W</a><br>demo: <a href="https://t.co/32PVuRdvUD">https://t.co/32PVuRdvUD</a> <a href="https://t.co/lgilaKNwsi">pic.twitter.com/lgilaKNwsi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356467679536828416?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Self-Supervised Equivariant Scene Synthesis from Video

Cinjon Resnick, Or Litany, Cosmas Heiß, Hugo Larochelle, Joan Bruna, Kyunghyun Cho

- retweets: 156, favorites: 76 (02/03/2021 10:20:11)

- links: [abs](https://arxiv.org/abs/2102.00863) | [pdf](https://arxiv.org/pdf/2102.00863)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a self-supervised framework to learn scene representations from video that are automatically delineated into background, characters, and their animations. Our method capitalizes on moving characters being equivariant with respect to their transformation across frames and the background being constant with respect to that same transformation. After training, we can manipulate image encodings in real time to create unseen combinations of the delineated components. As far as we know, we are the first method to perform unsupervised extraction and synthesis of interpretable background, character, and animation. We demonstrate results on three datasets: Moving MNIST with backgrounds, 2D video game sprites, and Fashion Modeling.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Self-supervised Equivariant Scene Synthesis from Video&quot; (<a href="https://t.co/KX0MoGP7Ms">https://t.co/KX0MoGP7Ms</a>). In stop motion animation, each character moves one affine step at a time. Can we learn the transformation, the background, and the character encoding simultaneously ... without supervision?</p>&mdash; Cinjon Resnick (@cinjoncin) <a href="https://twitter.com/cinjoncin/status/1356599546307051526?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Supervised Equivariant Scene Synthesis from Video<br>pdf: <a href="https://t.co/tDeZBkgVv8">https://t.co/tDeZBkgVv8</a><br>abs: <a href="https://t.co/u5MnGbxKn9">https://t.co/u5MnGbxKn9</a> <a href="https://t.co/qbF6Cbhnfb">pic.twitter.com/qbF6Cbhnfb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356491385390645249?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. The effect of differential victim crime reporting on predictive policing  systems

Nil-Jana Akpinar, Alexandra Chouldechova

- retweets: 132, favorites: 24 (02/03/2021 10:20:11)

- links: [abs](https://arxiv.org/abs/2102.00128) | [pdf](https://arxiv.org/pdf/2102.00128)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Police departments around the world have been experimenting with forms of place-based data-driven proactive policing for over two decades. Modern incarnations of such systems are commonly known as hot spot predictive policing. These systems predict where future crime is likely to concentrate such that police can allocate patrols to these areas and deter crime before it occurs. Previous research on fairness in predictive policing has concentrated on the feedback loops which occur when models are trained on discovered crime data, but has limited implications for models trained on victim crime reporting data. We demonstrate how differential victim crime reporting rates across geographical areas can lead to outcome disparities in common crime hot spot prediction models. Our analysis is based on a simulation patterned after district-level victimization and crime reporting survey data for Bogot\'a, Colombia. Our results suggest that differential crime reporting rates can lead to a displacement of predicted hotspots from high crime but low reporting areas to high or medium crime and high reporting areas. This may lead to misallocations both in the form of over-policing and under-policing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My first <a href="https://twitter.com/FAccTConference?ref_src=twsrc%5Etfw">@FAccTConference</a> paper with <a href="https://twitter.com/achould?ref_src=twsrc%5Etfw">@achould</a> is online!🙂<br>Link: <a href="https://t.co/wGlVjgTmtr">https://t.co/wGlVjgTmtr</a><br><br>We demonstrate how common crime hot spot prediction models can suffer from spatial bias even if trained entirely on victim crime reporting data 🧵👇 <a href="https://t.co/uUjnN6EeVG">pic.twitter.com/uUjnN6EeVG</a></p>&mdash; Nil-Jana Akpinar (@niljanaakpinar) <a href="https://twitter.com/niljanaakpinar/status/1356632095121166338?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Evaluating Large-Vocabulary Object Detectors: The Devil is in the  Details

Achal Dave, Piotr Dollár, Deva Ramanan, Alexander Kirillov, Ross Girshick

- retweets: 102, favorites: 45 (02/03/2021 10:20:11)

- links: [abs](https://arxiv.org/abs/2102.01066) | [pdf](https://arxiv.org/pdf/2102.01066)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

By design, average precision (AP) for object detection aims to treat all classes independently: AP is computed independently per category and averaged. On the one hand, this is desirable as it treats all classes, rare to frequent, equally. On the other hand, it ignores cross-category confidence calibration, a key property in real-world use cases. Unfortunately, we find that on imbalanced, large-vocabulary datasets, the default implementation of AP is neither category independent, nor does it directly reward properly calibrated detectors. In fact, we show that the default implementation produces a gameable metric, where a simple, nonsensical re-ranking policy can improve AP by a large margin. To address these limitations, we introduce two complementary metrics. First, we present a simple fix to the default AP implementation, ensuring that it is truly independent across categories as originally intended. We benchmark recent advances in large-vocabulary detection and find that many reported gains do not translate to improvements under our new per-class independent evaluation, suggesting recent improvements may arise from difficult to interpret changes to cross-category rankings. Given the importance of reliably benchmarking cross-category rankings, we consider a pooled version of AP (AP-pool) that rewards properly calibrated detectors by directly comparing cross-category rankings. Finally, we revisit classical approaches for calibration and find that explicitly calibrating detectors improves state-of-the-art on AP-pool by 1.7 points.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Rossさん達がAP計算のハック要素を指摘。実装上の都合で画像あたりの検出数に制限があるせいで、どの検出結果を残すかを操作して平均APを上げられる。頻出クラスは信頼度が高くても削除し、逆に信頼度が低いレアクラスを検出結果に含めるような不自然なやり方でAPが上がる<a href="https://t.co/2bbBzm7M1k">https://t.co/2bbBzm7M1k</a> <a href="https://t.co/k6WGe0TKdS">pic.twitter.com/k6WGe0TKdS</a></p>&mdash; Kazuyuki Miyazawa (@kzykmyzw) <a href="https://twitter.com/kzykmyzw/status/1356628424878485508?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Neural 3D Clothes Retargeting from a Single Image

Jae Shin Yoon, Kihwan Kim, Jan Kautz, Hyun Soo Park

- retweets: 83, favorites: 54 (02/03/2021 10:20:11)

- links: [abs](https://arxiv.org/abs/2102.00062) | [pdf](https://arxiv.org/pdf/2102.00062)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we present a method of clothes retargeting; generating the potential poses and deformations of a given 3D clothing template model to fit onto a person in a single RGB image. The problem is fundamentally ill-posed as attaining the ground truth data is impossible, i.e., images of people wearing the different 3D clothing template model at exact same pose. We address this challenge by utilizing large-scale synthetic data generated from physical simulation, allowing us to map 2D dense body pose to 3D clothing deformation. With the simulated data, we propose a semi-supervised learning framework that validates the physical plausibility of the 3D deformation by matching with the prescribed body-to-cloth contact points and clothing silhouette to fit onto the unlabeled real images. A new neural clothes retargeting network (CRNet) is designed to integrate the semi-supervised retargeting task in an end-to-end fashion. In our evaluation, we show that our method can predict the realistic 3D pose and deformation field needed for retargeting clothes models in real-world examples.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural 3D Clothes Retargeting from a Single Image<br>pdf: <a href="https://t.co/wX8yzHFfaB">https://t.co/wX8yzHFfaB</a><br>abs: <a href="https://t.co/amFinRZT5L">https://t.co/amFinRZT5L</a> <a href="https://t.co/vUymIDdFTk">pic.twitter.com/vUymIDdFTk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356492231927099393?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Modeling how social network algorithms can influence opinion  polarization

Henrique F. de Arruda, Felipe M. Cardoso, Guilherme F. de Arruda, Alexis R. Hernández, Luciano da F. Costa, Yamir Moreno

- retweets: 76, favorites: 27 (02/03/2021 10:20:11)

- links: [abs](https://arxiv.org/abs/2102.00099) | [pdf](https://arxiv.org/pdf/2102.00099)
- [cs.SI](https://arxiv.org/list/cs.SI/recent)

Among different aspects of social networks, dynamics have been proposed to simulate how opinions can be transmitted. In this study, we propose a model that simulates the communication in an online social network, in which the posts are created from external information. We considered the nodes and edges of a network as users and their friendship, respectively. A real number is associated with each user representing its opinion. The dynamics starts with a user that has contact with a random opinion, and, according to a given probability function, this individual can post this opinion. This step is henceforth called post transmission. In the next step, called post distribution, another probability function is employed to select the user's friends that could see the post. Post transmission and distribution represent the user and the social network algorithm, respectively. If an individual has contact with a post, its opinion can be attracted or repulsed. Furthermore, individuals that are repulsed can change their friendship through a rewiring. These steps are executed various times until the dynamics converge. Several impressive results were obtained, which include the formation of scenarios of polarization and consensus of opinions. In the case of echo chambers, the possibility of rewiring probability is found to be decisive. However, for particular network topologies, with a well-defined community structure, this effect can also happen. All in all, the results indicate that the post distribution strategy is crucial to mitigate or promote polarization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new preprint, &quot;Modeling how social<br>network algorithms can influence opinion polarization&quot;, is out. <a href="https://t.co/9YcqVryMnf">https://t.co/9YcqVryMnf</a> <a href="https://twitter.com/fmacielcardoso?ref_src=twsrc%5Etfw">@fmacielcardoso</a> <a href="https://twitter.com/GuiFdeArruda?ref_src=twsrc%5Etfw">@GuiFdeArruda</a> <a href="https://twitter.com/er_chechi?ref_src=twsrc%5Etfw">@er_chechi</a> <a href="https://twitter.com/LdaFCosta?ref_src=twsrc%5Etfw">@LdaFCosta</a> <a href="https://twitter.com/cosnet_bifi?ref_src=twsrc%5Etfw">@cosnet_bifi</a></p>&mdash; Henrique F. de Arruda (@hfarruda) <a href="https://twitter.com/hfarruda/status/1356623855926865923?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. High Fidelity Speech Regeneration with Application to Speech Enhancement

Adam Polyak, Lior Wolf, Yossi Adi, Ori Kabeli, Yaniv Taigman

- retweets: 42, favorites: 30 (02/03/2021 10:20:11)

- links: [abs](https://arxiv.org/abs/2102.00429) | [pdf](https://arxiv.org/pdf/2102.00429)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Speech enhancement has seen great improvement in recent years mainly through contributions in denoising, speaker separation, and dereverberation methods that mostly deal with environmental effects on vocal audio. To enhance speech beyond the limitations of the original signal, we take a regeneration approach, in which we recreate the speech from its essence, including the semi-recognized speech, prosody features, and identity. We propose a wav-to-wav generative model for speech that can generate 24khz speech in a real-time manner and which utilizes a compact speech representation, composed of ASR and identity features, to achieve a higher level of intelligibility. Inspired by voice conversion methods, we train to augment the speech characteristics while preserving the identity of the source using an auxiliary identity network. Perceptual acoustic metrics and subjective tests show that the method obtains valuable improvements over recent baselines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High Fidelity Speech Regeneration with Application to Speech Enhancement<br>pdf: <a href="https://t.co/vgkNbTPbtJ">https://t.co/vgkNbTPbtJ</a><br>abs: <a href="https://t.co/FNlnR7emyv">https://t.co/FNlnR7emyv</a><br>project page: <a href="https://t.co/1Bhe8iX6pQ">https://t.co/1Bhe8iX6pQ</a> <a href="https://t.co/7GstbrPCg6">pic.twitter.com/7GstbrPCg6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1356470175541964800?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Beyond the Command: Feminist STS Research and Critical Issues for the  Design of Social Machines

Kelly B. Wagman, Lisa Parks

- retweets: 30, favorites: 28 (02/03/2021 10:20:11)

- links: [abs](https://arxiv.org/abs/2102.00464) | [pdf](https://arxiv.org/pdf/2102.00464)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Machines, from artificially intelligent digital assistants to embodied robots, are becoming more pervasive in everyday life. Drawing on feminist science and technology studies (STS) perspectives, we demonstrate how machine designers are not just crafting neutral objects, but relationships between machines and humans that are entangled in human social issues such as gender and power dynamics. Thus, in order to create a more ethical and just future, the dominant assumptions currently underpinning the design of these human-machine relations must be challenged and reoriented toward relations of justice and inclusivity. This paper contributes the "social machine" as a model for technology designers who seek to recognize the importance, diversity and complexity of the social in their work, and to engage with the agential power of machines. In our model, the social machine is imagined as a potentially equitable relationship partner that has agency and as an "other" that is distinct from, yet related to, humans, objects, and animals. We critically examine and contrast our model with tendencies in robotics that consider robots as tools, human companions, animals or creatures, and/or slaves. In doing so, we demonstrate ingrained dominant assumptions about human-machine relations and reveal the challenges of radical thinking in the social machine design space. Finally, we present two design challenges based on non-anthropomorphic figuration and mutuality, and call for experimentation, unlearning dominant tendencies, and reimagining of sociotechnical futures.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share that my paper with Lisa Parks on how designers &amp; technologists can conceptualize human-machine relations in a way that is equitable/just has been accepted to <a href="https://twitter.com/hashtag/CSCW21?src=hash&amp;ref_src=twsrc%5Etfw">#CSCW21</a>! Really poured my heart and soul into this one... <a href="https://t.co/P9TyGBbA8x">https://t.co/P9TyGBbA8x</a> <a href="https://t.co/bksngozXgc">pic.twitter.com/bksngozXgc</a></p>&mdash; Kelly Wagman (@kellybwagman) <a href="https://twitter.com/kellybwagman/status/1356672782487158785?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. Can Machine Learning Help in Solving Cargo Capacity Management Booking  Control Problems?

Justin Dumouchelle, Emma Frejinger, Andrea Lodi

- retweets: 46, favorites: 8 (02/03/2021 10:20:12)

- links: [abs](https://arxiv.org/abs/2102.00092) | [pdf](https://arxiv.org/pdf/2102.00092)
- [math.OC](https://arxiv.org/list/math.OC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Revenue management is important for carriers (e.g., airlines and railroads). In this paper, we focus on cargo capacity management which has received less attention in the literature than its passenger counterpart. More precisely, we focus on the problem of controlling booking accept/reject decisions: Given a limited capacity, accept a booking request or reject it to reserve capacity for future bookings with potentially higher revenue. We formulate the problem as a finite-horizon stochastic dynamic program. The cost of fulfilling the accepted bookings, incurred at the end of the horizon, depends on the packing and routing of the cargo. This is a computationally challenging aspect as the latter are solutions to an operational decision-making problem, in our application a vehicle routing problem (VRP). Seeking a balance between online and offline computation, we propose to train a predictor of the solution costs to the VRPs using supervised learning. In turn, we use the predictions online in approximate dynamic programming and reinforcement learning algorithms to solve the booking control problem. We compare the results to an existing approach in the literature and show that we are able to obtain control policies that provide increased profit at a reduced evaluation time. This is achieved thanks to accurate approximation of the operational costs and negligible computing time in comparison to solving the VRPs.




# 20. Machine Translationese: Effects of Algorithmic Bias on Linguistic  Complexity in Machine Translation

Eva Vanmassenhove, Dimitar Shterionov, Matthew Gwilliam

- retweets: 12, favorites: 42 (02/03/2021 10:20:12)

- links: [abs](https://arxiv.org/abs/2102.00287) | [pdf](https://arxiv.org/pdf/2102.00287)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Recent studies in the field of Machine Translation (MT) and Natural Language Processing (NLP) have shown that existing models amplify biases observed in the training data. The amplification of biases in language technology has mainly been examined with respect to specific phenomena, such as gender bias. In this work, we go beyond the study of gender in MT and investigate how bias amplification might affect language in a broader sense. We hypothesize that the 'algorithmic bias', i.e. an exacerbation of frequently observed patterns in combination with a loss of less frequent ones, not only exacerbates societal biases present in current datasets but could also lead to an artificially impoverished language: 'machine translationese'. We assess the linguistic richness (on a lexical and morphological level) of translations created by different data-driven MT paradigms - phrase-based statistical (PB-SMT) and neural MT (NMT). Our experiments show that there is a loss of lexical and morphological richness in the translations produced by all investigated MT paradigms for two language pairs (EN<=>FR and EN<=>ES).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our latest paper on algorithmic bias, machine translationese and lexical and morphological richness is now available (<a href="https://t.co/9lzCBDQiuw">https://t.co/9lzCBDQiuw</a>) and will soon be presented at <a href="https://twitter.com/eaclmeeting?ref_src=twsrc%5Etfw">@eaclmeeting</a>. <br><br>Co-authored with<a href="https://twitter.com/DShterionov?ref_src=twsrc%5Etfw">@DShterionov</a> from <a href="https://twitter.com/TilburgU?ref_src=twsrc%5Etfw">@TilburgU</a> and Matthew Gwilliam from <a href="https://twitter.com/UofMaryland?ref_src=twsrc%5Etfw">@UofMaryland</a>.</p>&mdash; Eva Vanmassenhove (@Evanmassenhove) <a href="https://twitter.com/Evanmassenhove/status/1356515934870134784?ref_src=twsrc%5Etfw">February 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



