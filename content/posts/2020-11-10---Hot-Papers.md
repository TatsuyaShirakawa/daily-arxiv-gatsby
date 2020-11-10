---
title: Hot Papers 2020-11-10
date: 2020-11-11T08:57:47.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-10"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-10"
socialImage: "/media/flying-marine.jpg"

---

# 1. Graph Kernels: State-of-the-Art and Future Challenges

Karsten Borgwardt, Elisabetta Ghisu, Felipe Llinares-López, Leslie O'Bray, Bastian Rieck

- retweets: 5748, favorites: 423 (11/11/2020 08:57:47)

- links: [abs](https://arxiv.org/abs/2011.03854) | [pdf](https://arxiv.org/pdf/2011.03854)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Graph-structured data are an integral part of many application domains, including chemoinformatics, computational biology, neuroimaging, and social network analysis. Over the last fifteen years, numerous graph kernels, i.e. kernel functions between graphs, have been proposed to solve the problem of assessing the similarity between graphs, thereby making it possible to perform predictions in both classification and regression settings. This manuscript provides a review of existing graph kernels, their applications, software plus data resources, and an empirical comparison of state-of-the-art graph kernels.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited by the current boom in learning on graphs, we have reviewed the foundations and trends in <a href="https://twitter.com/hashtag/graphkernel?src=hash&amp;ref_src=twsrc%5Etfw">#graphkernel</a> research. We hope that it will be a reference and starting point for lots of future work in this domain! <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/Kernels?src=hash&amp;ref_src=twsrc%5Etfw">#Kernels</a> <a href="https://t.co/8vwiwWHbRM">https://t.co/8vwiwWHbRM</a> <a href="https://t.co/Ssxfo5MfR5">pic.twitter.com/Ssxfo5MfR5</a></p>&mdash; Karsten Borgwardt (@kmborgwardt) <a href="https://twitter.com/kmborgwardt/status/1326171326357319680?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Graph-structured data will continue to enable some of the most interesting applications in the field of machine learning -- ranging from social network analysis to neuroimaging.<br><br>Check out this recent paper reviewing graph kernels and future challenges.<a href="https://t.co/XDjV5927lb">https://t.co/XDjV5927lb</a> <a href="https://t.co/oUw2w48hb1">pic.twitter.com/oUw2w48hb1</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1326182523588780033?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Wave-Tacotron: Spectrogram-free end-to-end text-to-speech synthesis

Ron J. Weiss, RJ Skerry-Ryan, Eric Battenberg, Soroosh Mariooryad, Diederik P. Kingma

- retweets: 888, favorites: 129 (11/11/2020 08:57:47)

- links: [abs](https://arxiv.org/abs/2011.03568) | [pdf](https://arxiv.org/pdf/2011.03568)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We describe a sequence-to-sequence neural network which can directly generate speech waveforms from text inputs. The architecture extends the Tacotron model by incorporating a normalizing flow into the autoregressive decoder loop. Output waveforms are modeled as a sequence of non-overlapping fixed-length frames, each one containing hundreds of samples. The interdependencies of waveform samples within each frame are modeled using the normalizing flow, enabling parallel training and synthesis. Longer-term dependencies are handled autoregressively by conditioning each flow on preceding frames. This model can be optimized directly with maximum likelihood, without using intermediate, hand-designed features nor additional loss terms. Contemporary state-of-the-art text-to-speech (TTS) systems use a cascade of separately learned models: one (such as Tacotron) which generates intermediate features (such as spectrograms) from text, followed by a vocoder (such as WaveRNN) which generates waveform samples from the intermediate features. The proposed system, in contrast, does not use a fixed intermediate representation, and learns all parameters end-to-end. Experiments show that the proposed model generates speech with quality approaching a state-of-the-art neural TTS system, with significantly improved generation speed.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wave-Tacotron: Spectrogram-free end-to-end text-to-speech synthesis<br>pdf: <a href="https://t.co/gx4xC0fWrz">https://t.co/gx4xC0fWrz</a><br>abs: <a href="https://t.co/zWz2ZIPVOR">https://t.co/zWz2ZIPVOR</a><br>project page: <a href="https://t.co/CEexNbhUs8">https://t.co/CEexNbhUs8</a> <a href="https://t.co/30AhWoVsDo">pic.twitter.com/30AhWoVsDo</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1326017294284316674?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Text-to-Image Generation Grounded by Fine-Grained User Attention

Jing Yu Koh, Jason Baldridge, Honglak Lee, Yinfei Yang

- retweets: 522, favorites: 226 (11/11/2020 08:57:47)

- links: [abs](https://arxiv.org/abs/2011.03775) | [pdf](https://arxiv.org/pdf/2011.03775)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Localized Narratives is a dataset with detailed natural language descriptions of images paired with mouse traces that provide a sparse, fine-grained visual grounding for phrases. We propose TReCS, a sequential model that exploits this grounding to generate images. TReCS uses descriptions to retrieve segmentation masks and predict object labels aligned with mouse traces. These alignments are used to select and position masks to generate a fully covered segmentation canvas; the final image is produced by a segmentation-to-image generator using this canvas. This multi-step, retrieval-based approach outperforms existing direct text-to-image generation models on both automatic metrics and human evaluations: overall, its generated images are more photo-realistic and better match descriptions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">These text-to-image results look pretty cool.<a href="https://t.co/CPkC56bxst">https://t.co/CPkC56bxst</a> <a href="https://t.co/TbAaYZ9pwl">https://t.co/TbAaYZ9pwl</a> <a href="https://t.co/GNShK9NSmp">pic.twitter.com/GNShK9NSmp</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1326162876080418822?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our work on user attention grounded text-to-image generation: <a href="https://t.co/aEOXwhkDWP">https://t.co/aEOXwhkDWP</a><br><br>We generate images from mouse pointer traces and natural language, allowing users to dictate the layout of objects in a scene.<br><br>With <a href="https://twitter.com/jasonbaldridge?ref_src=twsrc%5Etfw">@jasonbaldridge</a>, <a href="https://twitter.com/honglaklee?ref_src=twsrc%5Etfw">@honglaklee</a>, and <a href="https://twitter.com/yinfeiy?ref_src=twsrc%5Etfw">@yinfeiy</a> <a href="https://t.co/7YUcC9n8fI">pic.twitter.com/7YUcC9n8fI</a></p>&mdash; Jing Yu Koh (@kohjingyu) <a href="https://twitter.com/kohjingyu/status/1326149554039787521?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Text-to-Image Generation Grounded by Fine-Grained User Attention<br>pdf: <a href="https://t.co/wkgZL4V9Gf">https://t.co/wkgZL4V9Gf</a><br>abs: <a href="https://t.co/ftqgcOYGMn">https://t.co/ftqgcOYGMn</a> <a href="https://t.co/3kaxsm4ZnA">pic.twitter.com/3kaxsm4ZnA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1326027910252294144?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Long Range Arena: A Benchmark for Efficient Transformers

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler

- retweets: 239, favorites: 67 (11/11/2020 08:57:48)

- links: [abs](https://arxiv.org/abs/2011.04006) | [pdf](https://arxiv.org/pdf/2011.04006)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

Transformers do not scale very well to long sequence lengths largely because of quadratic self-attention complexity. In the recent months, a wide spectrum of efficient, fast Transformers have been proposed to tackle this problem, more often than not claiming superior or comparable model quality to vanilla Transformer models. To this date, there is no well-established consensus on how to evaluate this class of models. Moreover, inconsistent benchmarking on a wide spectrum of tasks and datasets makes it difficult to assess relative model quality amongst many models. This paper proposes a systematic and unified benchmark, LRA, specifically focused on evaluating model quality under long-context scenarios. Our benchmark is a suite of tasks consisting of sequences ranging from $1K$ to $16K$ tokens, encompassing a wide range of data types and modalities such as text, natural, synthetic images, and mathematical expressions requiring similarity, structural, and visual-spatial reasoning. We systematically evaluate ten well-established long-range Transformer models (Reformers, Linformers, Linear Transformers, Sinkhorn Transformers, Performers, Synthesizers, Sparse Transformers, and Longformers) on our newly proposed benchmark suite. LRA paves the way towards better understanding this class of efficient Transformer models, facilitates more research in this direction, and presents new challenging tasks to tackle. Our benchmark code will be released at https://github.com/google-research/long-range-arena.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">As a companion to our recent efficient Transformer survey, we designed &quot;Long Range Arena&quot; a new challenging benchmark to help understand and analyze trade-offs between recent efficient Transformer models. Check out our paper at <a href="https://t.co/hnMkari1Oa">https://t.co/hnMkari1Oa</a>. <a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> <a href="https://twitter.com/DeepMind?ref_src=twsrc%5Etfw">@DeepMind</a> <a href="https://t.co/VHXjadES71">pic.twitter.com/VHXjadES71</a></p>&mdash; Yi Tay (@ytay017) <a href="https://twitter.com/ytay017/status/1326281833282072576?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Single and Multi-Agent Deep Reinforcement Learning for AI-Enabled  Wireless Networks: A Tutorial

Amal Feriani, Ekram Hossain

- retweets: 254, favorites: 21 (11/11/2020 08:57:48)

- links: [abs](https://arxiv.org/abs/2011.03615) | [pdf](https://arxiv.org/pdf/2011.03615)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NI](https://arxiv.org/list/cs.NI/recent)

Deep Reinforcement Learning (DRL) has recently witnessed significant advances that have led to multiple successes in solving sequential decision-making problems in various domains, particularly in wireless communications. The future sixth-generation (6G) networks are expected to provide scalable, low-latency, ultra-reliable services empowered by the application of data-driven Artificial Intelligence (AI). The key enabling technologies of future 6G networks, such as intelligent meta-surfaces, aerial networks, and AI at the edge, involve more than one agent which motivates the importance of multi-agent learning techniques. Furthermore, cooperation is central to establishing self-organizing, self-sustaining, and decentralized networks. In this context, this tutorial focuses on the role of DRL with an emphasis on deep Multi-Agent Reinforcement Learning (MARL) for AI-enabled 6G networks. The first part of this paper will present a clear overview of the mathematical frameworks for single-agent RL and MARL. The main idea of this work is to motivate the application of RL beyond the model-free perspective which was extensively adopted in recent years. Thus, we provide a selective description of RL algorithms such as Model-Based RL (MBRL) and cooperative MARL and we highlight their potential applications in 6G wireless networks. Finally, we overview the state-of-the-art of MARL in fields such as Mobile Edge Computing (MEC), Unmanned Aerial Vehicles (UAV) networks, and cell-free massive MIMO, and identify promising future research directions. We expect this tutorial to stimulate more research endeavors to build scalable and decentralized systems based on MARL.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Single and Multi-Agent Deep Reinforcement Learning for AI-Enabled Wireless Networks. <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/Cloud?src=hash&amp;ref_src=twsrc%5Etfw">#Cloud</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/ReinforcementLearning?src=hash&amp;ref_src=twsrc%5Etfw">#ReinforcementLearning</a><a href="https://t.co/u9pOI0GfEz">https://t.co/u9pOI0GfEz</a> <a href="https://t.co/YJsHbpJTBG">pic.twitter.com/YJsHbpJTBG</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1326292713952735232?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Closing the Generalization Gap in One-Shot Object Detection

Claudio Michaelis, Matthias Bethge, Alexander S. Ecker

- retweets: 158, favorites: 69 (11/11/2020 08:57:48)

- links: [abs](https://arxiv.org/abs/2011.04267) | [pdf](https://arxiv.org/pdf/2011.04267)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Despite substantial progress in object detection and few-shot learning, detecting objects based on a single example - one-shot object detection - remains a challenge: trained models exhibit a substantial generalization gap, where object categories used during training are detected much more reliably than novel ones. Here we show that this generalization gap can be nearly closed by increasing the number of object categories used during training. Our results show that the models switch from memorizing individual categories to learning object similarity over the category distribution, enabling strong generalization at test time. Importantly, in this regime standard methods to improve object detection models like stronger backbones or longer training schedules also benefit novel categories, which was not the case for smaller datasets like COCO. Our results suggest that the key to strong few-shot detection models may not lie in sophisticated metric learning approaches, but instead in scaling the number of categories. Future data annotation efforts should therefore focus on wider datasets and annotate a larger number of categories rather than gathering more images or instances per category.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">More classes is all you need for one-shot object detection. <br><br>It turns out we can almost close the gap between the detection of known and novel objects simply by increasing the number of categories. <a href="https://t.co/jpCEGb3sOA">https://t.co/jpCEGb3sOA</a> <a href="https://twitter.com/clmich?ref_src=twsrc%5Etfw">@clmich</a> <a href="https://twitter.com/alxecker?ref_src=twsrc%5Etfw">@alxecker</a> <a href="https://twitter.com/MatthiasBethge?ref_src=twsrc%5Etfw">@MatthiasBethge</a> <br>[1/7] <a href="https://t.co/8mlHhSBI5n">pic.twitter.com/8mlHhSBI5n</a></p>&mdash; Claudio Michaelis (@clmich) <a href="https://twitter.com/clmich/status/1326094157954297857?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. NeuralSim: Augmenting Differentiable Simulators with Neural Networks

Eric Heiden, David Millard, Erwin Coumans, Yizhou Sheng, Gaurav S. Sukhatme

- retweets: 72, favorites: 127 (11/11/2020 08:57:48)

- links: [abs](https://arxiv.org/abs/2011.04217) | [pdf](https://arxiv.org/pdf/2011.04217)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

Differentiable simulators provide an avenue for closing the sim-to-real gap by enabling the use of efficient, gradient-based optimization algorithms to find the simulation parameters that best fit the observed sensor readings. Nonetheless, these analytical models can only predict the dynamical behavior of systems for which they have been designed. In this work, we study the augmentation of a novel differentiable rigid-body physics engine via neural networks that is able to learn nonlinear relationships between dynamic quantities and can thus learn effects not accounted for in traditional simulators.Such augmentations require less data to train and generalize better compared to entirely data-driven models. Through extensive experiments, we demonstrate the ability of our hybrid simulator to learn complex dynamics involving frictional contacts from real data, as well as match known models of viscous friction, and present an approach for automatically discovering useful augmentations. We show that, besides benefiting dynamics modeling, inserting neural networks can accelerate model-based control architectures. We observe a ten-fold speed-up when replacing the QP solver inside a model-predictive gait controller for quadruped robots with a neural network, allowing us to significantly improve control delays as we demonstrate in real-hardware experiments.   We publish code, additional results and videos from our experiments on our project webpage at https://sites.google.com/usc.edu/neuralsim.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our ICRA2021 submission &quot;NeuralSim: Augmenting Differentiable Simulators with Neural Networks&quot; is public now. Neural networks can mimic the QP solver in whole body locomotion very well. <a href="https://t.co/rEScdk1uuA">https://t.co/rEScdk1uuA</a> <a href="https://t.co/CiYj8MO7qk">https://t.co/CiYj8MO7qk</a> and source code: <a href="https://t.co/KgjzC4vaH4">https://t.co/KgjzC4vaH4</a></p>&mdash; Erwin Coumans (@erwincoumans) <a href="https://twitter.com/erwincoumans/status/1325997556510334977?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeuralSim: Augmenting Differentiable Simulators<br>with Neural Networks<br>pdf: <a href="https://t.co/gpoIVriOi7">https://t.co/gpoIVriOi7</a><br>abs: <a href="https://t.co/Of44VM7LN8">https://t.co/Of44VM7LN8</a><br>project page: <a href="https://t.co/7pcXeXEVJG">https://t.co/7pcXeXEVJG</a> <a href="https://t.co/OOhvi1erF7">pic.twitter.com/OOhvi1erF7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1326011308840525830?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeuralSim: Augmenting Differentiable Simulators with Neural Networks<a href="https://t.co/3SxcrwMkBT">https://t.co/3SxcrwMkBT</a> <a href="https://t.co/bDgx0r7qIF">pic.twitter.com/bDgx0r7qIF</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1325994331749036032?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Fairness Under Partial Compliance

Jessica Dai, Sina Fazelpour, Zachary C. Lipton

- retweets: 48, favorites: 77 (11/11/2020 08:57:49)

- links: [abs](https://arxiv.org/abs/2011.03654) | [pdf](https://arxiv.org/pdf/2011.03654)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Typically, fair machine learning research focuses on a single decisionmaker and assumes that the underlying population is stationary. However, many of the critical domains motivating this work are characterized by competitive marketplaces with many decisionmakers. Realistically, we might expect only a subset of them to adopt any non-compulsory fairness-conscious policy, a situation that political philosophers call partial compliance. This possibility raises important questions: how does the strategic behavior of decision subjects in partial compliance settings affect the allocation outcomes? If k% of employers were to voluntarily adopt a fairness-promoting intervention, should we expect k% progress (in aggregate) towards the benefits of universal adoption, or will the dynamics of partial compliance wash out the hoped-for benefits? How might adopting a global (versus local) perspective impact the conclusions of an auditor? In this paper, we propose a simple model of an employment market, leveraging simulation as a tool to explore the impact of both interaction effects and incentive effects on outcomes and auditing metrics. Our key findings are that at equilibrium: (1) partial compliance (k% of employers) can result in far less than proportional (k%) progress towards the full compliance outcomes; (2) the gap is more severe when fair employers match global (vs local) statistics; (3) choices of local vs global statistics can paint dramatically different pictures of the performance vis-a-vis fairness desiderata of compliant versus non-compliant employers; and (4) partial compliance to local parity measures can induce extreme segregation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper &quot;Fair Machine Learning under Partial Compliance&quot; by <a href="https://twitter.com/acmi_lab?ref_src=twsrc%5Etfw">@acmi_lab</a> ***Visiting Undergrad*** <a href="https://twitter.com/jessicadai_?ref_src=twsrc%5Etfw">@jessicadai_</a> shows why we cannot continue to focus myopically on a decision-maker. <br><br>(w postdoc <a href="https://twitter.com/sinafazelpour?ref_src=twsrc%5Etfw">@sinafazelpour</a> &amp; <a href="https://twitter.com/zacharylipton?ref_src=twsrc%5Etfw">@zacharylipton</a>)<a href="https://t.co/mSt5G4sPls">https://t.co/mSt5G4sPls</a></p>&mdash; ACMI Lab (CMU) (@acmi_lab) <a href="https://twitter.com/acmi_lab/status/1326026290672103424?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Super excited to share *phenomenal undergrad <a href="https://twitter.com/jessicadai_?ref_src=twsrc%5Etfw">@jessicadai_</a>*&#39;s first paper, integrating the philosophy of partial compliance &amp; local justice w fair ML. Through simulation studies, we shed light on settings where a subset of employers adopt fairness measures<a href="https://t.co/nOjGXafVuc">https://t.co/nOjGXafVuc</a> <a href="https://t.co/DbaRMpKNUX">https://t.co/DbaRMpKNUX</a></p>&mdash; Zachary Lipton (@zacharylipton) <a href="https://twitter.com/zacharylipton/status/1326028325089898496?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Look Before You Leap: Trusted User Interfaces for the Immersive Web

Diane Hosfelt, Jessica Outlaw, Tyesha Snow, Sara Carbonneau

- retweets: 90, favorites: 30 (11/11/2020 08:57:49)

- links: [abs](https://arxiv.org/abs/2011.03570) | [pdf](https://arxiv.org/pdf/2011.03570)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

Part of what makes the web successful is that anyone can publish content and browsers maintain certain safety guarantees. For example, it's safe to travel between links and make other trust decisions on the web because users can always identify the location they are at. If we want virtual and augmented reality to be successful, we need that same safety. On the traditional, two-dimensional (2D) web, this user interface (UI) is provided by the browser bars and borders (also known as the chrome). However, the immersive, three-dimensional (3D) web has no concept of a browser chrome, preventing routine user inspection of URLs. In this paper, we discuss the unique challenges that fully immersive head-worn computing devices provide to this model, evaluate three different strategies for trusted immersive UI, and make specific recommendations to increase user safety and reduce the risks of spoofing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to announce the release of the last bit of <a href="https://twitter.com/mozillareality?ref_src=twsrc%5Etfw">@mozillareality</a> work I did--Look Before You Leap: Trusted User Interfaces For the Immersive Web<a href="https://t.co/grU8rwre86">https://t.co/grU8rwre86</a></p>&mdash; ddh (@avadacatavra) <a href="https://twitter.com/avadacatavra/status/1326241598099828742?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Reward Conditioned Neural Movement Primitives for Population Based  Variational Policy Optimization

M.Tuluhan Akbulut, Utku Bozdogan, Ahmet Tekden, Emre Ugur

- retweets: 49, favorites: 18 (11/11/2020 08:57:49)

- links: [abs](https://arxiv.org/abs/2011.04282) | [pdf](https://arxiv.org/pdf/2011.04282)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The aim of this paper is to study the reward based policy exploration problem in a supervised learning approach and enable robots to form complex movement trajectories in challenging reward settings and search spaces. For this, the experience of the robot, which can be bootstrapped from demonstrated trajectories, is used to train a novel Neural Processes-based deep network that samples from its latent space and generates the required trajectories given desired rewards. Our framework can generate progressively improved trajectories by sampling them from high reward landscapes, increasing the reward gradually. Variational inference is used to create a stochastic latent space to sample varying trajectories in generating population of trajectories given target rewards. We benefit from Evolutionary Strategies and propose a novel crossover operation, which is applied in the self-organized latent space of the individual policies, allowing blending of the individuals that might address different factors in the reward function. Using a number of tasks that require sequential reaching to multiple points or passing through gaps between objects, we showed that our method provides stable learning progress and significant sample efficiency compared to a number of state-of-the-art robotic reinforcement learning methods. Finally, we show the real-world suitability of our method through real robot execution involving obstacle avoidance.




# 11. Pathwise Conditioning of Gaussian Processes

James T. Wilson, Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth

- retweets: 30, favorites: 31 (11/11/2020 08:57:49)

- links: [abs](https://arxiv.org/abs/2011.04026) | [pdf](https://arxiv.org/pdf/2011.04026)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.ST](https://arxiv.org/list/math.ST/recent)

As Gaussian processes are integrated into increasingly complex problem settings, analytic solutions to quantities of interest become scarcer and scarcer. Monte Carlo methods act as a convenient bridge for connecting intractable mathematical expressions with actionable estimates via sampling. Conventional approaches for simulating Gaussian process posteriors view samples as vectors drawn from marginal distributions over process values at a finite number of input location. This distribution-based characterization leads to generative strategies that scale cubically in the size of the desired random vector. These methods are, therefore, prohibitively expensive in cases where high-dimensional vectors - let alone continuous functions - are required. In this work, we investigate a different line of reasoning. Rather than focusing on distributions, we articulate Gaussian conditionals at the level of random variables. We show how this pathwise interpretation of conditioning gives rise to a general family of approximations that lend themselves to fast sampling from Gaussian process posteriors. We analyze these methods, along with the approximation errors they introduce, from first principles. We then complement this theory, by exploring the practical ramifications of pathwise conditioning in a various applied settings.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pathwise Conditioning of Gaussian Processes<br><br>We study implications of the pathwise formula (f|y)(.) = f(.) + K_{(.)x} K_{xx}^{-1} (y - f(x)) on Gaussian processes, following our outstanding-paper-honorable-mention-winning ICML paper. Check it out!<a href="https://t.co/gCtmWf3KPJ">https://t.co/gCtmWf3KPJ</a><a href="https://twitter.com/mpd37?ref_src=twsrc%5Etfw">@mpd37</a> <a href="https://t.co/pgmuCTcpEA">pic.twitter.com/pgmuCTcpEA</a></p>&mdash; Alexander Terenin (@avt_im) <a href="https://twitter.com/avt_im/status/1326125319078686721?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Exploring the limits of Concurrency in ML Training on Google TPUs

Sameer Kumar, James Bradbury, Cliff Young, Yu Emma Wang, Anselm Levskaya, Blake Hechtman, Dehao Chen, HyoukJoong Lee, Mehmet Deveci, Naveen Kumar, Pankaj Kanwar, Shibo Wang, Skye Wanderman-Milne, Steve Lacy, Tao Wang, Tayo Oguntebi, Yazhou Zu, Yuanzhong Xu, Andy Swing

- retweets: 12, favorites: 46 (11/11/2020 08:57:49)

- links: [abs](https://arxiv.org/abs/2011.03641) | [pdf](https://arxiv.org/pdf/2011.03641)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Recent results in language understanding using neural networks have required training hardware of unprecedentedscale, with thousands of chips cooperating on a single training run. This paper presents techniques to scaleML models on the Google TPU Multipod, a mesh with 4096 TPU-v3 chips. We discuss model parallelism toovercome scaling limitations from the fixed batch size in data parallelism, communication/collective optimizations,distributed evaluation of training metrics, and host input processing scaling optimizations. These techniques aredemonstrated in both the TensorFlow and JAX programming frameworks. We also present performance resultsfrom the recent Google submission to the MLPerf-v0.7 benchmark contest, achieving record training times from16 to 28 seconds in four MLPerf models on the Google TPU-v3 Multipod machine.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Exploring the limits of Concurrency in ML Training on Google TPUs<br><br>- Presents methods to scale models on the TPU <br>Multipod, a mesh with 4096 TPU-v3 chips, on Tensorflow and JAX.<br><br>- Achieves record training times from 16 to 28 seconds in four MLPerf models<a href="https://t.co/mWEaHOqWsT">https://t.co/mWEaHOqWsT</a> <a href="https://t.co/MrHOLmhZ4j">pic.twitter.com/MrHOLmhZ4j</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1325992949692329984?ref_src=twsrc%5Etfw">November 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



