---
title: Hot Papers 2020-10-20
date: 2020-10-21T09:32:38.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-20"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-20"
socialImage: "/media/flying-marine.jpg"

---

# 1. Interpretable Machine Learning -- A Brief History, State-of-the-Art and  Challenges

Christoph Molnar, Giuseppe Casalicchio, Bernd Bischl

- retweets: 20484, favorites: 0 (10/21/2020 09:32:38)

- links: [abs](https://arxiv.org/abs/2010.09337) | [pdf](https://arxiv.org/pdf/2010.09337)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a brief history of the field of interpretable machine learning (IML), give an overview of state-of-the-art interpretation methods, and discuss challenges. Research in IML has boomed in recent years. As young as the field is, it has over 200 years old roots in regression modeling and rule-based machine learning, starting in the 1960s. Recently, many new IML methods have been proposed, many of them model-agnostic, but also interpretation techniques specific to deep learning and tree-based ensembles. IML methods either directly analyze model components, study sensitivity to input perturbations, or analyze local or global surrogate approximations of the ML model. The field approaches a state of readiness and stability, with many methods not only proposed in research, but also implemented in open-source software. But many important challenges remain for IML, such as dealing with dependent features, causal interpretation, and uncertainty estimation, which need to be resolved for its successful application to scientific problems. A further challenge is a missing rigorous definition of interpretability, which is accepted by the community. To address the challenges and advance the field, we urge to recall our roots of interpretable, data-driven modeling in statistics and (rule-based) ML, but also to consider other areas such as sensitivity analysis, causal inference, and the social sciences.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We have a new paper on arxiv 🎉🎉<br>Interpretable Machine Learning - A Brief History, State-of-the-Art and Challenges.<br><br>Best to read in a comfortable chair with a cup of coffee/tea. It&#39;s an extended abstract to a keynote I gave at the ECML XKDD workshop.   <a href="https://t.co/euATIkAetK">https://t.co/euATIkAetK</a></p>&mdash; Christoph Molnar (@ChristophMolnar) <a href="https://twitter.com/ChristophMolnar/status/1318445578943168512?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems

Robert David, Jared Duke, Advait Jain, Vijay Janapa Reddi, Nat Jeffries, Jian Li, Nick Kreeger, Ian Nappier, Meghna Natraj, Shlomi Regev, Rocky Rhodes, Tiezhen Wang, Pete Warden

- retweets: 2197, favorites: 180 (10/21/2020 09:32:38)

- links: [abs](https://arxiv.org/abs/2010.08678) | [pdf](https://arxiv.org/pdf/2010.08678)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Deep learning inference on embedded devices is a burgeoning field with myriad applications because tiny embedded devices are omnipresent. But we must overcome major challenges before we can benefit from this opportunity. Embedded processors are severely resource constrained. Their nearest mobile counterparts exhibit at least a 100---1,000x difference in compute capability, memory availability, and power consumption. As a result, the machine-learning (ML) models and associated ML inference framework must not only execute efficiently but also operate in a few kilobytes of memory. Also, the embedded devices' ecosystem is heavily fragmented. To maximize efficiency, system vendors often omit many features that commonly appear in mainstream systems, including dynamic memory allocation and virtual memory, that allow for cross-platform interoperability. The hardware comes in many flavors (e.g., instruction-set architecture and FPU support, or lack thereof). We introduce TensorFlow Lite Micro (TF Micro), an open-source ML inference framework for running deep-learning models on embedded systems. TF Micro tackles the efficiency requirements imposed by embedded-system resource constraints and the fragmentation challenges that make cross-platform interoperability nearly impossible. The framework adopts a unique interpreter-based approach that provides flexibility while overcoming these challenges. This paper explains the design decisions behind TF Micro and describes its implementation details. Also, we present an evaluation to demonstrate its low resource requirement and minimal run-time performance overhead.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Thanks to an amazing team of authors, we&#39;ve just posted the official <a href="https://twitter.com/TensorFlow?ref_src=twsrc%5Etfw">@TensorFlow</a> Lite Micro paper on Arxiv: <a href="https://t.co/sqxoHPyd6q">https://t.co/sqxoHPyd6q</a><br>Lots of juicy details about the design and tradeoffs involved, what worked and what didn&#39;t!</p>&mdash; Pete Warden (@petewarden) <a href="https://twitter.com/petewarden/status/1318361349500461056?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Enabling Fast Differentially Private SGD via Just-in-Time Compilation  and Vectorization

Pranav Subramani, Nicholas Vadivelu, Gautam Kamath

- retweets: 773, favorites: 147 (10/21/2020 09:32:38)

- links: [abs](https://arxiv.org/abs/2010.09063) | [pdf](https://arxiv.org/pdf/2010.09063)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

A common pain point in differentially private machine learning is the significant runtime overhead incurred when executing Differentially Private Stochastic Gradient Descent (DPSGD), which may be as large as two orders of magnitude. We thoroughly demonstrate that by exploiting powerful language primitives, including vectorization, just-in-time compilation, and static graph optimization, one can dramatically reduce these overheads, in many cases nearly matching the best non-private running times. These gains are realized in two frameworks: JAX and TensorFlow. JAX provides rich support for these primitives as core features of the language through the XLA compiler. We also rebuild core parts of TensorFlow Privacy, integrating features from TensorFlow 2 as well as XLA compilation, granting significant memory and runtime improvements over the current release version. These approaches allow us to achieve up to 50x speedups in comparison to the best alternatives. Our code is available at https://github.com/TheSalon/fast-dpsgd.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Feeling blue that differentially private machine learning is slow?😢<br>Answer is simple: learn JAX! 😮<br>50x speedups vs TF Privacy &amp; Opacus! 🏎️<br><br>Project carried by <a href="https://twitter.com/PranavSubramani?ref_src=twsrc%5Etfw">@PranavSubramani</a> and <a href="https://twitter.com/nicvadivelu?ref_src=twsrc%5Etfw">@nicvadivelu</a>.<br>📝Paper: <a href="https://t.co/9Dxb8pdYRb">https://t.co/9Dxb8pdYRb</a><br>💻Code: <a href="https://t.co/P7dZgECVop">https://t.co/P7dZgECVop</a><br>🧵Thread ⬇️ 1/8 <a href="https://t.co/F65Hu9ssxZ">pic.twitter.com/F65Hu9ssxZ</a></p>&mdash; Gautam Kamath (@thegautamkamath) <a href="https://twitter.com/thegautamkamath/status/1318596827948535808?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to announce our work on enabling fast DPSGD is up on arxiv now <a href="https://t.co/pPEz5uvNfx">https://t.co/pPEz5uvNfx</a><br><br>Joint work with <a href="https://twitter.com/nicvadivelu?ref_src=twsrc%5Etfw">@nicvadivelu</a> and <a href="https://twitter.com/thegautamkamath?ref_src=twsrc%5Etfw">@thegautamkamath</a>.<br><br>The paper is focused on getting DPSGD to run fast for Machine Learning models in particular. 1/n</p>&mdash; Pranav Subramani (@PranavSubramani) <a href="https://twitter.com/PranavSubramani/status/1318356534225612801?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Against Scale: Provocations and Resistances to Scale Thinking

Alex Hanna, Tina M. Park

- retweets: 639, favorites: 168 (10/21/2020 09:32:38)

- links: [abs](https://arxiv.org/abs/2010.08850) | [pdf](https://arxiv.org/pdf/2010.08850)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

At the heart of what drives the bulk of innovation and activity in Silicon Valley and elsewhere is scalability. This unwavering commitment to scalability -- to identify strategies for efficient growth -- is at the heart of what we refer to as "scale thinking." Whether people are aware of it or not, scale thinking is all-encompassing. It is not just an attribute of one's product, service, or company, but frames how one thinks about the world (what constitutes it and how it can be observed and measured), its problems (what is a problem worth solving versus not), and the possible technological fixes for those problems. This paper examines different facets of scale thinking and its implication on how we view technology and collaborative work. We argue that technological solutions grounded in scale thinking are unlikely to be as liberatory or effective at deep, systemic change as their purveyors imagine. Rather, solutions which resist scale thinking are necessary to undo the social structures which lie at the heart of social inequality. We draw on recent work on mutual aid networks and propose questions to ask of collaborative work systems as a means to evaluate technological solutions and guide designers in identifying sites of resistance to scale thinking.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper: <a href="https://twitter.com/teeepain?ref_src=twsrc%5Etfw">@teeepain</a> and I wrote a short piece for the recent <a href="https://twitter.com/hashtag/CSCW2020?src=hash&amp;ref_src=twsrc%5Etfw">#CSCW2020</a> workshop on Reconsidering Scale and Scaling, wherein we try to map the dimensions of &quot;scale thinking&quot; in Valley culture and map out resistances in mutual aid <a href="https://t.co/CetJ8WoM5C">https://t.co/CetJ8WoM5C</a></p>&mdash; Dr. Alex Hanna is just a witch, oh little old me (@alexhanna) <a href="https://twitter.com/alexhanna/status/1318356831878537216?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Against scale: how to resist Silicon Valley scale thinking - &quot;solutions which resist scale thinking are necessary to undo the social structures which lie at the heart of social inequality&quot; <a href="https://t.co/xjYRlpypNM">https://t.co/xjYRlpypNM</a> ht <a href="https://twitter.com/gleemie?ref_src=twsrc%5Etfw">@gleemie</a> cc <a href="https://twitter.com/Gina_labs?ref_src=twsrc%5Etfw">@Gina_labs</a></p>&mdash; giulio quaggiotto (@gquaggiotto) <a href="https://twitter.com/gquaggiotto/status/1318452389586685953?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Reinforcement Learning for Efficient and Tuning-Free Link Adaptation

Vidit Saxena, Hugo Tullberg, Joakim Jaldén

- retweets: 512, favorites: 22 (10/21/2020 09:32:39)

- links: [abs](https://arxiv.org/abs/2010.08651) | [pdf](https://arxiv.org/pdf/2010.08651)
- [eess.SP](https://arxiv.org/list/eess.SP/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Link adaptation (LA) optimizes the selection of modulation and coding schemes (MCS) for a stochastic wireless channel. The classical outer loop LA (OLLA) tracks the channel's signal-to-noise-and-interference ratio (SINR) based on the observed transmission outcomes. On the other hand, recent Reinforcement learning LA (RLLA) schemes sample the available MCSs to optimize the link performance objective. However, both OLLA and RLLA rely on tuning parameters that are challenging to configure. Further, OLLA optimizes for a target block error rate (BLER) that only indirectly relates to the common throughput-maximization objective, while RLLA does not fully exploit the inter-dependence between the MCSs. In this paper, we propose latent Thompson Sampling for LA (LTSLA), a RLLA scheme that does not require configuration tuning, and which fully exploits MCS inter-dependence for efficient learning. LTSLA models an SINR probability distribution for MCS selection, and refines this distribution through Bayesian updates with the transmission outcomes. LTSLA also automatically adapts to different channel fading profiles by utilizing their respective Doppler estimates. We perform simulation studies of LTSLA along with OLLA and RLLA schemes for frequency selective fading channels. Numerical results demonstrate that LTSLA improves the instantaneous link throughout by up to 50% compared to existing schemes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Reinforcement Learning for Efficient and Tuning-Free Link Adaptation. <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/Java?src=hash&amp;ref_src=twsrc%5Etfw">#Java</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/Coding?src=hash&amp;ref_src=twsrc%5Etfw">#Coding</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/Statistics?src=hash&amp;ref_src=twsrc%5Etfw">#Statistics</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://t.co/c5Qtf9UN9S">https://t.co/c5Qtf9UN9S</a> <a href="https://t.co/XrflsJshxz">pic.twitter.com/XrflsJshxz</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1318692712233766914?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Poisoned classifiers are not only backdoored, they are fundamentally  broken

Mingjie Sun, Siddhant Agarwal, J. Zico Kolter

- retweets: 140, favorites: 63 (10/21/2020 09:32:39)

- links: [abs](https://arxiv.org/abs/2010.09080) | [pdf](https://arxiv.org/pdf/2010.09080)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent)

Under a commonly-studied "backdoor" poisoning attack against classification models, an attacker adds a small "trigger" to a subset of the training data, such that the presence of this trigger at test time causes the classifier to always predict some target class. It is often implicitly assumed that the poisoned classifier is vulnerable exclusively to the adversary who possesses the trigger. In this paper, we show empirically that this view of backdoored classifiers is fundamentally incorrect. We demonstrate that anyone with access to the classifier, even without access to any original training data or trigger, can construct several alternative triggers that are as effective or more so at eliciting the target class at test time. We construct these alternative triggers by first generating adversarial examples for a smoothed version of the classifier, created with a recent process called Denoised Smoothing, and then extracting colors or cropped portions of adversarial images. We demonstrate the effectiveness of our attack through extensive experiments on ImageNet and TrojAI datasets, including a user study which demonstrates that our method allows users to easily determine the existence of such backdoors in existing poisoned classifiers. Furthermore, we demonstrate that our alternative triggers can in fact look entirely different from the original trigger, highlighting that the backdoor actually learned by the classifier differs substantially from the trigger image itself. Thus, we argue that there is no such thing as a "secret" backdoor in poisoned classifiers: poisoning a classifier invites attacks not just by the party that possesses the trigger, but from anyone with access to the classifier. Code is available at https://github.com/locuslab/breaking-poisoned-classifier.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Is the backdoor secret? Checkout our new work on &#39;&#39;breaking&#39;&#39; poisoned classifiers,  where we use neat ideas in adversarial robustness to analyze backdoored classifiers. Joint work with <a href="https://twitter.com/agsidd10?ref_src=twsrc%5Etfw">@agsidd10</a> &amp; <a href="https://twitter.com/zicokolter?ref_src=twsrc%5Etfw">@zicokolter</a>.<br><br>Paper: <a href="https://t.co/IRSS1q65Ky">https://t.co/IRSS1q65Ky</a><br>Code: <a href="https://t.co/KCDrek7FTP">https://t.co/KCDrek7FTP</a> <a href="https://t.co/yAyY8zsDD3">pic.twitter.com/yAyY8zsDD3</a></p>&mdash; Mingjie Sun (@Eric_jie_thu) <a href="https://twitter.com/Eric_jie_thu/status/1318567762407677952?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Current backdoor poisoning attacks aren&#39;t just vulnerable to attackers with a &quot;secret trigger&quot;. They can easily be broken (creating a new trigger that is just as effective) given access to the classifier.  New paper with <a href="https://twitter.com/Eric_jie_thu?ref_src=twsrc%5Etfw">@Eric_jie_thu</a> and <a href="https://twitter.com/agsidd10?ref_src=twsrc%5Etfw">@agsidd10</a>.<a href="https://t.co/afgPh8C2rE">https://t.co/afgPh8C2rE</a> <a href="https://t.co/EIY5alsXjx">https://t.co/EIY5alsXjx</a></p>&mdash; Zico Kolter (@zicokolter) <a href="https://twitter.com/zicokolter/status/1318576557808562178?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Light Stage Super-Resolution: Continuous High-Frequency Relighting

Tiancheng Sun, Zexiang Xu, Xiuming Zhang, Sean Fanello, Christoph Rhemann, Paul Debevec, Yun-Ta Tsai, Jonathan T. Barron, Ravi Ramamoorthi

- retweets: 110, favorites: 55 (10/21/2020 09:32:39)

- links: [abs](https://arxiv.org/abs/2010.08888) | [pdf](https://arxiv.org/pdf/2010.08888)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

The light stage has been widely used in computer graphics for the past two decades, primarily to enable the relighting of human faces. By capturing the appearance of the human subject under different light sources, one obtains the light transport matrix of that subject, which enables image-based relighting in novel environments. However, due to the finite number of lights in the stage, the light transport matrix only represents a sparse sampling on the entire sphere. As a consequence, relighting the subject with a point light or a directional source that does not coincide exactly with one of the lights in the stage requires interpolation and resampling the images corresponding to nearby lights, and this leads to ghosting shadows, aliased specularities, and other artifacts. To ameliorate these artifacts and produce better results under arbitrary high-frequency lighting, this paper proposes a learning-based solution for the "super-resolution" of scans of human faces taken from a light stage. Given an arbitrary "query" light direction, our method aggregates the captured images corresponding to neighboring lights in the stage, and uses a neural network to synthesize a rendering of the face that appears to be illuminated by a "virtual" light source at the query location. This neural network must circumvent the inherent aliasing and regularity of the light stage data that was used for training, which we accomplish through the use of regularized traditional interpolation methods within our network. Our learned model is able to produce renderings for arbitrary light directions that exhibit realistic shadows and specular highlights, and is able to generalize across a wide variety of subjects.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Light Stage Super-Resolution: Continuous High-Frequency Relighting<br>pdf: <a href="https://t.co/sQG9Hp98DH">https://t.co/sQG9Hp98DH</a><br>abs: <a href="https://t.co/pdvcGg9D5w">https://t.co/pdvcGg9D5w</a> <a href="https://t.co/2IoudaJE3n">pic.twitter.com/2IoudaJE3n</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1318393772661788673?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. An individual-level ground truth dataset for home location detection

Luca Pappalardo, Leo Ferres, Manuel Sacasa, Ciro Cattuto, Loreto Bravo

- retweets: 134, favorites: 29 (10/21/2020 09:32:39)

- links: [abs](https://arxiv.org/abs/2010.08814) | [pdf](https://arxiv.org/pdf/2010.08814)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

Home detection, assigning a phone device to its home antenna, is a ubiquitous part of most studies in the literature on mobile phone data. Despite its widespread use, home detection relies on a few assumptions that are difficult to check without ground truth, i.e., where the individual that owns the device resides. In this paper, we provide an unprecedented evaluation of the accuracy of home detection algorithms on a group of sixty-five participants for whom we know their exact home address and the antennas that might serve them. Besides, we analyze not only Call Detail Records (CDRs) but also two other mobile phone streams: eXtended Detail Records (XDRs, the ``data'' channel) and Control Plane Records (CPRs, the network stream). These data streams vary not only in their temporal granularity but also they differ in the data generation mechanism', e.g., CDRs are purely human-triggered while CPR is purely machine-triggered events. Finally, we quantify the amount of data that is needed for each stream to carry out successful home detection for each stream. We find that the choice of stream and the algorithm heavily influences home detection, with an hour-of-day algorithm for the XDRs performing the best, and with CPRs performing best for the amount of data needed to perform home detection. Our work is useful for researchers and practitioners in order to minimize data requests and to maximize the accuracy of home antenna location.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Good news, everyone! Our paper on home-tower identification for CDRs, XDRs, and CPRs using *ground truth data* is up on arxiv! <a href="https://t.co/HGjO69bSer">https://t.co/HGjO69bSer</a> 1/n <a href="https://t.co/Sla77bFrZb">pic.twitter.com/Sla77bFrZb</a></p>&mdash; Leo Ferres (@leoferres) <a href="https://twitter.com/leoferres/status/1318563326042013696?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Querent Intent in Multi-Sentence Questions

Laurie Burchell, Jie Chi, Tom Hosking, Nina Markl, Bonnie Webber

- retweets: 92, favorites: 32 (10/21/2020 09:32:39)

- links: [abs](https://arxiv.org/abs/2010.08980) | [pdf](https://arxiv.org/pdf/2010.08980)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Multi-sentence questions (MSQs) are sequences of questions connected by relations which, unlike sequences of standalone questions, need to be answered as a unit. Following Rhetorical Structure Theory (RST), we recognise that different "question discourse relations" between the subparts of MSQs reflect different speaker intents, and consequently elicit different answering strategies. Correctly identifying these relations is therefore a crucial step in automatically answering MSQs. We identify five different types of MSQs in English, and define five novel relations to describe them. We extract over 162,000 MSQs from Stack Exchange to enable future research. Finally, we implement a high-precision baseline classifier based on surface features.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work! With <a href="https://twitter.com/Edin_CDT_NLP?ref_src=twsrc%5Etfw">@Edin_CDT_NLP</a> pals <a href="https://twitter.com/very_laurie?ref_src=twsrc%5Etfw">@very_laurie</a> <a href="https://twitter.com/sociofauxnetic?ref_src=twsrc%5Etfw">@sociofauxnetic</a> Jie Chi and Bonnie Webber: <br>&quot;Querent Intent in Multi-Sentence Questions&quot;<a href="https://t.co/ueuGkZTdD9">https://t.co/ueuGkZTdD9</a><br>To appear at LAW <a href="https://twitter.com/coling2020?ref_src=twsrc%5Etfw">@coling2020</a></p>&mdash; Tom Hosking (@tomhosking) <a href="https://twitter.com/tomhosking/status/1318498896344195073?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Layer-wise Characterization of Latent Information Leakage in Federated  Learning

Fan Mo, Anastasia Borovykh, Mohammad Malekzadeh, Hamed Haddadi, Soteris Demetriou

- retweets: 62, favorites: 35 (10/21/2020 09:32:39)

- links: [abs](https://arxiv.org/abs/2010.08762) | [pdf](https://arxiv.org/pdf/2010.08762)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Training a deep neural network (DNN) via federated learning allows participants to share model updates (gradients), instead of the data itself. However, recent studies show that unintended latent information (e.g. gender or race) carried by the gradients can be discovered by attackers, compromising the promised privacy guarantee of federated learning. Existing privacy-preserving techniques (e.g. differential privacy) either have limited defensive capacity against the potential attacks, or suffer from considerable model utility loss. Moreover, characterizing the latent information carried by the gradients and the consequent privacy leakage has been a major theoretical and practical challenge. In this paper, we propose two new metrics to address these challenges: the empirical $\mathcal{V}$-information, a theoretically grounded notion of information which measures the amount of gradient information that is usable for an attacker, and the sensitivity analysis that utilizes the Jacobian matrix to measure the amount of changes in the gradients with respect to latent information which further quantifies private risk. We show that these metrics can localize the private information in each layer of a DNN and quantify the leakage depending on how sensitive the gradients are with respect to the latent information. As a practical application, we design LatenTZ: a federated learning framework that lets the most sensitive layers to run in the clients' Trusted Execution Environments (TEE). The implementation evaluation of LatenTZ shows that TEE-based approaches are promising for defending against powerful property inference attacks without a significant overhead in the clients' computing resources nor trading off the model's utility.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sharing model params/gradients in Federated Learning leaks sensitive information. We investigate the privacy threats and evaluate layer-wise leakages using information theory and sensitivity analysis. Then we use TEEs to minimize the risks on edge devices. <a href="https://t.co/Unvo715qvq">https://t.co/Unvo715qvq</a> <a href="https://t.co/xbi0o06q4N">pic.twitter.com/xbi0o06q4N</a></p>&mdash; Fan Vincent Mo (@VincentMo6) <a href="https://twitter.com/VincentMo6/status/1318474924810506242?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Permutationless Many-Jet Event Reconstruction with Symmetry Preserving  Attention Networks

Michael James Fenton, Alexander Shmakov, Ta-Wei Ho, Shih-Chieh Hsu, Daniel Whiteson, Pierre Baldi

- retweets: 30, favorites: 40 (10/21/2020 09:32:40)

- links: [abs](https://arxiv.org/abs/2010.09206) | [pdf](https://arxiv.org/pdf/2010.09206)
- [hep-ex](https://arxiv.org/list/hep-ex/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [hep-ph](https://arxiv.org/list/hep-ph/recent)

Top quarks are the most massive particle in the Standard Model and are produced in large numbers at the Large Hadron Collider. As the only quark to decay prior to hadronization, they have a complex detector signature and require special reconstruction techniques. The most common decay mode, the so-called "all-hadronic" channel, results in a 6-jet final state which is particularly difficult to reconstruct in $pp$ collisions due to the large number of permutations possible. We present a novel approach to this class of problem, based on neural networks using a generalized attention mechanism, that we call Symmetry Preserving Attention Networks (\ProjectName). We train one such network to identify and assign the decay products of each top quark unambiguously and without combinatorial explosion as an example of the power of this technique. This approach significantly outperforms existing state-of-the-art methods, correctly assigning all jets in 93.0\% of 6-jet events, 87.8\% of events with 7 jets, and 82.6\% of events with $\geq 8$ jets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper! <br><br>   Permutationless Many-Jet Event Reconstruction with Symmetry Preserving Attention Networks<a href="https://t.co/8K1wFbxmFW">https://t.co/8K1wFbxmFW</a><br><br>Led by <a href="https://twitter.com/mfentonHEP?ref_src=twsrc%5Etfw">@mfentonHEP</a> and Alex Shmakov. <a href="https://t.co/V54irMV0zh">pic.twitter.com/V54irMV0zh</a></p>&mdash; Daniel Whiteson (@DanielWhiteson) <a href="https://twitter.com/DanielWhiteson/status/1318359589205676032?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Paid and hypothetical time preferences are the same: Lab, field and  online evidence

Pablo Brañas-Garza, Diego Jorrat, Antonio M. Espín, Angel Sánchez

- retweets: 32, favorites: 23 (10/21/2020 09:32:40)

- links: [abs](https://arxiv.org/abs/2010.09262) | [pdf](https://arxiv.org/pdf/2010.09262)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.GT](https://arxiv.org/list/cs.GT/recent)

The use of hypothetical instead of real decision-making incentives remains under debate after decades of economic experiments. Standard incentivized experiments involve substantial monetary costs due to participants' earnings and often logistic costs as well. In time preferences experiments, which involve future payments, real payments are particularly problematic. Since immediate rewards frequently have lower transaction costs than delayed rewards in experimental tasks, among other issues, (quasi)hyperbolic functional forms cannot be accurately estimated. What if hypothetical payments provide accurate data which, moreover, avoid transaction cost problems? In this paper, we test whether the use of hypothetical - versus real - payments affects the elicitation of short-term and long-term discounting in a standard multiple price list task. One-out-of-ten participants probabilistic payment schemes are also considered. We analyze data from three studies: a lab experiment in Spain, a well-powered field experiment in Nigeria, and an online extension focused on probabilistic payments. Our results indicate that paid and hypothetical time preferences are mostly the same and, therefore, that hypothetical rewards are a good alternative to real rewards. However, our data suggest that probabilistic payments are not.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper on hypothetical vs real payments. “Paid and hypothetical time preferences are the same: Lab, field and online evidence<br><br>With ⁦<a href="https://twitter.com/dajorrat?ref_src=twsrc%5Etfw">@dajorrat</a>⁩ ⁦⁦<a href="https://twitter.com/A_M_Espin?ref_src=twsrc%5Etfw">@A_M_Espin</a>⁩ <a href="https://twitter.com/anxosan?ref_src=twsrc%5Etfw">@anxosan</a>⁩ <br><br>⁦<a href="https://twitter.com/LoyolaAnd?ref_src=twsrc%5Etfw">@LoyolaAnd</a>⁩ ⁦<a href="https://twitter.com/LoyolaBehLab?ref_src=twsrc%5Etfw">@LoyolaBehLab</a>⁩ ⁦<a href="https://twitter.com/EcScienceAssoc?ref_src=twsrc%5Etfw">@EcScienceAssoc</a>⁩  <a href="https://t.co/QOEgPFcD1L">https://t.co/QOEgPFcD1L</a></p>&mdash; BehaviouralSnapshots (@BehSnaps) <a href="https://twitter.com/BehSnaps/status/1318651943359795201?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Warrior1: A Performance Sanitizer for C++

Nadav Rotem, Lee Howes, David Goldblatt

- retweets: 4, favorites: 46 (10/21/2020 09:32:40)

- links: [abs](https://arxiv.org/abs/2010.09583) | [pdf](https://arxiv.org/pdf/2010.09583)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

This paper presents Warrior1, a tool that detects performance anti-patterns in C++ libraries. Many programs are slowed down by many small inefficiencies. Large-scale C++ applications are large, complex, and developed by large groups of engineers over a long period of time, which makes the task of identifying inefficiencies difficult. Warrior1 was designed to detect the numerous small performance issues that are the result of inefficient use of C++ libraries. The tool detects performance anti-patterns such as map double-lookup, vector reallocation, short lived objects, and lambda object capture by value. Warrior1 is implemented as an instrumented C++ standard library and an off-line diagnostics tool. The tool is very effective in detecting issues. We demonstrate that the tool is able to find a wide range of performance anti-patterns in a number of popular performance sensitive open source projects.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Warrior1: A Performance Sanitizer for C++<br><br>W1 detects performance anti-patterns in C++. Things like: map double-lookup, vector reallocation, short lived objects, and lambda object capture by value. <br><br>Paper: <a href="https://t.co/LQMxccFpai">https://t.co/LQMxccFpai</a> w/ <a href="https://twitter.com/LeeWHowes?ref_src=twsrc%5Etfw">@LeeWHowes</a> + <a href="https://twitter.com/davidtgoldblatt?ref_src=twsrc%5Etfw">@davidtgoldblatt</a></p>&mdash; Nadav Rotem (@nadavrot) <a href="https://twitter.com/nadavrot/status/1318404023385432066?ref_src=twsrc%5Etfw">October 20, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



