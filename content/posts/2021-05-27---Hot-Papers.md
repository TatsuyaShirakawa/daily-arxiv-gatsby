---
title: Hot Papers 2021-05-27
date: 2021-05-28T06:58:11.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-27"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-27"
socialImage: "/media/flying-marine.jpg"

---

# 1. From Motor Control to Team Play in Simulated Humanoid Football

Siqi Liu, Guy Lever, Zhe Wang, Josh Merel, S. M. Ali Eslami, Daniel Hennes, Wojciech M. Czarnecki, Yuval Tassa, Shayegan Omidshafiei, Abbas Abdolmaleki, Noah Y. Siegel, Leonard Hasenclever, Luke Marris, Saran Tunyasuvunakool, H. Francis Song, Markus Wulfmeier, Paul Muller, Tuomas Haarnoja, Brendan D. Tracey, Karl Tuyls, Thore Graepel, Nicolas Heess

- retweets: 12015, favorites: 13 (05/28/2021 06:58:11)

- links: [abs](https://arxiv.org/abs/2105.12196) | [pdf](https://arxiv.org/pdf/2105.12196)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Intelligent behaviour in the physical world exhibits structure at multiple spatial and temporal scales. Although movements are ultimately executed at the level of instantaneous muscle tensions or joint torques, they must be selected to serve goals defined on much longer timescales, and in terms of relations that extend far beyond the body itself, ultimately involving coordination with other agents. Recent research in artificial intelligence has shown the promise of learning-based approaches to the respective problems of complex movement, longer-term planning and multi-agent coordination. However, there is limited research aimed at their integration. We study this problem by training teams of physically simulated humanoid avatars to play football in a realistic virtual environment. We develop a method that combines imitation learning, single- and multi-agent reinforcement learning and population-based training, and makes use of transferable representations of behaviour for decision making at different levels of abstraction. In a sequence of stages, players first learn to control a fully articulated body to perform realistic, human-like movements such as running and turning; they then acquire mid-level football skills such as dribbling and shooting; finally, they develop awareness of others and play as a team, bridging the gap between low-level motor control at a timescale of milliseconds, and coordinated goal-directed behaviour as a team at the timescale of tens of seconds. We investigate the emergence of behaviours at different levels of abstraction, as well as the representations that underlie these behaviours using several analysis techniques, including statistics from real-world sports analytics. Our work constitutes a complete demonstration of integrated decision-making at multiple scales in a physically embodied multi-agent setting. See project video at https://youtu.be/KHMwq9pv7mg.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">From Motor Control to Team Play in Simulated Humanoid Football<br>pdf: <a href="https://t.co/rCLmDoicRd">https://t.co/rCLmDoicRd</a><br>abs: <a href="https://t.co/bOccFcOhf9">https://t.co/bOccFcOhf9</a><br>video: <a href="https://t.co/pebXTv7f7T">https://t.co/pebXTv7f7T</a> <a href="https://t.co/Iq3RqeUTbj">pic.twitter.com/Iq3RqeUTbj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1397725169930706944?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Aggregating Nested Transformers

Zizhao Zhang, Han Zhang, Long Zhao, Ting Chen, Tomas Pfister

- retweets: 1256, favorites: 279 (05/28/2021 06:58:11)

- links: [abs](https://arxiv.org/abs/2105.12723) | [pdf](https://arxiv.org/pdf/2105.12723)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Although hierarchical structures are popular in recent vision transformers, they require sophisticated designs and massive datasets to work well. In this work, we explore the idea of nesting basic local transformers on non-overlapping image blocks and aggregating them in a hierarchical manner. We find that the block aggregation function plays a critical role in enabling cross-block non-local information communication. This observation leads us to design a simplified architecture with minor code changes upon the original vision transformer and obtains improved performance compared to existing methods. Our empirical results show that the proposed method NesT converges faster and requires much less training data to achieve good generalization. For example, a NesT with 68M parameters trained on ImageNet for 100/300 epochs achieves $82.3\%/83.8\%$ accuracy evaluated on $224\times 224$ image size, outperforming previous methods with up to $57\%$ parameter reduction. Training a NesT with 6M parameters from scratch on CIFAR10 achieves $96\%$ accuracy using a single GPU, setting a new state of the art for vision transformers. Beyond image classification, we extend the key idea to image generation and show NesT leads to a strong decoder that is 8$\times$ faster than previous transformer based generators. Furthermore, we also propose a novel method for visually interpreting the learned model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Aggregating Nested Transformers<br>pdf: <a href="https://t.co/qbGFpbmGE0">https://t.co/qbGFpbmGE0</a><br>abs: <a href="https://t.co/Ju3GLt1l7M">https://t.co/Ju3GLt1l7M</a><br><br>68M achieves 82.3%/83.8% accuracy, NesT with 6M parameters from scratch on CIFAR10 achieves 96% accuracy using a single GPU, new SOTA, strong decoder 8× faster <a href="https://t.co/P20cmHlLVM">pic.twitter.com/P20cmHlLVM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1397717654383771651?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Aggregating Nested Transformers<br><br>NesT outperforms previous methods with up to 57% parameter reduction on Imagenet and leads to a strong generative model that is 8x faster than previous transformer based generators.<a href="https://t.co/1psqq7stsY">https://t.co/1psqq7stsY</a> <a href="https://t.co/LyI22Y0sfR">pic.twitter.com/LyI22Y0sfR</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1397716308842541057?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Neural Radiosity

Saeed Hadadan, Shuhong Chen, Matthias Zwicker

- retweets: 240, favorites: 86 (05/28/2021 06:58:11)

- links: [abs](https://arxiv.org/abs/2105.12319) | [pdf](https://arxiv.org/pdf/2105.12319)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We introduce Neural Radiosity, an algorithm to solve the rendering equation by minimizing the norm of its residual similar as in traditional radiosity techniques. Traditional basis functions used in radiosity techniques, such as piecewise polynomials or meshless basis functions are typically limited to representing isotropic scattering from diffuse surfaces. Instead, we propose to leverage neural networks to represent the full four-dimensional radiance distribution, directly optimizing network parameters to minimize the norm of the residual. Our approach decouples solving the rendering equation from rendering (perspective) images similar as in traditional radiosity techniques, and allows us to efficiently synthesize arbitrary views of a scene. In addition, we propose a network architecture using geometric learnable features that improves convergence of our solver compared to previous techniques. Our approach leads to an algorithm that is simple to implement, and we demonstrate its effectiveness on a variety of scenes with non-diffuse surfaces.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Radiosity<br>pdf: <a href="https://t.co/9euDT2JXuA">https://t.co/9euDT2JXuA</a><br>abs: <a href="https://t.co/JsVLXoHRf9">https://t.co/JsVLXoHRf9</a><br><br>leverage nns to represent the full four-dimensional radiance distribution, directly optimizing network parameters to minimize the norm of the residual <a href="https://t.co/JJ9EaD688x">pic.twitter.com/JJ9EaD688x</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1397721030421065730?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. SimNet: Learning Reactive Self-driving Simulations from Real-world  Observations

Luca Bergamini, Yawei Ye, Oliver Scheel, Long Chen, Chih Hu, Luca Del Pero, Blazej Osinski, Hugo Grimmett, Peter Ondruska

- retweets: 144, favorites: 38 (05/28/2021 06:58:12)

- links: [abs](https://arxiv.org/abs/2105.12332) | [pdf](https://arxiv.org/pdf/2105.12332)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this work, we present a simple end-to-end trainable machine learning system capable of realistically simulating driving experiences. This can be used for the verification of self-driving system performance without relying on expensive and time-consuming road testing. In particular, we frame the simulation problem as a Markov Process, leveraging deep neural networks to model both state distribution and transition function. These are trainable directly from the existing raw observations without the need for any handcrafting in the form of plant or kinematic models. All that is needed is a dataset of historical traffic episodes. Our formulation allows the system to construct never seen scenes that unfold realistically reacting to the self-driving car's behaviour. We train our system directly from 1,000 hours of driving logs and measure both realism, reactivity of the simulation as the two key properties of the simulation. At the same time, we apply the method to evaluate the performance of a recently proposed state-of-the-art ML planning system trained from human driving logs. We discover this planning system is prone to previously unreported causal confusion issues that are difficult to test by non-reactive simulation. To the best of our knowledge, this is the first work that directly merges highly realistic data-driven simulations with a closed-loop evaluation for self-driving vehicles. We make the data, code, and pre-trained models publicly available to further stimulate simulation development.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SimNet: Learning Reactive Self-driving Simulations<br>from Real-world Observations<br>pdf: <a href="https://t.co/ImQFTUEdsR">https://t.co/ImQFTUEdsR</a><br>abs: <a href="https://t.co/VcULUHDRZV">https://t.co/VcULUHDRZV</a><br>project page: <a href="https://t.co/AGzDUEA65H">https://t.co/AGzDUEA65H</a><br>code: <a href="https://t.co/M4IDEjxjQM">https://t.co/M4IDEjxjQM</a><br>colab: <a href="https://t.co/iaXgqvCv8m">https://t.co/iaXgqvCv8m</a><br>video: <a href="https://t.co/kYsT9ZroHw">https://t.co/kYsT9ZroHw</a> <a href="https://t.co/gClzozfhrt">pic.twitter.com/gClzozfhrt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1397766325687160832?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Provable Representation Learning for Imitation with Contrastive Fourier  Features

Ofir Nachum, Mengjiao Yang

- retweets: 56, favorites: 40 (05/28/2021 06:58:12)

- links: [abs](https://arxiv.org/abs/2105.12272) | [pdf](https://arxiv.org/pdf/2105.12272)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

In imitation learning, it is common to learn a behavior policy to match an unknown target policy via max-likelihood training on a collected set of target demonstrations. In this work, we consider using offline experience datasets - potentially far from the target distribution - to learn low-dimensional state representations that provably accelerate the sample-efficiency of downstream imitation learning. A central challenge in this setting is that the unknown target policy itself may not exhibit low-dimensional behavior, and so there is a potential for the representation learning objective to alias states in which the target policy acts differently. Circumventing this challenge, we derive a representation learning objective which provides an upper bound on the performance difference between the target policy and a lowdimensional policy trained with max-likelihood, and this bound is tight regardless of whether the target policy itself exhibits low-dimensional structure. Moving to the practicality of our method, we show that our objective can be implemented as contrastive learning, in which the transition dynamics are approximated by either an implicit energy-based model or, in some special cases, an implicit linear model with representations given by random Fourier features. Experiments on both tabular environments and high-dimensional Atari games provide quantitative evidence for the practical benefits of our proposed objective.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Provable Representation Learning for Imitation with Contrastive Fourier Features<br>pdf: <a href="https://t.co/BRjzpQRU9b">https://t.co/BRjzpQRU9b</a><br>abs: <a href="https://t.co/diwNGPfRhD">https://t.co/diwNGPfRhD</a> <a href="https://t.co/K5vkvgd47d">pic.twitter.com/K5vkvgd47d</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1397781274123722754?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Entropy and complexity unveil the landscape of memes evolution

Carlo Michele Valensise, Alessandra Serra, Alessandro Galeazzi, Gabriele Etta, Matteo Cinelli, Walter Quattrociocchi

- retweets: 56, favorites: 36 (05/28/2021 06:58:12)

- links: [abs](https://arxiv.org/abs/2105.12376) | [pdf](https://arxiv.org/pdf/2105.12376)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

On the Internet, information circulates fast and widely, and the form of content adapts to comply with users' cognitive abilities. Memes are an emerging aspect of the internet system of signification, and their visual schemes evolve by adapting to a heterogeneous context. A fundamental question is whether they present culturally and temporally transcendent characteristics in their organizing principles. In this work, we study the evolution of 2 million visual memes from Reddit over ten years, from 2011 to 2020, in terms of their statistical complexity and entropy. We find support for the hypothesis that memes are part of an emerging form of internet metalanguage: on one side, we observe an exponential growth with a doubling time of approximately 6 months; on the other side, the complexity of memes contents increases, allowing and adapting to represent social trends and attitudes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our latest work describing the evolution of 10 years of memes on <a href="https://twitter.com/reddit?ref_src=twsrc%5Etfw">@reddit</a> in terms of their visual complexity. Humor is layered and memes have to adapt to this stratified world. <a href="https://t.co/V585cq836c">https://t.co/V585cq836c</a><a href="https://twitter.com/valensic_?ref_src=twsrc%5Etfw">@valensic_</a>  <a href="https://twitter.com/Walter4C?ref_src=twsrc%5Etfw">@Walter4C</a>  <a href="https://twitter.com/DeveloperGale?ref_src=twsrc%5Etfw">@DeveloperGale</a>  <a href="https://twitter.com/gbrtte_?ref_src=twsrc%5Etfw">@gbrtte_</a>  <a href="https://twitter.com/AlessandraSerr1?ref_src=twsrc%5Etfw">@AlessandraSerr1</a> <a href="https://t.co/aSGbvgAf9S">pic.twitter.com/aSGbvgAf9S</a></p>&mdash; Matteo Cinelli (@matteo_cinelli) <a href="https://twitter.com/matteo_cinelli/status/1397840176320290821?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Networks of climate change: Connecting causes and consequences

Petter Holme, Juan C. Rocha

- retweets: 42, favorites: 36 (05/28/2021 06:58:12)

- links: [abs](https://arxiv.org/abs/2105.12537) | [pdf](https://arxiv.org/pdf/2105.12537)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.ao-ph](https://arxiv.org/list/physics.ao-ph/recent) | [q-bio.PE](https://arxiv.org/list/q-bio.PE/recent)

Understanding the causes and consequences of, and devising countermeasures to, global warming is a profoundly complex problem. Even when researchers narrow down the focus to a publishable investigation, their analysis often contains enough interacting components to require a network visualization. Networks are thus both necessary and natural elements of climate science. Furthermore, networks form a mathematical foundation for a multitude of computational and analytical techniques. We are only beginning to see the benefits of this connection between the sciences of climate change and networks. In this review, we cover use-cases of networks in the climate-change literature -- what they represent, how they are analyzed, and what insights they bring. We also discuss network data, tools, and problems yet to be explored.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Networks of climate change<br>Finally past the scrutiny of arXiv moderators: <a href="https://t.co/owvYm2aSWH">https://t.co/owvYm2aSWH</a><br>The arXiv version doesn&#39;t have a link to the code/data: <a href="https://t.co/tNZjMoCo4O">https://t.co/tNZjMoCo4O</a></p>&mdash; Petter Holme (@pholme) <a href="https://twitter.com/pholme/status/1397718457001517056?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Project CodeNet: A Large-Scale AI for Code Dataset for Learning a  Diversity of Coding Tasks

Ruchir Puri, David S. Kung, Geert Janssen, Wei Zhang, Giacomo Domeniconi, Vladmir Zolotov, Julian Dolby, Jie Chen, Mihir Choudhury, Lindsey Decker, Veronika Thost, Luca Buratti, Saurabh Pujar, Ulrich Finkler

- retweets: 36, favorites: 30 (05/28/2021 06:58:12)

- links: [abs](https://arxiv.org/abs/2105.12655) | [pdf](https://arxiv.org/pdf/2105.12655)
- [cs.SE](https://arxiv.org/list/cs.SE/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Advancements in deep learning and machine learning algorithms have enabled breakthrough progress in computer vision, speech recognition, natural language processing and beyond. In addition, over the last several decades, software has been built into the fabric of every aspect of our society. Together, these two trends have generated new interest in the fast-emerging research area of AI for Code. As software development becomes ubiquitous across all industries and code infrastructure of enterprise legacy applications ages, it is more critical than ever to increase software development productivity and modernize legacy applications. Over the last decade, datasets like ImageNet, with its large scale and diversity, have played a pivotal role in algorithmic advancements from computer vision to language and speech understanding. In this paper, we present Project CodeNet, a first-of-its-kind, very large scale, diverse, and high-quality dataset to accelerate the algorithmic advancements in AI for Code. It consists of 14M code samples and about 500M lines of code in 55 different programming languages. Project CodeNet is not only unique in its scale, but also in the diversity of coding tasks it can help benchmark: from code similarity and classification for advances in code recommendation algorithms, and code translation between a large variety programming languages, to advances in code performance (both runtime, and memory) improvement techniques. CodeNet also provides sample input and output test sets for over 7M code samples, which can be critical for determining code equivalence in different languages. As a usability feature, we provide several preprocessing tools in Project CodeNet to transform source codes into representations that can be readily used as inputs into machine learning models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Project CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks<br>pdf: <a href="https://t.co/OxnMLbyngD">https://t.co/OxnMLbyngD</a><br>abs: <a href="https://t.co/rcV0RWqCjk">https://t.co/rcV0RWqCjk</a><br><br>14M code samples and about 500M lines of code in 55 different programming languages <a href="https://t.co/VqkDGi0cuD">pic.twitter.com/VqkDGi0cuD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1397715826497703938?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. IntelliCAT: Intelligent Machine Translation Post-Editing with Quality  Estimation and Translation Suggestion

Dongjun Lee, Junhyeong Ahn, Heesoo Park, Jaemin Jo

- retweets: 44, favorites: 21 (05/28/2021 06:58:12)

- links: [abs](https://arxiv.org/abs/2105.12172) | [pdf](https://arxiv.org/pdf/2105.12172)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We present IntelliCAT, an interactive translation interface with neural models that streamline the post-editing process on machine translation output. We leverage two quality estimation (QE) models at different granularities: sentence-level QE, to predict the quality of each machine-translated sentence, and word-level QE, to locate the parts of the machine-translated sentence that need correction. Additionally, we introduce a novel translation suggestion model conditioned on both the left and right contexts, providing alternatives for specific words or phrases for correction. Finally, with word alignments, IntelliCAT automatically preserves the original document's styles in the translated document. The experimental results show that post-editing based on the proposed QE and translation suggestions can significantly improve translation quality. Furthermore, a user study reveals that three features provided in IntelliCAT significantly accelerate the post-editing task, achieving a 52.9\% speedup in translation time compared to translating from scratch. The interface is publicly available at https://intellicat.beringlab.com/.



