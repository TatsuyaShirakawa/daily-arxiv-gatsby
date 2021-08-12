---
title: Hot Papers 2021-08-12
date: 2021-08-13T08:41:50.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. Post-hoc Interpretability for Neural NLP: A Survey

Andreas Madsen, Siva Reddy, Sarath Chandar

- retweets: 3135, favorites: 247 (08/13/2021 08:41:50)

- links: [abs](https://arxiv.org/abs/2108.04840) | [pdf](https://arxiv.org/pdf/2108.04840)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Natural Language Processing (NLP) models have become increasingly more complex and widespread. With recent developments in neural networks, a growing concern is whether it is responsible to use these models. Concerns such as safety and ethics can be partially addressed by providing explanations. Furthermore, when models do fail, providing explanations is paramount for accountability purposes. To this end, interpretability serves to provide these explanations in terms that are understandable to humans. Central to what is understandable is how explanations are communicated. Therefore, this survey provides a categorization of how recent interpretability methods communicate explanations and discusses the methods in depth. Furthermore, the survey focuses on post-hoc methods, which provide explanations after a model is learned and generally model-agnostic. A common concern for this class of methods is whether they accurately reflect the model. Hence, how these post-hoc methods are evaluated is discussed throughout the paper.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new survey on post-hoc interpretability methods for NLP is out! This covers 19 specific interpretability methods, cites more than 100 publications, and took 1 year to write. I&#39;m very happy this is now public, do consider sharing.<br>Read <a href="https://t.co/03TmDZRsfy">https://t.co/03TmDZRsfy</a>. A thread 🧵 1/6 <a href="https://t.co/2Fx1RpwmG6">pic.twitter.com/2Fx1RpwmG6</a></p>&mdash; Andreas Madsen (@andreas_madsen) <a href="https://twitter.com/andreas_madsen/status/1425794989741727748?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. DEMix Layers: Disentangling Domains for Modular Language Modeling

Suchin Gururangan, Mike Lewis, Ari Holtzman, Noah A. Smith, Luke Zettlemoyer

- retweets: 2892, favorites: 260 (08/13/2021 08:41:51)

- links: [abs](https://arxiv.org/abs/2108.05036) | [pdf](https://arxiv.org/pdf/2108.05036)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We introduce a new domain expert mixture (DEMix) layer that enables conditioning a language model (LM) on the domain of the input text. A DEMix layer is a collection of expert feedforward networks, each specialized to a domain, that makes the LM modular: experts can be mixed, added or removed after initial training. Extensive experiments with autoregressive transformer LMs (up to 1.3B parameters) show that DEMix layers reduce test-time perplexity, increase training efficiency, and enable rapid adaptation with little overhead. We show that mixing experts during inference, using a parameter-free weighted ensemble, allows the model to better generalize to heterogeneous or unseen domains. We also show that experts can be added to iteratively incorporate new domains without forgetting older ones, and that experts can be removed to restrict access to unwanted domains, without additional training. Overall, these results demonstrate benefits of explicitly conditioning on textual domains during language modeling.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to introduce DEMix layers, a module with domain &quot;experts&quot; that make a language model modular! You can mix, add, or remove experts, enabling rapid adaptation. 🧵👇<br><br>Paper: <a href="https://t.co/dK9ZJBBbzw">https://t.co/dK9ZJBBbzw</a><br>Work with <a href="https://twitter.com/ml_perception?ref_src=twsrc%5Etfw">@ml_perception</a>, <a href="https://twitter.com/universeinanegg?ref_src=twsrc%5Etfw">@universeinanegg</a>, <a href="https://twitter.com/nlpnoah?ref_src=twsrc%5Etfw">@nlpnoah</a>, and <a href="https://twitter.com/LukeZettlemoyer?ref_src=twsrc%5Etfw">@LukeZettlemoyer</a> <a href="https://t.co/kK0JLA6YES">pic.twitter.com/kK0JLA6YES</a></p>&mdash; Suchin Gururangan (@ssgrn) <a href="https://twitter.com/ssgrn/status/1425615542837075968?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Simple black-box universal adversarial attacks on medical image  classification based on deep neural networks

Kazuki Koga, Kazuhiro Takemoto

- retweets: 288, favorites: 52 (08/13/2021 08:41:51)

- links: [abs](https://arxiv.org/abs/2108.04979) | [pdf](https://arxiv.org/pdf/2108.04979)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Universal adversarial attacks, which hinder most deep neural network (DNN) tasks using only a small single perturbation called a universal adversarial perturbation (UAP), is a realistic security threat to the practical application of a DNN. In particular, such attacks cause serious problems in medical imaging. Given that computer-based systems are generally operated under a black-box condition in which only queries on inputs are allowed and outputs are accessible, the impact of UAPs seems to be limited because well-used algorithms for generating UAPs are limited to a white-box condition in which adversaries can access the model weights and loss gradients. Nevertheless, we demonstrate that UAPs are easily generatable using a relatively small dataset under black-box conditions. In particular, we propose a method for generating UAPs using a simple hill-climbing search based only on DNN outputs and demonstrate the validity of the proposed method using representative DNN-based medical image classifications. Black-box UAPs can be used to conduct both non-targeted and targeted attacks. Overall, the black-box UAPs showed high attack success rates (40% to 90%), although some of them had relatively low success rates because the method only utilizes limited information to generate UAPs. The vulnerability of black-box UAPs was observed in several model architectures. The results indicate that adversaries can also generate UAPs through a simple procedure under the black-box condition to foil or control DNN-based medical image diagnoses, and that UAPs are a more realistic security threat.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">プレプリントです。深層ニューラルネットワーク（DNN）のほとんどのタスクを失敗させるまたは制御する単一の摂動（UAP）をブラックボックス（敵対者がモデルパラメータに直接アクセスできないという）条件下で生成する手法を提案しました。<a href="https://t.co/BznYajozIm">https://t.co/BznYajozIm</a></p>&mdash; Kazuhiro Takemoto (@kztakemoto) <a href="https://twitter.com/kztakemoto/status/1425631218981236736?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Differentiable Surface Rendering via Non-Differentiable Sampling

Forrester Cole, Kyle Genova, Avneesh Sud, Daniel Vlasic, Zhoutong Zhang

- retweets: 164, favorites: 82 (08/13/2021 08:41:51)

- links: [abs](https://arxiv.org/abs/2108.04886) | [pdf](https://arxiv.org/pdf/2108.04886)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a method for differentiable rendering of 3D surfaces that supports both explicit and implicit representations, provides derivatives at occlusion boundaries, and is fast and simple to implement. The method first samples the surface using non-differentiable rasterization, then applies differentiable, depth-aware point splatting to produce the final image. Our approach requires no differentiable meshing or rasterization steps, making it efficient for large 3D models and applicable to isosurfaces extracted from implicit surface definitions. We demonstrate the effectiveness of our method for implicit-, mesh-, and parametric-surface-based inverse rendering and neural-network training applications. In particular, we show for the first time efficient, differentiable rendering of an isosurface extracted from a neural radiance field (NeRF), and demonstrate surface-based, rather than volume-based, rendering of a NeRF.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Differentiable Surface Rendering via Non-Differentiable Sampling<br>abs: <a href="https://t.co/sMR8JGXZ5w">https://t.co/sMR8JGXZ5w</a><br><br>efficient, differentiable rendering of an isosurface extracted from a neural radiance field (NeRF), and demonstrate surface based, rather than volume-based, rendering of a NeRF <a href="https://t.co/pthlfcu0eL">pic.twitter.com/pthlfcu0eL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1425621248265031689?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We have a new paper on differentiable rendering! The core idea is to produce differentiable occlusions by using non-diff. rasterization to point-sample the geometry, then differentiable splatting to draw the points to the screen. 1/6 <a href="https://t.co/acp7JXAu7w">https://t.co/acp7JXAu7w</a></p>&mdash; Forrester Cole (@forrestercole) <a href="https://twitter.com/forrestercole/status/1425837764545757186?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Composing games into complex institutions

Seth Frey, Jules Hedges, Joshua Tan, Philipp Zahn

- retweets: 133, favorites: 66 (08/13/2021 08:41:51)

- links: [abs](https://arxiv.org/abs/2108.05318) | [pdf](https://arxiv.org/pdf/2108.05318)
- [cs.GT](https://arxiv.org/list/cs.GT/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Game theory is used by all behavioral sciences, but its development has long centered around tools for relatively simple games and toy systems, such as the economic interpretation of equilibrium outcomes. Our contribution, compositional game theory, permits another approach of equally general appeal: the high-level design of large games for expressing complex architectures and representing real-world institutions faithfully. Compositional game theory, grounded in the mathematics underlying programming languages, and introduced here as a general computational framework, increases the parsimony of game representations with abstraction and modularity, accelerates search and design, and helps theorists across disciplines express real-world institutional complexity in well-defined ways.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint!<br><br>“Composing games into complex institutions” with Seth Frey, Josh Tan and Philipp Zahn<br><br>This is a general-audience social science paper, summarising our explorations of using open games to think about institutions and governance<a href="https://t.co/JcYPaVvScH">https://t.co/JcYPaVvScH</a> <a href="https://t.co/IMjBvwBwXw">pic.twitter.com/IMjBvwBwXw</a></p>&mdash; julesh (@_julesh_) <a href="https://twitter.com/_julesh_/status/1425784358154158082?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Variable-Length Music Score Infilling via XLNet and Musically  Specialized Positional Encoding

Chin-Jui Chang, Chun-Yi Lee, Yi-Hsuan Yang

- retweets: 84, favorites: 63 (08/13/2021 08:41:52)

- links: [abs](https://arxiv.org/abs/2108.05064) | [pdf](https://arxiv.org/pdf/2108.05064)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

This paper proposes a new self-attention based model for music score infilling, i.e., to generate a polyphonic music sequence that fills in the gap between given past and future contexts. While existing approaches can only fill in a short segment with a fixed number of notes, or a fixed time span between the past and future contexts, our model can infill a variable number of notes (up to 128) for different time spans. We achieve so with three major technical contributions. First, we adapt XLNet, an autoregressive model originally proposed for unsupervised model pre-training, to music score infilling. Second, we propose a new, musically specialized positional encoding called relative bar encoding that better informs the model of notes' position within the past and future context. Third, to capitalize relative bar encoding, we perform look-ahead onset prediction to predict the onset of a note one time step before predicting the other attributes of the note. We compare our proposed model with two strong baselines and show that our model is superior in both objective and subjective analyses.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/ismir2021?ref_src=twsrc%5Etfw">@ismir2021</a> paper on *variable-length piano infilling* is now on arXiv!   Our model can generate up to 128 notes to fill the gap between two music segments. Done by <a href="https://twitter.com/ChinjuiChang?ref_src=twsrc%5Etfw">@ChinjuiChang</a>.<br><br>🐋paper- <a href="https://t.co/DEB6gwyCwE">https://t.co/DEB6gwyCwE</a><br>🐋code- <a href="https://t.co/MUS6uDayBm">https://t.co/MUS6uDayBm</a><br>🐋demo- <a href="https://t.co/rSCc7nfsjU">https://t.co/rSCc7nfsjU</a> <a href="https://t.co/EuwYgvRAcg">https://t.co/EuwYgvRAcg</a></p>&mdash; Yi-Hsuan Yang (@affige_yang) <a href="https://twitter.com/affige_yang/status/1425633001187155970?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Variable-Length Music Score Infilling via XLNet and Musically Specialized Positional Encoding<br>abs: <a href="https://t.co/3Mx6RESGoK">https://t.co/3Mx6RESGoK</a><br><br>replace the token-based distance attention mechanism<br>in Transformers with a musically specialized one considering relative bar distance <a href="https://t.co/JGbpHRmsWf">pic.twitter.com/JGbpHRmsWf</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1425629507621433352?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Embodied BERT: A Transformer Model for Embodied, Language-guided Visual  Task Completion

Alessandro Suglia, Qiaozi Gao, Jesse Thomason, Govind Thattai, Gaurav Sukhatme

- retweets: 86, favorites: 60 (08/13/2021 08:41:52)

- links: [abs](https://arxiv.org/abs/2108.04927) | [pdf](https://arxiv.org/pdf/2108.04927)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Language-guided robots performing home and office tasks must navigate in and interact with the world. Grounding language instructions against visual observations and actions to take in an environment is an open challenge. We present Embodied BERT (EmBERT), a transformer-based model which can attend to high-dimensional, multi-modal inputs across long temporal horizons for language-conditioned task completion. Additionally, we bridge the gap between successful object-centric navigation models used for non-interactive agents and the language-guided visual task completion benchmark, ALFRED, by introducing object navigation targets for EmBERT training. We achieve competitive performance on the ALFRED benchmark, and EmBERT marks the first transformer-based model to successfully handle the long-horizon, dense, multi-modal histories of ALFRED, and the first ALFRED model to utilize object-centric navigation targets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Embodied BERT: A Transformer Model for Embodied, Language-guided Visual Task Completion<br>paper: <a href="https://t.co/OtNYzgYWJn">https://t.co/OtNYzgYWJn</a><br><br>successfully handles the long-horizon, dense, multi-modal histories of ALFRED, and the first ALFRED model to utilize object-centric navigation targets <a href="https://t.co/IuChkILCwa">pic.twitter.com/IuChkILCwa</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1425620397874155521?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">When we&#39;re cooking and need an ingredient, we navigate the kitchen with it and its location, such as the cabinet or the fridge, in mind. In &quot;Embodied BERT: A Transformer Model for Embodied, Language-guided Visual Task Completion&quot;, we apply this intuition.<a href="https://t.co/isVNuz9Mgi">https://t.co/isVNuz9Mgi</a> <a href="https://t.co/WCW0SVNGKh">pic.twitter.com/WCW0SVNGKh</a></p>&mdash; Alessandro Suglia (@ale_suglia) <a href="https://twitter.com/ale_suglia/status/1425846389188202502?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. FLAME-in-NeRF : Neural control of Radiance Fields for Free View Face  Animation

ShahRukh Athar, Zhixin Shu, Dimitris Samaras

- retweets: 66, favorites: 46 (08/13/2021 08:41:52)

- links: [abs](https://arxiv.org/abs/2108.04913) | [pdf](https://arxiv.org/pdf/2108.04913)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper presents a neural rendering method for controllable portrait video synthesis. Recent advances in volumetric neural rendering, such as neural radiance fields (NeRF), has enabled the photorealistic novel view synthesis of static scenes with impressive results. However, modeling dynamic and controllable objects as part of a scene with such scene representations is still challenging. In this work, we design a system that enables both novel view synthesis for portrait video, including the human subject and the scene background, and explicit control of the facial expressions through a low-dimensional expression representation. We leverage the expression space of a 3D morphable face model (3DMM) to represent the distribution of human facial expressions, and use it to condition the NeRF volumetric function. Furthermore, we impose a spatial prior brought by 3DMM fitting to guide the network to learn disentangled control for scene appearance and facial actions. We demonstrate the effectiveness of our method on free view synthesis of portrait videos with expression controls. To train a scene, our method only requires a short video of a subject captured by a mobile device.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FLAME-in-NeRF : Neural control of Radiance Fields for Free View Face Animation<br>pdf: <a href="https://t.co/tVqP3u8MCU">https://t.co/tVqP3u8MCU</a><br>abs: <a href="https://t.co/N4tn2SwrKD">https://t.co/N4tn2SwrKD</a><br><br>method capable of arbitrary facial expression control and novel view synthesis for portrait video <a href="https://t.co/2uYSUHy78I">pic.twitter.com/2uYSUHy78I</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1425621892510232579?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Logic Explained Networks

Gabriele Ciravegna, Pietro Barbiero, Francesco Giannini, Marco Gori, Pietro Lió, Marco Maggini, Stefano Melacci

- retweets: 56, favorites: 39 (08/13/2021 08:41:52)

- links: [abs](https://arxiv.org/abs/2108.05149) | [pdf](https://arxiv.org/pdf/2108.05149)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LO](https://arxiv.org/list/cs.LO/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

The large and still increasing popularity of deep learning clashes with a major limit of neural network architectures, that consists in their lack of capability in providing human-understandable motivations of their decisions. In situations in which the machine is expected to support the decision of human experts, providing a comprehensible explanation is a feature of crucial importance. The language used to communicate the explanations must be formal enough to be implementable in a machine and friendly enough to be understandable by a wide audience. In this paper, we propose a general approach to Explainable Artificial Intelligence in the case of neural architectures, showing how a mindful design of the networks leads to a family of interpretable deep learning models called Logic Explained Networks (LENs). LENs only require their inputs to be human-understandable predicates, and they provide explanations in terms of simple First-Order Logic (FOL) formulas involving such predicates. LENs are general enough to cover a large number of scenarios. Amongst them, we consider the case in which LENs are directly used as special classifiers with the capability of being explainable, or when they act as additional networks with the role of creating the conditions for making a black-box classifier explainable by FOL formulas. Despite supervised learning problems are mostly emphasized, we also show that LENs can learn and provide explanations in unsupervised learning settings. Experimental results on several datasets and tasks show that LENs may yield better classifications than established white-box models, such as decision trees and Bayesian rule lists, while providing more compact and meaningful explanations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Logic Explained Networks&quot; have been finally released: <a href="https://t.co/pkX94YkuSg">https://t.co/pkX94YkuSg</a>!<a href="https://twitter.com/hashtag/LENs?src=hash&amp;ref_src=twsrc%5Etfw">#LENs</a> are novel <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> models providing first order <a href="https://twitter.com/hashtag/logic?src=hash&amp;ref_src=twsrc%5Etfw">#logic</a> explanations for their predictions.<br><br>A new paradigm for explainable <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a>! <a href="https://t.co/xEK8kG9WJV">pic.twitter.com/xEK8kG9WJV</a></p>&mdash; Pietro Barbiero (@pietro_barbiero) <a href="https://twitter.com/pietro_barbiero/status/1425758277007183876?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Optimal learning of quantum Hamiltonians from high-temperature Gibbs  states

Jeongwan Haah, Robin Kothari, Ewin Tang

- retweets: 30, favorites: 54 (08/13/2021 08:41:53)

- links: [abs](https://arxiv.org/abs/2108.04842) | [pdf](https://arxiv.org/pdf/2108.04842)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We study the problem of learning a Hamiltonian $H$ to precision $\varepsilon$, supposing we are given copies of its Gibbs state $\rho=\exp(-\beta H)/\operatorname{Tr}(\exp(-\beta H))$ at a known inverse temperature $\beta$. Anshu, Arunachalam, Kuwahara, and Soleimanifar (Nature Physics, 2021) recently studied the sample complexity (number of copies of $\rho$ needed) of this problem for geometrically local $N$-qubit Hamiltonians. In the high-temperature (low $\beta$) regime, their algorithm has sample complexity poly$(N, 1/\beta,1/\varepsilon)$ and can be implemented with polynomial, but suboptimal, time complexity.   In this paper, we study the same question for a more general class of Hamiltonians. We show how to learn the coefficients of a Hamiltonian to error $\varepsilon$ with sample complexity $S = O(\log N/(\beta\varepsilon)^{2})$ and time complexity linear in the sample size, $O(S N)$. Furthermore, we prove a matching lower bound showing that our algorithm's sample complexity is optimal, and hence our time complexity is also optimal.   In the appendix, we show that virtually the same algorithm can be used to learn $H$ from a real-time evolution unitary $e^{-it H}$ in a small $t$ regime with similar sample and time complexity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint! <a href="https://t.co/RplXGhYguL">https://t.co/RplXGhYguL</a><br><br>And here&#39;s a 30-minute talk on this work that I gave at Simons, if you&#39;re into that: <a href="https://t.co/xG2CwIDRe8">https://t.co/xG2CwIDRe8</a> <a href="https://t.co/kLAABLjnWC">https://t.co/kLAABLjnWC</a></p>&mdash; ewin (@ewintang) <a href="https://twitter.com/ewintang/status/1425743064786636804?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Adaptive Multi-Resolution Attention with Linear Complexity

Yao Zhang, Yunpu Ma, Thomas Seidl, Volker Tresp

- retweets: 20, favorites: 49 (08/13/2021 08:41:53)

- links: [abs](https://arxiv.org/abs/2108.04962) | [pdf](https://arxiv.org/pdf/2108.04962)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Transformers have improved the state-of-the-art across numerous tasks in sequence modeling. Besides the quadratic computational and memory complexity w.r.t the sequence length, the self-attention mechanism only processes information at the same scale, i.e., all attention heads are in the same resolution, resulting in the limited power of the Transformer. To remedy this, we propose a novel and efficient structure named Adaptive Multi-Resolution Attention (AdaMRA for short), which scales linearly to sequence length in terms of time and space. Specifically, we leverage a multi-resolution multi-head attention mechanism, enabling attention heads to capture long-range contextual information in a coarse-to-fine fashion. Moreover, to capture the potential relations between query representation and clues of different attention granularities, we leave the decision of which resolution of attention to use to query, which further improves the model's capacity compared to vanilla Transformer. In an effort to reduce complexity, we adopt kernel attention without degrading the performance. Extensive experiments on several benchmarks demonstrate the effectiveness and efficiency of our model by achieving a state-of-the-art performance-efficiency-memory trade-off. To facilitate AdaMRA utilization by the scientific community, the code implementation will be made publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Adaptive Multi-Resolution Attention with Linear<br>Complexity<br>pdf: <a href="https://t.co/TDkBQk72ZQ">https://t.co/TDkBQk72ZQ</a><br>abs: <a href="https://t.co/MZjfq84I1z">https://t.co/MZjfq84I1z</a><br><br>scales linearly to sequence length in terms of time and space <a href="https://t.co/j49AZ8xX1A">pic.twitter.com/j49AZ8xX1A</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1425622686932344832?ref_src=twsrc%5Etfw">August 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



