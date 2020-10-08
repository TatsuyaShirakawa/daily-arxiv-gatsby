---
title: Hot Papers 2020-10-07
date: 2020-10-08T10:09:26.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-07"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-07"
socialImage: "/media/flying-marine.jpg"

---

# 1. Participatory Research for Low-resourced Machine Translation: A Case  Study in African Languages

Wilhelmina Nekoto, Vukosi Marivate, Tshinondiwa Matsila, Timi Fasubaa, Tajudeen Kolawole, Taiwo Fagbohungbe, Solomon Oluwole Akinola, Shamsuddee Hassan Muhammad, Salomon Kabongo, Salomey Osei, Sackey Freshia, Rubungo Andre Niyongabo, Ricky Macharm, Perez Ogayo, Orevaoghene Ahia, Musie Meressa, Mofe Adeyemi, Masabata Mokgesi-Selinga, Lawrence Okegbemi, Laura Jane Martinus, Kolawole Tajudeen, Kevin Degila, Kelechi Ogueji, Kathleen Siminyu, Julia Kreutzer, Jason Webster, Jamiil Toure Ali, Jade Abbott, Iroro Orife, Ignatius Ezeani, Idris Abdulkabir Dangana, Herman Kamper, Hady Elsahar, Goodness Duru, Ghollah Kioko, Espoir Murhabazi, Elan van Biljon, Daniel Whitenack, Christopher Onyefuluchi, Chris Emezue, Bonaventure Dossou, Blessing Sibanda, Blessing Itoro Bassey, Ayodele Olabiyi, Arshath Ramkilowan

- retweets: 6596, favorites: 19 (10/08/2020 10:09:26)

- links: [abs](https://arxiv.org/abs/2010.02353) | [pdf](https://arxiv.org/pdf/2010.02353)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Research in NLP lacks geographic diversity, and the question of how NLP can be scaled to low-resourced languages has not yet been adequately solved. "Low-resourced"-ness is a complex problem going beyond data availability and reflects systemic problems in society. In this paper, we focus on the task of Machine Translation (MT), that plays a crucial role for information accessibility and communication worldwide. Despite immense improvements in MT over the past decade, MT is centered around a few high-resourced languages. As MT researchers cannot solve the problem of low-resourcedness alone, we propose participatory research as a means to involve all necessary agents required in the MT development process. We demonstrate the feasibility and scalability of participatory research with a case study on MT for African languages. Its implementation leads to a collection of novel translation datasets, MT benchmarks for over 30 languages, with human evaluations for a third of them, and enables participants without formal training to make a unique scientific contribution. Benchmarks, models, data, code, and evaluation results are released under https://github.com/masakhane-io/masakhane-mt.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">With overwhelming excitement that we can finally share with you &quot;Participatory Research for Low-resourced Machine Translation: A Case Study in African Languages&quot; to be published at ✨Findings of <a href="https://twitter.com/emnlp2020?ref_src=twsrc%5Etfw">@emnlp2020</a> ✨💕🌍💪🏾<br><br>Preprint: <a href="https://t.co/r01i7C9jFs">https://t.co/r01i7C9jFs</a> <br><br>/1 <a href="https://t.co/iLgMmDbZPK">pic.twitter.com/iLgMmDbZPK</a></p>&mdash; Masakhane (@MasakhaneNLP) <a href="https://twitter.com/MasakhaneNLP/status/1313717627722838022?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. If beam search is the answer, what was the question?

Clara Meister, Tim Vieira, Ryan Cotterell

- retweets: 841, favorites: 160 (10/08/2020 10:09:27)

- links: [abs](https://arxiv.org/abs/2010.02650) | [pdf](https://arxiv.org/pdf/2010.02650)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Quite surprisingly, exact maximum a posteriori (MAP) decoding of neural language generators frequently leads to low-quality results. Rather, most state-of-the-art results on language generation tasks are attained using beam search despite its overwhelmingly high search error rate. This implies that the MAP objective alone does not express the properties we desire in text, which merits the question: if beam search is the answer, what was the question? We frame beam search as the exact solution to a different decoding objective in order to gain insights into why high probability under a model alone may not indicate adequacy. We find that beam search enforces uniform information density in text, a property motivated by cognitive science. We suggest a set of decoding objectives that explicitly enforce this property and find that exact decoding with these objectives alleviates the problems encountered when decoding poorly calibrated language generation models. Additionally, we analyze the text produced using various decoding strategies and see that, in our neural machine translation experiments, the extent to which this property is adhered to strongly correlates with BLEU.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Beam search is a hack -- we all know it. So, why does it work so damn well? It’s a SOTA algorithm for decoding neural text generators! Our new EMNLP paper presents a framing of beam search that demonstrates it has a cognitive inductive bias. <a href="https://t.co/JtLtTFVWEO">https://t.co/JtLtTFVWEO</a> <a href="https://t.co/BG1m2JJRPo">pic.twitter.com/BG1m2JJRPo</a></p>&mdash; Clara Isabel Meister (@ClaraIsabelMei1) <a href="https://twitter.com/ClaraIsabelMei1/status/1313816344580808704?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Directional Graph Networks

Dominique Beaini, Saro Passaro, Vincent Létourneau, William L. Hamilton, Gabriele Corso, Pietro Liò

- retweets: 506, favorites: 90 (10/08/2020 10:09:27)

- links: [abs](https://arxiv.org/abs/2010.02863) | [pdf](https://arxiv.org/pdf/2010.02863)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CG](https://arxiv.org/list/cs.CG/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

In order to overcome the expressive limitations of graph neural networks (GNNs), we propose the first method that exploits vector flows over graphs to develop globally consistent directional and asymmetric aggregation functions. We show that our directional graph networks (DGNs) generalize convolutional neural networks (CNNs) when applied on a grid. Whereas recent theoretical works focus on understanding local neighbourhoods, local structures and local isomorphism with no global information flow, our novel theoretical framework allows directional convolutional kernels in any graph. First, by defining a vector field in the graph, we develop a method of applying directional derivatives and smoothing by projecting node-specific messages into the field. Then we propose the use of the Laplacian eigenvectors as such vector field, and we show that the method generalizes CNNs on an n-dimensional grid. Finally, we bring the power of CNN data augmentation to graphs by providing a means of doing reflection, rotation and distortion on the underlying directional field. We evaluate our method on different standard benchmarks and see a relative error reduction of 8% on the CIFAR10 graph dataset and 11% to 32% on the molecular ZINC dataset. An important outcome of this work is that it enables to translate any physical or biological problems with intrinsic directional axes into a graph network formalism with an embedded directional field.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Proud to announce our newest graph <a href="https://twitter.com/hashtag/research?src=hash&amp;ref_src=twsrc%5Etfw">#research</a> <a href="https://twitter.com/hashtag/paper?src=hash&amp;ref_src=twsrc%5Etfw">#paper</a>, we introduce directional aggregations, generalize convolutional <a href="https://twitter.com/hashtag/neuralnetworks?src=hash&amp;ref_src=twsrc%5Etfw">#neuralnetworks</a> in <a href="https://twitter.com/hashtag/graphs?src=hash&amp;ref_src=twsrc%5Etfw">#graphs</a> and solve bottlenecks in GNNs  1/5<a href="https://t.co/n7Jp3Z6UJA">https://t.co/n7Jp3Z6UJA</a><br>Authors:<a href="https://twitter.com/Saro2000?ref_src=twsrc%5Etfw">@Saro2000</a> <a href="https://twitter.com/vincentmillions?ref_src=twsrc%5Etfw">@vincentmillions</a> <a href="https://twitter.com/pl219_Cambridge?ref_src=twsrc%5Etfw">@pl219_Cambridge</a> <a href="https://twitter.com/williamleif?ref_src=twsrc%5Etfw">@williamleif</a> <a href="https://twitter.com/GabriCorso?ref_src=twsrc%5Etfw">@GabriCorso</a> <a href="https://t.co/Vkf4cEGbYW">pic.twitter.com/Vkf4cEGbYW</a></p>&mdash; Dominique Beaini (@dom_beaini) <a href="https://twitter.com/dom_beaini/status/1313831556604141569?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Meta-Learning of Compositional Task Distributions in Humans and Machines

Sreejan Kumar, Ishita Dasgupta, Jonathan D. Cohen, Nathaniel D. Daw, Thomas L. Griffiths

- retweets: 272, favorites: 75 (10/08/2020 10:09:27)

- links: [abs](https://arxiv.org/abs/2010.02317) | [pdf](https://arxiv.org/pdf/2010.02317)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Modern machine learning systems struggle with sample efficiency and are usually trained with enormous amounts of data for each task. This is in sharp contrast with humans, who often learn with very little data. In recent years, meta-learning, in which one trains on a family of tasks (i.e. a task distribution), has emerged as an approach to improving the sample complexity of machine learning systems and to closing the gap between human and machine learning. However, in this paper, we argue that current meta-learning approaches still differ significantly from human learning. We argue that humans learn over tasks by constructing compositional generative models and using these to generalize, whereas current meta-learning methods are biased toward the use of simpler statistical patterns. To highlight this difference, we construct a new meta-reinforcement learning task with a compositional task distribution. We also introduce a novel approach to constructing a "null task distribution" with the same statistical complexity as the compositional distribution but without explicit compositionality. We train a standard meta-learning agent, a recurrent network trained with model-free reinforcement learning, and compare it with human performance across the two task distributions. We find that humans do better in the compositional task distribution whereas the agent does better in the non-compositional null task distribution -- despite comparable statistical complexity. This work highlights a particular difference between human learning and current meta-learning models, introduces a task that displays this difference, and paves the way for future work on human-like meta-learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share a new preprint w/ great collaborators, Ishita Dasgupta, Jon Cohen, Nathaniel Daw (<a href="https://twitter.com/nathanieldaw?ref_src=twsrc%5Etfw">@nathanieldaw</a> ), and Tom Griffiths( <a href="https://twitter.com/cocosci_lab?ref_src=twsrc%5Etfw">@cocosci_lab</a>) on meta-learning in humans and artificial agents! <a href="https://t.co/8gY3rDU5TS">https://t.co/8gY3rDU5TS</a> <a href="https://t.co/Fgsn5kfdb3">pic.twitter.com/Fgsn5kfdb3</a></p>&mdash; Sreejan Kumar (@sreejank97) <a href="https://twitter.com/sreejank97/status/1313637206599389186?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Conditional Generative Adversarial Networks to Model Urban Outdoor Air  Pollution

Jamal Toutouh

- retweets: 285, favorites: 12 (10/08/2020 10:09:27)

- links: [abs](https://arxiv.org/abs/2010.02244) | [pdf](https://arxiv.org/pdf/2010.02244)
- [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This is a relevant problem because the design of most cities prioritizes the use of motorized vehicles, which has degraded air quality in recent years, having a negative effect on urban health. Modeling, predicting, and forecasting ambient air pollution is an important way to deal with this issue because it would be helpful for decision-makers and urban city planners to understand the phenomena and to take solutions. In general, data-driven methods for modeling, predicting, and forecasting outdoor pollution requires an important amount of data, which may limit their accuracy. In order to deal with such a lack of data, we propose to train models able to generate synthetic nitrogen dioxide daily time series according to a given classification that will allow an unlimited generation of realistic data. The main experimental results indicate that the proposed approach is able to generate accurate and diverse pollution daily time series, while requiring reduced computational time.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Conditional Generative Adversarial Networks to Model Urban Outdoor Air Pollution. <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/Java?src=hash&amp;ref_src=twsrc%5Etfw">#Java</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/Coding?src=hash&amp;ref_src=twsrc%5Etfw">#Coding</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a><a href="https://t.co/yJbqcrUmQ1">https://t.co/yJbqcrUmQ1</a> <a href="https://t.co/vc0PFeT4NV">pic.twitter.com/vc0PFeT4NV</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1313887970282344449?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. QADiscourse -- Discourse Relations as QA Pairs: Representation,  Crowdsourcing and Baselines

Valentina Pyatkin, Ayal Klein, Reut Tsarfaty, Ido Dagan

- retweets: 182, favorites: 50 (10/08/2020 10:09:28)

- links: [abs](https://arxiv.org/abs/2010.02815) | [pdf](https://arxiv.org/pdf/2010.02815)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Discourse relations describe how two propositions relate to one another, and identifying them automatically is an integral part of natural language understanding. However, annotating discourse relations typically requires expert annotators. Recently, different semantic aspects of a sentence have been represented and crowd-sourced via question-and-answer (QA) pairs. This paper proposes a novel representation of discourse relations as QA pairs, which in turn allows us to crowd-source wide-coverage data annotated with discourse relations, via an intuitively appealing interface for composing such questions and answers. Based on our proposed representation, we collect a novel and wide-coverage QADiscourse dataset, and present baseline algorithms for predicting QADiscourse relations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can you crowdsource QA annotations of discourse relations?<br>We relate propositions through questions and answers, for ex. by asking about concessions or substitutions.<br>Joint work with <a href="https://twitter.com/kleinay2?ref_src=twsrc%5Etfw">@kleinay2</a> <a href="https://twitter.com/rtsarfaty?ref_src=twsrc%5Etfw">@rtsarfaty</a> and Ido Dagan to appear at <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a> <a href="https://t.co/asby90vR8o">https://t.co/asby90vR8o</a> <a href="https://t.co/vf4gRHcnhY">pic.twitter.com/vf4gRHcnhY</a></p>&mdash; Valentina Pyatkin (@valentina__py) <a href="https://twitter.com/valentina__py/status/1313787340402417665?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. ZX-Calculus and Extended Hypergraph Rewriting Systems I: A Multiway  Approach to Categorical Quantum Information Theory

Jonathan Gorard, Manojna Namuduri, Xerxes D. Arsiwalla

- retweets: 105, favorites: 90 (10/08/2020 10:09:28)

- links: [abs](https://arxiv.org/abs/2010.02752) | [pdf](https://arxiv.org/pdf/2010.02752)
- [cs.LO](https://arxiv.org/list/cs.LO/recent) | [cs.DM](https://arxiv.org/list/cs.DM/recent)

Categorical quantum mechanics and the Wolfram model offer distinct but complementary approaches to studying the relationship between diagrammatic rewriting systems over combinatorial structures and the foundations of physics; the objective of the present article is to begin elucidating the formal correspondence between the two methodologies in the context of the ZX-calculus formalism of Coecke and Duncan for reasoning diagrammatically about linear maps between qubits. After briefly summarizing the relevant formalisms, and presenting a categorical formulation of the Wolfram model in terms of adhesive categories and double-pushout rewriting systems, we illustrate how the diagrammatic rewritings of the ZX-calculus can be embedded and realized within the broader context of Wolfram model multiway systems, and illustrate some of the capabilities of the software framework (ZXMultiwaySystem) that we have developed specifically for this purpose. Finally, we present a proof (along with an explicitly computed example) based on the methods of Dixon and Kissinger that the multiway evolution graphs and branchial graphs of the Wolfram model are naturally endowed with a monoidal structure based on rulial composition that is, furthermore, compatible with the monoidal product of ZX-diagrams.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Starting to build a bridge between <a href="https://twitter.com/wolframphysics?ref_src=twsrc%5Etfw">@wolframphysics</a> and the categorical quantum mechanics of <a href="https://twitter.com/coecke?ref_src=twsrc%5Etfw">@coecke</a> and others. With some potentially interesting implications for quantum information theory. Part II coming soon! <a href="https://t.co/dPQJtXneET">https://t.co/dPQJtXneET</a></p>&mdash; Jonathan Gorard (@getjonwithit) <a href="https://twitter.com/getjonwithit/status/1313651011542814720?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Robust priors for regularized regression

Sebastian Bobadilla-Suarez, Matt Jones, Bradley C. Love

- retweets: 145, favorites: 33 (10/08/2020 10:09:28)

- links: [abs](https://arxiv.org/abs/2010.02610) | [pdf](https://arxiv.org/pdf/2010.02610)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Induction benefits from useful priors. Penalized regression approaches, like ridge regression, shrink weights toward zero but zero association is usually not a sensible prior. Inspired by simple and robust decision heuristics humans use, we constructed non-zero priors for penalized regression models that provide robust and interpretable solutions across several tasks. Our approach enables estimates from a constrained model to serve as a prior for a more general model, yielding a principled way to interpolate between models of differing complexity. We successfully applied this approach to a number of decision and classification problems, as well as analyzing simulated brain imaging data. Models with robust priors had excellent worst-case performance. Solutions followed from the form of the heuristic that was used to derive the prior. These new algorithms can serve applications in data analysis and machine learning, as well as help in understanding how people transition from novice to expert performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new preprint “Robust priors for regularized regression” with <a href="https://twitter.com/ProfData?ref_src=twsrc%5Etfw">@ProfData</a> and Matt Jones is now up on arXiv! We find priors inspired by human decision-making heuristics lead to robust and interpretable models for data analysis. <a href="https://t.co/houLLrOFp7">https://t.co/houLLrOFp7</a> 1/n <a href="https://t.co/lqf9BE0e0S">pic.twitter.com/lqf9BE0e0S</a></p>&mdash; Sebastian Bobadilla Suarez (@seb_bobadilla) <a href="https://twitter.com/seb_bobadilla/status/1313781137806950400?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Please Mind the Root: Decoding Arborescences for Dependency Parsing

Ran Zmigrod, Tim Vieira, Ryan Cotterell

- retweets: 132, favorites: 30 (10/08/2020 10:09:28)

- links: [abs](https://arxiv.org/abs/2010.02550) | [pdf](https://arxiv.org/pdf/2010.02550)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The connection between dependency trees and spanning trees is exploited by the NLP community to train and to decode graph-based dependency parsers. However, the NLP literature has missed an important difference between the two structures: only one edge may emanate from the root in a dependency tree. We analyzed the output of state-of-the-art parsers on many languages from the Universal Dependency Treebank: although these parsers are often able to learn that trees which violate the constraint should be assigned lower probabilities, their ability to do so unsurprisingly de-grades as the size of the training set decreases. In fact, the worst constraint-violation rate we observe is 24%. Prior work has proposed an inefficient algorithm to enforce the constraint, which adds a factor of n to the decoding runtime. We adapt an algorithm due to Gabow and Tarjan (1984) to dependency parsing, which satisfies the constraint without compromising the original runtime.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">⚠️ Attention all NLPers, when decoding dependency trees, please mind the root! ⚠️<br>Check out our new <a href="https://twitter.com/emnlp2020?ref_src=twsrc%5Etfw">@emnlp2020</a> short paper about efficient root-constrained decoding for graph-based dependency parsers!<br>🌲 <a href="https://t.co/UOdFEcKvP1">https://t.co/UOdFEcKvP1</a><br>Joint work with <a href="https://twitter.com/xtimv?ref_src=twsrc%5Etfw">@xtimv</a> and <a href="https://twitter.com/ryandcotterell?ref_src=twsrc%5Etfw">@ryandcotterell</a> <a href="https://t.co/FTgUr70n0X">pic.twitter.com/FTgUr70n0X</a></p>&mdash; Ran Zmigrod (@RanZmigrod) <a href="https://twitter.com/RanZmigrod/status/1313754611770167296?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. NCP-VAE: Variational Autoencoders with Noise Contrastive Priors

Jyoti Aneja, Alexander Schwing, Jan Kautz, Arash Vahdat

- retweets: 72, favorites: 56 (10/08/2020 10:09:28)

- links: [abs](https://arxiv.org/abs/2010.02917) | [pdf](https://arxiv.org/pdf/2010.02917)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Variational autoencoders (VAEs) are one of the powerful likelihood-based generative models with applications in various domains. However, they struggle to generate high-quality images, especially when samples are obtained from the prior without any tempering. One explanation for VAEs' poor generative quality is the prior hole problem: the prior distribution fails to match the aggregate approximate posterior. Due to this mismatch, there exist areas in the latent space with high density under the prior that do not correspond to any encoded image. Samples from those areas are decoded to corrupted images. To tackle this issue, we propose an energy-based prior defined by the product of a base prior distribution and a reweighting factor, designed to bring the base closer to the aggregate posterior. We train the reweighting factor by noise contrastive estimation, and we generalize it to hierarchical VAEs with many latent variable groups. Our experiments confirm that the proposed noise contrastive priors improve the generative performance of state-of-the-art VAEs by a large margin on the MNIST, CIFAR-10, CelebA 64, and CelebA HQ 256 datasets.

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">NCP-VAE: Variational Autoencoders with Noise Contrastive Priors<br>pdf: <a href="https://t.co/SOF8I58ISe">https://t.co/SOF8I58ISe</a><br>abs: <a href="https://t.co/wOj71Ll7Nt">https://t.co/wOj71Ll7Nt</a> <a href="https://t.co/PtQeubqqLK">pic.twitter.com/PtQeubqqLK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1313664057262706695?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Spectral clustering of annotated graphs using a factor graph  representation

Tatsuro Kawamoto

- retweets: 92, favorites: 34 (10/08/2020 10:09:28)

- links: [abs](https://arxiv.org/abs/2010.02791) | [pdf](https://arxiv.org/pdf/2010.02791)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

Graph-structured data commonly have node annotations. A popular approach for inference and learning involving annotated graphs is to incorporate annotations into a statistical model or algorithm. By contrast, we consider a more direct method named scotch-taping, in which the structural information in a graph and its node annotations are encoded as a factor graph. Specifically, we establish the mathematical basis of this method in the spectral framework.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Spectral clustering of annotated graphs using a factor graph representation&quot;<a href="https://t.co/e2TRhJakeW">https://t.co/e2TRhJakeW</a><br><br>When you have an annotated graph, perhaps the first thing that crosses your mind would be “Why can’t I simply add edges?”. ...but how does that contribute to graph spectra? <a href="https://t.co/H7cv43E92P">pic.twitter.com/H7cv43E92P</a></p>&mdash; Tatsuro KAWAMOTO (@Tatmann9) <a href="https://twitter.com/Tatmann9/status/1313762134929469446?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Improving Efficient Neural Ranking Models with Cross-Architecture  Knowledge Distillation

Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, Allan Hanbury

- retweets: 110, favorites: 13 (10/08/2020 10:09:29)

- links: [abs](https://arxiv.org/abs/2010.02666) | [pdf](https://arxiv.org/pdf/2010.02666)
- [cs.IR](https://arxiv.org/list/cs.IR/recent)

The latency of neural ranking models at query time is largely dependent on the architecture and deliberate choices by their designers to trade-off effectiveness for higher efficiency. This focus on low query latency of a rising number of efficient ranking architectures make them feasible for production deployment. In machine learning an increasingly common approach to close the effectiveness gap of more efficient models is to apply knowledge distillation from a large teacher model to a smaller student model. We find that different ranking architectures tend to produce output scores in different magnitudes. Based on this finding, we propose a cross-architecture training procedure with a margin focused loss (Margin-MSE), that adapts knowledge distillation to the varying score output distributions of different BERT and non-BERT ranking architectures. We apply the teachable information as additional fine-grained labels to existing training triples of the MSMARCO-Passage collection. We evaluate our procedure of distilling knowledge from state-of-the-art concatenated BERT models to four different efficient architectures (TK, ColBERT, PreTT, and a BERT CLS dot product model). We show that across our evaluated architectures our Margin-MSE knowledge distillation significantly improves their effectiveness without compromising their efficiency. To benefit the community, we publish the costly teacher-score training files in a ready-to-use package.




# 13. A Characterization of the COVID-19 Pandemic Impact on a Mobile Network  Operator Traffic

Andra Lutu, Diego Perino, Marcelo Bagnulo, Enrique Frias-Martinez, Javad Khangosstar

- retweets: 42, favorites: 30 (10/08/2020 10:09:29)

- links: [abs](https://arxiv.org/abs/2010.02781) | [pdf](https://arxiv.org/pdf/2010.02781)
- [cs.NI](https://arxiv.org/list/cs.NI/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

During early 2020, the SARS-CoV-2 virus rapidly spread worldwide, forcing many governments to impose strict lockdown measures to tackle the pandemic. This significantly changed people's mobility and habits, subsequently impacting how they use telecommunication networks. In this paper, we investigate the effects of the COVID-19 emergency on a UK Mobile Network Operator (MNO). We quantify the changes in users' mobility and investigate how this impacted the cellular network usage and performance. Our analysis spans from the entire country to specific regions, and geodemographic area clusters. We also provide a detailed analysis for London. Our findings bring insights at different geo-temporal granularity on the status of the cellular network, from the decrease in data traffic volume in the cellular network and lower load on the radio network, counterposed to a surge in the conversational voice traffic volume.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper about the impact of the COVID-19 crisis on the mobile traffic of <a href="https://twitter.com/O2?ref_src=twsrc%5Etfw">@O2</a> in the UK will appear in <a href="https://twitter.com/hashtag/IMC2020?src=hash&amp;ref_src=twsrc%5Etfw">#IMC2020</a> later this month, now open on arxiv: <a href="https://t.co/Z8fo8S6pvv">https://t.co/Z8fo8S6pvv</a> <a href="https://twitter.com/Diego_Perino?ref_src=twsrc%5Etfw">@Diego_Perino</a> <a href="https://twitter.com/Khangosstar?ref_src=twsrc%5Etfw">@Khangosstar</a> <a href="https://twitter.com/TEFresearch?ref_src=twsrc%5Etfw">@TEFresearch</a>  <a href="https://twitter.com/MSCActions?ref_src=twsrc%5Etfw">@MSCActions</a> <a href="https://t.co/JARop4QMUR">pic.twitter.com/JARop4QMUR</a></p>&mdash; Andra Lutu (@dadalusa) <a href="https://twitter.com/dadalusa/status/1313763364129120257?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Efficient One-Pass End-to-End Entity Linking for Questions

Belinda Z. Li, Sewon Min, Srinivasan Iyer, Yashar Mehdad, Wen-tau Yih

- retweets: 30, favorites: 36 (10/08/2020 10:09:29)

- links: [abs](https://arxiv.org/abs/2010.02413) | [pdf](https://arxiv.org/pdf/2010.02413)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We present ELQ, a fast end-to-end entity linking model for questions, which uses a biencoder to jointly perform mention detection and linking in one pass. Evaluated on WebQSP and GraphQuestions with extended annotations that cover multiple entities per question, ELQ outperforms the previous state of the art by a large margin of +12.7% and +19.6% F1, respectively. With a very fast inference time (1.57 examples/s on a single CPU), ELQ can be useful for downstream question answering systems. In a proof-of-concept experiment, we demonstrate that using ELQ significantly improves the downstream QA performance of GraphRetriever (arXiv:1911.03868). Code and data available at https://github.com/facebookresearch/BLINK/tree/master/elq

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Presenting ELQ, a fast entity linker for questions!<br>Joint work with <a href="https://twitter.com/sewon__min?ref_src=twsrc%5Etfw">@sewon__min</a> <a href="https://twitter.com/sriniiyer88?ref_src=twsrc%5Etfw">@sriniiyer88</a> <a href="https://twitter.com/YasharMehdad?ref_src=twsrc%5Etfw">@YasharMehdad</a> <a href="https://twitter.com/scottyih?ref_src=twsrc%5Etfw">@scottyih</a> , to appear in <a href="https://twitter.com/emnlp2020?ref_src=twsrc%5Etfw">@emnlp2020</a> <br><br>Paper: <a href="https://t.co/ihGJoA1xRe">https://t.co/ihGJoA1xRe</a><br>Code (integrated with BLINK!): <a href="https://t.co/cP6U63InGE">https://t.co/cP6U63InGE</a><br><br>More details in thread 1/6 <a href="https://t.co/wMdkMZQxM2">pic.twitter.com/wMdkMZQxM2</a></p>&mdash; Belinda Li (@belindazli) <a href="https://twitter.com/belindazli/status/1313888478325690369?ref_src=twsrc%5Etfw">October 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. KGPT: Knowledge-Grounded Pre-Training for Data-to-Text Generation

Wenhu Chen, Yu Su, Xifeng Yan, William Yang Wang

- retweets: 24, favorites: 31 (10/08/2020 10:09:29)

- links: [abs](https://arxiv.org/abs/2010.02307) | [pdf](https://arxiv.org/pdf/2010.02307)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Data-to-text generation has recently attracted substantial interests due to its wide applications. Existing methods have shown impressive performance on an array of tasks. However, they rely on a significant amount of labeled data for each task, which is costly to acquire and thus limits their application to new tasks and domains. In this paper, we propose to leverage pre-training and transfer learning to address this issue. We propose a knowledge-grounded pre-training (KGPT), which consists of two parts, 1) a general knowledge-grounded generation model to generate knowledge-enriched text. 2) a pre-training paradigm on a massive knowledge-grounded text corpus crawled from the web. The pre-trained model can be fine-tuned on various data-to-text generation tasks to generate task-specific text. We adopt three settings, namely fully-supervised, zero-shot, few-shot to evaluate its effectiveness. Under the fully-supervised setting, our model can achieve remarkable gains over the known baselines. Under zero-shot setting, our model without seeing any examples achieves over 30 ROUGE-L on WebNLG while all other baselines fail. Under the few-shot setting, our model only needs about one-fifteenth as many labeled examples to achieve the same level of performance as baseline models. These experiments consistently prove the strong generalization ability of our proposed framework https://github.com/wenhuchen/KGPT.



