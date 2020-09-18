---
title: Hot Papers 2020-09-17
date: 2020-09-19T02:04:31.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-17"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-17"
socialImage: "/media/flying-marine.jpg"

---

# 1. Layered Neural Rendering for Retiming People in Video

Erika Lu, Forrester Cole, Tali Dekel, Weidi Xie, Andrew Zisserman, David Salesin, William T. Freeman, Michael Rubinstein

- retweets: 6496, favorites: 160 (09/19/2020 02:04:31)

- links: [abs](https://arxiv.org/abs/2009.07833) | [pdf](https://arxiv.org/pdf/2009.07833)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a method for retiming people in an ordinary, natural video---manipulating and editing the time in which different motions of individuals in the video occur. We can temporally align different motions, change the speed of certain actions (speeding up/slowing down, or entirely "freezing" people), or "erase" selected people from the video altogether. We achieve these effects computationally via a dedicated learning-based layered video representation, where each frame in the video is decomposed into separate RGBA layers, representing the appearance of different people in the video. A key property of our model is that it not only disentangles the direct motions of each person in the input video, but also correlates each person automatically with the scene changes they generate---e.g., shadows, reflections, and motion of loose clothing. The layers can be individually retimed and recombined into a new video, allowing us to achieve realistic, high-quality renderings of retiming effects for real-world videos depicting complex actions and involving multiple individuals, including dancing, trampoline jumping, or group running.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">動画内の一部分だけの人間の時間の流れを操作して、違和感なく新しい動画にする研究<br>一部の人だけの時間を止めたり、逆に時間の流れを早くしたりと見ていて面白い<br>Layered Neural Rendering for Retiming People in Video<a href="https://t.co/bQA1l62ca5">https://t.co/bQA1l62ca5</a><br> <a href="https://t.co/J7njrQml5o">pic.twitter.com/J7njrQml5o</a></p>&mdash; えるエル (@ImAI_Eruel) <a href="https://twitter.com/ImAI_Eruel/status/1306574201197793281?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Layered Neural Rendering for Retiming People in Video<br>pdf: <a href="https://t.co/NlRaJ5rElQ">https://t.co/NlRaJ5rElQ</a><br>abs: <a href="https://t.co/RPk6zluNOb">https://t.co/RPk6zluNOb</a><br>project page: <a href="https://t.co/BNuXnSsQck">https://t.co/BNuXnSsQck</a> <a href="https://t.co/rbFpAjZbxQ">pic.twitter.com/rbFpAjZbxQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306392510642749440?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Path Planning using Neural A* Search

Ryo Yonetani, Tatsunori Taniai, Mohammadamin Barekatain, Mai Nishimura, Asako Kanezaki

- retweets: 760, favorites: 148 (09/19/2020 02:04:31)

- links: [abs](https://arxiv.org/abs/2009.07476) | [pdf](https://arxiv.org/pdf/2009.07476)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We present Neural A*, a novel data-driven search algorithm for path planning problems. Although data-driven planning has received much attention in recent years, little work has focused on how search-based methods can learn from demonstrations to plan better. In this work, we reformulate a canonical A* search algorithm to be differentiable and couple it with a convolutional encoder to form an end-to-end trainable neural network planner. Neural A* solves a path planning problem by (1) encoding a visual representation of the problem to estimate a movement cost map and (2) performing the A* search on the cost map to output a solution path. By minimizing the difference between the search results and ground-truth paths in demonstrations, the encoder learns to capture a variety of visual planning cues in input images, such as shapes of dead-end obstacles, bypasses, and shortcuts, which makes estimated cost maps informative. Our extensive experiments confirmed that Neural A* (a) outperformed state-of-the-art data-driven planners in terms of the search optimality and efficiency trade-off and (b) predicted realistic pedestrian paths by directly performing a search on raw image inputs.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">arXivに新しい論文をアップロードしました: &quot;Path Planning using Neural A* Search&quot; <a href="https://t.co/4e0Sxvtfrn">https://t.co/4e0Sxvtfrn</a> 米谷・谷合・西村（オムロンサイニックエックス）、Amin（インターン）、技術アドバイザの金崎先生（東工大）による研究で、微分可能なA*探索による経路計画の話です。</p>&mdash; Ryo Yonetani (@RYonetani) <a href="https://twitter.com/RYonetani/status/1306394732310921216?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">arXivに新しい論文をアップロードしました: &quot;Path Planning using Neural A* Search&quot; <a href="https://t.co/4e0Sxvtfrn">https://t.co/4e0Sxvtfrn</a> 米谷・谷合・西村（オムロンサイニックエックス）、Amin（インターン）、技術アドバイザの金崎先生（東工大）による研究で、微分可能なA*探索による経路計画の話です。</p>&mdash; Ryo Yonetani (@RYonetani) <a href="https://twitter.com/RYonetani/status/1306394732310921216?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Detectability of hierarchical communities in networks

Leto Peel, Michael T. Schaub

- retweets: 218, favorites: 56 (09/19/2020 02:04:32)

- links: [abs](https://arxiv.org/abs/2009.07525) | [pdf](https://arxiv.org/pdf/2009.07525)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We study the problem of recovering a planted hierarchy of partitions in a network. The detectability of a single planted partition has previously been analysed in detail and a phase transition has been identified below which the partition cannot be detected. Here we show that, in the hierarchical setting, there exist additional phases in which the presence of multiple consistent partitions can either help or hinder detection. Accordingly, the detectability limit for non-hierarchical partitions typically provides insufficient information about the detectability of the complete hierarchical structure, as we highlight with several constructive examples.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Out now on the arxiv, the first outputs (yes plural!) from the Schaub &amp; Peel collaboration: hierarchical communities in networks! <br>These have been some time in the making and we&#39;d love to hear your feedback!<a href="https://t.co/WpysTjMA5F">https://t.co/WpysTjMA5F</a><a href="https://t.co/7mE6Hl1z3x">https://t.co/7mE6Hl1z3x</a><br><br>A tree 🌲 1/n</p>&mdash; Leto Peel (@PiratePeel) <a href="https://twitter.com/PiratePeel/status/1306699724683120640?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Grounded Adaptation for Zero-shot Executable Semantic Parsing

Victor Zhong, Mike Lewis, Sida I. Wang, Luke Zettlemoyer

- retweets: 192, favorites: 64 (09/19/2020 02:04:32)

- links: [abs](https://arxiv.org/abs/2009.07396) | [pdf](https://arxiv.org/pdf/2009.07396)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose Grounded Adaptation for Zero-shot Executable Semantic Parsing (GAZP) to adapt an existing semantic parser to new environments (e.g. new database schemas). GAZP combines a forward semantic parser with a backward utterance generator to synthesize data (e.g. utterances and SQL queries) in the new environment, then selects cycle-consistent examples to adapt the parser. Unlike data-augmentation, which typically synthesizes unverified examples in the training environment, GAZP synthesizes examples in the new environment whose input-output consistency are verified. On the Spider, Sparc, and CoSQL zero-shot semantic parsing tasks, GAZP improves logical form and execution accuracy of the baseline parser. Our analyses show that GAZP outperforms data-augmentation in the training environment, performance increases with the amount of GAZP-synthesized data, and cycle-consistency is central to successful adaptation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a> paper Grounded Adaptation for Zero-shot Semantic Parsing proposes GAZP, a framework for zero-shot language-to-SQL parsing by synthesizing data in new DBs after reading their schema. 💡<br><br>Paper 📰 <a href="https://t.co/lGxja3Z2L2">https://t.co/lGxja3Z2L2</a><br>Thread 👇1/9<a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://t.co/p4qwrCyec4">pic.twitter.com/p4qwrCyec4</a></p>&mdash; Victor Zhong (@hllo_wrld) <a href="https://twitter.com/hllo_wrld/status/1306623010275696641?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. BOP Challenge 2020 on 6D Object Localization

Tomas Hodan, Martin Sundermeyer, Bertram Drost, Yann Labbe, Eric Brachmann, Frank Michel, Carsten Rother, Jiri Matas

- retweets: 112, favorites: 36 (09/19/2020 02:04:32)

- links: [abs](https://arxiv.org/abs/2009.07378) | [pdf](https://arxiv.org/pdf/2009.07378)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

This paper presents the evaluation methodology, datasets, and results of the BOP Challenge 2020, the third in a series of public competitions organized with the goal to capture the status quo in the field of 6D object pose estimation from an RGB-D image. In 2020, to reduce the domain gap between synthetic training and real test RGB images, the participants were provided 350K photorealistic trainining images generated by BlenderProc4BOP, a~new open-source and light-weight physically-based renderer (PBR) and procedural data generator. Methods based on deep neural networks have finally caught up with methods based on point pair features, which were dominating previous editions of the challenge. Although the top-performing methods rely on RGB-D image channels, strong results were achieved when only RGB channels were used at both training and test time -- out of 26 evaluated methods, the third method was trained on RGB channels of PBR and real images, while the fifth was trained on PBR images only. Strong data augmentation was identified as a key component of the top-performing CosyPose method, and the photorealism of PBR images was demonstrated effective despite the augmentation. The online evaluation system stays open and is available at the project website: bop.felk.cvut.cz.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hot off the arXiv press: Results of the <a href="https://twitter.com/hashtag/BOPchallenge?src=hash&amp;ref_src=twsrc%5Etfw">#BOPchallenge</a> of this years <a href="https://twitter.com/hashtag/ECCV20?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV20</a> in handy paper format:<a href="https://t.co/6I11ryhIT7">https://t.co/6I11ryhIT7</a><br>What&#39;s the best 6D object pose estimator in town? Do we still need depth for high accuracy? Thanks <a href="https://twitter.com/tomhodan?ref_src=twsrc%5Etfw">@tomhodan</a> and <a href="https://twitter.com/ma_sundermeyer?ref_src=twsrc%5Etfw">@ma_sundermeyer</a> for putting it together.</p>&mdash; Eric Brachmann (@eric_brachmann) <a href="https://twitter.com/eric_brachmann/status/1306518806009872385?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A Human-Computer Duet System for Music Performance

Yuen-Jen Lin, Hsuan-Kai Kao, Yih-Chih Tseng, Ming Tsai, Li Su

- retweets: 110, favorites: 34 (09/19/2020 02:04:32)

- links: [abs](https://arxiv.org/abs/2009.07816) | [pdf](https://arxiv.org/pdf/2009.07816)
- [cs.MM](https://arxiv.org/list/cs.MM/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Virtual musicians have become a remarkable phenomenon in the contemporary multimedia arts. However, most of the virtual musicians nowadays have not been endowed with abilities to create their own behaviors, or to perform music with human musicians. In this paper, we firstly create a virtual violinist, who can collaborate with a human pianist to perform chamber music automatically without any intervention. The system incorporates the techniques from various fields, including real-time music tracking, pose estimation, and body movement generation. In our system, the virtual musician's behavior is generated based on the given music audio alone, and such a system results in a low-cost, efficient and scalable way to produce human and virtual musicians' co-performance. The proposed system has been validated in public concerts. Objective quality assessment approaches and possible ways to systematically improve the system are also discussed.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Human-Computer Duet System for Music Performance<br>pdf: <a href="https://t.co/rPpEMjoxAB">https://t.co/rPpEMjoxAB</a><br>abs: <a href="https://t.co/52RGHcTwLj">https://t.co/52RGHcTwLj</a><br>project page: <a href="https://t.co/yfNZtGeKVP">https://t.co/yfNZtGeKVP</a> <a href="https://t.co/SR9m8uSZpq">pic.twitter.com/SR9m8uSZpq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306466736158527488?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Generative Language-Grounded Policy in Vision-and-Language Navigation  with Bayes' Rule

Shuhei Kurita, Kyunghyun Cho

- retweets: 53, favorites: 45 (09/19/2020 02:04:32)

- links: [abs](https://arxiv.org/abs/2009.07783) | [pdf](https://arxiv.org/pdf/2009.07783)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Vision-and-language navigation (VLN) is a task in which an agent is embodied in a realistic 3D environment and follows an instruction to reach the goal node. While most of the previous studies have built and investigated a discriminative approach, we notice that there are in fact two possible approaches to building such a VLN agent: discriminative and generative. In this paper, we design and investigate a generative language-grounded policy which computes the distribution over all possible instructions given action and the transition history. In experiments, we show that the proposed generative approach outperforms the discriminative approach in the Room-2-Room (R2R) dataset, especially in the unseen environments. We further show that the combination of the generative and discriminative policies achieves close to the state-of-the art results in the R2R dataset, demonstrating that the generative and discriminative policies capture the different aspects of VLN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I have posted the latest manuscript for vision-and-language navigation (VLN) on arXiv with Kyunghyun Cho <a href="https://twitter.com/kchonyc?ref_src=twsrc%5Etfw">@kchonyc</a> !<br>&quot;Generative Language-Grounded Policy in Vision-and-Language Navigation with Bayes&#39; Rule&quot; (1/3)<a href="https://t.co/oO9UPY39PM">https://t.co/oO9UPY39PM</a></p>&mdash; Shuhei Kurita (@ShuheiKurita) <a href="https://twitter.com/ShuheiKurita/status/1306530719674507264?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Automated Source Code Generation and Auto-completion Using Deep  Learning: Comparing and Discussing Current Language-Model-Related Approaches

Juan Cruz-Benito, Sanjay Vishwakarma, Francisco Martin-Fernandez, Ismael Faro

- retweets: 78, favorites: 17 (09/19/2020 02:04:32)

- links: [abs](https://arxiv.org/abs/2009.07740) | [pdf](https://arxiv.org/pdf/2009.07740)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.PL](https://arxiv.org/list/cs.PL/recent) | [cs.SE](https://arxiv.org/list/cs.SE/recent)

In recent years, the use of deep learning in language models, text auto-completion, and text generation has made tremendous progress and gained much attention from the research community. Some products and research projects claim that they can generate text that can be interpreted as human-writing, enabling new possibilities in many application areas. Among the different areas related to language processing, one of the most notable in applying this type of modeling is the processing of programming languages. For years, the Machine Learning community has been researching in this Big Code area, pursuing goals like applying different approaches to auto-complete generate, fix, or evaluate code programmed by humans. One of the approaches followed in recent years to pursue these goals is the use of Deep-Learning-enabled language models. Considering the increasing popularity of that approach, we detected a lack of empirical papers that compare different methods and deep learning architectures to create and use language models based on programming code. In this paper, we compare different neural network (NN) architectures like AWD-LSTMs, AWD-QRNNs, and Transformer, while using transfer learning, and different tokenizations to see how they behave in building language models using a Python dataset for code generation and filling mask tasks. Considering the results, we discuss the different strengths and weaknesses of each approach and technique and what lacks do we find to evaluate the language models or apply them in a real programming context while including humans-in-the-loop.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper from our <a href="https://twitter.com/hashtag/IBMQuantum?src=hash&amp;ref_src=twsrc%5Etfw">#IBMQuantum</a> Cloud Team c/ Sanjay Vishwakarma, <a href="https://twitter.com/pacomartinfdez?ref_src=twsrc%5Etfw">@pacomartinfdez</a>, and <a href="https://twitter.com/ismaelfaro?ref_src=twsrc%5Etfw">@ismaelfaro</a><a href="https://t.co/uvED9GBpIj">https://t.co/uvED9GBpIj</a></p>&mdash; Juan Cruz-Benito (😷) (@_juancb) <a href="https://twitter.com/_juancb/status/1306541068901588993?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. GenASM: A High-Performance, Low-Power Approximate String Matching  Acceleration Framework for Genome Sequence Analysis

Damla Senol Cali, Gurpreet S. Kalsi, Zülal Bingöl, Can Firtina, Lavanya Subramanian, Jeremie S. Kim, Rachata Ausavarungnirun, Mohammed Alser, Juan Gomez-Luna, Amirali Boroumand, Anant Nori, Allison Scibisz, Sreenivas Subramoney, Can Alkan, Saugata Ghose, Onur Mutlu

- retweets: 54, favorites: 39 (09/19/2020 02:04:32)

- links: [abs](https://arxiv.org/abs/2009.07692) | [pdf](https://arxiv.org/pdf/2009.07692)
- [cs.AR](https://arxiv.org/list/cs.AR/recent) | [q-bio.GN](https://arxiv.org/list/q-bio.GN/recent)

Genome sequence analysis has enabled significant advancements in medical and scientific areas such as personalized medicine, outbreak tracing, and the understanding of evolution. Unfortunately, it is currently bottlenecked by the computational power and memory bandwidth limitations of existing systems, as many of the steps in genome sequence analysis must process a large amount of data. A major contributor to this bottleneck is approximate string matching (ASM).   We propose GenASM, the first ASM acceleration framework for genome sequence analysis. We modify the underlying ASM algorithm (Bitap) to significantly increase its parallelism and reduce its memory footprint, and we design the first hardware accelerator for Bitap. Our hardware accelerator consists of specialized compute units and on-chip SRAMs that are designed to match the rate of computation with memory capacity and bandwidth.   We demonstrate that GenASM is a flexible, high-performance, and low-power framework, which provides significant performance and power benefits for three different use cases in genome sequence analysis: 1) GenASM accelerates read alignment for both long reads and short reads. For long reads, GenASM outperforms state-of-the-art software and hardware accelerators by 116x and 3.9x, respectively, while consuming 37x and 2.7x less power. For short reads, GenASM outperforms state-of-the-art software and hardware accelerators by 111x and 1.9x. 2) GenASM accelerates pre-alignment filtering for short reads, with 3.7x the performance of a state-of-the-art pre-alignment filter, while consuming 1.7x less power and significantly improving the filtering accuracy. 3) GenASM accelerates edit distance calculation, with 22-12501x and 9.3-400x speedups over the state-of-the-art software library and FPGA-based accelerator, respectively, while consuming 548-582x and 67x less power.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/MICRO20?src=hash&amp;ref_src=twsrc%5Etfw">#MICRO20</a> preprint &quot;GenASM: A High-Performance, Low-Power Approximate String Matching Acceleration Framework for Genome Sequence Analysis&quot; is now online:<a href="https://t.co/IMnBhLJbzM">https://t.co/IMnBhLJbzM</a><a href="https://twitter.com/MicroArchConf?ref_src=twsrc%5Etfw">@MicroArchConf</a> <a href="https://twitter.com/SAFARI_ETH_CMU?ref_src=twsrc%5Etfw">@SAFARI_ETH_CMU</a> <a href="https://twitter.com/CarnegieMellon?ref_src=twsrc%5Etfw">@CarnegieMellon</a> <a href="https://twitter.com/ETH_en?ref_src=twsrc%5Etfw">@ETH_en</a> <a href="https://twitter.com/intel?ref_src=twsrc%5Etfw">@intel</a> <a href="https://twitter.com/BilkentCompGen?ref_src=twsrc%5Etfw">@BilkentCompGen</a> <a href="https://t.co/Oxez1yBdYV">pic.twitter.com/Oxez1yBdYV</a></p>&mdash; Damla Şenol Çalı (@damlasenolcali) <a href="https://twitter.com/damlasenolcali/status/1306408023947128834?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. RDF2Vec Light -- A Lightweight Approach for Knowledge Graph Embeddings

Jan Portisch, Michael Hladik, Heiko Paulheim

- retweets: 42, favorites: 44 (09/19/2020 02:04:33)

- links: [abs](https://arxiv.org/abs/2009.07659) | [pdf](https://arxiv.org/pdf/2009.07659)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Knowledge graph embedding approaches represent nodes and edges of graphs as mathematical vectors. Current approaches focus on embedding complete knowledge graphs, i.e. all nodes and edges. This leads to very high computational requirements on large graphs such as DBpedia or Wikidata. However, for most downstream application scenarios, only a small subset of concepts is of actual interest. In this paper, we present RDF2Vec Light, a lightweight embedding approach based on RDF2Vec which generates vectors for only a subset of entities. To that end, RDF2Vec Light only traverses and processes a subgraph of the knowledge graph. Our method allows the application of embeddings of very large knowledge graphs in scenarios where such embeddings were not possible before due to a significantly lower runtime and significantly reduced hardware requirements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Training <a href="https://twitter.com/hashtag/embedding?src=hash&amp;ref_src=twsrc%5Etfw">#embedding</a> vectors on a large <a href="https://twitter.com/hashtag/knowledgegraph?src=hash&amp;ref_src=twsrc%5Etfw">#knowledgegraph</a> takes time, but many tasks only require vectors for very few entities. For those cases, the <a href="https://twitter.com/hashtag/rdf2vec?src=hash&amp;ref_src=twsrc%5Etfw">#rdf2vec</a> variant RDF2vec Light is a competitive alternative. <a href="https://t.co/oxpywCb3Xg">https://t.co/oxpywCb3Xg</a> <a href="https://t.co/QufqeQAyiL">https://t.co/QufqeQAyiL</a> <a href="https://twitter.com/JanPortisch?ref_src=twsrc%5Etfw">@JanPortisch</a> <a href="https://twitter.com/dwsunima?ref_src=twsrc%5Etfw">@dwsunima</a></p>&mdash; Heiko Paulheim (@heikopaulheim) <a href="https://twitter.com/heikopaulheim/status/1306475581186375688?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. EfficientNet-eLite: Extremely Lightweight and Efficient CNN Models for  Edge Devices by Network Candidate Search

Ching-Chen Wang, Ching-Te Chiu, Jheng-Yi Chang

- retweets: 40, favorites: 26 (09/19/2020 02:04:33)

- links: [abs](https://arxiv.org/abs/2009.07409) | [pdf](https://arxiv.org/pdf/2009.07409)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Embedding Convolutional Neural Network (CNN) into edge devices for inference is a very challenging task because such lightweight hardware is not born to handle this heavyweight software, which is the common overhead from the modern state-of-the-art CNN models. In this paper, targeting at reducing the overhead with trading the accuracy as less as possible, we propose a novel of Network Candidate Search (NCS), an alternative way to study the trade-off between the resource usage and the performance through grouping concepts and elimination tournament. Besides, NCS can also be generalized across any neural network. In our experiment, we collect candidate CNN models from EfficientNet-B0 to be scaled down in varied way through width, depth, input resolution and compound scaling down, applying NCS to research the scaling-down trade-off. Meanwhile, a family of extremely lightweight EfficientNet is obtained, called EfficientNet-eLite. For further embracing the CNN edge application with Application-Specific Integrated Circuit (ASIC), we adjust the architectures of EfficientNet-eLite to build the more hardware-friendly version, EfficientNet-HF. Evaluation on ImageNet dataset, both proposed EfficientNet-eLite and EfficientNet-HF present better parameter usage and accuracy than the previous start-of-the-art CNNs. Particularly, the smallest member of EfficientNet-eLite is more lightweight than the best and smallest existing MnasNet with 1.46x less parameters and 0.56% higher accuracy. Code is available at https://github.com/Ching-Chen-Wang/EfficientNet-eLite




# 12. ChoreoNet: Towards Music to Dance Synthesis with Choreographic Action  Unit

Zijie Ye, Haozhe Wu, Jia Jia, Yaohua Bu, Wei Chen, Fanbo Meng, Yanfeng Wang

- retweets: 42, favorites: 23 (09/19/2020 02:04:33)

- links: [abs](https://arxiv.org/abs/2009.07637) | [pdf](https://arxiv.org/pdf/2009.07637)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Dance and music are two highly correlated artistic forms. Synthesizing dance motions has attracted much attention recently. Most previous works conduct music-to-dance synthesis via directly music to human skeleton keypoints mapping. Meanwhile, human choreographers design dance motions from music in a two-stage manner: they firstly devise multiple choreographic dance units (CAUs), each with a series of dance motions, and then arrange the CAU sequence according to the rhythm, melody and emotion of the music. Inspired by these, we systematically study such two-stage choreography approach and construct a dataset to incorporate such choreography knowledge. Based on the constructed dataset, we design a two-stage music-to-dance synthesis framework ChoreoNet to imitate human choreography procedure. Our framework firstly devises a CAU prediction model to learn the mapping relationship between music and CAU sequences. Afterwards, we devise a spatial-temporal inpainting model to convert the CAU sequence into continuous dance motions. Experimental results demonstrate that the proposed ChoreoNet outperforms baseline methods (0.622 in terms of CAU BLEU score and 1.59 in terms of user study score).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ChoreoNet: Towards Music to Dance Synthesis with Choreographic Action Unit<br>pdf: <a href="https://t.co/76p6zaxMkk">https://t.co/76p6zaxMkk</a><br>abs: <a href="https://t.co/h2nlND3GKx">https://t.co/h2nlND3GKx</a> <a href="https://t.co/aO0L5cL2Ub">pic.twitter.com/aO0L5cL2Ub</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306404454141505537?ref_src=twsrc%5Etfw">September 17, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. GLUCOSE: GeneraLized and COntextualized Story Explanations

Nasrin Mostafazadeh, Aditya Kalyanpur, Lori Moon, David Buchanan, Lauren Berkowitz, Or Biran, Jennifer Chu-Carroll

- retweets: 45, favorites: 20 (09/19/2020 02:04:33)

- links: [abs](https://arxiv.org/abs/2009.07758) | [pdf](https://arxiv.org/pdf/2009.07758)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

When humans read or listen, they make implicit commonsense inferences that frame their understanding of what happened and why. As a step toward AI systems that can build similar mental models, we introduce GLUCOSE, a large-scale dataset of implicit commonsense causal knowledge, encoded as causal mini-theories about the world, each grounded in a narrative context. To construct GLUCOSE, we drew on cognitive psychology to identify ten dimensions of causal explanation, focusing on events, states, motivations, and emotions. Each GLUCOSE entry includes a story-specific causal statement paired with an inference rule generalized from the statement. This paper details two concrete contributions: First, we present our platform for effectively crowdsourcing GLUCOSE data at scale, which uses semi-structured templates to elicit causal explanations. Using this platform, we collected 440K specific statements and general rules that capture implicit commonsense knowledge about everyday situations. Second, we show that existing knowledge resources and pretrained language models do not include or readily predict GLUCOSE's rich inferential content. However, when state-of-the-art neural models are trained on this knowledge, they can start to make commonsense inferences on unseen stories that match humans' mental models.



