---
title: Hot Papers 2021-07-15
date: 2021-07-16T07:00:57.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-15"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-15"
socialImage: "/media/flying-marine.jpg"

---

# 1. Deduplicating Training Data Makes Language Models Better

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini

- retweets: 3793, favorites: 479 (07/16/2021 07:00:57)

- links: [abs](https://arxiv.org/abs/2107.06499) | [pdf](https://arxiv.org/pdf/2107.06499)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We find that existing language modeling datasets contain many near-duplicate examples and long repetitive substrings. As a result, over 1% of the unprompted output of language models trained on these datasets is copied verbatim from the training data. We develop two tools that allow us to deduplicate training datasets -- for example removing from C4 a single 61 word English sentence that is repeated over 60,000 times. Deduplication allows us to train models that emit memorized text ten times less frequently and require fewer train steps to achieve the same or better accuracy. We can also reduce train-test overlap, which affects over 4% of the validation set of standard datasets, thus allowing for more accurate evaluation. We release code for reproducing our work and performing dataset deduplication at https://github.com/google-research/deduplicate-text-datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deduplicating Training Data Makes Language Models Better<br><br>Finds that deduplication allows us to train models that emit memorized text ten times less frequently and require fewer train steps to achieve the same or better accuracy.<a href="https://t.co/vWjKaGXFfx">https://t.co/vWjKaGXFfx</a> <a href="https://t.co/sE37pw2T1Z">pic.twitter.com/sE37pw2T1Z</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1415472192100339714?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Data duplication is serious business! <br><br>3% of documents in the large language dataset, C4, have near-duplicates. <br><br>Deduplication reduces model memorization while training faster and without reducing accuracy.<br><br>Paper:  <a href="https://t.co/ENRVYgjnOw">https://t.co/ENRVYgjnOw</a><br>Code: coming soon! <br><br>🧵⬇️ (1/9)</p>&mdash; Katherine Lee (@katherine1ee) <a href="https://twitter.com/katherine1ee/status/1415496898241339400?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deduplicating Training Data Makes Language Models Better<br>pdf: <a href="https://t.co/w8J8NZ5v7t">https://t.co/w8J8NZ5v7t</a><br>abs: <a href="https://t.co/4Woo78QjST">https://t.co/4Woo78QjST</a><br>Deduplication allows us to train models that emit memorized text ten times less frequently and require fewer train steps to achieve the same or better accuracy <a href="https://t.co/sMMud34rj7">pic.twitter.com/sMMud34rj7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415471595091595264?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Deep Neural Networks are Surprisingly Reversible: A Baseline for  Zero-Shot Inversion

Xin Dong, Hongxu Yin, Jose M. Alvarez, Jan Kautz, Pavlo Molchanov

- retweets: 1184, favorites: 175 (07/16/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.06304) | [pdf](https://arxiv.org/pdf/2107.06304)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Understanding the behavior and vulnerability of pre-trained deep neural networks (DNNs) can help to improve them. Analysis can be performed via reversing the network's flow to generate inputs from internal representations. Most existing work relies on priors or data-intensive optimization to invert a model, yet struggles to scale to deep architectures and complex datasets. This paper presents a zero-shot direct model inversion framework that recovers the input to the trained model given only the internal representation. The crux of our method is to inverse the DNN in a divide-and-conquer manner while re-syncing the inverted layers via cycle-consistency guidance with the help of synthesized data. As a result, we obtain a single feed-forward model capable of inversion with a single forward pass without seeing any real data of the original task. With the proposed approach, we scale zero-shot direct inversion to deep architectures and complex datasets. We empirically show that modern classification models on ImageNet can, surprisingly, be inverted, allowing an approximate recovery of the original 224x224px images from a representation after more than 20 layers. Moreover, inversion of generators in GANs unveils latent code of a given synthesized face image at 128x128px, which can even, in turn, improve defective synthesized images from GANs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Neural Networks are Surprisingly Reversible: A Baseline for Zero-Shot Inversion<br>pdf: <a href="https://t.co/rETZDP6rW2">https://t.co/rETZDP6rW2</a><br>abs: <a href="https://t.co/037Rx1nB5M">https://t.co/037Rx1nB5M</a><br><br>a zero-shot direct model inversion framework that recovers the input to the trained model given only the internal representation <a href="https://t.co/FVUQahDlu5">pic.twitter.com/FVUQahDlu5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415479327341350917?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Tortured phrases: A dubious writing style emerging in science. Evidence  of critical issues affecting established journals

Guillaume Cabanac, Cyril Labbé, Alexander Magazinov

- retweets: 702, favorites: 51 (07/16/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.06751) | [pdf](https://arxiv.org/pdf/2107.06751)
- [cs.DL](https://arxiv.org/list/cs.DL/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

Probabilistic text generators have been used to produce fake scientific papers for more than a decade. Such nonsensical papers are easily detected by both human and machine. Now more complex AI-powered generation techniques produce texts indistinguishable from that of humans and the generation of scientific texts from a few keywords has been documented. Our study introduces the concept of tortured phrases: unexpected weird phrases in lieu of established ones, such as 'counterfeit consciousness' instead of 'artificial intelligence.' We combed the literature for tortured phrases and study one reputable journal where these concentrated en masse. Hypothesising the use of advanced language models we ran a detector on the abstracts of recent articles of this journal and on several control sets. The pairwise comparisons reveal a concentration of abstracts flagged as 'synthetic' in the journal. We also highlight irregularities in its operation, such as abrupt changes in editorial timelines. We substantiate our call for investigation by analysing several individual dubious articles, stressing questionable features: tortured writing style, citation of non-existent literature, and unacknowledged image reuse. Surprisingly, some websites offer to rewrite texts for free, generating gobbledegook full of tortured phrases. We believe some authors used rewritten texts to pad their manuscripts. We wish to raise the awareness on publications containing such questionable AI-generated or rewritten texts that passed (poor) peer review. Deception with synthetic texts threatens the integrity of the scientific literature.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Just out <a href="https://twitter.com/arxiv?ref_src=twsrc%5Etfw">@arxiv</a>: “Tortured phrases: A dubious writing style emerging in science. Evidence of critical issues affecting established journals” w/ Labbé &amp; Magazinov <a href="https://t.co/eh0YNO5T8v">https://t.co/eh0YNO5T8v</a> Probable systematic manipulation of the publication process of Microprocessors in Microsystems <a href="https://t.co/vrVQUi3TNn">pic.twitter.com/vrVQUi3TNn</a></p>&mdash; Guillaume Cabanac (@gcabanac) <a href="https://twitter.com/gcabanac/status/1415559427743174661?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Multi-Label Generalized Zero Shot Learning for the Classification of  Disease in Chest Radiographs

Nasir Hayat, Hazem Lashen, Farah E. Shamout

- retweets: 570, favorites: 150 (07/16/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.06563) | [pdf](https://arxiv.org/pdf/2107.06563)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Despite the success of deep neural networks in chest X-ray (CXR) diagnosis, supervised learning only allows the prediction of disease classes that were seen during training. At inference, these networks cannot predict an unseen disease class. Incorporating a new class requires the collection of labeled data, which is not a trivial task, especially for less frequently-occurring diseases. As a result, it becomes inconceivable to build a model that can diagnose all possible disease classes. Here, we propose a multi-label generalized zero shot learning (CXR-ML-GZSL) network that can simultaneously predict multiple seen and unseen diseases in CXR images. Given an input image, CXR-ML-GZSL learns a visual representation guided by the input's corresponding semantics extracted from a rich medical text corpus. Towards this ambitious goal, we propose to map both visual and semantic modalities to a latent feature space using a novel learning objective. The objective ensures that (i) the most relevant labels for the query image are ranked higher than irrelevant labels, (ii) the network learns a visual representation that is aligned with its semantics in the latent feature space, and (iii) the mapped semantics preserve their original inter-class representation. The network is end-to-end trainable and requires no independent pre-training for the offline feature extractor. Experiments on the NIH Chest X-ray dataset show that our network outperforms two strong baselines in terms of recall, precision, f1 score, and area under the receiver operating characteristic curve. Our code is publicly available at: https://github.com/nyuad-cai/CXR-ML-GZSL.git

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Are deep neural networks able to predict diseases they haven’t been trained on? Our new work accepted at <a href="https://twitter.com/mlforhc?ref_src=twsrc%5Etfw">@mlforhc</a> investigates this question via Multi-Label Generalized Zero Shot Learning (ML-GZSL) for chest X-rays (CXR): <a href="https://t.co/OGNSG8IuNW">https://t.co/OGNSG8IuNW</a> <br><br>Here’s a brief summary!<br><br>1/5</p>&mdash; Farah Shamout (@farahshamout) <a href="https://twitter.com/farahshamout/status/1415544266391179264?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Nowcasting transmission and suppression of the Delta variant of  SARS-CoV-2 in Australia

Sheryl L. Chang, Oliver M. Cliff, Mikhail Prokopenko

- retweets: 426, favorites: 67 (07/16/2021 07:00:58)

- links: [abs](https://arxiv.org/abs/2107.06617) | [pdf](https://arxiv.org/pdf/2107.06617)
- [q-bio.PE](https://arxiv.org/list/q-bio.PE/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent)

As of July 2021, there is a continuing outbreak of the B.1.617.2 (Delta) variant of SARS-CoV-2 in Sydney, Australia. The outbreak is of major concern as the Delta variant is estimated to have twice the reproductive number to previous variants that circulated in Australia in 2020, which is worsened by low levels of acquired immunity in the population. Using a re-calibrated agent-based model, we explored a feasible range of non-pharmaceutical interventions, in terms of both mitigation (case isolation, home quarantine) and suppression (school closures, social distancing). Our nowcasting modelling indicated that the level of social distancing currently attained in Sydney is inadequate for the outbreak control. A counter-factual analysis suggested that if 80% of agents comply with social distancing, then at least a month is needed for the new daily cases to reduce from their peak to below ten. A small reduction in social distancing compliance to 70% lengthens this period to over two months.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Model suggests we are not winning in Greater Sydney yet. Nowcasting transmission and suppression of the Delta variant of SARS-CoV-2 in Australia <a href="https://t.co/gPXl4Pxilr">https://t.co/gPXl4Pxilr</a></p>&mdash; MJA Editor in Chief (@MJA_Editor) <a href="https://twitter.com/MJA_Editor/status/1415552495011524613?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Differentiable Programming of Reaction-Diffusion Patterns

Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson

- retweets: 295, favorites: 76 (07/16/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.06862) | [pdf](https://arxiv.org/pdf/2107.06862)
- [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Reaction-Diffusion (RD) systems provide a computational framework that governs many pattern formation processes in nature. Current RD system design practices boil down to trial-and-error parameter search. We propose a differentiable optimization method for learning the RD system parameters to perform example-based texture synthesis on a 2D plane. We do this by representing the RD system as a variant of Neural Cellular Automata and using task-specific differentiable loss functions. RD systems generated by our method exhibit robust, non-trivial 'life-like' behavior.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Differentiable Programming of Reaction-Diffusion Patterns<br>pdf: <a href="https://t.co/WNpD7bzWyd">https://t.co/WNpD7bzWyd</a><br>project page: <a href="https://t.co/vaVaO9kuo0">https://t.co/vaVaO9kuo0</a><br><br>a differentiable optimization method for learning the RD system parameters to perform example-based texture synthesis on a 2D plane <a href="https://t.co/ksb4XF2duV">pic.twitter.com/ksb4XF2duV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415484921523671041?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. How Much Can CLIP Benefit Vision-and-Language Tasks?

Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, Kurt Keutzer

- retweets: 272, favorites: 95 (07/16/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.06383) | [pdf](https://arxiv.org/pdf/2107.06383)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Most existing Vision-and-Language (V&L) models rely on pre-trained visual encoders, using a relatively small set of manually-annotated data (as compared to web-crawled data), to perceive the visual world. However, it has been observed that large-scale pretraining usually can result in better generalization performance, e.g., CLIP (Contrastive Language-Image Pre-training), trained on a massive amount of image-caption pairs, has shown a strong zero-shot capability on various vision tasks. To further study the advantage brought by CLIP, we propose to use CLIP as the visual encoder in various V&L models in two typical scenarios: 1) plugging CLIP into task-specific fine-tuning; 2) combining CLIP with V&L pre-training and transferring to downstream tasks. We show that CLIP significantly outperforms widely-used visual encoders trained with in-domain annotated data, such as BottomUp-TopDown. We achieve competitive or better results on diverse V&L tasks, while establishing new state-of-the-art results on Visual Question Answering, Visual Entailment, and V&L Navigation tasks. We release our code at https://github.com/clip-vil/CLIP-ViL.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How Much Can CLIP Benefit Vision-and-Language Tasks?<br>pdf: <a href="https://t.co/JvlKycMcBj">https://t.co/JvlKycMcBj</a><br>github: <a href="https://t.co/ilqUdlozPw">https://t.co/ilqUdlozPw</a><br><br>competitive or better results on diverse V&amp;L tasks, while establishing new sota results on Visual Question Answering, Visual Entailment, and V&amp;L Navigation tasks <a href="https://t.co/1FgZf0eQxu">pic.twitter.com/1FgZf0eQxu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415470537229406210?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A Generalized Lottery Ticket Hypothesis

Ibrahim Alabdulmohsin, Larisa Markeeva, Daniel Keysers, Ilya Tolstikhin

- retweets: 266, favorites: 91 (07/16/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.06825) | [pdf](https://arxiv.org/pdf/2107.06825)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce a generalization to the lottery ticket hypothesis in which the notion of "sparsity" is relaxed by choosing an arbitrary basis in the space of parameters. We present evidence that the original results reported for the canonical basis continue to hold in this broader setting. We describe how structured pruning methods, including pruning units or factorizing fully-connected layers into products of low-rank matrices, can be cast as particular instances of this "generalized" lottery ticket hypothesis. The investigations reported here are preliminary and are provided to encourage further research along this direction.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Generalized Lottery Ticket Hypothesis<br>pdf: <a href="https://t.co/UTEOQ9jm4Y">https://t.co/UTEOQ9jm4Y</a><br>abs: <a href="https://t.co/zxvAGYcpWL">https://t.co/zxvAGYcpWL</a> <a href="https://t.co/FOiph4VEQO">pic.twitter.com/FOiph4VEQO</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415502940815638529?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Short preprint where we claim that the Lottery Ticket Hypothesis holds for *any* notion of sparsity. <br><br>Outcome: Actual speedups in inference (still based on the expensive Iterative Magnitude Pruning). <br><br>Work together with <a href="https://twitter.com/ibomohsin?ref_src=twsrc%5Etfw">@ibomohsin</a>  <a href="https://twitter.com/re_rayne?ref_src=twsrc%5Etfw">@re_rayne</a> <a href="https://twitter.com/keysers?ref_src=twsrc%5Etfw">@keysers</a> <a href="https://t.co/Ui9kmlV9QL">https://t.co/Ui9kmlV9QL</a></p>&mdash; Ilya Tolstikhin (@tolstikhini) <a href="https://twitter.com/tolstikhini/status/1415620207201116164?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. GgViz: Accelerating Large-Scale Esports Game Analysis

Peter Xenopoulos, Joao Rulff, Claudio Silva

- retweets: 144, favorites: 38 (07/16/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.06495) | [pdf](https://arxiv.org/pdf/2107.06495)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

Game review is crucial for teams, players and media staff in sports. Despite its importance, game review is work-intensive and hard to scale. Recent advances in sports data collection have introduced systems that couple video with clustering techniques to allow for users to query sports situations of interest through sketching. However, due to data limitations, as well as differences in the sport itself, esports has seen a dearth of such systems. In this paper, we leverage emerging data for Counter-Strike: Global Offensive (CSGO) to develop ggViz, a novel visual analytics system that allows users to query a large esports data set for similar plays by drawing situations of interest. Along with ggViz, we also present a performant retrieval algorithm that can easily scale to hundreds of millions of game situations. We demonstrate ggViz's utility through detailed cases studies and interviews with staff from professional esports teams.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">For most sports, but especially esports, we analyze the game by watching film, which makes finding specific player setups time-consuming. In ggViz, we introduce a system to query a large CSGO dataset to find player setups. Link to paper: <a href="https://t.co/39OkWDynqk">https://t.co/39OkWDynqk</a> <a href="https://t.co/xx2J1fDk0g">pic.twitter.com/xx2J1fDk0g</a></p>&mdash; Peter Xenopoulos (@peterxeno) <a href="https://twitter.com/peterxeno/status/1415711039862288386?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Generative and reproducible benchmarks for comprehensive evaluation of  machine learning classifiers

Patryk Orzechowski, Jason H. Moore

- retweets: 51, favorites: 7 (07/16/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.06475) | [pdf](https://arxiv.org/pdf/2107.06475)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Understanding the strengths and weaknesses of machine learning (ML) algorithms is crucial for determine their scope of application. Here, we introduce the DIverse and GENerative ML Benchmark (DIGEN) - a collection of synthetic datasets for comprehensive, reproducible, and interpretable benchmarking of machine learning algorithms for classification of binary outcomes. The DIGEN resource consists of 40 mathematical functions which map continuous features to discrete endpoints for creating synthetic datasets. These 40 functions were discovered using a heuristic algorithm designed to maximize the diversity of performance among multiple popular machine learning algorithms thus providing a useful test suite for evaluating and comparing new methods. Access to the generative functions facilitates understanding of why a method performs poorly compared to other algorithms thus providing ideas for improvement. The resource with extensive documentation and analyses is open-source and available on GitHub.




# 11. How to make qubits speak

Bob Coecke, Giovanni de Felice, Konstantinos Meichanetzidis, Alexis Toumi

- retweets: 20, favorites: 34 (07/16/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.06776) | [pdf](https://arxiv.org/pdf/2107.06776)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

This is a story about making quantum computers speak, and doing so in a quantum-native, compositional and meaning-aware manner. Recently we did question-answering with an actual quantum computer. We explain what we did, stress that this was all done in terms of pictures, and provide many pointers to the related literature. In fact, besides natural language, many other things can be implemented in a quantum-native, compositional and meaning-aware manner, and we provide the reader with some indications of that broader pictorial landscape, including our account on the notion of compositionality. We also provide some guidance for the actual execution, so that the reader can give it a go as well.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper on arXiv with <a href="https://twitter.com/konstantinosmei?ref_src=twsrc%5Etfw">@konstantinosmei</a> <a href="https://twitter.com/AlexisToumi?ref_src=twsrc%5Etfw">@AlexisToumi</a> <a href="https://twitter.com/gio_defel?ref_src=twsrc%5Etfw">@gio_defel</a> with easy reading QNLP, and also some digressions on &quot;compositionality&quot;. <a href="https://t.co/4tvNlMCqx1">https://t.co/4tvNlMCqx1</a><br><br>To appear in <a href="https://twitter.com/bio_computer?ref_src=twsrc%5Etfw">@bio_computer</a>&#39;s book on quantum and the arts.</p>&mdash; bOb cOeCke (@coecke) <a href="https://twitter.com/coecke/status/1415653572558077957?ref_src=twsrc%5Etfw">July 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Learning Algebraic Recombination for Compositional Generalization

Chenyao Liu, Shengnan An, Zeqi Lin, Qian Liu, Bei Chen, Jian-Guang Lou, Lijie Wen, Nanning Zheng, Dongmei Zhang

- retweets: 31, favorites: 20 (07/16/2021 07:00:59)

- links: [abs](https://arxiv.org/abs/2107.06516) | [pdf](https://arxiv.org/pdf/2107.06516)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Neural sequence models exhibit limited compositional generalization ability in semantic parsing tasks. Compositional generalization requires algebraic recombination, i.e., dynamically recombining structured expressions in a recursive manner. However, most previous studies mainly concentrate on recombining lexical units, which is an important but not sufficient part of algebraic recombination. In this paper, we propose LeAR, an end-to-end neural model to learn algebraic recombination for compositional generalization. The key insight is to model the semantic parsing task as a homomorphism between a latent syntactic algebra and a semantic algebra, thus encouraging algebraic recombination. Specifically, we learn two modules jointly: a Composer for producing latent syntax, and an Interpreter for assigning semantic operations. Experiments on two realistic and comprehensive compositional generalization benchmarks demonstrate the effectiveness of our model. The source code is publicly available at https://github.com/microsoft/ContextualSP.



