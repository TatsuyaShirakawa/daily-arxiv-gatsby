---
title: Hot Papers 2021-01-22
date: 2021-01-24T09:44:31.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-22"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-22"
socialImage: "/media/flying-marine.jpg"

---

# 1. Characterizing signal propagation to close the performance gap in  unnormalized ResNets

Andrew Brock, Soham De, Samuel L. Smith

- retweets: 5563, favorites: 389 (01/24/2021 09:44:31)

- links: [abs](https://arxiv.org/abs/2101.08692) | [pdf](https://arxiv.org/pdf/2101.08692)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Batch Normalization is a key component in almost all state-of-the-art image classifiers, but it also introduces practical challenges: it breaks the independence between training examples within a batch, can incur compute and memory overhead, and often results in unexpected bugs. Building on recent theoretical analyses of deep ResNets at initialization, we propose a simple set of analysis tools to characterize signal propagation on the forward pass, and leverage these tools to design highly performant ResNets without activation normalization layers. Crucial to our success is an adapted version of the recently proposed Weight Standardization. Our analysis tools show how this technique preserves the signal in networks with ReLU or Swish activation functions by ensuring that the per-channel activation means do not grow with depth. Across a range of FLOP budgets, our networks attain performance competitive with the state-of-the-art EfficientNets on ImageNet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Normalizer-Free ResNets: Our ICLR2021 paper w/ <a href="https://twitter.com/sohamde_?ref_src=twsrc%5Etfw">@sohamde_</a>&amp; <a href="https://twitter.com/SamuelMLSmith?ref_src=twsrc%5Etfw">@SamuelMLSmith</a> <br><br>We show how to train deep ResNets w/o *any* normalization to ImageNet test accuracies competitive with ResNets, and EfficientNets at a range of FLOP budgets, while training faster.<a href="https://t.co/2WMhCkaxJh">https://t.co/2WMhCkaxJh</a> <a href="https://t.co/nwF7lT25BK">pic.twitter.com/nwF7lT25BK</a></p>&mdash; Andy Brock (@ajmooch) <a href="https://twitter.com/ajmooch/status/1352614051352899585?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Learn to Dance with AIST++: Music Conditioned 3D Dance Generation

Ruilong Li, Shan Yang, David A. Ross, Angjoo Kanazawa

- retweets: 3672, favorites: 304 (01/24/2021 09:44:31)

- links: [abs](https://arxiv.org/abs/2101.08779) | [pdf](https://arxiv.org/pdf/2101.08779)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

In this paper, we present a transformer-based learning framework for 3D dance generation conditioned on music. We carefully design our network architecture and empirically study the keys for obtaining qualitatively pleasing results. The critical components include a deep cross-modal transformer, which well learns the correlation between the music and dance motion; and the full-attention with future-N supervision mechanism which is essential in producing long-range non-freezing motion. In addition, we propose a new dataset of paired 3D motion and music called AIST++, which we reconstruct from the AIST multi-view dance videos. This dataset contains 1.1M frames of 3D dance motion in 1408 sequences, covering 10 genres of dance choreographies and accompanied with multi-view camera parameters. To our knowledge it is the largest dataset of this kind. Rich experiments on AIST++ demonstrate our method produces much better results than the state-of-the-art methods both qualitatively and quantitatively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learn to Dance with AIST++: Music Conditioned 3D Dance Generation<br>pdf: <a href="https://t.co/8t9PneGpgm">https://t.co/8t9PneGpgm</a><br>abs: <a href="https://t.co/qLCqzeK15y">https://t.co/qLCqzeK15y</a><br>project page: <a href="https://t.co/hujZbGhhqv">https://t.co/hujZbGhhqv</a> <a href="https://t.co/JMP96A6gGW">pic.twitter.com/JMP96A6gGW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1352441653873750022?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Noisy intermediate-scale quantum (NISQ) algorithms

Kishor Bharti, Alba Cervera-Lierta, Thi Ha Kyaw, Tobias Haug, Sumner Alperin-Lea, Abhinav Anand, Matthias Degroote, Hermanni Heimonen, Jakob S. Kottmann, Tim Menke, Wai-Keong Mok, Sukin Sim, Leong-Chuan Kwek, Alán Aspuru-Guzik

- retweets: 2405, favorites: 201 (01/24/2021 09:44:32)

- links: [abs](https://arxiv.org/abs/2101.08448) | [pdf](https://arxiv.org/pdf/2101.08448)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A universal fault-tolerant quantum computer that can solve efficiently problems such as integer factorization and unstructured database search requires millions of qubits with low error rates and long coherence times. While the experimental advancement towards realizing such devices will potentially take decades of research, noisy intermediate-scale quantum (NISQ) computers already exist. These computers are composed of hundreds of noisy qubits, i.e. qubits that are not error-corrected, and therefore perform imperfect operations in a limited coherence time. In the search for quantum advantage with these devices, algorithms have been proposed for applications in various disciplines spanning physics, machine learning, quantum chemistry and combinatorial optimization. The goal of such algorithms is to leverage the limited available resources to perform classically challenging tasks. In this review, we provide a thorough summary of NISQ computational paradigms and algorithms. We discuss the key structure of these algorithms, their limitations, and advantages. We additionally provide a comprehensive overview of various benchmarking and software tools useful for programming and testing NISQ devices.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A fantastic team of <a href="https://twitter.com/quantumlah?ref_src=twsrc%5Etfw">@quantumlah</a> and <a href="https://twitter.com/UofTCompSci?ref_src=twsrc%5Etfw">@UofTCompSci</a> <a href="https://twitter.com/chemuoft?ref_src=twsrc%5Etfw">@chemuoft</a> <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@UofT</a> <a href="https://twitter.com/VectorInst?ref_src=twsrc%5Etfw">@VectorInst</a> <a href="https://twitter.com/hashtag/matterlab?src=hash&amp;ref_src=twsrc%5Etfw">#matterlab</a> folks and I wrote a review of <a href="https://twitter.com/hashtag/NISQ?src=hash&amp;ref_src=twsrc%5Etfw">#NISQ</a> <a href="https://twitter.com/hashtag/quantumcomputing?src=hash&amp;ref_src=twsrc%5Etfw">#quantumcomputing</a> algorithms. If you have any suggestions for what we missed please reach out. <a href="https://t.co/dg0LKI7QoO">https://t.co/dg0LKI7QoO</a> 1/n <a href="https://t.co/d6qWa3EB0Q">https://t.co/d6qWa3EB0Q</a></p>&mdash; Alan Aspuru-Guzik (@A_Aspuru_Guzik) <a href="https://twitter.com/A_Aspuru_Guzik/status/1352560143876173824?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">everything you’d want to know about how to make quantum computers do useful stuff soonish, many authors to follow:<a href="https://twitter.com/CQT_Kishor?ref_src=twsrc%5Etfw">@CQT_Kishor</a> <a href="https://twitter.com/albaclierta?ref_src=twsrc%5Etfw">@albaclierta</a> <a href="https://twitter.com/thihakyaw_phy?ref_src=twsrc%5Etfw">@thihakyaw_phy</a> <a href="https://twitter.com/theanandabhinav?ref_src=twsrc%5Etfw">@theanandabhinav</a> <a href="https://twitter.com/whynotquantum?ref_src=twsrc%5Etfw">@whynotquantum</a> <a href="https://twitter.com/HermanniHei?ref_src=twsrc%5Etfw">@HermanniHei</a> <a href="https://twitter.com/JakobKottmann?ref_src=twsrc%5Etfw">@JakobKottmann</a> <a href="https://twitter.com/tim_menke1?ref_src=twsrc%5Etfw">@tim_menke1</a> <a href="https://twitter.com/sukin_sim?ref_src=twsrc%5Etfw">@sukin_sim</a> <a href="https://twitter.com/A_Aspuru_Guzik?ref_src=twsrc%5Etfw">@A_Aspuru_Guzik</a><a href="https://t.co/NtufSwqfm0">https://t.co/NtufSwqfm0</a> <a href="https://t.co/F4swwL0MO5">pic.twitter.com/F4swwL0MO5</a></p>&mdash; nick farina 💘 (@nick_farina) <a href="https://twitter.com/nick_farina/status/1353039233125134336?ref_src=twsrc%5Etfw">January 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Noisy intermediate-scale quantum algorithms&quot;, a comprehensive review about NISQ era algorithms. <br>We would like to hear your comments and opinions about it <a href="https://twitter.com/hashtag/quantumtwitter?src=hash&amp;ref_src=twsrc%5Etfw">#quantumtwitter</a>! 👀👀<br>Terrific collaboration! 🥳👇<br>arXiv: <a href="https://t.co/HTIB4AFkAj">https://t.co/HTIB4AFkAj</a><br>SciRate: <a href="https://t.co/603lh0l41Q">https://t.co/603lh0l41Q</a></p>&mdash; Alba Cervera-Lierta (@albaclierta) <a href="https://twitter.com/albaclierta/status/1352528323956658177?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Can a Fruit Fly Learn Word Embeddings?

Yuchen Liang, Chaitanya K. Ryali, Benjamin Hoover, Leopold Grinberg, Saket Navlakha, Mohammed J. Zaki, Dmitry Krotov

- retweets: 1587, favorites: 223 (01/24/2021 09:44:32)

- links: [abs](https://arxiv.org/abs/2101.06887) | [pdf](https://arxiv.org/pdf/2101.06887)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The mushroom body of the fruit fly brain is one of the best studied systems in neuroscience. At its core it consists of a population of Kenyon cells, which receive inputs from multiple sensory modalities. These cells are inhibited by the anterior paired lateral neuron, thus creating a sparse high dimensional representation of the inputs. In this work we study a mathematical formalization of this network motif and apply it to learning the correlational structure between words and their context in a corpus of unstructured text, a common natural language processing (NLP) task. We show that this network can learn semantic representations of words and can generate both static and context-dependent word embeddings. Unlike conventional methods (e.g., BERT, GloVe) that use dense representations for word embedding, our algorithm encodes semantic meaning of words and their context in the form of sparse binary hash codes. The quality of the learned representations is evaluated on word similarity analysis, word-sense disambiguation, and document classification. It is shown that not only can the fruit fly network motif achieve performance comparable to existing methods in NLP, but, additionally, it uses only a fraction of the computational resources (shorter training time and smaller memory footprint).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can a Fruit Fly Learn Word Embeddings?<br>pdf: <a href="https://t.co/zhbnf3Vyvl">https://t.co/zhbnf3Vyvl</a><br>abs: <a href="https://t.co/9tep5IG80l">https://t.co/9tep5IG80l</a> <a href="https://t.co/3wsVAkdyDd">pic.twitter.com/3wsVAkdyDd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351355495072886785?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our <a href="https://twitter.com/hashtag/ICLR2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICLR2021</a> paper we study a well-established neurobiological network motif from the fruit fly brain and investigate the possibility of reusing its architecture for solving common natural language processing tasks.<br>Paper: <a href="https://t.co/vQYJePQG7Y">https://t.co/vQYJePQG7Y</a> <a href="https://t.co/Y7kWU5VDWH">pic.twitter.com/Y7kWU5VDWH</a></p>&mdash; Yuchen Liang (@YuchenLiangRPI) <a href="https://twitter.com/YuchenLiangRPI/status/1351642932223373312?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Unifying Cardiovascular Modelling with Deep Reinforcement Learning for  Uncertainty Aware Control of Sepsis Treatment

Thesath Nanayakkara, Gilles Clermont, Christopher James Langmead, David Swigon

- retweets: 1354, favorites: 27 (01/24/2021 09:44:33)

- links: [abs](https://arxiv.org/abs/2101.08477) | [pdf](https://arxiv.org/pdf/2101.08477)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

Sepsis is the leading cause of mortality in the the ICU, responsible for 6% of all hospitalizations and 35% of all in-hospital deaths in USA. However, there is no universally agreed upon strategy for vasopressor and fluid administration. It has also been observed that different patients respond differently to treatment, highlighting the need for individualized treatment. Vasopressors and fluids are administrated with specific effects to cardiovascular physiology in mind and medical research has suggested that physiologic, hemodynamically guided, approaches to treatment. Thus we propose a novel approach, exploiting and unifying complementary strengths of Mathematical Modelling, Deep Learning, Reinforcement Learning and Uncertainty Quantification, to learn individualized, safe, and uncertainty aware treatment strategies. We first infer patient-specific, dynamic cardiovascular states using a novel physiology-driven recurrent neural network trained in an unsupervised manner. This information, along with a learned low dimensional representation of the patient's lab history and observable data, is then used to derive value distributions using Batch Distributional Reinforcement Learning. Moreover in a safety critical domain it is essential to know what our agent does and does not know, for this we also quantity the model uncertainty associated with each patient state and action, and propose a general framework for uncertainty aware, interpretable treatment policies. This framework can be tweaked easily, to reflect a clinician's own confidence of of the framework, and can be easily modified to factor in human expert opinion, whenever it's accessible. Using representative patients and a validation cohort, we show that our method has learned physiologically interpretable generalizable policies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unifying Cardiovascular Modelling with Deep Reinforcement Learning for Uncertainty Aware Control of Sepsis Treatment<a href="https://t.co/gIzxFIjCxD">https://t.co/gIzxFIjCxD</a><a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/ML?src=hash&amp;ref_src=twsrc%5Etfw">#ML</a> <a href="https://twitter.com/hashtag/Healthcare?src=hash&amp;ref_src=twsrc%5Etfw">#Healthcare</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/serverless?src=hash&amp;ref_src=twsrc%5Etfw">#serverless</a> <a href="https://twitter.com/hashtag/100daysofcode?src=hash&amp;ref_src=twsrc%5Etfw">#100daysofcode</a> <a href="https://twitter.com/hashtag/programming?src=hash&amp;ref_src=twsrc%5Etfw">#programming</a> <a href="https://twitter.com/hashtag/womenwhocode?src=hash&amp;ref_src=twsrc%5Etfw">#womenwhocode</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://t.co/k6wuf7hhSI">pic.twitter.com/k6wuf7hhSI</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1353104732060061700?ref_src=twsrc%5Etfw">January 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. LEAF: A Learnable Frontend for Audio Classification

Neil Zeghidour, Olivier Teboul, Félix de Chaumont Quitry, Marco Tagliasacchi

- retweets: 930, favorites: 150 (01/24/2021 09:44:33)

- links: [abs](https://arxiv.org/abs/2101.08596) | [pdf](https://arxiv.org/pdf/2101.08596)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Mel-filterbanks are fixed, engineered audio features which emulate human perception and have been used through the history of audio understanding up to today. However, their undeniable qualities are counterbalanced by the fundamental limitations of handmade representations. In this work we show that we can train a single learnable frontend that outperforms mel-filterbanks on a wide range of audio signals, including speech, music, audio events and animal sounds, providing a general-purpose learned frontend for audio classification. To do so, we introduce a new principled, lightweight, fully learnable architecture that can be used as a drop-in replacement of mel-filterbanks. Our system learns all operations of audio features extraction, from filtering to pooling, compression and normalization, and can be integrated into any neural network at a negligible parameter cost. We perform multi-task training on eight diverse audio classification tasks, and show consistent improvements of our model over mel-filterbanks and previous learnable alternatives. Moreover, our system outperforms the current state-of-the-art learnable frontend on Audioset, with orders of magnitude fewer parameters.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I will present LEAF, a learnable frontend for audio classification, at ICLR 2021.<br>* Learns filtering, pooling, compression, normalization<br>* Evaluated on 8 tasks, incl. speech, music, birds<br>* Outperforms mel-fbanks, SincNet, and others<br>* SOTA on AudioSet<a href="https://t.co/S6uFyKj0HT">https://t.co/S6uFyKj0HT</a> <a href="https://t.co/ccgsVVRX31">pic.twitter.com/ccgsVVRX31</a></p>&mdash; Neil Zeghidour (@neilzegh) <a href="https://twitter.com/neilzegh/status/1352533224854065152?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. DAF:re: A Challenging, Crowd-Sourced, Large-Scale, Long-Tailed Dataset  For Anime Character Recognition

Edwin Arkel Rios, Wen-Huang Cheng, Bo-Cheng Lai

- retweets: 650, favorites: 52 (01/24/2021 09:44:33)

- links: [abs](https://arxiv.org/abs/2101.08674) | [pdf](https://arxiv.org/pdf/2101.08674)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this work we tackle the challenging problem of anime character recognition. Anime, referring to animation produced within Japan and work derived or inspired from it. For this purpose we present DAF:re (DanbooruAnimeFaces:revamped), a large-scale, crowd-sourced, long-tailed dataset with almost 500 K images spread across more than 3000 classes. Additionally, we conduct experiments on DAF:re and similar datasets using a variety of classification models, including CNN based ResNets and self-attention based Vision Transformer (ViT). Our results give new insights into the generalization and transfer learning properties of ViT models on substantially different domain datasets from those used for the upstream pre-training, including the influence of batch and image size in their training. Additionally, we share our dataset, source-code, pre-trained checkpoints and results, as Animesion, the first end-to-end framework for large-scale anime character recognition: https://github.com/arkel23/animesion

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DAF:re: A Challenging, Crowd-Sourced, Large-Scale, Long-Tailed Dataset For Anime Character Recognition<br>pdf: <a href="https://t.co/xd5FYybeQV">https://t.co/xd5FYybeQV</a><br>abs: <a href="https://t.co/mDVbuqeG0B">https://t.co/mDVbuqeG0B</a><br>github: <a href="https://t.co/s2l0F8bLQ9">https://t.co/s2l0F8bLQ9</a> <a href="https://t.co/fgD82pnF3s">pic.twitter.com/fgD82pnF3s</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1352439894149365762?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Pre-training without Natural Images

Hirokatsu Kataoka, Kazushige Okayasu, Asato Matsumoto, Eisuke Yamagata, Ryosuke Yamada, Nakamasa Inoue, Akio Nakamura, Yutaka Satoh

- retweets: 110, favorites: 41 (01/24/2021 09:44:33)

- links: [abs](https://arxiv.org/abs/2101.08515) | [pdf](https://arxiv.org/pdf/2101.08515)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Is it possible to use convolutional neural networks pre-trained without any natural images to assist natural image understanding? The paper proposes a novel concept, Formula-driven Supervised Learning. We automatically generate image patterns and their category labels by assigning fractals, which are based on a natural law existing in the background knowledge of the real world. Theoretically, the use of automatically generated images instead of natural images in the pre-training phase allows us to generate an infinite scale dataset of labeled images. Although the models pre-trained with the proposed Fractal DataBase (FractalDB), a database without natural images, does not necessarily outperform models pre-trained with human annotated datasets at all settings, we are able to partially surpass the accuracy of ImageNet/Places pre-trained models. The image representation with the proposed FractalDB captures a unique feature in the visualization of convolutional layers and attentions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pre-training without Natural Images (ACCV 2020 Best Paper Honorable Mention Award) is published in arXiv!<br>--<br>abs: <a href="https://t.co/F64VUongG2">https://t.co/F64VUongG2</a><br>pdf: <a href="https://t.co/vzyGxjuLsJ">https://t.co/vzyGxjuLsJ</a><br>project: <a href="https://t.co/iqRJuvpncx">https://t.co/iqRJuvpncx</a><br>code: <a href="https://t.co/pROhocn0em">https://t.co/pROhocn0em</a></p>&mdash; Hirokatsu Kataoka | 片岡裕雄 (@HirokatuKataoka) <a href="https://twitter.com/HirokatuKataoka/status/1352432710392836096?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. MLPF: Efficient machine-learned particle-flow reconstruction using graph  neural networks

Joosep Pata, Javier Duarte, Jean-Roch Vlimant, Maurizio Pierini, Maria Spiropulu

- retweets: 90, favorites: 50 (01/24/2021 09:44:34)

- links: [abs](https://arxiv.org/abs/2101.08578) | [pdf](https://arxiv.org/pdf/2101.08578)
- [physics.data-an](https://arxiv.org/list/physics.data-an/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [hep-ex](https://arxiv.org/list/hep-ex/recent) | [physics.ins-det](https://arxiv.org/list/physics.ins-det/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In general-purpose particle detectors, the particle flow algorithm may be used to reconstruct a coherent particle-level view of the event by combining information from the calorimeters and the trackers, significantly improving the detector resolution for jets and the missing transverse momentum. In view of the planned high-luminosity upgrade of the CERN Large Hadron Collider, it is necessary to revisit existing reconstruction algorithms and ensure that both the physics and computational performance are sufficient in a high-pileup environment. Recent developments in machine learning may offer a prospect for efficient event reconstruction based on parametric models. We introduce MLPF, an end-to-end trainable machine-learned particle flow algorithm for reconstructing particle flow candidates based on parallelizable, computationally efficient, scalable graph neural networks and a multi-task objective. We report the physics and computational performance of the MLPF algorithm on on a synthetic dataset of ttbar events in HL-LHC running conditions, including the simulation of multiple interaction effects, and discuss potential next steps and considerations towards ML-based reconstruction in a general purpose particle detector.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A graph net can learn particle flow reconstruction and be computationally efficient &amp; scalable! Dataset &amp; code on zenodo! 🙂🙂<br>Work with <a href="https://twitter.com/jmgduarte?ref_src=twsrc%5Etfw">@jmgduarte</a> <a href="https://twitter.com/xmpierinix?ref_src=twsrc%5Etfw">@xmpierinix</a> <a href="https://twitter.com/vlimant?ref_src=twsrc%5Etfw">@vlimant</a> <a href="https://twitter.com/MariaSpiropulu?ref_src=twsrc%5Etfw">@MariaSpiropulu</a>  <a href="https://t.co/s1SJnJI0Tn">https://t.co/s1SJnJI0Tn</a> <a href="https://t.co/67pdflErKE">pic.twitter.com/67pdflErKE</a></p>&mdash; Joosep Pata (@JoosepPata) <a href="https://twitter.com/JoosepPata/status/1352500808089296898?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Aesthetics, Personalization and Recommendation: A survey on Deep  Learning in Fashion

Wei Gong, Laila Khalid

- retweets: 83, favorites: 20 (01/24/2021 09:44:34)

- links: [abs](https://arxiv.org/abs/2101.08301) | [pdf](https://arxiv.org/pdf/2101.08301)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Machine learning is completely changing the trends in the fashion industry. From big to small every brand is using machine learning techniques in order to improve their revenue, increase customers and stay ahead of the trend. People are into fashion and they want to know what looks best and how they can improve their style and elevate their personality. Using Deep learning technology and infusing it with Computer Vision techniques one can do so by utilizing Brain-inspired Deep Networks, and engaging into Neuroaesthetics, working with GANs and Training them, playing around with Unstructured Data,and infusing the transformer architecture are just some highlights which can be touched with the Fashion domain. Its all about designing a system that can tell us information regarding the fashion aspect that can come in handy with the ever growing demand. Personalization is a big factor that impacts the spending choices of customers.The survey also shows remarkable approaches that encroach the subject of achieving that by divulging deep into how visual data can be interpreted and leveraged into different models and approaches. Aesthetics play a vital role in clothing recommendation as users' decision depends largely on whether the clothing is in line with their aesthetics, however the conventional image features cannot portray this directly. For that the survey also highlights remarkable models like tensor factorization model, conditional random field model among others to cater the need to acknowledge aesthetics as an important factor in Apparel recommendation.These AI inspired deep models can pinpoint exactly which certain style resonates best with their customers and they can have an understanding of how the new designs will set in with the community. With AI and machine learning your businesses can stay ahead of the fashion trends.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Aesthetics, Personalization and Recommendation: A survey on Deep Learning in Fashion. <a href="https://t.co/ebkoAgj8bf">https://t.co/ebkoAgj8bf</a> <a href="https://t.co/I5OdsYE5rH">pic.twitter.com/I5OdsYE5rH</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1352685471415951360?ref_src=twsrc%5Etfw">January 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Robust Reinforcement Learning on State Observations with Learned Optimal  Adversary

Huan Zhang, Hongge Chen, Duane Boning, Cho-Jui Hsieh

- retweets: 32, favorites: 24 (01/24/2021 09:44:34)

- links: [abs](https://arxiv.org/abs/2101.08452) | [pdf](https://arxiv.org/pdf/2101.08452)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We study the robustness of reinforcement learning (RL) with adversarially perturbed state observations, which aligns with the setting of many adversarial attacks to deep reinforcement learning (DRL) and is also important for rolling out real-world RL agent under unpredictable sensing noise. With a fixed agent policy, we demonstrate that an optimal adversary to perturb state observations can be found, which is guaranteed to obtain the worst case agent reward. For DRL settings, this leads to a novel empirical adversarial attack to RL agents via a learned adversary that is much stronger than previous ones. To enhance the robustness of an agent, we propose a framework of alternating training with learned adversaries (ATLA), which trains an adversary online together with the agent using policy gradient following the optimal adversarial attack framework. Additionally, inspired by the analysis of state-adversarial Markov decision process (SA-MDP), we show that past states and actions (history) can be useful for learning a robust agent, and we empirically find a LSTM based policy can be more robust under adversaries. Empirical evaluations on a few continuous control environments show that ATLA achieves state-of-the-art performance under strong adversaries. Our code is available at https://github.com/huanzhang12/ATLA_robust_RL.




# 12. Zero-shot Generalization in Dialog State Tracking through Generative  Question Answering

Shuyang Li, Jin Cao, Mukund Sridhar, Henghui Zhu, Shang-Wen Li, Wael Hamza, Julian McAuley

- retweets: 17, favorites: 34 (01/24/2021 09:44:34)

- links: [abs](https://arxiv.org/abs/2101.08333) | [pdf](https://arxiv.org/pdf/2101.08333)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Dialog State Tracking (DST), an integral part of modern dialog systems, aims to track user preferences and constraints (slots) in task-oriented dialogs. In real-world settings with constantly changing services, DST systems must generalize to new domains and unseen slot types. Existing methods for DST do not generalize well to new slot names and many require known ontologies of slot types and values for inference. We introduce a novel ontology-free framework that supports natural language queries for unseen constraints and slots in multi-domain task-oriented dialogs. Our approach is based on generative question-answering using a conditional language model pre-trained on substantive English sentences. Our model improves joint goal accuracy in zero-shot domain adaptation settings by up to 9% (absolute) over the previous state-of-the-art on the MultiWOZ 2.1 dataset.



