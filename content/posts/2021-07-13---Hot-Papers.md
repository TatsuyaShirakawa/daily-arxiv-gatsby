---
title: Hot Papers 2021-07-13
date: 2021-07-14T10:56:40.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-13"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-13"
socialImage: "/media/flying-marine.jpg"

---

# 1. Optimally Reliable & Cheap Payment Flows on the Lightning Network

Rene Pickhardt, Stefan Richter

- retweets: 15582, favorites: 1 (07/14/2021 10:56:40)

- links: [abs](https://arxiv.org/abs/2107.05322) | [pdf](https://arxiv.org/pdf/2107.05322)
- [cs.NI](https://arxiv.org/list/cs.NI/recent)

Today, payment paths in Bitcoin's Lightning Network are found by searching for shortest paths on the fee graph. We enhance this approach in two dimensions. Firstly, we take into account the probability of a payment actually being possible due to the unknown balance distributions in the channels. Secondly, we use minimum cost flows as a proper generalization of shortest paths to multi-part payments (MPP). In particular we show that under plausible assumptions about the balance distributions we can find the most likely MPP for any given set of senders, recipients and amounts by solving for a (generalized) integer minimum cost flow with a separable and convex cost function. Polynomial time exact algorithms as well as approximations are known for this optimization problem. We present a round-based algorithm of min-cost flow computations for delivering large payment amounts over the Lightning Network. This algorithm works by updating the probability distributions with the information gained from both successful and unsuccessful paths on prior rounds. In all our experiments a single digit number of rounds sufficed to deliver payments of sizes that were close to the total local balance of the sender. Early experiments indicate that our approach increases the size of payments that can be reliably delivered by several orders of magnitude compared to the current state of the art. We observe that finding the cheapest multi-part payments is an NP-hard problem considering the current fee structure and propose dropping the base fee to make it a linear min-cost flow problem. Finally, we discuss possibilities for maximizing the probability while at the same time minimizing the fees of a flow. While this turns out to be a hard problem in general as well - even in the single path case - it appears to be surprisingly tractable in practice.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Optimally Reliable &amp; Cheap Payment Flows on the Lightning Network<br><br>the joint work with <a href="https://twitter.com/stefanwouldgo?ref_src=twsrc%5Etfw">@stefanwouldgo</a> is now (as promised) in the <a href="https://twitter.com/hashtag/open?src=hash&amp;ref_src=twsrc%5Etfw">#open</a>: <a href="https://t.co/CjzaZ1edVG">https://t.co/CjzaZ1edVG</a> <br><br>This method should allow us to <br>⚡️ send large  <a href="https://twitter.com/hashtag/Bitcoin?src=hash&amp;ref_src=twsrc%5Etfw">#Bitcoin</a> amounts over the <a href="https://twitter.com/hashtag/LightningNetwork?src=hash&amp;ref_src=twsrc%5Etfw">#LightningNetwork</a> ⚡️<br><br>Thread: 1 / 15</p>&mdash; Rene Pickhardt (@renepickhardt) <a href="https://twitter.com/renepickhardt/status/1414895844889960450?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Real-Time Super-Resolution System of 4K-Video Based on Deep Learning

Yanpeng Cao, Chengcheng Wang, Changjun Song, He Li, Yongming Tang

- retweets: 11968, favorites: 4 (07/14/2021 10:56:40)

- links: [abs](https://arxiv.org/abs/2107.05307) | [pdf](https://arxiv.org/pdf/2107.05307)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Video super-resolution (VSR) technology excels in reconstructing low-quality video, avoiding unpleasant blur effect caused by interpolation-based algorithms. However, vast computation complexity and memory occupation hampers the edge of deplorability and the runtime inference in real-life applications, especially for large-scale VSR task. This paper explores the possibility of real-time VSR system and designs an efficient and generic VSR network, termed EGVSR. The proposed EGVSR is based on spatio-temporal adversarial learning for temporal coherence. In order to pursue faster VSR processing ability up to 4K resolution, this paper tries to choose lightweight network structure and efficient upsampling method to reduce the computation required by EGVSR network under the guarantee of high visual quality. Besides, we implement the batch normalization computation fusion, convolutional acceleration algorithm and other neural network acceleration techniques on the actual hardware platform to optimize the inference process of EGVSR network. Finally, our EGVSR achieves the real-time processing capacity of 4K@29.61FPS. Compared with TecoGAN, the most advanced VSR network at present, we achieve 85.04% reduction of computation density and 7.92x performance speedups. In terms of visual quality, the proposed EGVSR tops the list of most metrics (such as LPIPS, tOF, tLP, etc.) on the public test dataset Vid4 and surpasses other state-of-the-art methods in overall performance score. The source code of this project can be found on https://github.com/Thmen/EGVSR.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Real-Time Super-Resolution System of 4K-Video Based on Deep Learning<br>pdf: <a href="https://t.co/YaDX8Om9SI">https://t.co/YaDX8Om9SI</a><br>abs: <a href="https://t.co/p7rtATT6gq">https://t.co/p7rtATT6gq</a><br>github: <a href="https://t.co/HGYWqR54nC">https://t.co/HGYWqR54nC</a> <a href="https://t.co/1LYrEXoyHB">pic.twitter.com/1LYrEXoyHB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414761877238915073?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. You Really Shouldn't Roll Your Own Crypto: An Empirical Study of  Vulnerabilities in Cryptographic Libraries

Jenny Blessing, Michael A. Specter, Daniel J. Weitzner

- retweets: 10179, favorites: 8 (07/14/2021 10:56:40)

- links: [abs](https://arxiv.org/abs/2107.04940) | [pdf](https://arxiv.org/pdf/2107.04940)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

The security of the Internet rests on a small number of open-source cryptographic libraries: a vulnerability in any one of them threatens to compromise a significant percentage of web traffic. Despite this potential for security impact, the characteristics and causes of vulnerabilities in cryptographic software are not well understood. In this work, we conduct the first comprehensive analysis of cryptographic libraries and the vulnerabilities affecting them. We collect data from the National Vulnerability Database, individual project repositories and mailing lists, and other relevant sources for eight widely used cryptographic libraries.   Among our most interesting findings is that only 27.2% of vulnerabilities in cryptographic libraries are cryptographic issues while 37.2% of vulnerabilities are memory safety issues, indicating that systems-level bugs are a greater security concern than the actual cryptographic procedures. In our investigation of the causes of these vulnerabilities, we find evidence of a strong correlation between the complexity of these libraries and their (in)security, empirically demonstrating the potential risks of bloated cryptographic codebases. We further compare our findings with non-cryptographic systems, observing that these systems are, indeed, more complex than similar counterparts, and that this excess complexity appears to produce significantly more vulnerabilities in cryptographic libraries than in non-cryptographic software.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&gt; Among our most interesting findings is that only 27.2% of vulnerabilities in cryptographic libraries are cryptographic issues while 37.2% of vulnerabilities are memory safety issues<br><br>Please stop writing crypto in C/C++/assembly, people<a href="https://t.co/ixuDX135GP">https://t.co/ixuDX135GP</a></p>&mdash; Pratyush Mishra (@zkproofs) <a href="https://twitter.com/zkproofs/status/1414794511679557640?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. MidiBERT-Piano: Large-scale Pre-training for Symbolic Music  Understanding

Yi-Hui Chou, I-Chun Chen, Chin-Jui Chang, Joann Ching, Yi-Hsuan Yang

- retweets: 529, favorites: 99 (07/14/2021 10:56:40)

- links: [abs](https://arxiv.org/abs/2107.05223) | [pdf](https://arxiv.org/pdf/2107.05223)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper presents an attempt to employ the mask language modeling approach of BERT to pre-train a 12-layer Transformer model over 4,166 pieces of polyphonic piano MIDI files for tackling a number of symbolic-domain discriminative music understanding tasks. These include two note-level classification tasks, i.e., melody extraction and velocity prediction, as well as two sequence-level classification tasks, i.e., composer classification and emotion classification. We find that, given a pre-trained Transformer, our models outperform recurrent neural network based baselines with less than 10 epochs of fine-tuning. Ablation studies show that the pre-training remains effective even if none of the MIDI data of the downstream tasks are seen at the pre-training stage, and that freezing the self-attention layers of the Transformer at the fine-tuning stage slightly degrades performance. All the five datasets employed in this work are publicly available, as well as checkpoints of our pre-trained and fine-tuned models. As such, our research can be taken as a benchmark for symbolic-domain music understanding.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MidiBERT-Piano: Large-scale Pre-training for<br>Symbolic Music Understanding<br>pdf: <a href="https://t.co/uCBADS9XDG">https://t.co/uCBADS9XDG</a><br>github: <a href="https://t.co/gHarqjUR6U">https://t.co/gHarqjUR6U</a><br><br>pre-train a 12-layer Transformer model over 4,166 pieces of polyphonic piano MIDI files <a href="https://t.co/SKWfQPC388">pic.twitter.com/SKWfQPC388</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414753121650323459?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Impossibility of What? Formal and Substantive Equality in Algorithmic  Fairness

Ben Green

- retweets: 400, favorites: 72 (07/14/2021 10:56:40)

- links: [abs](https://arxiv.org/abs/2107.04642) | [pdf](https://arxiv.org/pdf/2107.04642)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In the face of compounding crises of social and economic inequality, many have turned to algorithmic decision-making to achieve greater fairness in society. As these efforts intensify, reasoning within the burgeoning field of "algorithmic fairness" increasingly shapes how fairness manifests in practice. This paper interrogates whether algorithmic fairness provides the appropriate conceptual and practical tools for enhancing social equality. I argue that the dominant, "formal" approach to algorithmic fairness is ill-equipped as a framework for pursuing equality, as its narrow frame of analysis generates restrictive approaches to reform. In light of these shortcomings, I propose an alternative: a "substantive" approach to algorithmic fairness that centers opposition to social hierarchies and provides a more expansive analysis of how to address inequality. This substantive approach enables more fruitful theorizing about the role of algorithms in combatting oppression. The distinction between formal and substantive algorithmic fairness is exemplified by each approach's responses to the "impossibility of fairness" (an incompatibility between mathematical definitions of algorithmic fairness). While the formal approach requires us to accept the "impossibility of fairness" as a harsh limit on efforts to enhance equality, the substantive approach allows us to escape the "impossibility of fairness" by suggesting reforms that are not subject to this false dilemma and that are better equipped to ameliorate conditions of social oppression.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint: The dominant, “formal” approach to algorithmic fairness is a limited framework for enhancing equality. I propose an expanded, “substantive” approach that suggests better strategies for how algorithms can (and cannot) address inequality.<br><br>📄: <a href="https://t.co/iFErWwu9jW">https://t.co/iFErWwu9jW</a> <a href="https://t.co/9UQ7IfZEZG">pic.twitter.com/9UQ7IfZEZG</a></p>&mdash; Ben Green (@benzevgreen) <a href="https://twitter.com/benzevgreen/status/1414979168677376004?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Industry and Academic Research in Computer Vision

Iuliia Kotseruba

- retweets: 272, favorites: 114 (07/14/2021 10:56:40)

- links: [abs](https://arxiv.org/abs/2107.04902) | [pdf](https://arxiv.org/pdf/2107.04902)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This work aims to study the dynamic between research in the industry and academia in computer vision. The results are demonstrated on a set of top-5 vision conferences that are representative of the field. Since data for such analysis was not readily available, significant effort was spent on gathering and processing meta-data from the original publications. First, this study quantifies the share of industry-sponsored research. Specifically, it shows that the proportion of papers published by industry-affiliated researchers is increasing and that more academics join companies or collaborate with them. Next, the possible impact of industry presence is further explored, namely in the distribution of research topics and citation patterns. The results indicate that the distribution of the research topics is similar in industry and academic papers. However, there is a strong preference towards citing industry papers. Finally, possible reasons for citation bias, such as code availability and influence, are investigated.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Industry and Academic Research in Computer Vision <a href="https://t.co/fO0WDPtp95">https://t.co/fO0WDPtp95</a> <br><br>Not your typical <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> research paper, but a study of the dynamic between research in the industry and academia in computer vision. <a href="https://t.co/4SQEjhRAkG">pic.twitter.com/4SQEjhRAkG</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1414765196585885703?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ro" dir="ltr">Iuliia Kotseruba, “Industry and Academic Research in <a href="https://twitter.com/hashtag/ComputerVision?src=hash&amp;ref_src=twsrc%5Etfw">#ComputerVision</a>”, arXiv, 2021<br><br>Preprint: <a href="https://t.co/m9gqWyYjCz">https://t.co/m9gqWyYjCz</a> <a href="https://t.co/T4VCReXk1G">pic.twitter.com/T4VCReXk1G</a></p>&mdash; Kosta Derpanis (@CSProfKGD) <a href="https://twitter.com/CSProfKGD/status/1414765793192128512?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Designing Recommender Systems to Depolarize

Jonathan Stray

- retweets: 272, favorites: 64 (07/14/2021 10:56:41)

- links: [abs](https://arxiv.org/abs/2107.04953) | [pdf](https://arxiv.org/pdf/2107.04953)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Polarization is implicated in the erosion of democracy and the progression to violence, which makes the polarization properties of large algorithmic content selection systems (recommender systems) a matter of concern for peace and security. While algorithm-driven social media does not seem to be a primary driver of polarization at the country level, it could be a useful intervention point in polarized societies. This paper examines algorithmic depolarization interventions with the goal of conflict transformation: not suppressing or eliminating conflict but moving towards more constructive conflict. Algorithmic intervention is considered at three stages: which content is available (moderation), how content is selected and personalized (ranking), and content presentation and controls (user interface). Empirical studies of online conflict suggest that the exposure diversity intervention proposed as an antidote to "filter bubbles" can be improved and can even worsen polarization under some conditions. Using civility metrics in conjunction with diversity in content selection may be more effective. However, diversity-based interventions have not been tested at scale and may not work in the diverse and dynamic contexts of real platforms. Instead, intervening in platform polarization dynamics will likely require continuous monitoring of polarization metrics, such as the widely used "feeling thermometer." These metrics can be used to evaluate product features, and potentially engineered as algorithmic objectives. It may further prove necessary to include polarization measures in the objective functions of recommender algorithms to prevent optimization processes from creating conflict as a side effect.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I wrote a paper containing everything I know about recommenders polarization, and conflict, and how these systems could be designed to depolarize. <a href="https://t.co/whzYOjoKXx">https://t.co/whzYOjoKXx</a><br><br>Here&#39;s my argument in a nutshell -- a THREAD <a href="https://t.co/remAgU3aen">pic.twitter.com/remAgU3aen</a></p>&mdash; jonathanstray (@jonathanstray) <a href="https://twitter.com/jonathanstray/status/1415046628872888322?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Generating stable molecules using imitation and reinforcement learning

Søren Ager Meldgaard, Jonas Köhler, Henrik Lund Mortensen, Mads-Peter V. Christiansen, Frank Noé, Bjørk Hammer

- retweets: 180, favorites: 61 (07/14/2021 10:56:41)

- links: [abs](https://arxiv.org/abs/2107.05007) | [pdf](https://arxiv.org/pdf/2107.05007)
- [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Chemical space is routinely explored by machine learning methods to discover interesting molecules, before time-consuming experimental synthesizing is attempted. However, these methods often rely on a graph representation, ignoring 3D information necessary for determining the stability of the molecules. We propose a reinforcement learning approach for generating molecules in cartesian coordinates allowing for quantum chemical prediction of the stability. To improve sample-efficiency we learn basic chemical rules from imitation learning on the GDB-11 database to create an initial model applicable for all stoichiometries. We then deploy multiple copies of the model conditioned on a specific stoichiometry in a reinforcement learning setting. The models correctly identify low energy molecules in the database and produce novel isomers not found in the training set. Finally, we apply the model to larger molecules to show how reinforcement learning further refines the imitation learning model in domains far from the training data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wonderful collaboration with Søren Meldgaard and <a href="https://twitter.com/Bjork_Hammer?ref_src=twsrc%5Etfw">@Bjork_Hammer</a> on generating stable molecules using  reinforcement learning. With <a href="https://twitter.com/jonkhler?ref_src=twsrc%5Etfw">@jonkhler</a>, <a href="https://twitter.com/HenrikLundMort1?ref_src=twsrc%5Etfw">@HenrikLundMort1</a>, Mads-Peter Christiansen.<a href="https://t.co/sZiR9JXeMl">https://t.co/sZiR9JXeMl</a></p>&mdash; Frank Noe (@FrankNoeBerlin) <a href="https://twitter.com/FrankNoeBerlin/status/1414869020130557953?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. PonderNet: Learning to Ponder

Andrea Banino, Jan Balaguer, Charles Blundell

- retweets: 112, favorites: 83 (07/14/2021 10:56:41)

- links: [abs](https://arxiv.org/abs/2107.05407) | [pdf](https://arxiv.org/pdf/2107.05407)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CC](https://arxiv.org/list/cs.CC/recent)

In standard neural networks the amount of computation used grows with the size of the inputs, but not with the complexity of the problem being learnt. To overcome this limitation we introduce PonderNet, a new algorithm that learns to adapt the amount of computation based on the complexity of the problem at hand. PonderNet learns end-to-end the number of computational steps to achieve an effective compromise between training prediction accuracy, computational cost and generalization. On a complex synthetic problem, PonderNet dramatically improves performance over previous adaptive computation methods and additionally succeeds at extrapolation tests where traditional neural networks fail. Also, our method matched the current state of the art results on a real world question and answering dataset, but using less compute. Finally, PonderNet reached state of the art results on a complex task designed to test the reasoning capabilities of neural networks.1

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PonderNet: Learning to Ponder 🤔<br>pdf: <a href="https://t.co/AUxWIFFNCX">https://t.co/AUxWIFFNCX</a><br>abs: <a href="https://t.co/5yx0aCgt7s">https://t.co/5yx0aCgt7s</a><br>PonderNet reached sota results on a complex task designed to test the reasoning capabilities of neural networks <a href="https://t.co/PyjDEUaJah">pic.twitter.com/PyjDEUaJah</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414758709637562373?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. A Persistent Spatial Semantic Representation for High-level Natural  Language Instruction Execution

Valts Blukis, Chris Paxton, Dieter Fox, Animesh Garg, Yoav Artzi

- retweets: 127, favorites: 50 (07/14/2021 10:56:41)

- links: [abs](https://arxiv.org/abs/2107.05612) | [pdf](https://arxiv.org/pdf/2107.05612)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Natural language provides an accessible and expressive interface to specify long-term tasks for robotic agents. However, non-experts are likely to specify such tasks with high-level instructions, which abstract over specific robot actions through several layers of abstraction. We propose that key to bridging this gap between language and robot actions over long execution horizons are persistent representations. We propose a persistent spatial semantic representation method, and show how it enables building an agent that performs hierarchical reasoning to effectively execute long-term tasks. We evaluate our approach on the ALFRED benchmark and achieve state-of-the-art results, despite completely avoiding the commonly used step-by-step instructions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Persistent Spatial Semantic Representation for High-level Natural Language Instruction Execution<br>pdf: <a href="https://t.co/LA2FYUsseS">https://t.co/LA2FYUsseS</a><br>abs: <a href="https://t.co/3RrBRCD1wK">https://t.co/3RrBRCD1wK</a><br>project page: <a href="https://t.co/syBy6wgDgj">https://t.co/syBy6wgDgj</a> <a href="https://t.co/0Kp8VNB8p3">pic.twitter.com/0Kp8VNB8p3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414780398928859137?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Neural Waveshaping Synthesis

Ben Hayes, Charalampos Saitis, György Fazekas

- retweets: 54, favorites: 82 (07/14/2021 10:56:41)

- links: [abs](https://arxiv.org/abs/2107.05050) | [pdf](https://arxiv.org/pdf/2107.05050)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent) | [eess.SP](https://arxiv.org/list/eess.SP/recent)

We present the Neural Waveshaping Unit (NEWT): a novel, lightweight, fully causal approach to neural audio synthesis which operates directly in the waveform domain, with an accompanying optimisation (FastNEWT) for efficient CPU inference. The NEWT uses time-distributed multilayer perceptrons with periodic activations to implicitly learn nonlinear transfer functions that encode the characteristics of a target timbre. Once trained, a NEWT can produce complex timbral evolutions by simple affine transformations of its input and output signals. We paired the NEWT with a differentiable noise synthesiser and reverb and found it capable of generating realistic musical instrument performances with only 260k total model parameters, conditioned on F0 and loudness features. We compared our method to state-of-the-art benchmarks with a multi-stimulus listening test and the Fr\'echet Audio Distance and found it performed competitively across the tested timbral domains. Our method significantly outperformed the benchmarks in terms of generation speed, and achieved real-time performance on a consumer CPU, both with and without FastNEWT, suggesting it is a viable basis for future creative sound design tools.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Let&#39;s talk about Neural Waveshaping Synthesis!<br><br>📜 paper: <a href="https://t.co/TezK9Kv9iC">https://t.co/TezK9Kv9iC</a><br>💻 code: <a href="https://t.co/62Ytf1HoU1">https://t.co/62Ytf1HoU1</a><br>⏯ colab: <a href="https://t.co/tN8nyE6EIC">https://t.co/tN8nyE6EIC</a><br>🔗 website: <a href="https://t.co/YFcZ5WhYOi">https://t.co/YFcZ5WhYOi</a><br>🎶 audio: <a href="https://t.co/U84bHLzn3v">https://t.co/U84bHLzn3v</a> <a href="https://t.co/4z8OV7rDFz">pic.twitter.com/4z8OV7rDFz</a></p>&mdash; Ben Hayes (@benhayesmusic) <a href="https://twitter.com/benhayesmusic/status/1414873223209136141?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Waveshaping Synthesis<br>pdf: <a href="https://t.co/IzScKgHtpp">https://t.co/IzScKgHtpp</a><br>abs: <a href="https://t.co/9V7w3XTsTi">https://t.co/9V7w3XTsTi</a><br>github: <a href="https://t.co/zVFg0k6OvJ">https://t.co/zVFg0k6OvJ</a><br>significantly outperformed the benchmarks in terms of generation speed, and achieved real-time performance on a consumer CPU, both with and without FastNEWT <a href="https://t.co/1Y5nDLrpVF">pic.twitter.com/1Y5nDLrpVF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414756821345214470?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. CoBERL: Contrastive BERT for Reinforcement Learning

Andrea Banino, Adrià Puidomenech Badia, Jacob Walker, Tim Scholtes, Jovana Mitrovic, Charles Blundell

- retweets: 51, favorites: 26 (07/14/2021 10:56:42)

- links: [abs](https://arxiv.org/abs/2107.05431) | [pdf](https://arxiv.org/pdf/2107.05431)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Many reinforcement learning (RL) agents require a large amount of experience to solve tasks. We propose Contrastive BERT for RL (CoBERL), an agent that combines a new contrastive loss and a hybrid LSTM-transformer architecture to tackle the challenge of improving data efficiency. CoBERL enables efficient, robust learning from pixels across a wide range of domains. We use bidirectional masked prediction in combination with a generalization of recent contrastive methods to learn better representations for transformers in RL, without the need of hand engineered data augmentations. We find that CoBERL consistently improves performance across the full Atari suite, a set of control tasks and a challenging 3D environment.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CoBERL: Contrastive BERT for Reinforcement Learning<br>pdf: <a href="https://t.co/CsLvVCLORb">https://t.co/CsLvVCLORb</a><br>abs: <a href="https://t.co/vSYMJ7ZjSb">https://t.co/vSYMJ7ZjSb</a><br><br>consistently improves performance across the full Atari suite, a set of control tasks and a challenging 3D environment <a href="https://t.co/Yq5YvMHfEt">pic.twitter.com/Yq5YvMHfEt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414755055304052739?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Collective intelligence and the blockchain: Technology, communities and  social experiments

Andrea Baronchelli

- retweets: 42, favorites: 32 (07/14/2021 10:56:42)

- links: [abs](https://arxiv.org/abs/2107.05527) | [pdf](https://arxiv.org/pdf/2107.05527)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [econ.GN](https://arxiv.org/list/econ.GN/recent)

Blockchains are still perceived chiefly as a new technology. But each blockchain is also a community and a social experiment, built around social consensus. Here I discuss three examples showing how collective intelligence can help, threat or capitalize on blockchain-based ecosystems. They concern the immutability of smart contracts, code transparency and new forms of property. The examples show that more research, new norms and, eventually, laws are needed to manage the interaction between collective behaviour and the blockchain technology. Insights from researchers in collective intelligence can help society rise up to the challenge.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I wrote a short commentary to stress that each blockchain is a community and a social experiment. <br>This challenges current norms and presents risks. But it also untaps big opportunities.<a href="https://t.co/5WOLLDe3Gf">https://t.co/5WOLLDe3Gf</a>  <a href="https://t.co/uFlPiGkkvm">https://t.co/uFlPiGkkvm</a></p>&mdash; Andrea Baronchelli (@a_baronca) <a href="https://twitter.com/a_baronca/status/1414888932731498502?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. "A Virus Has No Religion": Analyzing Islamophobia on Twitter During the  COVID-19 Outbreak

Mohit Chandra, Manvith Reddy, Shradha Sehgal, Saurabh Gupta, Arun Balaji Buduru, Ponnurangam Kumaraguru

- retweets: 0, favorites: 57 (07/14/2021 10:56:42)

- links: [abs](https://arxiv.org/abs/2107.05104) | [pdf](https://arxiv.org/pdf/2107.05104)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

The COVID-19 pandemic has disrupted people's lives driving them to act in fear, anxiety, and anger, leading to worldwide racist events in the physical world and online social networks. Though there are works focusing on Sinophobia during the COVID-19 pandemic, less attention has been given to the recent surge in Islamophobia. A large number of positive cases arising out of the religious Tablighi Jamaat gathering has driven people towards forming anti-Muslim communities around hashtags like #coronajihad, #tablighijamaatvirus on Twitter. In addition to the online spaces, the rise in Islamophobia has also resulted in increased hate crimes in the real world. Hence, an investigation is required to create interventions. To the best of our knowledge, we present the first large-scale quantitative study linking Islamophobia with COVID-19.   In this paper, we present CoronaIslam dataset which focuses on anti-Muslim hate spanning four months, with over 410,990 tweets from 244,229 unique users. We use this dataset to perform longitudinal analysis. We find the relation between the trend on Twitter with the offline events that happened over time, measure the qualitative changes in the context associated with the Muslim community, and perform macro and micro topic analysis to find prevalent topics. We also explore the nature of the content, focusing on the toxicity of the URLs shared within the tweets present in the CoronaIslam dataset. Apart from the content-based analysis, we focus on user analysis, revealing that the portrayal of religion as a symbol of patriotism played a crucial role in deciding how the Muslim community was perceived during the pandemic. Through these experiments, we reveal the existence of anti-Muslim rhetoric around COVID-19 in the Indian sub-continent.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Thrilled to share that our work &quot;A Virus Has No Religion: Analyzing <a href="https://twitter.com/hashtag/Islamophobia?src=hash&amp;ref_src=twsrc%5Etfw">#Islamophobia</a> on Twitter During the COVID-19 Outbreak&quot; has been accepted at <a href="https://twitter.com/ACMHT?ref_src=twsrc%5Etfw">@ACMHT</a>!  Work w/ the all-star team <a href="https://twitter.com/mohit__30?ref_src=twsrc%5Etfw">@mohit__30</a><a href="https://twitter.com/ManvithReddy18?ref_src=twsrc%5Etfw">@ManvithReddy18</a> <a href="https://twitter.com/reallysaurabh?ref_src=twsrc%5Etfw">@reallysaurabh</a> <a href="https://twitter.com/ponguru?ref_src=twsrc%5Etfw">@ponguru</a><br><br>arXiv: <a href="https://t.co/jGzJ7RYKWZ">https://t.co/jGzJ7RYKWZ</a><a href="https://twitter.com/hashtag/acmht21?src=hash&amp;ref_src=twsrc%5Etfw">#acmht21</a></p>&mdash; Shradha Sehgal (@shradhasgl) <a href="https://twitter.com/shradhasgl/status/1414840959339225088?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Topological synchronization: explosive transition and rhythmic phase

Lucille Calmon, Juan G. Restrepo, Joaquín J. Torres, Ginestra Bianconi

- retweets: 18, favorites: 34 (07/14/2021 10:56:42)

- links: [abs](https://arxiv.org/abs/2107.05107) | [pdf](https://arxiv.org/pdf/2107.05107)
- [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [nlin.AO](https://arxiv.org/list/nlin.AO/recent) | [physics.bio-ph](https://arxiv.org/list/physics.bio-ph/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

Topological signals defined on nodes, links and higher dimensional simplices define the dynamical state of a network or of a simplicial complex. As such, topological signals are attracting increasing attention in network theory, dynamical systems, signal processing and machine learning. Topological signals defined on the nodes are typically studied in network dynamics, while topological signals defined on links are much less explored. Here we investigate topological synchronization describing locally coupled topological signals defined on the nodes and on the links of a network. The dynamics of signals defined on the nodes is affected by a phase lag depending on the dynamical state of nearby links and vice versa, the dynamics of topological signals defined on the links is affected by a phase lag depending on the dynamical state of nearby nodes. We show that topological synchronization on a fully connected network is explosive and leads to a discontinuous forward transition and a continuous backward transition. The analytical investigation of the phase diagram provides an analytical expression for the critical threshold of the discontinuous explosive synchronization. The model also displays an exotic coherent synchronized phase, also called rhythmic phase, characterized by having non-stationary order parameters which can shed light on topological mechanisms for the emergence of brain rhythms.




# 16. ReconVAT: A Semi-Supervised Automatic Music Transcription Framework for  Low-Resource Real-World Data

Kin Wai Cheuk, Dorien Herremans, Li Su

- retweets: 30, favorites: 21 (07/14/2021 10:56:42)

- links: [abs](https://arxiv.org/abs/2107.04954) | [pdf](https://arxiv.org/pdf/2107.04954)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Most of the current supervised automatic music transcription (AMT) models lack the ability to generalize. This means that they have trouble transcribing real-world music recordings from diverse musical genres that are not presented in the labelled training data. In this paper, we propose a semi-supervised framework, ReconVAT, which solves this issue by leveraging the huge amount of available unlabelled music recordings. The proposed ReconVAT uses reconstruction loss and virtual adversarial training. When combined with existing U-net models for AMT, ReconVAT achieves competitive results on common benchmark datasets such as MAPS and MusicNet. For example, in the few-shot setting for the string part version of MusicNet, ReconVAT achieves F1-scores of 61.0% and 41.6% for the note-wise and note-with-offset-wise metrics respectively, which translates into an improvement of 22.2% and 62.5% compared to the supervised baseline model. Our proposed framework also demonstrates the potential of continual learning on new data, which could be useful in real-world applications whereby new data is constantly available.




# 17. Hierarchical Neural Dynamic Policies

Shikhar Bahl, Abhinav Gupta, Deepak Pathak

- retweets: 20, favorites: 31 (07/14/2021 10:56:42)

- links: [abs](https://arxiv.org/abs/2107.05627) | [pdf](https://arxiv.org/pdf/2107.05627)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent)

We tackle the problem of generalization to unseen configurations for dynamic tasks in the real world while learning from high-dimensional image input. The family of nonlinear dynamical system-based methods have successfully demonstrated dynamic robot behaviors but have difficulty in generalizing to unseen configurations as well as learning from image inputs. Recent works approach this issue by using deep network policies and reparameterize actions to embed the structure of dynamical systems but still struggle in domains with diverse configurations of image goals, and hence, find it difficult to generalize. In this paper, we address this dichotomy by leveraging embedding the structure of dynamical systems in a hierarchical deep policy learning framework, called Hierarchical Neural Dynamical Policies (H-NDPs). Instead of fitting deep dynamical systems to diverse data directly, H-NDPs form a curriculum by learning local dynamical system-based policies on small regions in state-space and then distill them into a global dynamical system-based policy that operates only from high-dimensional images. H-NDPs additionally provide smooth trajectories, a strong safety benefit in the real world. We perform extensive experiments on dynamic tasks both in the real world (digit writing, scooping, and pouring) and simulation (catching, throwing, picking). We show that H-NDPs are easily integrated with both imitation as well as reinforcement learning setups and achieve state-of-the-art results. Video results are at https://shikharbahl.github.io/hierarchical-ndps/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical Neural Dynamic Policies<br>pdf: <a href="https://t.co/SfwXA0Agiq">https://t.co/SfwXA0Agiq</a><br>abs: <a href="https://t.co/p2sCKWMFXP">https://t.co/p2sCKWMFXP</a><br>project page: <a href="https://t.co/uLzK0UW3b7">https://t.co/uLzK0UW3b7</a> <a href="https://t.co/hY4E7ytSo5">pic.twitter.com/hY4E7ytSo5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414763882233667591?ref_src=twsrc%5Etfw">July 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



