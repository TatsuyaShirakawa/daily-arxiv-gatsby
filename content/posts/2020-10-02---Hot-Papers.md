---
title: Hot Papers 2020-10-02
date: 2020-10-03T13:51:55.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-02"
socialImage: "/media/flying-marine.jpg"

---

# 1. Interpreting Graph Neural Networks for NLP With Differentiable Edge  Masking

Michael Sejr Schlichtkrull, Nicola De Cao, Ivan Titov

- retweets: 5162, favorites: 309 (10/03/2020 13:51:55)

- links: [abs](https://arxiv.org/abs/2010.00577) | [pdf](https://arxiv.org/pdf/2010.00577)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Graph neural networks (GNNs) have become a popular approach to integrating structural inductive biases into NLP models. However, there has been little work on interpreting them, and specifically on understanding which parts of the graphs (e.g. syntactic trees or co-reference structures) contribute to a prediction. In this work, we introduce a post-hoc method for interpreting the predictions of GNNs which identifies unnecessary edges. Given a trained GNN model, we learn a simple classifier that, for every edge in every layer, predicts if that edge can be dropped. We demonstrate that such a classifier can be trained in a fully differentiable fashion, employing stochastic gates and encouraging sparsity through the expected $L_0$ norm. We use our technique as an attribution method to analyze GNN models for two tasks -- question answering and semantic role labeling -- providing insights into the information flow in these models. We show that we can drop a large proportion of edges without deteriorating the performance of the model, while we can analyse the remaining edges for interpreting model predictions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New hot pre-print 🔥Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking🔥 <a href="https://t.co/1eRIMPzttX">https://t.co/1eRIMPzttX</a><br>We show you can learn to remove most of the edges in GNNs such that the remaning ones are interpretable!<br>with <a href="https://twitter.com/michael_sejr?ref_src=twsrc%5Etfw">@michael_sejr</a> <a href="https://twitter.com/iatitov?ref_src=twsrc%5Etfw">@iatitov</a> <a href="https://t.co/0xl3U8bxiB">pic.twitter.com/0xl3U8bxiB</a></p>&mdash; Nicola De Cao (@nicola_decao) <a href="https://twitter.com/nicola_decao/status/1311978724234067969?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very happy to share our new preprint on interpretability for graph neural networks <a href="https://t.co/HGacWpiL86">https://t.co/HGacWpiL86</a>!🚀<br><br>Differentiable masking reveals which edges a GNN uses, and which can be discarded. With <a href="https://twitter.com/nicola_decao?ref_src=twsrc%5Etfw">@nicola_decao</a> <a href="https://twitter.com/iatitov?ref_src=twsrc%5Etfw">@iatitov</a> <a href="https://t.co/oWRRwMkccL">pic.twitter.com/oWRRwMkccL</a></p>&mdash; Michael Schlichtkrull (@michael_sejr) <a href="https://twitter.com/michael_sejr/status/1311978720022990848?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical  Clustering

Ines Chami, Albert Gu, Vaggos Chatziafratis, Christopher Ré

- retweets: 2756, favorites: 249 (10/03/2020 13:51:55)

- links: [abs](https://arxiv.org/abs/2010.00402) | [pdf](https://arxiv.org/pdf/2010.00402)
- [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Similarity-based Hierarchical Clustering (HC) is a classical unsupervised machine learning algorithm that has traditionally been solved with heuristic algorithms like Average-Linkage. Recently, Dasgupta reframed HC as a discrete optimization problem by introducing a global cost function measuring the quality of a given tree. In this work, we provide the first continuous relaxation of Dasgupta's discrete optimization problem with provable quality guarantees. The key idea of our method, HypHC, is showing a direct correspondence from discrete trees to continuous representations (via the hyperbolic embeddings of their leaf nodes) and back (via a decoding algorithm that maps leaf embeddings to a dendrogram), allowing us to search the space of discrete binary trees with continuous optimization. Building on analogies between trees and hyperbolic space, we derive a continuous analogue for the notion of lowest common ancestor, which leads to a continuous relaxation of Dasgupta's discrete objective. We can show that after decoding, the global minimizer of our continuous relaxation yields a discrete tree with a (1 + epsilon)-factor approximation for Dasgupta's optimal tree, where epsilon can be made arbitrarily small and controls optimization challenges. We experimentally evaluate HypHC on a variety of HC benchmarks and find that even approximate solutions found with gradient descent have superior clustering quality than agglomerative heuristics or other gradient based algorithms. Finally, we highlight the flexibility of HypHC using end-to-end training in a downstream classification task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very happy to share our recent work on gradient-based hierarchical clustering via hyperbolic embeddings (to appear at NeurIPS)! 1/6<br><br>code: <a href="https://t.co/rPfTk42efM">https://t.co/rPfTk42efM</a><br>Paper: <a href="https://t.co/V0hHGn1SKI">https://t.co/V0hHGn1SKI</a><br><br>A huge thanks to my collaborators <a href="https://twitter.com/_albertgu?ref_src=twsrc%5Etfw">@_albertgu</a> Vaggos Chatziafratis <a href="https://twitter.com/HazyResearch?ref_src=twsrc%5Etfw">@HazyResearch</a> <a href="https://t.co/o01rTgVcfg">pic.twitter.com/o01rTgVcfg</a></p>&mdash; Ines Chami (@chamii22) <a href="https://twitter.com/chamii22/status/1311966730445611008?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. FSD50K: an Open Dataset of Human-Labeled Sound Events

Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra

- retweets: 2149, favorites: 153 (10/03/2020 13:51:56)

- links: [abs](https://arxiv.org/abs/2010.00475) | [pdf](https://arxiv.org/pdf/2010.00475)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Most existing datasets for sound event recognition (SER) are relatively small and/or domain-specific, with the exception of AudioSet, based on a massive amount of audio tracks from YouTube videos and encompassing over 500 classes of everyday sounds. However, AudioSet is not an open dataset---its release consists of pre-computed audio features (instead of waveforms), which limits the adoption of some SER methods. Downloading the original audio tracks is also problematic due to constituent YouTube videos gradually disappearing and usage rights issues, which casts doubts over the suitability of this resource for systems' benchmarking. To provide an alternative benchmark dataset and thus foster SER research, we introduce FSD50K, an open dataset containing over 51k audio clips totalling over 100h of audio manually labeled using 200 classes drawn from the AudioSet Ontology. The audio clips are licensed under Creative Commons licenses, making the dataset freely distributable (including waveforms). We provide a detailed description of the FSD50K creation process, tailored to the particularities of Freesound data, including challenges encountered and solutions adopted. We include a comprehensive dataset characterization along with discussion of limitations and key factors to allow its audio-informed usage. Finally, we conduct sound event classification experiments to provide baseline systems as well as insight on the main factors to consider when splitting Freesound audio data for SER. Our goal is to develop a dataset to be widely adopted by the community as a new open benchmark for SER research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔊Happy to announce FSD50K: the new open dataset of human-labeled sound events! Over 51k Freesound audio clips, totalling over 100h of audio manually labeled using 200 classes drawn from the AudioSet Ontology.<br><br>Paper: <a href="https://t.co/fn5NSsdkgy">https://t.co/fn5NSsdkgy</a><br>Dataset: <a href="https://t.co/DmeCDQj6yW">https://t.co/DmeCDQj6yW</a> <a href="https://t.co/oKzW55LGWp">pic.twitter.com/oKzW55LGWp</a></p>&mdash; Eduardo Fonseca (@edfonseca_) <a href="https://twitter.com/edfonseca_/status/1312093703780134914?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Dynamic Facial Asset and Rig Generation from a Single Scan

Jiaman Li, Zhengfei Kuang, Yajie Zhao, Mingming He, Karl Bladin, Hao Li

- retweets: 446, favorites: 81 (10/03/2020 13:51:56)

- links: [abs](https://arxiv.org/abs/2010.00560) | [pdf](https://arxiv.org/pdf/2010.00560)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

The creation of high-fidelity computer-generated (CG) characters used in film and gaming requires intensive manual labor and a comprehensive set of facial assets to be captured with complex hardware, resulting in high cost and long production cycles. In order to simplify and accelerate this digitization process, we propose a framework for the automatic generation of high-quality dynamic facial assets, including rigs which can be readily deployed for artists to polish. Our framework takes a single scan as input to generate a set of personalized blendshapes, dynamic and physically-based textures, as well as secondary facial components (e.g., teeth and eyeballs). Built upon a facial database consisting of pore-level details, with over $4,000$ scans of varying expressions and identities, we adopt a self-supervised neural network to learn personalized blendshapes from a set of template expressions. We also model the joint distribution between identities and expressions, enabling the inference of the full set of personalized blendshapes with dynamic appearances from a single neutral input scan. Our generated personalized face rig assets are seamlessly compatible with cutting-edge industry pipelines for facial animation and rendering. We demonstrate that our framework is robust and effective by inferring on a wide range of novel subjects, and illustrate compelling rendering results while animating faces with generated customized physically-based dynamic textures.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dynamic Facial Asset and Rig Generation from a Single Scan<br>pdf: <a href="https://t.co/k8p2uIqqpn">https://t.co/k8p2uIqqpn</a><br>abs: <a href="https://t.co/nYUoI5qQ11">https://t.co/nYUoI5qQ11</a> <a href="https://t.co/QqISLLhLsg">pic.twitter.com/QqISLLhLsg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1311832183078879232?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Multi-agent Social Reinforcement Learning Improves Generalization

Kamal Ndousse, Douglas Eck, Sergey Levine, Natasha Jaques

- retweets: 306, favorites: 121 (10/03/2020 13:51:56)

- links: [abs](https://arxiv.org/abs/2010.00581) | [pdf](https://arxiv.org/pdf/2010.00581)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Social learning is a key component of human and animal intelligence. By taking cues from the behavior of experts in their environment, social learners can acquire sophisticated behavior and rapidly adapt to new circumstances. This paper investigates whether independent reinforcement learning (RL) agents in a multi-agent environment can use social learning to improve their performance using cues from other agents. We find that in most circumstances, vanilla model-free RL agents do not use social learning, even in environments in which individual exploration is expensive. We analyze the reasons for this deficiency, and show that by introducing a model-based auxiliary loss we are able to train agents to lever-age cues from experts to solve hard exploration tasks. The generalized social learning policy learned by these agents allows them to not only outperform the experts with which they trained, but also achieve better zero-shot transfer performance than solo learners when deployed to novel environments with experts. In contrast, agents that have not learned to rely on social learning generalize poorly and do not succeed in the transfer task. Further,we find that by mixing multi-agent and solo training, we can obtain agents that use social learning to out-perform agents trained alone, even when experts are not avail-able. This demonstrates that social learning has helped improve agents' representation of the task itself. Our results indicate that social learning can enable RL agents to not only improve performance on the task at hand, but improve generalization to novel environments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Big personal milestone! My first ML paper is on arXiv: <a href="https://t.co/jRsGIBbFYF">https://t.co/jRsGIBbFYF</a> <br>We propose a simple method that helps RL agents in shared environments learn from one another, and show that the learned social policies improve zero-shot transfer performance in new environments. 👇</p>&mdash; Kamal Ndousse (@kandouss) <a href="https://twitter.com/kandouss/status/1312164325927383040?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Understanding Self-supervised Learning with Dual Deep Networks

Yuandong Tian, Lantao Yu, Xinlei Chen, Surya Ganguli

- retweets: 218, favorites: 99 (10/03/2020 13:51:56)

- links: [abs](https://arxiv.org/abs/2010.00578) | [pdf](https://arxiv.org/pdf/2010.00578)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We propose a novel theoretical framework to understand self-supervised learning methods that employ dual pairs of deep ReLU networks (e.g., SimCLR, BYOL). First, we prove that in each SGD update of SimCLR, the weights at each layer are updated by a \emph{covariance operator} that specifically amplifies initial random selectivities that vary across data samples but survive averages over data augmentations, which we show leads to the emergence of hierarchical features, if the input data are generated from a hierarchical latent tree model. With the same framework, we also show analytically that BYOL works due to an implicit contrastive term, acting as an approximate covariance operator. The term is formed by the inter-play between the zero-mean operation of BatchNorm and the extra predictor in the online network. Extensive ablation studies justify our theoretical findings.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr"><a href="https://t.co/EeFWVpSm66">https://t.co/EeFWVpSm66</a> We propose an analytic framework explaining how methods like SimCLR/BYOL learns mid-level features: we find &quot;covariance operator&quot; that drives weight update per-layer in deep ReLU nets. It also shows BN+predictor in BYOL gives an implicit contrastive term.</p>&mdash; Yuandong Tian (@tydsh) <a href="https://twitter.com/tydsh/status/1312117271284772864?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. How LSTM Encodes Syntax: Exploring Context Vectors and Semi-Quantization  on Natural Text

Chihiro Shibata, Kei Uchiumi, Daichi Mochihashi

- retweets: 132, favorites: 66 (10/03/2020 13:51:56)

- links: [abs](https://arxiv.org/abs/2010.00363) | [pdf](https://arxiv.org/pdf/2010.00363)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Long Short-Term Memory recurrent neural network (LSTM) is widely used and known to capture informative long-term syntactic dependencies. However, how such information are reflected in its internal vectors for natural text has not yet been sufficiently investigated. We analyze them by learning a language model where syntactic structures are implicitly given. We empirically show that the context update vectors, i.e. outputs of internal gates, are approximately quantized to binary or ternary values to help the language model to count the depth of nesting accurately, as Suzgun et al. (2019) recently show for synthetic Dyck languages. For some dimensions in the context vector, we show that their activations are highly correlated with the depth of phrase structures, such as VP and NP. Moreover, with an $L_1$ regularization, we also found that it can accurately predict whether a word is inside a phrase structure or not from a small number of components of the context vector. Even for the case of learning from raw text, context vectors are shown to still correlate well with the phrase structures. Finally, we show that natural clusters of the functional words and the part of speeches that trigger phrases are represented in a small but principal subspace of the context-update vector of LSTM.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">柴田さん(東京工科大)、内海さん(デンソーITLab)との共同研究がCOLING 2020に無事採択されました。&quot;How LSTM Encodes Syntax: Exploring Context Vectors and Semi-Quantization on Natural Text&quot;<a href="https://t.co/U9GCcuyzvW">https://t.co/U9GCcuyzvW</a><br>LSTMの内部状態がほぼ離散的にスタック状になっていることを示した論文です。</p>&mdash; Daichi Mochihashi (@daiti_m) <a href="https://twitter.com/daiti_m/status/1312027502332665857?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked  Language Models

Nikita Nangia, Clara Vania, Rasika Bhalerao, Samuel R. Bowman

- retweets: 156, favorites: 38 (10/03/2020 13:51:56)

- links: [abs](https://arxiv.org/abs/2010.00133) | [pdf](https://arxiv.org/pdf/2010.00133)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Pretrained language models, especially masked language models (MLMs) have seen success across many NLP tasks. However, there is ample evidence that they use the cultural biases that are undoubtedly present in the corpora they are trained on, implicitly creating harm with biased representations. To measure some forms of social bias in language models against protected demographic groups in the US, we introduce the Crowdsourced Stereotype Pairs benchmark (CrowS-Pairs). CrowS-Pairs has 1508 examples that cover stereotypes dealing with nine types of bias, like race, religion, and age. In CrowS-Pairs a model is presented with two sentences: one that is more stereotyping and another that is less stereotyping. The data focuses on stereotypes about historically disadvantaged groups and contrasts them with advantaged groups. We find that all three of the widely-used MLMs we evaluate substantially favor sentences that express stereotypes in every category in CrowS-Pairs. As work on building less biased models advances, this dataset can be used as a benchmark to evaluate progress.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hello <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> twitter 👋 <br>New <a href="https://twitter.com/hashtag/EMNLP2020?src=hash&amp;ref_src=twsrc%5Etfw">#EMNLP2020</a> paper! We introduce CrowS-Pairs, a crowdsourced dataset to measure the degree to which US social stereotypes are captured by MLMs. <a href="https://t.co/v68HIntq8X">https://t.co/v68HIntq8X</a> 1/7</p>&mdash; Nikita Nangia (@meloncholist) <a href="https://twitter.com/meloncholist/status/1312115601574248448?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Computing Graph Neural Networks: A Survey from Algorithms to  Accelerators

Sergi Abadal, Akshay Jain, Robert Guirado, Jorge López-Alonso, Eduard Alarcón

- retweets: 91, favorites: 31 (10/03/2020 13:51:57)

- links: [abs](https://arxiv.org/abs/2010.00130) | [pdf](https://arxiv.org/pdf/2010.00130)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Graph Neural Networks (GNNs) have exploded onto the machine learning scene in recent years owing to their capability to model and learn from graph-structured data. Such an ability has strong implications in a wide variety of fields whose data is inherently relational, for which conventional neural networks do not perform well. Indeed, as recent reviews can attest, research in the area of GNNs has grown rapidly and has lead to the development of a variety of GNN algorithm variants as well as to the exploration of groundbreaking applications in chemistry, neurology, electronics, or communication networks, among others. At the current stage of research, however, the efficient processing of GNNs is still an open challenge for several reasons. Besides of their novelty, GNNs are hard to compute due to their dependence on the input graph, their combination of dense and very sparse operations, or the need to scale to huge graphs in some applications. In this context, this paper aims to make two main contributions. On the one hand, a review of the field of GNNs is presented from the perspective of computing. This includes a brief tutorial on the GNN fundamentals, an overview of the evolution of the field in the last decade, and a summary of operations carried out in the multiple phases of different GNN algorithm variants. On the other hand, an in-depth analysis of current software and hardware acceleration schemes is provided, from which a hardware-software, graph-aware, and communication-centric vision for GNN accelerators is distilled.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Computing Graph Neural Networks: A Survey from Algorithms to Accelerators <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a><a href="https://t.co/hdvZEjdExu">https://t.co/hdvZEjdExu</a> <a href="https://t.co/g6A268HPiD">pic.twitter.com/g6A268HPiD</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1312044680281026562?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Agnostic Learning of Halfspaces with Gradient Descent via Soft Margins

Spencer Frei, Yuan Cao, Quanquan Gu

- retweets: 42, favorites: 20 (10/03/2020 13:51:57)

- links: [abs](https://arxiv.org/abs/2010.00539) | [pdf](https://arxiv.org/pdf/2010.00539)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.OC](https://arxiv.org/list/math.OC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We analyze the properties of gradient descent on convex surrogates for the zero-one loss for the agnostic learning of linear halfspaces. If $\mathsf{OPT}$ is the best classification error achieved by a halfspace, by appealing to the notion of soft margins we are able to show that gradient descent finds halfspaces with classification error $\tilde O(\mathsf{OPT}^{1/2}) + \varepsilon$ in $\mathrm{poly}(d,1/\varepsilon)$ time and sample complexity for a broad class of distributions that includes log-concave isotropic distributions as a subclass. Along the way we answer a question recently posed by Ji et al. (2020) on how the tail behavior of a loss function can affect sample complexity and runtime guarantees for gradient descent.




# 11. Metrics for Benchmarking and Uncertainty Quantification: Quality,  Applicability, and a Path to Best Practices for Machine Learning in Chemistry

Gaurav Vishwakarma, Aditya Sonpal, Johannes Hachmann

- retweets: 39, favorites: 17 (10/03/2020 13:51:57)

- links: [abs](https://arxiv.org/abs/2010.00110) | [pdf](https://arxiv.org/pdf/2010.00110)
- [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This review aims to draw attention to two issues of concern when we set out to make machine learning work in the chemical and materials domain, i.e., statistical loss function metrics for the validation and benchmarking of data-derived models, and the uncertainty quantification of predictions made by them. They are often overlooked or underappreciated topics as chemists typically only have limited training in statistics. Aside from helping to assess the quality, reliability, and applicability of a given model, these metrics are also key to comparing the performance of different models and thus for developing guidelines and best practices for the successful application of machine learning in chemistry.



