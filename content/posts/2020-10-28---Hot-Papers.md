---
title: Hot Papers 2020-10-28
date: 2020-10-29T09:41:19.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-28"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-28"
socialImage: "/media/flying-marine.jpg"

---

# 1. COG: Connecting New Skills to Past Experience with Offline Reinforcement  Learning

Avi Singh, Albert Yu, Jonathan Yang, Jesse Zhang, Aviral Kumar, Sergey Levine

- retweets: 556, favorites: 131 (10/29/2020 09:41:19)

- links: [abs](https://arxiv.org/abs/2010.14500) | [pdf](https://arxiv.org/pdf/2010.14500)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Reinforcement learning has been applied to a wide variety of robotics problems, but most of such applications involve collecting data from scratch for each new task. Since the amount of robot data we can collect for any single task is limited by time and cost considerations, the learned behavior is typically narrow: the policy can only execute the task in a handful of scenarios that it was trained on. What if there was a way to incorporate a large amount of prior data, either from previously solved tasks or from unsupervised or undirected environment interaction, to extend and generalize learned behaviors? While most prior work on extending robotic skills using pre-collected data focuses on building explicit hierarchies or skill decompositions, we show in this paper that we can reuse prior data to extend new skills simply through dynamic programming. We show that even when the prior data does not actually succeed at solving the new task, it can still be utilized for learning a better policy, by providing the agent with a broader understanding of the mechanics of its environment. We demonstrate the effectiveness of our approach by chaining together several behaviors seen in prior datasets for solving a new task, with our hardest experimental setting involving composing four robotic skills in a row: picking, placing, drawer opening, and grasping, where a +1/0 sparse reward is provided only on task completion. We train our policies in an end-to-end fashion, mapping high-dimensional image observations to low-level robot control commands, and present results in both simulated and real world domains. Additional materials and source code can be found on our project website: https://sites.google.com/view/cog-rl

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">COG uses offline RL (CQL) to incorporate past data to generalize new skills. E.g., a robot picking a ball from a drawer will know it must open the drawer if it is closed, even if it never performed these steps together.<a href="https://t.co/i4rVXZoQbF">https://t.co/i4rVXZoQbF</a><a href="https://t.co/6A9INvrB5G">https://t.co/6A9INvrB5G</a><br><br>thread -&gt; <a href="https://t.co/LzqiYyLJmd">pic.twitter.com/LzqiYyLJmd</a></p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1321259357141528576?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Scientific intuition inspired by machine learning generated hypotheses

Pascal Friederich, Mario Krenn, Isaac Tamblyn, Alan Aspuru-Guzik

- retweets: 384, favorites: 111 (10/29/2020 09:41:19)

- links: [abs](https://arxiv.org/abs/2010.14236) | [pdf](https://arxiv.org/pdf/2010.14236)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CE](https://arxiv.org/list/cs.CE/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [quant-ph](https://arxiv.org/list/quant-ph/recent)

Machine learning with application to questions in the physical sciences has become a widely used tool, successfully applied to classification, regression and optimization tasks in many areas. Research focus mostly lies in improving the accuracy of the machine learning models in numerical predictions, while scientific understanding is still almost exclusively generated by human researchers analysing numerical results and drawing conclusions. In this work, we shift the focus on the insights and the knowledge obtained by the machine learning models themselves. In particular, we study how it can be extracted and used to inspire human scientists to increase their intuitions and understanding of natural systems. We apply gradient boosting in decision trees to extract human interpretable insights from big data sets from chemistry and physics. In chemistry, we not only rediscover widely know rules of thumb but also find new interesting motifs that tell us how to control solubility and energy levels of organic molecules. At the same time, in quantum physics, we gain new understanding on experiments for quantum entanglement. The ability to go beyond numerics and to enter the realm of scientific insight and hypothesis generation opens the door to use machine learning to accelerate the discovery of conceptual understanding in some of the most challenging domains of science.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How to get new concepts &amp; ideas with ML? 🤔<br><br>We use Graph-based gradient-boosting to get scientific insights. Beyond rediscovery, we learn new things!<br><br>For chemistry &amp; physics: <a href="https://t.co/KbPkwgoJkv">https://t.co/KbPkwgoJkv</a><br><br>Spearheaded by <a href="https://twitter.com/P_Friederich?ref_src=twsrc%5Etfw">@P_Friederich</a> w/ <a href="https://twitter.com/itamblyn?ref_src=twsrc%5Etfw">@itamblyn</a> &amp; <a href="https://twitter.com/A_Aspuru_Guzik?ref_src=twsrc%5Etfw">@A_Aspuru_Guzik</a> <a href="https://twitter.com/hashtag/TeamCanada?src=hash&amp;ref_src=twsrc%5Etfw">#TeamCanada</a> 🇨🇦😁 <a href="https://t.co/gSo20V2YYf">pic.twitter.com/gSo20V2YYf</a></p>&mdash; Mario Krenn (@MarioKrenn6240) <a href="https://twitter.com/MarioKrenn6240/status/1321261468902477824?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. MELD: Meta-Reinforcement Learning from Images via Latent State Models

Tony Z. Zhao, Anusha Nagabandi, Kate Rakelly, Chelsea Finn, Sergey Levine

- retweets: 230, favorites: 118 (10/29/2020 09:41:19)

- links: [abs](https://arxiv.org/abs/2010.13957) | [pdf](https://arxiv.org/pdf/2010.13957)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Meta-reinforcement learning algorithms can enable autonomous agents, such as robots, to quickly acquire new behaviors by leveraging prior experience in a set of related training tasks. However, the onerous data requirements of meta-training compounded with the challenge of learning from sensory inputs such as images have made meta-RL challenging to apply to real robotic systems. Latent state models, which learn compact state representations from a sequence of observations, can accelerate representation learning from visual inputs. In this paper, we leverage the perspective of meta-learning as task inference to show that latent state models can \emph{also} perform meta-learning given an appropriately defined observation space. Building on this insight, we develop meta-RL with latent dynamics (MELD), an algorithm for meta-RL from images that performs inference in a latent state model to quickly acquire new skills given observations and rewards. MELD outperforms prior meta-RL methods on several simulated image-based robotic control problems, and enables a real WidowX robotic arm to insert an Ethernet cable into new locations given a sparse task completion signal after only $8$ hours of real world meta-training. To our knowledge, MELD is the first meta-RL algorithm trained in a real-world robotic control setting from images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper: MELD allows a real robot to adapt to new goals directly from image pixels<a href="https://t.co/lyhUgA61JJ">https://t.co/lyhUgA61JJ</a><br>The key idea is to _meld_ meta-RL and latent dynamics models for unified state &amp; task inference.<br><br>w/ <a href="https://twitter.com/tonyzzhao?ref_src=twsrc%5Etfw">@tonyzzhao</a>, A Nagabandi, K Rakelly, <a href="https://twitter.com/svlevine?ref_src=twsrc%5Etfw">@svlevine</a><br>to appear at <a href="https://twitter.com/corl_conf?ref_src=twsrc%5Etfw">@corl_conf</a> <a href="https://t.co/tyvMLZWjIS">pic.twitter.com/tyvMLZWjIS</a></p>&mdash; Chelsea Finn (@chelseabfinn) <a href="https://twitter.com/chelseabfinn/status/1321270766327877634?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Wavelet Flow: Fast Training of High Resolution Normalizing Flows

Jason J. Yu, Konstantinos G. Derpanis, Marcus A. Brubaker

- retweets: 121, favorites: 69 (10/29/2020 09:41:19)

- links: [abs](https://arxiv.org/abs/2010.13821) | [pdf](https://arxiv.org/pdf/2010.13821)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Normalizing flows are a class of probabilistic generative models which allow for both fast density computation and efficient sampling and are effective at modelling complex distributions like images. A drawback among current methods is their significant training cost, sometimes requiring months of GPU training time to achieve state-of-the-art results. This paper introduces Wavelet Flow, a multi-scale, normalizing flow architecture based on wavelets. A Wavelet Flow has an explicit representation of signal scale that inherently includes models of lower resolution signals and conditional generation of higher resolution signals, i.e., super resolution. A major advantage of Wavelet Flow is the ability to construct generative models for high resolution data (e.g., 1024 x 1024 images) that are impractical with previous models. Furthermore, Wavelet Flow is competitive with previous normalizing flows in terms of bits per dimension on standard (low resolution) benchmarks while being up to 15x faster to train.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wavelet Flow: Fast Training of High Resolution Normalizing Flows<br>pdf: <a href="https://t.co/7J1ylyfx5i">https://t.co/7J1ylyfx5i</a><br>abs: <a href="https://t.co/BEZCJ07AsD">https://t.co/BEZCJ07AsD</a><br>project page: <a href="https://t.co/uqDeiNR9JO">https://t.co/uqDeiNR9JO</a> <a href="https://t.co/OCyjJ8SZgi">pic.twitter.com/OCyjJ8SZgi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1321280699194888193?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Toward Better Generalization Bounds with Locally Elastic Stability

Zhun Deng, Hangfeng He, Weijie J. Su

- retweets: 42, favorites: 51 (10/29/2020 09:41:19)

- links: [abs](https://arxiv.org/abs/2010.13988) | [pdf](https://arxiv.org/pdf/2010.13988)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Classical approaches in learning theory are often seen to yield very loose generalization bounds for deep neural networks. Using the example of "stability and generalization" \citep{bousquet2002stability}, however, we demonstrate that generalization bounds can be significantly improved by taking into account refined characteristics of modern neural networks. Specifically, this paper proposes a new notion of algorithmic stability termed \textit{locally elastic stability} in light of a certain phenomenon in the training of neural networks \citep{he2020local}. We prove that locally elastic stability implies a tighter generalization bound than that derived based on uniform stability in many situations. When applied to deep neural networks, our new generalization bound attaches much more meaningful confidence statements to the performance on unseen data than existing algorithmic stability notions, thereby shedding light on the effectiveness of modern neural networks in real-world applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Why do we need a *phenomenological* approach for understanding deep learning? Two new papers that use *local elasticity* for understanding the effectiveness of neural networks: <a href="https://t.co/3WtECW0y2g">https://t.co/3WtECW0y2g</a> and <a href="https://t.co/6MoYSoGlap">https://t.co/6MoYSoGlap</a> (NeurIPS 2020), from a phenomenological perspective</p>&mdash; Weijie Su (@weijie444) <a href="https://twitter.com/weijie444/status/1321483467042017280?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Apps Against the Spread: Privacy Implications and User Acceptance of  COVID-19-Related Smartphone Apps on Three Continents

Christine Utz, Steffen Becker, Theodor Schnitzler, Florian M. Farke, Franziska Herbert, Leonie Schaewitz, Martin Degeling, Markus Dürmuth

- retweets: 72, favorites: 12 (10/29/2020 09:41:20)

- links: [abs](https://arxiv.org/abs/2010.14245) | [pdf](https://arxiv.org/pdf/2010.14245)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

The COVID-19 pandemic has fueled the development of smartphone applications to assist disease management. These "corona apps" may require widespread adoption to be effective, which has sparked public debates about the privacy, security, and societal implications of government-backed health applications. We conducted a representative online study in Germany (n = 1,003), the US (n = 1,003), and China (n = 1,019) to investigate user acceptance of corona apps, using a vignette design based on the contextual integrity framework. We explored apps for contact tracing, symptom checks, quarantine enforcement, health certificates, and mere information. Our results provide insights into data processing practices that foster adoption and reveal significant differences between countries, with user acceptance being highest in China and lowest in the US. Chinese participants prefer the collection of personalized data, while German and US participants favor anonymity. Across countries, contact tracing is viewed more positively than quarantine enforcement, and technical malfunctions negatively impact user acceptance.




# 7. FragmentVC: Any-to-Any Voice Conversion by End-to-End Extracting and  Fusing Fine-Grained Voice Fragments With Attention

Yist Y. Lin, Chung-Ming Chien, Jheng-Hao Lin, Hung-yi Lee, Lin-shan Lee

- retweets: 45, favorites: 36 (10/29/2020 09:41:20)

- links: [abs](https://arxiv.org/abs/2010.14150) | [pdf](https://arxiv.org/pdf/2010.14150)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Any-to-any voice conversion aims to convert the voice from and to any speakers even unseen during training, which is much more challenging compared to one-to-one or many-to-many tasks, but much more attractive in real-world scenarios. In this paper we proposed FragmentVC, in which the latent phonetic structure of the utterance from the source speaker is obtained from Wav2Vec 2.0, while the spectral features of the utterance(s) from the target speaker are obtained from log mel-spectrograms. By aligning the hidden structures of the two different feature spaces with a two-stage training process, FragmentVC is able to extract fine-grained voice fragments from the target speaker utterance(s) and fuse them into the desired utterance, all based on the attention mechanism of Transformer as verified with analysis on attention maps, and is accomplished end-to-end. This approach is trained with reconstruction loss only without any disentanglement considerations between content and speaker information and doesn't require parallel data. Objective evaluation based on speaker verification and subjective evaluation with MOS both showed that this approach outperformed SOTA approaches, such as AdaIN-VC and AutoVC.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FragmentVC: Any-to-Any Voice Conversion by End-to-End Extracting and Fusing Fine-Grained Voice Fragments With Attention<br>pdf: <a href="https://t.co/dBkWbKisxB">https://t.co/dBkWbKisxB</a><br>abs: <a href="https://t.co/EzI3CbtR3f">https://t.co/EzI3CbtR3f</a> <a href="https://t.co/eyC2W3Y1yN">pic.twitter.com/eyC2W3Y1yN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1321256440661745670?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Generating 3D Molecular Structures Conditional on a Receptor Binding  Site with Deep Generative Models

Tomohide Masuda, Matthew Ragoza, David Ryan Koes

- retweets: 38, favorites: 33 (10/29/2020 09:41:20)

- links: [abs](https://arxiv.org/abs/2010.14442) | [pdf](https://arxiv.org/pdf/2010.14442)
- [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent)

Deep generative models have been applied with increasing success to the generation of two dimensional molecules as SMILES strings and molecular graphs. In this work we describe for the first time a deep generative model that can generate 3D molecular structures conditioned on a three-dimensional (3D) binding pocket. Using convolutional neural networks, we encode atomic density grids into separate receptor and ligand latent spaces. The ligand latent space is variational to support sampling of new molecules. A decoder network generates atomic densities of novel ligands conditioned on the receptor. Discrete atoms are then fit to these continuous densities to create molecular structures. We show that valid and unique molecules can be readily sampled from the variational latent space defined by a reference `seed' structure and generated structures have reasonable interactions with the binding site. As structures are sampled farther in latent space from the seed structure, the novelty of the generated structures increases, but the predicted binding affinity decreases. Overall, we demonstrate the feasibility of conditional 3D molecular structure generation and provide a starting point for methods that also explicitly optimize for desired molecular properties, such as high binding affinity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating 3D Molecular Structures Conditional on a Receptor Binding Site with Deep Generative Models<a href="https://t.co/Cn6w4Tetvm">https://t.co/Cn6w4Tetvm</a> <a href="https://twitter.com/hashtag/compchem?src=hash&amp;ref_src=twsrc%5Etfw">#compchem</a></p>&mdash; Jan Jensen (@janhjensen) <a href="https://twitter.com/janhjensen/status/1321364836635332609?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Parallel waveform synthesis based on generative adversarial networks  with voicing-aware conditional discriminators

Ryuichi Yamamoto, Eunwoo Song, Min-Jae Hwang, Jae-Min Kim

- retweets: 24, favorites: 46 (10/29/2020 09:41:20)

- links: [abs](https://arxiv.org/abs/2010.14151) | [pdf](https://arxiv.org/pdf/2010.14151)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.SP](https://arxiv.org/list/eess.SP/recent)

This paper proposes voicing-aware conditional discriminators for Parallel WaveGAN-based waveform synthesis systems. In this framework, we adopt a projection-based conditioning method that can significantly improve the discriminator's performance. Furthermore, the conventional discriminator is separated into two waveform discriminators for modeling voiced and unvoiced speech. As each discriminator learns the distinctive characteristics of the harmonic and noise components, respectively, the adversarial training process becomes more efficient, allowing the generator to produce more realistic speech waveforms. Subjective test results demonstrate the superiority of the proposed method over the conventional Parallel WaveGAN and WaveNet systems. In particular, our speaker-independently trained model within a FastSpeech 2 based text-to-speech framework achieves the mean opinion scores of 4.20, 4.18, 4.21, and 4.31 for four Japanese speakers, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new work is out on arXiv!<br><br>arXiv: <a href="https://t.co/pCopVzky5j">https://t.co/pCopVzky5j</a><br>Demo: <a href="https://t.co/qEtOcNnwe7">https://t.co/qEtOcNnwe7</a> <a href="https://t.co/5YrJcfhZUF">https://t.co/5YrJcfhZUF</a></p>&mdash; 山本りゅういち / Ryuichi Yamamoto (@r9y9) <a href="https://twitter.com/r9y9/status/1321416774504493056?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Random walks and community detection in hypergraphs

Timoteo Carletti, Duccio Fanelli, Renaud Lambiotte

- retweets: 42, favorites: 19 (10/29/2020 09:41:20)

- links: [abs](https://arxiv.org/abs/2010.14355) | [pdf](https://arxiv.org/pdf/2010.14355)
- [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [math.DS](https://arxiv.org/list/math.DS/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

We propose a one parameter family of random walk processes on hypergraphs, where a parameter biases the dynamics of the walker towards hyperedges of low or high cardinality. We show that for each value of the parameter the resulting process defines its own hypergraph projection on a weighted network. We then explore the differences between them by considering the community structure associated to each random walk process. To do so, we generalise the Markov stability framework to hypergraphs and test it on artificial and real-world hypergraphs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Random walks and community detection in hypergraphs. (arXiv:2010.14355v1 [cond-mat.stat-mech]) <a href="https://t.co/smxByp4RcG">https://t.co/smxByp4RcG</a></p>&mdash; NetScience (@net_science) <a href="https://twitter.com/net_science/status/1321382817272680449?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Examining the causal structures of deep neural networks using  information theory

Simon Mattsson, Eric J. Michaud, Erik Hoel

- retweets: 30, favorites: 25 (10/29/2020 09:41:20)

- links: [abs](https://arxiv.org/abs/2010.13871) | [pdf](https://arxiv.org/pdf/2010.13871)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Deep Neural Networks (DNNs) are often examined at the level of their response to input, such as analyzing the mutual information between nodes and data sets. Yet DNNs can also be examined at the level of causation, exploring "what does what" within the layers of the network itself. Historically, analyzing the causal structure of DNNs has received less attention than understanding their responses to input. Yet definitionally, generalizability must be a function of a DNN's causal structure since it reflects how the DNN responds to unseen or even not-yet-defined future inputs. Here, we introduce a suite of metrics based on information theory to quantify and track changes in the causal structure of DNNs during training. Specifically, we introduce the effective information (EI) of a feedforward DNN, which is the mutual information between layer input and output following a maximum-entropy perturbation. The EI can be used to assess the degree of causal influence nodes and edges have over their downstream targets in each layer. We show that the EI can be further decomposed in order to examine the sensitivity of a layer (measured by how well edges transmit perturbations) and the degeneracy of a layer (measured by how edge overlap interferes with transmission), along with estimates of the amount of integrated information of a layer. Together, these properties define where each layer lies in the "causal plane" which can be used to visualize how layer connectivity becomes more sensitive or degenerate over time, and how integration changes during training, revealing how the layer-by-layer causal structure differentiates. These results may help in understanding the generalization capabilities of DNNs and provide foundational tools for making DNNs both more generalizable and more explainable.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How do artificial neural networks generalize? The answer may be in their causal structure. In this new paper we use information theory to track nodes’ causal relationships becoming more sensitive or degenerate. Training traces a path in this “causal plane” <a href="https://t.co/fKRvYWr0mQ">https://t.co/fKRvYWr0mQ</a> <a href="https://t.co/w2YQrNvMnK">pic.twitter.com/w2YQrNvMnK</a></p>&mdash; Erik Hoel (@erikphoel) <a href="https://twitter.com/erikphoel/status/1321462223097790468?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Benchmarking Deep Learning Interpretability in Time Series Predictions

Aya Abdelsalam Ismail, Mohamed Gunady, Héctor Corrada Bravo, Soheil Feizi

- retweets: 30, favorites: 23 (10/29/2020 09:41:20)

- links: [abs](https://arxiv.org/abs/2010.13924) | [pdf](https://arxiv.org/pdf/2010.13924)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Saliency methods are used extensively to highlight the importance of input features in model predictions. These methods are mostly used in vision and language tasks, and their applications to time series data is relatively unexplored. In this paper, we set out to extensively compare the performance of various saliency-based interpretability methods across diverse neural architectures, including Recurrent Neural Network, Temporal Convolutional Networks, and Transformers in a new benchmark of synthetic time series data. We propose and report multiple metrics to empirically evaluate the performance of saliency methods for detecting feature importance over time using both precision (i.e., whether identified features contain meaningful signals) and recall (i.e., the number of features with signal identified as important). Through several experiments, we show that (i) in general, network architectures and saliency methods fail to reliably and accurately identify feature importance over time in time series data, (ii) this failure is mainly due to the conflation of time and feature domains, and (iii) the quality of saliency maps can be improved substantially by using our proposed two-step temporal saliency rescaling (TSR) approach that first calculates the importance of each time step before calculating the importance of each feature at a time step.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How well deep learning interpretation (saliency) methods work in &quot;time series predictions&quot;? We study this question in our <a href="https://twitter.com/hashtag/NeurIPS2020?src=hash&amp;ref_src=twsrc%5Etfw">#NeurIPS2020</a>  paper by developing a benchmark: <a href="https://t.co/dtYdQ1pbSe">https://t.co/dtYdQ1pbSe</a><br>Code/data: <a href="https://t.co/afTVSsyYe6">https://t.co/afTVSsyYe6</a><br>with <a href="https://twitter.com/asalam_91?ref_src=twsrc%5Etfw">@asalam_91</a>, M. Gunady and <a href="https://twitter.com/hcorrada?ref_src=twsrc%5Etfw">@hcorrada</a>  1/3 <a href="https://t.co/dha6gqIWyn">pic.twitter.com/dha6gqIWyn</a></p>&mdash; Soheil Feizi (@FeiziSoheil) <a href="https://twitter.com/FeiziSoheil/status/1321443840448958465?ref_src=twsrc%5Etfw">October 28, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



