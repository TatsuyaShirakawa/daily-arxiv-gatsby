---
title: Hot Papers 2021-07-01
date: 2021-07-02T11:58:00.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-01"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-01"
socialImage: "/media/flying-marine.jpg"

---

# 1. Small in-distribution changes in 3D perspective and lighting fool both  CNNs and Transformers

Spandan Madan, Tomotake Sasaki, Tzu-Mao Li, Xavier Boix, Hanspeter Pfister

- retweets: 876, favorites: 150 (07/02/2021 11:58:00)

- links: [abs](https://arxiv.org/abs/2106.16198) | [pdf](https://arxiv.org/pdf/2106.16198)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Neural networks are susceptible to small transformations including 2D rotations and shifts, image crops, and even changes in object colors. This is often attributed to biases in the training dataset, and the lack of 2D shift-invariance due to not respecting the sampling theorem. In this paper, we challenge this hypothesis by training and testing on unbiased datasets, and showing that networks are brittle to both small 3D perspective changes and lighting variations which cannot be explained by dataset bias or lack of shift-invariance. To find these in-distribution errors, we introduce an evolution strategies (ES) based approach, which we call CMA-Search. Despite training with a large-scale (0.5 million images), unbiased dataset of camera and light variations, in over 71% cases CMA-Search can find camera parameters in the vicinity of a correctly classified image which lead to in-distribution misclassifications with < 3.6% change in parameters. With lighting changes, CMA-Search finds misclassifications in 33% cases with < 11.6% change in parameters. Finally, we extend this method to find misclassifications in the vicinity of ImageNet images for both ResNet and OpenAI's CLIP model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Small in-distribution changes in 3D perspective and lighting fool both CNNs and Transformers<br>pdf: <a href="https://t.co/1zhF7xYao4">https://t.co/1zhF7xYao4</a><br><br>networks are brittle to both small 3D perspective changes and lighting variations which cannot be explained by dataset bias or lack of shift-invariance <a href="https://t.co/yzh7XfdM8N">pic.twitter.com/yzh7XfdM8N</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410407006188486656?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint! We show that CNNs and Transformers are brittle to small changes in 3D perspective and lighting. We propose an evolution strategies (ES) based search method for finding failures within training distribution! 1/5<br><br>pdf: <a href="https://t.co/4mOeBJZAjK">https://t.co/4mOeBJZAjK</a> <a href="https://t.co/Wj3me52Iw2">pic.twitter.com/Wj3me52Iw2</a></p>&mdash; Spandan Madan (@spandan_madan) <a href="https://twitter.com/spandan_madan/status/1410648136016560129?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. The MultiBERTs: BERT Reproductions for Robustness Analysis

Thibault Sellam, Steve Yadlowsky, Jason Wei, Naomi Saphra, Alexander D'Amour, Tal Linzen, Jasmijn Bastings, Iulia Turc, Jacob Eisenstein, Dipanjan Das, Ian Tenney, Ellie Pavlick

- retweets: 536, favorites: 154 (07/02/2021 11:58:01)

- links: [abs](https://arxiv.org/abs/2106.16163) | [pdf](https://arxiv.org/pdf/2106.16163)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Experiments with pretrained models such as BERT are often based on a single checkpoint. While the conclusions drawn apply to the artifact (i.e., the particular instance of the model), it is not always clear whether they hold for the more general procedure (which includes the model architecture, training data, initialization scheme, and loss function). Recent work has shown that re-running pretraining can lead to substantially different conclusions about performance, suggesting that alternative evaluations are needed to make principled statements about procedures. To address this question, we introduce MultiBERTs: a set of 25 BERT-base checkpoints, trained with similar hyper-parameters as the original BERT model but differing in random initialization and data shuffling. The aim is to enable researchers to draw robust and statistically justified conclusions about pretraining procedures. The full release includes 25 fully trained checkpoints, as well as statistical guidelines and a code library implementing our recommended hypothesis testing methods. Finally, for five of these models we release a set of 28 intermediate checkpoints in order to support research on learning dynamics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">BERT&#39;s performance varies when you re-train it. How much does this affect your results?<br><br>We release MultiBERTs: 25 pre-trained checkpoints and a statistical library to support robust research on BERT.<br><br>Paper: <a href="https://t.co/cdu725VkZ3">https://t.co/cdu725VkZ3</a><br>Models: <a href="https://t.co/Dqz3NGCMGw">https://t.co/Dqz3NGCMGw</a><a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a><br><br>1/X <a href="https://t.co/IjNEDwfPOF">pic.twitter.com/IjNEDwfPOF</a></p>&mdash; Thibault Sellam (@ThiboIbo) <a href="https://twitter.com/ThiboIbo/status/1410688788985176064?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The MultiBERTs: BERT Reproductions for Robustness Analysis<br>pdf: <a href="https://t.co/XNGvWEwulL">https://t.co/XNGvWEwulL</a><br>github: <a href="https://t.co/Tuca9t0Xo2">https://t.co/Tuca9t0Xo2</a><br><br>a set of 25 BERTbase checkpoints, trained with similar hyperparameters as the original BERT model but differing in random initialization and data shuffling <a href="https://t.co/QhjG0jpldT">pic.twitter.com/QhjG0jpldT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410402842729385999?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. DF-Conformer: Integrated architecture of Conv-TasNet and Conformer using  linear complexity self-attention for speech enhancement

Yuma Koizumi, Shigeki Karita, Scott Wisdom, Hakan Erdogan, John R. Hershey, Llion Jones, Michiel Bacchiani

- retweets: 302, favorites: 160 (07/02/2021 11:58:01)

- links: [abs](https://arxiv.org/abs/2106.15813) | [pdf](https://arxiv.org/pdf/2106.15813)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Single-channel speech enhancement (SE) is an important task in speech processing. A widely used framework combines an analysis/synthesis filterbank with a mask prediction network, such as the Conv-TasNet architecture. In such systems, the denoising performance and computational efficiency are mainly affected by the structure of the mask prediction network. In this study, we aim to improve the sequential modeling ability of Conv-TasNet architectures by integrating Conformer layers into a new mask prediction network. To make the model computationally feasible, we extend the Conformer using linear complexity attention and stacked 1-D dilated depthwise convolution layers. We trained the model on 3,396 hours of noisy speech data, and show that (i) the use of linear complexity attention avoids high computational complexity, and (ii) our model achieves higher scale-invariant signal-to-noise ratio than the improved time-dilated convolution network (TDCN++), an extended version of Conv-TasNet.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">arXivに現職で最初の論文を投稿しました！<a href="https://twitter.com/kari_tech?ref_src=twsrc%5Etfw">@kari_tech</a> <a href="https://twitter.com/ScottTWisdom?ref_src=twsrc%5Etfw">@ScottTWisdom</a> <a href="https://twitter.com/YesThisIsLion?ref_src=twsrc%5Etfw">@YesThisIsLion</a>との共著で、O(N)のattentionを持つConformerを使った音声強調です。デモページもあります。めっちゃ音いいです！<br>arXiv: <a href="https://t.co/rdpNGBzWpi">https://t.co/rdpNGBzWpi</a><br>Demo: <a href="https://t.co/P5T2Ih21hb">https://t.co/P5T2Ih21hb</a> <a href="https://t.co/rbf2hqzLuj">pic.twitter.com/rbf2hqzLuj</a></p>&mdash; Yuma Koizumi (@yuma_koizumi) <a href="https://twitter.com/yuma_koizumi/status/1410397020494340100?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We have published a new speech enhancement paper using a linear-complexity Conformer, named DF-Conformer. Using the dilated convolution and FAVOR+ attention. Thanks <a href="https://twitter.com/kari_tech?ref_src=twsrc%5Etfw">@kari_tech</a> <a href="https://twitter.com/ScottTWisdom?ref_src=twsrc%5Etfw">@ScottTWisdom</a> <a href="https://twitter.com/YesThisIsLion?ref_src=twsrc%5Etfw">@YesThisIsLion</a>!<br>arXiv: <a href="https://t.co/rdpNGBzWpi">https://t.co/rdpNGBzWpi</a><br>Demo: <a href="https://t.co/P5T2Ih21hb">https://t.co/P5T2Ih21hb</a> <a href="https://t.co/Iv0ozrvnjF">pic.twitter.com/Iv0ozrvnjF</a></p>&mdash; Yuma Koizumi (@yuma_koizumi) <a href="https://twitter.com/yuma_koizumi/status/1410396645678743552?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DF-Conformer: Integrated architecture of Conv-TasNet and Conformer using linear complexity self-attention for speech enhancement<br>pdf: <a href="https://t.co/uhYXIn6Kv6">https://t.co/uhYXIn6Kv6</a><br>abs: <a href="https://t.co/3yKdgjjdd1">https://t.co/3yKdgjjdd1</a> <a href="https://t.co/ZMk8oAdIU7">pic.twitter.com/ZMk8oAdIU7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410454294394986497?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. A Generative Model for Raw Audio Using Transformer Architectures

Prateek Verma, Chris Chafe

- retweets: 224, favorites: 111 (07/02/2021 11:58:01)

- links: [abs](https://arxiv.org/abs/2106.16036) | [pdf](https://arxiv.org/pdf/2106.16036)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper proposes a novel way of doing audio synthesis at the waveform level using Transformer architectures. We propose a deep neural network for generating waveforms, similar to wavenet \cite{oord2016wavenet}. This is fully probabilistic, auto-regressive, and causal, i.e. each sample generated depends only on the previously observed samples. Our approach outperforms a widely used wavenet architecture by up to 9\% on a similar dataset for predicting the next step. Using the attention mechanism, we enable the architecture to learn which audio samples are important for the prediction of the future sample. We show how causal transformer generative models can be used for raw waveform synthesis. We also show that this performance can be improved by another 2\% by conditioning samples over a wider context. The flexibility of the current model to synthesize audio from latent representations suggests a large number of potential applications. The novel approach of using generative transformer architectures for raw audio synthesis is, however, still far away from generating any meaningful music, without using latent codes/meta-data to aid the generation process.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Generative Model for Raw Audio Using Transformer Architectures<br>pdf: <a href="https://t.co/7dmsiQiaWg">https://t.co/7dmsiQiaWg</a><br>abs: <a href="https://t.co/g1Uj6aYbXU">https://t.co/g1Uj6aYbXU</a><br><br>approach outperforms a widely used wavenet architecture by up to 9% on a similar dataset for predicting the next step <a href="https://t.co/23JkxlfwA3">pic.twitter.com/23JkxlfwA3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410397827096846343?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Augmented Shortcuts for Vision Transformers

Yehui Tang, Kai Han, Chang Xu, An Xiao, Yiping Deng, Chao Xu, Yunhe Wang

- retweets: 272, favorites: 56 (07/02/2021 11:58:02)

- links: [abs](https://arxiv.org/abs/2106.15941) | [pdf](https://arxiv.org/pdf/2106.15941)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformer models have achieved great progress on computer vision tasks recently. The rapid development of vision transformers is mainly contributed by their high representation ability for extracting informative features from input images. However, the mainstream transformer models are designed with deep architectures, and the feature diversity will be continuously reduced as the depth increases, i.e., feature collapse. In this paper, we theoretically analyze the feature collapse phenomenon and study the relationship between shortcuts and feature diversity in these transformer models. Then, we present an augmented shortcut scheme, which inserts additional paths with learnable parameters in parallel on the original shortcuts. To save the computational costs, we further explore an efficient approach that uses the block-circulant projection to implement augmented shortcuts. Extensive experiments conducted on benchmark datasets demonstrate the effectiveness of the proposed method, which brings about 1% accuracy increase of the state-of-the-art visual transformers without obviously increasing their parameters and FLOPs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Augmented Shortcuts for Vision Transformers<br>pdf: <a href="https://t.co/68X2iVPoQd">https://t.co/68X2iVPoQd</a><br>abs: <a href="https://t.co/bo2BI1Suoe">https://t.co/bo2BI1Suoe</a><br><br>brings about 1% accuracy increase of the sota visual transformers without obviously increasing their parameters and FLOPs <a href="https://t.co/UnaaJKvIUv">pic.twitter.com/UnaaJKvIUv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410404384316530691?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Revisiting the Primacy of English in Zero-shot Cross-lingual Transfer

Iulia Turc, Kenton Lee, Jacob Eisenstein, Ming-Wei Chang, Kristina Toutanova

- retweets: 140, favorites: 55 (07/02/2021 11:58:02)

- links: [abs](https://arxiv.org/abs/2106.16171) | [pdf](https://arxiv.org/pdf/2106.16171)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Despite their success, large pre-trained multilingual models have not completely alleviated the need for labeled data, which is cumbersome to collect for all target languages. Zero-shot cross-lingual transfer is emerging as a practical solution: pre-trained models later fine-tuned on one transfer language exhibit surprising performance when tested on many target languages. English is the dominant source language for transfer, as reinforced by popular zero-shot benchmarks. However, this default choice has not been systematically vetted. In our study, we compare English against other transfer languages for fine-tuning, on two pre-trained multilingual models (mBERT and mT5) and multiple classification and question answering tasks. We find that other high-resource languages such as German and Russian often transfer more effectively, especially when the set of target languages is diverse or unknown a priori. Unexpectedly, this can be true even when the training sets were automatically translated from English. This finding can have immediate impact on multilingual zero-shot systems, and should inform future benchmark designs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New from Google Research: <a href="https://t.co/tMcLH1PAb8">https://t.co/tMcLH1PAb8</a><br><br>To build multilingual NLP systems, a successful recipe is to pre-train on a multilingual corpus, and then fine-tune on labeled data in a single transfer language -- usually English. But is English best? <a href="https://t.co/kye8JtwUJV">pic.twitter.com/kye8JtwUJV</a></p>&mdash; Iulia Turc (@IuliaTurc) <a href="https://twitter.com/IuliaTurc/status/1410745252688797699?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. On the Power of Saturated Transformers: A View from Circuit Complexity

William Merrill, Yoav Goldberg, Roy Schwartz, Noah A. Smith

- retweets: 45, favorites: 40 (07/02/2021 11:58:02)

- links: [abs](https://arxiv.org/abs/2106.16213) | [pdf](https://arxiv.org/pdf/2106.16213)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CC](https://arxiv.org/list/cs.CC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers have become a standard architecture for many NLP problems. This has motivated theoretically analyzing their capabilities as models of language, in order to understand what makes them successful, and what their potential weaknesses might be. Recent work has shown that transformers with hard attention are quite limited in capacity, and in fact can be simulated by constant-depth circuits. However, hard attention is a restrictive assumption, which may complicate the relevance of these results for practical transformers. In this work, we analyze the circuit complexity of transformers with saturated attention: a generalization of hard attention that more closely captures the attention patterns learnable in practical transformers. We show that saturated transformers transcend the limitations of hard-attention transformers. With some minor assumptions, we prove that the number of bits needed to represent a saturated transformer memory vector is $O(\log n)$, which implies saturated transformers can be simulated by log-depth circuits. Thus, the jump from hard to saturated attention can be understood as increasing the transformer's effective circuit depth by a factor of $O(\log n)$.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint analyzing the power of uniform attention patterns in transformers in terms of circuit complexity classes<br><br>w/ <a href="https://twitter.com/yoavgo?ref_src=twsrc%5Etfw">@yoavgo</a> <a href="https://twitter.com/royschwartzNLP?ref_src=twsrc%5Etfw">@royschwartzNLP</a> <a href="https://twitter.com/nlpnoah?ref_src=twsrc%5Etfw">@nlpnoah</a> <br><br>Very curious for feedback/suggestions!<a href="https://t.co/1yYaB2pC8f">https://t.co/1yYaB2pC8f</a></p>&mdash; Will Merrill (@lambdaviking) <a href="https://twitter.com/lambdaviking/status/1410643916110499847?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Probabilistic Graphical Models and Tensor Networks: A Hybrid Framework

Jacob Miller, Geoffrey Roeder, Tai-Danae Bradley

- retweets: 31, favorites: 53 (07/02/2021 11:58:02)

- links: [abs](https://arxiv.org/abs/2106.15666) | [pdf](https://arxiv.org/pdf/2106.15666)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [quant-ph](https://arxiv.org/list/quant-ph/recent)

We investigate a correspondence between two formalisms for discrete probabilistic modeling: probabilistic graphical models (PGMs) and tensor networks (TNs), a powerful modeling framework for simulating complex quantum systems. The graphical calculus of PGMs and TNs exhibits many similarities, with discrete undirected graphical models (UGMs) being a special case of TNs. However, more general probabilistic TN models such as Born machines (BMs) employ complex-valued hidden states to produce novel forms of correlation among the probabilities. While representing a new modeling resource for capturing structure in discrete probability distributions, this behavior also renders the direct application of standard PGM tools impossible. We aim to bridge this gap by introducing a hybrid PGM-TN formalism that integrates quantum-like correlations into PGM models in a principled manner, using the physically-motivated concept of decoherence. We first prove that applying decoherence to the entirety of a BM model converts it into a discrete UGM, and conversely, that any subgraph of a discrete UGM can be represented as a decohered BM. This method allows a broad family of probabilistic TN models to be encoded as partially decohered BMs, a fact we leverage to combine the representational strengths of both model families. We experimentally verify the performance of such hybrid models in a sequential modeling task, and identify promising uses of our method within the context of existing applications of graphical models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">i am not yet at a stage where i can parse all of this without some hand-holding, but the parts which i can follow seem very cool!<a href="https://t.co/vHP2uXdlCs">https://t.co/vHP2uXdlCs</a><br>`Probabilistic Graphical Models and Tensor Networks: A Hybrid Framework&#39;<br>- Jacob Miller, Geoffrey Roeder, Tai-Danae Bradley</p>&mdash; Sam Power (@sam_power_825) <a href="https://twitter.com/sam_power_825/status/1410552944907501575?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Reasoning about conscious experience with axiomatic and graphical  mathematics

Camilo Miguel Signorelli, Quanlong Wang, Bob Coecke

- retweets: 26, favorites: 50 (07/02/2021 11:58:03)

- links: [abs](https://arxiv.org/abs/2106.16061) | [pdf](https://arxiv.org/pdf/2106.16061)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We cast aspects of consciousness in axiomatic mathematical terms, using the graphical calculus of general process theories (a.k.a symmetric monoidal categories and Frobenius algebras therein). This calculus exploits the ontological neutrality of process theories. A toy example using the axiomatic calculus is given to show the power of this approach, recovering other aspects of conscious experience, such as external and internal subjective distinction, privacy or unreadability of personal subjective experience, and phenomenal unity, one of the main issues for scientific studies of consciousness. In fact, these features naturally arise from the compositional nature of axiomatic calculus.




# 10. Diffusion Priors In Variational Autoencoders

Antoine Wehenkel, Gilles Louppe

- retweets: 42, favorites: 31 (07/02/2021 11:58:03)

- links: [abs](https://arxiv.org/abs/2106.15671) | [pdf](https://arxiv.org/pdf/2106.15671)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Among likelihood-based approaches for deep generative modelling, variational autoencoders (VAEs) offer scalable amortized posterior inference and fast sampling. However, VAEs are also more and more outperformed by competing models such as normalizing flows (NFs), deep-energy models, or the new denoising diffusion probabilistic models (DDPMs). In this preliminary work, we improve VAEs by demonstrating how DDPMs can be used for modelling the prior distribution of the latent variables. The diffusion prior model improves upon Gaussian priors of classical VAEs and is competitive with NF-based priors. Finally, we hypothesize that hierarchical VAEs could similarly benefit from the enhanced capacity of diffusion priors.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Quite happy to release our latest short work with <a href="https://twitter.com/WehenkelAntoine?ref_src=twsrc%5Etfw">@WehenkelAntoine</a> on Diffusion Priors in Variational Autoencoders, accepted at INNF+ 2021 (at ICML) <a href="https://t.co/zorZgw6EaW">https://t.co/zorZgw6EaW</a></p>&mdash; Gilles Louppe (@glouppe) <a href="https://twitter.com/glouppe/status/1410601060880707590?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Two-Stage TMLE to Reduce Bias and Improve Efficiency in Cluster  Randomized Trials

Laura B. Balzer, Mark van der Laan, James Ayieko, Moses Kamya, Gabriel Chamie, Joshua Schwab, Diane V. Havlir, Maya L. Petersen

- retweets: 30, favorites: 40 (07/02/2021 11:58:03)

- links: [abs](https://arxiv.org/abs/2106.15737) | [pdf](https://arxiv.org/pdf/2106.15737)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.AP](https://arxiv.org/list/stat.AP/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Cluster randomized trials (CRTs) randomly assign an intervention to groups of individuals (e.g., clinics or communities), and measure outcomes on individuals in those groups. While offering many advantages, this experimental design introduces challenges that are only partially addressed by existing analytic approaches. First, outcomes are often missing for some individuals within clusters. Failing to appropriately adjust for differential outcome measurement can result in biased estimates and inference. Second, CRTs often randomize limited numbers of clusters, resulting in chance imbalances on baseline outcome predictors between arms. Failing to adaptively adjust for these imbalances and other predictive covariates can result in efficiency losses. To address these methodological gaps, we propose and evaluate a novel two-stage targeted minimum loss-based estimator (TMLE) to adjust for baseline covariates in a manner that optimizes precision, after controlling for baseline and post-baseline causes of missing outcomes. Finite sample simulations illustrate that our approach can nearly eliminate bias due to differential outcome measurement, while other common CRT estimators yield misleading results and inferences. Application to real data from the SEARCH community randomized trial demonstrates the gains in efficiency afforded through adaptive adjustment for cluster-level covariates, after controlling for missingness on individual-level outcomes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">After 10+yrs in the works, I&#39;m THRILLED: <br>&quot;Two-Stage TMLE to Reduce Bias &amp; Improve Efficiency in Cluster Randomized Trials&quot; is finally here: <a href="https://t.co/wMqaLhUWJ0">https://t.co/wMqaLhUWJ0</a><a href="https://twitter.com/hashtag/epitwitter?src=hash&amp;ref_src=twsrc%5Etfw">#epitwitter</a> <a href="https://twitter.com/hashtag/statstwitter?src=hash&amp;ref_src=twsrc%5Etfw">#statstwitter</a> <a href="https://twitter.com/hashtag/causalinference?src=hash&amp;ref_src=twsrc%5Etfw">#causalinference</a> <a href="https://twitter.com/hashtag/machinelearning?src=hash&amp;ref_src=twsrc%5Etfw">#machinelearning</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/missingdata?src=hash&amp;ref_src=twsrc%5Etfw">#missingdata</a> <a href="https://twitter.com/UCBerkeleySPH?ref_src=twsrc%5Etfw">@UCBerkeleySPH</a> <a href="https://twitter.com/UCSF_HIVIDGM?ref_src=twsrc%5Etfw">@UCSF_HIVIDGM</a> <a href="https://t.co/p9y9sCOmYE">https://t.co/p9y9sCOmYE</a></p>&mdash; Laura B. Balzer (@LauraBBalzer) <a href="https://twitter.com/LauraBBalzer/status/1410635254797422596?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. When the Echo Chamber Shatters: Examining the Use of Community-Specific  Language Post-Subreddit Ban

Milo Z. Trujillo, Samuel F. Rosenblatt, Guillermo de Anda Jáuregui, Emily Moog, Briane Paul V. Samson, Laurent Hébert-Dufresne, Allison M. Roth

- retweets: 39, favorites: 24 (07/02/2021 11:58:03)

- links: [abs](https://arxiv.org/abs/2106.16207) | [pdf](https://arxiv.org/pdf/2106.16207)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Community-level bans are a common tool against groups that enable online harassment and harmful speech. Unfortunately, the efficacy of community bans has only been partially studied and with mixed results. Here, we provide a flexible unsupervised methodology to identify in-group language and track user activity on Reddit both before and after the ban of a community (subreddit). We use a simple word frequency divergence to identify uncommon words overrepresented in a given community, not as a proxy for harmful speech but as a linguistic signature of the community. We apply our method to 15 banned subreddits, and find that community response is heterogeneous between subreddits and between users of a subreddit. Top users were more likely to become less active overall, while random users often reduced use of in-group language without decreasing activity. Finally, we find some evidence that the effectiveness of bans aligns with the content of a community. Users of dark humor communities were largely unaffected by bans while users of communities organized around white supremacy and fascism were the most affected. Altogether, our results show that bans do not affect all groups or users equally, and pave the way to understanding the effect of bans across communities.




# 13. XLM-E: Cross-lingual Language Model Pre-training via ELECTRA

Zewen Chi, Shaohan Huang, Li Dong, Shuming Ma, Saksham Singhal, Payal Bajaj, Xia Song, Furu Wei

- retweets: 36, favorites: 26 (07/02/2021 11:58:03)

- links: [abs](https://arxiv.org/abs/2106.16138) | [pdf](https://arxiv.org/pdf/2106.16138)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

In this paper, we introduce ELECTRA-style tasks to cross-lingual language model pre-training. Specifically, we present two pre-training tasks, namely multilingual replaced token detection, and translation replaced token detection. Besides, we pretrain the model, named as XLM-E, on both multilingual and parallel corpora. Our model outperforms the baseline models on various cross-lingual understanding tasks with much less computation cost. Moreover, analysis shows that XLM-E tends to obtain better cross-lingual transferability.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">XLM-E: Cross-lingual Language Model Pre-training via ELECTRA<br>pdf: <a href="https://t.co/hqayvN0lo1">https://t.co/hqayvN0lo1</a><br>abs: <a href="https://t.co/dSsVXPHZwi">https://t.co/dSsVXPHZwi</a><br><br>outperforms the baseline models on various cross-lingual understanding tasks with much less computation cost <a href="https://t.co/j4OkbpnrvI">pic.twitter.com/j4OkbpnrvI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410398850544803840?ref_src=twsrc%5Etfw">July 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



