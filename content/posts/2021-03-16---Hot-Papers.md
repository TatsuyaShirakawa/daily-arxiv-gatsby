---
title: Hot Papers 2021-03-16
date: 2021-03-17T09:09:48.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-16"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-16"
socialImage: "/media/flying-marine.jpg"

---

# 1. Revisiting ResNets: Improved Training and Scaling Strategies

Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas, Tsung-Yi Lin, Jonathon Shlens, Barret Zoph

- retweets: 4111, favorites: 51 (03/17/2021 09:09:48)

- links: [abs](https://arxiv.org/abs/2103.07579) | [pdf](https://arxiv.org/pdf/2103.07579)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Novel computer vision architectures monopolize the spotlight, but the impact of the model architecture is often conflated with simultaneous changes to training methodology and scaling strategies. Our work revisits the canonical ResNet (He et al., 2015) and studies these three aspects in an effort to disentangle them. Perhaps surprisingly, we find that training and scaling strategies may matter more than architectural changes, and further, that the resulting ResNets match recent state-of-the-art models. We show that the best performing scaling strategy depends on the training regime and offer two new scaling strategies: (1) scale model depth in regimes where overfitting can occur (width scaling is preferable otherwise); (2) increase image resolution more slowly than previously recommended (Tan & Le, 2019). Using improved training and scaling strategies, we design a family of ResNet architectures, ResNet-RS, which are 1.7x - 2.7x faster than EfficientNets on TPUs, while achieving similar accuracies on ImageNet. In a large-scale semi-supervised learning setup, ResNet-RS achieves 86.2% top-1 ImageNet accuracy, while being 4.7x faster than EfficientNet NoisyStudent. The training techniques improve transfer performance on a suite of downstream tasks (rivaling state-of-the-art self-supervised algorithms) and extend to video classification on Kinetics-400. We recommend practitioners use these simple revised ResNets as baselines for future research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">You don&#39;t need EfficientNets. Simple tricks make ResNets better and faster than EfficientNets<br> <br>Revisiting ResNets: Improved Training and Scaling Strategies<br><br>🤙<a href="https://t.co/poXZtzH4Bh">https://t.co/poXZtzH4Bh</a> <a href="https://t.co/YSqzKkCfRd">pic.twitter.com/YSqzKkCfRd</a></p>&mdash; Artsiom Sanakoyeu (@artsiom_s) <a href="https://twitter.com/artsiom_s/status/1371742955984277504?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Approximating How Single Head Attention Learns

Charlie Snell, Ruiqi Zhong, Dan Klein, Jacob Steinhardt

- retweets: 1480, favorites: 173 (03/17/2021 09:09:48)

- links: [abs](https://arxiv.org/abs/2103.07601) | [pdf](https://arxiv.org/pdf/2103.07601)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Why do models often attend to salient words, and how does this evolve throughout training? We approximate model training as a two stage process: early on in training when the attention weights are uniform, the model learns to translate individual input word `i` to `o` if they co-occur frequently. Later, the model learns to attend to `i` while the correct output is $o$ because it knows `i` translates to `o`. To formalize, we define a model property, Knowledge to Translate Individual Words (KTIW) (e.g. knowing that `i` translates to `o`), and claim that it drives the learning of the attention. This claim is supported by the fact that before the attention mechanism is learned, KTIW can be learned from word co-occurrence statistics, but not the other way around. Particularly, we can construct a training distribution that makes KTIW hard to learn, the learning of the attention fails, and the model cannot even learn the simple task of copying the input words to the output. Our approximation explains why models sometimes attend to salient words, and inspires a toy example where a multi-head attention model can overcome the above hard training distribution by improving learning dynamics rather than expressiveness.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Approximating How Single Head Attention Learns<br>pdf: <a href="https://t.co/nxuHyGGiw2">https://t.co/nxuHyGGiw2</a><br>abs: <a href="https://t.co/j0twraCq4P">https://t.co/j0twraCq4P</a> <a href="https://t.co/Vv4Kjs5EJQ">pic.twitter.com/Vv4Kjs5EJQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1371663135900307460?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Modular Interactive Video Object Segmentation: Interaction-to-Mask,  Propagation and Difference-Aware Fusion

Ho Kei Cheng, Yu-Wing Tai, Chi-Keung Tang

- retweets: 1368, favorites: 215 (03/17/2021 09:09:48)

- links: [abs](https://arxiv.org/abs/2103.07941) | [pdf](https://arxiv.org/pdf/2103.07941)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present Modular interactive VOS (MiVOS) framework which decouples interaction-to-mask and mask propagation, allowing for higher generalizability and better performance. Trained separately, the interaction module converts user interactions to an object mask, which is then temporally propagated by our propagation module using a novel top-$k$ filtering strategy in reading the space-time memory. To effectively take the user's intent into account, a novel difference-aware module is proposed to learn how to properly fuse the masks before and after each interaction, which are aligned with the target frames by employing the space-time memory. We evaluate our method both qualitatively and quantitatively with different forms of user interactions (e.g., scribbles, clicks) on DAVIS to show that our method outperforms current state-of-the-art algorithms while requiring fewer frame interactions, with the additional advantage in generalizing to different types of user interactions. We contribute a large-scale synthetic VOS dataset with pixel-accurate segmentation of 4.8M frames to accompany our source codes to facilitate future research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion<br>pdf: <a href="https://t.co/hTVNplXBQY">https://t.co/hTVNplXBQY</a><br>abs: <a href="https://t.co/930WeSQ8Np">https://t.co/930WeSQ8Np</a><br>project page: <a href="https://t.co/iyklauxO6V">https://t.co/iyklauxO6V</a> <a href="https://t.co/DX8QPVxjCr">pic.twitter.com/DX8QPVxjCr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1371641571154792450?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Multi-view Subword Regularization

Xinyi Wang, Sebastian Ruder, Graham Neubig

- retweets: 405, favorites: 163 (03/17/2021 09:09:49)

- links: [abs](https://arxiv.org/abs/2103.08490) | [pdf](https://arxiv.org/pdf/2103.08490)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Multilingual pretrained representations generally rely on subword segmentation algorithms to create a shared multilingual vocabulary. However, standard heuristic algorithms often lead to sub-optimal segmentation, especially for languages with limited amounts of data. In this paper, we take two major steps towards alleviating this problem. First, we demonstrate empirically that applying existing subword regularization methods(Kudo, 2018; Provilkov et al., 2020) during fine-tuning of pre-trained multilingual representations improves the effectiveness of cross-lingual transfer. Second, to take full advantage of different possible input segmentations, we propose Multi-view Subword Regularization (MVR), a method that enforces the consistency between predictions of using inputs tokenized by the standard and probabilistic segmentations. Results on the XTREME multilingual benchmark(Hu et al., 2020) show that MVR brings consistent improvements of up to 2.5 points over using standard segmentation algorithms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Subword segmentation for multilingual pretrained models are suboptimal, especially for under-represented languages. Our NAACL 2021 paper(<a href="https://t.co/L9UhV2j8dT">https://t.co/L9UhV2j8dT</a>) proposes a simple fix at fine-tuning time for better cross-lingual transfer. Joint work with  <a href="https://twitter.com/seb_ruder?ref_src=twsrc%5Etfw">@seb_ruder</a> <a href="https://twitter.com/gneubig?ref_src=twsrc%5Etfw">@gneubig</a> <a href="https://t.co/26lpco4A1V">pic.twitter.com/26lpco4A1V</a></p>&mdash; Xinyi Wang (Cindy) (@cindyxinyiwang) <a href="https://twitter.com/cindyxinyiwang/status/1371828172522717188?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-view subword regularization is simple but yields consistent improvements over pre-trained multilingual models. The best thing: It only needs to be applied during fine-tuning.<br><br>Paper: <a href="https://t.co/gxTgbzVvWN">https://t.co/gxTgbzVvWN</a><br>Code: <a href="https://t.co/FqUyZgEnOQ">https://t.co/FqUyZgEnOQ</a> <a href="https://t.co/sTFxot6yan">https://t.co/sTFxot6yan</a></p>&mdash; Sebastian Ruder (@seb_ruder) <a href="https://twitter.com/seb_ruder/status/1371868675347734528?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. The Public Life of Data: Investigating Reactions to Visualizations on  Reddit

Tobias Kauer, Arran Ridley, Marian Dörk, Benjamin Bach

- retweets: 323, favorites: 53 (03/17/2021 09:09:49)

- links: [abs](https://arxiv.org/abs/2103.08525) | [pdf](https://arxiv.org/pdf/2103.08525)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

This research investigates how people engage with data visualizations when commenting on the social platform Reddit. There has been considerable research on collaborative sensemaking with visualizations and the personal relation of people with data. Yet, little is known about how public audiences without specific expertise and shared incentives openly express their thoughts, feelings, and insights in response to data visualizations. Motivated by the extensive social exchange around visualizations in online communities, this research examines characteristics and motivations of people's reactions to posts featuring visualizations. Following a Grounded Theory approach, we study 475 reactions from the /r/dataisbeautiful community, identify ten distinguishable reaction types, and consider their contribution to the discourse. A follow-up survey with 168 Reddit users clarified their intentions to react. Our results help understand the role of personal perspectives on data and inform future interfaces that integrate audience reactions into visualizations to foster a public discourse about data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We’re excited to present our paper “The public life of data: Investigating Reactions to Visualizations on Reddit” at <a href="https://twitter.com/hashtag/CHI2021?src=hash&amp;ref_src=twsrc%5Etfw">#CHI2021</a> with <a href="https://twitter.com/nrchtct?ref_src=twsrc%5Etfw">@nrchtct</a> <a href="https://twitter.com/arranarranarran?ref_src=twsrc%5Etfw">@arranarranarran</a> <a href="https://twitter.com/benjbach?ref_src=twsrc%5Etfw">@benjbach</a>: <a href="https://t.co/VDd75JErMD">https://t.co/VDd75JErMD</a> <a href="https://t.co/FQPzFqpM7C">pic.twitter.com/FQPzFqpM7C</a></p>&mdash; Tobias Kauer (@tobi_vierzwo) <a href="https://twitter.com/tobi_vierzwo/status/1371747110073274369?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Wav2vec-C: A Self-supervised Model for Speech Representation Learning

Samik Sadhu, Di He, Che-Wei Huang, Sri Harish Mallidi, Minhua Wu, Ariya Rastrow, Andreas Stolcke, Jasha Droppo, Roland Maas

- retweets: 272, favorites: 65 (03/17/2021 09:09:49)

- links: [abs](https://arxiv.org/abs/2103.08393) | [pdf](https://arxiv.org/pdf/2103.08393)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Wav2vec-C introduces a novel representation learning technique combining elements from wav2vec 2.0 and VQ-VAE. Our model learns to reproduce quantized representations from partially masked speech encoding using a contrastive loss in a way similar to Wav2vec 2.0. However, the quantization process is regularized by an additional consistency network that learns to reconstruct the input features to the wav2vec 2.0 network from the quantized representations in a way similar to a VQ-VAE model. The proposed self-supervised model is trained on 10k hours of unlabeled data and subsequently used as the speech encoder in a RNN-T ASR model and fine-tuned with 1k hours of labeled data. This work is one of only a few studies of self-supervised learning on speech tasks with a large volume of real far-field labeled data. The Wav2vec-C encoded representations achieves, on average, twice the error reduction over baseline and a higher codebook utilization in comparison to wav2vec 2.0

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wav2vec-C: A Self-supervised Model for Speech Representation Learning<br>pdf: <a href="https://t.co/Fhn3sD6XCf">https://t.co/Fhn3sD6XCf</a><br>abs: <a href="https://t.co/Nxru1UbZF3">https://t.co/Nxru1UbZF3</a> <a href="https://t.co/wR0W5R842Z">pic.twitter.com/wR0W5R842Z</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1371630212996288513?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Few-Shot Text Classification with Triplet Networks, Data Augmentation,  and Curriculum Learning

Jason Wei, Chengyu Huang, Soroush Vosoughi, Yu Cheng, Shiqi Xu

- retweets: 196, favorites: 73 (03/17/2021 09:09:49)

- links: [abs](https://arxiv.org/abs/2103.07552) | [pdf](https://arxiv.org/pdf/2103.07552)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Few-shot text classification is a fundamental NLP task in which a model aims to classify text into a large number of categories, given only a few training examples per category. This paper explores data augmentation -- a technique particularly suitable for training with limited data -- for this few-shot, highly-multiclass text classification setting. On four diverse text classification tasks, we find that common data augmentation techniques can improve the performance of triplet networks by up to 3.0% on average.   To further boost performance, we present a simple training strategy called curriculum data augmentation, which leverages curriculum learning by first training on only original examples and then introducing augmented data as training progresses. We explore a two-stage and a gradual schedule, and find that, compared with standard single-stage training, curriculum data augmentation trains faster, improves performance, and remains robust to high amounts of noising from augmentation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Few-Shot Text Classification with Triplet Networks, Data Augmentation, and Curriculum Learning<br>pdf: <a href="https://t.co/uyzxL1tu4L">https://t.co/uyzxL1tu4L</a><br>abs: <a href="https://t.co/qX90yC7nAH">https://t.co/qX90yC7nAH</a> <a href="https://t.co/Y15zxoaJo8">pic.twitter.com/Y15zxoaJo8</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1371657275249659906?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Binary R Packages for Linux: Past, Present and Future

Iñaki Ucar, Dirk Eddelbuettel

- retweets: 156, favorites: 26 (03/17/2021 09:09:49)

- links: [abs](https://arxiv.org/abs/2103.08069) | [pdf](https://arxiv.org/pdf/2103.08069)
- [stat.CO](https://arxiv.org/list/stat.CO/recent) | [cs.SE](https://arxiv.org/list/cs.SE/recent)

Pre-compiled binary packages provide a very convenient way of efficiently distributing software that has been adopted by most Linux package management systems. However, the heterogeneity of the Linux ecosystem, combined with the growing number of R extensions available, poses a scalability problem. As a result, efforts to bring binary R packages to Linux have been scattered, and lack a proper mechanism to fully integrate them with R's package manager. This work reviews past and present of binary distribution for Linux, and presents a path forward by showcasing the `cran2copr' project, an RPM-based proof-of-concept implementation of an automated scalable binary distribution system with the capability of building, maintaining and distributing thousands of packages, while providing a portable and extensible bridge to the system package manager.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr"><a href="https://twitter.com/hashtag/arXiv?src=hash&amp;ref_src=twsrc%5Etfw">#arXiv</a> [<a href="https://t.co/7PoS7wuZWd">https://t.co/7PoS7wuZWd</a>] Binary R Packages for Linux: Past, Present and Future, by <a href="https://twitter.com/eddelbuettel?ref_src=twsrc%5Etfw">@eddelbuettel</a> and myself, <a href="https://t.co/Z22HHvPeLM">https://t.co/Z22HHvPeLM</a>. We review efforts over the last 20+ years, and present a PoC of an automated and scalable system with full <a href="https://twitter.com/hashtag/rstats?src=hash&amp;ref_src=twsrc%5Etfw">#rstats</a> integration via BSPM. <a href="https://t.co/rgXWeCD6I2">pic.twitter.com/rgXWeCD6I2</a></p>&mdash; Iñaki Úcar (@Enchufa2) <a href="https://twitter.com/Enchufa2/status/1371842703655383053?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Efficient estimation of Pauli observables by derandomization

Hsin-Yuan Huang, Richard Kueng, John Preskill

- retweets: 63, favorites: 95 (03/17/2021 09:09:49)

- links: [abs](https://arxiv.org/abs/2103.07510) | [pdf](https://arxiv.org/pdf/2103.07510)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent)

We consider the problem of jointly estimating expectation values of many Pauli observables, a crucial subroutine in variational quantum algorithms. Starting with randomized measurements, we propose an efficient derandomization procedure that iteratively replaces random single-qubit measurements with fixed Pauli measurements; the resulting deterministic measurement procedure is guaranteed to perform at least as well as the randomized one. In particular, for estimating any $L$ low-weight Pauli observables, a deterministic measurement on only of order $\log(L)$ copies of a quantum state suffices. In some cases, for example when some of the Pauli observables have a high weight, the derandomized procedure is substantially better than the randomized one. Specifically, numerical experiments highlight the advantages of our derandomized protocol over various previous methods for estimating the ground-state energies of small molecules.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Is randomness necessary to estimate M observables from only log M quantum measurements, e.g., as in <a href="https://t.co/eZIUYy2Rc0">https://t.co/eZIUYy2Rc0</a>? In <a href="https://t.co/fglWrO6EXy">https://t.co/fglWrO6EXy</a>, we show that randomness could be removed to yield even better performance (with application to quantum chemistry).</p>&mdash; Hsin-Yuan (Robert) Huang (@RobertHuangHY) <a href="https://twitter.com/RobertHuangHY/status/1371658159102717955?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Diagrammatic Differentiation for Quantum Machine Learning

Alexis Toumi, Richie Yeung, Giovanni de Felice

- retweets: 82, favorites: 75 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.07960) | [pdf](https://arxiv.org/pdf/2103.07960)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.CT](https://arxiv.org/list/math.CT/recent)

We introduce diagrammatic differentiation for tensor calculus by generalising the dual number construction from rigs to monoidal categories. Applying this to ZX diagrams, we show how to calculate diagrammatically the gradient of a linear map with respect to a phase parameter. For diagrams of parametrised quantum circuits, we get the well-known parameter-shift rule at the basis of many variational quantum algorithms. We then extend our method to the automatic differentation of hybrid classical-quantum circuits, using diagrams with bubbles to encode arbitrary non-linear operators. Moreover, diagrammatic differentiation comes with an open-source implementation in DisCoPy, the Python library for monoidal categories. Diagrammatic gradients of classical-quantum circuits can then be simplified using the PyZX library and executed on quantum hardware via the tket compiler. This opens the door to many practical applications harnessing both the structure of string diagrams and the computational power of quantum machine learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper with <a href="https://twitter.com/richie_yeung?ref_src=twsrc%5Etfw">@richie_yeung</a> and <a href="https://twitter.com/gio_defel?ref_src=twsrc%5Etfw">@gio_defel</a> on diagrammatic differentiation for QML is out on the arXiv! We give rules for computing the gradients of ZX diagrams, quantum circuits and their classical post processing. <a href="https://t.co/jJ2x3M1mGk">https://t.co/jJ2x3M1mGk</a> <a href="https://t.co/cJVLXqTYcv">pic.twitter.com/cJVLXqTYcv</a></p>&mdash; alexis.toumi (@AlexisToumi) <a href="https://twitter.com/AlexisToumi/status/1371764460524859393?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Accelerating the timeline for climate action in California

Daniel M Kammen, Teenie Matlock, Manuel Pastor, David Pellow, Veerabhadran Ramanathan, Tom Steyer, Leah Stokes, Feliz Ventura

- retweets: 87, favorites: 40 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.07801) | [pdf](https://arxiv.org/pdf/2103.07801)
- [eess.SY](https://arxiv.org/list/eess.SY/recent)

The climate emergency increasingly threatens our communities, ecosystems, food production, health, and economy. It disproportionately impacts lower income communities, communities of color, and the elderly. Assessments since the 2018 IPCC 1.5 Celsius report show that current national and sub-national commitments and actions are insufficient. Fortunately, a suite of solutions exists now to mitigate the climate crisis if we initiate and sustain actions today. California, which has a strong set of current targets in place and is home to clean energy and high technology innovation, has fallen behind in its climate ambition compared to a number of major governments. California, a catalyst for climate action globally, can and should ramp up its leadership by aligning its climate goals with the most recent science, coordinating actions to make 2030 a point of significant accomplishment. This entails dramatically accelerating its carbon neutrality and net-negative emissions goal from 2045 to 2030, including advancing clean energy and clean transportation standards, and accelerating nature-based solutions on natural and working lands. It also means changing its current greenhouse gas reduction goals both in the percentage and the timing: cutting emissions by 80 percent (instead of 40 percent) below 1990 levels much closer to 2030 than 2050. These actions will enable California to save lives, benefit underserved and frontline communities, and save trillions of dollars. This rededication takes heed of the latest science, accelerating equitable, job-creating climate policies. While there are significant challenges to achieving these goals, California can establish policy now that will unleash innovation and channel market forces, as has happened with solar, and catalyze positive upward-scaling tipping points for accelerated global climate action.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Time to up our game, California and get back in the lead on energy, climate and justice innovation! <a href="https://twitter.com/AirResources?ref_src=twsrc%5Etfw">@AirResources</a> <a href="https://twitter.com/californiapuc?ref_src=twsrc%5Etfw">@californiapuc</a> <a href="https://twitter.com/GavinNewsom?ref_src=twsrc%5Etfw">@GavinNewsom</a> <a href="https://twitter.com/TeenieMatlock?ref_src=twsrc%5Etfw">@TeenieMatlock</a>, <a href="https://twitter.com/Prof_MPastor?ref_src=twsrc%5Etfw">@Prof_MPastor</a>, <a href="https://twitter.com/david_pellow?ref_src=twsrc%5Etfw">@david_pellow</a>, V.Ramanathan, <a href="https://twitter.com/TomSteyer?ref_src=twsrc%5Etfw">@TomSteyer</a>, <a href="https://twitter.com/leahstokes?ref_src=twsrc%5Etfw">@leahstokes</a> <br><br>Open access version here: <a href="https://t.co/y29OSor2iR">https://t.co/y29OSor2iR</a> <a href="https://t.co/lKSNjdYEsa">https://t.co/lKSNjdYEsa</a></p>&mdash; Daniel M Kammen (@dan_kammen) <a href="https://twitter.com/dan_kammen/status/1371657372129583105?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Toward a Union-Find decoder for quantum LDPC codes

Nicolas Delfosse, Vivien Londe, Michael Beverland

- retweets: 49, favorites: 53 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.08049) | [pdf](https://arxiv.org/pdf/2103.08049)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent)

Quantum LDPC codes are a promising direction for low overhead quantum computing. In this paper, we propose a generalization of the Union-Find decoder as adecoder for quantum LDPC codes. We prove that this decoder corrects all errors with weight up to An^{\alpha} for some A, {\alpha} > 0 for different classes of quantum LDPC codes such as toric codes and hyperbolic codes in any dimension D \geq 3 and quantum expander codes. To prove this result, we introduce a notion of covering radius which measures the spread of an error from its syndrome. We believe this notion could find application beyond the decoding problem. We also perform numerical simulations, which show that our Union-Find decoder outperforms the belief propagation decoder in the low error rate regime in the case of a quantum LDPC code with length 3600.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Quantum LDPC codes are the future and they need better decoders.<br><br>Check out our new paper on a Union-Find decoder for LDPC codes with <a href="https://twitter.com/vivien_londe?ref_src=twsrc%5Etfw">@vivien_londe</a> and Michael Beverland.<a href="https://t.co/0f7l3ahmtM">https://t.co/0f7l3ahmtM</a></p>&mdash; Nicolas Delfosse (@nic_delfosse) <a href="https://twitter.com/nic_delfosse/status/1371637308319965185?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. S2AND: A Benchmark and Evaluation System for Author Name Disambiguation

Shivashankar Subramanian, Daniel King, Doug Downey, Sergey Feldman

- retweets: 90, favorites: 11 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.07534) | [pdf](https://arxiv.org/pdf/2103.07534)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

Author Name Disambiguation (AND) is the task of resolving which author mentions in a bibliographic database refer to the same real-world person, and is a critical ingredient of digital library applications such as search and citation analysis. While many AND algorithms have been proposed, comparing them is difficult because they often employ distinct features and are evaluated on different datasets. In response to this challenge, we present S2AND, a unified benchmark dataset for AND on scholarly papers, as well as an open-source reference model implementation. Our dataset harmonizes eight disparate AND datasets into a uniform format, with a single rich feature set drawn from the Semantic Scholar S2 database. Our evaluation suite for S2AND reports performance split by facets like publication year and number of papers, allowing researchers to track both global performance and measures of fairness across facet values. Our experiments show that because previous datasets tend to cover idiosyncratic and biased slices of the literature, algorithms trained to perform well on one on them may generalize poorly to others. By contrast, we show how training on a union of datasets in S2AND results in more robust models that perform well even on datasets unseen in training. The resulting AND model also substantially improves over the production algorithm in S2, reducing error by over 50% in terms of B^3 F1. We release our unified dataset, model code, trained models, and evaluation suite to the research community. https://github.com/allenai/S2AND/




# 14. Automated Fact-Checking for Assisting Human Fact-Checkers

Preslav Nakov, David Corney, Maram Hasanain, Firoj Alam, Tamer Elsayed, Alberto Barrón-Cedeño, Paolo Papotti, Shaden Shaar, Giovanni Da San Martino

- retweets: 26, favorites: 52 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.07769) | [pdf](https://arxiv.org/pdf/2103.07769)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The reporting and analysis of current events around the globe has expanded from professional, editor-lead journalism all the way to citizen journalism. Politicians and other key players enjoy direct access to their audiences through social media, bypassing the filters of official cables or traditional media. However, the multiple advantages of free speech and direct communication are dimmed by the misuse of the media to spread inaccurate or misleading claims. These phenomena have led to the modern incarnation of the fact-checker -- a professional whose main aim is to examine claims using available evidence to assess their veracity. As in other text forensics tasks, the amount of information available makes the work of the fact-checker more difficult. With this in mind, starting from the perspective of the professional fact-checker, we survey the available intelligent technologies that can support the human expert in the different steps of her fact-checking endeavor. These include identifying claims worth fact-checking; detecting relevant previously fact-checked claims; retrieving relevant evidence to fact-check a claim; and actually verifying a claim. In each case, we pay attention to the challenges in future work and the potential impact on real-world fact-checking.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Automated Fact-Checking for Assisting Human Fact-Checkers&quot; <a href="https://t.co/XwqYDeOs1u">https://t.co/XwqYDeOs1u</a> -- We survey AI to can support the human expert in  identifying claims worth fact-checking, detecting  previously fact-checked claims, retrieving evidence, and  verifying a claim <a href="https://twitter.com/hashtag/fakenews?src=hash&amp;ref_src=twsrc%5Etfw">#fakenews</a> <a href="https://t.co/XG9FSUEPhR">pic.twitter.com/XG9FSUEPhR</a></p>&mdash; Preslav Nakov (@preslav_nakov) <a href="https://twitter.com/preslav_nakov/status/1371738922838032386?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Rotation Coordinate Descent for Fast Globally Optimal Rotation Averaging

Álvaro Parra, Shin-Fang Chng, Tat-Jun Chin, Anders Eriksson, Ian Reid

- retweets: 20, favorites: 58 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.08292) | [pdf](https://arxiv.org/pdf/2103.08292)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Under mild conditions on the noise level of the measurements, rotation averaging satisfies strong duality, which enables global solutions to be obtained via semidefinite programming (SDP) relaxation. However, generic solvers for SDP are rather slow in practice, even on rotation averaging instances of moderate size, thus developing specialised algorithms is vital. In this paper, we present a fast algorithm that achieves global optimality called rotation coordinate descent (RCD). Unlike block coordinate descent (BCD) which solves SDP by updating the semidefinite matrix in a row-by-row fashion, RCD directly maintains and updates all valid rotations throughout the iterations. This obviates the need to store a large dense semidefinite matrix. We mathematically prove the convergence of our algorithm and empirically show its superior efficiency over state-of-the-art global methods on a variety of problem configurations. Maintaining valid rotations also facilitates incorporating local optimisation routines for further speed-ups. Moreover, our algorithm is simple to implement; see supplementary material for a demonstration program.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper on rotation averaging accepted as oral at <a href="https://twitter.com/hashtag/CVPR21?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR21</a>!<br><br>RCD is able to efficiently find the optimal camera orientations on large problems. <a href="https://t.co/aSVjEIPyaj">https://t.co/aSVjEIPyaj</a> <a href="https://t.co/KWDgvzS1B3">pic.twitter.com/KWDgvzS1B3</a></p>&mdash; Álvaro Parra (@DrAlvaroParra) <a href="https://twitter.com/DrAlvaroParra/status/1371636340903735296?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. PhotoApp: Photorealistic Appearance Editing of Head Portraits

Mallikarjun B R, Ayush Tewari, Abdallah Dib, Tim Weyrich, Bernd Bickel, Hans-Peter Seidel, Hanspeter Pfister, Wojciech Matusik, Louis Chevallier, Mohamed Elgharib, Christian Theobalt

- retweets: 36, favorites: 36 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.07658) | [pdf](https://arxiv.org/pdf/2103.07658)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Photorealistic editing of portraits is a challenging task as humans are very sensitive to inconsistencies in faces. We present an approach for high-quality intuitive editing of the camera viewpoint and scene illumination in a portrait image. This requires our method to capture and control the full reflectance field of the person in the image. Most editing approaches rely on supervised learning using training data captured with setups such as light and camera stages. Such datasets are expensive to acquire, not readily available and do not capture all the rich variations of in-the-wild portrait images. In addition, most supervised approaches only focus on relighting, and do not allow camera viewpoint editing. Thus, they only capture and control a subset of the reflectance field. Recently, portrait editing has been demonstrated by operating in the generative model space of StyleGAN. While such approaches do not require direct supervision, there is a significant loss of quality when compared to the supervised approaches. In this paper, we present a method which learns from limited supervised training data. The training images only include people in a fixed neutral expression with eyes closed, without much hair or background variations. Each person is captured under 150 one-light-at-a-time conditions and under 8 camera poses. Instead of training directly in the image space, we design a supervised problem which learns transformations in the latent space of StyleGAN. This combines the best of supervised learning and generative adversarial modeling. We show that the StyleGAN prior allows for generalisation to different expressions, hairstyles and backgrounds. This produces high-quality photorealistic results for in-the-wild images and significantly outperforms existing methods. Our approach can edit the illumination and pose simultaneously, and runs at interactive rates.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PhotoApp: Photorealistic Appearance Editing of Head Portraits<br>pdf: <a href="https://t.co/2PudYMHkk0">https://t.co/2PudYMHkk0</a><br>abs: <a href="https://t.co/rkFtRgqmBW">https://t.co/rkFtRgqmBW</a><br>project page: <a href="https://t.co/wLRQp70brf">https://t.co/wLRQp70brf</a> <a href="https://t.co/86Q98riFdF">pic.twitter.com/86Q98riFdF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1371632970453426178?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Unsupervised Image Transformation Learning via Generative Adversarial  Networks

Kaiwen Zha, Yujun Shen, Bolei Zhou

- retweets: 16, favorites: 49 (03/17/2021 09:09:50)

- links: [abs](https://arxiv.org/abs/2103.07751) | [pdf](https://arxiv.org/pdf/2103.07751)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this work, we study the image transformation problem by learning the underlying transformations from a collection of images using Generative Adversarial Networks (GANs). Specifically, we propose an unsupervised learning framework, termed as TrGAN, to project images onto a transformation space that is shared by the generator and the discriminator. Any two points in this projected space define a transformation that can guide the image generation process, leading to continuous semantic change. By projecting a pair of images onto the transformation space, we are able to adequately extract the semantic variation between them and further apply the extracted semantic to facilitating image editing, including not only transferring image styles (e.g., changing day to night) but also manipulating image contents (e.g., adding clouds in the sky). Code and models are available at https://genforce.github.io/trgan.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Image Transformation Learning via Generative Adversarial Networks<br>pdf: <a href="https://t.co/pFe5XPXrYb">https://t.co/pFe5XPXrYb</a><br>abs: <a href="https://t.co/Mwcg3bKv16">https://t.co/Mwcg3bKv16</a> <a href="https://t.co/wMhWf0C0nd">pic.twitter.com/wMhWf0C0nd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1371636322247540736?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Learning One Representation to Optimize All Rewards

Ahmed Touati, Yann Ollivier

- retweets: 16, favorites: 46 (03/17/2021 09:09:51)

- links: [abs](https://arxiv.org/abs/2103.07945) | [pdf](https://arxiv.org/pdf/2103.07945)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [math.OC](https://arxiv.org/list/math.OC/recent)

We introduce the forward-backward (FB) representation of the dynamics of a reward-free Markov decision process. It provides explicit near-optimal policies for any reward specified a posteriori. During an unsupervised phase, we use reward-free interactions with the environment to learn two representations via off-the-shelf deep learning methods and temporal difference (TD) learning. In the test phase, a reward representation is estimated either from observations or an explicit reward description (e.g., a target state). The optimal policy for that reward is directly obtained from these representations, with no planning.   The unsupervised FB loss is well-principled: if training is perfect, the policies obtained are provably optimal for any reward function. With imperfect training, the sub-optimality is proportional to the unsupervised approximation error. The FB representation learns long-range relationships between states and actions, via a predictive occupancy map, without having to synthesize states as in model-based approaches.   This is a step towards learning controllable agents in arbitrary black-box stochastic environments. This approach compares well to goal-oriented RL algorithms on discrete and continuous mazes, pixel-based MsPacman, and the FetchReach virtual robot arm. We also illustrate how the agent can immediately adapt to new tasks beyond goal-oriented RL.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning One Representation to Optimize All<br>Rewards<br>pdf: <a href="https://t.co/o2xjtwyWej">https://t.co/o2xjtwyWej</a><br>abs: <a href="https://t.co/yNXG5LvBQP">https://t.co/yNXG5LvBQP</a><br>github: <a href="https://t.co/U3gJyL68uC">https://t.co/U3gJyL68uC</a> <a href="https://t.co/AGGCgB5JR3">pic.twitter.com/AGGCgB5JR3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1371686285035909120?ref_src=twsrc%5Etfw">March 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



