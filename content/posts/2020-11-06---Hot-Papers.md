---
title: Hot Papers 2020-11-06
date: 2020-11-09T15:08:09.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-06"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-06"
socialImage: "/media/flying-marine.jpg"

---

# 1. Physics-Informed Neural Network Super Resolution for Advection-Diffusion  Models

Chulin Wang, Eloisa Bentivegna, Wang Zhou, Levente Klein, Bruce Elmegreen

- retweets: 1962, favorites: 52 (11/09/2020 15:08:09)

- links: [abs](https://arxiv.org/abs/2011.02519) | [pdf](https://arxiv.org/pdf/2011.02519)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [physics.geo-ph](https://arxiv.org/list/physics.geo-ph/recent)

Physics-informed neural networks (NN) are an emerging technique to improve spatial resolution and enforce physical consistency of data from physics models or satellite observations. A super-resolution (SR) technique is explored to reconstruct high-resolution images ($4\times$) from lower resolution images in an advection-diffusion model of atmospheric pollution plumes. SR performance is generally increased when the advection-diffusion equation constrains the NN in addition to conventional pixel-based constraints. The ability of SR techniques to also reconstruct missing data is investigated by randomly removing image pixels from the simulations and allowing the system to learn the content of missing data. Improvements in S/N of $11\%$ are demonstrated when physics equations are included in SR with $40\%$ pixel loss. Physics-informed NNs accurately reconstruct corrupted images and generate better results compared to the standard SR approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Physics-Informed Neural Network Super Resolution for Advection-Diffusion Models. <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/Cloud?src=hash&amp;ref_src=twsrc%5Etfw">#Cloud</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/Developers?src=hash&amp;ref_src=twsrc%5Etfw">#Developers</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a><a href="https://t.co/QjG5MuX5zI">https://t.co/QjG5MuX5zI</a> <a href="https://t.co/iZLZUEYrbx">pic.twitter.com/iZLZUEYrbx</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1324894650340167680?ref_src=twsrc%5Etfw">November 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Intriguing Properties of Contrastive Losses

Ting Chen, Lala Li

- retweets: 1215, favorites: 284 (11/09/2020 15:08:09)

- links: [abs](https://arxiv.org/abs/2011.02803) | [pdf](https://arxiv.org/pdf/2011.02803)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Contrastive loss and its variants have become very popular recently for learning visual representations without supervision. In this work, we first generalize the standard contrastive loss based on cross entropy to a broader family of losses that share an abstract form of $\mathcal{L}_{\text{alignment}} + \lambda \mathcal{L}_{\text{distribution}}$, where hidden representations are encouraged to (1) be aligned under some transformations/augmentations, and (2) match a prior distribution of high entropy. We show that various instantiations of the generalized loss perform similarly under the presence of a multi-layer non-linear projection head, and the temperature scaling ($\tau$) widely used in the standard contrastive loss is (within a range) inversely related to the weighting ($\lambda$) between the two loss terms. We then study an intriguing phenomenon of feature suppression among competing features shared acros augmented views, such as "color distribution" vs "object class". We construct datasets with explicit and controllable competing features, and show that, for contrastive learning, a few bits of easy-to-learn shared features could suppress, and even fully prevent, the learning of other sets of competing features. Interestingly, this characteristic is much less detrimental in autoencoders based on a reconstruction loss. Existing contrastive learning methods critically rely on data augmentation to favor certain sets of features than others, while one may wish that a network would learn all competing features as much as its capacity allows.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contrastive loss has been very successful in self-supervised learning recently, but how/why does it actually work, and is there any limitation?<br><br>Our recent tech report hopes to spark more discussions - &quot;Intriguing Properties of Contrastive Losses&quot;<a href="https://t.co/7R1KYWtyfA">https://t.co/7R1KYWtyfA</a></p>&mdash; Ting Chen (@tingchenai) <a href="https://twitter.com/tingchenai/status/1324779707775225857?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene  Understanding

Mike Roberts, Nathan Paczan

- retweets: 959, favorites: 218 (11/09/2020 15:08:10)

- links: [abs](https://arxiv.org/abs/2011.02523) | [pdf](https://arxiv.org/pdf/2011.02523)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

For many fundamental scene understanding tasks, it is difficult or impossible to obtain per-pixel ground truth labels from real images. We address this challenge by introducing Hypersim, a photorealistic synthetic dataset for holistic indoor scene understanding. To create our dataset, we leverage a large repository of synthetic scenes created by professional artists, and we generate 77,400 images of 461 indoor scenes with detailed per-pixel labels and corresponding ground truth geometry. Our dataset: (1) relies exclusively on publicly available 3D assets; (2) includes complete scene geometry, material information, and lighting information for every scene; (3) includes dense per-pixel semantic instance segmentations for every image; and (4) factors every image into diffuse reflectance, diffuse illumination, and a non-diffuse residual term that captures view-dependent lighting effects. Together, these features make our dataset well-suited for geometric learning problems that require direct 3D supervision, multi-task learning problems that require reasoning jointly over multiple input and output modalities, and inverse rendering problems. We analyze our dataset at the level of scenes, objects, and pixels, and we analyze costs in terms of money, annotation effort, and computation time. Remarkably, we find that it is possible to generate our entire dataset from scratch, for roughly half the cost of training a state-of-the-art natural language processing model. All the code we used to generate our dataset will be made available online.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Yay! I can finally share my latest work 😀🤓 Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding <a href="https://t.co/r4ootWiAVs">https://t.co/r4ootWiAVs</a> <a href="https://twitter.com/hashtag/3d?src=hash&amp;ref_src=twsrc%5Etfw">#3d</a> <a href="https://twitter.com/hashtag/vision?src=hash&amp;ref_src=twsrc%5Etfw">#vision</a> <a href="https://twitter.com/hashtag/graphics?src=hash&amp;ref_src=twsrc%5Etfw">#graphics</a> <a href="https://twitter.com/hashtag/deeplearning?src=hash&amp;ref_src=twsrc%5Etfw">#deeplearning</a> <a href="https://twitter.com/hashtag/ai?src=hash&amp;ref_src=twsrc%5Etfw">#ai</a> <a href="https://t.co/WjYxpb0gTC">pic.twitter.com/WjYxpb0gTC</a></p>&mdash; Mike Roberts (@mikeroberts3000) <a href="https://twitter.com/mikeroberts3000/status/1324557957292457984?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding<a href="https://t.co/4aGi6LDRzA">https://t.co/4aGi6LDRzA</a> <a href="https://t.co/lW5OE0KXxo">pic.twitter.com/lW5OE0KXxo</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1324526188656619521?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding. <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/Developers?src=hash&amp;ref_src=twsrc%5Etfw">#Developers</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a><a href="https://t.co/MXuCbQbtsl">https://t.co/MXuCbQbtsl</a> <a href="https://t.co/Ary4iLgwO3">pic.twitter.com/Ary4iLgwO3</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1324908455623266304?ref_src=twsrc%5Etfw">November 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding. <a href="https://t.co/BuVEesSYuf">https://t.co/BuVEesSYuf</a> <a href="https://t.co/yrm2I9RGMU">pic.twitter.com/yrm2I9RGMU</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1324606593913262082?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Multi-task learning for electronic structure to predict and explore  molecular potential energy surfaces

Zhuoran Qiao, Feizhi Ding, Matthew Welborn, Peter J. Bygrave, Animashree Anandkumar, Frederick R. Manby, Thomas F. Miller III

- retweets: 237, favorites: 116 (11/09/2020 15:08:10)

- links: [abs](https://arxiv.org/abs/2011.02680) | [pdf](https://arxiv.org/pdf/2011.02680)
- [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We refine the OrbNet model to accurately predict energy, forces, and other response properties for molecules using a graph neural-network architecture based on features from low-cost approximated quantum operators in the symmetry-adapted atomic orbital basis. The model is end-to-end differentiable due to the derivation of analytic gradients for all electronic structure terms, and is shown to be transferable across chemical space due to the use of domain-specific features. The learning efficiency is improved by incorporating physically motivated constraints on the electronic structure through multi-task learning. The model outperforms existing methods on energy prediction tasks for the QM9 dataset and for molecular geometry optimizations on conformer datasets, at a computational cost that is thousand-fold or more reduced compared to conventional quantum-chemistry calculations (such as density functional theory) that offer similar accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">OrbNet 🕷️ gradients are available, yielding excellent accuracy for geometry optimizations with 1000-fold reduction in DFT cost.<a href="https://t.co/WWaoU7ugtz">https://t.co/WWaoU7ugtz</a><br><br>Accepted to the <a href="https://twitter.com/hashtag/ML4Molecules?src=hash&amp;ref_src=twsrc%5Etfw">#ML4Molecules</a> workshop at <a href="https://twitter.com/NeurIPSConf?ref_src=twsrc%5Etfw">@NeurIPSConf</a> <br><br>Great collaboration with <a href="https://twitter.com/ZhuoranQ?ref_src=twsrc%5Etfw">@ZhuoranQ</a> <a href="https://twitter.com/AnimaAnandkumar?ref_src=twsrc%5Etfw">@AnimaAnandkumar</a> and <a href="https://twitter.com/EntosAI?ref_src=twsrc%5Etfw">@EntosAI</a></p>&mdash; Thomas Miller (@tfmiller3) <a href="https://twitter.com/tfmiller3/status/1324631647531069440?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to have our work accepted to the <a href="https://twitter.com/hashtag/ML4Molecules?src=hash&amp;ref_src=twsrc%5Etfw">#ML4Molecules</a> workshop at <a href="https://twitter.com/NeurIPSConf?ref_src=twsrc%5Etfw">@NeurIPSConf</a> as contributed talk (2-3)! Come to my presentation on Dec. 12 to hear about multi-task &amp; differentiable learning for molecular electronic structure. <a href="https://t.co/cTPMzY9f0i">https://t.co/cTPMzY9f0i</a> <a href="https://t.co/Q360SnuMQa">https://t.co/Q360SnuMQa</a></p>&mdash; Zhuoran Qiao (@ZhuoranQ) <a href="https://twitter.com/ZhuoranQ/status/1324748683129188352?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Semi-supervised Learning for Singing Synthesis Timbre

Jordi Bonada, Merlijn Blaauw

- retweets: 225, favorites: 62 (11/09/2020 15:08:10)

- links: [abs](https://arxiv.org/abs/2011.02809) | [pdf](https://arxiv.org/pdf/2011.02809)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We propose a semi-supervised singing synthesizer, which is able to learn new voices from audio data only, without any annotations such as phonetic segmentation. Our system is an encoder-decoder model with two encoders, linguistic and acoustic, and one (acoustic) decoder. In a first step, the system is trained in a supervised manner, using a labelled multi-singer dataset. Here, we ensure that the embeddings produced by both encoders are similar, so that we can later use the model with either acoustic or linguistic input features. To learn a new voice in an unsupervised manner, the pretrained acoustic encoder is used to train a decoder for the target singer. Finally, at inference, the pretrained linguistic encoder is used together with the decoder of the new voice, to produce acoustic features from linguistic input. We evaluate our system with a listening test and show that the results are comparable to those obtained with an equivalent supervised approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Semi-supervised Learning for Singing Synthesis Timbre<br>pdf: <a href="https://t.co/6r11V7bST8">https://t.co/6r11V7bST8</a><br>abs: <a href="https://t.co/gULtSD2rSP">https://t.co/gULtSD2rSP</a><br>project page: <a href="https://t.co/aZh3DfoUcV">https://t.co/aZh3DfoUcV</a> <a href="https://t.co/nfCJDNsPcN">pic.twitter.com/nfCJDNsPcN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1324540469221666816?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Disentangling Latent Space for Unsupervised Semantic Face Editing

Kanglin Liu, Gaofeng Cao, Fei Zhou, Bozhi Liu, Jiang Duan, Guoping Qiu

- retweets: 169, favorites: 115 (11/09/2020 15:08:11)

- links: [abs](https://arxiv.org/abs/2011.02638) | [pdf](https://arxiv.org/pdf/2011.02638)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Editing facial images created by StyleGAN is a popular research topic with important applications. Through editing the latent vectors, it is possible to control the facial attributes such as smile, age, \textit{etc}. However, facial attributes are entangled in the latent space and this makes it very difficult to independently control a specific attribute without affecting the others. The key to developing neat semantic control is to completely disentangle the latent space and perform image editing in an unsupervised manner. In this paper, we present a new technique termed Structure-Texture Independent Architecture with Weight Decomposition and Orthogonal Regularization (STIA-WO) to disentangle the latent space. The GAN model, applying STIA-WO, is referred to as STGAN-WO. STGAN-WO performs weight decomposition by utilizing the style vector to construct a fully controllable weight matrix for controlling the image synthesis, and utilizes orthogonal regularization to ensure each entry of the style vector only controls one factor of variation. To further disentangle the facial attributes, STGAN-WO introduces a structure-texture independent architecture which utilizes two independently and identically distributed (i.i.d.) latent vectors to control the synthesis of the texture and structure components in a disentangled way.Unsupervised semantic editing is achieved by moving the latent code in the coarse layers along its orthogonal directions to change texture related attributes or changing the latent code in the fine layers to manipulate structure related ones. We present experimental results which show that our new STGAN-WO can achieve better attribute editing than state of the art methods (The code is available at https://github.com/max-liu-112/STGAN-WO)

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Disentangling Latent Space for Unsupervised Semantic Face Editing<br>pdf: <a href="https://t.co/y03DK7L78K">https://t.co/y03DK7L78K</a><br>abs: <a href="https://t.co/qAFpcQmziI">https://t.co/qAFpcQmziI</a> <a href="https://t.co/cnEoSSvnka">pic.twitter.com/cnEoSSvnka</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1324532387066368002?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Absence of Barren Plateaus in Quantum Convolutional Neural Networks

Arthur Pesah, M. Cerezo, Samson Wang, Tyler Volkoff, Andrew T. Sornborger, Patrick J. Coles

- retweets: 138, favorites: 103 (11/09/2020 15:08:11)

- links: [abs](https://arxiv.org/abs/2011.02966) | [pdf](https://arxiv.org/pdf/2011.02966)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Quantum neural networks (QNNs) have generated excitement around the possibility of efficiently analyzing quantum data. But this excitement has been tempered by the existence of exponentially vanishing gradients, known as barren plateau landscapes, for many QNN architectures. Recently, Quantum Convolutional Neural Networks (QCNNs) have been proposed, involving a sequence of convolutional and pooling layers that reduce the number of qubits while preserving information about relevant data features. In this work we rigorously analyze the gradient scaling for the parameters in the QCNN architecture. We find that the variance of the gradient vanishes no faster than polynomially, implying that QCNNs do not exhibit barren plateaus. This provides an analytical guarantee for the trainability of randomly initialized QCNNs, which singles out QCNNs as being trainable unlike many other QNN architectures. To derive our results we introduce a novel graph-based method to analyze expectation values over Haar-distributed unitaries, which will likely be useful in other contexts. Finally, we perform numerical simulations to verify our analytical results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My project from the <a href="https://twitter.com/LosAlamosNatLab?ref_src=twsrc%5Etfw">@LosAlamosNatLab</a> Summer School is finally out! We are happy to announce the...<br><br>Absence of Barren Plateau in Quantum Convolutional Neural Networks 🔥<br><br>with <a href="https://twitter.com/MvsCerezo?ref_src=twsrc%5Etfw">@MvsCerezo</a>, <a href="https://twitter.com/samson_wang?ref_src=twsrc%5Etfw">@samson_wang</a>, T. Volkoff, <a href="https://twitter.com/sornborg?ref_src=twsrc%5Etfw">@sornborg</a> and <a href="https://twitter.com/ColesQuantum?ref_src=twsrc%5Etfw">@ColesQuantum</a>.<a href="https://t.co/JupLDz994a">https://t.co/JupLDz994a</a><br><br>Thread 👇 <a href="https://t.co/YeWp2cBz6Z">pic.twitter.com/YeWp2cBz6Z</a></p>&mdash; Arthur Pesah (@artix41) <a href="https://twitter.com/artix41/status/1324727426430210048?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Finally some good news about quantum neural networks. No barren plateaus in quantum convolutional neural networks. Congrats to our summer students <a href="https://twitter.com/artix41?ref_src=twsrc%5Etfw">@artix41</a> <a href="https://twitter.com/samson_wang?ref_src=twsrc%5Etfw">@samson_wang</a> on proving this exciting result<a href="https://t.co/NWKOUJmrkY">https://t.co/NWKOUJmrkY</a><br><br>See awesome thread below with beautiful illustrations <a href="https://t.co/loUnTJPY0M">https://t.co/loUnTJPY0M</a></p>&mdash; Patrick Coles (@ColesQuantum) <a href="https://twitter.com/ColesQuantum/status/1324736156731379712?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Measuring Data Collection Quality for Community Healthcare

Ramesha Karunasena, Mohammad Sarparajul Ambiya, Arunesh Sinha, Ruchit Nagar, Saachi Dalal, Divy Thakkar, Milind Tambe

- retweets: 158, favorites: 9 (11/09/2020 15:08:11)

- links: [abs](https://arxiv.org/abs/2011.02962) | [pdf](https://arxiv.org/pdf/2011.02962)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Machine learning has tremendous potential to provide targeted interventions in low-resource communities, however the availability of high-quality public health data is a significant challenge. In this work, we partner with field experts at an non-governmental organization (NGO) in India to define and test a data collection quality score for each health worker who collects data. This challenging unlabeled data problem is handled by building upon domain-expert's guidance to design a useful data representation that is then clustered to infer a data quality score. We also provide a more interpretable version of the score. These scores already provide for a measurement of data collection quality; in addition, we also predict the quality for future time steps and find our results to be very accurate. Our work was successfully field tested and is in the final stages of deployment in Rajasthan, India.




# 9. End-to-end-Architekturen zur Datenmonetarisierung im IIoT. Konzepte und  Implementierungen

Christoph F. Strnadl

- retweets: 90, favorites: 42 (11/09/2020 15:08:11)

- links: [abs](https://arxiv.org/abs/2011.02801) | [pdf](https://arxiv.org/pdf/2011.02801)
- [cs.NI](https://arxiv.org/list/cs.NI/recent)

The value creation potential of the Internet of Things (IoT), that is the connection of arbitrary objects to the Internet, lies in the creation of business benefits through accessing and processing the circa 80 Zettabytes (1 ZB = 10^21 Bytes) of data produced by an estimated 40 billions of IoT endpoints (prognosis for 2025). This contribution derives and presents the information technology-related fundament and basis required to be able to reap this potential. Quantity and heterogeneity of the devices and machines especially encountered in the industry at large in the so-called industrial IoT (IIoT) require the use of a typically cloud-based IoT platform for logical concentration and more efficient management of the -- unavoidable in industry -- complexity. Stringent non-functional requirements especially regarding (low) latency, (high) bandwidth, access to large processing capacities, and security and privacy-related aspects necessitate the deployment of intermediary IoT gateways endowed with different capability sets in the edge continuum between IoT endpoints and the IoT platform in the cloud. This will be illustrated in the form of two use cases from corporate projects using a component architecture view point. Finally we argue that this classical concept of IoT projects needs to be strategically widened towards application integration (key word: IT/OT integration) and API management resulting in the coupling of a suitable integration and API management platforms to the IoT platform in order to use this end-to-end understanding of IoT/IIoT to fully leverage the innovation-stimulating and transformational character of IIoT and Industry 4.0.

<blockquote class="twitter-tweet"><p lang="de" dir="ltr">Für ein Buchprojekt von <a href="https://twitter.com/rwth_wzl?ref_src=twsrc%5Etfw">@rwth_wzl</a> &amp; <a href="https://twitter.com/Fraunhofer_FIT?ref_src=twsrc%5Etfw">@Fraunhofer_FIT</a> (mehr darf ich noch nicht verraten) habe ich einen Beitrag (Deutsch) zum Thema <a href="https://twitter.com/hashtag/End2end?src=hash&amp;ref_src=twsrc%5Etfw">#End2end</a> <a href="https://twitter.com/hashtag/Architekturen?src=hash&amp;ref_src=twsrc%5Etfw">#Architekturen</a> im <a href="https://twitter.com/hashtag/IIoT?src=hash&amp;ref_src=twsrc%5Etfw">#IIoT</a> verfasst, den es als Preprint geben &quot;darf&quot;: <a href="https://t.co/uTIuC7YPPB">https://t.co/uTIuC7YPPB</a><a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Edge?src=hash&amp;ref_src=twsrc%5Etfw">#Edge</a> <a href="https://twitter.com/hashtag/EdgeContinuum?src=hash&amp;ref_src=twsrc%5Etfw">#EdgeContinuum</a> <a href="https://twitter.com/hashtag/platform?src=hash&amp;ref_src=twsrc%5Etfw">#platform</a> <a href="https://twitter.com/hashtag/LPWAN?src=hash&amp;ref_src=twsrc%5Etfw">#LPWAN</a></p>&mdash; Christoph F. Strnadl (@archimate) <a href="https://twitter.com/archimate/status/1324670960381894657?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Revisiting Stereo Depth Estimation From a Sequence-to-Sequence  Perspective with Transformers

Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, Mathias Unberath

- retweets: 41, favorites: 66 (11/09/2020 15:08:11)

- links: [abs](https://arxiv.org/abs/2011.02910) | [pdf](https://arxiv.org/pdf/2011.02910)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Stereo depth estimation relies on optimal correspondence matching between pixels on epipolar lines in the left and right image to infer depth. Rather than matching individual pixels, in this work, we revisit the problem from a sequence-to-sequence correspondence perspective to replace cost volume construction with dense pixel matching using position information and attention. This approach, named STereo TRansformer (STTR), has several advantages: It 1) relaxes the limitation of a fixed disparity range, 2) identifies occluded regions and provides confidence of estimation, and 3) imposes uniqueness constraints during the matching process. We report promising results on both synthetic and real-world datasets and demonstrate that STTR generalizes well across different domains, even without fine-tuning. Our code is publicly available at https://github.com/mli0603/stereo-transformer.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">STTRはステレオ深度推定で従来利用されているコストボリューム法の代わりに、自己と対応するもう片方の画像上のエピポーラ線上を注意対象とするTransformerと画素間対応が1対1となる制約を達成する最適輸送を利用。深度範囲が固定でなく、オクルージョンを陽に扱える。 <a href="https://t.co/VCWb6zFRuM">https://t.co/VCWb6zFRuM</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1325240113379467265?ref_src=twsrc%5Etfw">November 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. DTGAN: Dual Attention Generative Adversarial Networks for Text-to-Image  Generation

Zhenxing Zhang, Lambert Schomaker

- retweets: 56, favorites: 41 (11/09/2020 15:08:11)

- links: [abs](https://arxiv.org/abs/2011.02709) | [pdf](https://arxiv.org/pdf/2011.02709)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Most existing text-to-image generation methods adopt a multi-stage modular architecture which has three significant problems: (1) Training multiple networks can increase the run time and affect the convergence and stability of the generative model; (2) These approaches ignore the quality of early-stage generator images; (3) Many discriminators need to be trained. To this end, we propose the Dual Attention Generative Adversarial Network (DTGAN) which can synthesize high quality and visually realistic images only employing a single generator/discriminator pair. The proposed model introduces channel-aware and pixel-aware attention modules that can guide the generator to focus on text-relevant channels and pixels based on the global sentence vector and to fine-tune original feature maps using attention weights. Also, Conditional Adaptive Instance-Layer Normalization (CAdaILN) is presented to help our attention modules flexibly control the amount of change in shape and texture by the input natural-language description. Furthermore, a new type of visual loss is utilized to enhance the image quality by ensuring the vivid shape and the perceptually uniform color distributions of generated images. Experimental results on benchmark datasets demonstrate the superiority of our proposed method compared to the state-of-the-art models with a multi-stage framework. Visualization of the attention maps shows that the channel-aware attention module is able to localize the discriminative regions, while the pixel-aware attention module has the ability to capture the globally visual contents for the generation of an image.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DTGAN: Dual Attention Generative Adversarial Networks for Text-to-Image Generation<br>pdf: <a href="https://t.co/Ywr0FXA3to">https://t.co/Ywr0FXA3to</a><br>abs: <a href="https://t.co/yhs4KtE3Qk">https://t.co/yhs4KtE3Qk</a> <a href="https://t.co/aqd9rUUw28">pic.twitter.com/aqd9rUUw28</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1324534263375630337?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Teaching with Commentaries

Aniruddh Raghu, Maithra Raghu, Simon Kornblith, David Duvenaud, Geoffrey Hinton

- retweets: 36, favorites: 60 (11/09/2020 15:08:11)

- links: [abs](https://arxiv.org/abs/2011.03037) | [pdf](https://arxiv.org/pdf/2011.03037)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Effective training of deep neural networks can be challenging, and there remain many open questions on how to best learn these models. Recently developed methods to improve neural network training examine teaching: providing learned information during the training process to improve downstream model performance. In this paper, we take steps towards extending the scope of teaching. We propose a flexible teaching framework using commentaries, meta-learned information helpful for training on a particular task or dataset. We present an efficient and scalable gradient-based method to learn commentaries, leveraging recent work on implicit differentiation. We explore diverse applications of commentaries, from learning weights for individual training examples, to parameterizing label-dependent data augmentation policies, to representing attention masks that highlight salient image regions. In these settings, we find that commentaries can improve training speed and/or performance and also provide fundamental insights about the dataset and training process.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Teaching with Commentaries: <a href="https://t.co/7ObLGMbXIs">https://t.co/7ObLGMbXIs</a><br><br>We study the use of commentaries, metalearned auxiliary information, to improve neural network training and provide insights.<br><br>With <a href="https://twitter.com/maithra_raghu?ref_src=twsrc%5Etfw">@maithra_raghu</a>, <a href="https://twitter.com/skornblith?ref_src=twsrc%5Etfw">@skornblith</a>, <a href="https://twitter.com/DavidDuvenaud?ref_src=twsrc%5Etfw">@DavidDuvenaud</a>, <a href="https://twitter.com/geoffreyhinton?ref_src=twsrc%5Etfw">@geoffreyhinton</a><br><br>Thread⬇️ <a href="https://t.co/qqA6CVf2eO">pic.twitter.com/qqA6CVf2eO</a></p>&mdash; Aniruddh Raghu (@RaghuAniruddh) <a href="https://twitter.com/RaghuAniruddh/status/1324787707684352000?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Learning from Human Feedback: Challenges for Real-World Reinforcement  Learning in NLP

Julia Kreutzer, Stefan Riezler, Carolin Lawrence

- retweets: 54, favorites: 33 (11/09/2020 15:08:12)

- links: [abs](https://arxiv.org/abs/2011.02511) | [pdf](https://arxiv.org/pdf/2011.02511)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Large volumes of interaction logs can be collected from NLP systems that are deployed in the real world. How can this wealth of information be leveraged? Using such interaction logs in an offline reinforcement learning (RL) setting is a promising approach. However, due to the nature of NLP tasks and the constraints of production systems, a series of challenges arise. We present a concise overview of these challenges and discuss possible solutions.




# 14. Pseudo Random Number Generation through Reinforcement Learning and  Recurrent Neural Networks

Luca Pasqualini, Maurizio Parton

- retweets: 70, favorites: 17 (11/09/2020 15:08:12)

- links: [abs](https://arxiv.org/abs/2011.02909) | [pdf](https://arxiv.org/pdf/2011.02909)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A Pseudo-Random Number Generator (PRNG) is any algorithm generating a sequence of numbers approximating properties of random numbers. These numbers are widely employed in mid-level cryptography and in software applications. Test suites are used to evaluate PRNGs quality by checking statistical properties of the generated sequences. These sequences are commonly represented bit by bit. This paper proposes a Reinforcement Learning (RL) approach to the task of generating PRNGs from scratch by learning a policy to solve a partially observable Markov Decision Process (MDP), where the full state is the period of the generated sequence and the observation at each time step is the last sequence of bits appended to such state. We use a Long-Short Term Memory (LSTM) architecture to model the temporal relationship between observations at different time steps, by tasking the LSTM memory with the extraction of significant features of the hidden portion of the MDP's states. We show that modeling a PRNG with a partially observable MDP and a LSTM architecture largely improves the results of the fully observable feedforward RL approach introduced in previous work.




# 15. Imagining Grounded Conceptual Representations from Perceptual  Information in Situated Guessing Games

Alessandro Suglia, Antonio Vergari, Ioannis Konstas, Yonatan Bisk, Emanuele Bastianelli, Andrea Vanzo, Oliver Lemon

- retweets: 44, favorites: 32 (11/09/2020 15:08:12)

- links: [abs](https://arxiv.org/abs/2011.02917) | [pdf](https://arxiv.org/pdf/2011.02917)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In visual guessing games, a Guesser has to identify a target object in a scene by asking questions to an Oracle. An effective strategy for the players is to learn conceptual representations of objects that are both discriminative and expressive enough to ask questions and guess correctly. However, as shown by Suglia et al. (2020), existing models fail to learn truly multi-modal representations, relying instead on gold category labels for objects in the scene both at training and inference time. This provides an unnatural performance advantage when categories at inference time match those at training time, and it causes models to fail in more realistic "zero-shot" scenarios where out-of-domain object categories are involved. To overcome this issue, we introduce a novel "imagination" module based on Regularized Auto-Encoders, that learns context-aware and category-aware latent embeddings without relying on category labels at inference time. Our imagination module outperforms state-of-the-art competitors by 8.26% gameplay accuracy in the CompGuessWhat?! zero-shot scenario (Suglia et al., 2020), and it improves the Oracle and Guesser accuracy by 2.08% and 12.86% in the GuessWhat?! benchmark, when no gold categories are available at inference time. The imagination module also boosts reasoning about object properties and attributes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Camera-ready for our <a href="https://twitter.com/coling2020?ref_src=twsrc%5Etfw">@coling2020</a> paper &quot;Imagining Grounded Conceptual Representations in Situated Guessing Games&quot; is finally out: <a href="https://t.co/CTCS1sZ0Wf">https://t.co/CTCS1sZ0Wf</a>. Twitter thread and sneak preview of the presentation below. <a href="https://twitter.com/hashtag/NLP?src=hash&amp;ref_src=twsrc%5Etfw">#NLP</a> <a href="https://twitter.com/hashtag/dlearn?src=hash&amp;ref_src=twsrc%5Etfw">#dlearn</a> <a href="https://t.co/pA6hgQYxqE">https://t.co/pA6hgQYxqE</a> <a href="https://t.co/Ji4WkDzBrZ">pic.twitter.com/Ji4WkDzBrZ</a></p>&mdash; Alessandro Suglia (@ale_suglia) <a href="https://twitter.com/ale_suglia/status/1324778344454434816?ref_src=twsrc%5Etfw">November 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Language Model is All You Need: Natural Language Understanding as  Question Answering

Mahdi Namazifar, Alexandros Papangelis, Gokhan Tur, Dilek Hakkani-Tür

- retweets: 8, favorites: 57 (11/09/2020 15:08:12)

- links: [abs](https://arxiv.org/abs/2011.03023) | [pdf](https://arxiv.org/pdf/2011.03023)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Different flavors of transfer learning have shown tremendous impact in advancing research and applications of machine learning. In this work we study the use of a specific family of transfer learning, where the target domain is mapped to the source domain. Specifically we map Natural Language Understanding (NLU) problems to QuestionAnswering (QA) problems and we show that in low data regimes this approach offers significant improvements compared to other approaches to NLU. Moreover we show that these gains could be increased through sequential transfer learning across NLU problems from different domains. We show that our approach could reduce the amount of required data for the same performance by up to a factor of 10.



