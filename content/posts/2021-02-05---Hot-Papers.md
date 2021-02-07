---
title: Hot Papers 2021-02-05
date: 2021-02-07T12:58:11.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-05"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-05"
socialImage: "/media/flying-marine.jpg"

---

# 1. Understanding the Capabilities, Limitations, and Societal Impact of  Large Language Models

Alex Tamkin, Miles Brundage, Jack Clark, Deep Ganguli

- retweets: 5190, favorites: 100 (02/07/2021 12:58:11)

- links: [abs](https://arxiv.org/abs/2102.02503) | [pdf](https://arxiv.org/pdf/2102.02503)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

On October 14th, 2020, researchers from OpenAI, the Stanford Institute for Human-Centered Artificial Intelligence, and other universities convened to discuss open research questions surrounding GPT-3, the largest publicly-disclosed dense language model at the time. The meeting took place under Chatham House Rules. Discussants came from a variety of research backgrounds including computer science, linguistics, philosophy, political science, communications, cyber policy, and more. Broadly, the discussion centered around two main questions: 1) What are the technical capabilities and limitations of large language models? 2) What are the societal effects of widespread use of large language models? Here, we provide a detailed summary of the discussion organized by the two themes above.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">2020年10月14日、OpenAIを含む様々な研究者たちが集まり、多様なタスクを解ける言語モデル「GPT-3」について議論した。その時の詳細が公開された。読み物として面白かった。<br>主な議論は<br>①大規模言語モデルの能力と制限は何か<br>②大規模言語モデルの普及による社会的影響は何か<a href="https://t.co/tmUF9LevnF">https://t.co/tmUF9LevnF</a></p>&mdash; 小猫遊りょう（たかにゃし・りょう） (@jaguring1) <a href="https://twitter.com/jaguring1/status/1357675138129403905?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A few months back, OpenAI and Stanford&#39; Institute for Human-Centered AI co-hosted a workshop on the capabilities/limitations/societal implications of large language models like GPT-3. <br><br>The proceedings can be found here: <a href="https://t.co/7rLNa619sg">https://t.co/7rLNa619sg</a></p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1357506274544373760?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Unifying Vision-and-Language Tasks via Text Generation

Jaemin Cho, Jie Lei, Hao Tan, Mohit Bansal

- retweets: 2206, favorites: 323 (02/07/2021 12:58:11)

- links: [abs](https://arxiv.org/abs/2102.02779) | [pdf](https://arxiv.org/pdf/2102.02779)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Existing methods for vision-and-language learning typically require designing task-specific architectures and objectives for each task. For example, a multi-label answer classifier for visual question answering, a region scorer for referring expression comprehension, and a language decoder for image captioning, etc. To alleviate these hassles, in this work, we propose a unified framework that learns different tasks in a single architecture with the same language modeling objective, i.e., multimodal conditional text generation, where our models learn to generate labels in text based on the visual and textual inputs. On 7 popular vision-and-language benchmarks, including visual question answering, referring expression comprehension, visual commonsense reasoning, most of which have been previously modeled as discriminative tasks, our generative approach (with a single unified architecture) reaches comparable performance to recent task-specific state-of-the-art vision-and-language models. Moreover, our generative approach shows better generalization ability on answering questions that have rare answers. In addition, we show that our framework allows multi-task learning in a single architecture with a single set of parameters, which achieves similar performance to separately optimized single-task models. Our code will be publicly available at: https://github.com/j-min/VL-T5

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Presenting our new V+L pretraining work: “Unifying Vision-and-Language Tasks via Text Generation”,<br>a single unified generative framework (VL-T5 / VL-BART) for diverse multimodal tasks!<br><br>Arxiv: <a href="https://t.co/QWLWgRIhp7">https://t.co/QWLWgRIhp7</a><br><br>Work done w/ <a href="https://twitter.com/jayleicn?ref_src=twsrc%5Etfw">@jayleicn</a>  <a href="https://twitter.com/HaoTan5?ref_src=twsrc%5Etfw">@HaoTan5</a>  <a href="https://twitter.com/mohitban47?ref_src=twsrc%5Etfw">@mohitban47</a> (<a href="https://twitter.com/uncnlp?ref_src=twsrc%5Etfw">@uncnlp</a>)<br><br>🧵1/n <a href="https://t.co/SLweqOWbI2">pic.twitter.com/SLweqOWbI2</a></p>&mdash; Jaemin Cho (@jmin__cho) <a href="https://twitter.com/jmin__cho/status/1357515015264161792?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">視覚理解能力を持つように事前学習済み言語モデル「T5」と「BART」を拡張する研究。「VL-T5」と「VL-BART」と名付けている。生成的アプローチ（単一の構造）で7つの視覚+言語タスクで最高性能に匹敵するとのこと（VQA, GQA, VCR, NLVR2, RefCOCOg, COCO caption, Multi30k）<a href="https://t.co/HjcdJaswQm">https://t.co/HjcdJaswQm</a></p>&mdash; 小猫遊りょう（たかにゃし・りょう） (@jaguring1) <a href="https://twitter.com/jaguring1/status/1358022771620499458?ref_src=twsrc%5Etfw">February 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unifying Vision-and-Language Tasks via Text Generation<br>pdf: <a href="https://t.co/6HQ52pyIT0">https://t.co/6HQ52pyIT0</a><br>abs: <a href="https://t.co/OqYYbT1laJ">https://t.co/OqYYbT1laJ</a> <a href="https://t.co/fiepBmlqZJ">pic.twitter.com/fiepBmlqZJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357508112417226753?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Regenerating Soft Robots through Neural Cellular Automata

Kazuya Horibe, Kathryn Walker, Sebastian Risi

- retweets: 818, favorites: 146 (02/07/2021 12:58:12)

- links: [abs](https://arxiv.org/abs/2102.02579) | [pdf](https://arxiv.org/pdf/2102.02579)
- [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent) | [q-bio.PE](https://arxiv.org/list/q-bio.PE/recent)

Morphological regeneration is an important feature that highlights the environmental adaptive capacity of biological systems. Lack of this regenerative capacity significantly limits the resilience of machines and the environments they can operate in. To aid in addressing this gap, we develop an approach for simulated soft robots to regrow parts of their morphology when being damaged. Although numerical simulations using soft robots have played an important role in their design, evolving soft robots with regenerative capabilities have so far received comparable little attention. Here we propose a model for soft robots that regenerate through a neural cellular automata. Importantly, this approach only relies on local cell information to regrow damaged components, opening interesting possibilities for physical regenerable soft robots in the future. Our approach allows simulated soft robots that are damaged to partially regenerate their original morphology through local cell interactions alone and regain some of their ability to locomote. These results take a step towards equipping artificial systems with regenerative capacities and could potentially allow for more robust operations in a variety of situations and environments. The code for the experiments in this paper is available at: \url{github.com/KazuyaHoribe/RegeneratingSoftRobots}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are happy to present &quot;Regenerating Soft Robots through Neural Cellular Automata&quot; w/ <a href="https://twitter.com/khoribe3?ref_src=twsrc%5Etfw">@khoribe3</a>, <a href="https://twitter.com/katt_walker?ref_src=twsrc%5Etfw">@katt_walker</a> <br><br>The approach allows soft robots to regrow parts of their morphology when being damaged only based on local cell communication.<br><br>PDF: <a href="https://t.co/WnQ64kb03z">https://t.co/WnQ64kb03z</a> <a href="https://t.co/DzItkC4Qz1">pic.twitter.com/DzItkC4Qz1</a></p>&mdash; Sebastian Risi (@risi1979) <a href="https://twitter.com/risi1979/status/1358018266824912897?ref_src=twsrc%5Etfw">February 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Sovereign Smartphone: To Enjoy Freedom We Have to Control Our Phones

Friederike Groschupp, Moritz Schneider, Ivan Puddu, Shweta Shinde, Srdjan Capkun

- retweets: 841, favorites: 79 (02/07/2021 12:58:12)

- links: [abs](https://arxiv.org/abs/2102.02743) | [pdf](https://arxiv.org/pdf/2102.02743)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

The majority of smartphones either run iOS or Android operating systems. This has created two distinct ecosystems largely controlled by Apple and Google - they dictate which applications can run, how they run, and what kind of phone resources they can access. Barring some exceptions in Android where different phone manufacturers may have influence, users, developers, and governments are left with little to no choice. Specifically, users need to entrust their security and privacy to OS vendors and accept the functionality constraints they impose. Given the wide use of Android and iOS, immediately leaving these ecosystems is not practical, except in niche application areas. In this work, we draw attention to the magnitude of this problem and why it is an undesirable situation. As an alternative, we advocate the development of a new smartphone architecture that securely transfers the control back to the users while maintaining compatibility with the rich existing smartphone ecosystems. We propose and analyze one such design based on advances in trusted execution environments for ARM and RISC-V.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Too much control over smartphones is in hands of few companies. Gatekeeping limits developers, users, governments. A different phone architecture can hand control back to the users while still protecting existing ecosystems.  With <a href="https://twitter.com/shw3ta_shinde?ref_src=twsrc%5Etfw">@shw3ta_shinde</a> <a href="https://twitter.com/dn0sar?ref_src=twsrc%5Etfw">@dn0sar</a>.  <a href="https://t.co/Yzcps6UPFG">https://t.co/Yzcps6UPFG</a> <a href="https://t.co/oNX5PNsUeX">pic.twitter.com/oNX5PNsUeX</a></p>&mdash; Srdjan Čapkun (@SrdjanCapkun) <a href="https://twitter.com/SrdjanCapkun/status/1357597373082517504?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Im2Vec: Synthesizing Vector Graphics without Vector Supervision

Pradyumna Reddy, Michael Gharbi, Michal Lukac, Niloy J. Mitra

- retweets: 703, favorites: 135 (02/07/2021 12:58:12)

- links: [abs](https://arxiv.org/abs/2102.02798) | [pdf](https://arxiv.org/pdf/2102.02798)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Vector graphics are widely used to represent fonts, logos, digital artworks, and graphic designs. But, while a vast body of work has focused on generative algorithms for raster images, only a handful of options exists for vector graphics. One can always rasterize the input graphic and resort to image-based generative approaches, but this negates the advantages of the vector representation. The current alternative is to use specialized models that require explicit supervision on the vector graphics representation at training time. This is not ideal because large-scale high quality vector-graphics datasets are difficult to obtain. Furthermore, the vector representation for a given design is not unique, so models that supervise on the vector representation are unnecessarily constrained. Instead, we propose a new neural network that can generate complex vector graphics with varying topologies, and only requires indirect supervision from readily-available raster training images (i.e., with no vector counterparts). To enable this, we use a differentiable rasterization pipeline that renders the generated vector shapes and composites them together onto a raster canvas. We demonstrate our method on a range of datasets, and provide comparison with state-of-the-art SVG-VAE and DeepSVG, both of which require explicit vector graphics supervision. Finally, we also demonstrate our approach on the MNIST dataset, for which no groundtruth vector representation is available. Source code, datasets, and more results are available at http://geometry.cs.ucl.ac.uk/projects/2020/Im2Vec/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Im2Vec: Synthesizing Vector Graphics without Vector Supervision<br>pdf: <a href="https://t.co/ywASUSzvuv">https://t.co/ywASUSzvuv</a><br>abs: <a href="https://t.co/9ByeBDRNH7">https://t.co/9ByeBDRNH7</a> <a href="https://t.co/q2DwZ6Fxf9">pic.twitter.com/q2DwZ6Fxf9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357516072719691777?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Designing an Encoder for StyleGAN Image Manipulation

Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, Daniel Cohen-Or

- retweets: 256, favorites: 76 (02/07/2021 12:58:13)

- links: [abs](https://arxiv.org/abs/2102.02766) | [pdf](https://arxiv.org/pdf/2102.02766)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, there has been a surge of diverse methods for performing image editing by employing pre-trained unconditional generators. Applying these methods on real images, however, remains a challenge, as it necessarily requires the inversion of the images into their latent space. To successfully invert a real image, one needs to find a latent code that reconstructs the input image accurately, and more importantly, allows for its meaningful manipulation. In this paper, we carefully study the latent space of StyleGAN, the state-of-the-art unconditional generator. We identify and analyze the existence of a distortion-editability tradeoff and a distortion-perception tradeoff within the StyleGAN latent space. We then suggest two principles for designing encoders in a manner that allows one to control the proximity of the inversions to regions that StyleGAN was originally trained on. We present an encoder based on our two principles that is specifically designed for facilitating editing on real images by balancing these tradeoffs. By evaluating its performance qualitatively and quantitatively on numerous challenging domains, including cars and horses, we show that our inversion method, followed by common editing techniques, achieves superior real-image editing quality, with only a small reconstruction accuracy drop.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Designing an Encoder for StyleGAN Image Manipulation<br>pdf: <a href="https://t.co/6A33wc9xxn">https://t.co/6A33wc9xxn</a><br>abs: <a href="https://t.co/zNv4LrvY7r">https://t.co/zNv4LrvY7r</a> <a href="https://t.co/GTicRuDFpQ">pic.twitter.com/GTicRuDFpQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357513853458931712?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Sampling in Combinatorial Spaces with SurVAE Flow Augmented MCMC

Priyank Jaini, Didrik Nielsen, Max Welling

- retweets: 130, favorites: 46 (02/07/2021 12:58:13)

- links: [abs](https://arxiv.org/abs/2102.02374) | [pdf](https://arxiv.org/pdf/2102.02374)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Hybrid Monte Carlo is a powerful Markov Chain Monte Carlo method for sampling from complex continuous distributions. However, a major limitation of HMC is its inability to be applied to discrete domains due to the lack of gradient signal. In this work, we introduce a new approach based on augmenting Monte Carlo methods with SurVAE Flows to sample from discrete distributions using a combination of neural transport methods like normalizing flows and variational dequantization, and the Metropolis-Hastings rule. Our method first learns a continuous embedding of the discrete space using a surjective map and subsequently learns a bijective transformation from the continuous space to an approximately Gaussian distributed latent variable. Sampling proceeds by simulating MCMC chains in the latent space and mapping these samples to the target discrete space via the learned transformations. We demonstrate the efficacy of our algorithm on a range of examples from statistics, computational physics and machine learning, and observe improvements compared to alternative algorithms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to present our new work “Sampling in Combinatorial Spaces with SurVAE Flow augmented MCMC” with <a href="https://twitter.com/nielsen_didrik?ref_src=twsrc%5Etfw">@nielsen_didrik</a>  and <a href="https://twitter.com/wellingmax?ref_src=twsrc%5Etfw">@wellingmax</a> to appear <a href="https://twitter.com/aistats_conf?ref_src=twsrc%5Etfw">@aistats_conf</a>. <a href="https://twitter.com/hashtag/AISTATS2021?src=hash&amp;ref_src=twsrc%5Etfw">#AISTATS2021</a><br><br>Paper: <a href="https://t.co/ALC1QLj9Wg">https://t.co/ALC1QLj9Wg</a><br>Code: [coming soon] <br><br>1/6</p>&mdash; Priyank Jaini (@priyankjaini) <a href="https://twitter.com/priyankjaini/status/1357797386576400385?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. RoI Tanh-polar Transformer Network for Face Parsing in the Wild

Yiming Lin, Jie Shen, Yujiang Wang, Maja Pantic

- retweets: 72, favorites: 34 (02/07/2021 12:58:13)

- links: [abs](https://arxiv.org/abs/2102.02717) | [pdf](https://arxiv.org/pdf/2102.02717)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Face parsing aims to predict pixel-wise labels for facial components of a target face in an image. Existing approaches usually crop the target face from the input image with respect to a bounding box calculated during pre-processing, and thus can only parse inner facial Regions of Interest (RoIs). Peripheral regions like hair are ignored and nearby faces that are partially included in the bounding box can cause distractions. Moreover, these methods are only trained and evaluated on near-frontal portrait images and thus their performance for in-the-wild cases were unexplored. To address these issues, this paper makes three contributions. First, we introduce iBugMask dataset for face parsing in the wild containing 1,000 manually annotated images with large variations in sizes, poses, expressions and background, and Helen-LP, a large-pose training set containing 21,866 images generated using head pose augmentation. Second, we propose RoI Tanh-polar transform that warps the whole image to a Tanh-polar representation with a fixed ratio between the face area and the context, guided by the target bounding box. The new representation contains all information in the original image, and allows for rotation equivariance in the convolutional neural networks (CNNs). Third, we propose a hybrid residual representation learning block, coined HybridBlock, that contains convolutional layers in both the Tanh-polar space and the Tanh-Cartesian space, allowing for receptive fields of different shapes in CNNs. Through extensive experiments, we show that the proposed method significantly improves the state-of-the-art for face parsing in the wild.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RoI Tanh-polar Transformer Network for Face Parsing in the<br>Wild<br>pdf: <a href="https://t.co/pikSR1qJdJ">https://t.co/pikSR1qJdJ</a><br>abs: <a href="https://t.co/c3jtkmmLhh">https://t.co/c3jtkmmLhh</a> <a href="https://t.co/gJV8mVcbxK">pic.twitter.com/gJV8mVcbxK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357509868521988096?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Only a Matter of Style: Age Transformation Using a Style-Based  Regression Model

Yuval Alaluf, Or Patashnik, Daniel Cohen-Or

- retweets: 63, favorites: 32 (02/07/2021 12:58:13)

- links: [abs](https://arxiv.org/abs/2102.02754) | [pdf](https://arxiv.org/pdf/2102.02754)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The task of age transformation illustrates the change of an individual's appearance over time. Accurately modeling this complex transformation over an input facial image is extremely challenging as it requires making convincing and possibly large changes to facial features and head shape, while still preserving the input identity. In this work, we present an image-to-image translation method that learns to directly encode real facial images into the latent space of a pre-trained unconditional GAN (e.g., StyleGAN) subject to a given aging shift. We employ a pre-trained age regression network used to explicitly guide the encoder in generating the latent codes corresponding to the desired age. In this formulation, our method approaches the continuous aging process as a regression task between the input age and desired target age, providing fine-grained control over the generated image. Moreover, unlike other approaches that operate solely in the latent space using a prior on the path controlling age, our method learns a more disentangled, non-linear path. Finally, we demonstrate that the end-to-end nature of our approach, coupled with the rich semantic latent space of StyleGAN, allows for further editing of the generated images. Qualitative and quantitative evaluations show the advantages of our method compared to state-of-the-art approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Only a Matter of Style: Age Transformation Using a Style-Based Regression Model<br>pdf: <a href="https://t.co/dNVUN4mXCP">https://t.co/dNVUN4mXCP</a><br>abs: <a href="https://t.co/pDGmHGa5u7">https://t.co/pDGmHGa5u7</a><br>github: <a href="https://t.co/fyQwuOzVgW">https://t.co/fyQwuOzVgW</a> <a href="https://t.co/2SBB9G3Tqv">pic.twitter.com/2SBB9G3Tqv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357512178706886656?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Adaptive Semiparametric Language Models

Dani Yogatama, Cyprien de Masson d'Autume, Lingpeng Kong

- retweets: 66, favorites: 27 (02/07/2021 12:58:13)

- links: [abs](https://arxiv.org/abs/2102.02557) | [pdf](https://arxiv.org/pdf/2102.02557)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We present a language model that combines a large parametric neural network (i.e., a transformer) with a non-parametric episodic memory component in an integrated architecture. Our model uses extended short-term context by caching local hidden states -- similar to transformer-XL -- and global long-term memory by retrieving a set of nearest neighbor tokens at each timestep. We design a gating function to adaptively combine multiple information sources to make a prediction. This mechanism allows the model to use either local context, short-term memory, or long-term memory (or any combination of them) on an ad hoc basis depending on the context. Experiments on word-based and character-based language modeling datasets demonstrate the efficacy of our proposed method compared to strong baselines.

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">Adaptive Semiparametric Language Models<br>pdf: <a href="https://t.co/n8OVzEU9YC">https://t.co/n8OVzEU9YC</a><br>abs: <a href="https://t.co/A5qhcN1c9V">https://t.co/A5qhcN1c9V</a> <a href="https://t.co/46faweWQ1r">pic.twitter.com/46faweWQ1r</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357507446399467532?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. MUFASA: Multimodal Fusion Architecture Search for Electronic Health  Records

Zhen Xu, David R. So, Andrew M. Dai

- retweets: 49, favorites: 31 (02/07/2021 12:58:13)

- links: [abs](https://arxiv.org/abs/2102.02340) | [pdf](https://arxiv.org/pdf/2102.02340)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

One important challenge of applying deep learning to electronic health records (EHR) is the complexity of their multimodal structure. EHR usually contains a mixture of structured (codes) and unstructured (free-text) data with sparse and irregular longitudinal features -- all of which doctors utilize when making decisions. In the deep learning regime, determining how different modality representations should be fused together is a difficult problem, which is often addressed by handcrafted modeling and intuition. In this work, we extend state-of-the-art neural architecture search (NAS) methods and propose MUltimodal Fusion Architecture SeArch (MUFASA) to simultaneously search across multimodal fusion strategies and modality-specific architectures for the first time. We demonstrate empirically that our MUFASA method outperforms established unimodal NAS on public EHR data with comparable computation costs. In addition, MUFASA produces architectures that outperform Transformer and Evolved Transformer. Compared with these baselines on CCS diagnosis code prediction, our discovered models improve top-5 recall from 0.88 to 0.91 and demonstrate the ability to generalize to other EHR tasks. Studying our top architecture in depth, we provide empirical evidence that MUFASA's improvements are derived from its ability to both customize modeling for each data modality and find effective fusion strategies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MUFASA: Multimodal Fusion Architecture Search for Electronic Health Records<br>pdf: <a href="https://t.co/70O9mgpW9k">https://t.co/70O9mgpW9k</a><br>abs: <a href="https://t.co/oKU75SARUo">https://t.co/oKU75SARUo</a> <a href="https://t.co/n5v4LoBdnA">pic.twitter.com/n5v4LoBdnA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357508861301776389?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. A formalization of Dedekind domains and class groups of global fields

Anne Baanen, Sander R. Dahmen, Ashvni Narayanan, Filippo A. E. Nuccio Mortarino Majno di Capriglio

- retweets: 42, favorites: 34 (02/07/2021 12:58:13)

- links: [abs](https://arxiv.org/abs/2102.02600) | [pdf](https://arxiv.org/pdf/2102.02600)
- [cs.LO](https://arxiv.org/list/cs.LO/recent) | [math.NT](https://arxiv.org/list/math.NT/recent)

Dedekind domains and their class groups are notions in commutative algebra that are essential in algebraic number theory. We formalized these structures and several fundamental properties, including number theoretic finiteness results for class groups, in the Lean prover as part of the mathlib mathematical library. This paper describes the formalization process, noting the idioms we found useful in our development and mathlib's decentralized collaboration processes involved in this project.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I say (angrily) &quot;30+ years of computer proof verifiers and still no undergraduate maths curriculum&quot;. Sometimes I hear &quot;oh come on, it must have been done by now in Coq+Lean+Isabelle/HOL&quot;. My standard counterexample: finiteness of class group. But today: <a href="https://t.co/cFFu1aEQEX">https://t.co/cFFu1aEQEX</a></p>&mdash; The Xena Project (@XenaProject) <a href="https://twitter.com/XenaProject/status/1357701143502151681?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. MeInGame: Create a Game Character Face from a Single Portrait

Jiangke Lin, Yi Yuan, Zhengxia Zou

- retweets: 14, favorites: 37 (02/07/2021 12:58:14)

- links: [abs](https://arxiv.org/abs/2102.02371) | [pdf](https://arxiv.org/pdf/2102.02371)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Many deep learning based 3D face reconstruction methods have been proposed recently, however, few of them have applications in games. Current game character customization systems either require players to manually adjust considerable face attributes to obtain the desired face, or have limited freedom of facial shape and texture. In this paper, we propose an automatic character face creation method that predicts both facial shape and texture from a single portrait, and it can be integrated into most existing 3D games. Although 3D Morphable Face Model (3DMM) based methods can restore accurate 3D faces from single images, the topology of 3DMM mesh is different from the meshes used in most games. To acquire fidelity texture, existing methods require a large amount of face texture data for training, while building such datasets is time-consuming and laborious. Besides, such a dataset collected under laboratory conditions may not generalized well to in-the-wild situations. To tackle these problems, we propose 1) a low-cost facial texture acquisition method, 2) a shape transfer algorithm that can transform the shape of a 3DMM mesh to games, and 3) a new pipeline for training 3D game face reconstruction networks. The proposed method not only can produce detailed and vivid game characters similar to the input portrait, but can also eliminate the influence of lighting and occlusions. Experiments show that our method outperforms state-of-the-art methods used in games.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MeInGame: Create a Game Character Face from a Single Portrait<br>pdf: <a href="https://t.co/tyro9CXAco">https://t.co/tyro9CXAco</a><br>abs: <a href="https://t.co/mmg5jFKwY0">https://t.co/mmg5jFKwY0</a> <a href="https://t.co/cjN106WKoa">pic.twitter.com/cjN106WKoa</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1357520250112212996?ref_src=twsrc%5Etfw">February 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



