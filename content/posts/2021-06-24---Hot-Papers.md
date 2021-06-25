---
title: Hot Papers 2021-06-24
date: 2021-06-25T10:13:42.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-24"
socialImage: "/media/flying-marine.jpg"

---

# 1. Fine-Tuning StyleGAN2 For Cartoon Face Generation

Jihye Back

- retweets: 1558, favorites: 153 (06/25/2021 10:13:42)

- links: [abs](https://arxiv.org/abs/2106.12445) | [pdf](https://arxiv.org/pdf/2106.12445)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Recent studies have shown remarkable success in the unsupervised image to image (I2I) translation. However, due to the imbalance in the data, learning joint distribution for various domains is still very challenging. Although existing models can generate realistic target images, it's difficult to maintain the structure of the source image. In addition, training a generative model on large data in multiple domains requires a lot of time and computer resources. To address these limitations, we propose a novel image-to-image translation method that generates images of the target domain by finetuning a stylegan2 pretrained model. The stylegan2 model is suitable for unsupervised I2I translation on unbalanced datasets; it is highly stable, produces realistic images, and even learns properly from limited data when applied with simple fine-tuning techniques. Thus, in this paper, we propose new methods to preserve the structure of the source images and generate realistic images in the target domain. The code and results are available at https://github.com/happy-jihye/Cartoon-StyleGan2

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fine-Tuning StyleGAN2 For Cartoon Face Generation<br>pdf: <a href="https://t.co/0nR0KxfyLF">https://t.co/0nR0KxfyLF</a><br>abs: <a href="https://t.co/v7Dq9lISAj">https://t.co/v7Dq9lISAj</a><br>github: <a href="https://t.co/pc6NeQR2is">https://t.co/pc6NeQR2is</a> <a href="https://t.co/UQQFhwliuB">pic.twitter.com/UQQFhwliuB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407866961732767749?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Vision Permutator: A Permutable MLP-Like Architecture for Visual  Recognition

Qibin Hou, Zihang Jiang, Li Yuan, Ming-Ming Cheng, Shuicheng Yan, Jiashi Feng

- retweets: 957, favorites: 136 (06/25/2021 10:13:43)

- links: [abs](https://arxiv.org/abs/2106.12368) | [pdf](https://arxiv.org/pdf/2106.12368)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we present Vision Permutator, a conceptually simple and data efficient MLP-like architecture for visual recognition. By realizing the importance of the positional information carried by 2D feature representations, unlike recent MLP-like models that encode the spatial information along the flattened spatial dimensions, Vision Permutator separately encodes the feature representations along the height and width dimensions with linear projections. This allows Vision Permutator to capture long-range dependencies along one spatial direction and meanwhile preserve precise positional information along the other direction. The resulting position-sensitive outputs are then aggregated in a mutually complementing manner to form expressive representations of the objects of interest. We show that our Vision Permutators are formidable competitors to convolutional neural networks (CNNs) and vision transformers. Without the dependence on spatial convolutions or attention mechanisms, Vision Permutator achieves 81.5% top-1 accuracy on ImageNet without extra large-scale training data (e.g., ImageNet-22k) using only 25M learnable parameters, which is much better than most CNNs and vision transformers under the same model size constraint. When scaling up to 88M, it attains 83.2% top-1 accuracy. We hope this work could encourage research on rethinking the way of encoding spatial information and facilitate the development of MLP-like models. Code is available at https://github.com/Andrew-Qibin/VisionPermutator.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition<br>pdf: <a href="https://t.co/2Kb6zoFwR1">https://t.co/2Kb6zoFwR1</a><br>github: <a href="https://t.co/Ahe17BEDwf">https://t.co/Ahe17BEDwf</a><br><br>achieves 81.5% top-1 accuracy on ImageNet without extra large-scale training data (e.g., ImageNet-22k) using only 25M learnable parameters <a href="https://t.co/KO1Kx7OnSi">pic.twitter.com/KO1Kx7OnSi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407860562445443074?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Real-time Neural Radiance Caching for Path Tracing

Thomas Müller, Fabrice Rousselle, Jan Novák, Alexander Keller

- retweets: 342, favorites: 62 (06/25/2021 10:13:43)

- links: [abs](https://arxiv.org/abs/2106.12372) | [pdf](https://arxiv.org/pdf/2106.12372)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a real-time neural radiance caching method for path-traced global illumination. Our system is designed to handle fully dynamic scenes, and makes no assumptions about the lighting, geometry, and materials. The data-driven nature of our approach sidesteps many difficulties of caching algorithms, such as locating, interpolating, and updating cache points. Since pretraining neural networks to handle novel, dynamic scenes is a formidable generalization challenge, we do away with pretraining and instead achieve generalization via adaptation, i.e. we opt for training the radiance cache while rendering. We employ self-training to provide low-noise training targets and simulate infinite-bounce transport by merely iterating few-bounce training updates. The updates and cache queries incur a mild overhead -- about 2.6ms on full HD resolution -- thanks to a streaming implementation of the neural network that fully exploits modern hardware. We demonstrate significant noise reduction at the cost of little induced bias, and report state-of-the-art, real-time performance on a number of challenging scenarios.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Real-time Neural Radiance Caching for Path Tracing<br>pdf: <a href="https://t.co/4bjpRxyBkq">https://t.co/4bjpRxyBkq</a><br>abs: <a href="https://t.co/PsGQbUdtso">https://t.co/PsGQbUdtso</a><br><br>significant noise reduction at the cost of little induced bias, and sota real-time performance on a number of challenging scenarios <a href="https://t.co/yW1UXgovLL">pic.twitter.com/yW1UXgovLL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407865296669581317?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Strategic Liquidity Provision in Uniswap v3

Michael Neuder, Rithvik Rao, Daniel J. Moroz, David C. Parkes

- retweets: 204, favorites: 88 (06/25/2021 10:13:43)

- links: [abs](https://arxiv.org/abs/2106.12033) | [pdf](https://arxiv.org/pdf/2106.12033)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.GT](https://arxiv.org/list/cs.GT/recent)

Uniswap is the largest decentralized exchange for digital currencies. The newest version, called Uniswap v3, allows liquidity providers to allocate liquidity to one or more closed intervals of the price of an asset, instead of over the total range of prices. While the price of the asset remains in that interval, the liquidity provider earns rewards proportionally to the amount of liquidity allocated. This induces the problem of strategic liquidity provision: smaller intervals result in higher concentration of liquidity and correspondingly larger rewards when the price remains in the interval, but with higher risk. We formalize this problem and study three classes of strategies for liquidity providers: uniform, proportional, and optimal (via a constrained optimization problem). We present experimental results based on the historical price data of Ethereum, which show that simple liquidity provision strategies can yield near-optimal utility and earn over 200x more than Uniswap v2 liquidity provision.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Nice work on Uniswap V3 strategies, <a href="https://twitter.com/dmoroz?ref_src=twsrc%5Etfw">@dmoroz</a> <a href="https://t.co/iDEioO5kzt">https://t.co/iDEioO5kzt</a></p>&mdash; Tarun Chitra (@tarunchitra) <a href="https://twitter.com/tarunchitra/status/1408167132299411457?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Weisfeiler and Lehman Go Cellular: CW Networks

Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yu Guang Wang, Pietro Liò, Guido Montúfar, Michael Bronstein

- retweets: 169, favorites: 69 (06/25/2021 10:13:43)

- links: [abs](https://arxiv.org/abs/2106.12575) | [pdf](https://arxiv.org/pdf/2106.12575)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Graph Neural Networks (GNNs) are limited in their expressive power, struggle with long-range interactions and lack a principled way to model higher-order structures. These problems can be attributed to the strong coupling between the computational graph and the input graph structure. The recently proposed Message Passing Simplicial Networks naturally decouple these elements by performing message passing on the clique complex of the graph. Nevertheless, these models are severely constrained by the rigid combinatorial structure of Simplicial Complexes (SCs). In this work, we extend recent theoretical results on SCs to regular Cell Complexes, topological objects that flexibly subsume SCs and graphs. We show that this generalisation provides a powerful set of graph ``lifting'' transformations, each leading to a unique hierarchical message passing procedure. The resulting methods, which we collectively call CW Networks (CWNs), are strictly more powerful than the WL test and, in certain cases, not less powerful than the 3-WL test. In particular, we demonstrate the effectiveness of one such scheme, based on rings, when applied to molecular graph problems. The proposed architecture benefits from provably larger expressivity than commonly used GNNs, principled modelling of higher-order signals and from compressing the distances between nodes. We demonstrate that our model achieves state-of-the-art results on a variety of molecular datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Weisfeiler and Lehman continue their topological journey. Their next stop? Cell complexes. We obtain SOTA results on molecular prediction tasks: <a href="https://t.co/stIVpIwiMk">https://t.co/stIVpIwiMk</a> <br><br>w/ an amazing team: <a href="https://twitter.com/ffabffrasca?ref_src=twsrc%5Etfw">@ffabffrasca</a>* <a href="https://twitter.com/kneppkatt?ref_src=twsrc%5Etfw">@kneppkatt</a> <a href="https://twitter.com/wangyg85?ref_src=twsrc%5Etfw">@wangyg85</a> <a href="https://twitter.com/pl219_Cambridge?ref_src=twsrc%5Etfw">@pl219_Cambridge</a> <a href="https://twitter.com/guidomontufar?ref_src=twsrc%5Etfw">@guidomontufar</a> <a href="https://twitter.com/mmbronstein?ref_src=twsrc%5Etfw">@mmbronstein</a>  1/7 <a href="https://t.co/ul6AjZZaMz">pic.twitter.com/ul6AjZZaMz</a></p>&mdash; Cristian Bodnar (@CristianBodnar) <a href="https://twitter.com/CristianBodnar/status/1408066831676444676?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Deep Gaussian Processes: A Survey

Kalvik Jakkala

- retweets: 192, favorites: 0 (06/25/2021 10:13:43)

- links: [abs](https://arxiv.org/abs/2106.12135) | [pdf](https://arxiv.org/pdf/2106.12135)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Gaussian processes are one of the dominant approaches in Bayesian learning. Although the approach has been applied to numerous problems with great success, it has a few fundamental limitations. Multiple methods in literature have addressed these limitations. However, there has not been a comprehensive survey of the topics as of yet. Most existing surveys focus on only one particular variant of Gaussian processes and their derivatives. This survey details the core motivations for using Gaussian processes, their mathematical formulations, limitations, and research themes that have flourished over the years to address said limitations. Furthermore, one particular research area is Deep Gaussian Processes (DGPs), it has improved substantially in the past decade. The significant publications that advanced the forefront of this research area are outlined in their survey. Finally, a brief discussion on open problems and research directions for future work is presented at the end.




# 7. LegoFormer: Transformers for Block-by-Block Multi-view 3D Reconstruction

Farid Yagubbayli, Alessio Tonioni, Federico Tombari

- retweets: 100, favorites: 50 (06/25/2021 10:13:43)

- links: [abs](https://arxiv.org/abs/2106.12102) | [pdf](https://arxiv.org/pdf/2106.12102)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Most modern deep learning-based multi-view 3D reconstruction techniques use RNNs or fusion modules to combine information from multiple images after encoding them. These two separate steps have loose connections and do not consider all available information while encoding each view. We propose LegoFormer, a transformer-based model that unifies object reconstruction under a single framework and parametrizes the reconstructed occupancy grid by its decomposition factors. This reformulation allows the prediction of an object as a set of independent structures then aggregated to obtain the final reconstruction. Experiments conducted on ShapeNet display the competitive performance of our network with respect to the state-of-the-art methods. We also demonstrate how the use of self-attention leads to increased interpretability of the model output.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LegoFormer: Transformers for Block-by-Block Multi-view 3D Reconstruction<br>pdf: <a href="https://t.co/kFr0WpFI4L">https://t.co/kFr0WpFI4L</a><br><br>transformer-based model that unifies object reconstruction under a single framework and parametrizes the reconstructed occupancy grid by its decomposition<br>factors <a href="https://t.co/jkyAZH1PrD">pic.twitter.com/jkyAZH1PrD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407862493725270027?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Volume Rendering of Neural Implicit Surfaces

Lior Yariv, Jiatao Gu, Yoni Kasten, Yaron Lipman

- retweets: 63, favorites: 53 (06/25/2021 10:13:44)

- links: [abs](https://arxiv.org/abs/2106.12052) | [pdf](https://arxiv.org/pdf/2106.12052)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Neural volume rendering became increasingly popular recently due to its success in synthesizing novel views of a scene from a sparse set of input images. So far, the geometry learned by neural volume rendering techniques was modeled using a generic density function. Furthermore, the geometry itself was extracted using an arbitrary level set of the density function leading to a noisy, often low fidelity reconstruction. The goal of this paper is to improve geometry representation and reconstruction in neural volume rendering. We achieve that by modeling the volume density as a function of the geometry. This is in contrast to previous work modeling the geometry as a function of the volume density. In more detail, we define the volume density function as Laplace's cumulative distribution function (CDF) applied to a signed distance function (SDF) representation. This simple density representation has three benefits: (i) it provides a useful inductive bias to the geometry learned in the neural volume rendering process; (ii) it facilitates a bound on the opacity approximation error, leading to an accurate sampling of the viewing ray. Accurate sampling is important to provide a precise coupling of geometry and radiance; and (iii) it allows efficient unsupervised disentanglement of shape and appearance in volume rendering. Applying this new density representation to challenging scene multiview datasets produced high quality geometry reconstructions, outperforming relevant baselines. Furthermore, switching shape and appearance between scenes is possible due to the disentanglement of the two.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Volume Rendering of Neural Implicit Surfaces<br>pdf: <a href="https://t.co/NEV0bxF37o">https://t.co/NEV0bxF37o</a><br>abs: <a href="https://t.co/dx1FUjeYPc">https://t.co/dx1FUjeYPc</a> <a href="https://t.co/7IFkzefwKs">pic.twitter.com/7IFkzefwKs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407864593616154629?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Co-advise: Cross Inductive Bias Distillation

Sucheng Ren, Zhengqi Gao, Tianyu Hua, Zihui Xue, Yonglong Tian, Shengfeng He, Hang Zhao

- retweets: 42, favorites: 32 (06/25/2021 10:13:44)

- links: [abs](https://arxiv.org/abs/2106.12378) | [pdf](https://arxiv.org/pdf/2106.12378)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers recently are adapted from the community of natural language processing as a promising substitute of convolution-based neural networks for visual learning tasks. However, its supremacy degenerates given an insufficient amount of training data (e.g., ImageNet). To make it into practical utility, we propose a novel distillation-based method to train vision transformers. Unlike previous works, where merely heavy convolution-based teachers are provided, we introduce lightweight teachers with different architectural inductive biases (e.g., convolution and involution) to co-advise the student transformer. The key is that teachers with different inductive biases attain different knowledge despite that they are trained on the same dataset, and such different knowledge compounds and boosts the student's performance during distillation. Equipped with this cross inductive bias distillation method, our vision transformers (termed as CivT) outperform all previous transformers of the same architecture on ImageNet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Co-advise: Cross Inductive Bias Distillation<br>pdf: <a href="https://t.co/PBHJ0gI5ix">https://t.co/PBHJ0gI5ix</a><br>abs: <a href="https://t.co/TJFCL8eZmu">https://t.co/TJFCL8eZmu</a><br><br>distilling from teacher networks with diverse inductive biases <a href="https://t.co/Qf2geAnuhQ">pic.twitter.com/Qf2geAnuhQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407863960670457856?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Approximate Bayesian Computation with Path Signatures

Joel Dyer, Patrick Cannon, Sebastian M Schmon

- retweets: 26, favorites: 34 (06/25/2021 10:13:44)

- links: [abs](https://arxiv.org/abs/2106.12555) | [pdf](https://arxiv.org/pdf/2106.12555)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.CO](https://arxiv.org/list/stat.CO/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Simulation models of scientific interest often lack a tractable likelihood function, precluding standard likelihood-based statistical inference. A popular likelihood-free method for inferring simulator parameters is approximate Bayesian computation, where an approximate posterior is sampled by comparing simulator output and observed data. However, effective measures of closeness between simulated and observed data are generally difficult to construct, particularly for time series data which are often high-dimensional and structurally complex. Existing approaches typically involve manually constructing summary statistics, requiring substantial domain expertise and experimentation, or rely on unrealistic assumptions such as iid data. Others are inappropriate in more complex settings like multivariate or irregularly sampled time series data. In this paper, we introduce the use of path signatures as a natural candidate feature set for constructing distances between time series data for use in approximate Bayesian computation algorithms. Our experiments show that such an approach can generate more accurate approximate Bayesian posteriors than existing techniques for time series models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Approximate Bayesian computation with time series models doesn&#39;t have to be difficult! Path signatures can help to automatically design expressive summary statistics. New preprint lead by fantastic PhD student Joel Dyer, joint work with <a href="https://twitter.com/pw_cannon?ref_src=twsrc%5Etfw">@pw_cannon</a>:<a href="https://t.co/tlzB4QY1wB">https://t.co/tlzB4QY1wB</a></p>&mdash; Sebastian Schmon (@SebastianSchmon) <a href="https://twitter.com/SebastianSchmon/status/1407992872230608896?ref_src=twsrc%5Etfw">June 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



