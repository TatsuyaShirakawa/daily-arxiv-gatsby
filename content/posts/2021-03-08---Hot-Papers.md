---
title: Hot Papers 2021-03-08
date: 2021-03-09T08:38:00.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-08"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-08"
socialImage: "/media/flying-marine.jpg"

---

# 1. Attention is Not All You Need: Pure Attention Loses Rank Doubly  Exponentially with Depth

Yihe Dong, Jean-Baptiste Cordonnier, Andreas Loukas

- retweets: 10132, favorites: 35 (03/09/2021 08:38:00)

- links: [abs](https://arxiv.org/abs/2103.03404) | [pdf](https://arxiv.org/pdf/2103.03404)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Attention-based architectures have become ubiquitous in machine learning, yet our understanding of the reasons for their effectiveness remains limited. This work proposes a new way to understand self-attention networks: we show that their output can be decomposed into a sum of smaller terms, each involving the operation of a sequence of attention heads across layers. Using this decomposition, we prove that self-attention possesses a strong inductive bias towards "token uniformity". Specifically, without skip connections or multi-layer perceptrons (MLPs), the output converges doubly exponentially to a rank-1 matrix. On the other hand, skip connections and MLPs stop the output from degeneration. Our experiments verify the identified convergence phenomena on different variants of standard transformer architectures.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">様々なTransformer論文が量産されて，もう何もかもAll You Needなのではと思われる中，まさかの&quot;Attention is not all you need&quot;論文が投下され，混沌の時代へ・・・<br>&quot;Attention is not all you need: pure attention loses rank doubly exponentially with depth&quot;<a href="https://t.co/4lByFg3EhM">https://t.co/4lByFg3EhM</a> <a href="https://t.co/cviD5DIDCd">pic.twitter.com/cviD5DIDCd</a></p>&mdash; えるエル (@ImAI_Eruel) <a href="https://twitter.com/ImAI_Eruel/status/1368766314194497539?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Measuring Mathematical Problem Solving With the MATH Dataset

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt

- retweets: 4250, favorites: 501 (03/09/2021 08:38:00)

- links: [abs](https://arxiv.org/abs/2103.03874) | [pdf](https://arxiv.org/pdf/2103.03874)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. To facilitate future research and increase accuracy on MATH, we also contribute a large auxiliary pretraining dataset which helps teach models the fundamentals of mathematics. Even though we are able to increase accuracy on MATH, our results show that accuracy remains relatively low, even with enormous Transformer models. Moreover, we find that simply increasing budgets and model parameter counts will be impractical for achieving strong mathematical reasoning if scaling trends continue. While scaling Transformers is automatically solving most other text-based tasks, scaling is not currently solving MATH. To have more traction on mathematical problem solving we will likely need new algorithmic advancements from the broader research community.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">To find the limits of Transformers, we collected 12,500 math problems. While a three-time IMO gold medalist got 90%, GPT-3 models got ~5%, with accuracy increasing slowly.<br><br>If trends continue, ML models are far from achieving mathematical reasoning.<a href="https://t.co/X7dzRlut01">https://t.co/X7dzRlut01</a> <a href="https://t.co/coKAtgo09R">pic.twitter.com/coKAtgo09R</a></p>&mdash; Dan Hendrycks (@DanHendrycks) <a href="https://twitter.com/DanHendrycks/status/1368962213621415936?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Measuring Mathematical Problem Solving With the<br>MATH Dataset<br><br>Introduces MATH, a new dataset of 12, 500 challenging competition mathematics problems.<br><br>Observed that scaling is not currently solving MATH despite being helpful for most other datasets.<a href="https://t.co/LJDVTKOtEr">https://t.co/LJDVTKOtEr</a> <a href="https://t.co/R0jbPSwYW8">pic.twitter.com/R0jbPSwYW8</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1368740844665073664?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Generating Images with Sparse Representations

Charlie Nash, Jacob Menick, Sander Dieleman, Peter W. Battaglia

- retweets: 2396, favorites: 438 (03/09/2021 08:38:01)

- links: [abs](https://arxiv.org/abs/2103.03841) | [pdf](https://arxiv.org/pdf/2103.03841)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The high dimensionality of images presents architecture and sampling-efficiency challenges for likelihood-based generative models. Previous approaches such as VQ-VAE use deep autoencoders to obtain compact representations, which are more practical as inputs for likelihood-based models. We present an alternative approach, inspired by common image compression methods like JPEG, and convert images to quantized discrete cosine transform (DCT) blocks, which are represented sparsely as a sequence of DCT channel, spatial location, and DCT coefficient triples. We propose a Transformer-based autoregressive architecture, which is trained to sequentially predict the conditional distribution of the next element in such sequences, and which scales effectively to high resolution images. On a range of image datasets, we demonstrate that our approach can generate high quality, diverse images, with sample metric scores competitive with state of the art methods. We additionally show that simple modifications to our method yield effective image colorization and super-resolution models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to release our new paper &#39;Generating Images with Sparse Representations&#39;  (<a href="https://t.co/ErJXmaOE0C">https://t.co/ErJXmaOE0C</a>, <a href="https://twitter.com/jacobmenick?ref_src=twsrc%5Etfw">@jacobmenick</a> <a href="https://twitter.com/sedielem?ref_src=twsrc%5Etfw">@sedielem</a> <a href="https://twitter.com/PeterWBattaglia?ref_src=twsrc%5Etfw">@PeterWBattaglia</a>)<br><br>Our model picks where to place content in an image, and what content to place there (see vid).<br><br>Thread for more info: <a href="https://t.co/ihtnvM8Gzj">pic.twitter.com/ihtnvM8Gzj</a></p>&mdash; Charlie Nash (@charlietcnash) <a href="https://twitter.com/charlietcnash/status/1369000539371999235?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating images with sparse representations<br><br>Proposes a Transformer-based autoregressive model inspired by DCT/JPEG, which scales effectively to high<br>resolution images. <br><br>Demonstrate that it can generate high quality, diverse images, with SotA quality.<a href="https://t.co/3zJtIPqdUD">https://t.co/3zJtIPqdUD</a> <a href="https://t.co/8t6BwESRIQ">pic.twitter.com/8t6BwESRIQ</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1368745231621922820?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating Images with Sparse Representations<br>pdf: <a href="https://t.co/ze6TkvMiYR">https://t.co/ze6TkvMiYR</a><br>abs: <a href="https://t.co/QtJg9zf80M">https://t.co/QtJg9zf80M</a> <a href="https://t.co/dZjTSFBzqP">pic.twitter.com/dZjTSFBzqP</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1368740095260557313?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Lord of the Ring(s): Side Channel Attacks on the CPU On-Chip Ring  Interconnect Are Practical

Riccardo Paccagnella, Licheng Luo, Christopher W. Fletcher

- retweets: 925, favorites: 132 (03/09/2021 08:38:01)

- links: [abs](https://arxiv.org/abs/2103.03443) | [pdf](https://arxiv.org/pdf/2103.03443)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.AR](https://arxiv.org/list/cs.AR/recent)

We introduce the first microarchitectural side channel attacks that leverage contention on the CPU ring interconnect. There are two challenges that make it uniquely difficult to exploit this channel. First, little is known about the ring interconnect's functioning and architecture. Second, information that can be learned by an attacker through ring contention is noisy by nature and has coarse spatial granularity. To address the first challenge, we perform a thorough reverse engineering of the sophisticated protocols that handle communication on the ring interconnect. With this knowledge, we build a cross-core covert channel over the ring interconnect with a capacity of over 4 Mbps from a single thread, the largest to date for a cross-core channel not relying on shared memory. To address the second challenge, we leverage the fine-grained temporal patterns of ring contention to infer a victim program's secrets. We demonstrate our attack by extracting key bits from vulnerable EdDSA and RSA implementations, as well as inferring the precise timing of keystrokes typed by a victim user.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Another day, another CPU security bug.<br>Lord of the Ring(s): Side Channel Attacks on the CPU On-Chip Ring Interconnect. Works on Intel and may work on AMD and other cpus too. <a href="https://t.co/sAoFuVuH51">https://t.co/sAoFuVuH51</a> Apply patches when released.</p>&mdash; nixCraft (@nixcraft) <a href="https://twitter.com/nixcraft/status/1368839965111578624?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Lord of the Ring(s): Side Channel Attacks on the (Intel) CPU On-Chip Ring Interconnect<br><br>„In this paper, we introduced side channel attacks on the ring interconnect. … extracting key bits from vulner-able EdDSA and RSA implementations …“<a href="https://t.co/2ejltqWFel">https://t.co/2ejltqWFel</a> <a href="https://t.co/8sB5ajU56S">pic.twitter.com/8sB5ajU56S</a></p>&mdash; Andreas Schilling (@aschilling) <a href="https://twitter.com/aschilling/status/1368965455080226820?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our work on ring interconnect side channel attacks was accepted at <a href="https://twitter.com/USENIXSecurity?ref_src=twsrc%5Etfw">@USENIXSecurity</a> 2021 (<a href="https://twitter.com/hashtag/usesec21?src=hash&amp;ref_src=twsrc%5Etfw">#usesec21</a>)! Full paper and source code are now available at: <a href="https://t.co/bLXXhWmQZG">https://t.co/bLXXhWmQZG</a></p>&mdash; Riccardo Paccagnella (@ricpacca) <a href="https://twitter.com/ricpacca/status/1368731483603693568?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Rissanen Data Analysis: Examining Dataset Characteristics via  Description Length

Ethan Perez, Douwe Kiela, Kyunghyun Cho

- retweets: 552, favorites: 95 (03/09/2021 08:38:02)

- links: [abs](https://arxiv.org/abs/2103.03872) | [pdf](https://arxiv.org/pdf/2103.03872)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We introduce a method to determine if a certain capability helps to achieve an accurate model of given data. We view labels as being generated from the inputs by a program composed of subroutines with different capabilities, and we posit that a subroutine is useful if and only if the minimal program that invokes it is shorter than the one that does not. Since minimum program length is uncomputable, we instead estimate the labels' minimum description length (MDL) as a proxy, giving us a theoretically-grounded method for analyzing dataset characteristics. We call the method Rissanen Data Analysis (RDA) after the father of MDL, and we showcase its applicability on a wide variety of settings in NLP, ranging from evaluating the utility of generating subquestions before answering a question, to analyzing the value of rationales and explanations, to investigating the importance of different parts of speech, and uncovering dataset gender bias.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">There&#39;s a lot of work on probing models, but models are reflections of the training data. Can we probe datasets for what capabilities they require? <a href="https://twitter.com/kchonyc?ref_src=twsrc%5Etfw">@kchonyc</a> <a href="https://twitter.com/douwekiela?ref_src=twsrc%5Etfw">@douwekiela</a> &amp; I introduce Rissanen Data Analysis to do just that: <a href="https://t.co/f16EJF75qm">https://t.co/f16EJF75qm</a><br>Code: <a href="https://t.co/wG0dg0VhxD">https://t.co/wG0dg0VhxD</a><br>1/N</p>&mdash; Ethan Perez (@EthanJPerez) <a href="https://twitter.com/EthanJPerez/status/1368988171258580993?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Learning High Fidelity Depths of Dressed Humans by Watching Social Media  Dance Videos

Yasamin Jafarian, Hyun Soo Park

- retweets: 388, favorites: 203 (03/09/2021 08:38:02)

- links: [abs](https://arxiv.org/abs/2103.03319) | [pdf](https://arxiv.org/pdf/2103.03319)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

A key challenge of learning the geometry of dressed humans lies in the limited availability of the ground truth data (e.g., 3D scanned models), which results in the performance degradation of 3D human reconstruction when applying to real-world imagery. We address this challenge by leveraging a new data resource: a number of social media dance videos that span diverse appearance, clothing styles, performances, and identities. Each video depicts dynamic movements of the body and clothes of a single person while lacking the 3D ground truth geometry. To utilize these videos, we present a new method to use the local transformation that warps the predicted local geometry of the person from an image to that of another image at a different time instant. This allows self-supervision as enforcing a temporal coherence over the predictions. In addition, we jointly learn the depth along with the surface normals that are highly responsive to local texture, wrinkle, and shade by maximizing their geometric consistency. Our method is end-to-end trainable, resulting in high fidelity depth estimation that predicts fine geometry faithful to the input real image. We demonstrate that our method outperforms the state-of-the-art human depth estimation and human shape recovery approaches on both real and rendered images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos<br>pdf: <a href="https://t.co/kFHjax0T99">https://t.co/kFHjax0T99</a><br>abs: <a href="https://t.co/th4oERTtrY">https://t.co/th4oERTtrY</a> <a href="https://t.co/EREdkGVY8E">pic.twitter.com/EREdkGVY8E</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1368755478256312320?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ru" dir="ltr">Наконец-то что-то полезное вышло из тиктока.<a href="https://t.co/c6rzzghKqy">https://t.co/c6rzzghKqy</a> <a href="https://t.co/IHsevFE7NY">pic.twitter.com/IHsevFE7NY</a></p>&mdash; Yuri Krupenin (@turbojedi) <a href="https://twitter.com/turbojedi/status/1368893838757138432?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Social contagion on higher-order structures

Alain Barrat, Guilherme Ferraz de Arruda, Iacopo Iacopini, Yamir Moreno

- retweets: 438, favorites: 79 (03/09/2021 08:38:02)

- links: [abs](https://arxiv.org/abs/2103.03709) | [pdf](https://arxiv.org/pdf/2103.03709)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

In this Chapter, we discuss the effects of higher-order structures on SIS-like processes of social contagion. After a brief motivational introduction where we illustrate the standard SIS process on networks and the difference between simple and complex contagions, we introduce spreading processes on higher-order structures starting from the most general formulation on hypergraphs and then moving to several mean-field and heterogeneous mean-field approaches. The results highlight the rich phenomenology brought by taking into account higher-order contagion effects: both continuous and discontinuous transitions are observed, and critical mass effects emerge. We conclude with a short discussion on the theoretical results regarding the nature of the epidemic transition and the general need for data to validate these models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Two preprints out today on dynamics of higher-order interactions:<br>1- &quot;Evolutionary games on simplicial complexes&quot; (<a href="https://t.co/oBeapACgck">https://t.co/oBeapACgck</a>)<br><br>2- &quot;Social contagion on higher-order structures&quot; (<a href="https://t.co/bmUyZ4Blfd">https://t.co/bmUyZ4Blfd</a>) <a href="https://t.co/ZilIVxZp7t">pic.twitter.com/ZilIVxZp7t</a></p>&mdash; Yamir Moreno (@cosnet_bifi) <a href="https://twitter.com/cosnet_bifi/status/1368853958106767360?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In &quot;Social contagion on higher-order structures&quot; (<a href="https://t.co/bmUyZ4Blfd">https://t.co/bmUyZ4Blfd</a>), we revise what we know about social contagion in higher-order structures. Work lead by <a href="https://twitter.com/GuiFdeArruda?ref_src=twsrc%5Etfw">@GuiFdeArruda</a> <a href="https://twitter.com/iacopoiacopini?ref_src=twsrc%5Etfw">@iacopoiacopini</a> &amp; <a href="https://twitter.com/alainbarrat?ref_src=twsrc%5Etfw">@alainbarrat</a> <a href="https://t.co/xdjadA9WOt">pic.twitter.com/xdjadA9WOt</a></p>&mdash; Yamir Moreno (@cosnet_bifi) <a href="https://twitter.com/cosnet_bifi/status/1368853967451619328?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. An Effective Loss Function for Generating 3D Models from Single 2D Image  without Rendering

Nikola Zubić, Pietro Liò

- retweets: 308, favorites: 121 (03/09/2021 08:38:02)

- links: [abs](https://arxiv.org/abs/2103.03390) | [pdf](https://arxiv.org/pdf/2103.03390)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Differentiable rendering is a very successful technique that applies to a Single-View 3D Reconstruction. Current renderers use losses based on pixels between a rendered image of some 3D reconstructed object and ground-truth images from given matched viewpoints to optimise parameters of the 3D shape.   These models require a rendering step, along with visibility handling and evaluation of the shading model. The main goal of this paper is to demonstrate that we can avoid these steps and still get reconstruction results as other state-of-the-art models that are equal or even better than existing category-specific reconstruction methods. First, we use the same CNN architecture for the prediction of a point cloud shape and pose prediction like the one used by Insafutdinov \& Dosovitskiy. Secondly, we propose the novel effective loss function that evaluates how well the projections of reconstructed 3D point clouds cover the ground truth object's silhouette. Then we use Poisson Surface Reconstruction to transform the reconstructed point cloud into a 3D mesh. Finally, we perform a GAN-based texture mapping on a particular 3D mesh and produce a textured 3D mesh from a single 2D image. We evaluate our method on different datasets (including ShapeNet, CUB-200-2011, and Pascal3D+) and achieve state-of-the-art results, outperforming all the other supervised and unsupervised methods and 3D representations, all in terms of performance, accuracy, and training time.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">An Effective Loss Function for Generating 3D Models from Single 2D Image without Rendering<br>pdf: <a href="https://t.co/faTvBgq3fP">https://t.co/faTvBgq3fP</a><br>abs: <a href="https://t.co/Hooi2VlcMp">https://t.co/Hooi2VlcMp</a> <a href="https://t.co/P4ijNIKASY">pic.twitter.com/P4ijNIKASY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1368748894759366657?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Causal Attention for Vision-Language Tasks

Xu Yang, Hanwang Zhang, Guojun Qi, Jianfei Cai

- retweets: 210, favorites: 73 (03/09/2021 08:38:02)

- links: [abs](https://arxiv.org/abs/2103.03493) | [pdf](https://arxiv.org/pdf/2103.03493)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a novel attention mechanism: Causal Attention (CATT), to remove the ever-elusive confounding effect in existing attention-based vision-language models. This effect causes harmful bias that misleads the attention module to focus on the spurious correlations in training data, damaging the model generalization. As the confounder is unobserved in general, we use the front-door adjustment to realize the causal intervention, which does not require any knowledge on the confounder. Specifically, CATT is implemented as a combination of 1) In-Sample Attention (IS-ATT) and 2) Cross-Sample Attention (CS-ATT), where the latter forcibly brings other samples into every IS-ATT, mimicking the causal intervention. CATT abides by the Q-K-V convention and hence can replace any attention module such as top-down attention and self-attention in Transformers. CATT improves various popular attention-based vision-language models by considerable margins. In particular, we show that CATT has great potential in large-scale pre-training, e.g., it can promote the lighter LXMERT~\cite{tan2019lxmert}, which uses fewer data and less computational power, comparable to the heavier UNITER~\cite{chen2020uniter}. Code is published in \url{https://github.com/yangxuntu/catt}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Causal Attention for Vision-Language Tasks<br>pdf: <a href="https://t.co/bnxcdWKahC">https://t.co/bnxcdWKahC</a><br>abs: <a href="https://t.co/qHpxPNklUT">https://t.co/qHpxPNklUT</a> <a href="https://t.co/SOGkR5ngTY">pic.twitter.com/SOGkR5ngTY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1368742114935402496?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Addressing Research Software Sustainability via Institutes

Daniel S. Katz, Jeffrey C. Carver, Neil P. Chue Hong, Sandra Gesing, Simon Hettrick, Tom Honeyman, Karthik Ram, Nicholas Weber

- retweets: 240, favorites: 28 (03/09/2021 08:38:02)

- links: [abs](https://arxiv.org/abs/2103.03690) | [pdf](https://arxiv.org/pdf/2103.03690)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

Research software is essential to modern research, but it requires ongoing human effort to sustain: to continually adapt to changes in dependencies, to fix bugs, and to add new features. Software sustainability institutes, amongst others, develop, maintain, and disseminate best practices for research software sustainability, and build community around them. These practices can both reduce the amount of effort that is needed and create an environment where the effort is appreciated and rewarded. The UK SSI is such an institute, and the US URSSI and the Australian AuSSI are planning to become institutes, and this extended abstract discusses them and the strengths and weaknesses of this approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Addressing Research Software Sustainability via Institutes<br><br>by <a href="https://twitter.com/danielskatz?ref_src=twsrc%5Etfw">@danielskatz</a> <a href="https://twitter.com/JeffCarver32?ref_src=twsrc%5Etfw">@JeffCarver32</a> <a href="https://twitter.com/npch?ref_src=twsrc%5Etfw">@npch</a> <a href="https://twitter.com/sandragesing?ref_src=twsrc%5Etfw">@sandragesing</a> <a href="https://twitter.com/sjh5000?ref_src=twsrc%5Etfw">@sjh5000</a> <a href="https://twitter.com/TomHoneyman3?ref_src=twsrc%5Etfw">@TomHoneyman3</a> <a href="https://twitter.com/_inundata?ref_src=twsrc%5Etfw">@_inundata</a> <a href="https://twitter.com/nniiicc?ref_src=twsrc%5Etfw">@nniiicc</a> <br><br>an <a href="https://twitter.com/hashtag/icse2021?src=hash&amp;ref_src=twsrc%5Etfw">#icse2021</a> <a href="https://twitter.com/hashtag/bokss2021?src=hash&amp;ref_src=twsrc%5Etfw">#bokss2021</a> workshop paper<br><br>cc <a href="https://twitter.com/SoftwareSaved?ref_src=twsrc%5Etfw">@SoftwareSaved</a> <a href="https://twitter.com/si2urssi?ref_src=twsrc%5Etfw">@si2urssi</a> <a href="https://twitter.com/hashtag/AuSSI?src=hash&amp;ref_src=twsrc%5Etfw">#AuSSI</a> <a href="https://twitter.com/ICSEconf?ref_src=twsrc%5Etfw">@ICSEconf</a><a href="https://t.co/I8O0lWAYEq">https://t.co/I8O0lWAYEq</a> <a href="https://t.co/xHiYoPRVAg">pic.twitter.com/xHiYoPRVAg</a></p>&mdash; Daniel S. Katz (@danielskatz) <a href="https://twitter.com/danielskatz/status/1368930713639260166?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. There Once Was a Really Bad Poet, It Was Automated but You Didn't Know  It

Jianyou Wang, Xiaoxuan Zhang, Yuren Zhou, Christopher Suh, Cynthia Rudin

- retweets: 156, favorites: 54 (03/09/2021 08:38:03)

- links: [abs](https://arxiv.org/abs/2103.03775) | [pdf](https://arxiv.org/pdf/2103.03775)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Limerick generation exemplifies some of the most difficult challenges faced in poetry generation, as the poems must tell a story in only five lines, with constraints on rhyme, stress, and meter. To address these challenges, we introduce LimGen, a novel and fully automated system for limerick generation that outperforms state-of-the-art neural network-based poetry models, as well as prior rule-based poetry models. LimGen consists of three important pieces: the Adaptive Multi-Templated Constraint algorithm that constrains our search to the space of realistic poems, the Multi-Templated Beam Search algorithm which searches efficiently through the space, and the probabilistic Storyline algorithm that provides coherent storylines related to a user-provided prompt word. The resulting limericks satisfy poetic constraints and have thematically coherent storylines, which are sometimes even funny (when we are lucky).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">There Once Was a Really Bad Poet, It Was Automated but You Didn’t Know It<br>pdf: <a href="https://t.co/jcDJXppKo3">https://t.co/jcDJXppKo3</a><br>abs: <a href="https://t.co/ZeEnAv21jY">https://t.co/ZeEnAv21jY</a> <a href="https://t.co/lTbsiRpEKj">pic.twitter.com/lTbsiRpEKj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1368753107463766016?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Compositional Explanations for Image Classifiers

Hana Chockler, Daniel Kroening, Youcheng Sun

- retweets: 72, favorites: 21 (03/09/2021 08:38:03)

- links: [abs](https://arxiv.org/abs/2103.03622) | [pdf](https://arxiv.org/pdf/2103.03622)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Existing algorithms for explaining the output of image classifiers perform poorly on inputs where the object of interest is partially occluded. We present a novel, black-box algorithm for computing explanations that uses a principled approach based on causal theory. We implement the method in the tool CET (Compositional Explanation Tool). Owing to the compositionality in its algorithm, CET computes explanations that are much more accurate than those generated by the existing explanation tools on images with occlusions and delivers a level of performance comparable to the state of the art when explaining images without occlusions.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Compositional Explanations for Image Classifiers<br>pdf: <a href="https://t.co/dmmlckvsv1">https://t.co/dmmlckvsv1</a><br>abs: <a href="https://t.co/cdX3tqr61Z">https://t.co/cdX3tqr61Z</a><br>project page: <a href="https://t.co/qgXG67DAXg">https://t.co/qgXG67DAXg</a> <a href="https://t.co/ficAJ5Bo1X">pic.twitter.com/ficAJ5Bo1X</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1368763873466187785?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. IOT: Instance-wise Layer Reordering for Transformer Structures

Jinhua Zhu, Lijun Wu, Yingce Xia, Shufang Xie, Tao Qin, Wengang Zhou, Houqiang Li, Tie-Yan Liu

- retweets: 28, favorites: 42 (03/09/2021 08:38:03)

- links: [abs](https://arxiv.org/abs/2103.03457) | [pdf](https://arxiv.org/pdf/2103.03457)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

With sequentially stacked self-attention, (optional) encoder-decoder attention, and feed-forward layers, Transformer achieves big success in natural language processing (NLP), and many variants have been proposed. Currently, almost all these models assume that the layer order is fixed and kept the same across data samples. We observe that different data samples actually favor different orders of the layers. Based on this observation, in this work, we break the assumption of the fixed layer order in the Transformer and introduce instance-wise layer reordering into the model structure. Our Instance-wise Ordered Transformer (IOT) can model variant functions by reordered layers, which enables each sample to select the better one to improve the model performance under the constraint of almost the same number of parameters. To achieve this, we introduce a light predictor with negligible parameter and inference cost to decide the most capable and favorable layer order for any input sequence. Experiments on 3 tasks (neural machine translation, abstractive summarization, and code generation) and 9 datasets demonstrate consistent improvements of our method. We further show that our method can also be applied to other architectures beyond Transformer. Our code is released at Github.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">IOT: Instance-wise Layer Reordering for Transformer Structures<br>pdf: <a href="https://t.co/ND2idxVDUC">https://t.co/ND2idxVDUC</a><br>abs: <a href="https://t.co/F6DEXrTrl1">https://t.co/F6DEXrTrl1</a><br>github: <a href="https://t.co/QUShQ8nmmX">https://t.co/QUShQ8nmmX</a> <a href="https://t.co/OYp72REwZO">pic.twitter.com/OYp72REwZO</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1368741698822680583?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Golem: An algorithm for robust experiment and process optimization

Matteo Aldeghi, Florian Häse, Riley J. Hickman, Isaac Tamblyn, Alán Aspuru-Guzik

- retweets: 22, favorites: 34 (03/09/2021 08:38:03)

- links: [abs](https://arxiv.org/abs/2103.03716) | [pdf](https://arxiv.org/pdf/2103.03716)
- [math.OC](https://arxiv.org/list/math.OC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent)

Numerous challenges in science and engineering can be framed as optimization tasks, including the maximization of reaction yields, the optimization of molecular and materials properties, and the fine-tuning of automated hardware protocols. Design of experiment and optimization algorithms are often adopted to solve these tasks efficiently. Increasingly, these experiment planning strategies are coupled with automated hardware to enable autonomous experimental platforms. The vast majority of the strategies used, however, do not consider robustness against the variability of experiment and process conditions. In fact, it is generally assumed that these parameters are exact and reproducible. Yet some experiments may have considerable noise associated with some of their conditions, and process parameters optimized under precise control may be applied in the future under variable operating conditions. In either scenario, the optimal solutions found might not be robust against input variability, affecting the reproducibility of results and returning suboptimal performance in practice. Here, we introduce Golem, an algorithm that is agnostic to the choice of experiment planning strategy and that enables robust experiment and process optimization. Golem identifies optimal solutions that are robust to input uncertainty, thus ensuring the reproducible performance of optimized experimental protocols and processes. It can be used to analyze the robustness of past experiments, or to guide experiment planning algorithms toward robust solutions on the fly. We assess the performance and domain of applicability of Golem through extensive benchmark studies and demonstrate its practical relevance by optimizing an analytical chemistry protocol under the presence of significant noise in its experimental conditions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ROBUST <a href="https://twitter.com/hashtag/optimization?src=hash&amp;ref_src=twsrc%5Etfw">#optimization</a> in <a href="https://twitter.com/hashtag/chemistry?src=hash&amp;ref_src=twsrc%5Etfw">#chemistry</a> is an important requirement for e.g. scale up. Work with <a href="https://twitter.com/matteo_aldeghi?ref_src=twsrc%5Etfw">@matteo_aldeghi</a> <a href="https://twitter.com/florian_hase?ref_src=twsrc%5Etfw">@florian_hase</a> <a href="https://twitter.com/riley_hickman?ref_src=twsrc%5Etfw">@riley_hickman</a> <a href="https://twitter.com/itamblyn?ref_src=twsrc%5Etfw">@itamblyn</a> on GOLEM the new <a href="https://twitter.com/hashtag/matterlab?src=hash&amp;ref_src=twsrc%5Etfw">#matterlab</a> algorithm <a href="https://t.co/lC3PrU6qfh">https://t.co/lC3PrU6qfh</a> <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@UofT</a> <a href="https://twitter.com/VectorInst?ref_src=twsrc%5Etfw">@VectorInst</a> <a href="https://twitter.com/chemuoft?ref_src=twsrc%5Etfw">@chemuoft</a> <a href="https://twitter.com/UofTCompSci?ref_src=twsrc%5Etfw">@UofTCompSci</a> check it out!</p>&mdash; Alan Aspuru-Guzik (@A_Aspuru_Guzik) <a href="https://twitter.com/A_Aspuru_Guzik/status/1369009301470199809?ref_src=twsrc%5Etfw">March 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



