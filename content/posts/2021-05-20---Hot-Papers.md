---
title: Hot Papers 2021-05-20
date: 2021-05-21T07:15:34.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-20"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-20"
socialImage: "/media/flying-marine.jpg"

---

# 1. E(n) Equivariant Normalizing Flows for Molecule Generation in 3D

Victor Garcia Satorras, Emiel Hoogeboom, Fabian B. Fuchs, Ingmar Posner, Max Welling

- retweets: 2619, favorites: 288 (05/21/2021 07:15:34)

- links: [abs](https://arxiv.org/abs/2105.09016) | [pdf](https://arxiv.org/pdf/2105.09016)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

This paper introduces a generative model equivariant to Euclidean symmetries: E(n) Equivariant Normalizing Flows (E-NFs). To construct E-NFs, we take the discriminative E(n) graph neural networks and integrate them as a differential equation to obtain an invertible equivariant function: a continuous-time normalizing flow. We demonstrate that E-NFs considerably outperform baselines and existing methods from the literature on particle systems such as DW4 and LJ13, and on molecules from QM9 in terms of log-likelihood. To the best of our knowledge, this is the first likelihood-based deep generative model that generates molecules in 3D.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very excited to share our latest work E(n) Equivariant Normalizing Flows for Molecule Generation in 3D. Joint work with <a href="https://twitter.com/emiel_hoogeboom?ref_src=twsrc%5Etfw">@emiel_hoogeboom</a> <a href="https://twitter.com/FabianFuchsML?ref_src=twsrc%5Etfw">@FabianFuchsML</a> <a href="https://twitter.com/IngmarPosner?ref_src=twsrc%5Etfw">@IngmarPosner</a> <a href="https://twitter.com/wellingmax?ref_src=twsrc%5Etfw">@wellingmax</a>.<br><br>Paper: <a href="https://t.co/glBZVfzKTu">https://t.co/glBZVfzKTu</a> <a href="https://t.co/DXSQZ44nev">pic.twitter.com/DXSQZ44nev</a></p>&mdash; Víctor Garcia Satorras (@vgsatorras) <a href="https://twitter.com/vgsatorras/status/1395277545545863169?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Compositional Processing Emerges in Neural Networks Solving Math  Problems

Jacob Russin, Roland Fernandez, Hamid Palangi, Eric Rosen, Nebojsa Jojic, Paul Smolensky, Jianfeng Gao

- retweets: 962, favorites: 157 (05/21/2021 07:15:35)

- links: [abs](https://arxiv.org/abs/2105.08961) | [pdf](https://arxiv.org/pdf/2105.08961)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

A longstanding question in cognitive science concerns the learning mechanisms underlying compositionality in human cognition. Humans can infer the structured relationships (e.g., grammatical rules) implicit in their sensory observations (e.g., auditory speech), and use this knowledge to guide the composition of simpler meanings into complex wholes. Recent progress in artificial neural networks has shown that when large models are trained on enough linguistic data, grammatical structure emerges in their representations. We extend this work to the domain of mathematical reasoning, where it is possible to formulate precise hypotheses about how meanings (e.g., the quantities corresponding to numerals) should be composed according to structured rules (e.g., order of operations). Our work shows that neural networks are not only able to infer something about the structured relationships implicit in their training data, but can also deploy this knowledge to guide the composition of individual meanings into composite wholes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Compositional Processing Emerges in Neural Networks Solving Math Problems<br>pdf: <a href="https://t.co/2GwgUxHmSz">https://t.co/2GwgUxHmSz</a><br>abs: <a href="https://t.co/aKriRDhlpI">https://t.co/aKriRDhlpI</a> <a href="https://t.co/Ufb4MrFWpA">pic.twitter.com/Ufb4MrFWpA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395184676764061700?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Recursive-NeRF: An Efficient and Dynamically Growing NeRF

Guo-Wei Yang, Wen-Yang Zhou, Hao-Yang Peng, Dun Liang, Tai-Jiang Mu, Shi-Min Hu

- retweets: 306, favorites: 114 (05/21/2021 07:15:35)

- links: [abs](https://arxiv.org/abs/2105.09103) | [pdf](https://arxiv.org/pdf/2105.09103)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

View synthesis methods using implicit continuous shape representations learned from a set of images, such as the Neural Radiance Field (NeRF) method, have gained increasing attention due to their high quality imagery and scalability to high resolution. However, the heavy computation required by its volumetric approach prevents NeRF from being useful in practice; minutes are taken to render a single image of a few megapixels. Now, an image of a scene can be rendered in a level-of-detail manner, so we posit that a complicated region of the scene should be represented by a large neural network while a small neural network is capable of encoding a simple region, enabling a balance between efficiency and quality. Recursive-NeRF is our embodiment of this idea, providing an efficient and adaptive rendering and training approach for NeRF. The core of Recursive-NeRF learns uncertainties for query coordinates, representing the quality of the predicted color and volumetric intensity at each level. Only query coordinates with high uncertainties are forwarded to the next level to a bigger neural network with a more powerful representational capability. The final rendered image is a composition of results from neural networks of all levels. Our evaluation on three public datasets shows that Recursive-NeRF is more efficient than NeRF while providing state-of-the-art quality. The code will be available at https://github.com/Gword/Recursive-NeRF.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Recursive-NeRF: An Efficient and Dynamically Growing NeRF<br>pdf: <a href="https://t.co/x7qdU1BgKV">https://t.co/x7qdU1BgKV</a><br>abs: <a href="https://t.co/eLxxBeWotE">https://t.co/eLxxBeWotE</a><br><br>learns uncertainties for query coordinates, representing the quality of the predicted color and volumetric intensity at each level <a href="https://t.co/JFqCvSK8VT">pic.twitter.com/JFqCvSK8VT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395187969078726656?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Ab-initio study of interacting fermions at finite temperature with  neural canonical transformation

Hao Xie, Linfeng Zhang, Lei Wang

- retweets: 240, favorites: 72 (05/21/2021 07:15:35)

- links: [abs](https://arxiv.org/abs/2105.08644) | [pdf](https://arxiv.org/pdf/2105.08644)
- [cond-mat.str-el](https://arxiv.org/list/cond-mat.str-el/recent) | [cond-mat.quant-gas](https://arxiv.org/list/cond-mat.quant-gas/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent)

We present a variational density matrix approach to the thermal properties of interacting fermions in the continuum. The variational density matrix is parametrized by a permutation equivariant many-body unitary transformation together with a discrete probabilistic model. The unitary transformation is implemented as a quantum counterpart of neural canonical transformation, which incorporates correlation effects via a flow of fermion coordinates. As the first application, we study electrons in a two-dimensional quantum dot with an interaction-induced crossover from Fermi liquid to Wigner molecule. The present approach provides accurate results in the low-temperature regime, where conventional quantum Monte Carlo methods face severe difficulties due to the fermion sign problem. The approach is general and flexible for further extensions, thus holds the promise to deliver new physical results on strongly correlated fermions in the context of ultracold quantum gases, condensed matter, and warm dense matter physics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FermiFlow: Ab-initio study of fermions at finite temperature <br><br>Code: <a href="https://t.co/qlryo5fzT5">https://t.co/qlryo5fzT5</a>…<br>Paper: <a href="https://t.co/1NPzfrzROK">https://t.co/1NPzfrzROK</a><br><br>Animation shows the flow of electrons in a quantum dot towards the so called &quot;Wigner molecule&quot; structure. <a href="https://t.co/ZgMhqicj65">pic.twitter.com/ZgMhqicj65</a></p>&mdash; Lei Wang (@wangleiphy) <a href="https://twitter.com/wangleiphy/status/1395304671523135489?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. High-Resolution Photorealistic Image Translation in Real-Time: A  Laplacian Pyramid Translation Network

Jie Liang, Hui Zeng, Lei Zhang

- retweets: 225, favorites: 85 (05/21/2021 07:15:35)

- links: [abs](https://arxiv.org/abs/2105.09188) | [pdf](https://arxiv.org/pdf/2105.09188)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Existing image-to-image translation (I2IT) methods are either constrained to low-resolution images or long inference time due to their heavy computational burden on the convolution of high-resolution feature maps. In this paper, we focus on speeding-up the high-resolution photorealistic I2IT tasks based on closed-form Laplacian pyramid decomposition and reconstruction. Specifically, we reveal that the attribute transformations, such as illumination and color manipulation, relate more to the low-frequency component, while the content details can be adaptively refined on high-frequency components. We consequently propose a Laplacian Pyramid Translation Network (LPTN) to simultaneously perform these two tasks, where we design a lightweight network for translating the low-frequency component with reduced resolution and a progressive masking strategy to efficiently refine the high-frequency ones. Our model avoids most of the heavy computation consumed by processing high-resolution feature maps and faithfully preserves the image details. Extensive experimental results on various tasks demonstrate that the proposed method can translate 4K images in real-time using one normal GPU while achieving comparable transformation performance against existing methods. Datasets and codes are available: https://github.com/csjliang/LPTN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network<br>pdf: <a href="https://t.co/VcTs6HbIzd">https://t.co/VcTs6HbIzd</a><br>abs: <a href="https://t.co/sbXrdixFot">https://t.co/sbXrdixFot</a><br>github: <a href="https://t.co/cc3c0wQ3up">https://t.co/cc3c0wQ3up</a> <a href="https://t.co/gcLUdbzcn4">pic.twitter.com/gcLUdbzcn4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395197097863942146?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Pathdreamer: A World Model for Indoor Navigation

Jing Yu Koh, Honglak Lee, Yinfei Yang, Jason Baldridge, Peter Anderson

- retweets: 120, favorites: 75 (05/21/2021 07:15:35)

- links: [abs](https://arxiv.org/abs/2105.08756) | [pdf](https://arxiv.org/pdf/2105.08756)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

People navigating in unfamiliar buildings take advantage of myriad visual, spatial and semantic cues to efficiently achieve their navigation goals. Towards equipping computational agents with similar capabilities, we introduce Pathdreamer, a visual world model for agents navigating in novel indoor environments. Given one or more previous visual observations, Pathdreamer generates plausible high-resolution 360 visual observations (RGB, semantic segmentation and depth) for viewpoints that have not been visited, in buildings not seen during training. In regions of high uncertainty (e.g. predicting around corners, imagining the contents of an unseen room), Pathdreamer can predict diverse scenes, allowing an agent to sample multiple realistic outcomes for a given trajectory. We demonstrate that Pathdreamer encodes useful and accessible visual, spatial and semantic knowledge about human environments by using it in the downstream task of Vision-and-Language Navigation (VLN). Specifically, we show that planning ahead with Pathdreamer brings about half the benefit of looking ahead at actual observations from unobserved parts of the environment. We hope that Pathdreamer will help unlock model-based approaches to challenging embodied navigation tasks such as navigating to specified objects and VLN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pathdreamer: A World Model for Indoor Navigation<br>pdf: <a href="https://t.co/Ia9jYWAUMN">https://t.co/Ia9jYWAUMN</a><br>abs: <a href="https://t.co/eNEBInFwPX">https://t.co/eNEBInFwPX</a> <a href="https://t.co/4WzEGcR19F">pic.twitter.com/4WzEGcR19F</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395195478208286723?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pathdreamer: A World Model for Indoor Navigation<br><br>Pathdreamer generates plausible high-res 360 visual observations for viewpoints that have not been visited, in buildings not seen during training. <br><br>abs: <a href="https://t.co/xhwvq91jTt">https://t.co/xhwvq91jTt</a><br>video: <a href="https://t.co/F18Wjq9wyN">https://t.co/F18Wjq9wyN</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1395181491584262144?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Effective Attention Sheds Light On Interpretability

Kaiser Sun, Ana Marasović

- retweets: 78, favorites: 72 (05/21/2021 07:15:35)

- links: [abs](https://arxiv.org/abs/2105.08855) | [pdf](https://arxiv.org/pdf/2105.08855)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

An attention matrix of a transformer self-attention sublayer can provably be decomposed into two components and only one of them (effective attention) contributes to the model output. This leads us to ask whether visualizing effective attention gives different conclusions than interpretation of standard attention. Using a subset of the GLUE tasks and BERT, we carry out an analysis to compare the two attention matrices, and show that their interpretations differ. Effective attention is less associated with the features related to the language modeling pretraining such as the separator token, and it has more potential to illustrate linguistic features captured by the model for solving the end-task. Given the found differences, we recommend using effective attention for studying a transformer's behavior since it is more pertinent to the model output by design.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper “Effective Attention Sheds Light On Interpretability”(w/ <a href="https://twitter.com/anmarasovic?ref_src=twsrc%5Etfw">@anmarasovic</a>) was accepted into Findings of ACL2021  <a href="https://twitter.com/hashtag/ACL2021NLP?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021NLP</a> <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a><br><br>Pre-print available at: <a href="https://t.co/730HNeNiUC">https://t.co/730HNeNiUC</a><br>Thread⬇️ <a href="https://t.co/7fThrXDCmY">pic.twitter.com/7fThrXDCmY</a></p>&mdash; Kaiser Sun (@KaiserWhoLearns) <a href="https://twitter.com/KaiserWhoLearns/status/1395184065897123840?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Effective Attention Sheds Light On Interpretability<br>pdf: <a href="https://t.co/07oPOjXsnR">https://t.co/07oPOjXsnR</a><br>abs: <a href="https://t.co/22pqtKfeFX">https://t.co/22pqtKfeFX</a><br><br>Effective attention is less associated with the features related to the language modeling pretraining such as the separator token <a href="https://t.co/aG7mvKO4ax">pic.twitter.com/aG7mvKO4ax</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395178009217355779?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Large-scale Localization Datasets in Crowded Indoor Spaces

Donghwan Lee, Soohyun Ryu, Suyong Yeon, Yonghan Lee, Deokhwa Kim, Cheolho Han, Yohann Cabon, Philippe Weinzaepfel, Nicolas Guérin, Gabriela Csurka, Martin Humenberger

- retweets: 81, favorites: 55 (05/21/2021 07:15:36)

- links: [abs](https://arxiv.org/abs/2105.08941) | [pdf](https://arxiv.org/pdf/2105.08941)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Estimating the precise location of a camera using visual localization enables interesting applications such as augmented reality or robot navigation. This is particularly useful in indoor environments where other localization technologies, such as GNSS, fail. Indoor spaces impose interesting challenges on visual localization algorithms: occlusions due to people, textureless surfaces, large viewpoint changes, low light, repetitive textures, etc. Existing indoor datasets are either comparably small or do only cover a subset of the mentioned challenges. In this paper, we introduce 5 new indoor datasets for visual localization in challenging real-world environments. They were captured in a large shopping mall and a large metro station in Seoul, South Korea, using a dedicated mapping platform consisting of 10 cameras and 2 laser scanners. In order to obtain accurate ground truth camera poses, we developed a robust LiDAR SLAM which provides initial poses that are then refined using a novel structure-from-motion based optimization. We present a benchmark of modern visual localization algorithms on these challenging datasets showing superior performance of structure-based methods using robust image features. The datasets are available at: https://naverlabs.com/datasets

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large-scale Localization Datasets in Crowded Indoor Spaces<br>pdf: <a href="https://t.co/jfWfeBeiiF">https://t.co/jfWfeBeiiF</a><br>abs: <a href="https://t.co/xNfgD1KaQ8">https://t.co/xNfgD1KaQ8</a><br><br>5 new indoor datasets for visual localization in challenging real-world environments <a href="https://t.co/wOAgLgFiId">pic.twitter.com/wOAgLgFiId</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395187158735114241?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large-scale Localization Datasets in Crowded Indoor Spaces<br>Donghwan Lee et al (incl. <a href="https://twitter.com/WeinzaepfelP?ref_src=twsrc%5Etfw">@WeinzaepfelP</a> and <a href="https://twitter.com/naverlabseurope?ref_src=twsrc%5Etfw">@naverlabseurope</a> )<br><br>tl;dr: multisensor dataset + benchmark for indoor VisLoc. <br>P.S. ESAC and PoseNet fail, structure methods rule.<a href="https://t.co/x5yKa5NGuE">https://t.co/x5yKa5NGuE</a> <a href="https://t.co/8K8rR1y8f7">pic.twitter.com/8K8rR1y8f7</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1395351705471135744?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Multi-Person Extreme Motion Prediction with Cross-Interaction Attention

Wen Guo, Xiaoyu Bie, Xavier Alameda-Pineda, Francesc Moreno

- retweets: 74, favorites: 28 (05/21/2021 07:15:36)

- links: [abs](https://arxiv.org/abs/2105.08825) | [pdf](https://arxiv.org/pdf/2105.08825)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Human motion prediction aims to forecast future human poses given a sequence of past 3D skeletons. While this problem has recently received increasing attention, it has mostly been tackled for single humans in isolation. In this paper we explore this problem from a novel perspective, involving humans performing collaborative tasks. We assume that the input of our system are two sequences of past skeletons for two interacting persons, and we aim to predict the future motion for each of them. For this purpose, we devise a novel cross interaction attention mechanism that exploits historical information of both persons and learns to predict cross dependencies between self poses and the poses of the other person in spite of their spatial or temporal distance. Since no dataset to train such interactive situations is available, we have captured ExPI (Extreme Pose Interaction), a new lab-based person interaction dataset of professional dancers performing acrobatics. ExPI contains 115 sequences with 30k frames and 60k instances with annotated 3D body poses and shapes. We thoroughly evaluate our cross-interaction network on this dataset and show that both in short-term and long-term predictions, it consistently outperforms baselines that independently reason for each person. We plan to release our code jointly with the dataset and the train/test splits to spur future research on the topic.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Interested on extreme pose estimation in interactive scenarios, check our pre-print (<a href="https://t.co/8xOpnNEJ6v">https://t.co/8xOpnNEJ6v</a>) and dataset (<a href="https://t.co/GEFtIDAleU">https://t.co/GEFtIDAleU</a>). Joint work with <a href="https://twitter.com/fmorenoguer?ref_src=twsrc%5Etfw">@fmorenoguer</a> <a href="https://twitter.com/wen80560669?ref_src=twsrc%5Etfw">@wen80560669</a> <a href="https://twitter.com/BieXiaoyu?ref_src=twsrc%5Etfw">@BieXiaoyu</a> <a href="https://t.co/CtTBc8MSuw">pic.twitter.com/CtTBc8MSuw</a></p>&mdash; Xavier Alameda-Pineda (@xavirema) <a href="https://twitter.com/xavirema/status/1395418255548985354?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Tool- and Domain-Agnostic Parameterization of Style Transfer Effects  Leveraging Pretrained Perceptual Metrics

Hiromu Yakura, Yuki Koyama, Masataka Goto

- retweets: 42, favorites: 46 (05/21/2021 07:15:36)

- links: [abs](https://arxiv.org/abs/2105.09207) | [pdf](https://arxiv.org/pdf/2105.09207)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

Current deep learning techniques for style transfer would not be optimal for design support since their "one-shot" transfer does not fit exploratory design processes. To overcome this gap, we propose parametric transcription, which transcribes an end-to-end style transfer effect into parameter values of specific transformations available in an existing content editing tool. With this approach, users can imitate the style of a reference sample in the tool that they are familiar with and thus can easily continue further exploration by manipulating the parameters. To enable this, we introduce a framework that utilizes an existing pretrained model for style transfer to calculate a perceptual style distance to the reference sample and uses black-box optimization to find the parameters that minimize this distance. Our experiments with various third-party tools, such as Instagram and Blender, show that our framework can effectively leverage deep learning techniques for computational design support.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">ACT-Xなどで取り組んでいた研究が、採択率13.9%という思っていたより激しい競争をくぐり抜けてIJCAI2021に採択されました！ 既存の訓練済みモデルをうまく使ってコンピュテーショナルデザインを行う面白い研究になっていると思います。arXivにも上げているのでよければぜひ！ <a href="https://t.co/nVkYaOW9nY">https://t.co/nVkYaOW9nY</a> <a href="https://t.co/EwK5aeF0UI">https://t.co/EwK5aeF0UI</a></p>&mdash; Hiromu Yakura (@hiromu1996) <a href="https://twitter.com/hiromu1996/status/1395213335474233347?ref_src=twsrc%5Etfw">May 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



