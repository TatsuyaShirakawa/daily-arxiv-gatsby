---
title: Hot Papers 2020-12-22
date: 2020-12-23T10:50:00.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-22"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-22"
socialImage: "/media/flying-marine.jpg"

---

# 1. LieTransformer: Equivariant self-attention for Lie Groups

Michael Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye Teh, Hyunjik Kim

- retweets: 1000, favorites: 155 (12/23/2020 10:50:00)

- links: [abs](https://arxiv.org/abs/2012.10885) | [pdf](https://arxiv.org/pdf/2012.10885)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Group equivariant neural networks are used as building blocks of group invariant neural networks, which have been shown to improve generalisation performance and data efficiency through principled parameter sharing. Such works have mostly focused on group equivariant convolutions, building on the result that group equivariant linear maps are necessarily convolutions. In this work, we extend the scope of the literature to non-linear neural network modules, namely self-attention, that is emerging as a prominent building block of deep learning models. We propose the LieTransformer, an architecture composed of LieSelfAttention layers that are equivariant to arbitrary Lie groups and their discrete subgroups. We demonstrate the generality of our approach by showing experimental results that are competitive to baseline methods on a wide range of tasks: shape counting on point clouds, molecular property regression and modelling particle trajectories under Hamiltonian dynamics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LieTransformer: Equivariant self-attention for Lie Groups<br>pdf: <a href="https://t.co/NTiZMzwFoO">https://t.co/NTiZMzwFoO</a><br>abs: <a href="https://t.co/Lho5i79dLy">https://t.co/Lho5i79dLy</a> <a href="https://t.co/gVZol7tiza">pic.twitter.com/gVZol7tiza</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341202937956233217?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Online Bag-of-Visual-Words Generation for Unsupervised Representation  Learning

Spyros Gidaris, Andrei Bursuc, Gilles Puy, Nikos Komodakis, Matthieu Cord, Patrick Pérez

- retweets: 399, favorites: 88 (12/23/2020 10:50:00)

- links: [abs](https://arxiv.org/abs/2012.11552) | [pdf](https://arxiv.org/pdf/2012.11552)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Learning image representations without human supervision is an important and active research field. Several recent approaches have successfully leveraged the idea of making such a representation invariant under different types of perturbations, especially via contrastive-based instance discrimination training. Although effective visual representations should indeed exhibit such invariances, there are other important characteristics, such as encoding contextual reasoning skills, for which alternative reconstruction-based approaches might be better suited.   With this in mind, we propose a teacher-student scheme to learn representations by training a convnet to reconstruct a bag-of-visual-words (BoW) representation of an image, given as input a perturbed version of that same image. Our strategy performs an online training of both the teacher network (whose role is to generate the BoW targets) and the student network (whose role is to learn representations), along with an online update of the visual-words vocabulary (used for the BoW targets). This idea effectively enables fully online BoW-guided unsupervised learning. Extensive experiments demonstrate the interest of our BoW-based strategy which surpasses previous state-of-the-art methods (including contrastive-based ones) in several applications. For instance, in downstream tasks such Pascal object detection, Pascal classification and Places205 classification, our method improves over all prior unsupervised approaches, thus establishing new state-of-the-art results that are also significantly better even than those of supervised pre-training. We provide the implementation code at https://github.com/valeoai/obow.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work spearheaded by S. Gidaris on self-supervised learning: OBoW - Online Bag-of-Visual-Words Generation for Unsupervised Representation Learning<br><br>Paper: <a href="https://t.co/l0yyeIAc16">https://t.co/l0yyeIAc16</a> <br>Code: <a href="https://t.co/Z3cYeHcSly">https://t.co/Z3cYeHcSly</a> <br>🧵👇<br>1/N <a href="https://t.co/wAIr6PT2l5">pic.twitter.com/wAIr6PT2l5</a></p>&mdash; Andrei Bursuc (@abursuc) <a href="https://twitter.com/abursuc/status/1341365348403179520?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Evaluating Agents without Rewards

Brendon Matusch, Jimmy Ba, Danijar Hafner

- retweets: 380, favorites: 78 (12/23/2020 10:50:01)

- links: [abs](https://arxiv.org/abs/2012.11538) | [pdf](https://arxiv.org/pdf/2012.11538)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Reinforcement learning has enabled agents to solve challenging tasks in unknown environments. However, manually crafting reward functions can be time consuming, expensive, and error prone to human error. Competing objectives have been proposed for agents to learn without external supervision, but it has been unclear how well they reflect task rewards or human behavior. To accelerate the development of intrinsic objectives, we retrospectively compute potential objectives on pre-collected datasets of agent behavior, rather than optimizing them online, and compare them by analyzing their correlations. We study input entropy, information gain, and empowerment across seven agents, three Atari games, and the 3D game Minecraft. We find that all three intrinsic objectives correlate more strongly with a human behavior similarity metric than with task reward. Moreover, input entropy and information gain correlate more strongly with human similarity than task reward does, suggesting the use of intrinsic objectives for designing agents that behave similarly to human players.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share Evaluating Agents without Rewards!<br><br>We compare intrinsic objectives with task reward and similarity to human players. Turns out they all correlate more w/ human than w/ reward. Two of them even correlate more w/ human than reward does.<a href="https://t.co/hctzH7cWFz">https://t.co/hctzH7cWFz</a><br>👇 <a href="https://t.co/TmiDQcRUqL">pic.twitter.com/TmiDQcRUqL</a></p>&mdash; Danijar Hafner (@danijarh) <a href="https://twitter.com/danijarh/status/1341425926861631491?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Curiosity in exploring chemical space: Intrinsic rewards for deep  molecular reinforcement learning

Luca A. Thiede, Mario Krenn, AkshatKumar Nigam, Alan Aspuru-Guzik

- retweets: 208, favorites: 127 (12/23/2020 10:50:01)

- links: [abs](https://arxiv.org/abs/2012.11293) | [pdf](https://arxiv.org/pdf/2012.11293)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent)

Computer-aided design of molecules has the potential to disrupt the field of drug and material discovery. Machine learning, and deep learning, in particular, have been topics where the field has been developing at a rapid pace. Reinforcement learning is a particularly promising approach since it allows for molecular design without prior knowledge. However, the search space is vast and efficient exploration is desirable when using reinforcement learning agents. In this study, we propose an algorithm to aid efficient exploration. The algorithm is inspired by a concept known in the literature as curiosity. We show on three benchmarks that a curious agent finds better performing molecules. This indicates an exciting new research direction for reinforcement learning agents that can explore the chemical space out of their own motivation. This has the potential to eventually lead to unexpected new molecules that no human has thought about so far.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Human&#39;s curiosity drives science.<br><br>We find that artificial curiosity (pioneered by <a href="https://twitter.com/SchmidhuberAI?ref_src=twsrc%5Etfw">@SchmidhuberAI</a>, <a href="https://twitter.com/pathak2206?ref_src=twsrc%5Etfw">@pathak2206</a>) helps deep <a href="https://twitter.com/hashtag/ReinforcementLearning?src=hash&amp;ref_src=twsrc%5Etfw">#ReinforcementLearning</a> agents to efficiently explore the chemical universe: <a href="https://t.co/3997zzLINe">https://t.co/3997zzLINe</a><br><br>Spearheaded by Luca Thiede w/ <a href="https://twitter.com/akshat_ai?ref_src=twsrc%5Etfw">@akshat_ai</a> <a href="https://twitter.com/A_Aspuru_Guzik?ref_src=twsrc%5Etfw">@A_Aspuru_Guzik</a> <a href="https://t.co/z70PUcflW0">pic.twitter.com/z70PUcflW0</a></p>&mdash; Mario Krenn (@MarioKrenn6240) <a href="https://twitter.com/MarioKrenn6240/status/1341201038855028738?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Offline Reinforcement Learning from Images with Latent Space Models

Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, Chelsea Finn

- retweets: 166, favorites: 82 (12/23/2020 10:50:01)

- links: [abs](https://arxiv.org/abs/2012.11547) | [pdf](https://arxiv.org/pdf/2012.11547)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Offline reinforcement learning (RL) refers to the problem of learning policies from a static dataset of environment interactions. Offline RL enables extensive use and re-use of historical datasets, while also alleviating safety concerns associated with online exploration, thereby expanding the real-world applicability of RL. Most prior work in offline RL has focused on tasks with compact state representations. However, the ability to learn directly from rich observation spaces like images is critical for real-world applications such as robotics. In this work, we build on recent advances in model-based algorithms for offline RL, and extend them to high-dimensional visual observation spaces. Model-based offline RL algorithms have achieved state of the art results in state based tasks and have strong theoretical guarantees. However, they rely crucially on the ability to quantify uncertainty in the model predictions, which is particularly challenging with image observations. To overcome this challenge, we propose to learn a latent-state dynamics model, and represent the uncertainty in the latent space. Our approach is both tractable in practice and corresponds to maximizing a lower bound of the ELBO in the unknown POMDP. In experiments on a range of challenging image-based locomotion and manipulation tasks, we find that our algorithm significantly outperforms previous offline model-free RL methods as well as state-of-the-art online visual model-based RL methods. Moreover, we also find that our approach excels on an image-based drawer closing task on a real robot using a pre-existing dataset. All results including videos can be found online at https://sites.google.com/view/lompo/ .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Interested in offline RL? Handling image observations &amp; continuous actions is important for offline RL in the real world.<br><br>We introduce LOMPO to tackle this setting.<a href="https://t.co/8xzfNaipRQ">https://t.co/8xzfNaipRQ</a><br><br>with Rafael Rafailov, <a href="https://twitter.com/TianheYu?ref_src=twsrc%5Etfw">@TianheYu</a>, <a href="https://twitter.com/aravindr93?ref_src=twsrc%5Etfw">@aravindr93</a><br><br>🧵👇(1/4) <a href="https://t.co/ucfAxaGHXX">pic.twitter.com/ucfAxaGHXX</a></p>&mdash; Chelsea Finn (@chelseabfinn) <a href="https://twitter.com/chelseabfinn/status/1341514843765936128?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Hardware and Software Optimizations for Accelerating Deep Neural  Networks: Survey of Current Trends, Challenges, and the Road Ahead

Maurizio Capra, Beatrice Bussolino, Alberto Marchisio, Guido Masera, Maurizio Martina, Muhammad Shafique

- retweets: 156, favorites: 79 (12/23/2020 10:50:01)

- links: [abs](https://arxiv.org/abs/2012.11233) | [pdf](https://arxiv.org/pdf/2012.11233)
- [cs.AR](https://arxiv.org/list/cs.AR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Currently, Machine Learning (ML) is becoming ubiquitous in everyday life. Deep Learning (DL) is already present in many applications ranging from computer vision for medicine to autonomous driving of modern cars as well as other sectors in security, healthcare, and finance. However, to achieve impressive performance, these algorithms employ very deep networks, requiring a significant computational power, both during the training and inference time. A single inference of a DL model may require billions of multiply-and-accumulated operations, making the DL extremely compute- and energy-hungry. In a scenario where several sophisticated algorithms need to be executed with limited energy and low latency, the need for cost-effective hardware platforms capable of implementing energy-efficient DL execution arises. This paper first introduces the key properties of two brain-inspired models like Deep Neural Network (DNN), and Spiking Neural Network (SNN), and then analyzes techniques to produce efficient and high-performance designs. This work summarizes and compares the works for four leading platforms for the execution of algorithms such as CPU, GPU, FPGA and ASIC describing the main solutions of the state-of-the-art, giving much prominence to the last two solutions since they offer greater design flexibility and bear the potential of high energy-efficiency, especially for the inference process. In addition to hardware solutions, this paper discusses some of the important security issues that these DNN and SNN models may have during their execution, and offers a comprehensive section on benchmarking, explaining how to assess the quality of different networks and hardware systems designed for them.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Designing efficient and high-performing deep neural networks requires good understanding of their properties and how to optimize and execute them on hardware platforms like GPU and FPGA. Here is a recent report on the subject of accelerating DNNs.<a href="https://t.co/hwq4oQwmIu">https://t.co/hwq4oQwmIu</a> <a href="https://t.co/WghEUIXnHC">pic.twitter.com/WghEUIXnHC</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1341345085351473153?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Three Ways to Improve Semantic Segmentation with Self-Supervised Depth  Estimation

Lukas Hoyer, Dengxin Dai, Yuhua Chen, Adrian Köring, Suman Saha, Luc Van Gool

- retweets: 83, favorites: 81 (12/23/2020 10:50:01)

- links: [abs](https://arxiv.org/abs/2012.10782) | [pdf](https://arxiv.org/pdf/2012.10782)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Training deep networks for semantic segmentation requires large amounts of labeled training data, which presents a major challenge in practice, as labeling segmentation masks is a highly labor-intensive process. To address this issue, we present a framework for semi-supervised semantic segmentation, which is enhanced by self-supervised monocular depth estimation from unlabeled images. In particular, we propose three key contributions: (1) We transfer knowledge from features learned during self-supervised depth estimation to semantic segmentation, (2) we implement a strong data augmentation by blending images and labels using the structure of the scene, and (3) we utilize the depth feature diversity as well as the level of difficulty of learning depth in a student-teacher framework to select the most useful samples to be annotated for semantic segmentation. We validate the proposed model on the Cityscapes dataset, where all three modules demonstrate significant performance gains, and we achieve state-of-the-art results for semi-supervised semantic segmentation. The implementation is available at https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation<br>pdf: <a href="https://t.co/ggOGTeYaQm">https://t.co/ggOGTeYaQm</a><br>abs: <a href="https://t.co/mgQ1GVl5US">https://t.co/mgQ1GVl5US</a><br>github: <a href="https://t.co/R0I0Ooagj0">https://t.co/R0I0Ooagj0</a> <a href="https://t.co/dl9FajvTmw">pic.twitter.com/dl9FajvTmw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341225878563467264?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Populating 3D Scenes by Learning Human-Scene Interaction

Mohamed Hassan, Partha Ghosh, Joachim Tesch, Dimitrios Tzionas, Michael J. Black

- retweets: 54, favorites: 71 (12/23/2020 10:50:01)

- links: [abs](https://arxiv.org/abs/2012.11581) | [pdf](https://arxiv.org/pdf/2012.11581)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Humans live within a 3D space and constantly interact with it to perform tasks. Such interactions involve physical contact between surfaces that is semantically meaningful. Our goal is to learn how humans interact with scenes and leverage this to enable virtual characters to do the same. To that end, we introduce a novel Human-Scene Interaction (HSI) model that encodes proximal relationships, called POSA for "Pose with prOximitieS and contActs". The representation of interaction is body-centric, which enables it to generalize to new scenes. Specifically, POSA augments the SMPL-X parametric human body model such that, for every mesh vertex, it encodes (a) the contact probability with the scene surface and (b) the corresponding semantic scene label. We learn POSA with a VAE conditioned on the SMPL-X vertices, and train on the PROX dataset, which contains SMPL-X meshes of people interacting with 3D scenes, and the corresponding scene semantics from the PROX-E dataset. We demonstrate the value of POSA with two applications. First, we automatically place 3D scans of people in scenes. We use a SMPL-X model fit to the scan as a proxy and then find its most likely placement in 3D. POSA provides an effective representation to search for "affordances" in the scene that match the likely contact relationships for that pose. We perform a perceptual study that shows significant improvement over the state of the art on this task. Second, we show that POSA's learned representation of body-scene interaction supports monocular human pose estimation that is consistent with a 3D scene, improving on the state of the art. Our model and code will be available for research purposes at https://posa.is.tue.mpg.de.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Populating 3D Scenes by Learning Human-Scene Interaction<br>pdf: <a href="https://t.co/7xrxtckTEE">https://t.co/7xrxtckTEE</a><br>abs: <a href="https://t.co/Q3MsrR9wKm">https://t.co/Q3MsrR9wKm</a><br>project page: <a href="https://t.co/tQrNBrQib4">https://t.co/tQrNBrQib4</a> <a href="https://t.co/Nc7JxV70a3">pic.twitter.com/Nc7JxV70a3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341247945086291968?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Sub-Linear Memory: How to Make Performers SLiM

Valerii Likhosherstov, Krzysztof Choromanski, Jared Davis, Xingyou Song, Adrian Weller

- retweets: 35, favorites: 59 (12/23/2020 10:50:01)

- links: [abs](https://arxiv.org/abs/2012.11346) | [pdf](https://arxiv.org/pdf/2012.11346)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

The Transformer architecture has revolutionized deep learning on sequential data, becoming ubiquitous in state-of-the-art solutions for a wide variety of applications. Yet vanilla Transformers are notoriously resource-expensive, requiring $O(L^2)$ in serial time and memory as functions of input length $L$. Recent works proposed various linear self-attention mechanisms, scaling only as $O(L)$ for serial computation. We perform a thorough analysis of recent Transformer mechanisms with linear self-attention, Performers, in terms of overall computational complexity. We observe a remarkable computational flexibility: forward and backward propagation can be performed with no approximations using sublinear memory as a function of $L$ (in addition to negligible storage for the input sequence), at a cost of greater time complexity in the parallel setting. In the extreme case, a Performer consumes only $O(1)$ memory during training, and still requires $O(L)$ time. This discovered time-memory tradeoff can be used for training or, due to complete backward-compatibility, for fine-tuning on a low-memory device, e.g. a smartphone or an earlier-generation GPU, thus contributing towards decentralized and democratized deep learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sub-Linear Memory: How to Make Performers SLiM<br>pdf: <a href="https://t.co/2SXaIogkTU">https://t.co/2SXaIogkTU</a><br>abs: <a href="https://t.co/iiygjBP5jh">https://t.co/iiygjBP5jh</a><br>github: <a href="https://t.co/TaYv26nL1M">https://t.co/TaYv26nL1M</a> <a href="https://t.co/88WYDrDW2r">pic.twitter.com/88WYDrDW2r</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341207622800842753?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. No Shadow Left Behind: Removing Objects and their Shadows using  Approximate Lighting and Geometry

Edward Zhang, Ricardo Martin-Brualla, Janne Kontkanen, Brian Curless

- retweets: 56, favorites: 34 (12/23/2020 10:50:02)

- links: [abs](https://arxiv.org/abs/2012.10565) | [pdf](https://arxiv.org/pdf/2012.10565)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Removing objects from images is a challenging problem that is important for many applications, including mixed reality. For believable results, the shadows that the object casts should also be removed. Current inpainting-based methods only remove the object itself, leaving shadows behind, or at best require specifying shadow regions to inpaint. We introduce a deep learning pipeline for removing a shadow along with its caster. We leverage rough scene models in order to remove a wide variety of shadows (hard or soft, dark or subtle, large or thin) from surfaces with a wide variety of textures. We train our pipeline on synthetically rendered data, and show qualitative and quantitative results on both synthetic and real scenes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">No Shadow Left Behind: Removing Objects and their Shadows using Approximate Lighting and Geometry<br>pdf: <a href="https://t.co/AbVhMMfZXb">https://t.co/AbVhMMfZXb</a><br>abs: <a href="https://t.co/Hr4h2sFmJ0">https://t.co/Hr4h2sFmJ0</a> <a href="https://t.co/c2RTbx3HiW">pic.twitter.com/c2RTbx3HiW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341213523280588801?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Self-Supervised Learning for Visual Summary Identification in Scientific  Publications

Shintaro Yamamoto, Anne Lauscher, Simone Paolo Ponzetto, Goran Glavaš, Shigeo Morishima

- retweets: 42, favorites: 43 (12/23/2020 10:50:02)

- links: [abs](https://arxiv.org/abs/2012.11213) | [pdf](https://arxiv.org/pdf/2012.11213)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Providing visual summaries of scientific publications can increase information access for readers and thereby help deal with the exponential growth in the number of scientific publications. Nonetheless, efforts in providing visual publication summaries have been few and fart apart, primarily focusing on the biomedical domain. This is primarily because of the limited availability of annotated gold standards, which hampers the application of robust and high-performing supervised learning techniques. To address these problems we create a new benchmark dataset for selecting figures to serve as visual summaries of publications based on their abstracts, covering several domains in computer science. Moreover, we develop a self-supervised learning approach, based on heuristic matching of inline references to figures with figure captions. Experiments in both biomedical and computer science domains show that our model is able to outperform the state of the art despite being self-supervised and therefore not relying on any annotated training data.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">論文中から研究の概要図を探すというタスクに関する研究をarXivに公開しました<a href="https://t.co/KFpT50pbAU">https://t.co/KFpT50pbAU</a></p>&mdash; Shintaro Yamamoto (@yshin55) <a href="https://twitter.com/yshin55/status/1341212184240504832?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Building LEGO Using Deep Generative Models of Graphs

Rylee Thompson, Elahe Ghalebi, Terrance DeVries, Graham W. Taylor

- retweets: 42, favorites: 37 (12/23/2020 10:50:02)

- links: [abs](https://arxiv.org/abs/2012.11543) | [pdf](https://arxiv.org/pdf/2012.11543)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Generative models are now used to create a variety of high-quality digital artifacts. Yet their use in designing physical objects has received far less attention. In this paper, we advocate for the construction toy, LEGO, as a platform for developing generative models of sequential assembly. We develop a generative model based on graph-structured neural networks that can learn from human-built structures and produce visually compelling designs. Our code is released at: https://github.com/uoguelph-mlrg/GenerativeLEGO.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Building LEGO Using Deep Generative Models of Graphs<br>pdf: <a href="https://t.co/qa8B2lREHE">https://t.co/qa8B2lREHE</a><br>abs: <a href="https://t.co/ej4LVucQsS">https://t.co/ej4LVucQsS</a><br>github: <a href="https://t.co/83NWaWxLmH">https://t.co/83NWaWxLmH</a> <a href="https://t.co/h2osyz2VST">pic.twitter.com/h2osyz2VST</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341230933731651584?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain  Knowledge-Based VQA

Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, Marcus Rohrbach

- retweets: 42, favorites: 31 (12/23/2020 10:50:02)

- links: [abs](https://arxiv.org/abs/2012.11014) | [pdf](https://arxiv.org/pdf/2012.11014)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

One of the most challenging question types in VQA is when answering the question requires outside knowledge not present in the image. In this work we study open-domain knowledge, the setting when the knowledge required to answer a question is not given/annotated, neither at training nor test time. We tap into two types of knowledge representations and reasoning. First, implicit knowledge which can be learned effectively from unsupervised language pre-training and supervised training data with transformer-based models. Second, explicit, symbolic knowledge encoded in knowledge bases. Our approach combines both - exploiting the powerful implicit reasoning of transformer models for answer prediction, and integrating symbolic representations from a knowledge graph, while never losing their explicit semantics to an implicit embedding. We combine diverse sources of knowledge to cover the wide variety of knowledge needed to solve knowledge-based questions. We show our approach, KRISP (Knowledge Reasoning with Implicit and Symbolic rePresentations), significantly outperforms state-of-the-art on OK-VQA, the largest available dataset for open-domain knowledge-based VQA. We show with extensive ablations that while our model successfully exploits implicit knowledge reasoning, the symbolic answer module which explicitly connects the knowledge graph to the answer vocabulary is critical to the performance of our method and generalizes to rare answers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain Knowledge-Based VQA<br>pdf: <a href="https://t.co/zmalOt7PuQ">https://t.co/zmalOt7PuQ</a><br>abs: <a href="https://t.co/zvv5aa1vvA">https://t.co/zvv5aa1vvA</a> <a href="https://t.co/DJqsNXHsmh">pic.twitter.com/DJqsNXHsmh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341206425364819974?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Data Security for Machine Learning: Data Poisoning, Backdoor Attacks,  and Defenses

Micah Goldblum, Dimitris Tsipras, Chulin Xie, Xinyun Chen, Avi Schwarzschild, Dawn Song, Aleksander Madry, Bo Li, Tom Goldstein

- retweets: 36, favorites: 30 (12/23/2020 10:50:02)

- links: [abs](https://arxiv.org/abs/2012.10544) | [pdf](https://arxiv.org/pdf/2012.10544)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

As machine learning systems consume more and more data, practitioners are increasingly forced to automate and outsource the curation of training data in order to meet their data demands. This absence of human supervision over the data collection process exposes organizations to security vulnerabilities: malicious agents can insert poisoned examples into the training set to exploit the machine learning systems that are trained on it. Motivated by the emergence of this paradigm, there has been a surge in work on data poisoning including a variety of threat models as well as attack and defense methods. The goal of this work is to systematically categorize and discuss a wide range of data poisoning and backdoor attacks, approaches to defending against these threats, and an array of open problems in this space. In addition to describing these methods and the relationships among them in detail, we develop their unified taxonomy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Data Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defenses,&quot; Goldblum et al.: <a href="https://t.co/Fn1kN0vopd">https://t.co/Fn1kN0vopd</a></p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1341210832261193729?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. High-Fidelity Neural Human Motion Transfer from Monocular Video

Moritz Kappel, Vladislav Golyanik, Mohamed Elgharib, Jann-Ole Henningson, Hans-Peter Seidel, Susana Castillo, Christian Theobalt, Marcus Magnor

- retweets: 20, favorites: 45 (12/23/2020 10:50:02)

- links: [abs](https://arxiv.org/abs/2012.10974) | [pdf](https://arxiv.org/pdf/2012.10974)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Video-based human motion transfer creates video animations of humans following a source motion. Current methods show remarkable results for tightly-clad subjects. However, the lack of temporally consistent handling of plausible clothing dynamics, including fine and high-frequency details, significantly limits the attainable visual quality. We address these limitations for the first time in the literature and present a new framework which performs high-fidelity and temporally-consistent human motion transfer with natural pose-dependent non-rigid deformations, for several types of loose garments. In contrast to the previous techniques, we perform image generation in three subsequent stages, synthesizing human shape, structure, and appearance. Given a monocular RGB video of an actor, we train a stack of recurrent deep neural networks that generate these intermediate representations from 2D poses and their temporal derivatives. Splitting the difficult motion transfer problem into subtasks that are aware of the temporal motion context helps us to synthesize results with plausible dynamics and pose-dependent detail. It also allows artistic control of results by manipulation of individual framework stages. In the experimental results, we significantly outperform the state-of-the-art in terms of video realism. Our code and data will be made publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High-Fidelity Neural Human Motion Transfer from Monocular Video<br>pdf: <a href="https://t.co/fSkcqZPJEf">https://t.co/fSkcqZPJEf</a><br>abs: <a href="https://t.co/LNQrjqdu1Z">https://t.co/LNQrjqdu1Z</a><br>project page: <a href="https://t.co/r0qhrsUlCm">https://t.co/r0qhrsUlCm</a> <a href="https://t.co/h9zrivpBTC">pic.twitter.com/h9zrivpBTC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341215651986325505?ref_src=twsrc%5Etfw">December 22, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Domain specific BERT representation for Named Entity Recognition of lab  protocol

Tejas Vaidhya, Ayush Kaushal

- retweets: 30, favorites: 23 (12/23/2020 10:50:02)

- links: [abs](https://arxiv.org/abs/2012.11145) | [pdf](https://arxiv.org/pdf/2012.11145)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Supervised models trained to predict properties from representations have been achieving high accuracy on a variety of tasks. For instance, the BERT family seems to work exceptionally well on the downstream task from NER tagging to the range of other linguistic tasks. But the vocabulary used in the medical field contains a lot of different tokens used only in the medical industry such as the name of different diseases, devices, organisms, medicines, etc. that makes it difficult for traditional BERT model to create contextualized embedding. In this paper, we are going to illustrate the System for Named Entity Tagging based on Bio-Bert. Experimental results show that our model gives substantial improvements over the baseline and stood the fourth runner up in terms of F1 score, and first runner up in terms of Recall with just 2.21 F1 score behind the best one.



