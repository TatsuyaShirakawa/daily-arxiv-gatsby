---
title: Hot Papers 2021-06-28
date: 2021-06-29T10:22:50.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-28"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-28"
socialImage: "/media/flying-marine.jpg"

---

# 1. Brax -- A Differentiable Physics Engine for Large Scale Rigid Body  Simulation

C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, Olivier Bachem

- retweets: 1054, favorites: 182 (06/29/2021 10:22:50)

- links: [abs](https://arxiv.org/abs/2106.13281) | [pdf](https://arxiv.org/pdf/2106.13281)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We present Brax, an open source library for rigid body simulation with a focus on performance and parallelism on accelerators, written in JAX. We present results on a suite of tasks inspired by the existing reinforcement learning literature, but remade in our engine. Additionally, we provide reimplementations of PPO, SAC, ES, and direct policy optimization in JAX that compile alongside our environments, allowing the learning algorithm and the environment processing to occur on the same device, and to scale seamlessly on accelerators. Finally, we include notebooks that facilitate training of performant policies on common OpenAI Gym MuJoCo-like tasks in minutes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation<br>pdf: <a href="https://t.co/QkVCDxi3ch">https://t.co/QkVCDxi3ch</a><br>github: <a href="https://t.co/kfteGt8VO4">https://t.co/kfteGt8VO4</a><br><br>an open source library for rigid body simulation with a focus on performance and parallelism on accelerators, written in JAX <a href="https://t.co/6s9F6Z8gig">pic.twitter.com/6s9F6Z8gig</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1409358691342733314?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. The Price of Tolerance in Distribution Testing

Clément L. Canonne, Ayush Jain, Gautam Kamath, Jerry Li

- retweets: 162, favorites: 89 (06/29/2021 10:22:50)

- links: [abs](https://arxiv.org/abs/2106.13414) | [pdf](https://arxiv.org/pdf/2106.13414)
- [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [math.PR](https://arxiv.org/list/math.PR/recent) | [math.ST](https://arxiv.org/list/math.ST/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We revisit the problem of tolerant distribution testing. That is, given samples from an unknown distribution $p$ over $\{1, \dots, n\}$, is it $\varepsilon_1$-close to or $\varepsilon_2$-far from a reference distribution $q$ (in total variation distance)? Despite significant interest over the past decade, this problem is well understood only in the extreme cases. In the noiseless setting (i.e., $\varepsilon_1 = 0$) the sample complexity is $\Theta(\sqrt{n})$, strongly sublinear in the domain size. At the other end of the spectrum, when $\varepsilon_1 = \varepsilon_2/2$, the sample complexity jumps to the barely sublinear $\Theta(n/\log n)$. However, very little is known about the intermediate regime. We fully characterize the price of tolerance in distribution testing as a function of $n$, $\varepsilon_1$, $\varepsilon_2$, up to a single $\log n$ factor. Specifically, we show the sample complexity to be \[\tilde \Theta\left(\frac{\sqrt{n}}{\varepsilon_2^{2}} + \frac{n}{\log n} \cdot \max \left\{\frac{\varepsilon_1}{\varepsilon_2^2},\left(\frac{\varepsilon_1}{\varepsilon_2^2}\right)^{\!\!2}\right\}\right),\] providing a smooth tradeoff between the two previously known cases. We also provide a similar characterization for the problem of tolerant equivalence testing, where both $p$ and $q$ are unknown. Surprisingly, in both cases, the main quantity dictating the sample complexity is the ratio $\varepsilon_1/\varepsilon_2^2$, and not the more intuitive $\varepsilon_1/\varepsilon_2$. Of particular technical interest is our lower bound framework, which involves novel approximation-theoretic tools required to handle the asymmetry between $\varepsilon_1$ and $\varepsilon_2$, a challenge absent from previous works.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It took a little longer than hoped, but our recent work with Ayush Jain, <a href="https://twitter.com/thegautamkamath?ref_src=twsrc%5Etfw">@thegautamkamath</a>, and <a href="https://twitter.com/jerryzli?ref_src=twsrc%5Etfw">@jerryzli</a> is out on <a href="https://twitter.com/hashtag/ArXiv?src=hash&amp;ref_src=twsrc%5Etfw">#ArXiv</a>: &quot;The Price of Tolerance in Distribution Testing:&quot;<br>📝 <a href="https://t.co/mja1sBASJo">https://t.co/mja1sBASJo</a><br><br>Comments welcome! See below for a short overview 🧵<br><br>1/</p>&mdash; Clément Canonne (@ccanonne_) <a href="https://twitter.com/ccanonne_/status/1409309640102342657?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with  Language Models

Robert L. Logan IV, Ivana Balažević, Eric Wallace, Fabio Petroni, Sameer Singh, Sebastian Riedel

- retweets: 144, favorites: 67 (06/29/2021 10:22:50)

- links: [abs](https://arxiv.org/abs/2106.13353) | [pdf](https://arxiv.org/pdf/2106.13353)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Prompting language models (LMs) with training examples and task descriptions has been seen as critical to recent successes in few-shot learning. In this work, we show that finetuning LMs in the few-shot setting can considerably reduce the need for prompt engineering. In fact, one can use null prompts, prompts that contain neither task-specific templates nor training examples, and achieve competitive accuracy to manually-tuned prompts across a wide range of tasks. While finetuning LMs does introduce new parameters for each downstream task, we show that this memory overhead can be substantially reduced: finetuning only the bias terms can achieve comparable or better accuracy than standard finetuning while only updating 0.1% of the parameters. All in all, we recommend finetuning LMs for few-shot learning as it is more accurate, robust to different prompts, and can be made nearly as efficient as using frozen LMs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with Language Models<br>pdf: <a href="https://t.co/3GgcuFSvpr">https://t.co/3GgcuFSvpr</a><br>abs: <a href="https://t.co/kKDt36rbfR">https://t.co/kKDt36rbfR</a><br><br>null prompts, prompts that contain neither task specific templates nor training examples, and achieve competitive accuracy <a href="https://t.co/Zh6HUofQa4">pic.twitter.com/Zh6HUofQa4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1409314604799438850?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. To the Point: Efficient 3D Object Detection in the Range Image with  Graph Convolution Kernels

Yuning Chai, Pei Sun, Jiquan Ngiam, Weiyue Wang, Benjamin Caine, Vijay Vasudevan, Xiao Zhang, Dragomir Anguelov

- retweets: 156, favorites: 39 (06/29/2021 10:22:50)

- links: [abs](https://arxiv.org/abs/2106.13381) | [pdf](https://arxiv.org/pdf/2106.13381)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

3D object detection is vital for many robotics applications. For tasks where a 2D perspective range image exists, we propose to learn a 3D representation directly from this range image view. To this end, we designed a 2D convolutional network architecture that carries the 3D spherical coordinates of each pixel throughout the network. Its layers can consume any arbitrary convolution kernel in place of the default inner product kernel and exploit the underlying local geometry around each pixel. We outline four such kernels: a dense kernel according to the bag-of-words paradigm, and three graph kernels inspired by recent graph neural network advances: the Transformer, the PointNet, and the Edge Convolution. We also explore cross-modality fusion with the camera image, facilitated by operating in the perspective range image view. Our method performs competitively on the Waymo Open Dataset and improves the state-of-the-art AP for pedestrian detection from 69.7% to 75.5%. It is also efficient in that our smallest model, which still outperforms the popular PointPillars in quality, requires 180 times fewer FLOPS and model parameters

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">To the Point: Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels<br>pdf: <a href="https://t.co/ZB1EtxXq4g">https://t.co/ZB1EtxXq4g</a><br>abs: <a href="https://t.co/ezGE05dGIk">https://t.co/ezGE05dGIk</a><br>performs competitively on the Waymo Open Dataset and improves the sota AP for pedestrian detection from 69.7% to 75.5% <a href="https://t.co/OjJipTIkYr">pic.twitter.com/OjJipTIkYr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1409312937827880965?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. NP-DRAW: A Non-Parametric Structured Latent Variable Modelfor Image  Generation

Xiaohui Zeng, Raquel Urtasun, Richard Zemel, Sanja Fidler, Renjie Liao

- retweets: 156, favorites: 32 (06/29/2021 10:22:50)

- links: [abs](https://arxiv.org/abs/2106.13435) | [pdf](https://arxiv.org/pdf/2106.13435)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we present a non-parametric structured latent variable model for image generation, called NP-DRAW, which sequentially draws on a latent canvas in a part-by-part fashion and then decodes the image from the canvas. Our key contributions are as follows. 1) We propose a non-parametric prior distribution over the appearance of image parts so that the latent variable ``what-to-draw'' per step becomes a categorical random variable. This improves the expressiveness and greatly eases the learning compared to Gaussians used in the literature. 2) We model the sequential dependency structure of parts via a Transformer, which is more powerful and easier to train compared to RNNs used in the literature. 3) We propose an effective heuristic parsing algorithm to pre-train the prior. Experiments on MNIST, Omniglot, CIFAR-10, and CelebA show that our method significantly outperforms previous structured image models like DRAW and AIR and is competitive to other generic generative models. Moreover, we show that our model's inherent compositionality and interpretability bring significant benefits in the low-data learning regime and latent space editing. Code is available at \url{https://github.com/ZENGXH/NPDRAW}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share our UAI 2021 paper on learning discrete VAEs for iterative image generation, like human drawing! The key is a non-parametric prior over a sequence of discrete decisions (whether, where, &amp; what-to-draw).<br><br>Paper: <a href="https://t.co/ONLfPEAzFZ">https://t.co/ONLfPEAzFZ</a><br>Code: <a href="https://t.co/bUv57SABcr">https://t.co/bUv57SABcr</a> <a href="https://t.co/wV5iV23tnR">pic.twitter.com/wV5iV23tnR</a></p>&mdash; Renjie Liao (@lrjconan) <a href="https://twitter.com/lrjconan/status/1409621112917413896?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Using Machine Learning and Data Mining to Leverage Community Knowledge  for the Engineering of Stable Metal-Organic Frameworks

Aditya Nandy, Chenru Duan, Heather J. Kulik

- retweets: 72, favorites: 40 (06/29/2021 10:22:51)

- links: [abs](https://arxiv.org/abs/2106.13327) | [pdf](https://arxiv.org/pdf/2106.13327)
- [cond-mat.mtrl-sci](https://arxiv.org/list/cond-mat.mtrl-sci/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent)

Although the tailored metal active sites and porous architectures of MOFs hold great promise for engineering challenges ranging from gas separations to catalysis, a lack of understanding of how to improve their stability limits their use in practice. To overcome this limitation, we extract thousands of published reports of the key aspects of MOF stability necessary for their practical application: the ability to withstand high temperatures without degrading and the capacity to be activated by removal of solvent molecules. From nearly 4,000 manuscripts, we use natural language processing and automated image analysis to obtain over 2,000 solvent-removal stability measures and 3,000 thermal degradation temperatures. We analyze the relationships between stability properties and the chemical and geometric structures in this set to identify limits of prior heuristics derived from smaller sets of MOFs. By training predictive machine learning (ML, i.e., Gaussian process and artificial neural network) models to encode the structure-property relationships with graph- and pore-structure-based representations, we are able to make predictions of stability orders of magnitude faster than conventional physics-based modeling or experiment. Interpretation of important features in ML models provides insights that we use to identify strategies to engineer increased stability into typically unstable 3d-containing MOFs that are frequently targeted for catalytic applications. We expect our approach to accelerate the time to discovery of stable, practical MOF materials for a wide range of applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share the latest work from <a href="https://twitter.com/chenru_duan?ref_src=twsrc%5Etfw">@chenru_duan</a> (<a href="https://t.co/3X9rTnkSC8">https://t.co/3X9rTnkSC8</a>) on consensus-based molecular design and from <a href="https://twitter.com/realadityanandy?ref_src=twsrc%5Etfw">@realadityanandy</a> (<a href="https://t.co/9QrUTQFzJB">https://t.co/9QrUTQFzJB</a>) on data mining and machine learning to engineer stable MOFs at Computational Materials Chemistry <a href="https://twitter.com/TellurideSci?ref_src=twsrc%5Etfw">@TellurideSci</a>! <a href="https://t.co/v3opK9wDa6">pic.twitter.com/v3opK9wDa6</a></p>&mdash; the Kulik Group (@KulikGroup) <a href="https://twitter.com/KulikGroup/status/1409516193355112448?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Multi-Robot Deep Reinforcement Learning for Mobile Navigation

Katie Kang, Gregory Kahn, Sergey Levine

- retweets: 72, favorites: 36 (06/29/2021 10:22:51)

- links: [abs](https://arxiv.org/abs/2106.13280) | [pdf](https://arxiv.org/pdf/2106.13280)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep reinforcement learning algorithms require large and diverse datasets in order to learn successful policies for perception-based mobile navigation. However, gathering such datasets with a single robot can be prohibitively expensive. Collecting data with multiple different robotic platforms with possibly different dynamics is a more scalable approach to large-scale data collection. But how can deep reinforcement learning algorithms leverage such heterogeneous datasets? In this work, we propose a deep reinforcement learning algorithm with hierarchically integrated models (HInt). At training time, HInt learns separate perception and dynamics models, and at test time, HInt integrates the two models in a hierarchical manner and plans actions with the integrated model. This method of planning with hierarchically integrated models allows the algorithm to train on datasets gathered by a variety of different platforms, while respecting the physical capabilities of the deployment robot at test time. Our mobile navigation experiments show that HInt outperforms conventional hierarchical policies and single-source approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Robot Deep Reinforcement Learning for Mobile Navigation<br>pdf: <a href="https://t.co/b7PtjhVGQh">https://t.co/b7PtjhVGQh</a><br>abs: <a href="https://t.co/QgTlFvqa5o">https://t.co/QgTlFvqa5o</a><br>project page: <a href="https://t.co/g5o99hGeCU">https://t.co/g5o99hGeCU</a> <a href="https://t.co/X1T1S1k5nm">pic.twitter.com/X1T1S1k5nm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1409328400737751040?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Building Intelligent Autonomous Navigation Agents

Devendra Singh Chaplot

- retweets: 21, favorites: 66 (06/29/2021 10:22:52)

- links: [abs](https://arxiv.org/abs/2106.13415) | [pdf](https://arxiv.org/pdf/2106.13415)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Breakthroughs in machine learning in the last decade have led to `digital intelligence', i.e. machine learning models capable of learning from vast amounts of labeled data to perform several digital tasks such as speech recognition, face recognition, machine translation and so on. The goal of this thesis is to make progress towards designing algorithms capable of `physical intelligence', i.e. building intelligent autonomous navigation agents capable of learning to perform complex navigation tasks in the physical world involving visual perception, natural language understanding, reasoning, planning, and sequential decision making. Despite several advances in classical navigation methods in the last few decades, current navigation agents struggle at long-term semantic navigation tasks. In the first part of the thesis, we discuss our work on short-term navigation using end-to-end reinforcement learning to tackle challenges such as obstacle avoidance, semantic perception, language grounding, and reasoning. In the second part, we present a new class of navigation methods based on modular learning and structured explicit map representations, which leverage the strengths of both classical and end-to-end learning methods, to tackle long-term navigation tasks. We show that these methods are able to effectively tackle challenges such as localization, mapping, long-term planning, exploration and learning semantic priors. These modular learning methods are capable of long-term spatial and semantic understanding and achieve state-of-the-art results on various navigation tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My Ph.D. Thesis on Building Intelligent Autonomous Navigation Agents is now available at:<a href="https://t.co/pNw6E7lhfB">https://t.co/pNw6E7lhfB</a><br><br>Thesis Defense recording and slides:<a href="https://t.co/wZGSWElpuQ">https://t.co/wZGSWElpuQ</a><br><br>Thanks <a href="https://twitter.com/rsalakhu?ref_src=twsrc%5Etfw">@rsalakhu</a> for being an amazing advisor!<br><br>And I am very excited to join Facebook AI Research! <a href="https://t.co/waOrfuM2X1">pic.twitter.com/waOrfuM2X1</a></p>&mdash; Devendra Chaplot (@dchaplot) <a href="https://twitter.com/dchaplot/status/1409603508094001157?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Semantic annotation for computational pathology: Multidisciplinary  experience and best practice recommendations

Noorul Wahab, Islam M Miligy, Katherine Dodd, Harvir Sahota, Michael Toss, Wenqi Lu, Mostafa Jahanifar, Mohsin Bilal, Simon Graham, Young Park, Giorgos Hadjigeorghiou, Abhir Bhalerao, Ayat Lashen, Asmaa Ibrahim, Ayaka Katayama, Henry O Ebili, Matthew Parkin, Tom Sorell, Shan E Ahmed Raza, Emily Hero, Hesham Eldaly, Yee Wah Tsang, Kishore Gopalakrishnan, David Snead, Emad Rakha, Nasir Rajpoot, Fayyaz Minhas

- retweets: 56, favorites: 14 (06/29/2021 10:22:52)

- links: [abs](https://arxiv.org/abs/2106.13689) | [pdf](https://arxiv.org/pdf/2106.13689)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent advances in whole slide imaging (WSI) technology have led to the development of a myriad of computer vision and artificial intelligence (AI) based diagnostic, prognostic, and predictive algorithms. Computational Pathology (CPath) offers an integrated solution to utilize information embedded in pathology WSIs beyond what we obtain through visual assessment. For automated analysis of WSIs and validation of machine learning (ML) models, annotations at the slide, tissue and cellular levels are required. The annotation of important visual constructs in pathology images is an important component of CPath projects. Improper annotations can result in algorithms which are hard to interpret and can potentially produce inaccurate and inconsistent results. Despite the crucial role of annotations in CPath projects, there are no well-defined guidelines or best practices on how annotations should be carried out. In this paper, we address this shortcoming by presenting the experience and best practices acquired during the execution of a large-scale annotation exercise involving a multidisciplinary team of pathologists, ML experts and researchers as part of the Pathology image data Lake for Analytics, Knowledge and Education (PathLAKE) consortium. We present a real-world case study along with examples of different types of annotations, diagnostic algorithm, annotation data dictionary and annotation constructs. The analyses reported in this work highlight best practice recommendations that can be used as annotation guidelines over the lifecycle of a CPath project.



