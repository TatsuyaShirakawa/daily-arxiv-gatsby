---
title: Hot Papers 2020-12-01
date: 2020-12-02T10:43:17.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-01"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-01"
socialImage: "/media/flying-marine.jpg"

---

# 1. One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing

Ting-Chun Wang, Arun Mallya, Ming-Yu Liu

- retweets: 9751, favorites: 0 (12/02/2020 10:43:17)

- links: [abs](https://arxiv.org/abs/2011.15126) | [pdf](https://arxiv.org/pdf/2011.15126)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a neural talking-head video synthesis model and demonstrate its application to video conferencing. Our model learns to synthesize a talking-head video using a source image containing the target person's appearance and a driving video that dictates the motion in the output. Our motion is encoded based on a novel keypoint representation, where the identity-specific and motion-related information is decomposed unsupervisedly. Extensive experimental validation shows that our model outperforms competing methods on benchmark datasets. Moreover, our compact keypoint representation enables a video conferencing system that achieves the same visual quality as the commercial H.264 standard while only using one-tenth of the bandwidth. Besides, we show our keypoint representation allows the user to rotate the head during synthesis, which is useful for simulating a face-to-face video conferencing experience.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new work on face-vid2vid, a neural talking-head model for video conferencing that is 10x more bandwidth efficient than H264<br>arxiv <a href="https://t.co/g8nLvnQwnG">https://t.co/g8nLvnQwnG</a><br>project <a href="https://t.co/u7KnaTgxTr">https://t.co/u7KnaTgxTr</a><br>video <a href="https://t.co/eJdREPdWRB">https://t.co/eJdREPdWRB</a><a href="https://twitter.com/tcwang0509?ref_src=twsrc%5Etfw">@tcwang0509</a> <a href="https://twitter.com/arunmallya?ref_src=twsrc%5Etfw">@arunmallya</a> <a href="https://twitter.com/hashtag/GAN?src=hash&amp;ref_src=twsrc%5Etfw">#GAN</a> <a href="https://t.co/kI1R2KzQBI">pic.twitter.com/kI1R2KzQBI</a></p>&mdash; Ming-Yu Liu (@liu_mingyu) <a href="https://twitter.com/liu_mingyu/status/1333610989330202625?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Unsupervised Deep Video Denoising

Dev Yashpal Sheth, Sreyas Mohan, Joshua L. Vincent, Ramon Manzorro, Peter A. Crozier, Mitesh M. Khapra, Eero P. Simoncelli, Carlos Fernandez-Granda

- retweets: 2703, favorites: 293 (12/02/2020 10:43:17)

- links: [abs](https://arxiv.org/abs/2011.15045) | [pdf](https://arxiv.org/pdf/2011.15045)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Deep convolutional neural networks (CNNs) currently achieve state-of-the-art performance in denoising videos. They are typically trained with supervision, minimizing the error between the network output and ground-truth clean videos. However, in many applications, such as microscopy, noiseless videos are not available. To address these cases, we build on recent advances in unsupervised still image denoising to develop an Unsupervised Deep Video Denoiser (UDVD). UDVD is shown to perform competitively with current state-of-the-art supervised methods on benchmark datasets, even when trained only on a single short noisy video sequence. Experiments on fluorescence-microscopy and electron-microscopy data illustrate the promise of our approach for imaging modalities where ground-truth clean data is generally not available. In addition, we study the mechanisms used by trained CNNs to perform video denoising. An analysis of the gradient of the network output with respect to its input reveals that these networks perform spatio-temporal filtering that is adapted to the particular spatial structures and motion of the underlying content. We interpret this as an implicit and highly effective form of motion compensation, a widely used paradigm in traditional video denoising, compression, and analysis. Code and iPython notebooks for our analysis are available in https://sreyas-mohan.github.io/udvd/ .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Deep Video Denoising<br>pdf: <a href="https://t.co/9CIQ5khGW8">https://t.co/9CIQ5khGW8</a><br>abs: <a href="https://t.co/j9jo2FRtGB">https://t.co/j9jo2FRtGB</a><br>project page: <a href="https://t.co/AFDLROnnxz">https://t.co/AFDLROnnxz</a> <a href="https://t.co/MlPWWKJiSQ">pic.twitter.com/MlPWWKJiSQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333633563397382147?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Feature Learning in Infinite-Width Neural Networks

Greg Yang, Edward J. Hu

- retweets: 1406, favorites: 243 (12/02/2020 10:43:17)

- links: [abs](https://arxiv.org/abs/2011.14522) | [pdf](https://arxiv.org/pdf/2011.14522)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

As its width tends to infinity, a deep neural network's behavior under gradient descent can become simplified and predictable (e.g. given by the Neural Tangent Kernel (NTK)), if it is parametrized appropriately (e.g. the NTK parametrization). However, we show that the standard and NTK parametrizations of a neural network do not admit infinite-width limits that can learn features, which is crucial for pretraining and transfer learning such as with BERT. We propose simple modifications to the standard parametrization to allow for feature learning in the limit. Using the *Tensor Programs* technique, we derive explicit formulas for such limits. On Word2Vec and few-shot learning on Omniglot via MAML, two canonical tasks that rely crucially on feature learning, we compute these limits exactly. We find that they outperform both NTK baselines and finite-width networks, with the latter approaching the infinite-width feature learning performance as width increases.   More generally, we classify a natural space of neural network parametrizations that generalizes standard, NTK, and Mean Field parametrizations. We show 1) any parametrization in this space either admits feature learning or has an infinite-width training dynamics given by kernel gradient descent, but not both; 2) any such infinite-width limit can be computed using the Tensor Programs technique.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/ Existing theories of neural networks (NN) like NTK don&#39;t learn features so can&#39;t explain success of pretraining (e.g. BERT, GPT3). We derive the *feature learning* ∞-width limit of NNs &amp; pretrained such an ∞-width word2vec model: it learned semantics!<a href="https://t.co/vhvvXylq58">https://t.co/vhvvXylq58</a> <a href="https://t.co/1VJVFjY7JZ">pic.twitter.com/1VJVFjY7JZ</a></p>&mdash; Greg Yang (@TheGregYang) <a href="https://twitter.com/TheGregYang/status/1333773565515194371?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. General Invertible Transformations for Flow-based Generative Modeling

Jakub M. Tomczak

- retweets: 841, favorites: 171 (12/02/2020 10:43:17)

- links: [abs](https://arxiv.org/abs/2011.15056) | [pdf](https://arxiv.org/pdf/2011.15056)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In this paper, we present a new class of invertible transformations. We indicate that many well-known invertible tranformations in reversible logic and reversible neural networks could be derived from our proposition. Next, we propose two new coupling layers that are important building blocks of flow-based generative models. In the preliminary experiments on toy digit data, we present how these new coupling layers could be used in Integer Discrete Flows (IDF), and that they achieve better results than standard coupling layers used in IDF and RealNVP.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My recent work on general invertible transformations for flow-based generative models with some preliminary experiments. I present how we can generalize coupling layers and reversible logic operators. Paper: <a href="https://t.co/UOqfXpQZ1w">https://t.co/UOqfXpQZ1w</a> Code: <a href="https://t.co/ufyFB5DIdl">https://t.co/ufyFB5DIdl</a> <a href="https://t.co/gWir02DqRd">pic.twitter.com/gWir02DqRd</a></p>&mdash; Jakub Tomczak (@jmtomczak) <a href="https://twitter.com/jmtomczak/status/1333742066107637762?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Scaling *down* Deep Learning

Sam Greydanus

- retweets: 813, favorites: 127 (12/02/2020 10:43:18)

- links: [abs](https://arxiv.org/abs/2011.14439) | [pdf](https://arxiv.org/pdf/2011.14439)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Though deep learning models have taken on commercial and political relevance, many aspects of their training and operation remain poorly understood. This has sparked interest in "science of deep learning" projects, many of which are run at scale and require enormous amounts of time, money, and electricity. But how much of this research really needs to occur at scale? In this paper, we introduce MNIST-1D: a minimalist, low-memory, and low-compute alternative to classic deep learning benchmarks. The training examples are 20 times smaller than MNIST examples yet they differentiate more clearly between linear, nonlinear, and convolutional models which attain 32, 68, and 94% accuracy respectively (these models obtain 94, 99+, and 99+% on MNIST). Then we present example use cases which include measuring the spatial inductive biases of lottery tickets, observing deep double descent, and metalearning an activation function.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">“Scaling *down* Deep Learning” 🧪<br><br>Blog: <a href="https://t.co/qGQZRebx6w">https://t.co/qGQZRebx6w</a><br>Paper: <a href="https://t.co/z8geypeVoD">https://t.co/z8geypeVoD</a><br><br>In order to explore the limits of how large we can scale neural networks, we may need to explore the limits of how small we can scale them first. <a href="https://t.co/dNxUNhbC6I">pic.twitter.com/dNxUNhbC6I</a></p>&mdash; Sam Greydanus (@samgreydanus) <a href="https://twitter.com/samgreydanus/status/1333887306940387329?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Simulation-efficient marginal posterior estimation with swyft: stop  wasting your precious time

Benjamin Kurt Miller, Alex Cole, Gilles Louppe, Christoph Weniger

- retweets: 714, favorites: 138 (12/02/2020 10:43:18)

- links: [abs](https://arxiv.org/abs/2011.13951) | [pdf](https://arxiv.org/pdf/2011.13951)
- [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [astro-ph.CO](https://arxiv.org/list/astro-ph.CO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [hep-ph](https://arxiv.org/list/hep-ph/recent)

We present algorithms (a) for nested neural likelihood-to-evidence ratio estimation, and (b) for simulation reuse via an inhomogeneous Poisson point process cache of parameters and corresponding simulations. Together, these algorithms enable automatic and extremely simulator efficient estimation of marginal and joint posteriors. The algorithms are applicable to a wide range of physics and astronomy problems and typically offer an order of magnitude better simulator efficiency than traditional likelihood-based sampling methods. Our approach is an example of likelihood-free inference, thus it is also applicable to simulators which do not offer a tractable likelihood function. Simulator runs are never rejected and can be automatically reused in future analysis. As functional prototype implementation we provide the open-source software package swyft.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🎁We are happy to present swyft, a Python package for neural nested marginal posterior estimation. It provides parameter inference superpowers through our new nested ratio estimation and iP3 sample cache algorithms. A thread 👇.  <a href="https://t.co/UKedVTloPz">https://t.co/UKedVTloPz</a> <a href="https://t.co/0uXXNGQUci">https://t.co/0uXXNGQUci</a> <a href="https://t.co/6FkSpo9DxB">pic.twitter.com/6FkSpo9DxB</a></p>&mdash; Christoph Weniger (@C_Weniger) <a href="https://twitter.com/C_Weniger/status/1333716740509986817?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Revisiting Rainbow: Promoting more insightful and inclusive deep  reinforcement learning research

Johan S. Obando-Ceron, Pablo Samuel Castro

- retweets: 600, favorites: 105 (12/02/2020 10:43:18)

- links: [abs](https://arxiv.org/abs/2011.14826) | [pdf](https://arxiv.org/pdf/2011.14826)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Since the introduction of DQN, a vast majority of reinforcement learning research has focused on reinforcement learning with deep neural networks as function approximators. New methods are typically evaluated on a set of environments that have now become standard, such as Atari 2600 games. While these benchmarks help standardize evaluation, their computational cost has the unfortunate side effect of widening the gap between those with ample access to computational resources, and those without. In this work we argue that, despite the community's emphasis on large-scale environments, the traditional small-scale environments can still yield valuable scientific insights and can help reduce the barriers to entry for underprivileged communities. To substantiate our claims, we empirically revisit the paper which introduced the Rainbow algorithm [Hessel et al., 2018] and present some new insights into the algorithms used by Rainbow.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share &quot;Revisiting Rainbow&quot; w/ <a href="https://twitter.com/JS_Obando?ref_src=twsrc%5Etfw">@JS_Obando</a> where we argue small/mid-scale envs can promote more insightful &amp; inclusive deep RL research.<br>📜Paper: <a href="https://t.co/d5I62kAqdc">https://t.co/d5I62kAqdc</a><br>✍️🏾Blog: <a href="https://t.co/WMVJJjPaLm">https://t.co/WMVJJjPaLm</a><br>🐍Code: <a href="https://t.co/WdtgsZFP84">https://t.co/WdtgsZFP84</a><br>📽️Video: <a href="https://t.co/VfJFQqsGds">https://t.co/VfJFQqsGds</a><br>🧵1/X <a href="https://t.co/WTqMOUxEqi">pic.twitter.com/WTqMOUxEqi</a></p>&mdash; Pablo Samuel Castro (@pcastr) <a href="https://twitter.com/pcastr/status/1333795082370125830?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. D-NeRF: Neural Radiance Fields for Dynamic Scenes

Albert Pumarola, Enric Corona, Gerard Pons-Moll, Francesc Moreno-Noguer

- retweets: 396, favorites: 144 (12/02/2020 10:43:18)

- links: [abs](https://arxiv.org/abs/2011.13961) | [pdf](https://arxiv.org/pdf/2011.13961)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Neural rendering techniques combining machine learning with geometric reasoning have arisen as one of the most promising approaches for synthesizing novel views of a scene from a sparse set of images. Among these, stands out the Neural radiance fields (NeRF), which trains a deep network to map 5D input coordinates (representing spatial location and viewing direction) into a volume density and view-dependent emitted radiance. However, despite achieving an unprecedented level of photorealism on the generated images, NeRF is only applicable to static scenes, where the same spatial location can be queried from different images. In this paper we introduce D-NeRF, a method that extends neural radiance fields to a dynamic domain, allowing to reconstruct and render novel images of objects under rigid and non-rigid motions from a \emph{single} camera moving around the scene. For this purpose we consider time as an additional input to the system, and split the learning process in two main stages: one that encodes the scene into a canonical space and another that maps this canonical representation into the deformed scene at a particular time. Both mappings are simultaneously learned using fully-connected networks. Once the networks are trained, D-NeRF can render novel images, controlling both the camera view and the time variable, and thus, the object movement. We demonstrate the effectiveness of our approach on scenes with objects under rigid, articulated and non-rigid motions. Code, model weights and the dynamic scenes dataset will be released.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">D-NeRF: Neural Radiance Fields for Dynamic Scenes<br>pdf: <a href="https://t.co/JDiX4t4ZLY">https://t.co/JDiX4t4ZLY</a><br>abs: <a href="https://t.co/T2bP06K8xZ">https://t.co/T2bP06K8xZ</a> <a href="https://t.co/I6RRZ3JBnu">pic.twitter.com/I6RRZ3JBnu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333637845131653120?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We present D-NeRF, a method for synthesizing images of dynamic scenes with time and camera view control.<br><br>Work done with <a href="https://twitter.com/enric_corona?ref_src=twsrc%5Etfw">@enric_corona</a>, <a href="https://twitter.com/GerardPonsMoll1?ref_src=twsrc%5Etfw">@GerardPonsMoll1</a> and <a href="https://twitter.com/fmorenoguer?ref_src=twsrc%5Etfw">@fmorenoguer</a> at <a href="https://twitter.com/IRI_robotics?ref_src=twsrc%5Etfw">@IRI_robotics</a>.<br><br>🖥️Project: <a href="https://t.co/lO77rqzyti">https://t.co/lO77rqzyti</a><br>📄PDF: <a href="https://t.co/k8A6DgJ3l6">https://t.co/k8A6DgJ3l6</a> <a href="https://t.co/DK2nRWG1Ic">pic.twitter.com/DK2nRWG1Ic</a></p>&mdash; Albert Pumarola (@AlbertPumarola) <a href="https://twitter.com/AlbertPumarola/status/1333819522705485826?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. UniCon: Universal Neural Controller For Physics-based Character Motion

Tingwu Wang, Yunrong Guo, Maria Shugrina, Sanja Fidler

- retweets: 240, favorites: 82 (12/02/2020 10:43:19)

- links: [abs](https://arxiv.org/abs/2011.15119) | [pdf](https://arxiv.org/pdf/2011.15119)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

The field of physics-based animation is gaining importance due to the increasing demand for realism in video games and films, and has recently seen wide adoption of data-driven techniques, such as deep reinforcement learning (RL), which learn control from (human) demonstrations. While RL has shown impressive results at reproducing individual motions and interactive locomotion, existing methods are limited in their ability to generalize to new motions and their ability to compose a complex motion sequence interactively. In this paper, we propose a physics-based universal neural controller (UniCon) that learns to master thousands of motions with different styles by learning on large-scale motion datasets. UniCon is a two-level framework that consists of a high-level motion scheduler and an RL-powered low-level motion executor, which is our key innovation. By systematically analyzing existing multi-motion RL frameworks, we introduce a novel objective function and training techniques which make a significant leap in performance. Once trained, our motion executor can be combined with different high-level schedulers without the need for retraining, enabling a variety of real-time interactive applications. We show that UniCon can support keyboard-driven control, compose motion sequences drawn from a large pool of locomotion and acrobatics skills and teleport a person captured on video to a physics-based virtual avatar. Numerical and qualitative results demonstrate a significant improvement in efficiency, robustness and generalizability of UniCon over prior state-of-the-art, showcasing transferability to unseen motions, unseen humanoid models and unseen perturbation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">UniCon: Universal Neural Controller For Physics-based Character Motion<br>pdf: <a href="https://t.co/huuNpaqNkV">https://t.co/huuNpaqNkV</a><br>abs: <a href="https://t.co/dnXOTcgjC0">https://t.co/dnXOTcgjC0</a><br>project page: <a href="https://t.co/GLDIyBhwf8">https://t.co/GLDIyBhwf8</a> <a href="https://t.co/tvbmDNQ64a">pic.twitter.com/tvbmDNQ64a</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333612757107748866?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Google Searches and COVID-19 Cases in Saudi Arabia: A Correlation Study

Btool Hamoui, Abdulaziz Alashaikh, Eisa Alanazi

- retweets: 225, favorites: 40 (12/02/2020 10:43:19)

- links: [abs](https://arxiv.org/abs/2011.14386) | [pdf](https://arxiv.org/pdf/2011.14386)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Background: The outbreak of the new coronavirus disease (COVID-19) has affected human life to a great extent on a worldwide scale. During the coronavirus pandemic, public health professionals at the early outbreak faced an extraordinary challenge to track and quantify the spread of disease. Objective: To investigate whether a digital surveillance model using google trends (GT) is feasible to monitor the outbreak of coronavirus in the Kingdom of Saudi Arabia. Methods: We retrieve GT data using ten common COVID-19 symptoms related keywords from March 2, 2020, to October 31, 2020. Spearman correlation were performed to determine the correlation between COVID-19 cases and the Google search terms. Results: GT data related to Cough and Sore Throat were the most searched symptoms by the Internet users in Saudi Arabia. The highest daily correlation found with the Loss of Smell followed by Loss of Taste and Diarrhea. Strong correlation as well was found between the weekly confirmed cases and the same symptoms: Loss of Smell, Loss of Taste and Diarrhea. Conclusions: We conducted an investigation study utilizing Internet searches related to COVID-19 symptoms for surveillance of the pandemic spread. This study documents that google searches can be used as a supplementary surveillance tool in COVID-19 monitoring in Saudi Arabia.

<blockquote class="twitter-tweet"><p lang="ar" dir="rtl">ماالعلاقة بين بحث السعوديون في قوقل و عدد حالات كورونا اليومية بالمملكة؟ <br>قمنا بدراسة Google السعودي على مدى سبعة أشهر منذ بداية الجائحة وتحليل أبرز مابُحث عنه.<br>النتائج: حاسة الشم أكثر الأعراض إرتباطاً مع عدد الحالات اليومية.<br>👇 <a href="https://t.co/vHlmeZAkkf">https://t.co/vHlmeZAkkf</a> <a href="https://t.co/moHvTv3pvr">pic.twitter.com/moHvTv3pvr</a></p>&mdash; #G20...عيسى العنزي (@eisa_ayed) <a href="https://twitter.com/eisa_ayed/status/1333700482456559616?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. DUT:Learning Video Stabilization by Simply Watching Unstable Videos

Yufei Xu, Jing Zhang, Stephen J. Mayban, Dacheng Tao

- retweets: 156, favorites: 50 (12/02/2020 10:43:19)

- links: [abs](https://arxiv.org/abs/2011.14574) | [pdf](https://arxiv.org/pdf/2011.14574)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a Deep Unsupervised Trajectory-based stabilization framework (DUT) in this paper\footnote{Our code is available at https://github.com/Annbless/DUTCode.}. Traditional stabilizers focus on trajectory-based smoothing, which is controllable but fragile in occluded and textureless cases regarding the usage of hand-crafted features. On the other hand, previous deep video stabilizers directly generate stable videos in a supervised manner without explicit trajectory estimation, which is robust but less controllable and the appropriate paired data are hard to obtain. To construct a controllable and robust stabilizer, DUT makes the first attempt to stabilize unstable videos by explicitly estimating and smoothing trajectories in an unsupervised deep learning manner, which is composed of a DNN-based keypoint detector and motion estimator to generate grid-based trajectories, and a DNN-based trajectory smoother to stabilize videos. We exploit both the nature of continuity in motion and the consistency of keypoints and grid vertices before and after stabilization for unsupervised training. Experiment results on public benchmarks show that DUT outperforms representative state-of-the-art methods both qualitatively and quantitatively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DUT: Learning Video Stabilization by Simply Watching Unstable Videos<br>pdf: <a href="https://t.co/mztFhptyKE">https://t.co/mztFhptyKE</a><br>abs: <a href="https://t.co/t0k9YaJDzo">https://t.co/t0k9YaJDzo</a><br>github: <a href="https://t.co/doukOBVMMl">https://t.co/doukOBVMMl</a> <a href="https://t.co/g8xH3CTBwU">pic.twitter.com/g8xH3CTBwU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333615275606994946?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs

Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki, Desmond Elliott

- retweets: 132, favorites: 64 (12/02/2020 10:43:19)

- links: [abs](https://arxiv.org/abs/2011.15124) | [pdf](https://arxiv.org/pdf/2011.15124)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Large-scale pretraining and task-specific fine-tuning is now the standard methodology for many tasks in computer vision and natural language processing. Recently, a multitude of methods have been proposed for pretraining vision and language BERTs to tackle challenges at the intersection of these two key areas of AI. These models can be categorized into either single-stream or dual-stream encoders. We study the differences between these two categories, and show how they can be unified under a single theoretical framework. We then conduct controlled experiments to discern the empirical differences between five V&L BERTs. Our experiments show that training data and hyperparameters are responsible for most of the differences between the reported results, but they also reveal that the embedding layer plays a crucial role in these massive models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Which vision-and-language BERT should you use for your downstream task? There are so many 🤯<a href="https://twitter.com/delliott?ref_src=twsrc%5Etfw">@delliott</a>, <a href="https://twitter.com/ryandcotterell?ref_src=twsrc%5Etfw">@ryandcotterell</a>, <a href="https://twitter.com/chokkanorg?ref_src=twsrc%5Etfw">@chokkanorg</a> and I take a deep dive into these models:<br><br>“Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs”<br><br>📄 <a href="https://t.co/WodASSBs4n">https://t.co/WodASSBs4n</a> <a href="https://t.co/bNuFzbJh4T">pic.twitter.com/bNuFzbJh4T</a></p>&mdash; Emanuele Bugliarello (@ebugliarello) <a href="https://twitter.com/ebugliarello/status/1333719046337015810?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. End-to-End Video Instance Segmentation with Transformers

Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, Huaxia Xia

- retweets: 90, favorites: 71 (12/02/2020 10:43:19)

- links: [abs](https://arxiv.org/abs/2011.14503) | [pdf](https://arxiv.org/pdf/2011.14503)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Video instance segmentation (VIS) is the task that requires simultaneously classifying, segmenting and tracking object instances of interest in video. Recent methods typically develop sophisticated pipelines to tackle this task. Here, we propose a new video instance segmentation framework built upon Transformers, termed VisTR, which views the VIS task as a direct end-to-end parallel sequence decoding/prediction problem. Given a video clip consisting of multiple image frames as input, VisTR outputs the sequence of masks for each instance in the video in order directly. At the core is a new, effective instance sequence matching and segmentation strategy, which supervises and segments instances at the sequence level as a whole. VisTR frames the instance segmentation and tracking in the same perspective of similarity learning, thus considerably simplifying the overall pipeline and is significantly different from existing approaches. Without bells and whistles, VisTR achieves the highest speed among all existing VIS models, and achieves the best result among methods using single model on the YouTube-VIS dataset. For the first time, we demonstrate a much simpler and faster video instance segmentation framework built upon Transformers, achieving competitive accuracy. We hope that VisTR can motivate future research for more video understanding tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">End-to-End Video Instance Segmentation with Transformers<br>pdf: <a href="https://t.co/yfd47m2Z5C">https://t.co/yfd47m2Z5C</a><br>abs: <a href="https://t.co/iiyjkesoyD">https://t.co/iiyjkesoyD</a> <a href="https://t.co/itcxd8FyuD">pic.twitter.com/itcxd8FyuD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333599304259407873?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Forecasting Characteristic 3D Poses of Human Actions

Christian Diller, Thomas Funkhouser, Angela Dai

- retweets: 42, favorites: 36 (12/02/2020 10:43:19)

- links: [abs](https://arxiv.org/abs/2011.15079) | [pdf](https://arxiv.org/pdf/2011.15079)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose the task of forecasting characteristic 3D poses: from a single pose observation of a person, to predict a future 3D pose of that person in a likely action-defining, characteristic pose - for instance, from observing a person picking up a banana, predict the pose of the person eating the banana. Prior work on human motion prediction estimates future poses at fixed time intervals. Although easy to define, this frame-by-frame formulation confounds temporal and intentional aspects of human action. Instead, we define a goal-directed pose prediction task that decouples pose prediction from time, taking inspiration from human, goal-directed behavior. To predict characteristic goal poses, we propose a probabilistic approach that first models the possible multi-modality in the distribution of possible characteristic poses. It then samples future pose hypotheses from the predicted distribution in an autoregressive fashion to model dependencies between joints and then optimizes the final pose with bone length and angle constraints. To evaluate our method, we construct a dataset of manually annotated single-frame observations and characteristic 3D poses. Our experiments with this dataset suggest that our proposed probabilistic approach outperforms state-of-the-art approaches by 22% on average.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Forecasting Characteristic 3D Poses of Human Actions:<br><br>We take a goal-directed approach to human pose prediction instead of frame-by-frame forecasting<br><br>Video: <a href="https://t.co/IN3XSY1Obc">https://t.co/IN3XSY1Obc</a><br>Paper: <a href="https://t.co/MeRPlRFRLD">https://t.co/MeRPlRFRLD</a><br><br>Project with Tom Funkhouser and <a href="https://twitter.com/angelaqdai?ref_src=twsrc%5Etfw">@angelaqdai</a> <a href="https://t.co/g35mkrNBHE">pic.twitter.com/g35mkrNBHE</a></p>&mdash; Christian Diller (@chrdiller) <a href="https://twitter.com/chrdiller/status/1333864883348267009?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Towards constraining warm dark matter with stellar streams through  neural simulation-based inference

Joeri Hermans, Nilanjan Banik, Christoph Weniger, Gianfranco Bertone, Gilles Louppe

- retweets: 24, favorites: 48 (12/02/2020 10:43:19)

- links: [abs](https://arxiv.org/abs/2011.14923) | [pdf](https://arxiv.org/pdf/2011.14923)
- [astro-ph.GA](https://arxiv.org/list/astro-ph.GA/recent) | [astro-ph.CO](https://arxiv.org/list/astro-ph.CO/recent) | [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

A statistical analysis of the observed perturbations in the density of stellar streams can in principle set stringent contraints on the mass function of dark matter subhaloes, which in turn can be used to constrain the mass of the dark matter particle. However, the likelihood of a stellar density with respect to the stream and subhaloes parameters involves solving an intractable inverse problem which rests on the integration of all possible forward realisations implicitly defined by the simulation model. In order to infer the subhalo abundance, previous analyses have relied on Approximate Bayesian Computation (ABC) together with domain-motivated but handcrafted summary statistics. Here, we introduce a likelihood-free Bayesian inference pipeline based on Amortised Approximate Likelihood Ratios (AALR), which automatically learns a mapping between the data and the simulator parameters and obviates the need to handcraft a possibly insufficient summary statistic. We apply the method to the simplified case where stellar streams are only perturbed by dark matter subhaloes, thus neglecting baryonic substructures, and describe several diagnostics that demonstrate the effectiveness of the new method and the statistical quality of the learned estimator.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper today w/ <a href="https://twitter.com/joeri_hermans?ref_src=twsrc%5Etfw">@joeri_hermans</a>, Nil Banik, <a href="https://twitter.com/C_Weniger?ref_src=twsrc%5Etfw">@C_Weniger</a> and <a href="https://twitter.com/glouppe?ref_src=twsrc%5Etfw">@glouppe</a>: a likelihood-free Bayesian inference pipeline to infer properties of dark matter from observations of stellar streams <a href="https://t.co/gen57yIoMc">https://t.co/gen57yIoMc</a> <a href="https://t.co/hSIHRb8Z8T">pic.twitter.com/hSIHRb8Z8T</a></p>&mdash; Gianfranco Bertone (@gfbertone) <a href="https://twitter.com/gfbertone/status/1333726886543945728?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Hybrid quantum-classical classifier based on tensor network and  variational quantum circuit

Samuel Yen-Chi Chen, Chih-Min Huang, Chia-Wei Hsing, Ying-Jer Kao

- retweets: 26, favorites: 31 (12/02/2020 10:43:20)

- links: [abs](https://arxiv.org/abs/2011.14651) | [pdf](https://arxiv.org/pdf/2011.14651)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

One key step in performing quantum machine learning (QML) on noisy intermediate-scale quantum (NISQ) devices is the dimension reduction of the input data prior to their encoding. Traditional principle component analysis (PCA) and neural networks have been used to perform this task; however, the classical and quantum layers are usually trained separately. A framework that allows for a better integration of the two key components is thus highly desirable. Here we introduce a hybrid model combining the quantum-inspired tensor networks (TN) and the variational quantum circuits (VQC) to perform supervised learning tasks, which allows for an end-to-end training. We show that a matrix product state based TN with low bond dimensions performs better than PCA as a feature extractor to compress data for the input of VQCs in the binary classification of MNIST dataset. The architecture is highly adaptable and can easily incorporate extra quantum resource when available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hybrid quantum-classical classifier based on tensor network and variational quantum circuit<br><br>We propose a TN+VQC architecture that allows for an end-to-end training <a href="https://t.co/hkLGs9yNKh">https://t.co/hkLGs9yNKh</a></p>&mdash; Ying-Jer Kao (@yjkao) <a href="https://twitter.com/yjkao/status/1333610638501789697?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Applying Convolutional Neural Networks to Data on Unstructured Meshes  with Space-Filling Curves

Claire Heaney, Yuling Li, Omar Matar, Christopher Pain

- retweets: 42, favorites: 14 (12/02/2020 10:43:20)

- links: [abs](https://arxiv.org/abs/2011.14820) | [pdf](https://arxiv.org/pdf/2011.14820)
- [math.NA](https://arxiv.org/list/math.NA/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper presents the first classical Convolutional Neural Network (CNN) that can be applied directly to data from unstructured finite element meshes or control volume grids. CNNs have been hugely influential in the areas of image classification and image compression, both of which typically deal with data on structured grids. Unstructured meshes are frequently used to solve partial differential equations and are particularly suitable for problems that require the mesh to conform to complex geometries or for problems that require variable mesh resolution. Central to the approach are space-filling curves, which traverse the nodes or cells of a mesh tracing out a path that is as short as possible (in terms of numbers of edges) and that visits each node or cell exactly once. The space-filling curves (SFCs) are used to find an ordering of the nodes or cells that can transform multi-dimensional solutions on unstructured meshes into a one-dimensional (1D) representation, to which 1D convolutional layers can then be applied. Although developed in two dimensions, the approach is applicable to higher dimensional problems.   To demonstrate the approach, the network we choose is a convolutional autoencoder (CAE) although other types of CNN could be used. The approach is tested by applying CAEs to data sets that have been reordered with an SFC. Sparse layers are used at the input and output of the autoencoder, and the use of multiple SFCs is explored. We compare the accuracy of the SFC-based CAE with that of a classical CAE applied to two idealised problems on structured meshes, and then apply the approach to solutions of flow past a cylinder obtained using the finite-element method and an unstructured mesh.




# 18. Importance Weight Estimation and Generalization in Domain Adaptation  under Label Shift

Kamyar Azizzadenesheli

- retweets: 12, favorites: 39 (12/02/2020 10:43:20)

- links: [abs](https://arxiv.org/abs/2011.14251) | [pdf](https://arxiv.org/pdf/2011.14251)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We study generalization under label shift in domain adaptation where the learner has access to labeled samples from the source domain but unlabeled samples from the target domain. Prior works deploy label classifiers and introduce various methods to estimate the importance weights from source to target domains. They use these estimates in importance weighted empirical risk minimization to learn classifiers. In this work, we theoretically compare the prior approaches, relax their strong assumptions, and generalize them from requiring label classifiers to general functions. This latter generalization improves the conditioning on the inverse operator of the induced inverse problems by allowing for broader exploitation of the spectrum of the forward operator.   The prior works in the study of label shifts are limited to categorical label spaces. In this work, we propose a series of methods to estimate the importance weight functions for arbitrary normed label spaces. We introduce a new operator learning approach between Hilbert spaces defined on labels (rather than covariates) and show that it induces a perturbed inverse problem of compact operators. We propose a novel approach to solve the inverse problem in the presence of perturbation. This analysis has its own independent interest since such problems commonly arise in partial differential equations and reinforcement learning.   For both categorical and general normed spaces, we provide concentration bounds for the proposed estimators. Using the existing generalization analysis based on Rademacher complexity, R\'enyi divergence, and MDFR lemma in Azizzadenesheli et al. [2019], we show the generalization property of the importance weighted empirical risk minimization on the unseen target domain.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It&#39;d been for a while that this draft had been sitting in my drawer with no hope to come out. Finally, putting it out.<br>Theme: Avocado Boba<br>Paper: <a href="https://t.co/U68jlktaZ7">https://t.co/U68jlktaZ7</a><br><br>Label shift is cool, interesting, important &amp; fundamental.<br>Stay tuned to c soon these methods being applied;)</p>&mdash; Kamyar Azizzadenesheli (@kazizzad) <a href="https://twitter.com/kazizzad/status/1333613881021902848?ref_src=twsrc%5Etfw">December 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



