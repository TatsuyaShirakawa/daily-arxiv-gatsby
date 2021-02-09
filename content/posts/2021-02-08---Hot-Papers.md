---
title: Hot Papers 2021-02-08
date: 2021-02-09T15:29:21.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-08"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-08"
socialImage: "/media/flying-marine.jpg"

---

# 1. How to Train Your Robot with Deep Reinforcement Learning; Lessons We've  Learned

Julian Ibarz, Jie Tan, Chelsea Finn, Mrinal Kalakrishnan, Peter Pastor, Sergey Levine

- retweets: 10227, favorites: 25 (02/09/2021 15:29:21)

- links: [abs](https://arxiv.org/abs/2102.02915) | [pdf](https://arxiv.org/pdf/2102.02915)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep reinforcement learning (RL) has emerged as a promising approach for autonomously acquiring complex behaviors from low level sensor observations. Although a large portion of deep RL research has focused on applications in video games and simulated control, which does not connect with the constraints of learning in real environments, deep RL has also demonstrated promise in enabling physical robots to learn complex skills in the real world. At the same time,real world robotics provides an appealing domain for evaluating such algorithms, as it connects directly to how humans learn; as an embodied agent in the real world. Learning to perceive and move in the real world presents numerous challenges, some of which are easier to address than others, and some of which are often not considered in RL research that focuses only on simulated domains. In this review article, we present a number of case studies involving robotic deep RL. Building off of these case studies, we discuss commonly perceived challenges in deep RL and how they have been addressed in these works. We also provide an overview of other outstanding challenges, many of which are unique to the real-world robotics setting and are not often the focus of mainstream RL research. Our goal is to provide a resource both for roboticists and machine learning researchers who are interested in furthering the progress of deep RL in the real world.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What did we learn from 5 years of robotic deep RL? My colleagues at Google and I tried to distill our experience into a review-style journal paper, covering some of the practical aspects of real-world robotic deep RL:<a href="https://t.co/fYGQfFYlKu">https://t.co/fYGQfFYlKu</a><br><br>🧵-&gt; <a href="https://t.co/rtHOJhuImN">pic.twitter.com/rtHOJhuImN</a></p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1358855369364103168?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. CharacterGAN: Few-Shot Keypoint Character Animation and Reposing

Tobias Hinz, Matthew Fisher, Oliver Wang, Eli Shechtman, Stefan Wermter

- retweets: 2021, favorites: 187 (02/09/2021 15:29:21)

- links: [abs](https://arxiv.org/abs/2102.03141) | [pdf](https://arxiv.org/pdf/2102.03141)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce CharacterGAN, a generative model that can be trained on only a few samples (8 - 15) of a given character. Our model generates novel poses based on keypoint locations, which can be modified in real time while providing interactive feedback, allowing for intuitive reposing and animation. Since we only have very limited training samples, one of the key challenges lies in how to address (dis)occlusions, e.g. when a hand moves behind or in front of a body. To address this, we introduce a novel layering approach which explicitly splits the input keypoints into different layers which are processed independently. These layers represent different parts of the character and provide a strong implicit bias that helps to obtain realistic results even with strong (dis)occlusions. To combine the features of individual layers we use an adaptive scaling approach conditioned on all keypoints. Finally, we introduce a mask connectivity constraint to reduce distortion artifacts that occur with extreme out-of-distribution poses at test time. We show that our approach outperforms recent baselines and creates realistic animations for diverse characters. We also show that our model can handle discrete state changes, for example a profile facing left or right, that the different layers do indeed learn features specific for the respective keypoints in those layers, and that our model scales to larger datasets when more data is available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CharacterGAN: Few-Shot Keypoint Character Animation and Reposing<br>pdf: <a href="https://t.co/UNTQPM5pmh">https://t.co/UNTQPM5pmh</a><br>abs: <a href="https://t.co/U7QYYjqYn8">https://t.co/U7QYYjqYn8</a><br>github: <a href="https://t.co/56XScPd99B">https://t.co/56XScPd99B</a> <a href="https://t.co/a8xFB4NbBF">pic.twitter.com/a8xFB4NbBF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358606645228687360?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. baller2vec: A Multi-Entity Transformer For Multi-Agent Spatiotemporal  Modeling

Michael A. Alcorn, Anh Nguyen

- retweets: 1760, favorites: 217 (02/09/2021 15:29:22)

- links: [abs](https://arxiv.org/abs/2102.03291) | [pdf](https://arxiv.org/pdf/2102.03291)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent)

Multi-agent spatiotemporal modeling is a challenging task from both an algorithmic design and computational complexity perspective. Recent work has explored the efficacy of traditional deep sequential models in this domain, but these architectures are slow and cumbersome to train, particularly as model size increases. Further, prior attempts to model interactions between agents across time have limitations, such as imposing an order on the agents, or making assumptions about their relationships. In this paper, we introduce baller2vec, a multi-entity generalization of the standard Transformer that, with minimal assumptions, can simultaneously and efficiently integrate information across entities and time. We test the effectiveness of baller2vec for multi-agent spatiotemporal modeling by training it to perform two different basketball-related tasks: (1) simultaneously forecasting the trajectories of all players on the court and (2) forecasting the trajectory of the ball. Not only does baller2vec learn to perform these tasks well, it also appears to "understand" the game of basketball, encoding idiosyncratic qualities of players in its embeddings, and performing basketball-relevant functions with its attention heads.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">baller2vec: A Multi-Entity Transformer For Multi-Agent Spatiotemporal Modeling<br>pdf: <a href="https://t.co/Ms7Eszk9aw">https://t.co/Ms7Eszk9aw</a><br>abs: <a href="https://t.co/BYZoGr2gDQ">https://t.co/BYZoGr2gDQ</a><br>github: <a href="https://t.co/daljSmbtLa">https://t.co/daljSmbtLa</a> <a href="https://t.co/6OSWMtHZsX">pic.twitter.com/6OSWMtHZsX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358601564924084224?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. ViLT: Vision-and-Language Transformer Without Convolution or Region  Supervision

Wonjae Kim, Bokyung Son, Ildoo Kim

- retweets: 1498, favorites: 218 (02/09/2021 15:29:22)

- links: [abs](https://arxiv.org/abs/2102.03334) | [pdf](https://arxiv.org/pdf/2102.03334)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Vision-and-Language Pretraining (VLP) has improved performance on various joint vision-and-language downstream tasks. Current approaches for VLP heavily rely on image feature extraction processes, most of which involve region supervisions (e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more computation than the actual multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive power of the visual encoder and its predefined visual vocabulary. In this paper, we present a minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the sense that processing of visual inputs is drastically simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to 60 times faster than previous VLP models, yet with competitive or better downstream task performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision<br>pdf: <a href="https://t.co/2zTB8pxVyV">https://t.co/2zTB8pxVyV</a><br>abs: <a href="https://t.co/bOoRqGhP9i">https://t.co/bOoRqGhP9i</a> <a href="https://t.co/I2uypRaYbE">pic.twitter.com/I2uypRaYbE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358602192517804032?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Forget about messy vision backbones inside vision+language models?<br><br>Check out ViLT, a cool work by Kim et al., extending Vision Transformers to multimodal domains. <br><br>Link: <a href="https://t.co/7H1JYTRdqD">https://t.co/7H1JYTRdqD</a> <a href="https://t.co/m3YHaEQ71G">pic.twitter.com/m3YHaEQ71G</a></p>&mdash; Gabriel Ilharco (@gabriel_ilharco) <a href="https://twitter.com/gabriel_ilharco/status/1358828740285923328?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. The h-index is no longer an effective correlate of scientific reputation

Vladlen Koltun, David Hafner

- retweets: 624, favorites: 162 (02/09/2021 15:29:22)

- links: [abs](https://arxiv.org/abs/2102.03234) | [pdf](https://arxiv.org/pdf/2102.03234)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

The impact of individual scientists is commonly quantified using citation-based measures. The most common such measure is the h-index. A scientist's h-index affects hiring, promotion, and funding decisions, and thus shapes the progress of science. Here we report a large-scale study of scientometric measures, analyzing millions of articles and hundreds of millions of citations across four scientific fields and two data platforms. We find that the correlation of the h-index with awards that indicate recognition by the scientific community has substantially declined. These trends are associated with changing authorship patterns. We show that these declines can be mitigated by fractional allocation of citations among authors, which has been discussed in the literature but not implemented at scale. We find that a fractional analogue of the h-index outperforms other measures as a correlate and predictor of scientific awards. Our results suggest that the use of the h-index in ranking scientists should be reconsidered, and that fractional allocation measures such as h-frac provide more robust alternatives. An interactive visualization of our work can be found at https://h-frac.org

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vladlen Koltun, David Hafner The h-index is no longer an effective correlate of scientific reputation, arXiv, 2021<br><br>arXiv: <a href="https://t.co/l2bBeg1Dtg">https://t.co/l2bBeg1Dtg</a><br>Project page: <a href="https://t.co/7Lcx7PaeMW">https://t.co/7Lcx7PaeMW</a> <a href="https://t.co/w7eyuB48QY">pic.twitter.com/w7eyuB48QY</a></p>&mdash; Kosta Derpanis (@CSProfKGD) <a href="https://twitter.com/CSProfKGD/status/1358863632940023808?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">科学分野にて用いられる h-index はもはやあまり意味がないのでは？という提案。数百万論文と数億引用を分析し、受賞論文とh-indexの相関関係が大幅に低下していることを明らかにした。<br><br>The h-index is no longer an effective correlate of scientific reputation<a href="https://t.co/iKW1qRaVuX">https://t.co/iKW1qRaVuX</a></p>&mdash; Hirokatsu Kataoka | 片岡裕雄 (@HirokatuKataoka) <a href="https://twitter.com/HirokatuKataoka/status/1358910328428371972?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Unsupervised Novel View Synthesis from a Single Image

Pierluigi Zama Ramirez, Alessio Tonioni, Federico Tombari

- retweets: 99, favorites: 67 (02/09/2021 15:29:23)

- links: [abs](https://arxiv.org/abs/2102.03285) | [pdf](https://arxiv.org/pdf/2102.03285)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Novel view synthesis from a single image aims at generating novel views from a single input image of an object. Several works recently achieved remarkable results, though require some form of multi-view supervision at training time, therefore limiting their deployment in real scenarios. This work aims at relaxing this assumption enabling training of conditional generative model for novel view synthesis in a completely unsupervised manner. We first pre-train a purely generative decoder model using a GAN formulation while at the same time training an encoder network to invert the mapping from latent code to images. Then we swap encoder and decoder and train the network as a conditioned GAN with a mixture of auto-encoder-like objective and self-distillation. At test time, given a view of an object, our model first embeds the image content in a latent code and regresses its pose w.r.t. a canonical reference system, then generates novel views of it by keeping the code and varying the pose. We show that our framework achieves results comparable to the state of the art on ShapeNet and that it can be employed on unconstrained collections of natural images, where no competing method can be trained.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Novel View Synthesis from a Single Image<br>pdf: <a href="https://t.co/5gV8z1L3Ix">https://t.co/5gV8z1L3Ix</a><br>abs: <a href="https://t.co/39N1NQHeDV">https://t.co/39N1NQHeDV</a> <a href="https://t.co/z0XHBjBjNV">pic.twitter.com/z0XHBjBjNV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358616737583931392?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Measuring Utility and Privacy of Synthetic Genomic Data

Bristena Oprisanu, Georgi Ganev, Emiliano De Cristofaro

- retweets: 56, favorites: 44 (02/09/2021 15:29:23)

- links: [abs](https://arxiv.org/abs/2102.03314) | [pdf](https://arxiv.org/pdf/2102.03314)
- [q-bio.GN](https://arxiv.org/list/q-bio.GN/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent)

Genomic data provides researchers with an invaluable source of information to advance progress in biomedical research, personalized medicine, and drug development. At the same time, however, this data is extremely sensitive, which makes data sharing, and consequently availability, problematic if not outright impossible. As a result, organizations have begun to experiment with sharing synthetic data, which should mirror the real data's salient characteristics, without exposing it. In this paper, we provide the first evaluation of the utility and the privacy protection of five state-of-the-art models for generating synthetic genomic data.   First, we assess the performance of the synthetic data on a number of common tasks, such as allele and population statistics as well as linkage disequilibrium and principal component analysis. Then, we study the susceptibility of the data to membership inference attacks, i.e., inferring whether a target record was part of the data used to train the model producing the synthetic dataset. Overall, there is no single approach for generating synthetic genomic data that performs well across the board. We show how the size and the nature of the training dataset matter, especially in the case of generative models. While some combinations of datasets and models produce synthetic data with distributions close to the real data, there often are target data points that are vulnerable to membership inference. Our measurement framework can be used by practitioners to assess the risks of deploying synthetic genomic data in the wild, and will serve as a benchmark tool for researchers and practitioners in the future.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New pre-print &quot;Measuring Utility and Privacy of Synthetic Genomic Data&quot;<a href="https://t.co/ULmxX2p2fh">https://t.co/ULmxX2p2fh</a> <a href="https://t.co/FrWfmWBddA">pic.twitter.com/FrWfmWBddA</a></p>&mdash; Emiliano DC (@emilianoucl) <a href="https://twitter.com/emilianoucl/status/1358682515905646598?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. 1-bit Adam: Communication Efficient Large-Scale Training with Adam's  Convergence Speed

Hanlin Tang, Shaoduo Gan, Ammar Ahmad Awan, Samyam Rajbhandari, Conglong Li, Xiangru Lian, Ji Liu, Ce Zhang, Yuxiong He

- retweets: 36, favorites: 47 (02/09/2021 15:29:23)

- links: [abs](https://arxiv.org/abs/2102.02888) | [pdf](https://arxiv.org/pdf/2102.02888)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Scalable training of large models (like BERT and GPT-3) requires careful optimization rooted in model design, architecture, and system capabilities. From a system standpoint, communication has become a major bottleneck, especially on commodity systems with standard TCP interconnects that offer limited network bandwidth. Communication compression is an important technique to reduce training time on such systems. One of the most effective methods is error-compensated compression, which offers robust convergence speed even under 1-bit compression. However, state-of-the-art error compensation techniques only work with basic optimizers like SGD and momentum SGD, which are linearly dependent on the gradients. They do not work with non-linear gradient-based optimizers like Adam, which offer state-of-the-art convergence efficiency and accuracy for models like BERT. In this paper, we propose 1-bit Adam that reduces the communication volume by up to $5\times$, offers much better scalability, and provides the same convergence speed as uncompressed Adam. Our key finding is that Adam's variance (non-linear term) becomes stable (after a warmup phase) and can be used as a fixed precondition for the rest of the training (compression phase). Experiments on up to 256 GPUs show that 1-bit Adam enables up to $3.3\times$ higher throughput for BERT-Large pre-training and up to $2.9\times$ higher throughput for SQuAD fine-tuning. In addition, we provide theoretical analysis for our proposed work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1-bit Adam: Communication Efficient Large-Scale Training with Adam’s Convergence Speed<br><br>Proposes 1-bit Adam that reduces the communication volume by up to 5x with the same convergence speed as the regular Adam. <a href="https://t.co/TOXxaz5yk1">https://t.co/TOXxaz5yk1</a> <a href="https://t.co/1K7ytCuoj9">pic.twitter.com/1K7ytCuoj9</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1358596224220155906?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. PipeTransformer: Automated Elastic Pipelining for Distributed Training  of Transformers

Chaoyang He, Shen Li, Mahdi Soltanolkotabi, Salman Avestimehr

- retweets: 38, favorites: 33 (02/09/2021 15:29:23)

- links: [abs](https://arxiv.org/abs/2102.03161) | [pdf](https://arxiv.org/pdf/2102.03161)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

The size of Transformer models is growing at an unprecedented pace. It has only taken less than one year to reach trillion-level parameters after the release of GPT-3 (175B). Training such models requires both substantial engineering efforts and enormous computing resources, which are luxuries most research teams cannot afford. In this paper, we propose PipeTransformer, which leverages automated and elastic pipelining and data parallelism for efficient distributed training of Transformer models. PipeTransformer automatically adjusts the pipelining and data parallelism by identifying and freezing some layers during the training, and instead allocates resources for training of the remaining active layers. More specifically, PipeTransformer dynamically excludes converged layers from the pipeline, packs active layers into fewer GPUs, and forks more replicas to increase data-parallel width. We evaluate PipeTransformer using Vision Transformer (ViT) on ImageNet and BERT on GLUE and SQuAD datasets. Our results show that PipeTransformer attains a 2.4 fold speedup compared to the state-of-the-art baseline. We also provide various performance analyses for a more comprehensive understanding of our algorithmic and system-wise design. We also develop open-sourced flexible APIs for PipeTransformer, which offer a clean separation among the freeze algorithm, model definitions, and training accelerations, hence allowing it to be applied to other algorithms that require similar freezing strategies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PipeTransformer: Automated Elastic Pipelining for Distributed Training of Transformers<br>pdf: <a href="https://t.co/O7ymi5wSS7">https://t.co/O7ymi5wSS7</a><br>abs: <a href="https://t.co/3m71cp8igY">https://t.co/3m71cp8igY</a> <a href="https://t.co/81URHYdE5J">pic.twitter.com/81URHYdE5J</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358600738168057860?ref_src=twsrc%5Etfw">February 8, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Frontrunner Jones and the Raiders of the Dark Forest: An Empirical Study  of Frontrunning on the Ethereum Blockchain

Christof Ferreira Torres, Ramiro Camino, Radu State

- retweets: 42, favorites: 19 (02/09/2021 15:29:23)

- links: [abs](https://arxiv.org/abs/2102.03347) | [pdf](https://arxiv.org/pdf/2102.03347)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

Ethereum prospered the inception of a plethora of smart contract applications, ranging from gambling games to decentralized finance. However, Ethereum is also considered a highly adversarial environment, where vulnerable smart contracts will eventually be exploited. Recently, Ethereum's pool of pending transaction has become a far more aggressive environment. In the hope of making some profit, attackers continuously monitor the transaction pool and try to front-run their victims' transactions by either displacing or suppressing them, or strategically inserting their transactions. This paper aims to shed some light into what is known as a dark forest and uncover these predators' actions. We present a methodology to efficiently measure the three types of frontrunning: displacement, insertion, and suppression. We perform a large-scale analysis on more than 11M blocks and identify almost 200K attacks with an accumulated profit of 18.41M USD for the attackers, providing evidence that frontrunning is both, lucrative and a prevalent issue.




# 11. Positioning in 5G networks

Satyam Dwivedi, Ritesh Shreevastav, Florent Munier, Johannes Nygren, Iana Siomina, Yazid Lyazidi, Deep Shrestha, Gustav Lindmark, Per Ernström, Erik Stare, Sara M. Razavi, Siva Muruganathan, Gino Masini, Åke Busin, Fredrik Gunnarsson

- retweets: 49, favorites: 10 (02/09/2021 15:29:23)

- links: [abs](https://arxiv.org/abs/2102.03361) | [pdf](https://arxiv.org/pdf/2102.03361)
- [cs.NI](https://arxiv.org/list/cs.NI/recent)

In this paper we describe the recent 3GPP Release 16 specification for positioning in 5G networks. It specifies positioning signals, measurements, procedures, and architecture to meet requirements from a plethora of regulatory, commercial and industrial use cases. 5G thereby significantly extends positioning capabilities compared to what was possible with LTE. The indicative positioning performance is evaluated in agreed representative 3GPP simulation scenarios, showing a 90 percentile accuracy of a few meters down to a few decimeters depending on scenarios and assumptions.



