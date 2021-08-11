---
title: Hot Papers 2021-08-10
date: 2021-08-11T11:38:42.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-10"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-10"
socialImage: "/media/flying-marine.jpg"

---

# 1. Paint Transformer: Feed Forward Neural Painting with Stroke Prediction

Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Ruifeng Deng, Xin Li, Errui Ding, Hao Wang

- retweets: 4878, favorites: 3 (08/11/2021 11:38:42)

- links: [abs](https://arxiv.org/abs/2108.03798) | [pdf](https://arxiv.org/pdf/2108.03798)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Neural painting refers to the procedure of producing a series of strokes for a given image and non-photo-realistically recreating it using neural networks. While reinforcement learning (RL) based agents can generate a stroke sequence step by step for this task, it is not easy to train a stable RL agent. On the other hand, stroke optimization methods search for a set of stroke parameters iteratively in a large search space; such low efficiency significantly limits their prevalence and practicality. Different from previous methods, in this paper, we formulate the task as a set prediction problem and propose a novel Transformer-based framework, dubbed Paint Transformer, to predict the parameters of a stroke set with a feed forward network. This way, our model can generate a set of strokes in parallel and obtain the final painting of size 512 * 512 in near real time. More importantly, since there is no dataset available for training the Paint Transformer, we devise a self-training pipeline such that it can be trained without any off-the-shelf dataset while still achieving excellent generalization capability. Experiments demonstrate that our method achieves better painting performance than previous ones with cheaper training and inference costs. Codes and models are available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Paint Transformer: Feed Forward Neural Painting with Stroke Prediction<br>pdf: <a href="https://t.co/4sWy843kcj">https://t.co/4sWy843kcj</a><br>abs: <a href="https://t.co/3rBCp8ITNQ">https://t.co/3rBCp8ITNQ</a><br>github: <a href="https://t.co/UczmdRh5Bl">https://t.co/UczmdRh5Bl</a><br><br>model can generate a set of strokes in parallel and obtain the final painting of size 512 * 512 in near real time <a href="https://t.co/aTIlI6594Y">pic.twitter.com/aTIlI6594Y</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424901538095435779?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Pathfinder: Parallel quasi-Newton variational inference

Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari

- retweets: 1199, favorites: 228 (08/11/2021 11:38:43)

- links: [abs](https://arxiv.org/abs/2108.03782) | [pdf](https://arxiv.org/pdf/2108.03782)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We introduce Pathfinder, a variational method for approximately sampling from differentiable log densities. Starting from a random initialization, Pathfinder locates normal approximations to the target density along a quasi-Newton optimization path, with local covariance estimated using the inverse Hessian estimates produced by the optimizer. Pathfinder returns draws from the approximation with the lowest estimated Kullback-Leibler (KL) divergence to the true posterior. We evaluate Pathfinder on a wide range of posterior distributions, demonstrating that its approximate draws are better than those from automatic differentiation variational inference (ADVI) and comparable to those produced by short chains of dynamic Hamiltonian Monte Carlo (HMC), as measured by 1-Wasserstein distance. Compared to ADVI and short dynamic HMC runs, Pathfinder requires one to two orders of magnitude fewer log density and gradient evaluations, with greater reductions for more challenging posteriors. Importance resampling over multiple runs of Pathfinder improves the diversity of approximate draws, reducing 1-Wasserstein distance further and providing a measure of robustness to optimization failures on plateaus, saddle points, or in minor modes. The Monte Carlo KL-divergence estimates are embarrassingly parallelizable in the core Pathfinder algorithm, as are multiple runs in the resampling version, further increasing Pathfinder's speed advantage with multiple cores.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper &quot;Pathfinder: Parallel quasi-Newton variational inference&quot; with Lu Zhang, Bob Carpenter, and Andrew Gelman <a href="https://t.co/UHf116HvMj">https://t.co/UHf116HvMj</a>. We combine deterministic quasi-Newton optimization with variational KL-divergence minimization. <a href="https://t.co/z4QigsFcZL">pic.twitter.com/z4QigsFcZL</a></p>&mdash; Aki Vehtari (@avehtari) <a href="https://twitter.com/avehtari/status/1425008776709562397?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">quite enjoying this complexity annotation for algorithm blocks in this paper, e.g.<br><br>(from <a href="https://t.co/xStGWmJiyu">https://t.co/xStGWmJiyu</a>) <a href="https://t.co/ISegZR7orr">pic.twitter.com/ISegZR7orr</a></p>&mdash; Sam Power (@sam_power_825) <a href="https://twitter.com/sam_power_825/status/1425025196570857472?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Semantic Tracklets: An Object-Centric Representation for Visual  Multi-Agent Reinforcement Learning

Iou-Jen Liu, Zhongzheng Ren, Raymond A. Yeh, Alexander G. Schwing

- retweets: 552, favorites: 110 (08/11/2021 11:38:43)

- links: [abs](https://arxiv.org/abs/2108.03319) | [pdf](https://arxiv.org/pdf/2108.03319)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Solving complex real-world tasks, e.g., autonomous fleet control, often involves a coordinated team of multiple agents which learn strategies from visual inputs via reinforcement learning. Many existing multi-agent reinforcement learning (MARL) algorithms however don't scale to environments where agents operate on visual inputs. To address this issue, algorithmically, recent works have focused on non-stationarity and exploration. In contrast, we study whether scalability can also be achieved via a disentangled representation. For this, we explicitly construct an object-centric intermediate representation to characterize the states of an environment, which we refer to as `semantic tracklets.' We evaluate `semantic tracklets' on the visual multi-agent particle environment (VMPE) and on the challenging visual multi-agent GFootball environment. `Semantic tracklets' consistently outperform baselines on VMPE, and achieve a +2.4 higher score difference than baselines on GFootball. Notably, this method is the first to successfully learn a strategy for five players in the GFootball environment using only visual data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Semantic Tracklets: An Object-Centric Representation for Visual Multi-Agent Reinforcement Learning<br>pdf: <a href="https://t.co/pOjR5yBKca">https://t.co/pOjR5yBKca</a><br>abs: <a href="https://t.co/KkZ8xyHAKa">https://t.co/KkZ8xyHAKa</a><br>project page: <a href="https://t.co/d6rmh3iXKv">https://t.co/d6rmh3iXKv</a> <a href="https://t.co/hAFVRnCHrU">pic.twitter.com/hAFVRnCHrU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424944798075695104?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Noisy Channel Language Model Prompting for Few-Shot Text Classification

Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer

- retweets: 426, favorites: 98 (08/11/2021 11:38:43)

- links: [abs](https://arxiv.org/abs/2108.04106) | [pdf](https://arxiv.org/pdf/2108.04106)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We introduce a noisy channel approach for language model prompting in few-shot text classification. Instead of computing the likelihood of the label given the input (referred as direct models), channel models compute the conditional probability of the input given the label, and are thereby required to explain every word in the input. We use channel models for recently proposed few-shot learning methods with no or very limited updates to the language model parameters, via either in-context demonstration or prompt tuning. Our experiments show that, for both methods, channel models significantly outperform their direct counterparts, which we attribute to their stability, i.e., lower variance and higher worst-case accuracy. We also present extensive ablations that provide recommendations for when to use channel prompt tuning instead of other competitive models (e.g., direct head tuning): channel prompt tuning is preferred when the number of training examples is small, labels in the training data are imbalanced, or generalization to unseen labels is required.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper!✨We introduce a noisy channel approach for LM prompting in few-shot text classification. Channel models are more stable (much lower variance), and better with limited data / imbalanced labels.<a href="https://t.co/WXUeI97QRE">https://t.co/WXUeI97QRE</a><br>w/ <a href="https://twitter.com/ml_perception?ref_src=twsrc%5Etfw">@ml_perception</a> <a href="https://twitter.com/HannaHajishirzi?ref_src=twsrc%5Etfw">@HannaHajishirzi</a> <a href="https://twitter.com/LukeZettlemoyer?ref_src=twsrc%5Etfw">@LukeZettlemoyer</a></p>&mdash; Sewon Min (@sewon__min) <a href="https://twitter.com/sewon__min/status/1425099713997537280?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Impact of Aliasing on Generalization in Deep Convolutional Networks

Cristina Vasconcelos, Hugo Larochelle, Vincent Dumoulin, Rob Romijnders, Nicolas Le Roux, Ross Goroshin

- retweets: 140, favorites: 163 (08/11/2021 11:38:43)

- links: [abs](https://arxiv.org/abs/2108.03489) | [pdf](https://arxiv.org/pdf/2108.03489)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We investigate the impact of aliasing on generalization in Deep Convolutional Networks and show that data augmentation schemes alone are unable to prevent it due to structural limitations in widely used architectures. Drawing insights from frequency analysis theory, we take a closer look at ResNet and EfficientNet architectures and review the trade-off between aliasing and information loss in each of their major components. We show how to mitigate aliasing by inserting non-trainable low-pass filters at key locations, particularly where networks lack the capacity to learn them. These simple architectural changes lead to substantial improvements in generalization on i.i.d. and even more on out-of-distribution conditions, such as image classification under natural corruptions on ImageNet-C [11] and few-shot learning on Meta-Dataset [26]. State-of-the art results are achieved on both datasets without introducing additional trainable parameters and using the default hyper-parameters of open source codebases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am excited to share that our paper &quot;Impact of Aliasing on Generalization in Deep Convolutional Networks&quot; has been accepted to <a href="https://twitter.com/ICCV_2021?ref_src=twsrc%5Etfw">@ICCV_2021</a> <a href="https://t.co/UIkxwzd4Jf">https://t.co/UIkxwzd4Jf</a> <br>Special thanks to my coauthors <a href="https://twitter.com/hugo_larochelle?ref_src=twsrc%5Etfw">@hugo_larochelle</a> <a href="https://twitter.com/dumoulinv?ref_src=twsrc%5Etfw">@dumoulinv</a> <a href="https://twitter.com/robromijnders?ref_src=twsrc%5Etfw">@robromijnders</a>, <a href="https://twitter.com/le_roux_nicolas?ref_src=twsrc%5Etfw">@le_roux_nicolas</a>  and Ross Goroshin</p>&mdash; Cristina Vasconcelos (@Cristin13377923) <a href="https://twitter.com/Cristin13377923/status/1424951923027890192?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Impact of Aliasing on Generalization in Deep Convolutional Networks<br>pdf: <a href="https://t.co/qxUYtmXuNs">https://t.co/qxUYtmXuNs</a><br>abs: <a href="https://t.co/JhGKHxLbuR">https://t.co/JhGKHxLbuR</a><br><br>proposed simple architectural improvements to convolutional architectures to counter aliasing occurring at various stages <a href="https://t.co/pouJvssvsn">pic.twitter.com/pouJvssvsn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424923224572407808?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">CNNのサブサンプリング時にエイリアシングが発生し、大量のデータで学習したりデータオーグメンテーションを適用しても防げない。重要な発生箇所を特定し、そこでサブサンプリング時にローパスフィルタを適用することで、i.i.d, o.o.d両方で汎化性能を大きく改善できる<a href="https://t.co/Oo9zK3t1F6">https://t.co/Oo9zK3t1F6</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1425242959121813507?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. PSViT: Better Vision Transformer via Token Pooling and Attention Sharing

Boyu Chen, Peixia Li, Baopu Li, Chuming Li, Lei Bai, Chen Lin, Ming Sun, Junjie Yan, Wanli Ouyang

- retweets: 143, favorites: 55 (08/11/2021 11:38:44)

- links: [abs](https://arxiv.org/abs/2108.03428) | [pdf](https://arxiv.org/pdf/2108.03428)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we observe two levels of redundancies when applying vision transformers (ViT) for image recognition. First, fixing the number of tokens through the whole network produces redundant features at the spatial level. Second, the attention maps among different transformer layers are redundant. Based on the observations above, we propose a PSViT: a ViT with token Pooling and attention Sharing to reduce the redundancy, effectively enhancing the feature representation ability, and achieving a better speed-accuracy trade-off. Specifically, in our PSViT, token pooling can be defined as the operation that decreases the number of tokens at the spatial level. Besides, attention sharing will be built between the neighboring transformer layers for reusing the attention maps having a strong correlation among adjacent layers. Then, a compact set of the possible combinations for different token pooling and attention sharing mechanisms are constructed. Based on the proposed compact set, the number of tokens in each layer and the choices of layers sharing attention can be treated as hyper-parameters that are learned from data automatically. Experimental results show that the proposed scheme can achieve up to 6.6% accuracy improvement in ImageNet classification compared with the DeiT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PSViT: Better Vision Transformer via Token Pooling and Attention Sharing<br>pdf: <a href="https://t.co/8Ixk97V6nb">https://t.co/8Ixk97V6nb</a><br>abs: <a href="https://t.co/OOnONItfnX">https://t.co/OOnONItfnX</a><br><br>can achieve up to 6.6% accuracy improvement in ImageNet classification compared with DeiT <a href="https://t.co/5PVV2BAu8X">pic.twitter.com/5PVV2BAu8X</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424900668804960256?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. IntenT5: Search Result Diversification using Causal Language Models

Sean MacAvaney, Craig Macdonald, Roderick Murray-Smith, Iadh Ounis

- retweets: 156, favorites: 39 (08/11/2021 11:38:44)

- links: [abs](https://arxiv.org/abs/2108.04026) | [pdf](https://arxiv.org/pdf/2108.04026)
- [cs.IR](https://arxiv.org/list/cs.IR/recent)

Search result diversification is a beneficial approach to overcome under-specified queries, such as those that are ambiguous or multi-faceted. Existing approaches often rely on massive query logs and interaction data to generate a variety of possible query intents, which then can be used to re-rank documents. However, relying on user interaction data is problematic because one first needs a massive user base to build a sufficient log; public query logs are insufficient on their own. Given the recent success of causal language models (such as the Text-To-Text Transformer (T5) model) at text generation tasks, we explore the capacity of these models to generate potential query intents. We find that to encourage diversity in the generated queries, it is beneficial to adapt the model by including a new Distributional Causal Language Modeling (DCLM) objective during fine-tuning and a representation replacement during inference. Across six standard evaluation benchmarks, we find that our method (which we call IntenT5) improves search result diversity and attains (and sometimes exceeds) the diversity obtained when using query suggestions based on a proprietary query log. Our analysis shows that our approach is most effective for multi-faceted queries and is able to generalize effectively to queries that were unseen in training data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new preprint: IntenT5.<br><br>tl;dr: BERT and friends usually latch onto a single query intent, which isn&#39;t good for under-specified queries. We can use a T5 to generate a diverse set of possible intents!<a href="https://t.co/Kzzn9viU48">https://t.co/Kzzn9viU48</a><br><br>w/ <a href="https://twitter.com/craig_macdonald?ref_src=twsrc%5Etfw">@craig_macdonald</a> <a href="https://twitter.com/iadh?ref_src=twsrc%5Etfw">@iadh</a> <a href="https://twitter.com/MurraySmithRod?ref_src=twsrc%5Etfw">@MurraySmithRod</a></p>&mdash; Sean MacAvaney (@macavaney) <a href="https://twitter.com/macavaney/status/1425008990027763712?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer

Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling Wang, Xin Li, Zhengxing Sun, Qian Li, Errui Ding

- retweets: 110, favorites: 55 (08/11/2021 11:38:44)

- links: [abs](https://arxiv.org/abs/2108.03647) | [pdf](https://arxiv.org/pdf/2108.03647)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Fast arbitrary neural style transfer has attracted widespread attention from academic, industrial and art communities due to its flexibility in enabling various applications. Existing solutions either attentively fuse deep style feature into deep content feature without considering feature distributions, or adaptively normalize deep content feature according to the style such that their global statistics are matched. Although effective, leaving shallow feature unexplored and without locally considering feature statistics, they are prone to unnatural output with unpleasing local distortions. To alleviate this problem, in this paper, we propose a novel attention and normalization module, named Adaptive Attention Normalization (AdaAttN), to adaptively perform attentive normalization on per-point basis. Specifically, spatial attention score is learnt from both shallow and deep features of content and style images. Then per-point weighted statistics are calculated by regarding a style feature point as a distribution of attention-weighted output of all style feature points. Finally, the content feature is normalized so that they demonstrate the same local feature statistics as the calculated per-point weighted style feature statistics. Besides, a novel local feature loss is derived based on AdaAttN to enhance local visual quality. We also extend AdaAttN to be ready for video style transfer with slight modifications. Experiments demonstrate that our method achieves state-of-the-art arbitrary image/video style transfer. Codes and models are available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer<br>pdf: <a href="https://t.co/VaWfcR2kKh">https://t.co/VaWfcR2kKh</a><br>abs: <a href="https://t.co/FtEfKs8Ekk">https://t.co/FtEfKs8Ekk</a><br>github: <a href="https://t.co/Y3IFKiKZok">https://t.co/Y3IFKiKZok</a> <a href="https://t.co/xO9YwDZ3cK">pic.twitter.com/xO9YwDZ3cK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424924828021239814?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Asymmetry-aware Scalable Locking

Nian Liu, Jinyu Gu, Dahai Tang, Kenli Li, Binyu Zang, Haibo Chen

- retweets: 100, favorites: 28 (08/11/2021 11:38:44)

- links: [abs](https://arxiv.org/abs/2108.03355) | [pdf](https://arxiv.org/pdf/2108.03355)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

The pursuit of power-efficiency is popularizing asymmetric multicore processors (AMP) such as ARM big.LITTLE, Apple M1 and recent Intel Alder Lake with big and little cores. However, we find that existing scalable locks fail to scale on AMP and cause collapses in either throughput or latency, or both, because their implicit assumption of symmetric cores no longer holds. To address this issue, we propose the first asymmetry-aware scalable lock named LibASL. LibASL provides a new lock ordering guided by applications' latency requirements, which allows big cores to reorder with little cores for higher throughput under the condition of preserving applications' latency requirements. Using LibASL only requires linking the applications with it and, if latency-critical, inserting few lines of code to annotate the coarse-grained latency requirement. We evaluate LibASL in various benchmarks including five popular databases on Apple M1. Evaluation results show that LibASL can improve the throughput by up to 5 times while precisely preserving the tail latency designated by applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper, researchers have proposed LibASL, the first asymmetry-aware scalable lock which allows big cores to reorder with little cores for higher throughput under the condition of preserving applications’ latency requirements.<a href="https://t.co/3abilz796s">https://t.co/3abilz796s</a> <a href="https://t.co/AKEziiRQFC">pic.twitter.com/AKEziiRQFC</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1424972593887580160?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Exploring the potential of flow-based programming for machine learning  deployment in comparison with service-oriented architectures

Andrei Paleyes, Christian Cabrera, Neil D. Lawrence

- retweets: 72, favorites: 55 (08/11/2021 11:38:44)

- links: [abs](https://arxiv.org/abs/2108.04105) | [pdf](https://arxiv.org/pdf/2108.04105)
- [cs.SE](https://arxiv.org/list/cs.SE/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Despite huge successes reported by the field of machine learning, such as speech assistants or self-driving cars, businesses still observe very high failure rate when it comes to deployment of ML in production. We argue that part of the reason is infrastructure that was not designed for activities around data collection and analysis. We propose to consider flow-based programming with data streams as an alternative to commonly used service-oriented architectures for building software applications. To compare flow-based programming with the widespread service-oriented approach, we develop a data processing application, and formulate two subsequent ML-related tasks that constitute a complete cycle of ML deployment while allowing us to assess characteristics of each programming paradigm in the ML context. Employing both code metrics and empirical observations, we show that when it comes to ML deployment each paradigm has certain advantages and drawbacks. Our main conclusion is that while FBP shows great potential for providing infrastructural benefits for deployment of machine learning, it requires a lot of boilerplate code to define and manipulate the dataflow graph. We believe that with better developer tools in place this problem can be alleviated, establishing FBP as a strong alternative to currently prevalent SOA-driven software design approach. Additionally, we provide an insight into the trend of prioritising model development over data quality management.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We&#39;ve been thinking about better software engineering paradigms for maintaining and explaing production ML code.<br><br>Andrei Paleyes and Christian Cabrera have led our exploration of flow based programming with this in mind.<a href="https://t.co/oqk73ihMI1">https://t.co/oqk73ihMI1</a></p>&mdash; Neil Lawrence (@lawrennd) <a href="https://twitter.com/lawrennd/status/1425043627206656006?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. The Right to Talk: An Audio-Visual Transformer Approach

Thanh-Dat Truong, Chi Nhan Duong, De Vu, Hoang Anh Pham, Bhiksha Raj, Ngan Le, Khoa Luu

- retweets: 70, favorites: 36 (08/11/2021 11:38:45)

- links: [abs](https://arxiv.org/abs/2108.03256) | [pdf](https://arxiv.org/pdf/2108.03256)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Turn-taking has played an essential role in structuring the regulation of a conversation. The task of identifying the main speaker (who is properly taking his/her turn of speaking) and the interrupters (who are interrupting or reacting to the main speaker's utterances) remains a challenging task. Although some prior methods have partially addressed this task, there still remain some limitations. Firstly, a direct association of Audio and Visual features may limit the correlations to be extracted due to different modalities. Secondly, the relationship across temporal segments helping to maintain the consistency of localization, separation, and conversation contexts is not effectively exploited. Finally, the interactions between speakers that usually contain the tracking and anticipatory decisions about the transition to a new speaker are usually ignored. Therefore, this work introduces a new Audio-Visual Transformer approach to the problem of localization and highlighting the main speaker in both audio and visual channels of a multi-speaker conversation video in the wild. The proposed method exploits different types of correlations presented in both visual and audio signals. The temporal audio-visual relationships across spatial-temporal space are anticipated and optimized via the self-attention mechanism in a Transformerstructure. Moreover, a newly collected dataset is introduced for the main speaker detection. To the best of our knowledge, it is one of the first studies that is able to automatically localize and highlight the main speaker in both visual and audio channels in multi-speaker conversation videos.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Right to Talk: An Audio-Visual Transformer Approach<br>pdf: <a href="https://t.co/C2ISc6xbek">https://t.co/C2ISc6xbek</a><br>abs: <a href="https://t.co/vRLQN3Ub6D">https://t.co/vRLQN3Ub6D</a><br><br>method can effectively localize and highlight the main speaker in both visual and audio channels on multi-speaker conversation videos <a href="https://t.co/4l9tW2n0B9">pic.twitter.com/4l9tW2n0B9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424917447375794178?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. NeuralMVS: Bridging Multi-View Stereo and Novel View Synthesis

Radu Alexandru Rosu, Sven Behnke

- retweets: 62, favorites: 42 (08/11/2021 11:38:45)

- links: [abs](https://arxiv.org/abs/2108.03880) | [pdf](https://arxiv.org/pdf/2108.03880)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Multi-View Stereo (MVS) is a core task in 3D computer vision. With the surge of novel deep learning methods, learned MVS has surpassed the accuracy of classical approaches, but still relies on building a memory intensive dense cost volume. Novel View Synthesis (NVS) is a parallel line of research and has recently seen an increase in popularity with Neural Radiance Field (NeRF) models, which optimize a per scene radiance field. However, NeRF methods do not generalize to novel scenes and are slow to train and test. We propose to bridge the gap between these two methodologies with a novel network that can recover 3D scene geometry as a distance function, together with high-resolution color images. Our method uses only a sparse set of images as input and can generalize well to novel scenes. Additionally, we propose a coarse-to-fine sphere tracing approach in order to significantly increase speed. We show on various datasets that our method reaches comparable accuracy to per-scene optimized methods while being able to generalize and running significantly faster.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeuralMVS: Bridging Multi-View Stereo and Novel<br>View Synthesis<br>pdf: <a href="https://t.co/PboI1vZ3HB">https://t.co/PboI1vZ3HB</a><br>abs: <a href="https://t.co/D1iKTb7CeJ">https://t.co/D1iKTb7CeJ</a><br><br>proposed a network that jointly resolves scene geometry and novel view synthesis from multiview datasets and is supervised only by image reconstruction loss <a href="https://t.co/sfDziMzyK8">pic.twitter.com/sfDziMzyK8</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424904511148134406?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Image Retrieval on Real-life Images with Pre-trained Vision-and-Language  Models

Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, Stephen Gould

- retweets: 32, favorites: 50 (08/11/2021 11:38:45)

- links: [abs](https://arxiv.org/abs/2108.04024) | [pdf](https://arxiv.org/pdf/2108.04024)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

We extend the task of composed image retrieval, where an input query consists of an image and short textual description of how to modify the image. Existing methods have only been applied to non-complex images within narrow domains, such as fashion products, thereby limiting the scope of study on in-depth visual reasoning in rich image and language contexts. To address this issue, we collect the Compose Image Retrieval on Real-life images (CIRR) dataset, which consists of over 36,000 pairs of crowd-sourced, open-domain images with human-generated modifying text. To extend current methods to the open-domain, we propose CIRPLANT, a transformer based model that leverages rich pre-trained vision-and-language (V&L) knowledge for modifying visual features conditioned on natural language. Retrieval is then done by nearest neighbor lookup on the modified features. We demonstrate that with a relatively simple architecture, CIRPLANT outperforms existing methods on open-domain images, while matching state-of-the-art accuracy on the existing narrow datasets, such as fashion. Together with the release of CIRR, we believe this work will inspire further research on composed image retrieval.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models<br>pdf: <a href="https://t.co/lxvU0JuEIa">https://t.co/lxvU0JuEIa</a><br>abs: <a href="https://t.co/t9rGkT50Jf">https://t.co/t9rGkT50Jf</a><br>project page: <a href="https://t.co/jL61yiyc23">https://t.co/jL61yiyc23</a> <a href="https://t.co/GHFLnDXGQs">pic.twitter.com/GHFLnDXGQs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424969733137485824?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Multilingual Compositional Wikidata Questions

Ruixiang Cui, Rahul Aralikatte, Heather Lent, Daniel Hershcovich

- retweets: 58, favorites: 17 (08/11/2021 11:38:45)

- links: [abs](https://arxiv.org/abs/2108.03509) | [pdf](https://arxiv.org/pdf/2108.03509)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Semantic parsing allows humans to leverage vast knowledge resources through natural interaction. However, parsers are mostly designed for and evaluated on English resources, such as CFQ (Keysers et al., 2020), the current standard benchmark based on English data generated from grammar rules and oriented towards Freebase, an outdated knowledge base. We propose a method for creating a multilingual, parallel dataset of question-query pairs, grounded in Wikidata, and introduce such a dataset called Compositional Wikidata Questions (CWQ). We utilize this data to train and evaluate semantic parsers for Hebrew, Kannada, Chinese and English, to better understand the current strengths and weaknesses of multilingual semantic parsing. Experiments on zero-shot cross-lingual transfer demonstrate that models fail to generate valid queries even with pretrained multilingual encoders. Our methodology, dataset and results will facilitate future research on semantic parsing in more realistic and diverse settings than has been possible with existing resources.




# 15. 3D Human Reconstruction in the Wild with Collaborative Aerial Cameras

Cherie Ho, Andrew Jong, Harry Freeman, Rohan Rao, Rogerio Bonatti, Sebastian Scherer

- retweets: 42, favorites: 27 (08/11/2021 11:38:45)

- links: [abs](https://arxiv.org/abs/2108.03936) | [pdf](https://arxiv.org/pdf/2108.03936)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Aerial vehicles are revolutionizing applications that require capturing the 3D structure of dynamic targets in the wild, such as sports, medicine, and entertainment. The core challenges in developing a motion-capture system that operates in outdoors environments are: (1) 3D inference requires multiple simultaneous viewpoints of the target, (2) occlusion caused by obstacles is frequent when tracking moving targets, and (3) the camera and vehicle state estimation is noisy. We present a real-time aerial system for multi-camera control that can reconstruct human motions in natural environments without the use of special-purpose markers. We develop a multi-robot coordination scheme that maintains the optimal flight formation for target reconstruction quality amongst obstacles. We provide studies evaluating system performance in simulation, and validate real-world performance using two drones while a target performs activities such as jogging and playing soccer. Supplementary video: https://youtu.be/jxt91vx0cns

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">3D Human Reconstruction in the Wild with Collaborative Aerial Cameras<br>pdf: <a href="https://t.co/paPzWAzpaE">https://t.co/paPzWAzpaE</a><br>abs: <a href="https://t.co/VDmxavsJHk">https://t.co/VDmxavsJHk</a> <a href="https://t.co/TOl6NMqK9q">pic.twitter.com/TOl6NMqK9q</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1424948502061273092?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. A Machine Learning Tool to Determine State of Mind and Emotion

Rodrigo S. Jamisola Jr

- retweets: 56, favorites: 4 (08/11/2021 11:38:45)

- links: [abs](https://arxiv.org/abs/2108.03444) | [pdf](https://arxiv.org/pdf/2108.03444)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper investigates the possibility of creating a machine learning tool that automatically determines the state of mind and emotion of an individual through a questionnaire, without the aid of a human expert. The state of mind and emotion is defined in this work as pertaining to preference, feelings, or opinion that is not based on logic or reason. It is the case when a person gives out an answer to start by saying, "I feel...". The tool is designed to mimic the expertise of a psychologist and is built without any formal knowledge of psychology. The idea is to build the expertise by purely computational methods through thousands of questions collected from users. It is aimed towards possibly diagnosing substance addiction, alcoholism, sexual attraction, HIV status, degree of commitment, activity inclination, etc. First, the paper presents the related literature and classifies them according to data gathering methods. Another classification is created according to preference, emotion, grouping, and rules to achieve a deeper interpretation and better understanding of the state of mind and emotion. Second, the proposed tool is developed using an online addiction questionnaire with 10 questions and 292 respondents. In addition, an initial investigation on the dimension of addiction is presented through the built machine learning model. Machine learning methods, namely, artificial neural network (ANN) and support vector machine (SVM), are used to determine a true or false or degree of state of a respondent.




# 17. Johnson-Lindenstrauss Lemma, Linear and Nonlinear Random Projections,  Random Fourier Features, and Random Kitchen Sinks: Tutorial and Survey

Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, Mark Crowley

- retweets: 16, favorites: 35 (08/11/2021 11:38:45)

- links: [abs](https://arxiv.org/abs/2108.04172) | [pdf](https://arxiv.org/pdf/2108.04172)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.PR](https://arxiv.org/list/math.PR/recent)

This is a tutorial and survey paper on the Johnson-Lindenstrauss (JL) lemma and linear and nonlinear random projections. We start with linear random projection and then justify its correctness by JL lemma and its proof. Then, sparse random projections with $\ell_1$ norm and interpolation norm are introduced. Two main applications of random projection, which are low-rank matrix approximation and approximate nearest neighbor search by random projection onto hypercube, are explained. Random Fourier Features (RFF) and Random Kitchen Sinks (RKS) are explained as methods for nonlinear random projection. Some other methods for nonlinear random projection, including extreme learning machine, randomly weighted neural networks, and ensemble of random projections, are also introduced.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Johnson-Lindenstrauss Lemma, Linear and Nonlinear Random Projections, Random Fourier Features, and Random Kitchen Sinks: Tutorial and Survey. (arXiv:2108.04172v1 [<a href="https://t.co/zjV5HgYw5a">https://t.co/zjV5HgYw5a</a>]) <a href="https://t.co/JW9zQIrHsH">https://t.co/JW9zQIrHsH</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1424909281690898433?ref_src=twsrc%5Etfw">August 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



