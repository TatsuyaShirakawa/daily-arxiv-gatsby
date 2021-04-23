---
title: Hot Papers 2021-04-22
date: 2021-04-23T15:26:15.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-22"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-22"
socialImage: "/media/flying-marine.jpg"

---

# 1. The NLP Cookbook: Modern Recipes for Transformer based Deep Learning  Architectures

Sushant Singh, Ausif Mahmood

- retweets: 22397, favorites: 160 (04/23/2021 15:26:15)

- links: [abs](https://arxiv.org/abs/2104.10640) | [pdf](https://arxiv.org/pdf/2104.10640)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In recent years, Natural Language Processing (NLP) models have achieved phenomenal success in linguistic and semantic tasks like text classification, machine translation, cognitive dialogue systems, information retrieval via Natural Language Understanding (NLU), and Natural Language Generation (NLG). This feat is primarily attributed due to the seminal Transformer architecture, leading to designs such as BERT, GPT (I, II, III), etc. Although these large-size models have achieved unprecedented performances, they come at high computational costs. Consequently, some of the recent NLP architectures have utilized concepts of transfer learning, pruning, quantization, and knowledge distillation to achieve moderate model sizes while keeping nearly similar performances as achieved by their predecessors. Additionally, to mitigate the data size challenge raised by language models from a knowledge extraction perspective, Knowledge Retrievers have been built to extricate explicit data documents from a large corpus of databases with greater efficiency and accuracy. Recent research has also focused on superior inference by providing efficient attention to longer input sequences. In this paper, we summarize and examine the current state-of-the-art (SOTA) NLP models that have been employed for numerous NLP tasks for optimal performance and efficiency. We provide a detailed understanding and functioning of the different architectures, a taxonomy of NLP designs, comparative evaluations, and future directions in NLP.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔥 The NLP Cookbook: Modern Recipes for<br>Transformer based Deep Learning Architectures<br><br>This is one of the best overviews of Transformer based models for NLP I&#39;ve seen. A good read for ML engineers/researchers wanting to learn about modern NLP techniques.<a href="https://t.co/mYO5htK8Fp">https://t.co/mYO5htK8Fp</a> <a href="https://t.co/HrhUhGbG4O">pic.twitter.com/HrhUhGbG4O</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1385192526865387526?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Carbon Emissions and Large Neural Network Training

David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, Jeff Dean

- retweets: 6257, favorites: 311 (04/23/2021 15:26:16)

- links: [abs](https://arxiv.org/abs/2104.10350) | [pdf](https://arxiv.org/pdf/2104.10350)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

The computation demand for machine learning (ML) has grown rapidly recently, which comes with a number of costs. Estimating the energy cost helps measure its environmental impact and finding greener strategies, yet it is challenging without detailed information. We calculate the energy use and carbon footprint of several recent large models-T5, Meena, GShard, Switch Transformer, and GPT-3-and refine earlier estimates for the neural architecture search that found Evolved Transformer. We highlight the following opportunities to improve energy efficiency and CO2 equivalent emissions (CO2e): Large but sparsely activated DNNs can consume <1/10th the energy of large, dense DNNs without sacrificing accuracy despite using as many or even more parameters. Geographic location matters for ML workload scheduling since the fraction of carbon-free energy and resulting CO2e vary ~5X-10X, even within the same country and the same organization. We are now optimizing where and when large models are trained. Specific datacenter infrastructure matters, as Cloud datacenters can be ~1.4-2X more energy efficient than typical datacenters, and the ML-oriented accelerators inside them can be ~2-5X more effective than off-the-shelf systems. Remarkably, the choice of DNN, datacenter, and processor can reduce the carbon footprint up to ~100-1000X. These large factors also make retroactive estimates of energy cost difficult. To avoid miscalculations, we believe ML papers requiring large computational resources should make energy consumption and CO2e explicit when practical. We are working to be more transparent about energy use and CO2e in our future research. To help reduce the carbon footprint of ML, we believe energy usage and CO2e should be a key metric in evaluating models, and we are collaborating with MLPerf developers to include energy usage during training and inference in this industry standard benchmark.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A thorough study of CO2 emissions linked to large neural net training from Google.<br>It focuses on Google&#39;s NLP models &amp; computing infra.<br>But an equivalent study at FB would yield similar results.<br><br>Patterson &amp; al: &quot;Carbon Emissions and Large NN Training&quot;<a href="https://t.co/9yRgOy8KsU">https://t.co/9yRgOy8KsU</a></p>&mdash; Yann LeCun (@ylecun) <a href="https://twitter.com/ylecun/status/1385297474080985097?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr"><a href="https://t.co/yRgSGpdcKg">https://t.co/yRgSGpdcKg</a> -- a detailed study of CO2 emission in large models from Google</p>&mdash; Ilya Sutskever (@ilyasut) <a href="https://twitter.com/ilyasut/status/1385309813970661377?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pretty wild Google put out a long paper on &quot;Carbon Emissions and Large Neural Network Training&quot;, with a focus on LMs, and didn&#39;t cite <a href="https://twitter.com/emilymbender?ref_src=twsrc%5Etfw">@emilymbender</a> / <a href="https://twitter.com/timnitGebru?ref_src=twsrc%5Etfw">@timnitGebru</a> et al. 2021, from what I can tell - <a href="https://t.co/J9P8hKeJ5M">https://t.co/J9P8hKeJ5M</a> How hard would it have been?</p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1385040091194613762?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Contingencies from Observations: Tractable Contingency Planning with  Learned Behavior Models

Nicholas Rhinehart, Jeff He, Charles Packer, Matthew A. Wright, Rowan McAllister, Joseph E. Gonzalez, Sergey Levine

- retweets: 888, favorites: 239 (04/23/2021 15:26:17)

- links: [abs](https://arxiv.org/abs/2104.10558) | [pdf](https://arxiv.org/pdf/2104.10558)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Humans have a remarkable ability to make decisions by accurately reasoning about future events, including the future behaviors and states of mind of other agents. Consider driving a car through a busy intersection: it is necessary to reason about the physics of the vehicle, the intentions of other drivers, and their beliefs about your own intentions. If you signal a turn, another driver might yield to you, or if you enter the passing lane, another driver might decelerate to give you room to merge in front. Competent drivers must plan how they can safely react to a variety of potential future behaviors of other agents before they make their next move. This requires contingency planning: explicitly planning a set of conditional actions that depend on the stochastic outcome of future events. In this work, we develop a general-purpose contingency planner that is learned end-to-end using high-dimensional scene observations and low-dimensional behavioral observations. We use a conditional autoregressive flow model to create a compact contingency planning space, and show how this model can tractably learn contingencies from behavioral observations. We developed a closed-loop control benchmark of realistic multi-agent scenarios in a driving simulator (CARLA), on which we compare our method to various noncontingent methods that reason about multi-agent future behavior, including several state-of-the-art deep learning-based planning approaches. We illustrate that these noncontingent planning methods fundamentally fail on this benchmark, and find that our deep contingency planning method achieves significantly superior performance. Code to run our benchmark and reproduce our results is available at https://sites.google.com/view/contingency-planning

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Planning in multi-agent settings requires considering uncertain future behaviors of other agents. So we need contingent plans. In our recent work, we tackle this by learning models where open-loop latent space plans lead to closed loop control<a href="https://t.co/W7k5yegTIM">https://t.co/W7k5yegTIM</a><br>🧵&gt; <a href="https://t.co/Jy1yS9DDFv">pic.twitter.com/Jy1yS9DDFv</a></p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1385069792340570112?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contingencies from Observations: Tractable Contingency Planning with Learned Behavior Models<br>pdf: <a href="https://t.co/vNBBx4TxEM">https://t.co/vNBBx4TxEM</a><br>abs: <a href="https://t.co/i0izOdzsbA">https://t.co/i0izOdzsbA</a><br>project page: <a href="https://t.co/JNHLVZIJ1g">https://t.co/JNHLVZIJ1g</a><br>github: <a href="https://t.co/Etx5H11Vog">https://t.co/Etx5H11Vog</a> <a href="https://t.co/G9nwukWbWV">pic.twitter.com/G9nwukWbWV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385046696732790785?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. PP-YOLOv2: A Practical Object Detector

Xin Huang, Xinxin Wang, Wenyu Lv, Xiaying Bai, Xiang Long, Kaipeng Deng, Qingqing Dang, Shumin Han, Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma, Osamu Yoshie

- retweets: 156, favorites: 56 (04/23/2021 15:26:17)

- links: [abs](https://arxiv.org/abs/2104.10419) | [pdf](https://arxiv.org/pdf/2104.10419)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Being effective and efficient is essential to an object detector for practical use. To meet these two concerns, we comprehensively evaluate a collection of existing refinements to improve the performance of PP-YOLO while almost keep the infer time unchanged. This paper will analyze a collection of refinements and empirically evaluate their impact on the final model performance through incremental ablation study. Things we tried that didn't work will also be discussed. By combining multiple effective refinements, we boost PP-YOLO's performance from 45.9% mAP to 49.5% mAP on COCO2017 test-dev. Since a significant margin of performance has been made, we present PP-YOLOv2. In terms of speed, PP-YOLOv2 runs in 68.9FPS at 640x640 input size. Paddle inference engine with TensorRT, FP16-precision, and batch size = 1 further improves PP-YOLOv2's infer speed, which achieves 106.5 FPS. Such a performance surpasses existing object detectors with roughly the same amount of parameters (i.e., YOLOv4-CSP, YOLOv5l). Besides, PP-YOLOv2 with ResNet101 achieves 50.3% mAP on COCO2017 test-dev. Source code is at https://github.com/PaddlePaddle/PaddleDetection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PP-YOLOv2: A Practical Object Detector<br>pdf: <a href="https://t.co/PvF8XZOzqV">https://t.co/PvF8XZOzqV</a><br>abs: <a href="https://t.co/E9k1VbjQJ5">https://t.co/E9k1VbjQJ5</a> <a href="https://t.co/PIj1FPRNcz">pic.twitter.com/PIj1FPRNcz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385060912508375040?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Identify, Align, and Integrate: Matching Knowledge Graphs to Commonsense  Reasoning Tasks

Lisa Bauer, Mohit Bansal

- retweets: 104, favorites: 51 (04/23/2021 15:26:17)

- links: [abs](https://arxiv.org/abs/2104.10193) | [pdf](https://arxiv.org/pdf/2104.10193)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Integrating external knowledge into commonsense reasoning tasks has shown progress in resolving some, but not all, knowledge gaps in these tasks. For knowledge integration to yield peak performance, it is critical to select a knowledge graph (KG) that is well-aligned with the given task's objective. We present an approach to assess how well a candidate KG can correctly identify and accurately fill in gaps of reasoning for a task, which we call KG-to-task match. We show this KG-to-task match in 3 phases: knowledge-task identification, knowledge-task alignment, and knowledge-task integration. We also analyze our transformer-based KG-to-task models via commonsense probes to measure how much knowledge is captured in these models before and after KG integration. Empirically, we investigate KG matches for the SocialIQA (SIQA) (Sap et al., 2019b), Physical IQA (PIQA) (Bisk et al., 2020), and MCScript2.0 (Ostermann et al., 2019) datasets with 3 diverse KGs: ATOMIC (Sap et al., 2019a), ConceptNet (Speer et al., 2017), and an automatically constructed instructional KG based on WikiHow (Koupaee and Wang, 2018). With our methods we are able to demonstrate that ATOMIC, an event-inference focused KG, is the best match for SIQA and MCScript2.0, and that the taxonomic ConceptNet and WikiHow-based KGs are the best matches for PIQA across all 3 analysis phases. We verify our methods and findings with human evaluation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We&#39;ll present our <a href="https://twitter.com/hashtag/EACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#EACL2021</a> paper &quot;Identify, Align, Integrate: Matching Knowledge Graphs to Commonsense Reasoning Tasks&quot; on Apr23!🙂 <br>Join us: Oral/QA session Zoom8B 8-9am, Gather3A 9-11am EDT<br>cc <a href="https://twitter.com/mohitban47?ref_src=twsrc%5Etfw">@mohitban47</a><a href="https://t.co/v9LZmhR1s3">https://t.co/v9LZmhR1s3</a><a href="https://t.co/IiarBajALn">https://t.co/IiarBajALn</a><a href="https://t.co/tFxePl5AJl">https://t.co/tFxePl5AJl</a> <a href="https://t.co/gpioT5xlPR">pic.twitter.com/gpioT5xlPR</a></p>&mdash; Lisa Bauer (@lbauer119) <a href="https://twitter.com/lbauer119/status/1385242417012613123?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Should we Stop Training More Monolingual Models, and Simply Use Machine  Translation Instead?

Tim Isbister, Fredrik Carlsson, Magnus Sahlgren

- retweets: 49, favorites: 66 (04/23/2021 15:26:17)

- links: [abs](https://arxiv.org/abs/2104.10441) | [pdf](https://arxiv.org/pdf/2104.10441)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Most work in NLP makes the assumption that it is desirable to develop solutions in the native language in question. There is consequently a strong trend towards building native language models even for low-resource languages. This paper questions this development, and explores the idea of simply translating the data into English, thereby enabling the use of pretrained, and large-scale, English language models. We demonstrate empirically that a large English language model coupled with modern machine translation outperforms native language models in most Scandinavian languages. The exception to this is Finnish, which we assume is due to inferior translation quality. Our results suggest that machine translation is a mature technology, which raises a serious counter-argument for training native language models for low-resource languages. This paper therefore strives to make a provocative but important point. As English language models are improving at an unprecedented pace, which in turn improves machine translation, it is from an empirical and environmental stand-point more effective to translate data from low-resource languages into English, than to build language models for such languages.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Should we Stop Training More Monolingual Models, and Simply Use Machine Translation Instead?,&quot; Isbister et al.: <a href="https://t.co/ggv26936ZD">https://t.co/ggv26936ZD</a><br><br>&quot;We demonstrate empirically that a large English LM coupled with modern MT outperforms native LMs in most Scandinavian languages.&quot; 🤔🤔🤔</p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1385074626489851904?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Hierarchical Cross-Modal Agent for Robotics Vision-and-Language  Navigation

Muhammad Zubair Irshad, Chih-Yao Ma, Zsolt Kira

- retweets: 56, favorites: 32 (04/23/2021 15:26:18)

- links: [abs](https://arxiv.org/abs/2104.10674) | [pdf](https://arxiv.org/pdf/2104.10674)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

Deep Learning has revolutionized our ability to solve complex problems such as Vision-and-Language Navigation (VLN). This task requires the agent to navigate to a goal purely based on visual sensory inputs given natural language instructions. However, prior works formulate the problem as a navigation graph with a discrete action space. In this work, we lift the agent off the navigation graph and propose a more complex VLN setting in continuous 3D reconstructed environments. Our proposed setting, Robo-VLN, more closely mimics the challenges of real world navigation. Robo-VLN tasks have longer trajectory lengths, continuous action spaces, and challenges such as obstacles. We provide a suite of baselines inspired by state-of-the-art works in discrete VLN and show that they are less effective at this task. We further propose that decomposing the task into specialized high- and low-level policies can more effectively tackle this task. With extensive experiments, we show that by using layered decision making, modularized training, and decoupling reasoning and imitation, our proposed Hierarchical Cross-Modal (HCM) agent outperforms existing baselines in all key metrics and sets a new benchmark for Robo-VLN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical Cross-Modal Agent for Robotics Vision-and-Language Navigation<br>pdf: <a href="https://t.co/S52mmmnwjs">https://t.co/S52mmmnwjs</a><br>abs: <a href="https://t.co/9mC4xsqncJ">https://t.co/9mC4xsqncJ</a><br>github: <a href="https://t.co/WD3bT5tfX1">https://t.co/WD3bT5tfX1</a><br>project page: <a href="https://t.co/ETdE9Muxqa">https://t.co/ETdE9Muxqa</a> <a href="https://t.co/nbCjhUqhZr">pic.twitter.com/nbCjhUqhZr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1385065584266784772?ref_src=twsrc%5Etfw">April 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



