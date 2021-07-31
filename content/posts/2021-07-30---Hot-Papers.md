---
title: Hot Papers 2021-07-30
date: 2021-07-31T09:47:56.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-30"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-30"
socialImage: "/media/flying-marine.jpg"

---

# 1. Accurate Throughput Prediction of Basic Blocks on Recent Intel  Microarchitectures

Andreas Abel, Jan Reineke

- retweets: 5824, favorites: 293 (07/31/2021 09:47:56)

- links: [abs](https://arxiv.org/abs/2107.14210) | [pdf](https://arxiv.org/pdf/2107.14210)
- [cs.PF](https://arxiv.org/list/cs.PF/recent)

Tools to predict the throughput of basic blocks on a specific microarchitecture are useful to optimize software performance and to build optimizing compilers. In recent work, several such tools have been proposed. However, the accuracy of their predictions has been shown to be relatively low.   In this paper, we identify the most important factors for these inaccuracies. To a significant degree these inaccuracies are due to elements and parameters of the pipelines of recent CPUs that are not taken into account by previous tools. A primary reason for this is that the necessary details are often undocumented. In this paper, we build more precise models of relevant components by reverse engineering using microbenchmarks. Based on these models, we develop a simulator for predicting the throughput of basic blocks. In addition to predicting the throughput, our simulator also provides insights into how the code is executed.   Our tool supports all Intel Core microarchitecture generations released in the last decade. We evaluate it on an improved version of the BHive benchmark suite. On many recent microarchitectures, its predictions are more accurate than the predictions of state-of-the-art tools by more than an order of magnitude.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Researchers have developed a new simulator to predict the throughput of basic blocks of all Intel Core μarchs released in the last decade, demonstrating to be more accurate than the predictions of state-of-the-art tools by more than an order of magnitude.<a href="https://t.co/83UDBQSchX">https://t.co/83UDBQSchX</a> <a href="https://t.co/U22j4rZjnJ">pic.twitter.com/U22j4rZjnJ</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1421073084732395520?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Deeper Learning By Doing: Integrating Hands-On Research Projects Into a  Machine Learning Course

Sebastian Raschka

- retweets: 570, favorites: 209 (07/31/2021 09:47:56)

- links: [abs](https://arxiv.org/abs/2107.13671) | [pdf](https://arxiv.org/pdf/2107.13671)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Machine learning has seen a vast increase of interest in recent years, along with an abundance of learning resources. While conventional lectures provide students with important information and knowledge, we also believe that additional project-based learning components can motivate students to engage in topics more deeply. In addition to incorporating project-based learning in our courses, we aim to develop project-based learning components aligned with real-world tasks, including experimental design and execution, report writing, oral presentation, and peer-reviewing. This paper describes the organization of our project-based machine learning courses with a particular emphasis on the class project components and shares our resources with instructors who would like to include similar elements in their courses.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share a preprint of my &quot;Deeper Learning By Doing: Integrating Hands-On Research Projects Into an ML Course&quot; paper accepted to the Teaching ML Workshop at ECML 2021. Am hoping that the materials are useful for planning the upcoming Fall semester :) <a href="https://t.co/HLJzYQNS7b">https://t.co/HLJzYQNS7b</a></p>&mdash; Sebastian Raschka (@rasbt) <a href="https://twitter.com/rasbt/status/1420909157813784579?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ar" dir="rtl">جمعة مباركة <br>لمن ينوي تدريس مادة تعلم الآلة او التعلم العميق الفصل القادم ، هذه ورقة حديثة تفصل منهجية مقترحة تطبيقية ممتازه لتدرس هذه الكورس متمحوره على مشروع يسلمه الطلبة على مراحل. <br>اراها مناسبة لجميع المواد التطبيقية في علوم الحاسب <a href="https://t.co/XeZ9Wg1IGU">https://t.co/XeZ9Wg1IGU</a> <a href="https://t.co/0iOrDrUh1S">pic.twitter.com/0iOrDrUh1S</a></p>&mdash; Najwa Alghamdi (@NajwaGhamdi) <a href="https://twitter.com/NajwaGhamdi/status/1421063508666593280?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Rethinking and Improving Relative Position Encoding for Vision  Transformer

Kan Wu, Houwen Peng, Minghao Chen, Jianlong Fu, Hongyang Chao

- retweets: 304, favorites: 83 (07/31/2021 09:47:57)

- links: [abs](https://arxiv.org/abs/2107.14222) | [pdf](https://arxiv.org/pdf/2107.14222)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Relative position encoding (RPE) is important for transformer to capture sequence ordering of input tokens. General efficacy has been proven in natural language processing. However, in computer vision, its efficacy is not well studied and even remains controversial, e.g., whether relative position encoding can work equally well as absolute position? In order to clarify this, we first review existing relative position encoding methods and analyze their pros and cons when applied in vision transformers. We then propose new relative position encoding methods dedicated to 2D images, called image RPE (iRPE). Our methods consider directional relative distance modeling as well as the interactions between queries and relative position embeddings in self-attention mechanism. The proposed iRPE methods are simple and lightweight. They can be easily plugged into transformer blocks. Experiments demonstrate that solely due to the proposed encoding methods, DeiT and DETR obtain up to 1.5% (top-1 Acc) and 1.3% (mAP) stable improvements over their original versions on ImageNet and COCO respectively, without tuning any extra hyperparameters such as learning rate and weight decay. Our ablation and analysis also yield interesting findings, some of which run counter to previous understanding. Code and models are open-sourced at https://github.com/microsoft/Cream/tree/main/iRPE.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking and Improving Relative Position Encoding for Vision Transformer<br>pdf: <a href="https://t.co/AiMbLX2YPc">https://t.co/AiMbLX2YPc</a><br>abs: <a href="https://t.co/2S9xjTgvNr">https://t.co/2S9xjTgvNr</a><br><br>propose new relative position encoding methods dedicated to 2D images, called image RPE <a href="https://t.co/PR2o3WUdIE">pic.twitter.com/PR2o3WUdIE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420909102855819268?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Break, Perturb, Build: Automatic Perturbation of Reasoning Paths through  Question Decomposition

Mor Geva, Tomer Wolfson, Jonathan Berant

- retweets: 306, favorites: 55 (07/31/2021 09:47:57)

- links: [abs](https://arxiv.org/abs/2107.13935) | [pdf](https://arxiv.org/pdf/2107.13935)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Recent efforts to create challenge benchmarks that test the abilities of natural language understanding models have largely depended on human annotations. In this work, we introduce the "Break, Perturb, Build" (BPB) framework for automatic reasoning-oriented perturbation of question-answer pairs. BPB represents a question by decomposing it into the reasoning steps that are required to answer it, symbolically perturbs the decomposition, and then generates new question-answer pairs. We demonstrate the effectiveness of BPB by creating evaluation sets for three reading comprehension (RC) benchmarks, generating thousands of high-quality examples without human intervention. We evaluate a range of RC models on our evaluation sets, which reveals large performance gaps on generated examples compared to the original data. Moreover, symbolic perturbations enable fine-grained analysis of the strengths and limitations of models. Last, augmenting the training data with examples generated by BPB helps close performance gaps, without any drop on the original data distribution.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Automatically generating high-level semantic perturbations of questions is challenging! We introduce BPB, a framework for automatic reasoning-focused perturbations, and use it to generate large and accurate contrast sets for RC.<br>👉 <a href="https://t.co/a8johIgn01">https://t.co/a8johIgn01</a><a href="https://twitter.com/JonathanBerant?ref_src=twsrc%5Etfw">@JonathanBerant</a> Tomer W. <a href="https://t.co/o2uWAzMW6w">pic.twitter.com/o2uWAzMW6w</a></p>&mdash; Mor Geva (@megamor2) <a href="https://twitter.com/megamor2/status/1421079601741242375?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Open-World Entity Segmentation

Lu Qi, Jason Kuen, Yi Wang, Jiuxiang Gu, Hengshuang Zhao, Zhe Lin, Philip Torr, Jiaya Jia

- retweets: 306, favorites: 53 (07/31/2021 09:47:57)

- links: [abs](https://arxiv.org/abs/2107.14228) | [pdf](https://arxiv.org/pdf/2107.14228)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We introduce a new image segmentation task, termed Entity Segmentation (ES) with the aim to segment all visual entities in an image without considering semantic category labels. It has many practical applications in image manipulation/editing where the segmentation mask quality is typically crucial but category labels are less important. In this setting, all semantically-meaningful segments are equally treated as categoryless entities and there is no thing-stuff distinction. Based on our unified entity representation, we propose a center-based entity segmentation framework with two novel modules to improve mask quality. Experimentally, both our new task and framework demonstrate superior advantages as against existing work. In particular, ES enables the following: (1) merging multiple datasets to form a large training set without the need to resolve label conflicts; (2) any model trained on one dataset can generalize exceptionally well to other datasets with unseen domains. Our code is made publicly available at https://github.com/dvlab-research/Entity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Open-World Entity Segmentation<br>pdf: <a href="https://t.co/3fREwaE73c">https://t.co/3fREwaE73c</a><br>abs: <a href="https://t.co/PfRIjiNknI">https://t.co/PfRIjiNknI</a><br>github: <a href="https://t.co/1HbAlwnq8C">https://t.co/1HbAlwnq8C</a> <a href="https://t.co/e8VTvsQqsO">pic.twitter.com/e8VTvsQqsO</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420926568684265476?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods  in Natural Language Processing

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig

- retweets: 235, favorites: 71 (07/31/2021 09:47:57)

- links: [abs](https://arxiv.org/abs/2107.13586) | [pdf](https://arxiv.org/pdf/2107.13586)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper surveys and organizes research works in a new paradigm in natural language processing, which we dub "prompt-based learning". Unlike traditional supervised learning, which trains a model to take in an input x and predict an output y as P(y|x), prompt-based learning is based on language models that model the probability of text directly. To use these models to perform prediction tasks, the original input x is modified using a template into a textual string prompt x' that has some unfilled slots, and then the language model is used to probabilistically fill the unfilled information to obtain a final string x, from which the final output y can be derived. This framework is powerful and attractive for a number of reasons: it allows the language model to be pre-trained on massive amounts of raw text, and by defining a new prompting function the model is able to perform few-shot or even zero-shot learning, adapting to new scenarios with few or no labeled data. In this paper we introduce the basics of this promising paradigm, describe a unified set of mathematical notations that can cover a wide variety of existing work, and organize existing work along several dimensions, e.g.the choice of pre-trained models, prompts, and tuning strategies. To make the field more accessible to interested beginners, we not only make a systematic review of existing works and a highly structured typology of prompt-based concepts, but also release other resources, e.g., a website http://pretrain.nlpedia.ai/ including constantly-updated survey, and paperlist.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pre-train, Prompt, and Predict: A Systematic Survey of<br>Prompting Methods in Natural Language Processing<br>pdf: <a href="https://t.co/k67WAWv3XJ">https://t.co/k67WAWv3XJ</a><br>abs: <a href="https://t.co/fCqO2hgRP0">https://t.co/fCqO2hgRP0</a><br>project page: <a href="https://t.co/ng8OJBLedk">https://t.co/ng8OJBLedk</a> <a href="https://t.co/Xh9Okw1Kjl">pic.twitter.com/Xh9Okw1Kjl</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420908268411670532?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Improved quantum error correction using soft information

Christopher A. Pattison, Michael E. Beverland, Marcus P. da Silva, Nicolas Delfosse

- retweets: 81, favorites: 49 (07/31/2021 09:47:57)

- links: [abs](https://arxiv.org/abs/2107.13589) | [pdf](https://arxiv.org/pdf/2107.13589)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent)

The typical model for measurement noise in quantum error correction is to randomly flip the binary measurement outcome. In experiments, measurements yield much richer information - e.g., continuous current values, discrete photon counts - which is then mapped into binary outcomes by discarding some of this information. In this work, we consider methods to incorporate all of this richer information, typically called soft information, into the decoding of quantum error correction codes, and in particular the surface code. We describe how to modify both the Minimum Weight Perfect Matching and Union-Find decoders to leverage soft information, and demonstrate these soft decoders outperform the standard (hard) decoders that can only access the binary measurement outcomes. Moreover, we observe that the soft decoder achieves a threshold 25\% higher than any hard decoder for phenomenological noise with Gaussian soft measurement outcomes. We also introduce a soft measurement error model with amplitude damping, in which measurement time leads to a trade-off between measurement resolution and additional disturbance of the qubits. Under this model we observe that the performance of the surface code is very sensitive to the choice of the measurement time - for a distance-19 surface code, a five-fold increase in measurement time can lead to a thousand-fold increase in logical error rate. Moreover, the measurement time that minimizes the physical error rate is distinct from the one that minimizes the logical performance, pointing to the benefits of jointly optimizing the physical and quantum error correction layers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new paper on using soft information to improve quantum error correction with Chris Pattison, Michael Beverland, <a href="https://twitter.com/themarcusps?ref_src=twsrc%5Etfw">@themarcusps</a>.<br>That was great to have Chris as an intern in our group last Fall!<a href="https://t.co/Yk38I0MuMM">https://t.co/Yk38I0MuMM</a></p>&mdash; Nicolas Delfosse (@nic_delfosse) <a href="https://twitter.com/nic_delfosse/status/1420910060721512448?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient  Pre-trained Language Models

Yichun Yin, Cheng Chen, Lifeng Shang, Xin Jiang, Xiao Chen, Qun Liu

- retweets: 64, favorites: 28 (07/31/2021 09:47:57)

- links: [abs](https://arxiv.org/abs/2107.13686) | [pdf](https://arxiv.org/pdf/2107.13686)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Pre-trained language models (PLMs) have achieved great success in natural language processing. Most of PLMs follow the default setting of architecture hyper-parameters (e.g., the hidden dimension is a quarter of the intermediate dimension in feed-forward sub-networks) in BERT (Devlin et al., 2019). Few studies have been conducted to explore the design of architecture hyper-parameters in BERT, especially for the more efficient PLMs with tiny sizes, which are essential for practical deployment on resource-constrained devices. In this paper, we adopt the one-shot Neural Architecture Search (NAS) to automatically search architecture hyper-parameters. Specifically, we carefully design the techniques of one-shot learning and the search space to provide an adaptive and efficient development way of tiny PLMs for various latency constraints. We name our method AutoTinyBERT and evaluate its effectiveness on the GLUE and SQuAD benchmarks. The extensive experiments show that our method outperforms both the SOTA search-based baseline (NAS-BERT) and the SOTA distillation-based methods (such as DistilBERT, TinyBERT, MiniLM and MobileBERT). In addition, based on the obtained architectures, we propose a more efficient development method that is even faster than the development of a single PLM.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AutoTinyBERT: Automatic Hyper-parameter Optimization<br>for Efficient Pre-trained Language Models<br>pdf: <a href="https://t.co/5gRbzuYUQD">https://t.co/5gRbzuYUQD</a><br>abs: <a href="https://t.co/XEwLvwGM77">https://t.co/XEwLvwGM77</a><br><br>outperforms both the SOTA search-based baseline (NAS-BERT) and the SOTA distillation-based methods <a href="https://t.co/n5vMz0vGYe">pic.twitter.com/n5vMz0vGYe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420918264000090117?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. ReFormer: The Relational Transformer for Image Captioning

Xuewen Yang, Yingru Liu, Xin Wang

- retweets: 58, favorites: 30 (07/31/2021 09:47:57)

- links: [abs](https://arxiv.org/abs/2107.14178) | [pdf](https://arxiv.org/pdf/2107.14178)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image captioning is shown to be able to achieve a better performance by using scene graphs to represent the relations of objects in the image. The current captioning encoders generally use a Graph Convolutional Net (GCN) to represent the relation information and merge it with the object region features via concatenation or convolution to get the final input for sentence decoding. However, the GCN-based encoders in the existing methods are less effective for captioning due to two reasons. First, using the image captioning as the objective (i.e., Maximum Likelihood Estimation) rather than a relation-centric loss cannot fully explore the potential of the encoder. Second, using a pre-trained model instead of the encoder itself to extract the relationships is not flexible and cannot contribute to the explainability of the model. To improve the quality of image captioning, we propose a novel architecture ReFormer -- a RElational transFORMER to generate features with relation information embedded and to explicitly express the pair-wise relationships between objects in the image. ReFormer incorporates the objective of scene graph generation with that of image captioning using one modified Transformer model. This design allows ReFormer to generate not only better image captions with the bene-fit of extracting strong relational image features, but also scene graphs to explicitly describe the pair-wise relation-ships. Experiments on publicly available datasets show that our model significantly outperforms state-of-the-art methods on image captioning and scene graph generation

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ReFormer: The Relational Transformer for Image Captioning<br>paper: <a href="https://t.co/ch9N78qOWp">https://t.co/ch9N78qOWp</a><br><br>propose the use of ReFormer to integrate the extraction<br>of object relationship and caption generation into the same learning framework that the encoder can be more accurately trained <a href="https://t.co/dvypA7GNOp">pic.twitter.com/dvypA7GNOp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420910745613705217?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. ProtoTransformer: A Meta-Learning Approach to Providing Student Feedback

Mike Wu, Noah Goodman, Chris Piech, Chelsea Finn

- retweets: 49, favorites: 22 (07/31/2021 09:47:58)

- links: [abs](https://arxiv.org/abs/2107.14035) | [pdf](https://arxiv.org/pdf/2107.14035)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

High-quality computer science education is limited by the difficulty of providing instructor feedback to students at scale. While this feedback could in principle be automated, supervised approaches to predicting the correct feedback are bottlenecked by the intractability of annotating large quantities of student code. In this paper, we instead frame the problem of providing feedback as few-shot classification, where a meta-learner adapts to give feedback to student code on a new programming question from just a few examples annotated by instructors. Because data for meta-training is limited, we propose a number of amendments to the typical few-shot learning framework, including task augmentation to create synthetic tasks, and additional side information to build stronger priors about each task. These additions are combined with a transformer architecture to embed discrete sequences (e.g. code) to a prototypical representation of a feedback class label. On a suite of few-shot natural language processing tasks, we match or outperform state-of-the-art performance. Then, on a collection of student solutions to exam questions from an introductory university course, we show that our approach reaches an average precision of 88% on unseen questions, surpassing the 82% precision of teaching assistants. Our approach was successfully deployed to deliver feedback to 16,000 student exam-solutions in a programming course offered by a tier 1 university. This is, to the best of our knowledge, the first successful deployment of a machine learning based feedback to open-ended student code.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ProtoTransformer: A Meta-Learning Approach to Providing Student Feedback<br>pdf: <a href="https://t.co/K78JDjATzs">https://t.co/K78JDjATzs</a><br>abs: <a href="https://t.co/hD3ri0KWwV">https://t.co/hD3ri0KWwV</a><br><br>approach reaches an average precision of 88% on unseen questions, surpassing the 82% precision of teaching assistants <a href="https://t.co/ypWq9HJ0lx">pic.twitter.com/ypWq9HJ0lx</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420910076387340296?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Reuse Cache for Heterogeneous CPU-GPU Systems

Tejas Shah, Bobbi Yogatama, Kyle Roarty, Rami Dahman

- retweets: 20, favorites: 36 (07/31/2021 09:47:58)

- links: [abs](https://arxiv.org/abs/2107.13649) | [pdf](https://arxiv.org/pdf/2107.13649)
- [cs.AR](https://arxiv.org/list/cs.AR/recent)

It is generally observed that the fraction of live lines in shared last-level caches (SLLC) is very small for chip multiprocessors (CMPs). This can be tackled using promotion-based replacement policies like re-reference interval prediction (RRIP) instead of LRU, dead-block predictors, or reuse-based cache allocation schemes. In GPU systems, similar LLC issues are alleviated using various cache bypassing techniques. These issues are worsened in heterogeneous CPU-GPU systems because the two processors have different data access patterns and frequencies. GPUs generally work on streaming data, but have many more threads accessing memory as compared to CPUs. As such, most traditional cache replacement and allocation policies prove ineffective due to the higher number of cache accesses in GPU applications, resulting in higher allocation for GPU cache lines, despite their minimal reuse. In this work, we implement the Reuse Cache approach for heterogeneous CPU-GPU systems. The reuse cache is a decoupled tag/data SLLC which is designed to only store the data that is being accessed more than once. This design is based on the observation that most of the cache lines in the LLC are stored but do not get reused before being replaced. We find that the reuse cache achieves within 0.5% of the IPC gains of a statically partitioned LLC, while decreasing the area cost of the LLC by an average of 40%.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper, researchers implemented the reuse cache approach to heterogeneous CPU-GPU systems using the AMD APU model in the gem5 simulator.<a href="https://t.co/R3VxYOIL1O">https://t.co/R3VxYOIL1O</a> <a href="https://t.co/w3ia2APY8d">pic.twitter.com/w3ia2APY8d</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1421091942402777090?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Competitive Control

Gautam Goel, Babak Hassibi

- retweets: 12, favorites: 43 (07/31/2021 09:47:58)

- links: [abs](https://arxiv.org/abs/2107.13657) | [pdf](https://arxiv.org/pdf/2107.13657)
- [math.OC](https://arxiv.org/list/math.OC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We consider control from the perspective of competitive analysis. Unlike much prior work on learning-based control, which focuses on minimizing regret against the best controller selected in hindsight from some specific class, we focus on designing an online controller which competes against a clairvoyant offline optimal controller. A natural performance metric in this setting is competitive ratio, which is the ratio between the cost incurred by the online controller and the cost incurred by the offline optimal controller. Using operator-theoretic techniques from robust control, we derive a computationally efficient state-space description of the the controller with optimal competitive ratio in both finite-horizon and infinite-horizon settings. We extend competitive control to nonlinear systems using Model Predictive Control (MPC) and present numerical experiments which show that our competitive controller can significantly outperform standard $H_2$ and $H_{\infty}$ controllers in the MPC setting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This looks really cool! (by ⁦<a href="https://twitter.com/gautamcgoel?ref_src=twsrc%5Etfw">@gautamcgoel</a>⁩ and Babak Hassibi)  <a href="https://t.co/xIwPWbLPSK">https://t.co/xIwPWbLPSK</a></p>&mdash; Maxim Raginsky (@mraginsky) <a href="https://twitter.com/mraginsky/status/1420933421526880262?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



