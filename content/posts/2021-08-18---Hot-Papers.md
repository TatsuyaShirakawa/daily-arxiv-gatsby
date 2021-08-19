---
title: Hot Papers 2021-08-18
date: 2021-08-19T16:18:36.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Program Synthesis with Large Language Models

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, Charles Sutton

- retweets: 21435, favorites: 4 (08/19/2021 16:18:36)

- links: [abs](https://arxiv.org/abs/2108.07732) | [pdf](https://arxiv.org/pdf/2108.07732)
- [cs.PL](https://arxiv.org/list/cs.PL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper explores the limits of the current generation of large language models for program synthesis in general purpose programming languages. We evaluate a collection of such models (with between 244M and 137B parameters) on two new benchmarks, MBPP and MathQA-Python, in both the few-shot and fine-tuning regimes. Our benchmarks are designed to measure the ability of these models to synthesize short Python programs from natural language descriptions. The Mostly Basic Programming Problems (MBPP) dataset contains 974 programming tasks, designed to be solvable by entry-level programmers. The MathQA-Python dataset, a Python version of the MathQA benchmark, contains 23914 problems that evaluate the ability of the models to synthesize code from more complex text. On both datasets, we find that synthesis performance scales log-linearly with model size. Our largest models, even without finetuning on a code dataset, can synthesize solutions to 59.6 percent of the problems from MBPP using few-shot learning with a well-designed prompt. Fine-tuning on a held-out portion of the dataset improves performance by about 10 percentage points across most model sizes. On the MathQA-Python dataset, the largest fine-tuned model achieves 83.8 percent accuracy. Going further, we study the model's ability to engage in dialog about code, incorporating human feedback to improve its solutions. We find that natural language feedback from a human halves the error rate compared to the model's initial prediction. Additionally, we conduct an error analysis to shed light on where these models fall short and what types of programs are most difficult to generate. Finally, we explore the semantic grounding of these models by fine-tuning them to predict the results of program execution. We find that even our best models are generally unable to predict the output of a program given a specific input.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper! <a href="https://t.co/4oiC9A9SuI">https://t.co/4oiC9A9SuI</a><br>We use big language models to synthesize computer programs, execute programs, solve math problems, and dialog with humans to iteratively refine code.<br>The models can solve 60% and 81% of the programming and math problems, respectively. A thread: <a href="https://t.co/EDGQDVyzrq">pic.twitter.com/EDGQDVyzrq</a></p>&mdash; augustus odena (@gstsdn) <a href="https://twitter.com/gstsdn/status/1427794393373626368?ref_src=twsrc%5Etfw">August 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Mitigating harm in language models with conditional-likelihood  filtration

Helen Ngo, Cooper Raterink, João G.M. Araújo, Ivan Zhang, Carol Chen, Adrien Morisot, Nicholas Frosst

- retweets: 246, favorites: 120 (08/19/2021 16:18:36)

- links: [abs](https://arxiv.org/abs/2108.07790) | [pdf](https://arxiv.org/pdf/2108.07790)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Language models trained on large-scale unfiltered datasets curated from the open web acquire systemic biases, prejudices, and harmful views from their training data. We present a methodology for programmatically identifying and removing harmful text from web-scale datasets. A pretrained language model is used to calculate the log-likelihood of researcher-written trigger phrases conditioned on a specific document, which is used to identify and filter documents from the dataset. We demonstrate that models trained on this filtered dataset exhibit lower propensity to generate harmful text, with a marginal decrease in performance on standard language modeling benchmarks compared to unfiltered baselines. We provide a partial explanation for this performance gap by surfacing examples of hate speech and other undesirable content from standard language modeling benchmarks. Finally, we discuss the generalization of this method and how trigger phrases which reflect specific values can be used by researchers to build language models which are more closely aligned with their values.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share a <a href="https://twitter.com/CohereAI?ref_src=twsrc%5Etfw">@CohereAI</a>  preprint on detoxifying language models!<br><br>We use a language model to find hateful text by calculating the conditional likelihood of curated trigger phrases and filtering the training set! This results in nicer models :)<a href="https://t.co/Jl58TUoOi9">https://t.co/Jl58TUoOi9</a> <a href="https://t.co/DuD4mLXnoH">pic.twitter.com/DuD4mLXnoH</a></p>&mdash; nick frosst (@nickfrosst) <a href="https://twitter.com/nickfrosst/status/1428016832997429252?ref_src=twsrc%5Etfw">August 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Modeling Protein Using Large-scale Pretrain Language Model

Yijia Xiao, Jiezhong Qiu, Ziang Li, Chang-Yu Hsieh, Jie Tang

- retweets: 144, favorites: 59 (08/19/2021 16:18:37)

- links: [abs](https://arxiv.org/abs/2108.07435) | [pdf](https://arxiv.org/pdf/2108.07435)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent)

Protein is linked to almost every life process. Therefore, analyzing the biological structure and property of protein sequences is critical to the exploration of life, as well as disease detection and drug discovery. Traditional protein analysis methods tend to be labor-intensive and time-consuming. The emergence of deep learning models makes modeling data patterns in large quantities of data possible. Interdisciplinary researchers have begun to leverage deep learning methods to model large biological datasets, e.g. using long short-term memory and convolutional neural network for protein sequence classification. After millions of years of evolution, evolutionary information is encoded in protein sequences. Inspired by the similarity between natural language and protein sequences, we use large-scale language models to model evolutionary-scale protein sequences, encoding protein biology information in representation. Significant improvements are observed in both token-level and sequence-level tasks, demonstrating that our large-scale model can accurately capture evolution information from pretraining on evolutionary-scale individual sequences. Our code and model are available at https://github.com/THUDM/ProteinLM.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Modeling Protein Using Large-scale Pretrain Language Model<br>pdf: <a href="https://t.co/Eni6b9CV5v">https://t.co/Eni6b9CV5v</a><br>abs: <a href="https://t.co/XuwFcqjgbi">https://t.co/XuwFcqjgbi</a><br>github: <a href="https://t.co/2KH9kHjF3R">https://t.co/2KH9kHjF3R</a> <a href="https://t.co/xKUld7ksqb">pic.twitter.com/xKUld7ksqb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427827996895891461?ref_src=twsrc%5Etfw">August 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Generative Relation Linking for Question Answering over Knowledge Bases

Gaetano Rossiello, Nandana Mihindukulasooriya, Ibrahim Abdelaziz, Mihaela Bornea, Alfio Gliozzo, Tahira Naseem, Pavan Kapanipathi

- retweets: 56, favorites: 18 (08/19/2021 16:18:37)

- links: [abs](https://arxiv.org/abs/2108.07337) | [pdf](https://arxiv.org/pdf/2108.07337)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Relation linking is essential to enable question answering over knowledge bases. Although there are various efforts to improve relation linking performance, the current state-of-the-art methods do not achieve optimal results, therefore, negatively impacting the overall end-to-end question answering performance. In this work, we propose a novel approach for relation linking framing it as a generative problem facilitating the use of pre-trained sequence-to-sequence models. We extend such sequence-to-sequence models with the idea of infusing structured data from the target knowledge base, primarily to enable these models to handle the nuances of the knowledge base. Moreover, we train the model with the aim to generate a structured output consisting of a list of argument-relation pairs, enabling a knowledge validation step. We compared our method against the existing relation linking systems on four different datasets derived from DBpedia and Wikidata. Our method reports large improvements over the state-of-the-art while using a much simpler model that can be easily adapted to different knowledge bases.




# 5. TOOD: Task-aligned One-stage Object Detection

Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang

- retweets: 18, favorites: 42 (08/19/2021 16:18:37)

- links: [abs](https://arxiv.org/abs/2108.07755) | [pdf](https://arxiv.org/pdf/2108.07755)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

One-stage object detection is commonly implemented by optimizing two sub-tasks: object classification and localization, using heads with two parallel branches, which might lead to a certain level of spatial misalignment in predictions between the two tasks. In this work, we propose a Task-aligned One-stage Object Detection (TOOD) that explicitly aligns the two tasks in a learning-based manner. First, we design a novel Task-aligned Head (T-Head) which offers a better balance between learning task-interactive and task-specific features, as well as a greater flexibility to learn the alignment via a task-aligned predictor. Second, we propose Task Alignment Learning (TAL) to explicitly pull closer (or even unify) the optimal anchors for the two tasks during training via a designed sample assignment scheme and a task-aligned loss. Extensive experiments are conducted on MS-COCO, where TOOD achieves a 51.1 AP at single-model single-scale testing. This surpasses the recent one-stage detectors by a large margin, such as ATSS (47.7 AP), GFL (48.2 AP), and PAA (49.0 AP), with fewer parameters and FLOPs. Qualitative results also demonstrate the effectiveness of TOOD for better aligning the tasks of object classification and localization. Code is available at https://github.com/fcjian/TOOD.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">1ステージ物体検出における、ClassificationとLocalizationの位置ずれを解消<br><br>TOOD: Task-aligned One-stage Object Detection<br>(arXiv 2021)<a href="https://t.co/nHfwhnhelB">https://t.co/nHfwhnhelB</a> <a href="https://t.co/QqHfFn48in">pic.twitter.com/QqHfFn48in</a></p>&mdash; Machi (@MachiK06) <a href="https://twitter.com/MachiK06/status/1428153917116878853?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Panoramic Learning with A Standardized Machine Learning Formalism

Zhiting Hu, Eric P. Xing

- retweets: 25, favorites: 30 (08/19/2021 16:18:37)

- links: [abs](https://arxiv.org/abs/2108.07783) | [pdf](https://arxiv.org/pdf/2108.07783)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Machine Learning (ML) is about computational methods that enable machines to learn concepts from experiences. In handling a wide variety of experiences ranging from data instances, knowledge, constraints, to rewards, adversaries, and lifelong interplay in an ever-growing spectrum of tasks, contemporary ML/AI research has resulted in a multitude of learning paradigms and methodologies. Despite the continual progresses on all different fronts, the disparate narrowly-focused methods also make standardized, composable, and reusable development of learning solutions difficult, and make it costly if possible to build AI agents that panoramically learn from all types of experiences. This paper presents a standardized ML formalism, in particular a standard equation of the learning objective, that offers a unifying understanding of diverse ML algorithms, making them special cases due to different choices of modeling components. The framework also provides guidance for mechanic design of new ML solutions, and serves as a promising vehicle towards panoramic learning with all experiences.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The initial draft of the &quot;standard model&quot; that I have recently talked about is now online: <a href="https://t.co/WJzh7aF9Mh">https://t.co/WJzh7aF9Mh</a>. We present a unified mathematical formulation for learning with all experiences and a holistic understanding of algorithms and paradigms in ML. Feedback are welcome</p>&mdash; Eric Xing (@ericxing) <a href="https://twitter.com/ericxing/status/1427832735029207045?ref_src=twsrc%5Etfw">August 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



