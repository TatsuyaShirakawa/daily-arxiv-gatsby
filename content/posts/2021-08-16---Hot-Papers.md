---
title: Hot Papers 2021-08-16
date: 2021-08-17T03:47:56.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-16"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-16"
socialImage: "/media/flying-marine.jpg"

---

# 1. The Forgotten Threat of Voltage Glitching: A Case Study on Nvidia Tegra  X2 SoCs

Otto Bittner, Thilo Krachenfels, Andreas Galauner, Jean-Pierre Seifert

- retweets: 2438, favorites: 201 (08/17/2021 03:47:56)

- links: [abs](https://arxiv.org/abs/2108.06131) | [pdf](https://arxiv.org/pdf/2108.06131)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

Voltage fault injection (FI) is a well-known attack technique that can be used to force faulty behavior in processors during their operation. Glitching the supply voltage can cause data value corruption, skip security checks, or enable protected code paths. At the same time, modern systems on a chip (SoCs) are used in security-critical applications, such as self-driving cars and autonomous machines. Since these embedded devices are often physically accessible by attackers, vendors must consider device tampering in their threat models. However, while the threat of voltage FI is known since the early 2000s, it seems as if vendors still forget to integrate countermeasures. This work shows how the entire boot security of an Nvidia SoC, used in Tesla's autopilot and Mercedes-Benz's infotainment system, can be circumvented using voltage FI. We uncover a hidden bootloader that is only available to the manufacturer for testing purposes and disabled by fuses in shipped products. We demonstrate how to re-enable this bootloader using FI to gain code execution with the highest privileges, enabling us to extract the bootloader's firmware and decryption keys used in later boot stages. Using a hardware implant, an adversary might misuse the hidden bootloader to bypass trusted code execution even during the system's regular operation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I did a thing together with <a href="https://twitter.com/derpst3b?ref_src=twsrc%5Etfw">@derpst3b</a> and <a href="https://twitter.com/_tkrachenfels?ref_src=twsrc%5Etfw">@_tkrachenfels</a>: We successfully glitched the early boot process of the Nvidia Tegra X2 SoC and managed to dump every single secret crypto key and the full boot ROM. You can read our paper here: <a href="https://t.co/JkKJefZrSh">https://t.co/JkKJefZrSh</a></p>&mdash; Andy (@G33KatWork) <a href="https://twitter.com/G33KatWork/status/1427186000934154240?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Curriculum Learning: A Regularization Method for Efficient and Stable  Billion-Scale GPT Model Pre-Training

Conglong Li, Minjia Zhang, Yuxiong He

- retweets: 759, favorites: 214 (08/17/2021 03:47:56)

- links: [abs](https://arxiv.org/abs/2108.06084) | [pdf](https://arxiv.org/pdf/2108.06084)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Recent works have demonstrated great success in training high-capacity autoregressive language models (GPT, GPT-2, GPT-3) on a huge amount of unlabeled text corpus for text generation. Despite showing great results, this generates two training efficiency challenges. First, training large corpora can be extremely timing consuming, and how to present training samples to the model to improve the token-wise convergence speed remains a challenging and open question. Second, many of these large models have to be trained with hundreds or even thousands of processors using data-parallelism with a very large batch size. Despite of its better compute efficiency, it has been observed that large-batch training often runs into training instability issue or converges to solutions with bad generalization performance. To overcome these two challenges, we present a study of a curriculum learning based approach, which helps improves the pre-training convergence speed of autoregressive models. More importantly, we find that curriculum learning, as a regularization method, exerts a gradient variance reduction effect and enables to train autoregressive models with much larger batch sizes and learning rates without training instability, further improving the training speed. Our evaluations demonstrate that curriculum learning enables training GPT-2 models (with up to 1.5B parameters) with 8x larger batch size and 4x larger learning rate, whereas the baseline approach struggles with training divergence. To achieve the same validation perplexity targets during pre-training, curriculum learning reduces the required number of tokens and wall clock time by up to 59% and 54%, respectively. To achieve the same or better zero-shot WikiText-103/LAMBADA evaluation results at the end of pre-training, curriculum learning reduces the required number of tokens and wall clock time by up to 13% and 61%, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Curriculum Learning: A Regularization Method for Efficient and Stable Billion-Scale GPT Model Pre-Training<br><br>Curriculum learning saves ~2x computes of the training of GPT-2 models to reach the same perplexity.<a href="https://t.co/wi3HUkWwhE">https://t.co/wi3HUkWwhE</a> <a href="https://t.co/b8ChvT1T22">pic.twitter.com/b8ChvT1T22</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1427070420042268680?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Curriculum Learning: A Regularization Method for Efficient and Stable Billion-Scale GPT Model Pre-Training<br>paper: <a href="https://t.co/E7pAD7qDbR">https://t.co/E7pAD7qDbR</a><br><br>is able to reduce up to 61% training time while still achieve the same validation perplexity and zero-shot evaluation results <a href="https://t.co/4putEY9zPY">pic.twitter.com/4putEY9zPY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427066874550960136?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. W2v-BERT: Combining Contrastive Learning and Masked Language Modeling  for Self-Supervised Speech Pre-Training

Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, Yonghui Wu

- retweets: 346, favorites: 72 (08/17/2021 03:47:57)

- links: [abs](https://arxiv.org/abs/2108.06209) | [pdf](https://arxiv.org/pdf/2108.06209)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Motivated by the success of masked language modeling~(MLM) in pre-training natural language processing models, we propose w2v-BERT that explores MLM for self-supervised speech representation learning. w2v-BERT is a framework that combines contrastive learning and MLM, where the former trains the model to discretize input continuous speech signals into a finite set of discriminative speech tokens, and the latter trains the model to learn contextualized speech representations via solving a masked prediction task consuming the discretized tokens. In contrast to existing MLM-based speech pre-training frameworks such as HuBERT, which relies on an iterative re-clustering and re-training process, or vq-wav2vec, which concatenates two separately trained modules, w2v-BERT can be optimized in an end-to-end fashion by solving the two self-supervised tasks~(the contrastive task and MLM) simultaneously. Our experiments show that w2v-BERT achieves competitive results compared to current state-of-the-art pre-trained models on the LibriSpeech benchmarks when using the Libri-Light~60k corpus as the unsupervised data. In particular, when compared to published models such as conformer-based wav2vec~2.0 and HuBERT, our model shows~5\% to~10\% relative WER reduction on the test-clean and test-other subsets. When applied to the Google's Voice Search traffic dataset, w2v-BERT outperforms our internal conformer-based wav2vec~2.0 by more than~30\% relatively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training<br>abs: <a href="https://t.co/3HXsEMjQdd">https://t.co/3HXsEMjQdd</a><br><br>applied to Google’s Voice Search traffic dataset, w2v-BERT outperforms internal conformer-based wav2vec 2.0 by more than 30% relatively <a href="https://t.co/XcFIAd1SZe">pic.twitter.com/XcFIAd1SZe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427067735595495425?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. MUSIQ: Multi-scale Image Quality Transformer

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang

- retweets: 306, favorites: 68 (08/17/2021 03:47:57)

- links: [abs](https://arxiv.org/abs/2108.05997) | [pdf](https://arxiv.org/pdf/2108.05997)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image quality assessment (IQA) is an important research topic for understanding and improving visual experience. The current state-of-the-art IQA methods are based on convolutional neural networks (CNNs). The performance of CNN-based models is often compromised by the fixed shape constraint in batch training. To accommodate this, the input images are usually resized and cropped to a fixed shape, causing image quality degradation. To address this, we design a multi-scale image quality Transformer (MUSIQ) to process native resolution images with varying sizes and aspect ratios. With a multi-scale image representation, our proposed method can capture image quality at different granularities. Furthermore, a novel hash-based 2D spatial embedding and a scale embedding is proposed to support the positional embedding in the multi-scale representation. Experimental results verify that our method can achieve state-of-the-art performance on multiple large scale IQA datasets such as PaQ-2-PiQ, SPAQ and KonIQ-10k.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MUSIQ: Multi-scale Image Quality Transformer<br>pdf: <a href="https://t.co/LKoJ05canf">https://t.co/LKoJ05canf</a><br>abs: <a href="https://t.co/gUmNW7aX6Q">https://t.co/gUmNW7aX6Q</a><br><br>a multi-scale image quality Transformer, which can handle full-size image input with varying resolutions and aspect ratios <a href="https://t.co/Kf61upF7jC">pic.twitter.com/Kf61upF7jC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427068707575148550?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Co-GAIL: Learning Diverse Strategies for Human-Robot Collaboration

Chen Wang, Claudia Pérez-D'Arpino, Danfei Xu, Li Fei-Fei, C. Karen Liu, Silvio Savarese

- retweets: 110, favorites: 36 (08/17/2021 03:47:57)

- links: [abs](https://arxiv.org/abs/2108.06038) | [pdf](https://arxiv.org/pdf/2108.06038)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We present a method for learning a human-robot collaboration policy from human-human collaboration demonstrations. An effective robot assistant must learn to handle diverse human behaviors shown in the demonstrations and be robust when the humans adjust their strategies during online task execution. Our method co-optimizes a human policy and a robot policy in an interactive learning process: the human policy learns to generate diverse and plausible collaborative behaviors from demonstrations while the robot policy learns to assist by estimating the unobserved latent strategy of its human collaborator. Across a 2D strategy game, a human-robot handover task, and a multi-step collaborative manipulation task, our method outperforms the alternatives in both simulated evaluations and when executing the tasks with a real human operator in-the-loop. Supplementary materials and videos at https://sites.google.com/view/co-gail-web/home

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Co-GAIL: Learning Diverse Strategies for Human-Robot Collaboration<br>pdf: <a href="https://t.co/xPHK2Mr6FX">https://t.co/xPHK2Mr6FX</a><br>abs: <a href="https://t.co/av3zU5pgAN">https://t.co/av3zU5pgAN</a><br>project page: <a href="https://t.co/g2KRIDArFz">https://t.co/g2KRIDArFz</a> <a href="https://t.co/bSY4t3cQHt">pic.twitter.com/bSY4t3cQHt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427075967030374403?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Enhancing audio quality for expressive Neural Text-to-Speech

Abdelhamid Ezzerg, Adam Gabrys, Bartosz Putrycz, Daniel Korzekwa, Daniel Saez-Trigueros, David McHardy, Kamil Pokora, Jakub Lachowicz, Jaime Lorenzo-Trueba, Viacheslav Klimkov

- retweets: 81, favorites: 24 (08/17/2021 03:47:57)

- links: [abs](https://arxiv.org/abs/2108.06270) | [pdf](https://arxiv.org/pdf/2108.06270)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Artificial speech synthesis has made a great leap in terms of naturalness as recent Text-to-Speech (TTS) systems are capable of producing speech with similar quality to human recordings. However, not all speaking styles are easy to model: highly expressive voices are still challenging even to recent TTS architectures since there seems to be a trade-off between expressiveness in a generated audio and its signal quality. In this paper, we present a set of techniques that can be leveraged to enhance the signal quality of a highly-expressive voice without the use of additional data. The proposed techniques include: tuning the autoregressive loop's granularity during training; using Generative Adversarial Networks in acoustic modelling; and the use of Variational Auto-Encoders in both the acoustic model and the neural vocoder. We show that, when combined, these techniques greatly closed the gap in perceived naturalness between the baseline system and recordings by 39% in terms of MUSHRA scores for an expressive celebrity voice.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Enhancing audio quality for expressive Neural Text-to-Speech<br>pdf: <a href="https://t.co/Eu47qicr90">https://t.co/Eu47qicr90</a><br>abs: <a href="https://t.co/J6xWBQVaGG">https://t.co/J6xWBQVaGG</a> <a href="https://t.co/y7gMF5WhlA">pic.twitter.com/y7gMF5WhlA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427070136738013187?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Low-Resource Adaptation of Open-Domain Generative Chatbots

Greyson Gerhard-Young, Raviteja Anantha, Srinivas Chappidi, Björn Hoffmeister

- retweets: 56, favorites: 31 (08/17/2021 03:47:57)

- links: [abs](https://arxiv.org/abs/2108.06329) | [pdf](https://arxiv.org/pdf/2108.06329)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent work building open-domain chatbots has demonstrated that increasing model size improves performance. On the other hand, latency and connectivity considerations dictate the move of digital assistants on the device. Giving a digital assistant like Siri, Alexa, or Google Assistant the ability to discuss just about anything leads to the need for reducing the chatbot model size such that it fits on the user's device. We demonstrate that low parameter models can simultaneously retain their general knowledge conversational abilities while improving in a specific domain. Additionally, we propose a generic framework that accounts for variety in question types, tracks reference throughout multi-turn conversations, and removes inconsistent and potentially toxic responses. Our framework seamlessly transitions between chatting and performing transactional tasks, which will ultimately make interactions with digital assistants more human-like. We evaluate our framework on 1 internal and 4 public benchmark datasets using both automatic (Perplexity) and human (SSA - Sensibleness and Specificity Average) evaluation metrics and establish comparable performance while reducing model parameters by 90%.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Low-Resource Adaptation of Open-Domain Generative Chatbots<br>pdf: <a href="https://t.co/dHyXBYSUF6">https://t.co/dHyXBYSUF6</a><br>abs: <a href="https://t.co/s4MkdgJhYM">https://t.co/s4MkdgJhYM</a> <a href="https://t.co/I0RmScyPqF">pic.twitter.com/I0RmScyPqF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427074608516243458?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Algebraic Geometry and Representation theory in the study of matrix  multiplication complexity and other problems in theoretical computer science

J. M. Landsberg

- retweets: 53, favorites: 32 (08/17/2021 03:47:57)

- links: [abs](https://arxiv.org/abs/2108.06263) | [pdf](https://arxiv.org/pdf/2108.06263)
- [math.AG](https://arxiv.org/list/math.AG/recent) | [cs.CC](https://arxiv.org/list/cs.CC/recent) | [math.RT](https://arxiv.org/list/math.RT/recent)

Many fundamental questions in theoretical computer science are naturally expressed as special cases of the following problem: Let $G$ be a complex reductive group, let $V$ be a $G$-module, and let $v,w$ be elements of $V$. Determine if $w$ is in the $G$-orbit closure of $v$. I explain the computer science problems, the questions in representation theory and algebraic geometry that they give rise to, and the new perspectives on old areas such as invariant theory that have arisen in light of these questions. I focus primarily on the complexity of matrix multiplication.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Algebraic Geometry and Representation theory in the study of matrix multiplication complexity and other problems in theoretical computer science&quot; (by J. M. Landsberg): <a href="https://t.co/9OcUiQrh4j">https://t.co/9OcUiQrh4j</a></p>&mdash; DynamicalSystemsSIAM (@DynamicsSIAM) <a href="https://twitter.com/DynamicsSIAM/status/1427085328951640064?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Continual Backprop: Stochastic Gradient Descent with Persistent  Randomness

Shibhansh Dohare, A. Rupam Mahmood, Richard S. Sutton

- retweets: 43, favorites: 37 (08/17/2021 03:47:57)

- links: [abs](https://arxiv.org/abs/2108.06325) | [pdf](https://arxiv.org/pdf/2108.06325)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

The Backprop algorithm for learning in neural networks utilizes two mechanisms: first, stochastic gradient descent and second, initialization with small random weights, where the latter is essential to the effectiveness of the former. We show that in continual learning setups, Backprop performs well initially, but over time its performance degrades. Stochastic gradient descent alone is insufficient to learn continually; the initial randomness enables only initial learning but not continual learning. To the best of our knowledge, ours is the first result showing this degradation in Backprop's ability to learn. To address this issue, we propose an algorithm that continually injects random features alongside gradient descent using a new generate-and-test process. We call this the Continual Backprop algorithm. We show that, unlike Backprop, Continual Backprop is able to continually adapt in both supervised and reinforcement learning problems. We expect that as continual learning becomes more common in future applications, a method like Continual Backprop will be essential where the advantages of random initialization are present throughout learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Continual Backprop: Stochastic Gradient Descent<br>with Persistent Randomness<br>pdf: <a href="https://t.co/D36bGyAbO7">https://t.co/D36bGyAbO7</a><br>abs: <a href="https://t.co/mzrQr2JbbT">https://t.co/mzrQr2JbbT</a><br><br>propose an algorithm that continually injects random features alongside gradient descent using a new generate-and-test process <a href="https://t.co/b1eXERhuv0">pic.twitter.com/b1eXERhuv0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427072631719743492?ref_src=twsrc%5Etfw">August 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



