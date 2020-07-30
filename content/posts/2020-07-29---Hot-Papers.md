---
title: Hot Papers 2020-07-29
date: 2020-07-30T10:57:26.Z
template: "post"
draft: false
slug: "hot-papers-2020-07-29"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-07-29"
socialImage: "/media/flying-marine.jpg"

---

# 1. Big Bird: Transformers for Longer Sequences

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed

- retweets: 100, favorites: 451 (07/30/2020 10:57:26)

- links: [abs](https://arxiv.org/abs/2007.14062) | [pdf](https://arxiv.org/pdf/2007.14062)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Transformers-based models, such as BERT, have been one of the most successful deep learning models for NLP. Unfortunately, one of their core limitations is the quadratic dependency (mainly in terms of memory) on the sequence length due to their full attention mechanism. To remedy this, we propose, BigBird, a sparse attention mechanism that reduces this quadratic dependency to linear. We show that BigBird is a universal approximator of sequence functions and is Turing complete, thereby preserving these properties of the quadratic, full attention model. Along the way, our theoretical analysis reveals some of the benefits of having $O(1)$ global tokens (such as CLS), that attend to the entire sequence as part of the sparse attention mechanism. The proposed sparse attention can handle sequences of length up to 8x of what was previously possible using similar hardware. As a consequence of the capability to handle longer context, BigBird drastically improves performance on various NLP tasks such as question answering and summarization. We also propose novel applications to genomics data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Big Bird: Transformers for Longer Sequences 🐦<br>pdf: <a href="https://t.co/1ZH5oC2T2e">https://t.co/1ZH5oC2T2e</a><br>abs: <a href="https://t.co/DLt59rpbps">https://t.co/DLt59rpbps</a> <a href="https://t.co/XHuvaqPahM">pic.twitter.com/XHuvaqPahM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1288279345585426435?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">That didn’t take long at all! As predicted in the recent <a href="https://twitter.com/pagestlabs?ref_src=twsrc%5Etfw">@pagestlabs</a> issue, long-span attention cost for transformer models like GPT-3 and T5 came down from O(N√N) to O(N) in BigBird. Looking forward to these models becoming viable for everyone to build.<a href="https://t.co/ukS1TAWsX3">https://t.co/ukS1TAWsX3</a> <a href="https://t.co/4NVZ0j5Vss">https://t.co/4NVZ0j5Vss</a></p>&mdash; Delip Rao (@deliprao) <a href="https://twitter.com/deliprao/status/1288362975565213698?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">BigBirdはTransformerで注意対象を1)疎なランダム位置 2) 局所周辺 3) 定数個の全位置 の組み合わせで表す。系列長に対し線形計算量で、元のTransformerと同じ表現力を持ちTuring完全である。長距離依存を扱えNLPタスクのSOTAを更新、DNAのプロモーター領域をほぼ完璧に予測 <a href="https://t.co/iO5FEpUMvp">https://t.co/iO5FEpUMvp</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1288620979652980736?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Big Bird is a transformer-based model that more effectively supports NLP tasks requiring longer contexts.<br><br>It satisfies the theoretical properties of the full model while reducing the attention mechanism complexity to linear in # of tokens.<a href="https://t.co/zMJ5ZmSBUc">https://t.co/zMJ5ZmSBUc</a> <a href="https://t.co/HiMEMSlezY">pic.twitter.com/HiMEMSlezY</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1288426752168075264?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Big Bird: Transformers for Longer Sequences (Google) <a href="https://t.co/wQWVWFyfI9">https://t.co/wQWVWFyfI9</a> Transformerのスパースアテンションによる長期系列への対応．BigBird = Random + Local (Window) + Global．実験は最大長4096．MLM, QA, 要約などで評価．blockサイズ（図4, 表12）やglobalトークン数が結構大きい印象 <a href="https://t.co/jVkpY8smSb">pic.twitter.com/jVkpY8smSb</a></p>&mdash; Kyosuke Nishida (@kyoun) <a href="https://twitter.com/kyoun/status/1288394954620903424?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Link for the lazy: <a href="https://t.co/vGEdi2tWAf">https://t.co/vGEdi2tWAf</a></p>&mdash; Madison May (@pragmaticml) <a href="https://twitter.com/pragmaticml/status/1288293649554579456?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very nice work from colleagues in my team and in sibling teams, <a href="https://t.co/MGQZHi0AjW">https://t.co/MGQZHi0AjW</a> <a href="https://t.co/pCpIyuqzlR">https://t.co/pCpIyuqzlR</a></p>&mdash; D. Sivakumar (@dsivakumar) <a href="https://twitter.com/dsivakumar/status/1288341826626154497?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Noise-Induced Barren Plateaus in Variational Quantum Algorithms

Samson Wang, Enrico Fontana, M. Cerezo, Kunal Sharma, Akira Sone, Lukasz Cincio, Patrick J. Coles

- retweets: 45, favorites: 247 (07/30/2020 10:57:27)

- links: [abs](https://arxiv.org/abs/2007.14384) | [pdf](https://arxiv.org/pdf/2007.14384)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Variational Quantum Algorithms (VQAs) may be a path to quantum advantage on Noisy Intermediate-Scale Quantum (NISQ) computers. A natural question is whether the noise on NISQ devices places any fundamental limitations on the performance of VQAs. In this work, we rigorously prove a serious limitation for noisy VQAs, in that the noise causes the training landscape to have a barren plateau (i.e., vanishing gradient). Specifically, for the local Pauli noise considered, we prove that the gradient vanishes exponentially in the number of layers $L$. This implies exponential decay in the number of qubits $n$ when $L$ scales as $\operatorname{poly}(n)$, for sufficiently large coefficients in the polynomial. These noise-induced barren plateaus (NIBPs) are conceptually different from noise-free barren plateaus, which are linked to random parameter initialization. Our result is formulated for an abstract ansatz that includes as special cases the Quantum Alternating Operator Ansatz (QAOA) and the Unitary Coupled Cluster Ansatz, among others. In the case of the QAOA, we implement numerical heuristics that confirm the NIBP phenomenon for a realistic hardware noise model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Congrats to our students <a href="https://twitter.com/samson_wang?ref_src=twsrc%5Etfw">@samson_wang</a>,<a href="https://twitter.com/EnricoFontana19?ref_src=twsrc%5Etfw">@EnricoFontana19</a> for discovering a new phenomenon, in the first paper of our 2020 school. We prove that local Pauli noise if strong enough will cause a barren plateau in cost landscape. Deep ansatzes are untrainable. <a href="https://t.co/hHrP6HJJaZ">https://t.co/hHrP6HJJaZ</a> <a href="https://t.co/4YrPDZ3mOC">pic.twitter.com/4YrPDZ3mOC</a></p>&mdash; Plateaus Coles (@ColesQuantum) <a href="https://twitter.com/ColesQuantum/status/1288313953861406721?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to have a preprint out, the first I have been involved in! <br><br>Our message for variational algorithms: keep depth linear or lower in number of qubits for hope of avoiding barren plateaus. Go deeper, and they&#39;re inevitable (asymptotically)<a href="https://t.co/YaeqSomPhC">https://t.co/YaeqSomPhC</a> <a href="https://t.co/KgTjW2OFJG">pic.twitter.com/KgTjW2OFJG</a></p>&mdash; Samson Wang (@samson_wang) <a href="https://twitter.com/samson_wang/status/1288320799829565440?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new results: <a href="https://t.co/q7tKSOGhyv">https://t.co/q7tKSOGhyv</a><br><br>with <a href="https://twitter.com/samson_wang?ref_src=twsrc%5Etfw">@samson_wang</a>, <a href="https://twitter.com/EnricoFontana19?ref_src=twsrc%5Etfw">@EnricoFontana19</a>, <a href="https://twitter.com/MvsCerezo?ref_src=twsrc%5Etfw">@MvsCerezo</a>, <a href="https://twitter.com/SoneAkira?ref_src=twsrc%5Etfw">@SoneAkira</a>, <a href="https://twitter.com/LCincio?ref_src=twsrc%5Etfw">@LCincio</a>, <a href="https://twitter.com/ColesQuantum?ref_src=twsrc%5Etfw">@ColesQuantum</a>. <br><br>How does the noise on NISQ devices place limitations on the trainability of variational quantum algorithms (VQAs)? <a href="https://t.co/snDe6YaLrM">pic.twitter.com/snDe6YaLrM</a></p>&mdash; Kunal Sharma (@kunal_phy) <a href="https://twitter.com/kunal_phy/status/1288312781666545665?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work from out LANL Quantum Computing Summer School by 👉<a href="https://twitter.com/samson_wang?ref_src=twsrc%5Etfw">@samson_wang</a> &amp; <a href="https://twitter.com/EnricoFontana19?ref_src=twsrc%5Etfw">@EnricoFontana19</a> 👈 and my collaborators <a href="https://twitter.com/kunal_phy?ref_src=twsrc%5Etfw">@kunal_phy</a>, <a href="https://twitter.com/SoneAkira?ref_src=twsrc%5Etfw">@SoneAkira</a>, <a href="https://twitter.com/LCincio?ref_src=twsrc%5Etfw">@LCincio</a>, <a href="https://twitter.com/ColesQuantum?ref_src=twsrc%5Etfw">@ColesQuantum</a>. <a href="https://t.co/syHxyN5V40">https://t.co/syHxyN5V40</a><br><br>Below I explain the results... 🔥VIA MEMES!🔥 <a href="https://t.co/XbSGxzhO2E">pic.twitter.com/XbSGxzhO2E</a></p>&mdash; Marco Cerezo (@MvsCerezo) <a href="https://twitter.com/MvsCerezo/status/1288483731313799169?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Map from our latest work on Noise Induced Barren Plateaus.<br><br>🗺️We have recently return from exploring the untamed land where noise roams free and gate infidelity lurks behind every door. What we have found is a bleak land, barren of any features. <a href="https://t.co/syHxyN5V40">https://t.co/syHxyN5V40</a> <a href="https://t.co/JEonJRo9zv">pic.twitter.com/JEonJRo9zv</a></p>&mdash; Marco Cerezo (@MvsCerezo) <a href="https://twitter.com/MvsCerezo/status/1288500533263650817?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Self-supervised Neural Audio-Visual Sound Source Localization via  Probabilistic Spatial Modeling

Yoshiki Masuyama, Yoshiaki Bando, Kohei Yatabe, Yoko Sasaki, Masaki Onishi, Yasuhiro Oikawa

- retweets: 14, favorites: 54 (07/30/2020 10:57:28)

- links: [abs](https://arxiv.org/abs/2007.13976) | [pdf](https://arxiv.org/pdf/2007.13976)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Detecting sound source objects within visual observation is important for autonomous robots to comprehend surrounding environments. Since sounding objects have a large variety with different appearances in our living environments, labeling all sounding objects is impossible in practice. This calls for self-supervised learning which does not require manual labeling. Most of conventional self-supervised learning uses monaural audio signals and images and cannot distinguish sound source objects having similar appearances due to poor spatial information in audio signals. To solve this problem, this paper presents a self-supervised training method using 360{\deg} images and multichannel audio signals. By incorporating with the spatial information in multichannel audio signals, our method trains deep neural networks (DNNs) to distinguish multiple sound source objects. Our system for localizing sound source objects in the image is composed of audio and visual DNNs. The visual DNN is trained to localize sound source candidates within an input image. The audio DNN verifies whether each candidate actually produces sound or not. These DNNs are jointly trained in a self-supervised manner based on a probabilistic spatial audio model. Experimental results with simulated data showed that the DNNs trained by our method localized multiple speakers. We also demonstrate that the visual DNN detected objects including talking visitors and specific exhibits from real data recorded in a science museum.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/IROS2020?src=hash&amp;ref_src=twsrc%5Etfw">#IROS2020</a> paper titled &quot;Self-supervised Neural Audio-Visual Sound Source Localization via Probabilistic Spatial Modeling&quot; is now available online! Our self-supervised learning is based on probabilistic inference of a multichannel audio model (cGMM).<a href="https://t.co/zi986Lz9sU">https://t.co/zi986Lz9sU</a> <a href="https://t.co/5RU2kb65Zl">pic.twitter.com/5RU2kb65Zl</a></p>&mdash; まっすー (@ymas0315) <a href="https://twitter.com/ymas0315/status/1288313015851393025?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Toward Zero-Shot Unsupervised Image-to-Image Translation

Yuanqi Chen, Xiaoming Yu, Shan Liu, Ge Li

- retweets: 14, favorites: 52 (07/30/2020 10:57:28)

- links: [abs](https://arxiv.org/abs/2007.14050) | [pdf](https://arxiv.org/pdf/2007.14050)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent studies have shown remarkable success in unsupervised image-to-image translation. However, if there has no access to enough images in target classes, learning a mapping from source classes to the target classes always suffers from mode collapse, which limits the application of the existing methods. In this work, we propose a zero-shot unsupervised image-to-image translation framework to address this limitation, by associating categories with their side information like attributes. To generalize the translator to previous unseen classes, we introduce two strategies for exploiting the space spanned by the semantic attributes. Specifically, we propose to preserve semantic relations to the visual space and expand attribute space by utilizing attribute vectors of unseen classes, thus encourage the translator to explore the modes of unseen classes. Quantitative and qualitative results on different datasets demonstrate the effectiveness of our proposed approach. Moreover, we demonstrate that our framework can be applied to many tasks, such as zero-shot classification and fashion design.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Toward Zero-Shot Unsupervised Image-to-Image Translation<br>pdf: <a href="https://t.co/saEvcoPpCx">https://t.co/saEvcoPpCx</a><br>abs: <a href="https://t.co/5dy2hz3TfZ">https://t.co/5dy2hz3TfZ</a> <a href="https://t.co/UcsJ7NxJuI">pic.twitter.com/UcsJ7NxJuI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1288278351497564160?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. BUSTLE: Bottom-up program-Synthesis Through Learning-guided Exploration

Augustus Odena, Kensen Shi, David Bieber, Rishabh Singh, Charles Sutton

- retweets: 16, favorites: 48 (07/30/2020 10:57:28)

- links: [abs](https://arxiv.org/abs/2007.14381) | [pdf](https://arxiv.org/pdf/2007.14381)
- [cs.PL](https://arxiv.org/list/cs.PL/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Program synthesis is challenging largely because of the difficulty of search in a large space of programs. Human programmers routinely tackle the task of writing complex programs by writing sub-programs and then analysing their intermediate results to compose them in appropriate ways. Motivated by this intuition, we present a new synthesis approach that leverages learning to guide a bottom-up search over programs. In particular, we train a model to prioritize compositions of intermediate values during search conditioned on a given set of input-output examples. This is a powerful combination because of several emergent properties: First, in bottom-up search, intermediate programs can be executed, providing semantic information to the neural network. Second, given the concrete values from those executions, we can exploit rich features based on recent work on property signatures. Finally, bottom-up search allows the system substantial flexibility in what order to generate the solution, allowing the synthesizer to build up a program from multiple smaller sub-programs. Overall, our empirical evaluation finds that the combination of learning and bottom-up search is remarkably effective, even with simple supervised learning approaches. We demonstrate the effectiveness of our technique on a new data set for synthesis of string transformation programs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What&#39;s the fuss(le) about BUSTLE?<br>It&#39;s our new paper on program synthesis! (<a href="https://t.co/aJMmFAZRG6">https://t.co/aJMmFAZRG6</a>)<br>We perform bottom-up search over programs, with machine learning in the inner loop. <br>A thread: (1/8) <a href="https://t.co/78UD7FOPAX">pic.twitter.com/78UD7FOPAX</a></p>&mdash; augustus odena (@gstsdn) <a href="https://twitter.com/gstsdn/status/1288520054003261440?ref_src=twsrc%5Etfw">July 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



