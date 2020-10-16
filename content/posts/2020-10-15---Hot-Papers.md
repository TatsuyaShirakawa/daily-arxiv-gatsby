---
title: Hot Papers 2020-10-15
date: 2020-10-16T10:15:47.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-15"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-15"
socialImage: "/media/flying-marine.jpg"

---

# 1. With Little Power Comes Great Responsibility

Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky

- retweets: 722, favorites: 140 (10/16/2020 10:15:47)

- links: [abs](https://arxiv.org/abs/2010.06595) | [pdf](https://arxiv.org/pdf/2010.06595)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Despite its importance to experimental design, statistical power (the probability that, given a real effect, an experiment will reject the null hypothesis) has largely been ignored by the NLP community. Underpowered experiments make it more difficult to discern the difference between statistical noise and meaningful model improvements, and increase the chances of exaggerated findings. By meta-analyzing a set of existing NLP papers and datasets, we characterize typical power for a variety of settings and conclude that underpowered experiments are common in the NLP literature. In particular, for several tasks in the popular GLUE benchmark, small test sets mean that most attempted comparisons to state of the art models will not be adequately powered. Similarly, based on reasonable assumptions, we find that the most typical experimental design for human rating studies will be underpowered to detect small model differences, of the sort that are frequently studied. For machine translation, we find that typical test sets of 2000 sentences have approximately 75% power to detect differences of 1 BLEU point. To improve the situation going forward, we give an overview of best practices for power analysis in NLP and release a series of notebooks to assist with future power analyses.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Interesting paper by <a href="https://twitter.com/jurafsky?ref_src=twsrc%5Etfw">@jurafsky</a> et al. that estimates statistical power of <a href="https://twitter.com/hashtag/nlproc?src=hash&amp;ref_src=twsrc%5Etfw">#nlproc</a> experiments: <a href="https://t.co/ZUPxs8zfwl">https://t.co/ZUPxs8zfwl</a> Some benchmark datasets are too small to be useful.</p>&mdash; John Platt (@johnplattml) <a href="https://twitter.com/johnplattml/status/1316628661638852608?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New EMNLP paper with <a href="https://twitter.com/PeterHndrsn?ref_src=twsrc%5Etfw">@PeterHndrsn</a> <a href="https://twitter.com/ukhndlwl?ref_src=twsrc%5Etfw">@ukhndlwl</a> <a href="https://twitter.com/robinomial?ref_src=twsrc%5Etfw">@robinomial</a> <a href="https://twitter.com/kmahowald?ref_src=twsrc%5Etfw">@kmahowald</a> and <a href="https://twitter.com/jurafsky?ref_src=twsrc%5Etfw">@jurafsky</a> --  With Little Power Comes Great Responsibility -- <a href="https://t.co/JT9U4Ertur">https://t.co/JT9U4Ertur</a> (1/3)</p>&mdash; Dallas Card (@dallascard) <a href="https://twitter.com/dallascard/status/1316775765690601472?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Vokenization: Improving Language Understanding with Contextualized,  Visual-Grounded Supervision

Hao Tan, Mohit Bansal

- retweets: 555, favorites: 145 (10/16/2020 10:15:47)

- links: [abs](https://arxiv.org/abs/2010.06775) | [pdf](https://arxiv.org/pdf/2010.06775)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Humans learn language by listening, speaking, writing, reading, and also, via interaction with the multimodal real world. Existing language pre-training frameworks show the effectiveness of text-only self-supervision while we explore the idea of a visually-supervised language model in this paper. We find that the main reason hindering this exploration is the large divergence in magnitude and distributions between the visually-grounded language datasets and pure-language corpora. Therefore, we develop a technique named "vokenization" that extrapolates multimodal alignments to language-only data by contextually mapping language tokens to their related images (which we call "vokens"). The "vokenizer" is trained on relatively small image captioning datasets and we then apply it to generate vokens for large language corpora. Trained with these contextually generated vokens, our visually-supervised language models show consistent improvements over self-supervised alternatives on multiple pure-language tasks such as GLUE, SQuAD, and SWAG. Code and pre-trained models publicly available at https://github.com/airsplay/vokenization

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">*Vokenization*: a visually-supervised language model attempt in our <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a> paper: <a href="https://t.co/r9MZNniAhn">https://t.co/r9MZNniAhn</a> (w. <a href="https://twitter.com/mohitban47?ref_src=twsrc%5Etfw">@mohitban47</a>)<br><br>To improve language pre-training, we extrapolate multimodal alignments to lang-only data by contextually mapping tokens to related images (&quot;vokens&quot;) 1/4 <a href="https://t.co/wuXt1K58BH">pic.twitter.com/wuXt1K58BH</a></p>&mdash; Hao Tan (@HaoTan5) <a href="https://twitter.com/HaoTan5/status/1316785618278666241?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Vokens&quot; = Visually-grounded-tokens (contextual) to imprv lang-pretraining &amp; engl NLU tasks (imp divergence/grounding ratio issues, extrapolates frm small dataset)!<br><br>pdf: <a href="https://t.co/rNMnmDyJga">https://t.co/rNMnmDyJga</a><br>Full code: <a href="https://t.co/KELW6XVYbc">https://t.co/KELW6XVYbc</a><br><br>➡️Hao is on job market🙂: <a href="https://t.co/CB2Fty0f0A">https://t.co/CB2Fty0f0A</a> <a href="https://t.co/0tEVIFU5GJ">https://t.co/0tEVIFU5GJ</a></p>&mdash; Mohit Bansal (@🏡) (@mohitban47) <a href="https://twitter.com/mohitban47/status/1316816915873042441?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Learning Deep Features in Instrumental Variable Regression

Liyuan Xu, Yutian Chen, Siddarth Srinivasan, Nando de Freitas, Arnaud Doucet, Arthur Gretton

- retweets: 230, favorites: 144 (10/16/2020 10:15:48)

- links: [abs](https://arxiv.org/abs/2010.07154) | [pdf](https://arxiv.org/pdf/2010.07154)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Instrumental variable (IV) regression is a standard strategy for learning causal relationships between confounded treatment and outcome variables by utilizing an instrumental variable, which is conditionally independent of the outcome given the treatment. In classical IV regression, learning proceeds in two stages: stage 1 performs linear regression from the instrument to the treatment; and stage 2 performs linear regression from the treatment to the outcome, conditioned on the instrument. We propose a novel method, {\it deep feature instrumental variable regression (DFIV)}, to address the case where relations between instruments, treatments, and outcomes may be nonlinear. In this case, deep neural nets are trained to define informative nonlinear features on the instruments and treatments. We propose an alternating training regime for these features to ensure good end-to-end performance when composing stages 1 and 2, thus obtaining highly flexible feature maps in a computationally efficient manner. DFIV outperforms recent state-of-the-art methods on challenging IV benchmarks, including settings involving high dimensional image data. DFIV also exhibits competitive performance in off-policy policy evaluation for reinforcement learning, which can be understood as an IV regression task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Bridging the gap between causal inference with neural networks and off policy evaluation in deep RL ⁦<a href="https://twitter.com/yudapearl?ref_src=twsrc%5Etfw">@yudapearl</a>⁩  <a href="https://twitter.com/hashtag/causality?src=hash&amp;ref_src=twsrc%5Etfw">#causality</a> <a href="https://t.co/Y4HnDCm6rf">https://t.co/Y4HnDCm6rf</a></p>&mdash; Nando de Freitas (@NandoDF) <a href="https://twitter.com/NandoDF/status/1316685504650321920?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">留学してから最初の論文が出ました。因果推論における操作変数法をDeepでうまく学習する方法の提案と、それを使って強化学習のOffline Policy Evaluationが解けるよ、という主張です。<a href="https://t.co/YpsegpCUHW">https://t.co/YpsegpCUHW</a></p>&mdash; LY9988 (@ly9988) <a href="https://twitter.com/ly9988/status/1316723370969366528?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Are all negatives created equal in contrastive instance discrimination?

Tiffany, Jonathan Frankle, David J. Schwab, Ari S. Morcos

- retweets: 242, favorites: 102 (10/16/2020 10:15:48)

- links: [abs](https://arxiv.org/abs/2010.06682) | [pdf](https://arxiv.org/pdf/2010.06682)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Self-supervised learning has recently begun to rival supervised learning on computer vision tasks. Many of the recent approaches have been based on contrastive instance discrimination (CID), in which the network is trained to recognize two augmented versions of the same instance (a query and positive) while discriminating against a pool of other instances (negatives). The learned representation is then used on downstream tasks such as image classification. Using methodology from MoCo v2 (Chen et al., 2020), we divided negatives by their difficulty for a given query and studied which difficulty ranges were most important for learning useful representations. We found a minority of negatives -- the hardest 5% -- were both necessary and sufficient for the downstream task to reach nearly full accuracy. Conversely, the easiest 95% of negatives were unnecessary and insufficient. Moreover, the very hardest 0.1% of negatives were unnecessary and sometimes detrimental. Finally, we studied the properties of negatives that affect their hardness, and found that hard negatives were more semantically similar to the query, and that some negatives were more consistently easy or hard than we would expect by chance. Together, our results indicate that negatives vary in importance and that CID may benefit from more intelligent negative treatment.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Are all negatives created equal in contrastive instance discrimination? <br><br>In new work led by Tiffany Cai, we show that only the hardest 5% of negatives per query are both necessary and largely sufficient for self-supervised learning.<br><br>Tweetprint time!<a href="https://t.co/ijyrMb4zGG">https://t.co/ijyrMb4zGG</a> <a href="https://t.co/dIeR7DptVF">pic.twitter.com/dIeR7DptVF</a></p>&mdash; Ari Morcos (@arimorcos) <a href="https://twitter.com/arimorcos/status/1316804557192613888?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Video Action Understanding: A Tutorial

Matthew Hutchinson, Vijay Gadepally

- retweets: 90, favorites: 112 (10/16/2020 10:15:48)

- links: [abs](https://arxiv.org/abs/2010.06647) | [pdf](https://arxiv.org/pdf/2010.06647)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Many believe that the successes of deep learning on image understanding problems can be replicated in the realm of video understanding. However, the span of video action problems and the set of proposed deep learning solutions is arguably wider and more diverse than those of their 2D image siblings. Finding, identifying, and predicting actions are a few of the most salient tasks in video action understanding. This tutorial clarifies a taxonomy of video action problems, highlights datasets and metrics used to baseline each problem, describes common data preparation methods, and presents the building blocks of state-of-the-art deep learning model architectures.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Video Action Understanding: A Tutorial<a href="https://t.co/bICYnbbVNg">https://t.co/bICYnbbVNg</a><br>行動認識はこれが一番良くまとまっている。 <a href="https://t.co/8wlFRG8qJA">pic.twitter.com/8wlFRG8qJA</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1316583534748790784?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. fugashi, a Tool for Tokenizing Japanese in Python

Paul McCann

- retweets: 122, favorites: 34 (10/16/2020 10:15:48)

- links: [abs](https://arxiv.org/abs/2010.06858) | [pdf](https://arxiv.org/pdf/2010.06858)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Recent years have seen an increase in the number of large-scale multilingual NLP projects. However, even in such projects, languages with special processing requirements are often excluded. One such language is Japanese. Japanese is written without spaces, tokenization is non-trivial, and while high quality open source tokenizers exist they can be hard to use and lack English documentation. This paper introduces fugashi, a MeCab wrapper for Python, and gives an introduction to tokenizing Japanese.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Working on a multilingual NLP project but didn&#39;t integrate Japanese because you didn&#39;t know how to tokenize it? I&#39;ve got a paper for you! This is a brief introduction to fugashi, the Japanese tokenizer used in <a href="https://twitter.com/huggingface?ref_src=twsrc%5Etfw">@huggingface</a> Transformers and elsewhere. <a href="https://t.co/dQ5LNWgcy8">https://t.co/dQ5LNWgcy8</a> <a href="https://t.co/UCtfshVqbv">pic.twitter.com/UCtfshVqbv</a></p>&mdash; Paul O&#39;Leary McCann (@polm23) <a href="https://twitter.com/polm23/status/1316642739606282240?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Flexible mean field variational inference using mixtures of  non-overlapping exponential families

Jeffrey P. Spence

- retweets: 72, favorites: 47 (10/16/2020 10:15:48)

- links: [abs](https://arxiv.org/abs/2010.06768) | [pdf](https://arxiv.org/pdf/2010.06768)
- [math.ST](https://arxiv.org/list/math.ST/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Sparse models are desirable for many applications across diverse domains as they can perform automatic variable selection, aid interpretability, and provide regularization. When fitting sparse models in a Bayesian framework, however, analytically obtaining a posterior distribution over the parameters of interest is intractable for all but the simplest cases. As a result practitioners must rely on either sampling algorithms such as Markov chain Monte Carlo or variational methods to obtain an approximate posterior. Mean field variational inference is a particularly simple and popular framework that is often amenable to analytically deriving closed-form parameter updates. When all distributions in the model are members of exponential families and are conditionally conjugate, optimization schemes can often be derived by hand. Yet, I show that using standard mean field variational inference can fail to produce sensible results for models with sparsity-inducing priors, such as the spike-and-slab. Fortunately, such pathological behavior can be remedied as I show that mixtures of exponential family distributions with non-overlapping support form an exponential family. In particular, any mixture of a diffuse exponential family and a point mass at zero to model sparsity forms an exponential family. Furthermore, specific choices of these distributions maintain conditional conjugacy. I use two applications to motivate these results: one from statistical genetics that has connections to generalized least squares with a spike-and-slab prior on the regression coefficients; and sparse probabilistic principal component analysis. The theoretical results presented here are broadly applicable beyond these two examples.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m happy to say that I&#39;ve finally gotten my NeurIPS submission preprinted.  Short thread below! 1/6<a href="https://t.co/9vF43d14sN">https://t.co/9vF43d14sN</a></p>&mdash; jeffrey spence (@spence_jeffrey_) <a href="https://twitter.com/spence_jeffrey_/status/1316539817144938497?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Recipes for Safety in Open-domain Chatbots

Jing Xu, Da Ju, Margaret Li, Y-Lan Boureau, Jason Weston, Emily Dinan

- retweets: 49, favorites: 47 (10/16/2020 10:15:48)

- links: [abs](https://arxiv.org/abs/2010.07079) | [pdf](https://arxiv.org/pdf/2010.07079)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Models trained on large unlabeled corpora of human interactions will learn patterns and mimic behaviors therein, which include offensive or otherwise toxic behavior and unwanted biases. We investigate a variety of methods to mitigate these issues in the context of open-domain generative dialogue models. We introduce a new human-and-model-in-the-loop framework for both training safer models and for evaluating them, as well as a novel method to distill safety considerations inside generative models without the use of an external classifier at deployment time. We conduct experiments comparing these methods and find our new techniques are (i) safer than existing models as measured by automatic and human evaluations while (ii) maintaining usability metrics such as engagingness relative to the state of the art. We then discuss the limitations of this work by analyzing failure cases of our models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share this new work on safer conversational AI systems, just in time for the Safety for ConvAI Workshop today! <a href="https://t.co/EqpJHA6EKs">https://t.co/EqpJHA6EKs</a><br><br>Fun working with <a href="https://twitter.com/jingxu_ml?ref_src=twsrc%5Etfw">@jingxu_ml</a> <a href="https://twitter.com/dexterJu27?ref_src=twsrc%5Etfw">@dexterJu27</a> <a href="https://twitter.com/margs_li?ref_src=twsrc%5Etfw">@margs_li</a> Y-Lan and <a href="https://twitter.com/jaseweston?ref_src=twsrc%5Etfw">@jaseweston</a>!</p>&mdash; Emily Dinan (@em_dinan) <a href="https://twitter.com/em_dinan/status/1316715673280798720?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Re-evaluating Evaluation in Text Summarization

Manik Bhandari, Pranav Gour, Atabak Ashfaq, Pengfei Liu, Graham Neubig

- retweets: 26, favorites: 47 (10/16/2020 10:15:48)

- links: [abs](https://arxiv.org/abs/2010.07100) | [pdf](https://arxiv.org/pdf/2010.07100)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Automated evaluation metrics as a stand-in for manual evaluation are an essential part of the development of text-generation tasks such as text summarization. However, while the field has progressed, our standard metrics have not -- for nearly 20 years ROUGE has been the standard evaluation in most summarization papers. In this paper, we make an attempt to re-evaluate the evaluation method for text summarization: assessing the reliability of automatic metrics using top-scoring system outputs, both abstractive and extractive, on recently popular datasets for both system-level and summary-level evaluation settings. We find that conclusions about evaluation metrics on older datasets do not necessarily hold on modern datasets and systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our <a href="https://twitter.com/hashtag/EMNLP2020?src=hash&amp;ref_src=twsrc%5Etfw">#EMNLP2020</a>  work: <br>REALSum: Re-evaluating Evaluation in Text Summ: <a href="https://t.co/S3YCKy79RH">https://t.co/S3YCKy79RH</a><br>(super awesome coauthors: <a href="https://twitter.com/manikb20?ref_src=twsrc%5Etfw">@manikb20</a> <a href="https://twitter.com/pranav?ref_src=twsrc%5Etfw">@Pranav</a> <a href="https://twitter.com/ashatabak786?ref_src=twsrc%5Etfw">@ashatabak786</a> and <a href="https://twitter.com/gneubig?ref_src=twsrc%5Etfw">@gneubig</a>  )<br>Are existing automated metrics reliable???  All relevant resource has been released (1/n)! <a href="https://t.co/Ggj3M3vPr0">pic.twitter.com/Ggj3M3vPr0</a></p>&mdash; Pengfei Liu (@stefan_fee) <a href="https://twitter.com/stefan_fee/status/1316732208900603904?ref_src=twsrc%5Etfw">October 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



