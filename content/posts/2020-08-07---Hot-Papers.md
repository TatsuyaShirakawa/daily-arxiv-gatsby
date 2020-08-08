---
title: Hot Papers 2020-08-07
date: 2020-08-08T09:03:05.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-07"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-07"
socialImage: "/media/flying-marine.jpg"

---

# 1. Question and Answer Test-Train Overlap in Open-Domain Question Answering  Datasets

Patrick Lewis, Pontus Stenetorp, Sebastian Riedel

- retweets: 113, favorites: 476 (08/08/2020 09:03:05)

- links: [abs](https://arxiv.org/abs/2008.02637) | [pdf](https://arxiv.org/pdf/2008.02637)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Ideally Open-Domain Question Answering models should exhibit a number of competencies, ranging from simply memorizing questions seen at training time, to answering novel question formulations with answers seen during training, to generalizing to completely novel questions with novel answers. However, single aggregated test set scores do not show the full picture of what capabilities models truly have. In this work, we perform a detailed study of the test sets of three popular open-domain benchmark datasets with respect to these competencies. We find that 60-70% of test-time answers are also present somewhere in the training sets. We also find that 30% of test-set questions have a near-duplicate paraphrase in their corresponding training sets. Using these findings, we evaluate a variety of popular open-domain models to obtain greater insight into what extent they can actually generalize, and what drives their overall performance. We find that all models perform dramatically worse on questions that cannot be memorized from training sets, with a mean absolute performance difference of 63% between repeated and non-repeated data. Finally we show that simple nearest-neighbor models out-perform a BART closed-book QA model, further highlighting the role that training set memorization plays in these benchmarks

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Turns out a lot of open-domain QA datasets have test set leakage. If you control for it, model performance drops by a mean absolute of 63%. Yikes! If we missed this for such a long time, I wonder if there are problems with other NLP datasets too. <a href="https://t.co/uPT2uYqou7">https://t.co/uPT2uYqou7</a></p>&mdash; Tim Dettmers (@Tim_Dettmers) <a href="https://twitter.com/Tim_Dettmers/status/1291739379887562753?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New! Do you use NaturalQuestions, TriviaQA, or WebQuestions? It turns out 60% of test set answers are also in the train set. More surprising, 30% of test questions have a close paraphrase in the train set. What does it mean for models? Read <a href="https://t.co/hu3rFSe6tR">https://t.co/hu3rFSe6tR</a> to find out! 1/ <a href="https://t.co/jsW8qa3faL">pic.twitter.com/jsW8qa3faL</a></p>&mdash; Patrick Lewis (@PSH_Lewis) <a href="https://twitter.com/PSH_Lewis/status/1291739650567045121?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">woah 😲! 60% of overlap and 30% close-paraphrases is extreme... from the paper (<a href="https://t.co/nWxByQfHW8">https://t.co/nWxByQfHW8</a>): &quot;a greater emphasis should be placed on more behaviour-driven evaluation, rather than pursuing single-number overall accuracy figures.&quot; - yes! totally agree <a href="https://twitter.com/hashtag/beyondaccuracy?src=hash&amp;ref_src=twsrc%5Etfw">#beyondaccuracy</a> <a href="https://t.co/yrCfSM0T1C">https://t.co/yrCfSM0T1C</a></p>&mdash; Barbara Plank (@barbara_plank) <a href="https://twitter.com/barbara_plank/status/1291749777957412877?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Speculative Dereferencing of Registers:Reviving Foreshadow

Martin Schwarzl, Thomas Schuster, Michael Schwarz, Daniel Gruss

- retweets: 138, favorites: 328 (08/08/2020 09:03:05)

- links: [abs](https://arxiv.org/abs/2008.02307) | [pdf](https://arxiv.org/pdf/2008.02307)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

Since 2016, multiple microarchitectural attacks have exploited an effect that is attributed to prefetching. These works observe that certain user-space operations can fetch kernel addresses into the cache. Fetching user-inaccessible data into the cache enables KASLR breaks and assists various Meltdown-type attacks, especially Foreshadow.   In this paper, we provide a systematic analysis of the root cause of this prefetching effect. While we confirm the empirical results of previous papers, we show that the attribution to a prefetching mechanism is fundamentally incorrect in all previous papers describing or exploiting this effect. In particular, neither the prefetch instruction nor other user-space instructions actually prefetch kernel addresses into the cache, leading to incorrect conclusions and ineffectiveness of proposed defenses. The effect exploited in all of these papers is, in fact, caused by speculative dereferencing of user-space registers in the kernel. Hence, mitigation techniques such as KAISER do not eliminate this leakage as previously believed. Beyond our thorough analysis of these previous works, we also demonstrate new attacks enabled by understanding the root cause, namely an address-translation attack in more restricted contexts, direct leakage of register values in certain scenarios, and the first end-to-end Foreshadow (L1TF) exploit targeting non-L1 data. The latter is effective even with the recommended Foreshadow mitigations enabled and thus revives the Foreshadow attack. We demonstrate that these dereferencing effects exist even on the most recent Intel CPUs with the latest hardware mitigations, and on CPUs previously believed to be unaffected, i.e., ARM, IBM, and AMD CPUs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Researchers have discovered a novel way to exploit speculative dereferences,enabling direct leakage of data values stored in registers, showing that this effect can be adapted to Foreshadow by using addresses not valid in any address space of the guest.<a href="https://t.co/hV0bHF7FZ8">https://t.co/hV0bHF7FZ8</a> <a href="https://t.co/6HdjU2nDep">pic.twitter.com/6HdjU2nDep</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1291595231440384005?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I shouldn&#39;t comment on this, but I have repeatedly said (until my voice broke down): KASLR is not a mitigation that has a chance of surviving against a local attacker, and should not be treated as such.<br><br>Another case in point:<a href="https://t.co/nKcrdRm2x5">https://t.co/nKcrdRm2x5</a></p>&mdash; halvarflake (@halvarflake) <a href="https://twitter.com/halvarflake/status/1291661721900376066?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">So <a href="https://twitter.com/lavados?ref_src=twsrc%5Etfw">@lavados</a>, <a href="https://twitter.com/misc0110?ref_src=twsrc%5Etfw">@misc0110</a> et al released a new paper called “ Speculative Dereferencing of Registers: Reviving Foreshadow” getting at the root cause of Foreshadow making it clear that it’s not just Intel that’s affected but AMD, ARM etc too. Nice work! <a href="https://t.co/bncwCLgS4R">https://t.co/bncwCLgS4R</a> <a href="https://t.co/1oNdNuWclJ">pic.twitter.com/1oNdNuWclJ</a></p>&mdash; Adrian Rueegsegger (@Kensan42) <a href="https://twitter.com/Kensan42/status/1291661147251376128?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated  Images using Conditional Continuous Normalizing Flows

Rameen Abdal, Peihao Zhu, Niloy Mitra, Peter Wonka

- retweets: 41, favorites: 195 (08/08/2020 09:03:06)

- links: [abs](https://arxiv.org/abs/2008.02401) | [pdf](https://arxiv.org/pdf/2008.02401)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

High-quality, diverse, and photorealistic images can now be generated by unconditional GANs (e.g., StyleGAN). However, limited options exist to control the generation process using (semantic) attributes, while still preserving the quality of the output. Further, due to the entangled nature of the GAN latent space, performing edits along one attribute can easily result in unwanted changes along other attributes. In this paper, in the context of conditional exploration of entangled latent spaces, we investigate the two sub-problems of attribute-conditioned sampling and attribute-controlled editing. We present StyleFlow as a simple, effective, and robust solution to both the sub-problems by formulating conditional exploration as an instance of conditional continuous normalizing flows in the GAN latent space conditioned by attribute features. We evaluate our method using the face and the car latent space of StyleGAN, and demonstrate fine-grained disentangled edits along various attributes on both real photographs and StyleGAN generated images). For example, for faces, we vary camera pose, illumination variation, expression, facial hair, gender, and age. We show edits on synthetically generated as well as projected real images. Finally, via extensive qualitative and quantitative comparisons, we demonstrate the superiority of StyleFlow to other concurrent works.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows<br>pdf: <a href="https://t.co/BmLfGYYMTM">https://t.co/BmLfGYYMTM</a><br>abs: <a href="https://t.co/vpLT5d7J0r">https://t.co/vpLT5d7J0r</a><br>project page: <a href="https://t.co/K5y2Bra24I">https://t.co/K5y2Bra24I</a> <a href="https://t.co/dVnW5r9jm7">pic.twitter.com/dVnW5r9jm7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1291536423066259456?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Explore then Execute: Adapting without Rewards via Factorized  Meta-Reinforcement Learning

Evan Zheran Liu, Aditi Raghunathan, Percy Liang, Chelsea Finn

- retweets: 34, favorites: 174 (08/08/2020 09:03:06)

- links: [abs](https://arxiv.org/abs/2008.02790) | [pdf](https://arxiv.org/pdf/2008.02790)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We seek to efficiently learn by leveraging shared structure between different tasks and environments. For example, cooking is similar in different kitchens, even though the ingredients may change location. In principle, meta-reinforcement learning approaches can exploit this shared structure, but in practice, they fail to adapt to new environments when adaptation requires targeted exploration (e.g., exploring the cabinets to find ingredients in a new kitchen). We show that existing approaches fail due to a chicken-and-egg problem: learning what to explore requires knowing what information is critical for solving the task, but learning to solve the task requires already gathering this information via exploration. For example, exploring to find the ingredients only helps a robot prepare a meal if it already knows how to cook, but the robot can only learn to cook if it already knows where the ingredients are. To address this, we propose a new exploration objective (DREAM), based on identifying key information in the environment, independent of how this information will exactly be used solve the task. By decoupling exploration from task execution, DREAM explores and consequently adapts to new environments, requiring no reward signal when the task is specified via an instruction. Empirically, DREAM scales to more complex problems, such as sparse-reward 3D visual navigation, while existing approaches fail from insufficient exploration.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Want your robot to explore intelligently? We study how to learn to explore &amp; introduce a *efficient* meta-learning method that can lead to optimal exploration.<br><br>Paper: <a href="https://t.co/DNRJzlo8rw">https://t.co/DNRJzlo8rw</a><br>w Evan Liu, Raghunathan, Liang <a href="https://twitter.com/StanfordAILab?ref_src=twsrc%5Etfw">@StanfordAILab</a><br><br>Thread👇🏼(1/5)<a href="https://t.co/qcR6G1wfBk">https://t.co/qcR6G1wfBk</a></p>&mdash; Chelsea Finn (@chelseabfinn) <a href="https://twitter.com/chelseabfinn/status/1291565580827336705?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Network comparison and the within-ensemble graph distance

Harrison Hartle, Brennan Klein, Stefan McCabe, Alexander Daniels, Guillaume St-Onge, Charles Murphy, Laurent Hébert-Dufresne

- retweets: 33, favorites: 109 (08/08/2020 09:03:06)

- links: [abs](https://arxiv.org/abs/2008.02415) | [pdf](https://arxiv.org/pdf/2008.02415)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Quantifying the differences between networks is a challenging and ever-present problem in network science. In recent years a multitude of diverse, ad hoc solutions to this problem have been introduced. Here we propose that simple and well-understood ensembles of random networks (such as Erd\H{o}s-R\'{e}nyi graphs, random geometric graphs, Watts-Strogatz graphs, the configuration model, and preferential attachment networks) are natural benchmarks for network comparison methods. Moreover, we show that the expected distance between two networks independently sampled from a generative model is a useful property that encapsulates many key features of that model. To illustrate our results, we calculate this within-ensemble graph distance and related quantities for classic network models (and several parameterizations thereof) using 20 distance measures commonly used to compare graphs. The within-ensemble graph distance provides a new framework for developers of graph distances to better understand their creations and for practitioners to better choose an appropriate tool for their particular task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Ahh so excited about this! &quot;Network comparison and the within-ensemble graph distance&quot; <a href="https://t.co/bnMWwjJ76J">https://t.co/bnMWwjJ76J</a><br><br>It&#39;s quite simple:<br>1. Sample pairs of graphs from the same ensemble and w/ same params<br>2. Measure their graph distance<br>3. Vary params / ensembles / distances<br>4. Repeat <a href="https://t.co/sX3VjJb2BR">pic.twitter.com/sX3VjJb2BR</a></p>&mdash; Brennan Klein (@jkbren) <a href="https://twitter.com/jkbren/status/1291730627860070401?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Network comparison and the within-ensemble graph distance”<br><br>New preprint from faculty member <a href="https://twitter.com/LHDnets?ref_src=twsrc%5Etfw">@LHDnets</a> w/<a href="https://twitter.com/jkbren?ref_src=twsrc%5Etfw">@jkbren</a> &amp; team<a href="https://t.co/DP15QXG09d">https://t.co/DP15QXG09d</a> <a href="https://t.co/YjzudpKLCx">pic.twitter.com/YjzudpKLCx</a></p>&mdash; Vermont Complex Systems Center @ UVM (@uvmcomplexity) <a href="https://twitter.com/uvmcomplexity/status/1291694828284071942?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Mixed-Initiative Level Design with RL Brush

Omar Delarosa, Hang Dong, Mindy Ruan, Ahmed Khalifa, Julian Togelius

- retweets: 23, favorites: 101 (08/08/2020 09:03:07)

- links: [abs](https://arxiv.org/abs/2008.02778) | [pdf](https://arxiv.org/pdf/2008.02778)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper introduces RL Brush, a level-editing tool for tile-based games designed for mixed-initiative co-creation. The tool uses reinforcement-learning-based models to augment manual human level-design through the addition of AI-generated suggestions. Here, we apply RL Brush to designing levels for the classic puzzle game Sokoban. We put the tool online and tested it with 39 different sessions. The results show that users using the AI suggestions stay around longer and their created levels on average are more playable and more complex than without.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We can use reinforcement learning to learn to generate levels (and other functional objects). But how can we control and collaborate with these generators? We present RL Brush, a mixed-initiative level design tool.<br>Paper:<a href="https://t.co/orSOGaBGF7">https://t.co/orSOGaBGF7</a><br>Try it:<a href="https://t.co/kMQSwunShy">https://t.co/kMQSwunShy</a> <a href="https://t.co/5W6wz6O8KW">pic.twitter.com/5W6wz6O8KW</a></p>&mdash; Julian Togelius (@togelius) <a href="https://twitter.com/togelius/status/1291556128216821761?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. CrowDEA: Multi-view Idea Prioritization with Crowds

Yukino Baba, Jiyi Li, Hisashi Kashima

- retweets: 13, favorites: 60 (08/08/2020 09:03:07)

- links: [abs](https://arxiv.org/abs/2008.02354) | [pdf](https://arxiv.org/pdf/2008.02354)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Given a set of ideas collected from crowds with regard to an open-ended question, how can we organize and prioritize them in order to determine the preferred ones based on preference comparisons by crowd evaluators? As there are diverse latent criteria for the value of an idea, multiple ideas can be considered as "the best". In addition, evaluators can have different preference criteria, and their comparison results often disagree.   In this paper, we propose an analysis method for obtaining a subset of ideas, which we call frontier ideas, that are the best in terms of at least one latent evaluation criterion. We propose an approach, called CrowDEA, which estimates the embeddings of the ideas in the multiple-criteria preference space, the best viewpoint for each idea, and preference criterion for each evaluator, to obtain a set of frontier ideas. Experimental results using real datasets containing numerous ideas or designs demonstrate that the proposed approach can effectively prioritize ideas from multiple viewpoints, thereby detecting frontier ideas. The embeddings of ideas learned by the proposed approach provide a visualization that facilitates observation of the frontier ideas. In addition, the proposed approach prioritizes ideas from a wider variety of viewpoints, whereas the baselines tend to use to the same viewpoints; it can also handle various viewpoints and prioritize ideas in situations where only a limited number of evaluators or labels are available.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">集団意思決定支援に関する論文がHCOMP 2020に採択されました🥳 評価者の価値観の多様性に配慮して、一対比較結果を集約する手法を提案しています <a href="https://t.co/syZY6RdLpD">https://t.co/syZY6RdLpD</a> <a href="https://t.co/6HdkWF5lTg">pic.twitter.com/6HdkWF5lTg</a></p>&mdash; Yukino Baba (@yukino) <a href="https://twitter.com/yukino/status/1291540778737573888?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. HooliGAN: Robust, High Quality Neural Vocoding

Ollie McCarthy, Zohaib Ahmed

- retweets: 11, favorites: 53 (08/08/2020 09:03:07)

- links: [abs](https://arxiv.org/abs/2008.02493) | [pdf](https://arxiv.org/pdf/2008.02493)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Recent developments in generative models have shown that deep learning combined with traditional digital signal processing (DSP) techniques could successfully generate convincing violin samples [1], that source-excitation combined with WaveNet yields high-quality vocoders [2, 3] and that generative adversarial network (GAN) training can improve naturalness [4, 5]. By combining the ideas in these models we introduce HooliGAN, a robust vocoder that has state of the art results, finetunes very well to smaller datasets (<30 minutes of speechdata) and generates audio at 2.2MHz on GPU and 35kHz on CPU. We also show a simple modification to Tacotron-basedmodels that allows seamless integration with HooliGAN. Results from our listening tests show the proposed model's ability to consistently output high-quality audio with a variety of datasets, big and small. We provide samples at the following demo page: https://resemble-ai.github.io/hooligan_demo/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">HooliGAN: Robust, High Quality Neural Vocoding<br>pdf: <a href="https://t.co/BzruIZiW0E">https://t.co/BzruIZiW0E</a><br>abs: <a href="https://t.co/4fqotetdHd">https://t.co/4fqotetdHd</a><br>project page: <a href="https://t.co/vvEgfusY6p">https://t.co/vvEgfusY6p</a> <a href="https://t.co/OHI33nEXGf">pic.twitter.com/OHI33nEXGf</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1291548503823482881?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. CaSPR: Learning Canonical Spatiotemporal Point Cloud Representations

Davis Rempe, Tolga Birdal, Yongheng Zhao, Zan Gojcic, Srinath Sridhar, Leonidas J. Guibas

- retweets: 5, favorites: 55 (08/08/2020 09:03:07)

- links: [abs](https://arxiv.org/abs/2008.02792) | [pdf](https://arxiv.org/pdf/2008.02792)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose CaSPR, a method to learn object-centric canonical spatiotemporal point cloud representations of dynamically moving or evolving objects. Our goal is to enable information aggregation over time and the interrogation of object state at any spatiotemporal neighborhood in the past, observed or not. Different from previous work, CaSPR learns representations that support spacetime continuity, are robust to variable and irregularly spacetime-sampled point clouds, and generalize to unseen object instances. Our approach divides the problem into two subtasks. First, we explicitly encode time by mapping an input point cloud sequence to a spatiotemporally-canonicalized object space. We then leverage this canonicalization to learn a spatiotemporal latent representation using neural ordinary differential equations and a generative model of dynamically evolving shapes using continuous normalizing flows. We demonstrate the effectiveness of our method on several applications including shape reconstruction, camera pose estimation, continuous spatiotemporal sequence reconstruction, and correspondence estimation from irregularly or intermittently sampled observations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to announce CaSPR, a Swiss army knife for the perception of dynamic objects. We use Neural ODEs and Continuous Normalizing Flows to learn CAnonical Spatiotemporal Point Cloud Representations.<a href="https://t.co/aDuGTkB50v">https://t.co/aDuGTkB50v</a><br>Kudos to <a href="https://twitter.com/davrempe?ref_src=twsrc%5Etfw">@davrempe</a> <a href="https://twitter.com/drsrinathsridha?ref_src=twsrc%5Etfw">@drsrinathsridha</a> <a href="https://twitter.com/ZGojcic?ref_src=twsrc%5Etfw">@ZGojcic</a> <a href="https://twitter.com/hashtag/guibaslab?src=hash&amp;ref_src=twsrc%5Etfw">#guibaslab</a> <a href="https://t.co/PHZuf9MCjv">pic.twitter.com/PHZuf9MCjv</a></p>&mdash; Tolga Birdal (@tolga_birdal) <a href="https://twitter.com/tolga_birdal/status/1291621548676325376?ref_src=twsrc%5Etfw">August 7, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



