---
title: Hot Papers 2020-08-18
date: 2020-08-19T09:42:21.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Learning Gradient Fields for Shape Generation

Ruojin Cai, Guandao Yang, Hadar Averbuch-Elor, Zekun Hao, Serge Belongie, Noah Snavely, Bharath Hariharan

- retweets: 68, favorites: 306 (08/19/2020 09:42:21)

- links: [abs](https://arxiv.org/abs/2008.06520) | [pdf](https://arxiv.org/pdf/2008.06520)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this work, we propose a novel technique to generate shapes from point cloud data. A point cloud can be viewed as samples from a distribution of 3D points whose density is concentrated near the surface of the shape. Point cloud generation thus amounts to moving randomly sampled points to high-density areas. We generate point clouds by performing stochastic gradient ascent on an unnormalized probability density, thereby moving sampled points toward the high-likelihood regions. Our model directly predicts the gradient of the log density field and can be trained with a simple objective adapted from score-based generative models. We show that our method can reach state-of-the-art performance for point cloud auto-encoding and generation, while also allowing for extraction of a high-quality implicit surface. Code is available at https://github.com/RuojinCai/ShapeGF.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Gradient Fields for Shape Generation<br>pdf: <a href="https://t.co/tJHDw8Yh5y">https://t.co/tJHDw8Yh5y</a><br>abs: <a href="https://t.co/mfplJIIWBd">https://t.co/mfplJIIWBd</a><br>github: <a href="https://t.co/Xc4RhA7IXW">https://t.co/Xc4RhA7IXW</a><br>project page: <a href="https://t.co/v9IS6W0rPh">https://t.co/v9IS6W0rPh</a> <a href="https://t.co/RXJWQfqNQA">pic.twitter.com/RXJWQfqNQA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1295527690435530752?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our ECCV paper on 3D generation: Learning Gradient Fields for Shape Generation. <br><br>Arxiv: <a href="https://t.co/wjZvA6I0US">https://t.co/wjZvA6I0US</a><br>Project page: <a href="https://t.co/v0WsHUONt1">https://t.co/v0WsHUONt1</a><br>Code:<a href="https://t.co/ugTa58NxIN">https://t.co/ugTa58NxIN</a><br>Video: <a href="https://t.co/k2FBZthl1G">https://t.co/k2FBZthl1G</a> <br>Long video: <a href="https://t.co/d0pJXIlhAx">https://t.co/d0pJXIlhAx</a> <a href="https://t.co/dc3C8Tej18">pic.twitter.com/dc3C8Tej18</a></p>&mdash; Guandao Yang (@stevenygd) <a href="https://twitter.com/stevenygd/status/1295520418690822146?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Manticore: A 4096-core RISC-V Chiplet Architecture for Ultra-efficient  Floating-point Computing

Florian Zaruba, Fabian Schuiki, Luca Benini

- retweets: 20, favorites: 94 (08/19/2020 09:42:22)

- links: [abs](https://arxiv.org/abs/2008.06502) | [pdf](https://arxiv.org/pdf/2008.06502)
- [cs.AR](https://arxiv.org/list/cs.AR/recent)

Data-parallel problems, commonly found in data analytics, machine learning, and scientific computing demand ever growing floating-point operations per second under tight area- and energy-efficiency constraints. Application-specific architectures and accelerators, while efficient at a given task, are hard to adjust to algorithmic changes. In this work, we present Manticore, a general-purpose, ultra-efficient, RISC-V, chiplet-based architecture for data-parallel floating-point workloads. We have manufactured a 9$\text{mm}^2$ prototype of the chiplet's computational core in Globalfoundries 22nm FD-SOI process and demonstrate more than 2.5$\times$ improvement in energy efficiency on floating-point intensive workloads compared to high performance compute engines (CPUs and GPUs), despite their more advanced FinFET process. The prototype contains two 64-bit, application-class RISC-V Ariane management cores that run a full-fledged Linux OS. The compute capability at high energy and area efficiency is provided by Snitch clusters. Each cluster contains eight small (20kGE) 32-bit integer RISC-V cores, each controlling a large double-precision floating-point unit (120kGE). Each core supports two custom RISC-V ISA extensions: FREP and SSR. The SSR extension elides explicit load and store instructions by encoding them as register reads and writes. The FREP extension mostly decouples the integer core from the FPU by allowing a sequence buffer to issue instructions to the FPU independently. Both extensions allow the tiny, single-issue, integer core to saturate the instruction bandwidth of the FPU and achieve FPU utilization above 90%, with more than 80% of core area dedicated to the FPU.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In HC32, researchers have presented Manticore, a 22FDX general-purpose ultra-efficient <a href="https://twitter.com/hashtag/RISCV?src=hash&amp;ref_src=twsrc%5Etfw">#RISCV</a> chiplet-based architecture for data-parallel floating-point workloads.<a href="https://t.co/rFdANlxxEq">https://t.co/rFdANlxxEq</a> <a href="https://t.co/O8IweO8iYm">pic.twitter.com/O8IweO8iYm</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1295598895679131648?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Computational timeline reconstruction of the stories surrounding Trump:  Story turbulence, narrative control, and collective chronopathy

P. S. Dodds, J. R. Minot, M. V. Arnold, T. Alshaabi, J. L. Adams, A. J. Reagan, C. M. Danforth

- retweets: 24, favorites: 51 (08/19/2020 09:42:22)

- links: [abs](https://arxiv.org/abs/2008.07301) | [pdf](https://arxiv.org/pdf/2008.07301)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Measuring the specific kind, temporal ordering, diversity, and turnover rate of stories surrounding any given subject is essential to developing a complete reckoning of that subject's historical impact. Here, we use Twitter as a distributed news and opinion aggregation source to identify and track the dynamics of the dominant day-scale stories around Donald Trump, the 45th President of the United States. Working with a data set comprising around 20 billion 1-grams, we first compare each day's 1-gram and 2-gram usage frequencies to those of a year before, to create day- and week-scale timelines for Trump stories for 2016 onwards. We measure Trump's narrative control, the extent to which stories have been about Trump or put forward by Trump. We then quantify story turbulence and collective chronopathy -- the rate at which a population's stories for a subject seem to change over time. We show that 2017 was the most turbulent year for Trump, and that story generation slowed dramatically during the COVID-19 pandemic in 2020. Trump story turnover for 2 months during the COVID-19 pandemic was on par with that of 3 days in September 2017. Our methods may be applied to any well-discussed phenomenon, and have potential, in particular, to enable the computational aspects of journalism, history, and biography.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint:<br><br>“Computational timeline reconstruction of the stories surrounding Trump: Story turbulence, narrative control, and collective chronopathy”<a href="https://t.co/4zAfQmjWlW">https://t.co/4zAfQmjWlW</a><br><br>P. S. Dodds, J. R. Minot, M. V. Arnold, T. Alshaabi, J. L. Adams, A. J. Reagan, and C. M. Danforth <a href="https://t.co/tcxNmiB8Ul">pic.twitter.com/tcxNmiB8Ul</a></p>&mdash; ComputationlStoryLab (@compstorylab) <a href="https://twitter.com/compstorylab/status/1295726754070515713?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Did the news cycle really launch into warp speed when Trump was elected?<br><br>Bigly.<br><br>Our latest preprint:<a href="https://t.co/6YVG6jEsaQ">https://t.co/6YVG6jEsaQ</a> <a href="https://t.co/MGf4Tae56Y">pic.twitter.com/MGf4Tae56Y</a></p>&mdash; Chris Danforth (@ChrisDanforth) <a href="https://twitter.com/ChrisDanforth/status/1295752839210377218?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Bounds on the $\mathrm{QAC}^0$ Complexity of Approximating Parity

Gregory Rosenthal

- retweets: 9, favorites: 60 (08/19/2020 09:42:22)

- links: [abs](https://arxiv.org/abs/2008.07470) | [pdf](https://arxiv.org/pdf/2008.07470)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.CC](https://arxiv.org/list/cs.CC/recent)

$\mathrm{QAC}$ circuits are quantum circuits with one-qubit gates and Toffoli gates of arbitrary arity. $\mathrm{QAC}^0$ circuits are $\mathrm{QAC}$ circuits of constant depth, and are quantum analogues of $\mathrm{AC}^0$ circuits. We prove the following:   $\bullet$ For all $d \ge 7$ and $\varepsilon>0$ there is a depth-$d$ $\mathrm{QAC}$ circuit of size $\exp(\mathrm{poly}(n^{1/d}) \log(n/\varepsilon))$ that approximates the $n$-qubit parity function to within error $\varepsilon$ on worst-case quantum inputs. Previously it was unknown whether $\mathrm{QAC}$ circuits of sublogarithmic depth could approximate parity regardless of size.   $\bullet$ We introduce a class of "mostly classical" $\mathrm{QAC}$ circuits, including a major component of our circuit from the above upper bound, and prove a tight lower bound on the size of low-depth, mostly classical $\mathrm{QAC}$ circuits that approximate this component.   $\bullet$ Arbitrary depth-$d$ $\mathrm{QAC}$ circuits require at least $\Omega(n/d)$ multi-qubit gates to achieve a $1/2 + \exp(-o(n/d))$ approximation of parity. When $d = \Theta(\log n)$ this nearly matches an easy $O(n)$ size upper bound for computing parity exactly.   $\bullet$ $\mathrm{QAC}$ circuits with at most two layers of multi-qubit gates cannot achieve a $1/2 + \exp(-o(n))$ approximation of parity, even non-cleanly. Previously it was known only that such circuits could not cleanly compute parity exactly for sufficiently large $n$.   The proofs use a new normal form for quantum circuits which may be of independent interest, and are based on reductions to the problem of constructing certain generalizations of the cat state which we name "nekomata" after an analogous cat y\=okai.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Exciting paper by Gregory Rosenthal (<a href="https://twitter.com/gregrosent?ref_src=twsrc%5Etfw">@gregrosent</a>), a PhD student <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@UofT</a>. Proves new bounds on approximating the parity function with QAC0 circuits, making progress on a long-standing question in quantum circuit complexity. Also, his paper has cool notation. <a href="https://t.co/M5cJvRZOlX">https://t.co/M5cJvRZOlX</a> <a href="https://t.co/ViDFplJmV4">pic.twitter.com/ViDFplJmV4</a></p>&mdash; Henry Yuen (@henryquantum) <a href="https://twitter.com/henryquantum/status/1295571425638711297?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Crossing The Gap: A Deep Dive into Zero-Shot Sim-to-Real Transfer for  Dynamics

Eugene Valassakis, Zihan Ding, Edward Johns

- retweets: 11, favorites: 58 (08/19/2020 09:42:22)

- links: [abs](https://arxiv.org/abs/2008.06686) | [pdf](https://arxiv.org/pdf/2008.06686)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Zero-shot sim-to-real transfer of tasks with complex dynamics is a highly challenging and unsolved problem. A number of solutions have been proposed in recent years, but we have found that many works do not present a thorough evaluation in the real world, or underplay the significant engineering effort and task-specific fine tuning that is required to achieve the published results. In this paper, we dive deeper into the sim-to-real transfer challenge, investigate why this is such a difficult problem, and present objective evaluations of a number of transfer methods across a range of real-world tasks. Surprisingly, we found that a method which simply injects random forces into the simulation performs just as well as more complex methods, such as those which randomise the simulator's dynamics parameters, or adapt a policy online using recurrent network architectures.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Crossing the Gap: A Deep Dive into Zero-Shot Sim-to-Real Transfer for Dynamics<a href="https://t.co/SY98chnp5P">https://t.co/SY98chnp5P</a> <a href="https://t.co/qDqDoKkpB0">pic.twitter.com/qDqDoKkpB0</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1295549044794695680?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">For simple robotics tasks (reach, push, slide) injecting random noise performs as well as more complicated domain randomization: <a href="https://t.co/FvfSWyumgn">https://t.co/FvfSWyumgn</a><br>Suspicion: all these methods are basically just inducing feedback so additional complexity doesn&#39;t help</p>&mdash; Eugene Vinitsky (@EugeneVinitsky) <a href="https://twitter.com/EugeneVinitsky/status/1295556127917314050?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Is Supervised Syntactic Parsing Beneficial for Language Understanding?  An Empirical Investigation

Goran Glavaš, Ivan Vulić

- retweets: 13, favorites: 55 (08/19/2020 09:42:23)

- links: [abs](https://arxiv.org/abs/2008.06788) | [pdf](https://arxiv.org/pdf/2008.06788)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Traditional NLP has long held (supervised) syntactic parsing necessary for successful higher-level language understanding. The recent advent of end-to-end neural language learning, self-supervised via language modeling (LM), and its success on a wide range of language understanding tasks, however, questions this belief. In this work, we empirically investigate the usefulness of supervised parsing for semantic language understanding in the context of LM-pretrained transformer networks. Relying on the established fine-tuning paradigm, we first couple a pretrained transformer with a biaffine parsing head, aiming to infuse explicit syntactic knowledge from Universal Dependencies (UD) treebanks into the transformer. We then fine-tune the model for language understanding (LU) tasks and measure the effect of the intermediate parsing training (IPT) on downstream LU performance. Results from both monolingual English and zero-shot language transfer experiments (with intermediate target-language parsing) show that explicit formalized syntax, injected into transformers through intermediate supervised parsing, has very limited and inconsistent effect on downstream LU performance. Our results, coupled with our analysis of transformers' representation spaces before and after intermediate parsing, make a significant step towards providing answers to an essential question: how (un)availing is supervised parsing for high-level semantic language understanding in the era of large neural models?

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work (w. <a href="https://twitter.com/licwu?ref_src=twsrc%5Etfw">@licwu</a>): <a href="https://t.co/QaJmQRZRnd">https://t.co/QaJmQRZRnd</a><br>We know from recent work that pretrained Transformers implicitly encode some kind of syntax. But does this implicit syntax render formal/explicit syntax (i.e., treebanks and supervised parsing) unnecessary for language understanding?</p>&mdash; Goran Glavaš (@gg42554) <a href="https://twitter.com/gg42554/status/1295739316308672512?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Image Stylization for Robust Features

Iaroslav Melekhov, Gabriel J. Brostow, Juho Kannala, Daniyar Turmukhambetov

- retweets: 9, favorites: 50 (08/19/2020 09:42:23)

- links: [abs](https://arxiv.org/abs/2008.06959) | [pdf](https://arxiv.org/pdf/2008.06959)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Local features that are robust to both viewpoint and appearance changes are crucial for many computer vision tasks. In this work we investigate if photorealistic image stylization improves robustness of local features to not only day-night, but also weather and season variations. We show that image stylization in addition to color augmentation is a powerful method of learning robust features. We evaluate learned features on visual localization benchmarks, outperforming state of the art baseline models despite training without ground-truth 3D correspondences using synthetic homographies only.   We use trained feature networks to compete in Long-Term Visual Localization and Map-based Localization for Autonomous Driving challenges achieving competitive scores.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Image Stylization for Robust Features<br><br>TLDR: image stylization, in addition to color augmentation, is a powerful method of learning robust visual features<br><br>Creative use of style transfer Awesome work from <a href="https://twitter.com/iMelekhov?ref_src=twsrc%5Etfw">@iMelekhov</a> and <a href="https://twitter.com/NianticLabs?ref_src=twsrc%5Etfw">@NianticLabs</a> <a href="https://t.co/X0I5f1yZoP">https://t.co/X0I5f1yZoP</a><a href="https://twitter.com/hashtag/ComputerVision?src=hash&amp;ref_src=twsrc%5Etfw">#ComputerVision</a> <a href="https://t.co/P3rVq2NCUY">pic.twitter.com/P3rVq2NCUY</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1295794106657058817?ref_src=twsrc%5Etfw">August 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



