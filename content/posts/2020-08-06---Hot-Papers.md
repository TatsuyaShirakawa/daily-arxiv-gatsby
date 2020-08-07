---
title: Hot Papers 2020-08-06
date: 2020-08-07T09:11:52.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-06"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-06"
socialImage: "/media/flying-marine.jpg"

---

# 1. Hopfield Networks is All You Need

Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber, Markus Holzleitner, Milena Pavlović, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter

- retweets: 435, favorites: 1918 (08/07/2020 09:11:52)

- links: [abs](https://arxiv.org/abs/2008.02217) | [pdf](https://arxiv.org/pdf/2008.02217)
- [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We show that the transformer attention mechanism is the update rule of a modern Hopfield network with continuous states. This new Hopfield network can store exponentially (with the dimension) many patterns, converges with one update, and has exponentially small retrieval errors. The number of stored patterns is traded off against convergence speed and retrieval error. The new Hopfield network has three types of energy minima (fixed points of the update): (1) global fixed point averaging over all patterns, (2) metastable states averaging over a subset of patterns, and (3) fixed points which store a single pattern. Transformer and BERT models operate in their first layers preferably in the global averaging regime, while they operate in higher layers in metastable states. The gradient in transformers is maximal for metastable states, is uniformly distributed for global averaging, and vanishes for a fixed point near a stored pattern. Using the Hopfield network interpretation, we analyzed learning of transformer and BERT models. Learning starts with attention heads that average and then most of them switch to metastable states. However, the majority of heads in the first layers still averages and can be replaced by averaging, e.g. our proposed Gaussian weighting. In contrast, heads in the last layers steadily learn and seem to use metastable states to collect information created in lower layers. These heads seem to be a promising target for improving transformers. Neural networks with Hopfield networks outperform other methods on immune repertoire classification, where the Hopfield net stores several hundreds of thousands of patterns. We provide a new PyTorch layer called "Hopfield", which allows to equip deep learning architectures with modern Hopfield networks as a new powerful concept comprising pooling, memory, and attention. GitHub: https://github.com/ml-jku/hopfield-layers

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-attention mechanism can be viewed as the update rule of a Hopfield network with continuous states.<br><br>Deep learning models can take advantage of Hopfield networks as a powerful concept comprising pooling, memory, and attention.<a href="https://t.co/FL8PimjVo9">https://t.co/FL8PimjVo9</a><a href="https://t.co/HT79M95lkn">https://t.co/HT79M95lkn</a> <a href="https://t.co/Ld2eioVsDG">pic.twitter.com/Ld2eioVsDG</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1291250453263441922?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hopfield Networks is All You Need<br><br>Shows that attention mechanism of transformers is equivalent to the update rule of a modern Hopfield network with continuous states. <a href="https://t.co/gfvAgEeZM4">https://t.co/gfvAgEeZM4</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1291184750837690371?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Attention mechanism of transformers is equivalent to the update rule of a modern Hopfield network with continuous states! <br><br>Proud to announce the latest groundbreaking paper by Sepp Hochreiter team and our <a href="https://twitter.com/hashtag/IARAI?src=hash&amp;ref_src=twsrc%5Etfw">#IARAI</a> colleagues!<br>👉<a href="https://t.co/oFxpw7EPkk">https://t.co/oFxpw7EPkk</a><a href="https://twitter.com/hashtag/deeplearning?src=hash&amp;ref_src=twsrc%5Etfw">#deeplearning</a> <a href="https://twitter.com/hashtag/ai?src=hash&amp;ref_src=twsrc%5Etfw">#ai</a> <a href="https://twitter.com/LITAILab?ref_src=twsrc%5Etfw">@LITAILab</a> <a href="https://t.co/PwqZnBxv4E">pic.twitter.com/PwqZnBxv4E</a></p>&mdash; IARAI (@IARAInews) <a href="https://twitter.com/IARAInews/status/1291304013292609536?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hopfield Networks is All You Need<br>pdf: <a href="https://t.co/SWFnVFNS8h">https://t.co/SWFnVFNS8h</a><br>abs: <a href="https://t.co/erpgXRmPqJ">https://t.co/erpgXRmPqJ</a><br>github: <a href="https://t.co/MWrtQlsNNO">https://t.co/MWrtQlsNNO</a> <a href="https://t.co/0VmtHZK9QX">pic.twitter.com/0VmtHZK9QX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1291199195748208641?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Great paper but the elephant in the room is... Shouldn&#39;t it be &quot;Hopfield Networks *are* All You Need&quot;?<a href="https://t.co/kqOPs4Yxkr">https://t.co/kqOPs4Yxkr</a></p>&mdash; Tiago Ramalho (@tmramalho) <a href="https://twitter.com/tmramalho/status/1291232944598552576?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo  Collections

Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, Daniel Duckworth

- retweets: 102, favorites: 426 (08/07/2020 09:11:53)

- links: [abs](https://arxiv.org/abs/2008.02268) | [pdf](https://arxiv.org/pdf/2008.02268)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a learning-based method for synthesizing novel views of complex outdoor scenes using only unstructured collections of in-the-wild photographs. We build on neural radiance fields (NeRF), which uses the weights of a multilayer perceptron to implicitly model the volumetric density and color of a scene. While NeRF works well on images of static subjects captured under controlled settings, it is incapable of modeling many ubiquitous, real-world phenomena in uncontrolled images, such as variable illumination or transient occluders. In this work, we introduce a series of extensions to NeRF to address these issues, thereby allowing for accurate reconstructions from unstructured image collections taken from the internet. We apply our system, which we dub NeRF-W, to internet photo collections of famous landmarks, thereby producing photorealistic, spatially consistent scene representations despite unknown and confounding factors, resulting in significant improvement over the state of the art.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections<br>pdf: <a href="https://t.co/KHINkovxPi">https://t.co/KHINkovxPi</a><br>abs: <a href="https://t.co/XWSUJZeqMA">https://t.co/XWSUJZeqMA</a><br>project page: <a href="https://t.co/7S682vbr18">https://t.co/7S682vbr18</a> <a href="https://t.co/c3uH73XBJQ">pic.twitter.com/c3uH73XBJQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1291177315351900165?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This project wouldn’t have been possible without my amazing coauthors: <a href="https://twitter.com/rmbrualla?ref_src=twsrc%5Etfw">@rmbrualla</a>, Noha Radwan, Mehdi S. M. Sajjadi, <a href="https://twitter.com/jon_barron?ref_src=twsrc%5Etfw">@jon_barron</a>, and Alexey Dosovitskiy. Check out our paper: <a href="https://t.co/rEWPAGSAxE">https://t.co/rEWPAGSAxE</a></p>&mdash; Daniel Duckworth (@duck) <a href="https://twitter.com/duck/status/1291311679914024962?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Generalized Word Shift Graphs: A Method for Visualizing and Explaining  Pairwise Comparisons Between Texts

Ryan J. Gallagher, Morgan R. Frank, Lewis Mitchell, Aaron J. Schwartz, Andrew J. Reagan, Christopher M. Danforth, Peter Sheridan Dodds

- retweets: 94, favorites: 304 (08/07/2020 09:11:54)

- links: [abs](https://arxiv.org/abs/2008.02250) | [pdf](https://arxiv.org/pdf/2008.02250)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

A common task in computational text analyses is to quantify how two corpora differ according to a measurement like word frequency, sentiment, or information content. However, collapsing the texts' rich stories into a single number is often conceptually perilous, and it is difficult to confidently interpret interesting or unexpected textual patterns without looming concerns about data artifacts or measurement validity. To better capture fine-grained differences between texts, we introduce generalized word shift graphs, visualizations which yield a meaningful and interpretable summary of how individual words contribute to the variation between two texts for any measure that can be formulated as a weighted average. We show that this framework naturally encompasses many of the most commonly used approaches for comparing texts, including relative frequencies, dictionary scores, and entropy-based measures like the Kullback-Leibler and Jensen-Shannon divergences. Through several case studies, we demonstrate how generalized word shift graphs can be flexibly applied across domains for diagnostic investigation, hypothesis generation, and substantive interpretation. By providing a detailed lens into textual shifts between corpora, generalized word shift graphs help computational social scientists, digital humanists, and other text analysis practitioners fashion more robust scientific narratives.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Tired of word clouds? Want to do better sentiment analysis? Not sure how to look at the words underneath your measures?<br><br>Our long overdue paper on generalized word shift graphs is finally here!<a href="https://t.co/lIBXvbMJWX">https://t.co/lIBXvbMJWX</a><a href="https://t.co/vSL1REYT8V">https://t.co/vSL1REYT8V</a><br><br>So what are they?<br><br>1/n <a href="https://t.co/4NM6HoZcGg">pic.twitter.com/4NM6HoZcGg</a></p>&mdash; Ryan J. Gallagher (@ryanjgallag) <a href="https://twitter.com/ryanjgallag/status/1291348764213628930?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Word meaning in minds and machines

Brenden M. Lake, Gregory L. Murphy

- retweets: 19, favorites: 129 (08/07/2020 09:11:54)

- links: [abs](https://arxiv.org/abs/2008.01766) | [pdf](https://arxiv.org/pdf/2008.01766)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Machines show an increasingly broad set of linguistic competencies, thanks to recent progress in Natural Language Processing (NLP). Many algorithms stem from past computational work in psychology, raising the question of whether they understand words as people do. In this paper, we compare how humans and machines represent the meaning of words. We argue that contemporary NLP systems are promising models of human word similarity, but they fall short in many other respects. Current models are too strongly linked to the text-based patterns in large corpora, and too weakly linked to the desires, goals, and beliefs that people use words in order to express. Word meanings must also be grounded in vision and action, and capable of flexible combinations, in ways that current systems are not. We pose concrete challenges for developing machines with a more human-like, conceptual basis for word meaning. We also discuss implications for cognitive science and NLP.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How can we build machines that understand words as people do? Models must look beyond patterns in text to secure a more grounded, conceptual foundation for word meaning, with links to beliefs and desires while supporting flexible composition. w/<a href="https://twitter.com/glmurphy39?ref_src=twsrc%5Etfw">@glmurphy39</a> <a href="https://t.co/9MCDl4bMUV">https://t.co/9MCDl4bMUV</a></p>&mdash; Brenden Lake (@LakeBrenden) <a href="https://twitter.com/LakeBrenden/status/1291351434093563904?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Differentially Private Accelerated Optimization Algorithms

Nurdan Kuru, Ş. İlker Birbil, Mert Gurbuzbalaban, Sinan Yildirim

- retweets: 7, favorites: 97 (08/07/2020 09:11:54)

- links: [abs](https://arxiv.org/abs/2008.01989) | [pdf](https://arxiv.org/pdf/2008.01989)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [math.OC](https://arxiv.org/list/math.OC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We present two classes of differentially private optimization algorithms derived from the well-known accelerated first-order methods. The first algorithm is inspired by Polyak's heavy ball method and employs a smoothing approach to decrease the accumulated noise on the gradient steps required for differential privacy. The second class of algorithms are based on Nesterov's accelerated gradient method and its recent multi-stage variant. We propose a noise dividing mechanism for the iterations of Nesterov's method in order to improve the error behavior of the algorithm. The convergence rate analyses are provided for both the heavy ball and the Nesterov's accelerated gradient method with the help of the dynamical system analysis techniques. Finally, we conclude with our numerical experiments showing that the presented algorithms have advantages over the well-known differentially private algorithms.

<blockquote class="twitter-tweet"><p lang="tr" dir="ltr">Mahremiyet gözeten optimizasyon algoritmaları üzerine yazdığımız makaleyi de açık erişime koyduk.<a href="https://t.co/8Tc0i3DMe3">https://t.co/8Tc0i3DMe3</a></p>&mdash; İlker Birbil (@sibirbil) <a href="https://twitter.com/sibirbil/status/1291264757530492928?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. MuseGAN: Multi-track Sequential Generative Adversarial Networks for  Symbolic Music Generation and Accompaniment

Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, Yi-Hsuan Yang

- retweets: 15, favorites: 70 (08/07/2020 09:11:54)

- links: [abs](https://arxiv.org/abs/1709.06298) | [pdf](https://arxiv.org/pdf/1709.06298)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Generating music has a few notable differences from generating images and videos. First, music is an art of time, necessitating a temporal model. Second, music is usually composed of multiple instruments/tracks with their own temporal dynamics, but collectively they unfold over time interdependently. Lastly, musical notes are often grouped into chords, arpeggios or melodies in polyphonic music, and thereby introducing a chronological ordering of notes is not naturally suitable. In this paper, we propose three models for symbolic multi-track music generation under the framework of generative adversarial networks (GANs). The three models, which differ in the underlying assumptions and accordingly the network architectures, are referred to as the jamming model, the composer model and the hybrid model. We trained the proposed models on a dataset of over one hundred thousand bars of rock music and applied them to generate piano-rolls of five tracks: bass, drums, guitar, piano and strings. A few intra-track and inter-track objective metrics are also proposed to evaluate the generative results, in addition to a subjective user study. We show that our models can generate coherent music of four bars right from scratch (i.e. without human inputs). We also extend our models to human-AI cooperative music generation: given a specific track composed by human, we can generate four additional tracks to accompany it. All code, the dataset and the rendered audio samples are available at https://salu133445.github.io/musegan/ .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Some projects that really do produce music with GANs:<br><br>SeqGAN: <a href="https://t.co/LMDaba0NSA">https://t.co/LMDaba0NSA</a><br>MuseGAN: <a href="https://t.co/U5nU42Cg9U">https://t.co/U5nU42Cg9U</a><br>MidiNet: <a href="https://t.co/g9s4b8Jb31">https://t.co/g9s4b8Jb31</a><br><br>My comment about the video above is only a sarcastic analogy. The robotic arms video doesn&#39;t use a GAN as far as I know.</p>&mdash; Reza Zadeh (@Reza_Zadeh) <a href="https://twitter.com/Reza_Zadeh/status/945014469813354497?ref_src=twsrc%5Etfw">December 24, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Learning to Denoise Historical Music

Yunpeng Li, Beat Gfeller, Marco Tagliasacchi, Dominik Roblek

- retweets: 12, favorites: 64 (08/07/2020 09:11:54)

- links: [abs](https://arxiv.org/abs/2008.02027) | [pdf](https://arxiv.org/pdf/2008.02027)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose an audio-to-audio neural network model that learns to denoise old music recordings. Our model internally converts its input into a time-frequency representation by means of a short-time Fourier transform (STFT), and processes the resulting complex spectrogram using a convolutional neural network. The network is trained with both reconstruction and adversarial objectives on a synthetic noisy music dataset, which is created by mixing clean music with real noise samples extracted from quiet segments of old recordings. We evaluate our method quantitatively on held-out test examples of the synthetic dataset, and qualitatively by human rating on samples of actual historical recordings. Our results show that the proposed method is effective in removing noise, while preserving the quality and details of the original music.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning to Denoise Historical Music<br>pdf: <a href="https://t.co/xbcZ0MapfH">https://t.co/xbcZ0MapfH</a><br>abs: <a href="https://t.co/UJbYfeNYhy">https://t.co/UJbYfeNYhy</a> <a href="https://t.co/77VsaNAgp4">pic.twitter.com/77VsaNAgp4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1291211086285475841?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Domain-Specific Mappings for Generative Adversarial Style Transfer

Hsin-Yu Chang, Zhixiang Wang, Yung-Yu Chuang

- retweets: 11, favorites: 49 (08/07/2020 09:11:54)

- links: [abs](https://arxiv.org/abs/2008.02198) | [pdf](https://arxiv.org/pdf/2008.02198)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Style transfer generates an image whose content comes from one image and style from the other. Image-to-image translation approaches with disentangled representations have been shown effective for style transfer between two image categories. However, previous methods often assume a shared domain-invariant content space, which could compromise the content representation power. For addressing this issue, this paper leverages domain-specific mappings for remapping latent features in the shared content space to domain-specific content spaces. This way, images can be encoded more properly for style transfer. Experiments show that the proposed method outperforms previous style transfer methods, particularly on challenging scenarios that would require semantic correspondences between images. Code and results are available at https://acht7111020.github.io/DSMAP-demo/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Domain-Specific Mappings for Generative Adversarial Style Transfer<br>pdf: <a href="https://t.co/aw8iZyJyls">https://t.co/aw8iZyJyls</a><br>abs: <a href="https://t.co/U3rlndmPLe">https://t.co/U3rlndmPLe</a><br>project page: <a href="https://t.co/uOOAZdeAqa">https://t.co/uOOAZdeAqa</a> <a href="https://t.co/iLJkkUMDla">pic.twitter.com/iLJkkUMDla</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1291175097315852294?ref_src=twsrc%5Etfw">August 6, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



