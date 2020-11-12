---
title: Hot Papers 2020-11-12
date: 2020-11-13T08:40:46.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. Adversarial images for the primate brain

Li Yuan, Will Xiao, Gabriel Kreiman, Francis E.H. Tay, Jiashi Feng, Margaret S. Livingstone

- retweets: 240, favorites: 92 (11/13/2020 08:40:46)

- links: [abs](https://arxiv.org/abs/2011.05623) | [pdf](https://arxiv.org/pdf/2011.05623)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Deep artificial neural networks have been proposed as a model of primate vision. However, these networks are vulnerable to adversarial attacks, whereby introducing minimal noise can fool networks into misclassifying images. Primate vision is thought to be robust to such adversarial images. We evaluated this assumption by designing adversarial images to fool primate vision. To do so, we first trained a model to predict responses of face-selective neurons in macaque inferior temporal cortex. Next, we modified images, such as human faces, to match their model-predicted neuronal responses to a target category, such as monkey faces. These adversarial images elicited neuronal responses similar to the target category. Remarkably, the same images fooled monkeys and humans at the behavioral level. These results challenge fundamental assumptions about the similarity between computer and primate vision and show that a model of neuronal activity can selectively direct primate visual behavior.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Adversarial images for the primate brain&quot; (!)<br><br>1. Train a model to predict responses of face-selective neurons<br><br>2. Modify images to fool that model <br><br>3. Test actual behavioral &amp; neural responses to those images<br><br>Voila: Macaques (and humans) are fooled!<a href="https://t.co/FfoN8zbPvN">https://t.co/FfoN8zbPvN</a> <a href="https://t.co/E38aJfLRjQ">pic.twitter.com/E38aJfLRjQ</a></p>&mdash; Chaz Firestone (@chazfirestone) <a href="https://twitter.com/chazfirestone/status/1326752533176606720?ref_src=twsrc%5Etfw">November 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by  Transferring from GANs

Yaxing Wang, Lu Yu, Joost van de Weijer

- retweets: 56, favorites: 45 (11/13/2020 08:40:47)

- links: [abs](https://arxiv.org/abs/2011.05867) | [pdf](https://arxiv.org/pdf/2011.05867)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image-to-image translation has recently achieved remarkable results. But despite current success, it suffers from inferior performance when translations between classes require large shape changes. We attribute this to the high-resolution bottlenecks which are used by current state-of-the-art image-to-image methods. Therefore, in this work, we propose a novel deep hierarchical Image-to-Image Translation method, called DeepI2I. We learn a model by leveraging hierarchical features: (a) structural information contained in the shallow layers and (b) semantic information extracted from the deep layers. To enable the training of deep I2I models on small datasets, we propose a novel transfer learning method, that transfers knowledge from pre-trained GANs. Specifically, we leverage the discriminator of a pre-trained GANs (i.e. BigGAN or StyleGAN) to initialize both the encoder and the discriminator and the pre-trained generator to initialize the generator of our model. Applying knowledge transfer leads to an alignment problem between the encoder and generator. We introduce an adaptor network to address this. On many-class image-to-image translation on three datasets (Animal faces, Birds, and Foods) we decrease mFID by at least 35% when compared to the state-of-the-art. Furthermore, we qualitatively and quantitatively demonstrate that transfer learning significantly improves the performance of I2I systems, especially for small datasets. Finally, we are the first to perform I2I translations for domains with over 100 classes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by Transferring from GANs<br>pdf: <a href="https://t.co/qSG98rl2LZ">https://t.co/qSG98rl2LZ</a><br>abs: <a href="https://t.co/d2VoOoTpk9">https://t.co/d2VoOoTpk9</a> <a href="https://t.co/univDmJcFP">pic.twitter.com/univDmJcFP</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1326710310544224256?ref_src=twsrc%5Etfw">November 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Don't Read Too Much into It: Adaptive Computation for Open-Domain  Question Answering

Yuxiang Wu, Sebastian Riedel, Pasquale Minervini, Pontus Stenetorp

- retweets: 32, favorites: 45 (11/13/2020 08:40:47)

- links: [abs](https://arxiv.org/abs/2011.05435) | [pdf](https://arxiv.org/pdf/2011.05435)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Most approaches to Open-Domain Question Answering consist of a light-weight retriever that selects a set of candidate passages, and a computationally expensive reader that examines the passages to identify the correct answer. Previous works have shown that as the number of retrieved passages increases, so does the performance of the reader. However, they assume all retrieved passages are of equal importance and allocate the same amount of computation to them, leading to a substantial increase in computational cost. To reduce this cost, we propose the use of adaptive computation to control the computational budget allocated for the passages to be read. We first introduce a technique operating on individual passages in isolation which relies on anytime prediction and a per-layer estimation of an early exit probability. We then introduce SkylineBuilder, an approach for dynamically deciding on which passage to allocate computation at each step, based on a resource allocation policy trained via reinforcement learning. Our results on SQuAD-Open show that adaptive computation with global prioritisation improves over several strong static and adaptive methods, leading to a 4.3x reduction in computation while retaining 95% performance of the full model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Don’t Read Too Much Into It: Adaptive Computation for Open-Domain Question Answering<br><br>Achieves 4.3x reduction in computation for fine-tuning a LM on SQuAD-Open while retaining 95% performance of the full model. <a href="https://t.co/7yKf3PncAp">https://t.co/7yKf3PncAp</a> <a href="https://t.co/fZtKBv8SxO">pic.twitter.com/fZtKBv8SxO</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1326713763479785472?ref_src=twsrc%5Etfw">November 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. End-to-End Chinese Landscape Painting Creation Using Generative  Adversarial Networks

Alice Xue

- retweets: 20, favorites: 48 (11/13/2020 08:40:47)

- links: [abs](https://arxiv.org/abs/2011.05552) | [pdf](https://arxiv.org/pdf/2011.05552)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Current GAN-based art generation methods produce unoriginal artwork due to their dependence on conditional input. Here, we propose Sketch-And-Paint GAN (SAPGAN), the first model which generates Chinese landscape paintings from end to end, without conditional input. SAPGAN is composed of two GANs: SketchGAN for generation of edge maps, and PaintGAN for subsequent edge-to-painting translation. Our model is trained on a new dataset of traditional Chinese landscape paintings never before used for generative research. A 242-person Visual Turing Test study reveals that SAPGAN paintings are mistaken as human artwork with 55% frequency, significantly outperforming paintings from baseline GANs. Our work lays a groundwork for truly machine-original art generation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">End-to-End Chinese Landscape Painting Creation Using<br>Generative Adversarial Networks<br>pdf: <a href="https://t.co/w8RaBfc6A7">https://t.co/w8RaBfc6A7</a><br>abs: <a href="https://t.co/qQ3L0GNJU2">https://t.co/qQ3L0GNJU2</a> <a href="https://t.co/89vJLMs2V1">pic.twitter.com/89vJLMs2V1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1326714584137420805?ref_src=twsrc%5Etfw">November 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. The Impact of Text Presentation on Translator Performance

Samuel Läubli, Patrick Simianer, Joern Wuebker, Geza Kovacs, Rico Sennrich, Spence Green

- retweets: 42, favorites: 21 (11/13/2020 08:40:47)

- links: [abs](https://arxiv.org/abs/2011.05978) | [pdf](https://arxiv.org/pdf/2011.05978)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

Widely used computer-aided translation (CAT) tools divide documents into segments such as sentences and arrange them in a side-by-side, spreadsheet-like view. We present the first controlled evaluation of these design choices on translator performance, measuring speed and accuracy in three experimental text processing tasks. We find significant evidence that sentence-by-sentence presentation enables faster text reproduction and within-sentence error identification compared to unsegmented text, and that a top-and-bottom arrangement of source and target sentences enables faster text reproduction compared to a side-by-side arrangement. For revision, on the other hand, our results suggest that presenting unsegmented text results in the highest accuracy and time efficiency. Our findings have direct implications for best practices in designing CAT tools.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The way in which CAT tools divide documents into segments and arrange these segments on your screen probably impact your productivity. Check out the results of our full-day experiment with 20 professional translators: <a href="https://t.co/89cxQ53YF4">https://t.co/89cxQ53YF4</a></p>&mdash; Samuel Läubli (@samlaeubli) <a href="https://twitter.com/samlaeubli/status/1326790185233866752?ref_src=twsrc%5Etfw">November 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



