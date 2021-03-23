---
title: Hot Papers 2021-03-22
date: 2021-03-23T09:58:57.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-22"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-22"
socialImage: "/media/flying-marine.jpg"

---

# 1. ConViT: Improving Vision Transformers with Soft Convolutional Inductive  Biases

Stéphane d'Ascoli, Hugo Touvron, Matthew Leavitt, Ari Morcos, Giulio Biroli, Levent Sagun

- retweets: 1105, favorites: 178 (03/23/2021 09:58:57)

- links: [abs](https://arxiv.org/abs/2103.10697) | [pdf](https://arxiv.org/pdf/2103.10697)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Convolutional architectures have proven extremely successful for vision tasks. Their hard inductive biases enable sample-efficient learning, but come at the cost of a potentially lower performance ceiling. Vision Transformers (ViTs) rely on more flexible self-attention layers, and have recently outperformed CNNs for image classification. However, they require costly pre-training on large external datasets or distillation from pre-trained convolutional networks. In this paper, we ask the following question: is it possible to combine the strengths of these two architectures while avoiding their respective limitations? To this end, we introduce gated positional self-attention (GPSA), a form of positional self-attention which can be equipped with a "soft" convolutional inductive bias. We initialize the GPSA layers to mimic the locality of convolutional layers, then give each attention head the freedom to escape locality by adjusting a gating parameter regulating the attention paid to position versus content information. The resulting convolutional-like ViT architecture, ConViT, outperforms the DeiT on ImageNet, while offering a much improved sample efficiency. We further investigate the role of locality in learning by first quantifying how it is encouraged in vanilla self-attention layers, then analyzing how it is escaped in GPSA layers. We conclude by presenting various ablations to better understand the success of the ConViT. Our code and models are released publicly.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases<br><br>ConViT outperforms the DeiT on ImageNet, while offering a much improved sample efficiency.<br><br>abs: <a href="https://t.co/6zy7QBYJZD">https://t.co/6zy7QBYJZD</a><br>code: <a href="https://t.co/CjOreFrIuo">https://t.co/CjOreFrIuo</a> <a href="https://t.co/00VRLXR6sn">pic.twitter.com/00VRLXR6sn</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1373797309314605059?ref_src=twsrc%5Etfw">March 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases<br>pdf: <a href="https://t.co/gOtYCuubtC">https://t.co/gOtYCuubtC</a><br>abs: <a href="https://t.co/WxfHdbasA5">https://t.co/WxfHdbasA5</a> <a href="https://t.co/p0SakZ5Yho">pic.twitter.com/p0SakZ5Yho</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1373798213979889666?ref_src=twsrc%5Etfw">March 22, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



