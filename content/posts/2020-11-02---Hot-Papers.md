---
title: Hot Papers 2020-11-02
date: 2020-11-04T08:08:16.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-02"
socialImage: "/media/flying-marine.jpg"

---

# 1. Targeting for long-term outcomes

Jeremy Yang, Dean Eckles, Paramveer Dhillon, Sinan Aral

- retweets: 1162, favorites: 146 (11/04/2020 08:08:16)

- links: [abs](https://arxiv.org/abs/2010.15835) | [pdf](https://arxiv.org/pdf/2010.15835)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.AP](https://arxiv.org/list/stat.AP/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Decision-makers often want to target interventions (e.g., marketing campaigns) so as to maximize an outcome that is observed only in the long-term. This typically requires delaying decisions until the outcome is observed or relying on simple short-term proxies for the long-term outcome. Here we build on the statistical surrogacy and off-policy learning literature to impute the missing long-term outcomes and then approximate the optimal targeting policy on the imputed outcomes via a doubly-robust approach. We apply our approach in large-scale proactive churn management experiments at The Boston Globe by targeting optimal discounts to its digital subscribers to maximize their long-term revenue. We first show that conditions for validity of average treatment effect estimation with imputed outcomes are also sufficient for valid policy evaluation and optimization; furthermore, these conditions can be somewhat relaxed for policy optimization. We then validate this approach empirically by comparing it with a policy learned on the ground truth long-term outcomes and show that they are statistically indistinguishable. Our approach also outperforms a policy learned on short-term proxies for the long-term outcome. In a second field experiment, we implement the optimal targeting policy with additional randomized exploration, which allows us to update the optimal policy for each new cohort of customers to account for potential non-stationarity. Over three years, our approach had a net-positive revenue impact in the range of $4-5 million compared to The Boston Globe's current policies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🎯How can we learn to target interventions when we care about outcomes that are only observed after a delay?🗓️<br><br>This comes up in settings as varied as clinical trials (5 yr all-cause mortality) and marketing (customer lifetime value).<br><br>Our new paper:<a href="https://t.co/0TZNjc1phT">https://t.co/0TZNjc1phT</a> <a href="https://t.co/Pa4nVfUg8N">pic.twitter.com/Pa4nVfUg8N</a></p>&mdash; Dean Eckles (@deaneckles) <a href="https://twitter.com/deaneckles/status/1323396125601210372?ref_src=twsrc%5Etfw">November 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



