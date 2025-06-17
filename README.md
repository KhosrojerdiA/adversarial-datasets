# Datasets for Supervised Adversarial Attacks on Neural Rankers

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

![](images/Slide1.png)
<p align="center"><em>Model Architecture.</em></p>

## Overview

This repository contains the code, datasets, and experimental results associated with our paper:

"Datasets for Supervised Adversarial Attacks on Neural Rankers"

In this paper, we explore adversarial rank attacks on Neural Ranking Models (NRMs) using a structured framework based on Retrieval-Augmented Generation (RAG). Our approach generates adversarial datasets by strategically modifying documents with adversarially generated sentences to manipulate ranking results in Information Retrieval (IR) systems.

To implement our Retrieval-Augmented Generation (RAG) based adversarial dataset construction, we leveraged Qwen3 32B,a state-of-the-art large language model (LLM) optimized for complex language understanding and generation with chain of thoughts. The model was selected due to its strong performance in generating semantically coherent, reasoning, context-aware adversarial modifications while maintaining fluency and linguistic integrity. Below, we outline the key components of our implementation pipeline.

** Please check prompts folder to see our LLM Prompts **

** You can check the datasets in datasets folder **

## Abstract

Neural Ranking Models (NRMs) are vulnerable to adversarial attacks that manipulate document rankings, threatening the integrity of retrieval systems. Existing adversarial attacks rely on unsupervised methods and surrogate models, which limit their generalizability. In this work, we introduce a novel RAG-based dataset construction framework that employs Large Language Models (LLMs) to generate adversarially modified documents optimized for rank manipulation. Our dataset is released in two variations: Gold, and Diamond, categorized by attack effectiveness. The proposed approach facilitates robust evaluation of adversarial resilience in IR systems and the development of defense strategies against ranking attacks.

## Key Contributions

- A supervised adversarial dataset for benchmarking adversarial attacks against NRMs.
- An iterative self-refinement process using an LLM-NRM feedback loop to generate high-impact adversarial modifications.
- Two dataset variations (Gold, Diamond) categorized by ranking impact.
- Comprehensive evaluation against state-of-the-art adversarial attack methods on IR systems.

## Installation
To set up the project environment, run the following:

```
conda create -n rag_attack_env python=3.10
conda activate rag_attack_env
pip install -r requirements.txt
```

## Run
To run the code, you can use main_no_think_v2.py for no reasoning LLM. It will save the result on output folder. 
Then you can run main_think_v2.py that will run the reasoning LLM for those query-documents that did not achieve the goal.


## Source Data
Following Wu et al., we evaluate our attack methods on a randomly sampled subset of 1,000 queries from the Dev set. For each query, we target two distinct types of documents—Easy-5 and Hard-5—selected from the re-ranked results produced by the victim neural ranking model (NRM) after applying it to the top-1K BM25 candidates. This dual-target approach allows us to systematically assess the impact of our rank boosting techniques on documents with varying levels of initial ranking quality.

## Target Document Groups
- Easy-5: This group consists of five documents initially ranked between positions 51 and 100 in the search results. Specifically, one document is randomly sampled from every ten-ranked positions within this range (e.g., ranks 51, 63, 76, 84, and 91). By targeting these mid-ranked documents, we aim to evaluate how our augmentation strategies enhance the visibility of documents that are neither highly ranked nor too obscure.
- Hard-5: In contrast, this group comprises the five lowest-ranked documents from the re-ranked list, representing the most challenging cases for rank boosting. By focusing on these least visible documents, we critically examine the robustness of our augmentation approach when applied to content with minimal initial exposure.

## Datasets Variation
Our datasets are categorized as follows:
- Gold: Selects the best-performing adversarial modifications per document-query pair.
- Diamond: Strictly filters cases where documents achieve a rank of ≤10 for mid-ranked (Easy-5) and ≤50 for lower-ranked (Hard-5) documents.


## Attack Performance Evaluation Metrics
- Attack Success Rate: The percentage of cases where the target document’s rank improves following augmentation.
- %r ≤ 10: The proportion of documents that achieve a final ranking within the top 10 results, indicating a high level of retrieval performance.
- %r ≤ 50: The proportion of documents that move to rank 50 or better, providing insight into broader ranking improvements.
- Boost: The average rank increase for the document post augmentation, quantifying the magnitude of improvement.
- Adv Rank: The final average rank of the document after augmentation, serving as a summary measure of the method’s overall effectiveness.
- Perplexity: A measure of text fluency, with lower values indicating more predictable and coherent language after augmentation.
- Acceptability Score: A measure of perceived text quality, reflecting how natural or readable the augmented document is compared to the original document.

## Table 1: Gold Dataset statistics over Easy-5 and Hard-5.

| Group   | Type | SentPos | Counts | ASR  | %r ≤ 10 | %r ≤ 50 | Boost | PPL  | AcS |
|---------|------|---------|--------|------|---------|---------|-------|------|-----|
| Orig.   |      | –       | 5,000  | –    | –       | –       | –     | 37.3 | 0.78|
| Easy-5  | Adv. | all     | 5,000  | 99.7 | 62.2    | 93.1    | 59.2  | 42.4 | 0.77|
|         |      | v=0     | 3,134  | 100.0| 72.9    | 96.6    | 63.2  | 42.2 | 0.77|
|         |      | v=1     | 961    | 99.7 | 54.4    | 92.5    | 57.7  | 42.7 | 0.77|
|         |      | v=2     | 371    | 99.7 | 44.5    | 87.9    | 51.6  | 40.0 | 0.76|
|         |      | v=3     | 163    | 99.4 | 33.1    | 85.9    | 50.6  | 40.4 | 0.73|
|         |      | v≥4     | 371    | 98.1 | 22.4    | 72.5    | 40.5  | 46.5 | 0.77|
| Orig.   |      | –       | 5,000  | –    | –       | –       | –     | 51.4 | 0.72|
| Hard-5  | Adv. | all     | 5,000  | 99.9 | 29.1    | 49.4    | 781.1 | 67.4 | 0.72|
|         |      | v=0     | 3,281  | 99.9 | 34.5    | 55.0    | 802.5 | 66.9 | 0.73|
|         |      | v=1     | 854    | 100.0| 25.3    | 44.7    | 773.3 | 67.2 | 0.69|
|         |      | v=2     | 341    | 100.0| 14.1    | 36.1    | 726.9 | 66.1 | 0.70|
|         |      | v=3     | 142    | 100.0| 7.7     | 28.9    | 701.6 | 64.6 | 0.63|
|         |      | v≥4     | 382    | 100.0| 12.8    | 31.2    | 692.3 | 74.0 | 0.72|


## Table 2: Diamond Dataset statistics over Easy-5 and Hard-5.

| Group   | Type | SentPos | Counts | %r ≤ 10 | Boost | PPL  | AcS |
|---------|------|---------|--------|---------|-------|------|-----|
| Easy-5  | Orig.| –       | 3,110  | –       | –     | 37.6 | 0.78|
|         | Adv. | all     | 3,110  | 100.0   | 69.7  | 41.9 | 0.77|
|         |      | v=0     | 2,285  | 100.0   | 69.7  | 41.3 | 0.78|
|         |      | v=1     | 523    | 100.0   | 69.7  | 44.8 | 0.76|
|         |      | v=2     | 165    | 100.0   | 70.5  | 39.3 | 0.74|
|         |      | v=3     | 54     | 100.0   | 67.3  | 38.4 | 0.75|
|         |      | v≥4     | 83     | 100.0   | 70.2  | 48.7 | 0.74|
| Hard-5  | Orig.| –       | 2,470  | –       | –     | 55.8 | 0.71|
|         | Adv. | all     | 2,470  | 58.9    | 983.1 | 73.1 | 0.71|
|         |      | v=0     | 1,805  | 62.7    | 984.0 | 70.8 | 0.72|
|         |      | v=1     | 382    | 56.5    | 982.3 | 79.9 | 0.66|
|         |      | v=2     | 123    | 39.0    | 979.3 | 74.8 | 0.71|
|         |      | v=3     | 41     | 26.8    | 974.9 | 76.2 | 0.63|
|         |      | v≥4     | 119    | 41.2    | 979.3 | 82.0 | 0.69|



## Table 3: Performance comparison across baselines and ours.
We evaluate our approach against state-of-the-art adversarial ranking techniques, including:
- Query Injection (Query+)
- Embedding Perturbation (EMPRA)
- Trigger-Based Attacks (PAT, Brittle-BERT)
- Sentence-Level Attacks (IDEM)

| Model         | Easy-5 ASR | %r ≤ 10 | %r ≤ 50 | Boost | PPL  | Hard-5 ASR | %r ≤ 10 | %r ≤ 50 | Boost | PPL  |
|---------------|------------|---------|---------|-------|------|-------------|----------|----------|--------|------|
| PRADA         | 76.9       | 2.6     | 44.7    | 22.1  | 96.8 | 75.7        | 0.0      | 2.2      | 98.5   | 202.8|
| Brittle-BERT  | 94.1       | 61.6    | 90.7    | 56.8  | 118.1| 100.0       | 29.6     | 72.7     | 933.1  | 173.2|
| PAT           | 51.2       | 3.5     | 25.0    | 2.9   | 47.0 | 80.0        | 0.0      | 1.1      | 119.2  | 70.2 |
| IDEM          | 99.3       | 82.4    | 97.4    | 67.0  | 38.8 | 99.9        | 45.5     | 80.2     | 926.3  | 61.8 |
| EMPRA         | 99.7       | 87.9    | 99.2    | 69.3  | 37.3 | 99.4        | 51.3     | 80.9     | 904.8  | 55.0 |
| Ours          | 100.0      | 100.0   | 100.0   | 69.7  | 41.9 | 100.0       | 58.9     | 100.0    | 983.1  | 73.1 |






