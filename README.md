# Relation Extration on PubMed Dataset

Project Objective - This project aims to formulate a better framework
of obtaining entity relation dataset from text
documents. Specifically, we will perform entity extraction from text
documents in PubMed dataset, get the phrases connecting every pair of
entities, summarize the relations, and cluster the summarizations so
similar relationships can be described by a centroid term.

To install all the requirement for this repository
Run 
```
bash requirement.txt
```

####  Authors: + Ruilin Liu (rl3234)(Team Captain) + Zhifeng Zhang (zz2884) + Jessie Wang (yw3765) + Zhiqing Yang (zy2491) + Zhucheng Zhan (zz2783) 
####  Sponsor/Mentor: - John Labarga (john.labarga@unilever.com) from Unilever
####  CA: - Aayush Kumar Verma (av2955)
####  instructor: - Sining Chen (sc4549)

## Strategy Diagram

The objective of this capstone is to evaluate the feasibility and
performance of a novel strategy for producing training examples for
relation extractors. This strategy seeks to make a trade-off where
lower fidelity is tolerated to achieve lower cost and broader scope in
producing training examples.

The central hypothesis of this effort is that transformer-based
sumamrization models have gotten good enough to identify a term that
describes the relationship between two entities in a sentence or
paragraph (what is sometimes called the *verb* in the relation
extraction literature, when the relation extraction problem is framed
as identifying (subject, verb, object) triples). There is a secondary
hypothesis here that this verb is present in the text between two
entities, however this hypothesis seems plasuble since it also
underlies the premise of relation extraction.

The basic workflow of this approach is as follows:

1. Pass documents through an entity extractor to identify entities.

2. For each entity pair within a given proximity, use a summarization
model to extract a token or small set of tokens (3 tokens at most)
that describes the relationship between the entities. What "proximity"
means here will need to be tuned via experimentation, in the simplest
case this means two adjacent entities. These tokens or small sets of
tokens become the "relation candidates".

3. The output of step 2 is expected to be sparse and not immediately
usable. To condense the candidates, use semantic clustering to cluster
the relation candidates. Then identify a centroid vector for each
cluster. Finally, use a decoder (this could be as simple as Word2Vec)
to get a token (or small set of tokens) which is most indicative of
that cluster.

4. Produce a training set by tagging the entity pairs from step 2 with
the centroid relationships from step 3.

5. Fine-tune a relation extractor with the training examples fro step
4, and compare with an un-tuned relation extractor.