# Literature review

Status: Module A's two assigned sources have been checked against their original
papers and author releases. The other modules' contributions and the full
related-work synthesis remain pending. This is a focused contribution, not a
comprehensive survey or evidence of current state-of-the-art performance.

## Module A sources

### BANKING77 and the public benchmark

Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*,
introduce BANKING77: 13,083 labeled banking queries across 77 intents. Closely
related intents make classification harder than identifying broad domains. They
evaluate fixed USE/ConveRT sentence representations with an MLP classifier, and
compare them with BERT-based methods, in balanced settings with
10 examples per intent, 30 per intent, and full training data, keeping standard
test sets. Their central concern is effective intent detection with limited
labels and modest computing resources. Table 3 reports **accuracy**: at ten
examples per intent, fine-tuned cased BERT-Large scores **83.42%**, and USE+ConveRT scores
**85.19%**. These are contextual published references; neither is a DistilBERT
augmentation result or a macro-F1 score. The authors also emphasize reducing
dependence on validation-based tuning. [Original paper, Sections 3–5 and Table 3](https://aclanthology.org/2020.nlp4convai-1.5.pdf).

The [author dataset release](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
provides the category mapping and training/test files. Our pinned copy has
10,003 source training queries and 3,080 test queries. We reserve 770 training
queries for validation before drawing our training subsets. Our ten-shot setting
therefore uses **770 training labels plus 770 validation labels**. We must report
that extra supervision explicitly. Our released subset IDs, validation rule,
classifier and training policy differ from the paper's experiment. Sharing a
dataset and examples-per-class count does not establish an identical protocol.

For a direct numerical comparison, reproduce the cited method with our released
artifacts and supervision budget, or reproduce its original protocol separately.
Until then, place the paper's accuracy in a contextual table with these
differences; evaluate our own methods against B1 under the common protocol.

### EDA as a published augmentation method

Wei and Zou (2019), *EDA: Easy Data Augmentation Techniques for Boosting
Performance on Text Classification Tasks*, investigate synonym replacement,
random insertion, random swap and random deletion. They evaluate convolutional
and recurrent classifiers on five tasks, including sentiment, subjectivity and
question classification, using **accuracy**. Their experiments vary training-set
size, edit strength and augmentation quantity; the quantity study considers
1–32 augmented sentences per original. Gains are stronger with smaller training
sets. The discussion explicitly cautions that gains with pretrained models may
be smaller. Thus, the paper motivates a simple augmentation control but does not
establish that EDA improves our DistilBERT/BANKING77 setup. Its recommendations
also depend on training-set size. [Original paper, Sections 3–6](https://aclanthology.org/D19-1670.pdf).

The [author implementation](https://github.com/jasonwei20/eda_nlp) uses WordNet
synonyms and exposes operation strengths and the number of augmentations.
Our B3 implements the same four operation families in
[src/baselines.py](../src/baselines.py), with project-specific safeguards:

- Preserve punctuation, numeric tokens and explicit negation during edits.
- Keep original training rows and retain each augmentation's parent ID and label.
- Reject unchanged or duplicate candidates against real training and accepted
  EDA examples only; make at most 16 attempts per requested augmentation.
- Target one augmentation per real example, with edit/deletion rates of 0.1.

All six prepared B3 datasets meet that target without shortfalls. This produces
1,540 low-resource rows and 2,630 imbalanced rows per seed. Describe B3 as an
**adaptation of EDA**, with these parameters and safeguards, rather than claiming
an exact reproduction of the published experiment. The inherited label still
needs scrutiny: protecting tokens does not guarantee that an edit preserves a
fine-grained banking intent.

### Connection to the proposal

Our interpretation is that these papers motivate complementary controls:
BANKING77 supplies a demanding intent task, and EDA supplies a published, simple
augmentation method. Neither paper directly resolves the proposed comparison
of Qwen-generated examples with duplicate filtering, classifier-based selection,
rare-class behavior and full pipeline cost. That observation motivates our
experiment; it does not establish novelty across the wider literature.

| Proposal question | Evidence our experiments must supply |
|---|---|
| Filtering effectiveness | Compare F0/F1/F2 with matched per-class synthetic counts and the same trainer; use B1/B2/B3 as controls. |
| Augmentation quantity | Run the declared quantity sweep before claiming a quantity effect; otherwise state that RQ2 is partly unanswered. |
| Rare-class effects | Report the fixed rare-class F1/recall alongside overall macro-F1, including oversampling as a control. |
| Cost versus quality | Include generation, rejected examples, filtering and classifier time; identify the B1 teacher cost for F2. |

The completed learning-rate pilot is development evidence, not a final benchmark
comparison. Main baseline and filtering results must be reported only after
their runs and exports are verified. Do not infer augmentation gains from the
pilot's selected rate.

## Shared comparison matrix to complete

For each relevant paper record: task/dataset, real and validation supervision,
generator, selection method, classifier, augmentation quantity, metric, cost
measurement, findings, limitations, released artifacts, and relevance to an RQ.

Organize the narrative around generation, filtering/selection, low-resource and
imbalanced classification, and cost/quality/diversity. Do not claim novelty or
superiority from results obtained under incompatible protocols.

The final report must answer filtering effectiveness, quantity, rare-class
behaviour, and cost using completed evidence. If the optional quantity sweep is
not run, explicitly report that RQ2 remains partly unanswered.
