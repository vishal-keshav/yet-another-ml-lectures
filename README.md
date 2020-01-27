# yet-another-ml-lectures

> I have started providing some sessions(very informal) on machine learning to my fellow classmates who needed help with it. I mostly focus on mathematics and programming in pytorch. I will update my notes on a weakly basis after my session.

## Books I follow(strictly) to prepare the lecture notes
* [MML-book](https://mml-book.github.io/book/mml-book.pdf)
* [math4ml](https://gwthomas.github.io/docs/math4ml.pdf)
* [Deep Learning Book](https://www.deeplearningbook.org/)
* [Dive into Deep Learning](https://d2l.ai/d2l-en.pdf)
* [DiDL code](https://github.com/dsgiitr/d2l-pytorch)

|Week|         Topic        |     Extra notes      |
|:--:|:--------------------:|:--------------------:|
| [4](##Week 4) |Introduction          | ---      |
| [5](##Week 5) |Probability           | ---      |





--------------------------------------
## Week 4
* Introduction and the expectation from these sessions.
* Difference between classical ML approach and Deep learning approach. pros and cons.
* Importance of intuitive understanding by having basic math knowledge.
* Practical approach and importance of coding excersise. Gap between theory and practical implementation.
---------------------------------------

## Week 5
* Basics in probability
* Important descrete and continous distributions
* Self-information, mutual information, KL divergence and cross entropy.
* A good article on cross entropy theory [here](https://medium.com/@stepanulyanin/notes-on-deep-learning-theory-part-1-data-generating-process-31fdda2c8941)
* Clarification of binary and multi class cross entropy interface in pytorch
    - `torch.nn.functional.binary_cross_entropy` takes logistic sigmoid values as inputs
    - `torch.nn.functional.binary_cross_entropy_with_logits` takes logits as inputs
    - `torch.nn.functional.cross_entropy` takes logits as inputs (performs `log_softmax` internally)
    - `torch.nn.functional.nll_loss` is like `cross_entropy` but takes log-probabilities (log-softmax) values as inputs
* Notes  [here](https://www.dropbox.com/sh/b2e2rbc41kfi7rz/AADhtGrZbH-U-po2HBq8zCcqa?dl=0)
