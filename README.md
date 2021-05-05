# Naive-Bayes-for-text-classification-spam-ham

Download the spam/ham (ham is not spam) dataset available on my-
Courses. The data set is divided into two sets: training set and test set.
The dataset was used in the Metsis et al. paper [1]. Each set has two di-
rectories: spam and ham. All files in the spam folders are spam messages
and all files in the ham folder are legitimate (non spam) messages.

Implement the multinomial Naive Bayes algorithm for text classification
described here: http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
(see Figure 13.2). Note that the algorithm uses add-one laplace smooth-
ing. Ignore punctuation and special characters and normalize words by
converting them to lower case, converting plural words to singular (i.e.,
\Here" and \here" are the same word, \pens" and \pen" are the same
word). Normalize words by stemming them using an online stemmer such
as http://www.nltk.org/howto/stem.html. Make sure that you do all
the calculations in log-scale to avoid under
ow. Use your algorithm to
learn from the training set and report accuracy on the test set.

Improve your Naive Bayes by throwing away (i.e., filtering out) stop words
such as \the" \of" and \for" from all the documents. A list of stop words
can be found here: http://www.ranks.nl/stopwords. Report accuracy
for Naive Bayes for this filtered set. Does the accuracy improve? Explain
why the accuracy improves or why it does not?