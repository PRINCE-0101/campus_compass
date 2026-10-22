# Campus Compass — Project Approach

## 1. Problem and objective

Campus Compass is a small machine-learning project that classifies a student’s question into one of six broad campus categories: **academics, events, transport, clubs, facilities, or other**.

I treated this as an intent-classification problem. The input is a short text query and the output is one category. The system does not answer the question, search university records, or claim to know campus-specific facts. This boundary keeps the project focused: it demonstrates how a simple text classifier can support routing, while a real information service would need a trusted source of campus answers as well.

## 2. Interpreting the task and selecting data

The brief names CLINC150 as a resource and asks how the six campus categories are obtained: by mapping suitable CLINC150 intents, writing original campus questions with at least ten examples per category, or combining the two approaches. CLINC150 is a general intent dataset, so its labels do not directly represent campus departments. Forcing unrelated general intents into campus categories could create incorrect labels. I therefore chose the explicitly permitted original-query approach and recorded that choice in the data card and README.

The included dataset contains **216 English queries: 36 for each category**. These are original, AI-assisted examples written for this project; they are not real student submissions and are not copied from CLINC150. The README makes the dataset source clear so that the project does not imply that CLINC150 was used for model training. If the evaluator expects CLINC150 samples in addition to original queries, that would be a separate experiment and should be added explicitly rather than claimed after the fact.

## 3. Defining category boundaries

Before training, I wrote down what each label means. Academics covers courses, examinations, grades, faculty and academic records. Events covers attending or registering for a time-bound talk, workshop, competition or fest. Transport covers buses, shuttles, routes, passes and campus parking. Clubs covers society membership, recruitment, leadership and club projects. Facilities covers shared campus services and infrastructure such as WiFi, libraries, hostels, room bookings and repairs. Other covers general questions outside the campus-routing scope.

These boundaries matter because some queries can sound similar. “How do I join the coding club?” is about club membership, while “How do I register for the club’s hackathon?” is about an event. “Where can I book a room for the workshop?” is a facilities query. I used examples across each category to make those distinctions understandable.

## 4. Inspecting and preparing the data

The editable source examples are stored in `data/queries.json`. During training, the script converts these examples into a table and records the query, category, source and split. It normalizes text by applying Unicode normalization, lowercasing, replacing punctuation with spaces and collapsing repeated whitespace. This makes trivial differences such as capitalization or extra spaces less likely to be treated as separate vocabulary.

I check for empty cleaned queries and conflicting labels. Exact duplicates after normalization are removed before the split so the same normalized sentence cannot appear in both training and evaluation.

Cleaning is intentionally modest. I do not remove every common word or apply stemming, because phrases such as “how do I” and word combinations can carry useful context. The same cleaning function is used for both training and predictions, which reduces the chance that the model sees text in one form during training and another form when a student uses the demo.

## 5. Creating a reproducible evaluation split

I split the data into training, validation and test partitions using stratification and a fixed random seed of **42**. Stratification keeps all six categories represented in each partition. The current split contains **129 training, 43 validation and 44 test queries**. The training partition is used to learn the text vocabulary and model parameters. The validation partition is used to compare models and choose a fallback threshold. The test partition is held aside until those choices are fixed.

Keeping these partitions separate is important. If I used the test results to pick a model or adjust settings, the final score would no longer be a fair estimate of performance on unseen examples. The saved models remain fitted on the training partition, and the test report reflects those models.

A fixed seed makes the split repeatable, but it does not make a small dataset representative. Similar topics or paraphrases may still appear in different partitions. Exact duplicate removal prevents one direct form of leakage, but it cannot guarantee that semantically similar questions are independent. The report calls out this limitation and recommends collecting a new independently authored test set before drawing stronger conclusions.

## 6. Turning text into features

Machine-learning models need numerical input, so I use **TF-IDF** to represent each query as a vector of word features. TF-IDF gives more weight to terms that are useful within a query and less weight to terms that appear throughout many examples. I include both unigrams and bigrams: unigrams capture individual words, while bigrams capture short phrases such as “bus pass” or “exam timetable.” This helps the classifier use a little local context without requiring a large language model.

The vectorizer is placed inside a scikit-learn pipeline with the classifier. It is fitted only on the training texts, which means validation and test examples do not influence the vocabulary or inverse-document-frequency weights. That design reduces data leakage and keeps the training and inference steps together.

## 7. Comparing and choosing models

I compare two lightweight text-classification baselines. The first is **Logistic Regression**, which learns how strongly text features are associated with each category and produces a score for each class. The second is **Multinomial Naive Bayes**, a common and fast baseline for word-count and text-vector features. Comparing them helps show that the project tested a modelling choice rather than selecting one algorithm without evidence.

For both models, the validation set is used to compare macro-F1 across a small, documented set of confidence thresholds. Macro-F1 gives the six classes equal weight, which is useful even when a dataset has equal class counts, because it reveals the average per-class balance of precision and recall. The model with the better validation macro-F1 is selected. In this run, Logistic Regression achieved validation macro-F1 of about **0.471**, compared with about **0.427** for Naive Bayes, so Logistic Regression became the selected model. The confidence threshold is also chosen on validation data, not on the test data.

## 8. Handling predictions and unfamiliar queries

At inference time, the shared `Predictor` validates the user’s input, applies the fitted text pipeline and reads the model’s class probabilities. It returns the routed category, the original top model category, the top score, scores for all six classes, the threshold and a reason for the routing decision.

If the model’s top score is below the selected threshold, the query is routed to `other`. If the query contains no terms in the learned vocabulary, it is also routed to `other` with an unseen-vocabulary reason. They are not a reliable open-world detector: a query made from familiar words can still receive a confident, incorrect label. The displayed score is an uncalibrated model score, not a guarantee that the prediction is correct. For a fallback result, the score still belongs to the model’s original top category, not to the fallback label.

## 9. Evaluation and error analysis

The final test evaluation includes accuracy, per-class precision, recall and F1, plus macro and weighted averages. The selected Logistic Regression model reached **65.9% test accuracy**, with macro precision about **0.701**, macro recall about **0.661** and macro-F1 about **0.654** on 44 held-out queries. Multinomial Naive Bayes reached **56.8% accuracy** and about **0.572 macro-F1** on the same test partition. These figures are included as measured results, not as a promise of real-campus performance.

I also save a confusion matrix and a prediction table for each model. The evaluation report lists actual correct examples and actual mistakes, including their true labels, predicted labels and top model scores. This makes it possible to inspect which categories are getting confused instead of relying only on one overall score. Words such as “registration,” “certificate” or “workshop” can occur in more than one context, so a model trained on a small set may learn a misleading association. Looking at these cases helps identify whether the problem is a boundary ambiguity, insufficient examples, or a genuinely unclear query.

## 10. User interface and project structure

Campus Compass offers both a command-line interface and an optional Streamlit dashboard. The CLI accepts a query and prints a JSON result, which makes the predictor easy to test or integrate later. It also supports an interactive mode for trying several questions.

Both interfaces call the same inference code in `core.py`. This is deliberate: separate implementations could preprocess a query differently and produce results that do not match the evaluation report.

## 11. Testing, limitations and next steps

The automated tests check that each label appears in each split, normalized duplicates are absent, invalid inputs are rejected, unseen vocabulary uses the fallback, class scores sum to one, and serving predictions match the saved evaluation predictions. The dashboard was separately checked for startup, a transport prediction, unknown-vocabulary fallback, model selection and empty-input handling.

The main limitation is the small, English-only, AI-assisted dataset. It does not include actual student language, spelling variations, Hinglish, multilingual queries or measured campus traffic. It also does not provide answers to questions or connect to campus services.

## Conclusion

My approach was to build a transparent, reproducible baseline for routing campus questions. I defined six categories, wrote a documented original dataset because that option is allowed by the task, cleaned and split the queries, compared two models, selected settings on validation data and evaluated the final choice on a separate test set. The CLI and dashboard make the classifier easy to try, while the confidence fallback, confusion matrices and error examples show both how it behaves and where it can fail.

Campus Compass is a practical starting point for understanding text classification in a student-help context. Its measured test results are useful for comparing the two baselines on this small dataset, but they should not be treated as production accuracy. A larger, independently collected dataset and a fresh evaluation would be needed before using the model to route real student requests.

