# Campus Compass — interview preparation

## 30-second introduction
“Campus Compass classifies a student's question into six categories: academics, events, transport, clubs, facilities and other. I used an original AI-assisted dataset because the task permits custom campus queries and CLINC150 is not campus-specific. The pipeline cleans text, converts it into TF-IDF features and compares Logistic Regression with Naive Bayes. Validation chooses the model and fallback threshold, and a separate test set measures performance. There is a CLI and a small Streamlit interface.”

Be clear about AI assistance and explain only work you actually understand. This guide helps you learn the implementation; it is not a script for claiming unaided authorship.

## Most likely questions

**1. TF-IDF kya hai?**
TF = a word's frequency in one query; IDF reduces the importance of words common across many training queries. TF-IDF turns text into numerical vectors. “Bus” helps distinguish transport more than a common word like “the”. We use single words and two-word phrases. The vocabulary and IDF are learned only from training queries.

**2. Logistic Regression classification kaise karta hai?**
Each category gets learned weights for text features. A linear score is calculated per category, then converted into probabilities. Despite its name, Logistic Regression is used for classification. In this multiclass problem the largest score supplies the raw prediction.

**3. Naive Bayes kyun compare kiya?**
It is a fast, understandable text-classification baseline. It estimates class-associated feature likelihoods and assumes features are conditionally independent given the class. “Naive” refers to that simplifying assumption. Our model uses smoothing alpha=0.3 so unseen class-feature combinations do not zero out the estimate.

**4. Train, validation aur test alag kyun?**
Training learns vocabulary, IDF and model weights. Validation chooses between models and confidence thresholds. Test estimates the frozen choice on unseen examples. Here the sizes are 129, 43 and 44. We do not refit using test samples.

**5. Accuracy aur F1 mein difference?**
Accuracy = correct predictions / all predictions. Precision = true positives / all predicted positives. Recall = true positives / all actual positives. F1 = 2PR/(P+R). Compute these for each category; macro-F1 gives every class equal weight. Weighted-F1 weights by class support.

**6. Results kya hain?**
Logistic Regression: test accuracy 65.9%, macro precision 0.701, macro recall 0.661, macro-F1 0.654. Naive Bayes: accuracy 56.8%, macro-F1 0.572. Logistic Regression was chosen from validation, where its macro-F1 was 0.471. The test set has only 44 examples, so results can vary substantially across samples.

**7. Accuracy 90–99% kyun nahi?**
“The benchmark is small and includes distinct phrasings and overlapping topics. I reported measured results and real mistakes instead of training on the test set. The next step is independently collected, more diverse data and a new evaluation.” Do not claim the score proves production quality.

**8. CLINC150 use kiya?**
“No CLINC150 samples were used in training or testing. The task explicitly permits original campus queries, so this project includes 36 original AI-assisted examples per category and documents why generic intents should not be forced into campus labels.”

**9. Confidence score kya guarantee karta hai?**
It is the model's maximum estimated class probability. It is not calibrated, so 70% is not a proven 70% chance of correctness. Below the validation-selected threshold, the query routes to Other. Unknown vocabulary also routes to Other. For fallback results the displayed confidence still belongs to the original top class.

**10. Other aur unknown same hain?**
The output label is the same, but the reason differs. Other can be a learned class prediction, a low-confidence fallback or an unseen-vocabulary fallback. The JSON exposes the reason. This is heuristic OOD handling, not a dedicated verified open-world detector.

**11. Club workshop kis class mein jayega?**
Attending or registering for a workshop is Events; joining the organising society is Clubs; reserving its room is Facilities. The model may confuse these because they share words. Read the actual mistakes in `reports/evaluation.md`.

**12. Fixed random seed kyun?**
To repeat the same split and stochastic decisions. It improves reproducibility, but does not guarantee that a result is statistically reliable. Library versions also matter; our scikit-learn version is pinned.

**13. Data leakage kaise avoid kiya?**
Normalized duplicates are removed before splitting. TF-IDF is fitted inside the training pipeline only. Model/threshold selection uses validation only. Tests verify vocabulary matches training terms and serving predictions match evaluation. Related paraphrases may still cross splits, which is documented.

**14. Hinglish chalega?**
It may occasionally match known words, but the dataset and evaluation are English-only. We do not claim tested Hindi/Hinglish support.

**15. Deep learning kyun nahi?**
216 samples are too few to justify training a large neural model from scratch. Classical baselines are cheap, transparent and easy to reproduce on a CPU. Pretrained embeddings are a reasonable future comparison, but were not used here.

## Five-minute demo
1. Run `python predict.py "When does the campus shuttle leave?"`.
2. Show the six scores, top class and fallback reason in JSON.
3. Run `python -m streamlit run app.py`; classify an English query.
4. Try `zzzxqv pppqqzz` to demonstrate unseen-vocabulary fallback.
5. Open Model evidence: compare both models and show an actual error.
6. Show `data/DATA_CARD.md`, the split column in the CSV and `reports/evaluation.md`.
7. Explain one limitation and the proposed next experiment.

## Files to understand first
Read `core.py`, then `train.py`, then `predict.py`. The UI calls the same predictor; it does not use hardcoded category rules. The test suite verifies the training/serving contract rather than promising perfect predictions.
