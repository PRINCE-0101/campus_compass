# Verification record

- Trained both models successfully with seed 42 using scikit-learn 1.8.0.
- 7 unittest checks passed using `python -m unittest discover -s tests -v` from the project root.
- CLI prediction for the campus shuttle question returned transport.
- Python sources compiled successfully.

Checks cover normalized sample uniqueness and class coverage, training-only TF-IDF vocabulary, invalid input rejection, unknown-vocabulary fallback, inference/evaluation consistency, probability structure/determinism, and text cleaning. These checks verify software behavior, not real-world classification quality.

Streamlit 1.66.1 AppTest passed: startup, transport prediction, unknown-vocabulary fallback, comparison-model selection, and empty-input handling. This is a functional UI check, not a browser pixel-layout inspection.
