"""Optional local interactive demo: streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
from core import ROOT, Predictor

st.set_page_config(page_title='Campus Compass', page_icon='🧭', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1020; color: #f1f5f9;}
.block-container {max-width: 1100px; padding-top: 2.5rem;}
h1 {letter-spacing: -2px;}
[data-testid="stMetric"] {background: #172038; border:1px solid #293550; padding:18px; border-radius:16px;}
.stButton button, .stFormSubmitButton button {border-radius:10px;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return Predictor()

try:
    predictor = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.caption('CAMPUS COMPASS / INTENT INTELLIGENCE')
st.title('One question. The right direction.')
st.write('Classify student queries into six campus categories with a small, reproducible ML model.')
with st.sidebar:
    st.title('🧭 Compass')
    st.caption('A transparent ML screening project')
    st.markdown('**Categories**\n\nAcademics · Events · Transport · Clubs · Facilities · Other')
    st.divider()
    st.write('Active model', predictor.bundle['model'].replace('_', ' ').title())
    st.write('Rejection threshold', predictor.bundle['threshold'])
    st.info('English queries work best. This routes a question; it does not answer it or access university systems.')
    st.caption('No external AI API. Queries are processed locally and are not saved by this app.')

classify, evaluate, about = st.tabs(['Try a query', 'Model evidence', 'How it works'])
with classify:
    examples = ['When does the campus shuttle leave?', 'How do I join the coding club?',
                'Where do I submit my assignment?', 'What time does the library close?',
                'Are tickets available for the college fest?', 'Tell me a joke']
    st.caption('QUICK EXAMPLES · illustrative, some are from the authored dataset')
    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i % 3].button(example, width='stretch'):
            st.session_state['query_input'] = example
    with st.form('query_form'):
        query = st.text_area('Your campus question', key='query_input', placeholder='Ask about classes, clubs, transport, facilities…', max_chars=1000)
        submit = st.form_submit_button('Find the category →', type='primary')
    if submit:
        try:
            st.session_state['result'] = predictor.predict(query)
        except ValueError as exc:
            st.session_state.pop('result', None)
            st.warning(str(exc))
    if 'result' in st.session_state:
        result = st.session_state['result']
        st.caption('RESULT FOR: ' + result['query'])
        c1, c2, c3 = st.columns(3)
        c1.metric('Routed category', result['category'].title())
        c2.metric('Top model score', f"{result['confidence']:.1%}")
        c3.metric('Routing', 'Fallback' if result['fallback'] else 'Model prediction')
        if result['fallback']:
            st.warning(f"Routed to Other: {result['reason'].replace('_', ' ')}. Original top class: {result['top_model_category']}.")
        st.caption('Model scores are not calibrated guarantees. The top score belongs to the original model class, including when routed to Other.')
        st.bar_chart(pd.DataFrame({'Score': result['scores']}), horizontal=True)
        with st.expander('Prediction JSON'):
            st.json(result)
with evaluate:
    metrics = json.loads((ROOT / 'reports/metrics.json').read_text())
    rows = []
    for name, value in metrics['test'].items():
        macro = value['report']['macro avg']
        rows.append({'Model': name, 'Test accuracy': value['accuracy'], 'Macro precision': macro['precision'],
                     'Macro recall': macro['recall'], 'Macro F1': macro['f1-score']})
    st.subheader('Measured performance, visible mistakes')
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption('Model selected by validation macro-F1. These test results come from a small AI-assisted authored dataset, not real-world traffic.')
    model = st.selectbox('Inspect model', list(metrics['test']))
    st.image(str(ROOT / f'reports/{model}_confusion_matrix.png'))
    results = pd.read_csv(ROOT / f'reports/{model}_predictions.csv')
    st.write('Held-out mistakes')
    st.dataframe(results[~results.correct], hide_index=True, width='stretch')
    st.download_button('Download test predictions', results.to_csv(index=False), file_name='predictions.csv', mime='text/csv')
with about:
    st.markdown('''### A small pipeline you can explain
1. **Clean:** normalize Unicode, lowercase and tidy punctuation/whitespace.
2. **Represent:** TF-IDF word unigrams and bigrams; vocabulary learns from training only.
3. **Compare:** Logistic Regression and Multinomial Naive Bayes.
4. **Choose:** validation macro-F1 picks the model and confidence threshold.
5. **Evaluate:** untouched test queries measure accuracy, precision, recall and F1.
6. **Route:** low scores or entirely unknown vocabulary go to Other.

**Dataset:** 216 original AI-assisted English queries, 36 per class. The task allows custom queries. CLINC150 is a reference resource; no CLINC150 samples are used in this experiment.

**Boundary rule:** joining an organisation → Clubs; attending its workshop → Events; booking the room → Facilities; traveling there → Transport; coursework → Academics.

**Limitations:** small synthetic dataset, lexical features, no reliable multilingual or mixed-intent handling, no calibrated uncertainty, and no guaranteed unknown-intent detection.
''')
