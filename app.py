"""Streamlit application for hotel no-show prediction."""
import streamlit as st
import pandas as pd
import os
import json
import joblib
import plotly.express as px

from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.genai.interpreter import GenAIInterpreter
from src.genai.agent import DataScienceAgent

setup_logging()

st.set_page_config(
    page_title="Hotel No-Show Prediction",
    page_icon="🏨",
    layout="wide"
)


@st.cache_resource
def load_resources():
    """Load config, models, results, data, and preprocessor."""
    config = load_config('config/config.yaml')
    
    results = {}
    try:
        with open('models/results.json') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
        st.warning("results.json not found. Run: `python ml_pipeline.py`")
    
    models = {}
    try:
        champion_name = results.get('metadata', {}).get('champion_model', config.get('champion_model', 'xgboost'))
        champion_path = f'models/{champion_name}_model.pkl'
        champion_model = joblib.load(champion_path)
        models['champion'] = champion_model
        models['champion_name'] = champion_name
    except Exception as e:
        st.warning(f"Champion model not found: {e}. Run: `python ml_pipeline.py`")
    
    df = None
    if os.path.exists(config['data']['database_path']):
        query_file = 'config/queries.sql'
        if os.path.exists(query_file):
            with open(query_file, 'r') as f:
                query = f.read()
        else:
            query = "SELECT * FROM noshow"
        
        loader = DataLoader(config['data']['database_path'])
        df = loader.load_data(query)
        if df is not None:
            if 'price' in df.columns:
                df['price'] = (
                    df['price'].astype(str)
                    .str.replace(r'^[A-Z]+\$\s*', '', regex=True)
                    .str.replace(',', '', regex=False)
                )
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            if 'first_time' in df.columns:
                df['first_time'] = df['first_time'].map({'Yes': 1, 'No': 0})
                df['first_time'] = pd.to_numeric(df['first_time'], errors='coerce')
            if 'num_adults' in df.columns:
                df['num_adults'] = pd.to_numeric(df['num_adults'], errors='coerce')
    
    preprocessor = DataPreprocessor(config)
    for path in ['models/production_preprocessor.joblib', 'models/preprocessor.joblib']:
        if os.path.exists(path):
            saved = joblib.load(path)
            if isinstance(saved, dict):
                preprocessor.transformer = saved.get('transformer')
                preprocessor.feature_names = saved.get('feature_names')
            else:
                preprocessor = saved
            break
    
    return config, models, results, df, preprocessor


def create_bar_chart(df, x, y, title, color_scale='Reds'):
    fig = px.bar(df, x=x, y=y, title=title, color=y, color_continuous_scale=color_scale)
    return fig


def main():
    st.title("🏨 Hotel No-Show Prediction System")
    
    config, models, results, df, preprocessor = load_resources()
    
    st.sidebar.markdown("## 📍 Navigation")
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    nav_options = {
        "📊 Dashboard": "Dashboard",
        "🔮 Prediction": "Prediction", 
        "📈 Performance": "Performance",
        "🤖 AI Agent": "AI Agent",
        "💡 Insights": "Insights"
    }
    
    for label, page_name in nav_options.items():
        if st.sidebar.button(label, key=page_name, use_container_width=True):
            st.session_state.current_page = page_name
    
    page = st.session_state.current_page
    
    if page == "Dashboard":
        show_dashboard(df)
    elif page == "Prediction":
        show_prediction(models, preprocessor)
    elif page == "Performance":
        show_performance(results)
    elif page == "AI Agent":
        show_agent(config)
    else:
        show_insights(results)


def show_dashboard(df):
    """Dashboard with key metrics and visualizations."""
    st.header("📊 Dashboard")
    
    if df is None:
        st.error("Database not found")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bookings", f"{len(df):,}")
    col2.metric("No-Show Rate", f"{df['no_show'].mean():.1%}")
    col3.metric("Avg Price", f"${df['price'].mean():.2f}")
    col4.metric("Revenue at Risk", f"${df[df['no_show']==1]['price'].sum():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        platform = df.groupby('platform')['no_show'].mean().reset_index()
        platform['no_show'] *= 100
        st.plotly_chart(create_bar_chart(platform, 'platform', 'no_show', 
            'No-Show Rate by Platform'), use_container_width=True)
    with col2:
        room = df.groupby('room')['no_show'].mean().reset_index()
        room['no_show'] *= 100
        st.plotly_chart(create_bar_chart(room, 'room', 'no_show',
            'No-Show Rate by Room', 'Oranges'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        ft = df.dropna(subset=['first_time'])
        first = ft.groupby('first_time')['no_show'].mean().reset_index()
        first['first_time'] = first['first_time'].map({0: 'Returning', 1: 'First-Time'})
        fig = px.pie(first, values='no_show', names='first_time', 
            title='No-Show: First-Time vs Returning')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='price', color='no_show', nbins=30,
            title='Price Distribution', barmode='overlay', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)


def show_prediction(models, preprocessor):
    st.header("🔮 Make Prediction")
    
    if not models:
        st.error("Models not loaded")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        branch = st.selectbox("Branch", ['Changi', 'Orchard'])
        country = st.selectbox("Country", ['Australia', 'China', 'India', 'Indonesia', 'Japan', 'Malaysia', 'Singapore'])
        room = st.selectbox("Room", ['King', 'President Suite', 'Queen', 'Single'])
        platform = st.selectbox("Platform", ['Agent', 'Email', 'Phone', 'Website'])
    with col2:
        booking_month = st.selectbox("Booking Month", range(1, 13))
        arrival_month = st.selectbox("Arrival Month", range(1, 13))
        arrival_day = st.slider("Arrival Day", 1, 28, 15)
        checkout_month = st.selectbox("Checkout Month", range(1, 13))
        checkout_day = st.slider("Checkout Day", 1, 28, 20)
    with col3:
        price = st.number_input("Price ($)", 50.0, 1000.0, 150.0)
        num_adults = st.number_input("Adults", 1, 10, 2)
        num_children = st.number_input("Children", 0, 10, 0)
        first_time = st.radio("First Time?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    
    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame([{
            'booking_id': 'PRED_001', 'branch': branch, 'booking_month': booking_month,
            'arrival_month': arrival_month, 'arrival_day': arrival_day,
            'checkout_month': checkout_month, 'checkout_day': checkout_day,
            'country': country, 'first_time': first_time, 'room': room,
            'price': price, 'platform': platform, 'num_adults': num_adults,
            'num_children': num_children, 'no_show': 0
        }])
        
        X, _ = preprocessor.prepare_features(input_data, fit=False)
        model = models['champion']
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)
        no_show_prob = prob[0][1] if len(prob[0]) > 1 else prob[0][0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "⚠️ No-Show" if prediction else "✅ Show")
        col2.metric("No-Show Probability", f"{float(no_show_prob):.1%}")
        risk = "HIGH" if no_show_prob > 0.7 else "MEDIUM" if no_show_prob > 0.4 else "LOW"
        col3.metric("Risk Level", risk)
        
        genai = GenAIInterpreter()
        st.subheader("💡 AI Insights")
        display_data = input_data.to_dict('records')[0]
        display_data['first_time'] = 'Yes' if first_time == 1 else 'No'
        st.markdown(genai.explain_prediction(prediction, no_show_prob, display_data))


def show_performance(results):
    st.header("📊 Model Performance")
    
    if not results:
        st.warning("No results available")
        return
    
    data = []
    models_data = results.get('models', results)
    for name, res in models_data.items():
        if isinstance(res, dict):
            row = {
                'Model': res.get('model_name', name),
                'Precision': f"{res.get('precision', 0):.3f}",
                'Recall': f"{res.get('recall', 0):.3f}",
                'F1': f"{res.get('f1_score', 0):.3f}",
                'ROC-AUC': f"{res.get('roc_auc', 0):.3f}",
                'PR-AUC': f"{res.get('pr_auc', 0):.3f}",
                'CV ROC-AUC': f"{res.get('cv_roc_auc_mean', 0) or 0:.3f} ± {res.get('cv_roc_auc_std', 0) or 0:.3f}" if res.get('cv_roc_auc_mean') is not None else 'N/A'
            }
            if res.get('accuracy'):
                row['Accuracy'] = f"{res['accuracy']:.3f}"
            data.append(row)
    
    if data:
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.info("No model results available")
    
    plot_data = []
    models_data = results.get('models', results)
    for name, res in models_data.items():
        if isinstance(res, dict):
            for metric in ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
                if metric in res:
                    plot_data.append({
                        'Model': res.get('model_name', name),
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': res[metric]
                    })
    
    if plot_data:
        fig = px.bar(pd.DataFrame(plot_data), x='Metric', y='Score', color='Model',
            barmode='group', title='Performance Comparison')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No performance metrics available for visualization")
    
    models_data = results.get('models', results)
    rf_data = models_data.get('random_forest', {})
    if 'feature_importance' in rf_data:
        st.subheader("Feature Importance")
        feat = sorted(rf_data['feature_importance'].items(), 
            key=lambda x: x[1], reverse=True)[:15]
        df_feat = pd.DataFrame(feat, columns=['Feature', 'Importance'])
        fig = px.bar(df_feat, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)


def show_agent(config):
    st.header("💬 AI Assistant")
    
    if not os.path.exists(config['data']['database_path']):
        st.error("Database not found")
        return
    
    with st.expander("Example Questions"):
        for q in ["Top features for no-show?", "No-show by country?", "Model metrics?"]:
            st.markdown(f"- {q}")
    
    @st.cache_resource
    def get_agent():
        return DataScienceAgent(config, max_iterations=config['agent']['max_iterations'])
    
    agent = get_agent()
    
    if 'chat' not in st.session_state:
        st.session_state.chat = []
    
    query = st.text_input("Question:", placeholder="Ask about the data or models...")
    
    col1, col2 = st.columns([1, 5])
    if col1.button("Ask", type="primary") and query:
        with st.spinner("Analyzing..."):
            result = agent.execute_task(query)
        
        st.markdown(result)
        
        if agent.action_history:
            with st.expander("📋 Evidence & Sources", expanded=False):
                for i, action in enumerate(agent.action_history, 1):
                    st.markdown(f"**{i}. {action.tool}**")
                    st.text(f"Reasoning: {action.reasoning}")
                    with st.expander(f"Raw output from {action.tool}", expanded=False):
                        st.code(str(action.output)[:3000], language="text")
                    st.divider()
        
        st.session_state.chat.append({
            "q": query,
            "a": result,
        })
    
    if col2.button("Clear"):
        st.session_state.chat = []
        st.rerun()
    
    for i, chat in enumerate(reversed(st.session_state.chat)):
        with st.expander(f"Q{len(st.session_state.chat)-i}: {chat['q']}", expanded=(i == 0)):
            st.markdown(chat['a'])


def show_insights(results):
    st.header("💡 AI Insights")
    
    if not results:
        st.warning("No results")
        return
    
    if st.button("Generate Report", type="primary"):
        genai = GenAIInterpreter()
        sections = []
        models_data = results.get('models', results)
        
        with st.spinner("Analyzing..."):
            st.subheader("Model Comparison")
            comp = {}
            for m, r in models_data.items():
                if isinstance(r, dict):
                    comp[r.get('model_name', m)] = {
                        k: round(r[k], 4) for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'cv_roc_auc_mean']
                        if k in r and r[k]
                    }
            analysis = genai.generate_response(
                f"""Here are the performance metrics for our hotel no-show prediction models:
{json.dumps(comp, indent=2)}

In 3-4 sentences, state which model is the best performer and why. Do NOT explain what each metric means. Do NOT list individual metric values. This is for business decision-makers who want a quick summary, not a technical report.""")
            st.markdown(analysis)
            sections.append(f"## Model Analysis\n\n{analysis}")
            
            for model in ['random_forest', 'xgboost', 'lightgbm']:
                if model in models_data and 'feature_importance' in models_data[model]:
                    st.subheader("Feature Importance")
                    feat = dict(sorted(models_data[model]['feature_importance'].items(), 
                        key=lambda x: x[1], reverse=True)[:10])
                    text = genai.interpret_feature_importance(feat)
                    st.markdown(text)
                    sections.append(f"## Features\n\n{text}")
                    break
            
            st.subheader("Recommendations")
            best_model = max(comp.items(), key=lambda x: x[1].get('roc_auc', 0))
            recs = genai.generate_response(
                f"""Our best hotel no-show prediction model is {best_model[0]} with {best_model[1].get('roc_auc', 'N/A')} ROC-AUC.

Provide exactly 5 bullet-point recommendations for hotel management to reduce no-shows and improve revenue.
Each bullet should be 1-2 sentences max. Focus on operational strategies only (e.g., overbooking, communication, pricing, deposits).
Do NOT include any technical or data science recommendations. No introductions or conclusions — just the 5 bullets.""")
            st.markdown(recs)
            sections.append(f"## Recommendations\n\n{recs}")
            
            with open('models/genai_insights_report.md', 'w') as f:
                f.write(f"# AI Insights Report\n\n{pd.Timestamp.now()}\n\n")
                f.write("\n\n---\n\n".join(sections))
            
            st.success("Report saved to models/genai_insights_report.md")


if __name__ == "__main__":
    main()
