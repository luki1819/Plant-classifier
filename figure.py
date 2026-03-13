import plotly.graph_objects as go
import streamlit as st


def display_class_confidences(predictions):
    if predictions:
        fig = go.Figure()
        class_names_list, confidence_list = zip(*predictions)
        fig.add_trace(go.Bar(x=class_names_list, y=confidence_list, name="Class Confidence", marker_color='#4CAF50'))
        fig.update_layout(title="Class Confidence Breakdown", xaxis_title="Classes", yaxis_title="Confidence",
                          template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(fig)
    else:
        st.error("No predictions to display.")
