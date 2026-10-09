"""
Web UI for the churn model (not built yet).

Planned: a Gradio or Streamlit page that lets a user fill in a customer's
details and see the prediction. It should call the API in main.py over HTTP
(POST /predict) instead of loading the model itself, so the threshold and the
validation rules live in one place. The dropdown options can be read from the
Literal types in schemas.py.
"""
