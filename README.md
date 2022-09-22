# streamlit-presentation
A demo showcasing the usage of Streamlit in Machine Learning.

Link: https://arbinbici-streamlit-presentation-app-raj0sw.streamlitapp.com

### How To Run It

- Requirements:
    1. Streamlit
    2. Scikit-Learn

1. Create a virtual environment and access it:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
    _or_
    ```bash
    make requirements
    ```
3. Start the streamlit server:

    ```python
    streamlit run app.py
    ```
    _or_
    ```bash
    make streamlit
    ```