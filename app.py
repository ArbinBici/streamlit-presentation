import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# ---------------------------------#
# Page layout
st.set_page_config(
    page_title="The Machine Learning App", initial_sidebar_state="collapsed"
)

# ---------------------------------#
st.write(
    """
# The Machine Learning App
In this implementation, the *RandomForestRegressor()* and *LinearRegression()* functions are used to build a regression model using the **Random Forest** and **Linear Regression** algorithm.
Try adjusting the hyperparameters!
"""
)

tab1, tab2, tab3 = st.tabs(["Dataset", "Performance", "Parameters"])
# ---------------------------------#
# Model building
def build_model_LR(df, params):
    # Features and target are taken from an already prepared dataset.
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100 - params["split_size"]) / 100
    )
    with tab1:
        st.markdown("#### **Data Splits**")
        st.markdown("_Training Set:_")
        st.info(X_train.shape)
        st.markdown("_Test Set:_")
        st.info(X_test.shape)

        st.markdown("#### **Variable Details**")
        st.markdown("_X Variables:_")
        st.info(list(X.columns))
        st.markdown("_Y Variable:_")
        st.info(Y.name)

    lr = LinearRegression(
        positive=params["parameter_positive"],
        n_jobs=params["parameter_n_jobs"],
    )
    lr.fit(X_train, Y_train)
    with tab2:
        st.markdown("#### **Training Set**")
        Y_pred_train = lr.predict(X_train)
        st.markdown("*Coefficient of determination ($R^2$):*")
        st.info(r2_score(Y_train, Y_pred_train))

        st.markdown(f"*Error (Mean Squared Error):*")
        st.info(mean_squared_error(Y_train, Y_pred_train))

        st.markdown("#### **Test Set**")
        Y_pred_test = lr.predict(X_test)
        st.markdown("*Coefficient of determination ($R^2$):*")
        st.info(r2_score(Y_test, Y_pred_test))

        st.markdown(f"*Error (Mean Squared Error):*")
        st.info(mean_squared_error(Y_test, Y_pred_test))
    with tab3:
        st.markdown("#### Model Parameters")
        st.write(lr.get_params())


def build_model_RF(df, params):
    # Features and target are taken from an already prepared dataset.
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100 - params["split_size"]) / 100
    )
    with tab1:
        st.markdown("#### **Data Splits**")
        st.markdown("_Training Set:_")
        st.info(X_train.shape)
        st.markdown("_Test Set:_")
        st.info(X_test.shape)

        st.markdown("#### **Variable Details**")
        st.markdown("_X Variables:_")
        st.info(list(X.columns))
        st.markdown("_Y Variable:_")
        st.info(Y.name)

    rf = RandomForestRegressor(
        n_estimators=params["parameter_n_estimators"],
        random_state=params["parameter_random_state"],
        max_depth=params["parameter_max_depth"],
        criterion=params["parameter_criterion"],
        min_samples_split=params["parameter_min_samples_split"],
        min_samples_leaf=params["parameter_min_samples_leaf"],
        bootstrap=params["parameter_bootstrap"],
        oob_score=params["parameter_oob_score"],
        n_jobs=params["parameter_n_jobs"],
    )
    rf.fit(X_train, Y_train)
    criterion_choose = params["parameter_criterion_choose"]
    with tab2:
        st.markdown("#### **Training Set**")
        Y_pred_train = rf.predict(X_train)
        st.markdown("*Coefficient of determination ($R^2$):*")
        st.info(r2_score(Y_train, Y_pred_train))

        st.markdown(f"*Error ({criterion_choose}):*")
        st.info(mean_squared_error(Y_train, Y_pred_train))

        st.markdown("#### **Test Set**")
        Y_pred_test = rf.predict(X_test)
        st.markdown("*Coefficient of determination ($R^2$):*")
        st.info(r2_score(Y_test, Y_pred_test))

        st.markdown(f"*Error ({criterion_choose}):*")
        st.info(mean_squared_error(Y_test, Y_pred_test))
    with tab3:
        st.markdown("#### Model Parameters")
        st.write(rf.get_params())


# ---------------------------------#
def LR_hyperparams():
    st.markdown("### Set Parameters")
    split_size = st.sidebar.slider(
        "Data split ratio (% for Training Set)", 10, 90, 80, 5
    )
    st.markdown("### General Parameters")
    parameter_positive = st.radio(
        "Force coefficients to be positive (positive)", options=[True, False]
    )
    parameter_n_jobs = st.radio(
        "Number of jobs to run in parallel (n_jobs)", options=[1, -1]
    )
    params = {
        "split_size": split_size,
        "parameter_positive": parameter_positive,
        "parameter_n_jobs": parameter_n_jobs,
    }
    return params


def RF_hyperparams():
    st.markdown("### Set Parameters")
    split_size = st.sidebar.slider(
        "Data split ratio (% for Training Set)", 10, 90, 80, 5
    )
    st.markdown("### Learning Parameters")
    parameter_n_estimators = st.sidebar.slider(
        "Number of estimators (n_estimators)", 1, 1000, 100, 1
    )
    parameter_max_depth = st.sidebar.slider("Max depth (max_depth)", 1, 100, 1, 1)
    parameter_min_samples_split = st.sidebar.slider(
        "Minimum number of samples required to split an internal node (min_samples_split)",
        2,
        10,
        2,
        1,
    )
    parameter_min_samples_leaf = st.sidebar.slider(
        "Minimum number of samples required to be at a leaf node (min_samples_leaf)",
        1,
        10,
        2,
        1,
    )
    st.markdown("### General Parameters")
    parameter_random_state = st.sidebar.slider(
        "Seed number (random_state)", 0, 1000, 42, 1
    )
    parameter_criterion_choose = st.radio(
        "Performance measure (criterion)",
        options=["Mean Squared Error", "Mean Absolute Error"],
    )
    parameter_criterion = (
        "squared_error"
        if parameter_criterion_choose == "Mean Squared Error"
        else "absolute_error"
    )
    parameter_bootstrap = st.radio(
        "Bootstrap samples when building trees (bootstrap)", options=[True, False]
    )
    parameter_oob_score = st.radio(
        "Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)",
        options=[False, True],
    )
    parameter_n_jobs = st.radio(
        "Number of jobs to run in parallel (n_jobs)", options=[1, -1]
    )
    params = {
        "split_size": split_size,
        "parameter_n_estimators": parameter_n_estimators,
        "parameter_max_depth": parameter_max_depth,
        "parameter_min_samples_split": parameter_min_samples_split,
        "parameter_min_samples_leaf": parameter_min_samples_leaf,
        "parameter_random_state": parameter_random_state,
        "parameter_criterion_choose": parameter_criterion_choose,
        "parameter_criterion": parameter_criterion,
        "parameter_bootstrap": parameter_bootstrap,
        "parameter_oob_score": parameter_oob_score,
        "parameter_n_jobs": parameter_n_jobs,
    }
    return params


# Sidebar - Specify parameter settings
with st.sidebar:
    model_selection = st.selectbox(
        "Select the model to train:", ["Linear Regression", "Random Forest"]
    )
    if model_selection == "Linear Regression":
        LR_params = LR_hyperparams()
        model = 1
    else:
        RF_params = RF_hyperparams()
        model = 2

# ---------------------------------#
# Main panel

# Displays the dataset
with tab1:
    # Diabetes dataset
    st.markdown("### Dataset")
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    Y = pd.Series(diabetes.target, name="response")
    df = pd.concat([X, Y], axis=1)

    st.markdown("The Diabetes dataset is used as the example.")
    st.write(df)

    if model == 1:
        build_model_LR(df, LR_params)
    elif model == 2:
        build_model_RF(df, RF_params)
