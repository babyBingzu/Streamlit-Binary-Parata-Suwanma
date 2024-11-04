from tkinter import _test
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def main():
    ################ Step 1 Create Web Title #####################

    st.title("Binary Classification Streamlit App")
    st.sidebar.title("Binary Classification Streamlit App")
    st.markdown(" เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")
    st.sidebar.markdown(" เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")
    C = 1.0  # ค่าเริ่มต้นของ Regularization parameter
    max_iter = 100  # ค่าเริ่มต้นของจำนวน iterations




    ############### Step 2 Load dataset and Preprocessing data ##########

    # create read_csv function
    @st.cache_data(persist=True) #ไม่ให้ streamlit โหลดข้อมูลใหม่ทุกๆครั้งเมื่อ run
    def load_data():
        # Define the base directory as the folder containing your script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        file_path = os.path.join(DATA_DIR, 'mushrooms.csv')

        data = pd.read_csv(file_path)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def spliting_data(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=class_names)
            st.pyplot(fig)
    
            
        
        if 'ROC Curve' in metrics_list:
            
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
      
    df = load_data()
    x_train, x_test, y_train, y_test = spliting_data(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Choose Classifiers")
    classifier  = st.sidebar.selectbox("Classifier", ("Support Vectore Machine (SVM)", "Logistice Regression", "Random Forest"))

    



     ############### Step 3 Train a SVM Classifier ##########

    if classifier == 'Support Vectore Machine (SVM)': #เลือก model
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma  = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Supper Vector Machine (SVM) results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred   = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)
            
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)


    

     ############### Step 4 Training a Logistic Regression Classifier ##########
    # ตรวจสอบว่า classifier ถูกตั้งค่าไว้แล้ว
    if classifier == 'Logistice Regression':
        st.sidebar.subheader("Model Hyperparameters")
    
        # รับค่าพารามิเตอร์จากผู้ใช้
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0, key='C')
        max_iter = st.sidebar.slider("Maximum iterations", 50, 500, step=50, value=100, key='max_iter')
    
        # เลือก solver
        solver = st.sidebar.selectbox("Solver", 
                                   options=['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                                   index=0, key='solver')

        # เลือกเมตริกที่ต้องการแสดง
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metrics_multiselect')

        # เมื่อกดปุ่ม "Classify" ให้ทำการฝึกโมเดล
        if st.sidebar.button("Classify", key='classify_logistic_regression'):
            st.subheader("Logistic Regression Results")

            # สร้างโมเดล Logistic Regression และฝึกกับข้อมูล
            model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=0)
            model.fit(x_train, y_train)

            # คำนวณ Accuracy, Precision, และ Recall
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            # แสดงผลลัพธ์บนหน้าจอ
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)

        # แสดงกราฟเมตริกที่เลือก
        plot_metrics(metrics)

     






     ############### Step 5 Training a Random Forest Classifier ##########
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 10, 200, step=10, value=100)
        max_depth = st.sidebar.slider("Maximum depth of trees", 1, 20, step=1, value=None)

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metrics_multiselect')

        if st.sidebar.button("Classify", key='classify_Random_Forest'):
            st.subheader("Random Forest Results")

            # Create and train the Random Forest model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            model.fit(x_train, y_train)

            # Calculate Accuracy, Precision, and Recall
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            # Display results on the screen
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)

        # Display selected metrics
        plot_metrics(metrics)







    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset")
        st.write(df)

    
    
    




if __name__ == '__main__':
    main()


