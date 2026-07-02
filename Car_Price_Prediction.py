import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

MODEL_FILE="car_model.pkl"
PIPELINE_FILE="car_pipeline.pkl"


def build_pipeline(num_attribs,cat_attribs):
    #for numerical columns
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    # for categorical columns
    cat_pipline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    #Construct the full pipeline

    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipline,cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):

    car=pd.read_csv("cardekho_dataset.csv")
    car = car.drop(columns=["Unnamed: 0"])

    car["price_cat"] = pd.qcut(
    car["selling_price"],
    q=5,
    labels=[1,2,3,4,5]
    )

    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    print(car.shape)
    for train_index,test_index in split.split(car,car["price_cat"]):
        strat_test_set=car.loc[test_index].drop("price_cat",axis=1)
        strat_test_set.to_csv("car_test.csv", index=False)
        strat_test_set.drop("selling_price", axis=1).to_csv("car_input.csv", index=False)
        car=car.loc[train_index].drop("price_cat",axis=1)
        
    print(car.shape)

    car_labels=car["selling_price"].copy()  # y(output)
    car_features=car.drop("selling_price",axis=1)   #X (input)
    print(car_features.columns)

    num_attribs=car_features.drop(columns=["car_name","brand","model","seller_type","fuel_type","transmission_type"]).columns.tolist()
    cat_attribs=car_features.drop(columns=["vehicle_age","km_driven","mileage","engine","max_power","seats"]).columns.tolist()

    full_pipeline=build_pipeline(num_attribs,cat_attribs)
    car_prepared=full_pipeline.fit_transform(car_features)
    print("Pipeline complited")

    model=RandomForestRegressor(random_state=42)
    model.fit(car_prepared,car_labels)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(full_pipeline,PIPELINE_FILE)
    
    pred = model.predict(car_prepared)


    test_features = full_pipeline.transform(strat_test_set.drop("selling_price", axis=1))
    test_labels = strat_test_set["selling_price"]
    

    test_pred = model.predict(test_features)

    print("Test R2 score is ",r2_score(test_labels, test_pred))

    print("Training R2 score is ",r2_score(car_labels, pred))

    print("Trained and save")

else :
    model=joblib.load(MODEL_FILE)
    full_pipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv("car_input.csv")
    transform_input=full_pipeline.transform(input_data)
    predictions=model.predict(transform_input)
    input_data["selling_price"]=predictions.round().astype(int)

    input_data.to_csv("car_output.csv",index=False)
    print("Inference is complete, results saved to output.csv ")
    
    
