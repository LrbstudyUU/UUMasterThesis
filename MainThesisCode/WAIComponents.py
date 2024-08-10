#Helper script that creates csv files which extract WAI overall and subcomponent scores from original WAI score csv files
import pandas as pd
import numpy as np

def wais_btg(df):
    for col in df.columns[2:14]:
        df[col] = df[col].replace('', 0).replace(' ', 0).astype(int) 
        df[col] = pd.to_numeric(df[col])
    df["bond"] = df["3"] + df["5"] + df["7"] + df["9"]
    df["goal"] = df["1"] + df["4"] + df["8"] + df["11"]
    df["task"] = df["2"] + df["6"] + df["10"] + df["12"]
    df["wai"] = df["bond"] + df["goal"] + df["task"]
    return df

def waisrt_btg(df):
    for col in df.columns[2:13]:
        df[col] = df[col].replace('', 0).replace(' ', 0).astype(int) 
        df[col] = pd.to_numeric(df[col])
    df["bond"] = df["t2"] + df["t5"] + df["t7"] + df["t9"]
    df["goal"] = df["t3"] + df["t4"] + df["t8"]
    df["task"] = df["t1"] + df["t2"] + df["t10"]
    df["wai"] = df["bond"] + df["goal"] + df["task"]
    return df

patient = pd.read_csv("~/WAI/extracted_columns.csv", delimiter=';')
patient = wais_btg(patient)
patient.to_csv("~/WAI/patient_btg.csv", index=False)

observer = pd.read_csv("~/WAI/observer_ratings.csv", delimiter=';')
observer = wais_btg(observer)
observer.to_csv("~/WAI/observer_btg.csv", index=False)

therapist = pd.read_csv("~/WAI/Therapist_ratings.csv", delimiter=',')
therapist = waisrt_btg(therapist)
therapist.to_csv("~/WAI/Therapist_ratings.csv", index=False)