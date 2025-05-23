import casadi as ca
import numpy as np
import typing
import pandas as pd

def run (num_agents : int):
    dummy_data = np.array([num_agents])
    df = pd.DataFrame(dummy_data)
    df.to_csv("csv/dummy_data.csv")

    """
    TODO: define each agents system dynamics
    TODO: define cost function
    TODO: define geometries
    TODO: anyting casadi
    TODO: save data to csv
    """
