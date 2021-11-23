import pandas as pd 
import numpy as np
import numpy as geek

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from functools import partial

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from stargazer.stargazer import Stargazer, LineLocation
from IPython.core.display import HTML

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.discrete.discrete_model import Probit

import warnings;
warnings.filterwarnings('ignore');

#from aux_functions import *

from auxiliary.aux_plots import *
from auxiliary.aux_predictions import *
from auxiliary.aux_tables import *

def get_data_graphs(selected_columns, sort_data, treated):
    
    graph_data = pd.read_stata('data/BRIGHT_Formatted.dta')
    graph_data.fillna(np.nan, inplace = True)
    
    rel_score_redefined = graph_data["rel_score"]*10000 
    graph_data['rel_score_redefined'] = rel_score_redefined
    
    
    if sort_data == 0: 
        
        graph_data = graph_data[selected_columns]
        rslt = graph_data.dropna()
    
    if sort_data == 1: 
        
        graph_data = graph_data[selected_columns]
        
        rslt = graph_data.loc[graph_data["village_level"] == 1]
        rslt = rslt.dropna()
    
    if treated == 0:
        
        rslt_df = rslt.loc[rslt["proj_selected"] == 0].reset_index(drop= True)
        
    elif treated == 1: 
        
        rslt_df = rslt.loc[rslt["proj_selected"] == 1].reset_index(drop= True)
        
    elif treated == 2: 
        
        rslt_df = rslt.reset_index(drop= True)
    
    return rslt_df

#####################################

def plot_discontinuity(ylabel_lhs,title,data,bins_lhs_data,bins_rhs_data,r_sq_data,yaxis):
    
    sns.set(style="white")
    ax = sns.lineplot(x = "x-axis",y = "y-axis", data = data, color = "black")
    sns.scatterplot(x = "Xgrid",y = "Binned Averages", data = bins_lhs_data, ax = ax, color='orange')
    sns.scatterplot(x = "Xgrid",y = "Binned Averages", data = bins_rhs_data, ax = ax, color='orange')
    ax2 = ax.twinx()
    sns.lineplot(x = "Xgrid for R-squared Values", y = "R-squared Values", data = r_sq_data, ax=ax2)

    
    plt.axvline(x=0, color='red', alpha=1)
    plt.xlim(-300,300)
    
    if yaxis == 1: 
        
        ax.set_yticks(np.linspace(0, 1, 11))
        ax2.set_yticks(np.linspace(0,1, 11))
        ax.set_ylim(-0.05,1.05)
        ax2.set_ylim(-0.05,1.05)
        
    elif yaxis == 2: 
        
        ax.set_yticks(np.linspace(-0.4, 0.5, 10))
        ax2.set_yticks(np.linspace(0,1, 11))
        ax.set_ylim(-0.45,0.55)
        ax2.set_ylim(-0.05,1.05)
       
    ax.set_xlabel("Relative Score")
    ax2.set_ylabel('Fraction of Explained Variance')
    ax.set_ylabel(ylabel_lhs)
    plt.title(title)
    plt.show()

#################################

def plot_graph_2(ylabel, xlabel, title, data):
    
    ax = sns.histplot(data['rel_score_redefined'],stat = "density", kde = True, binwidth = 41.2, color = 'blue')
    plt.axvline(x=0, color='red', alpha=0.8)
    ax.set_yticks(np.linspace(0, 0.005, 6))
    ax.set_ylim(0,0.005)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
####################################

def plot_continuity_hh(variables, data): 
    
    lhs_store = []
    rhs_store = []
    
   
    for var in variables:
        
        lhs = compute_binned_averages(0,var,30,9, data)
        rhs = compute_binned_averages(1,var,30,9,data)
        
        lhs_store.append(lhs)
        rhs_store.append(rhs)
    
    fig, ax = plt.subplots(2,2, figsize=(26,16))
    fontsize = 18
    
    ax[0][0].set_title("Number of Household Members", fontsize = fontsize)
    ax[0][0].set_xlabel("Relative Score", fontsize = fontsize)
    ax[0][0].set_ylabel("Binned Average", fontsize = fontsize)
    ax[0][0].axvline(x=0, color='r', alpha = 1)
    ax[0][0].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[0], color='orange')
    ax[0][0].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[0], color='orange')
    ax[0][0].set_ylim(5,20)
    
    ax[0][1].set_title("Number of Children per Household", fontsize = fontsize)
    ax[0][1].set_xlabel("Relative Score", fontsize = fontsize)
    ax[0][1].set_ylabel("Binned Average", fontsize = fontsize)
    ax[0][1].axvline(x=0, color='r', alpha = 1)
    ax[0][1].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[1], color='orange')
    ax[0][1].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[1], color='orange')
    ax[0][1].set_ylim(2,12)
    
    ax[1][0].set_title("Age of Children", fontsize = fontsize)
    ax[1][0].set_xlabel("Relative Score", fontsize = fontsize)
    ax[1][0].set_ylabel("Binned Average", fontsize = fontsize)
    ax[1][0].axvline(x=0, color='r', alpha = 1)
    ax[1][0].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[2], color='orange')
    ax[1][0].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[2], color='orange')
    ax[1][0].set_ylim(7,10)
    
    ax[1][1].set_title("Proportion of Female Children", fontsize = fontsize)
    ax[1][1].set_xlabel("Relative Score", fontsize = fontsize)
    ax[1][1].set_ylabel("Binned Average", fontsize = fontsize)
    ax[1][1].axvline(x=0, color='r', alpha = 1)
    ax[1][1].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[3], color='orange')
    ax[1][1].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[3], color='orange')
    ax[1][1].set_ylim(0.2,0.8)

    plt.show()

###########################################
    
def plot_continuity_rel_lang(variables, data): 
    
    lhs_store = []
    rhs_store = []
    
    for var in variables:
        
        lhs = compute_binned_averages(0,var,30,9, data)
        rhs = compute_binned_averages(1,var,30,9,data)
        
        lhs_store.append(lhs)
        rhs_store.append(rhs)
    
    fig, ax = plt.subplots(2,3, figsize=(26,16))
    fontsize = 18
    
    ax[0][0].set_title("Proportion of Children from Animist Households", fontsize = fontsize)
    ax[0][0].set_xlabel("Relative Score", fontsize = fontsize)
    ax[0][0].set_ylabel("Binned Average", fontsize = fontsize)
    ax[0][0].axvline(x=0, color='r', alpha = 1)
    ax[0][0].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[0], color='orange')
    ax[0][0].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[0], color='orange')
    ax[0][0].set_ylim(0.3,0.8)
    
    ax[0][1].set_title("Proportion of Children from Muslim Households", fontsize = fontsize)
    ax[0][1].set_xlabel("Relative Score", fontsize = fontsize)
    ax[0][1].set_ylabel("Binned Average", fontsize = fontsize)
    ax[0][1].axvline(x=0, color='r', alpha = 1)
    ax[0][1].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[1], color='orange')
    ax[0][1].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[1], color='orange')
    ax[0][1].set_ylim(0,0.6)
    
    ax[0][2].set_title("Proportion of Children from Christian Households", fontsize = fontsize)
    ax[0][2].set_xlabel("Relative Score", fontsize = fontsize)
    ax[0][2].set_ylabel("Binned Average", fontsize = fontsize)
    ax[0][2].axvline(x=0, color='r', alpha = 1)
    ax[0][2].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[2], color='orange')
    ax[0][2].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[2], color='orange')
    ax[0][2].set_ylim(-0.1,0.4)
    
    ax[1][0].set_title("Proportion of Children from Fulfude-Speaking Households", fontsize = fontsize)
    ax[1][0].set_xlabel("Relative Score", fontsize = fontsize)
    ax[1][0].set_ylabel("Binned Average", fontsize = fontsize)
    ax[1][0].axvline(x=0, color='r', alpha = 1)
    ax[1][0].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[3], color='orange')
    ax[1][0].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[3], color='orange')
    ax[1][0].set_ylim(-0.1,0.4)
    
    ax[1][1].set_title("Proportion of Children from Gulmachema-Speaking Households", fontsize = fontsize)
    ax[1][1].set_xlabel("Relative Score", fontsize = fontsize)
    ax[1][1].set_ylabel("Binned Average", fontsize = fontsize)
    ax[1][1].axvline(x=0, color='r', alpha = 1)
    ax[1][1].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[4], color='orange')
    ax[1][1].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[4], color='orange')
    ax[1][1].set_ylim(-0.1,0.8)
    
    ax[1][2].set_title("Proportion of Children from Moore-Speaking Households ", fontsize = fontsize)
    ax[1][2].set_xlabel("Relative Score", fontsize = fontsize)
    ax[1][2].set_ylabel("Binned Average", fontsize = fontsize)
    ax[1][2].axvline(x=0, color='r', alpha = 1)
    ax[1][2].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[5], color='orange')
    ax[1][2].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[5], color='orange')
    ax[1][2].set_ylim(0,1.1)
    
    plt.show()  

##########################################
    
def plot_continuity_assets(variables, data): 
    
    lhs_store = []
    rhs_store = []
    
   
    for var in variables:
        
        lhs = compute_binned_averages(0,var,30,9, data)
        rhs = compute_binned_averages(1,var,30,9,data)
        
        lhs_store.append(lhs)
        rhs_store.append(rhs)
    
    fig, ax = plt.subplots(2,2, figsize=(26,16))
    fontsize = 18
    
    ax[0][0].set_title("Number of Radios per Household", fontsize = fontsize)
    ax[0][0].set_xlabel("Relative Score", fontsize = fontsize)
    ax[0][0].set_ylabel("Binned Average", fontsize = fontsize)
    ax[0][0].axvline(x=0, color='r', alpha = 1)
    ax[0][0].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[0], color='orange')
    ax[0][0].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[0], color='orange')
    ax[0][0].set_ylim(0.2,1.5)
    
    ax[0][1].set_title("Proportion of Households with a Mobile Device", fontsize = fontsize)
    ax[0][1].set_xlabel("Relative Score", fontsize = fontsize)
    ax[0][1].set_ylabel("Binned Average", fontsize = fontsize)
    ax[0][1].axvline(x=0, color='r', alpha = 1)
    ax[0][1].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[1], color='orange')
    ax[0][1].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[1], color='orange')
    ax[0][1].set_ylim(0,0.40)
    
    ax[1][0].set_title("Number of Cows Owned by Each Household", fontsize = fontsize)
    ax[1][0].set_xlabel("Relative Score", fontsize = fontsize)
    ax[1][0].set_ylabel("Binned Average", fontsize = fontsize)
    ax[1][0].axvline(x=0, color='r', alpha = 1)
    ax[1][0].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[2], color='orange')
    ax[1][0].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[2], color='orange')
    ax[1][0].set_ylim(1,9)
    
    ax[1][1].set_title("Proportion of Children in Households with Basic Flooring", fontsize = fontsize)
    ax[1][1].set_xlabel("Relative Score", fontsize = fontsize)
    ax[1][1].set_ylabel("Binned Average", fontsize = fontsize)
    ax[1][1].axvline(x=0, color='r', alpha = 1)
    ax[1][1].scatter(x = "Xgrid",y = "Binned Averages", data = lhs_store[3], color='orange')
    ax[1][1].scatter(x = "Xgrid",y = "Binned Averages", data = rhs_store[3], color='orange')
    ax[1][1].set_ylim(0.7, 1.1)

    plt.show()
    
###################
    
def plot_continuity_quad_reg(y_table, X_table, var, title, ymin, ymax): 
    
    df_store = pd.DataFrame()
    
    model = calculate_polynomial_features(1, y_table[var], X_table, 2)
    
    dep_proportion = X_table.iloc[:,4:50].mean()
    dep_coeff = model.params[4:50]
    
    predictions = model.params[2]+model.params[0]*X_table["rel_score"]+model.params[1]*X_table["rel_score"]*X_table["rel_score"]+model.params[3]*X_table["proj_selected"]+np.dot(dep_proportion,dep_coeff)
    df_store["predictions"] = predictions
    y_table["rel_score_redefined"] = X_table["rel_score"]*10000
    df_store["rel_score_redefined"] = X_table["rel_score"]*10000

    lhs = compute_binned_averages(0,var,30,9, y_table)
    rhs = compute_binned_averages(1,var,30,9,y_table)

    ax = sns.lineplot(x = "rel_score_redefined", y = "predictions", data = df_store)
    sns.scatterplot(x = "Xgrid",y = "Binned Averages", data = lhs, color='orange')
    sns.scatterplot(x = "Xgrid",y = "Binned Averages", data = rhs, color='orange')
    ax.set_xlim(-300,300)
    ax.set_ylim(ymin,ymax)
    ax.axvline(x=0, color='r', alpha = 1)
    ax.set_ylabel("Binned Averages, Predictions")
    ax.set_xlabel("Relative Score")
    ax.set_title(title)
        
    plt.show()

###############################
    
def plot_continuity_quad_reg_separated(y_df, X_df, dep_var, ymin, ymax, title):

    df_store_untreated = pd.DataFrame()
    df_store_treated = pd.DataFrame()
    
    y_untreated = y_df.loc[X_df["rel_score"] < 0].reset_index(drop = True)
    X_untreated = X_df.loc[X_df["rel_score"] < 0].reset_index(drop = True).drop("proj_selected", axis = 1)
    y_treated = y_df.loc[X_df["rel_score"] >= 0].reset_index(drop = True)
    X_treated = X_df.loc[X_df["rel_score"] >= 0].reset_index(drop = True).drop("proj_selected", axis = 1)

    model_untreated = calculate_polynomial_features(1, y_untreated[dep_var], X_untreated, 2)
    model_treated = calculate_polynomial_features(1, y_treated[dep_var], X_treated, 2)
    
    dep_proportion = X_df.iloc[:,4:50].mean()
    dep_coeff_untreated = model_untreated.params[3:50]
    dep_coeff_treated = model_treated.params[3:50]
    
    predictions_untreated = model_untreated.params[2]+model_untreated.params[0]*X_untreated["rel_score"]+model_untreated.params[1]*X_untreated["rel_score"]*X_untreated["rel_score"]+np.dot(dep_proportion,dep_coeff_untreated)
    predictions_treated = model_treated.params[2]+model_treated.params[0]*X_treated["rel_score"]+model_treated.params[1]*X_treated["rel_score"]*X_treated["rel_score"]+np.dot(dep_proportion,dep_coeff_treated)

    df_store_untreated["predictions"] = predictions_untreated
    y_untreated["rel_score_redefined"] = X_untreated["rel_score"]*10000
    df_store_untreated["rel_score_redefined"] = X_untreated["rel_score"]*10000

    df_store_treated["predictions"] = predictions_treated
    y_treated["rel_score_redefined"] = X_treated["rel_score"]*10000
    df_store_treated["rel_score_redefined"] = X_treated["rel_score"]*10000

    rhs = compute_binned_averages(1,dep_var,30,9, y_df)
    lhs = compute_binned_averages(0,dep_var,30,9, y_df)

    ax = sns.lineplot(x = "rel_score_redefined", y = "predictions", data = df_store_untreated)
    sns.lineplot(x = "rel_score_redefined", y = "predictions", data = df_store_treated)
    sns.scatterplot(x = "Xgrid",y = "Binned Averages", data = lhs, color='orange')
    sns.scatterplot(x = "Xgrid",y = "Binned Averages", data = rhs, color='orange')
    ax.set_xlim(-300,300)
    ax.set_ylim(ymin,ymax)
    ax.axvline(x=0, color='r', alpha = 1)
    ax.set_ylabel("Binned Averages, Predictions")
    ax.set_xlabel("Relative Score")
    ax.set_title(title)
        
    plt.show()



