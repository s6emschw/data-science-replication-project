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

def create_table(models_list, covariate_list, table_title, covariate_names_dict, notes):
    
    table = Stargazer(models_list)
    table.covariate_order(covariate_list)
    table.title(table_title)
    table.rename_covariates(covariate_names_dict)
    table.show_degrees_of_freedom(False)
    table.add_custom_notes(notes)
    
    return table

#################

def calculate_polynomial_features(cluster_se, y_var, initial_data, degree):
        
    rel_score = np.array(initial_data["rel_score"])
        
    running_var = rel_score[:, np.newaxis]
    
    polynomial_features= PolynomialFeatures(degree=degree)
    rel_score_poly = polynomial_features.fit_transform(running_var)

    column_names = ["rel_score" + str(x) for x in range(degree + 1)]
    drop_columns = ["rel_score", "rel_score0", "clustercode"]
    
    rel_score_poly_df = pd.DataFrame(rel_score_poly, columns = column_names)
    concated_df = pd.concat([rel_score_poly_df, initial_data], axis = 1).drop(drop_columns ,axis = 1)

    if cluster_se == 0:
        
        mod_poly = sm.OLS(y_var,concated_df, missing='drop')
        result_poly = mod_poly.fit()
        
    elif cluster_se == 1: 
        
        mod_poly = sm.OLS(y_var,concated_df, missing='drop')
        result_poly = mod_poly.fit(cov_type = 'cluster', cov_kwds={'groups': initial_data["clustercode"]}) 
                                
        
    #stargazer = Stargazer([result_poly])
    return result_poly
    
####################
    
def create_table_df(y_df,dep_variables, X_df, table_means, cluster_se):
    
    if table_means == 0:

        table = pd.DataFrame({'Regression Coefficient at Cutoff': [], 'Standard Error': [], 'P-value' : []})
    
        table['Outcome Variables'] = dep_variables
        table = table.set_index('Outcome Variables')
  
        for dep_var in dep_variables: 
            summary = calculate_polynomial_features(cluster_se,y_df[dep_var],X_df,2)
            results = [summary.params["proj_selected"],summary.bse["proj_selected"],summary.pvalues["proj_selected"]]
            table.loc[dep_var] = results
            
    if table_means == 1:

        table = pd.DataFrame({'Regression Coefficient at Cutoff': [], 'Standard Error': [], 'P-value' : [], 
                              'Sample Average': []})
    
        table['Outcome Variables'] = dep_variables
        table = table.set_index('Outcome Variables')
  
        for dep_var in dep_variables: 
            summary = calculate_polynomial_features(cluster_se,y_df[dep_var],X_df,2)
            means = y_df[dep_var].mean() 
            results = [summary.params["proj_selected"],summary.bse["proj_selected"],summary.pvalues["proj_selected"],
                      means]
            table.loc[dep_var] = results
    
    if table_means == 2:

        table = pd.DataFrame({'Sample Average for Non-Selected Villages' : [], 
                              'Sample Average for Selected Villages' : [],
                              'Regression Coefficient at Cutoff': [], 'Standard Error': [], 'P-value' : [], 
                              })
    
        table['Outcome Variables'] = dep_variables
        table = table.set_index('Outcome Variables')
  
        for dep_var in dep_variables: 
        
            means_non_sel = y_df[dep_var][X_df["proj_selected"] == 0].mean()
            means_sel = y_df[dep_var][X_df["proj_selected"] == 1].mean()
            summary = calculate_polynomial_features(cluster_se,y_df[dep_var],X_df,2)
            
            results = [means_non_sel, means_sel, summary.params["proj_selected"],
                       summary.bse["proj_selected"],summary.pvalues["proj_selected"]]
            table.loc[dep_var] = results
        
    
    table = table.round(3)
    
    return table
    
##################################
    
def compute_X_axis(xmax,xmin,gsize):
    
    st = (xmax - xmin)/(gsize - 1)
    
    store_X = []
    for x in range(1, gsize + 1): 
        xgrid = xmin + (x-1)*st 
        store_X.append(xgrid) 
    
    return store_X 

##################################

def compute_R_sq(y_var,running_var,data,X): 
    
    dummy = []
    
    for row in data[running_var]:
        
        if row >= X:
            dummy.append(1)
        else:
            dummy.append(0)
            
    data["dummy"] = dummy 
    
    y = data[y_var]
    X_matrix = sm.add_constant(data["dummy"])
    
    model = sm.OLS(y, X_matrix, data = data).fit().rsquared

    return model

####################################
    
def create_R_sq_df(y_var, running_var, data):
    
    X_grid = compute_X_axis(250,-250,501)
    R_sq_iterated = [compute_R_sq(y_var,running_var,data, item) for item in X_grid]

    R_sq_graph_df = pd.DataFrame({"Xgrid for R-squared Values": [], "R-squared Values": []}) 

    R_sq_graph_df["Index"] = list(range(len(R_sq_iterated)))
    R_sq_graph_df = R_sq_graph_df.set_index("Index")
    
    for x in range(len(R_sq_iterated)): 
        r_sq = R_sq_iterated
        xgrid = X_grid
        results = [xgrid[x],r_sq[x]]
        R_sq_graph_df.loc[x] = results
        R_sq_graph_df.reset_index(drop = True, inplace = True)
   
    return R_sq_graph_df

#####################################
    
def create_continuity_table(characteristics):
    
    mean_table_2_var = round(characteristics.groupby("proj_selected").mean(), 3).drop(columns = ["rel_score_redefined"])
    std_table_2_var = round(characteristics.groupby("proj_selected").std(), 3).drop(columns = ["rel_score_redefined"])

    table_selected = pd.DataFrame()
    table_nonselected = pd.DataFrame()

    table_nonselected["Mean"] = mean_table_2_var.loc[0]
    table_selected["Mean"] = mean_table_2_var.loc[1]

    table_nonselected["Standard Deviation"] = std_table_2_var.loc[0]
    table_selected["Standard Deviation"] = std_table_2_var.loc[1]

    table = pd.concat([table_nonselected, table_selected],axis = 1)
    table.columns = pd.MultiIndex.from_product([["Control Villages", "Treatment Villages"],
                                                ["Mean", "Standard Deviation"]])

    return table

##################################

def create_table_E1(y_df,dep_variables,X_df): 
    
    table_proj_sel = pd.DataFrame({"a":[], "b":[]})
    table_fem = pd.DataFrame({"a":[], "b":[]})
    
    for var in dep_variables: 
        
        summary = calculate_polynomial_features(1,y_df[var],X_df,2)
        results_proj_sel = [summary.params[3], summary.pvalues[3]]
        table_proj_sel.loc[var] = results_proj_sel
        
        results_fem = [summary.params[4], summary.pvalues[4]]                                             
        table_fem.loc[var] = results_fem
        
        table = pd.concat([table_proj_sel, table_fem],axis = 1)
        table.columns = pd.MultiIndex.from_product([["BRIGHT Program (Selected)", "Selected x Female"],
                                                    ["Coefficient", "P-value"]])

    return table.round(3)





