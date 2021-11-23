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

def get_data(sample_level, second_condition_school_info, add_controls, selected_y_var, selected_var): 
    
    data = pd.read_stata('data/BRIGHT_Formatted.dta')
    data.fillna(np.nan, inplace = True)
    
    if second_condition_school_info == 0:
    
        if sample_level == 0: 
            rslt_df = data.loc[data["child_level"] == 1]
    
        elif sample_level == 1: 
            rslt_df = data.loc[data["village_level"] == 1]
    
        elif sample_level == 2: 
            rslt_df = data.loc[data["school_level"] == 1]
    
    if second_condition_school_info == 1: 
        
        if sample_level == 0: 
            rslt_df = data.loc[data["child_level"] == 1]
    
        elif sample_level == 1: 
            rslt_df = data.loc[data["village_level"] == 1]
    
        elif sample_level == 2: 
            rslt_df = data.loc[data["school_level"] == 1] 
        
        rslt_df = rslt_df.loc[rslt_df["Vill_HasSchoolInfo"] == 1]   
    
    rslt_df = pd.get_dummies(rslt_df, columns = ["department"])
    filter_col = [col for col in rslt_df if col.startswith('department')]
    dummies = rslt_df[filter_col].drop(columns=['department_close', 'department_Bani'])
    controls = rslt_df[["Ch_Girl", "Ch_HeadChild", "Ch_HeadGrandChild", "Ch_HeadNephew", "age7_dummy", "age8_dummy",                             "age9_dummy", "age10_dummy", "age11_dummy", "age12_dummy", "Hh_NumMembers", "Hh_NumKids", 
                        "Hh_HeadMale", "Hh_HeadAge", "Hh_HeadSchool", "Hh_ReligionMuslin", "Hh_Animist", 
                        "Hh_Christian", "Hh_Lang_Fulfude", "Hh_Lang_Gulmachema", "Hh_Lang_Moore",
                        "Hh_Ethnicity_Gourmanche", "Hh_Ethnicity_Mossi", "Hh_Ethnicity_Peul", "Hh_FloorBasic",
                        "Hh_RoofBasic", "Hh_Radio", "Hh_Telmob", "Hh_Watch", "Hh_Bike", "Hh_Cows", "Hh_Motorbike",
                        "Hh_Cart"]]
    
    y = rslt_df[selected_y_var]
    y.reset_index(drop= True, inplace = True)
    X = rslt_df[selected_var]    
        
    if add_controls == 0: 
        
        X = pd.concat([X, dummies], axis = 1)
        X.reset_index(drop= True, inplace = True)
        X = sm.add_constant(X)
    
    if add_controls == 1: 
        
        X = pd.concat([X, dummies, controls], axis = 1)
        X.reset_index(drop= True, inplace = True)
        X = sm.add_constant(X)
    
    return y,X

#################

def calculate_polynomial_interaction(cluster_se, y_var, initial_data, interact_var1, interact_var2, degree): 
    
    rel_score = np.array(initial_data["rel_score"])
    running_var = rel_score[:, np.newaxis]
    
    interaction_term = np.array(initial_data[interact_var1] * initial_data[interact_var2])
    interaction_term = interaction_term[:, np.newaxis]
    
    polynomial_features = PolynomialFeatures(degree=degree)
    rel_score_poly = polynomial_features.fit_transform(running_var)
    interaction_term_poly = polynomial_features.fit_transform(interaction_term)
    
    column_names_rel_score = ["rel_score" + str(x) for x in range(degree + 1)]
    column_names_interaction = ["interaction_term" + str(x) for x in range(degree + 1)]
        
    drop_columns = ["interaction_term0", "rel_score0", "rel_score", "clustercode"]
    
    rel_score_poly_df = pd.DataFrame(rel_score_poly, columns = column_names_rel_score)
    interaction_term_poly_df = pd.DataFrame(interaction_term_poly, columns = column_names_interaction)

    concated_df = pd.concat([rel_score_poly_df, interaction_term_poly_df, 
                             initial_data], axis = 1).drop(drop_columns,axis = 1)
    
    if cluster_se == 0:
        
        mod_interaction_poly = sm.OLS(y_var,concated_df)
        result_interaction_poly = mod_interaction_poly.fit()
        
    elif cluster_se == 1: 
        
        mod_interaction_poly = sm.OLS(y_var,concated_df)
        result_interaction_poly = mod_interaction_poly.fit(cov_type = 'cluster', 
                                                           cov_kwds={'groups': initial_data["clustercode"]})
    #stargazer = Stargazer([result_interaction_poly])
    return result_interaction_poly
    
#################

def calculate_probit(cluster_se, y_var, initial_data, degree):
    
    rel_score = np.array(initial_data["rel_score"])
    running_var = rel_score[:, np.newaxis]

    polynomial_features= PolynomialFeatures(degree=2)
    rel_score_poly = polynomial_features.fit_transform(running_var)
    
    column_names = []   
    for poly_degree in range(rel_score_poly.shape[1]): #can leave this for loop in, or can replace w/ list comprehension
        column_names.append("rel_score" + str(poly_degree))
    
    drop_columns = ["rel_score", "rel_score0", "const", "clustercode"]
        
    rel_score_poly_df = pd.DataFrame(rel_score_poly, columns = column_names)
    concated_df = pd.concat([rel_score_poly_df, initial_data], axis = 1).drop(drop_columns, axis = 1)
    
    if cluster_se == 0:
        
        mod_probit = sm.Probit(y_var,concated_df)
        result_probit = mod_probit.fit()
        
    elif cluster_se == 1: 
        
        mod_probit = sm.Probit(y_var,concated_df)
        result_probit = mod_probit.fit(cov_type = 'cluster', cov_kwds={'groups': initial_data["clustercode"]}) 

    #stargazer = Stargazer([result_probit])
    return result_probit                                                       
    
#################

def calculate_reduced_range_40(cluster_se, y_var, initial_data):
       
    if cluster_se == 0:
        
        drop_columns = ["rel_score", "clustercode"]
    
        new_y_var = y_var.loc[abs(initial_data["rel_score"]) < .0040].reset_index(drop = True)
        new_X_matrix = initial_data.loc[abs(initial_data["rel_score"]) < .0040].drop(drop_columns,
                                                                                     axis = 1).reset_index(drop = True)
        
        mod_reduced_range_40 = sm.OLS(new_y_var, new_X_matrix)
        result_reduced_range_40 = mod_reduced_range_40.fit()
        
    elif cluster_se == 1: 
        
        drop_columns = ["rel_score"]
    
        new_y_var = y_var.loc[abs(initial_data["rel_score"]) < .0040].reset_index(drop = True)
        new_X_matrix = initial_data.loc[abs(initial_data["rel_score"]) < .0040].drop(drop_columns,
                                                                                     axis = 1).reset_index(drop = True)
        clusters = new_X_matrix["clustercode"]  
        new_X_matrix_dropped_clcode = new_X_matrix.drop("clustercode",axis = 1)
        
        mod_reduced_range_40 = sm.OLS(new_y_var, new_X_matrix_dropped_clcode)
        result_reduced_range_40 = mod_reduced_range_40.fit(cov_type = 'cluster', cov_kwds={'groups': clusters})
    
    #stargazer = Stargazer([result_reduced_range_40])
    return result_reduced_range_40 
    
#################
    
def regress_cluster_se_corrected(y_var, initial_data, degree):
    
    rel_score = np.array(initial_data["rel_score"])
        
    running_var = rel_score[:, np.newaxis]
    
    polynomial_features= PolynomialFeatures(degree=degree)
    rel_score_poly = polynomial_features.fit_transform(running_var)

    column_names = ["rel_score" + str(x) for x in range(degree + 1)]
    drop_columns = ["rel_score", "rel_score0"]
    
    rel_score_poly_df = pd.DataFrame(rel_score_poly, columns = column_names)
    concated_df = pd.concat([rel_score_poly_df, y_var, initial_data], axis = 1).drop(drop_columns ,axis = 1)

    dataset = concated_df.dropna()
    
    clusters = dataset["clustercode"]

    drop =  ["clustercode","Ch_Highest_Grade"]

    dataset_dropped_clcode = dataset.drop(drop, axis = 1)
    y_dropped_clcode = dataset["Ch_Highest_Grade"]

    mod_poly = sm.OLS(y_dropped_clcode, dataset_dropped_clcode)
    result_poly = mod_poly.fit(cov_type = 'cluster', cov_kwds={'groups': clusters}) 
                                
    return result_poly

####################
    
def regress_over_output(cluster_se, y, initial_data): 
    
    if cluster_se == 0:
        
        model = sm.OLS(y, initial_data, missing='drop')
        res = model.fit()
    
    elif cluster_se == 1: 
        drop_columns = "clustercode"
        X_matrix = initial_data.drop(drop_columns, axis = 1)
    
        model = sm.OLS(y, X_matrix, missing='drop')
        res = model.fit(cov_type = 'cluster', cov_kwds={'groups': initial_data["clustercode"]})
    
    #stargazer = Stargazer([res])
    return res
    
#################

def regress_for_diff_outcomes_poly_features(y_df,dep_variables,X_df,degree,cluster_se):

    varying_depend_var = []
    
    for dep_var in dep_variables:
        models = calculate_polynomial_features(cluster_se, y_df[dep_var], X_df, degree)
        varying_depend_var.append(models)
        
    return varying_depend_var

###################
    
def fit_poly_models_iterate(y_df, X_df, degree, cluster_se): 
    
    fitted_poly_models = []
    
    for x in range(1, degree + 1):
        models = calculate_polynomial_features(cluster_se, y_df, X_df, x)
        fitted_poly_models.append(models)
        
    return fitted_poly_models 

####################
    
def compute_locally_weighted_reg(y_var,x_var,X,h,data): 
    
    z_array = abs((data[x_var] - X)/h)
    
    kernels = []
               
    for z in z_array: 
        
        if z <= 1:
            kz = (3/4)*(1 - z**2)/h 
        else: 
            kz = 0
        kernels.append(kz)
    
    kernels_array = np.array(kernels)
        
    weights = kernels_array**0.5
    
    y_predicted = (data[y_var])*weights
    const_predicted = weights
    x_predicted = (data[x_var] - X)*weights
    
    data["weights"] = weights
    data['y_predicted'] = y_predicted
    data['const_predicted'] = const_predicted

    data['x_predicted'] = x_predicted
    
    ols_output = smf.ols(formula = 'y_predicted ~ const_predicted + x_predicted - 1', data = data).fit()
    param = ols_output.params["const_predicted"]
        
    return param

####################

def compute_binned_averages(treated, binned_variable,binsize,bins,data): 
    
    store_xgrid = []
    store_averages =[]
    
    table = pd.DataFrame({'Xgrid': [],'Binned Averages': []})
    table['Index'] = list(range(bins))
    table = table.set_index('Index')
    
    if treated == 0: 
        
        set_range = -1*np.array(range(0,250,binsize))
        
        for value in set_range: 
            xgrid = value - binsize/2 
            store_xgrid.append(xgrid)
    
        for value in set_range: 
            subset = data[(data["rel_score_redefined"] < value) & (data["rel_score_redefined"] >= value - binsize)] 
            means = subset[binned_variable].mean()
            store_averages.append(means)
        
    if treated == 1: 
    
        set_range = list(range(0,250,binsize))
    
        for value in set_range: 
            xgrid = value + binsize/2 
            store_xgrid.append(xgrid)
    
        for value in set_range: 
            subset = data[(data["rel_score_redefined"] > value) & (data["rel_score_redefined"] <= value + binsize)] 
            means = subset[binned_variable].mean()
            store_averages.append(means)
    
    for x in range(bins): 
            averages = store_averages 
            xgrid = store_xgrid
            results = [xgrid[x],averages[x]]
            table.loc[x] = results
            table.reset_index(drop = True, inplace = True)
    
    return table
    
#################################

def test_placebo_discontinuity(y_placebo, X_placebo, treated_side, placebo_cutoff):
    
    if treated_side == 0: 
        
        y_placebo_untreated = y_placebo.loc[X_placebo["rel_score"] < 0].reset_index(drop = True)
        X_placebo_untreated = X_placebo.loc[X_placebo["rel_score"] < 0].reset_index(drop = True)

        X_placebo_untreated["placebo_treatment_dummy"] = (X_placebo_untreated["rel_score"] >= placebo_cutoff).astype(int)
        clustercodes = X_placebo_untreated["clustercode"]

        X_placebo_untreated.drop(["proj_selected", "clustercode"], inplace = True, axis = 1)
        
        mod = sm.OLS(y_placebo_untreated,X_placebo_untreated, missing='drop')
        result = mod.fit(cov_type = 'cluster', cov_kwds={'groups': clustercodes})

    if treated_side == 1: 
        y_placebo_treated = y_placebo.loc[X_placebo["rel_score"] >= 0].reset_index(drop = True)
        X_placebo_treated = X_placebo.loc[X_placebo["rel_score"] >= 0].reset_index(drop = True)

        X_placebo_treated["placebo_treatment_dummy"] = (X_placebo_treated["rel_score"] >= placebo_cutoff).astype(int)
        clustercodes = X_placebo_treated["clustercode"]
        
        X_placebo_treated.drop(["proj_selected", "clustercode"], inplace = True, axis = 1)

        mod = sm.OLS(y_placebo_treated,X_placebo_treated, missing='drop')
        result = mod.fit(cov_type = 'cluster', cov_kwds={'groups': clustercodes}) 
        
    return result

##################################

def iterate_placebo_test(y_var, X_matrix, treated_side):

    treatment_coeffs = []
    pvalues = []
    CI_lower = []
    CI_upper = []
    
    table = pd.DataFrame({'Placebo Cutoff': [],'Coefficient Estimate': [],'Coefficient Estimate': [],'P-value': [], 'Conf. Int. Lower': [], 'Conf. Int. Upper': []})
    table['Index'] = list(range(4))
    table = table.set_index('Index')
    
    if treated_side == 0:
    
        placebo_cutoffs = [-0.0125, -0.00625, -0.003125, -0.0015625]

        for cutoff in placebo_cutoffs: 
        
            summary = test_placebo_discontinuity(y_var, X_matrix, 0, cutoff)
        
            treatment_coeffs.append(summary.params["placebo_treatment_dummy"])
            pvalues.append(summary.pvalues["placebo_treatment_dummy"]) 
            CI_lower.append(summary.conf_int(alpha=0.05, cols=None).loc["placebo_treatment_dummy",:][0])
            CI_upper.append(summary.conf_int(alpha=0.05, cols=None).loc["placebo_treatment_dummy",:][1])
    
    if treated_side == 1:
    
        placebo_cutoffs = [0.0125, 0.00625, 0.003125, 0.0015625]

        for cutoff in placebo_cutoffs: 
        
            summary = test_placebo_discontinuity(y_var, X_matrix, 1, cutoff)
        
            treatment_coeffs.append(summary.params["placebo_treatment_dummy"])
            pvalues.append(summary.pvalues["placebo_treatment_dummy"]) 
            CI_lower.append(summary.conf_int(alpha=0.05, cols=None).loc["placebo_treatment_dummy",:][0])
            CI_upper.append(summary.conf_int(alpha=0.05, cols=None).loc["placebo_treatment_dummy",:][1])
    
    for x in range(4): 
        results = [placebo_cutoffs[x]*10000, treatment_coeffs[x], pvalues[x], CI_lower[x], CI_upper[x]]
        table.loc[x] = results
         
    return table

####################################
    
def check_effect_bandwidth(y, X, attending):
    
    if attending == 0: 
    
        rslts = pd.DataFrame(columns=["Treatment Effect, std", "P-value"])
        rslts.index.set_names("Bandwidth", inplace=True)
    
        for h in [1, 0.0400,0.0300, 0.0250, 0.01875, 0.0125, 
                  0.003125, 0.0015625, 0.0010, 0.0005]: 
    
            y_limited = y[X["rel_score"].between(-h, h)].reset_index(drop = True)
            X_limited = X[X["rel_score"].between(-h, h)].reset_index(drop = True)
    
            output = calculate_polynomial_interaction(1,y_limited, X_limited, "proj_selected","rel_score",1)

            info = [output.params[3], round(output.pvalues[3],3)]
            
            rslts.loc[h*10000] = info    
    
    elif attending == 1: 
        
        rslts = pd.DataFrame(columns=["Treatment Effect, %", "P-value"])
        rslts.index.set_names("Bandwidth", inplace=True)
    
        for h in [1, 0.0400,0.0300, 0.0250, 0.01875, 0.0125,
                  0.003125, 0.0015625, 0.0010, 0.0005]: 
    
            y_limited = y[X["rel_score"].between(-h, h)].reset_index(drop = True)
            X_limited = X[X["rel_score"].between(-h, h)].reset_index(drop = True)
    
            output = calculate_polynomial_interaction(1,y_limited, X_limited, "proj_selected","rel_score",1)

            info = [output.params[3] * 100, round(output.pvalues[3],3)]
            
            rslts.loc[h*10000] = info
            
    return rslts 

################################
    
def create_bandwidth_check(y_bndwd, X_bndwd, bandwidth): 
    
    scoring = "neg_mean_squared_error"
    model = LinearRegression()
    cv = LeaveOneOut()

    cross_val_score_p = partial(cross_val_score, scoring=scoring, cv=cv)

    rslts = pd.DataFrame(columns=["below", "above", "joint"])
    rslts.index.set_names("Bandwidth", inplace=True)

    for label in ["below", "above"]:
        for h in bandwidth:

            if label == "below":
                df_subset_y = y_bndwd.loc[X_bndwd["rel_score"].between(-h, +0.00)].reset_index(drop = True)
                df_subset_X = X_bndwd.loc[X_bndwd["rel_score"].between(-h, +0.00)].reset_index(drop = True)
            else:
                df_subset_y = y_bndwd.loc[X_bndwd["rel_score"].between(+0.00, +h)].reset_index(drop = True)
                df_subset_X = X_bndwd.loc[X_bndwd["rel_score"].between(+0.00, +h)].reset_index(drop = True)

            y = df_subset_y
            X = df_subset_X.drop(["const"], axis = 1)

            rslts.loc[h*10000, label] = -cross_val_score_p(model, X, y).mean()
            rslts["joint"] = rslts[["below", "above"]].mean(axis=1)
    
    return rslts

################################
    
def compute_est_effect_separate(y_df, X_df, dep_var,attending): 

    y_untreated = y_df.loc[X_df["rel_score"] < 0].reset_index(drop = True)
    X_untreated = X_df.loc[X_df["rel_score"] < 0].reset_index(drop = True).drop("proj_selected", axis = 1)
    y_treated = y_df.loc[X_df["rel_score"] >= 0].reset_index(drop = True)
    X_treated = X_df.loc[X_df["rel_score"] >= 0].reset_index(drop = True).drop("proj_selected", axis = 1)

    model_untreated = calculate_polynomial_features(1, y_untreated[dep_var], X_untreated, 2)
    model_treated = calculate_polynomial_features(1, y_treated[dep_var], X_treated, 2)
    
    dep_proportion = X_df.iloc[:,4:50].mean()
    dep_coeff_untreated = model_untreated.params[3:49]
    dep_coeff_treated = model_treated.params[3:49]
    
    const_untreated = model_untreated.params["const"] + np.dot(dep_proportion,dep_coeff_untreated)
    const_treated = model_treated.params["const"] + np.dot(dep_proportion,dep_coeff_treated)
    
    if attending == 0:
        
        est_effect = (const_treated - const_untreated)
        
        print(f'The measured effect on normalized total test scores for separately plotted control and treatment regressions is {est_effect:.4f} standard deviations.')
    
    if attending == 1: 
        
        est_effect = (const_treated - const_untreated) * 100
    
        print(f'The measured effect on attendance for separately plotted control and treatment regressions is {est_effect:.4f}%.')
    
    
    
