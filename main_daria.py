import os
import numpy as np
import pandas as pd

from pyrepo_mcda.mcda_methods import PROMETHEE_II
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from daria import DARIA



def main():
    
    path = 'output_all'
    # Number of countries (evaluated alternatives)
    m = 32

    file = 'shares_' + '2020' + '.csv'
    pathfile = os.path.join(path, file)
    data = pd.read_csv(pathfile, index_col = 'Country')
    # Codes of countries (alternatives) loaded from the file with data for 2020
    list_alt_names = list(data.index)

    # list of evaluated years
    str_years = [str(y) for y in range(2013, 2021)]
    # dataframe for annual PROMETHEE II preference values
    preferences = pd.DataFrame(index = list_alt_names)

    # initialization of classical PROMETHEE II with default parameters: p, q, linear preference functions
    # for generation annual rankings for each year
    promethee_II = PROMETHEE_II()

    for el, year in enumerate(str_years):
        # load data for the following years
        file = 'shares_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        # load decision matrix
        data = pd.read_csv(pathfile, index_col = 'Country')
        # Transform dataframe with a decision matrix to NumPy array for following calculations
        matrix = data.to_numpy()
        
        # types (1 for profit type, -1 for cost type): here, all criteria are profit type
        types = np.ones(matrix.shape[1])
        
        # calculate criteria weights
        weights = mcda_weights.equal_weighting(matrix)

        # Select PROMETHEE II preference functions for criteria
        preference_functions = [promethee_II._linear_function for pf in range(len(weights))]

        # Calculate PROMETHEE II preference values for each alternative
        pref = promethee_II(matrix, weights, types, preference_functions=preference_functions)
        # Calculate PROMETHEE II ranks for each alternative
        rank = rank_preferences(pref, reverse = True)
        
        # Save PROMETHEE II preferences for each year in dataframe
        preferences[year] = pref
        

    # Create dataframe to save Temporal PROMETHEE II results
    results = pd.DataFrame(index = list_alt_names)
    
    preferences = preferences.rename_axis('Ai')

    # ======================================================================
    # Temporal PROMETHEE II
    df = preferences.T
    matrix = df.to_numpy()

    # PROMETHEE II orders preferences in descending order
    type = 1

    # Calculate efficiencies variability using methods from DARIA class
    # Create the DARIA class object
    daria = DARIA()
    # Calculate variability values for each alternative with Standard deviation using the method from DARIA class
    G = daria._std(matrix)
    # Calculate variability directions for each alternative using the method from DARIA class
    _, dir = daria._direction(matrix, type)

    # The most recent year will be updated by variability
    S = preferences['2020'].to_numpy()

    # update efficiencies using the method from DARIA class
    final_S = daria._update_efficiency(S, G, dir)

    # Calculate Temporal PROMETHEE II ranking
    rank = rank_preferences(final_S, reverse = True)
    # Save results in dataframe
    results['Variability'] = G
    results['Direction'] = dir
    results['Temporal PROMETHEE II pref.'] = final_S
    results['Temporal PROMETHEE II rank'] = rank

    print(results)

if __name__ == '__main__':
    main()