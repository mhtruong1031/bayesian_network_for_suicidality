import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
from variables import relationships

regression = LogisticRegression()

def get_uni_logit_results(input: pd.Series, output: pd.Series):
    clean_data = pd.concat([input, output], axis=1).dropna() # remove Nan values
    n_samples  = len(clean_data.index)

    x = clean_data[input.name].to_numpy().reshape((-1, 1))
    y = clean_data[output.name].to_numpy().reshape((n_samples,))

    regression.fit(x, y)

    return regression.score(x, y)

def get_uni_chi2_results(input: pd.Series, output: pd.Series):
    clean_data        = pd.concat([input, output], axis=1).dropna() # remove Nan values
    contingency_table = pd.crosstab(clean_data[output.name], clean_data[input.name])

    return chi2_contingency(contingency_table).pvalue

def main() -> None:
    log_logit = "Univariate Results\n--------------------------------------------------------------------\n"
    log_chi2  = "Univariate Results\n--------------------------------------------------------------------\n"
    for relationship in relationships:
        input  = relationship[0]
        output = relationship[1]

        for var_i in input:
            for var_o in output:
                log_logit += f"i: {var_i.name}  o: {var_o.name}  |  coef: {get_uni_logit_results(var_i, var_o)}\n"
                log_chi2  += f"i: {var_i.name}  o: {var_o.name}  |  coef: {get_uni_chi2_results(var_i, var_o)}\n"
    
    with open('results/uni_logit.txt', 'w') as f:
        f.write(log_logit)
    with open('results/uni_chi2.txt', 'w') as f:
        f.write(log_chi2)


if __name__ == '__main__':
    main()