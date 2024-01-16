import pandas as pd
from sklearn.linear_model import LogisticRegression
from variables import relationships

regression = LogisticRegression()

def get_univariate_results(input: pd.Series, output: pd.Series):
    clean_data = pd.concat([input, output], axis=1).dropna() # remove Nan values
    n_samples  = len(clean_data.index)

    x = clean_data[input.name].to_numpy().reshape((-1, 1))
    y = clean_data[output.name].to_numpy().reshape((n_samples,))

    regression.fit(x, y)

    return regression.score(x, y)

def main() -> None:
    log = "Univariate Results\n--------------------------------------------------------------------\n"
    for relationship in relationships:
        input  = relationship[0]
        output = relationship[1]

        for var_i in input:
            for var_o in output:
                log += f"i: {var_i.name}  o: {var_o.name}  |  coef: {get_univariate_results(var_i, var_o)}\n"
    print(log)
    with open('results/univariate.txt', 'a') as f:
        f.write(log)


if __name__ == '__main__':
    main()