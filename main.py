import pandas as pd
from model import model
from variables import testing_data

def main() -> None:
    predictions = testing_data.drop(labels=['q26', 'qn28', 'index', 'level_0'], axis=1) # pandas library correction
    predictions = model.predict(predictions)

    q26_correct  = 0
    qn28_correct = 0
    for i in range(len(testing_data.index)):
        if predictions.loc[i, 'q26'] == testing_data.loc[i, 'q26']:
            q26_correct += 1
        if predictions.loc[i, 'qn28'] == testing_data.loc[i, 'qn28']:
            qn28_correct += 1

    print(f"q26 accuracy: {q26_correct/len(testing_data.index)}") # print prediction accuracy
    print(f"qn28 accuracy: {qn28_correct/len(testing_data.index)}")

if __name__ == '__main__':
    main()