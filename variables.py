import pandas as pd


def get_binary_dist(column: pd.Series) -> tuple: 
    outcome_counts = column.value_counts().values # (2, 1)

    true_dist      = outcome_counts[1]/sum(outcome_counts)
    false_dist     = outcome_counts[0]/sum(outcome_counts)

    return (true_dist, false_dist) 

def get_recoded_var(var_list: list, var_name: str) -> pd.Series:
    new_data = []
    for row in pd.concat(var_list, axis=1).itertuples():
        if row[1] == 1 or row[2] == 1:
            new_data.append(1.0)
        else:
            new_data.append(2.0)

    return pd.Series(new_data, name=var_name) 

df  = pd.read_csv('resources/XXHq.csv')
dfN = pd.read_csv('resources/XXHqn.csv') # Contains calculated variables


# _pt - probability table
# _cpt - conditional probabability table
# _new - recoded variable

# // DIATHESIS //
QN87    = dfN['qn87'] 
QN87_pt = get_binary_dist(QN87) # Homelessness during pandemic


QN94    = dfN['qn94']
QN94_pt = get_binary_dist(QN94) # Job loss


# // MOTIVATIONAL STAGES //     
# Defeat
Q19  = df['q19'] # Sexual Violence
QN22 = dfN['qn22'] # Physical violence by a partner within past 12 months

QN22_new = get_recoded_var([Q19,QN22], "qn22_new") # Sexual/Dating Violence (collective)

Q23    = df['q23'] # Bullied at school
Q24    = df['q24'] # Electronically Bullied

Q23_new = get_recoded_var([Q23,Q24], "q23_new") # Bullied (collective)

# Entrapment


# // OUTCOMES //
# Ideation
Q26 = df['q26'] # Considered Suicide

# Actualization
QN28 = dfN['qn28'] # Attempted suicide


# // MODERATORS // 
# Threats to Self Moderators
Q98    = df['q98']
Q98_pt = get_binary_dist(Q98) # Difficulty concentrating

QNILLICT    = dfN['qnillict']
QNILLICT_pt = get_binary_dist(QNILLICT) # Ever used selct illicit drugs

# Motivational Moderators
Q25    = df['q25']
Q25_pt = get_binary_dist(Q25) # Sad or Hopeless

QN96    = dfN['qn96']
QN96_pt = get_binary_dist(QN96) # Feel close to people at school

# Volitional Moderators
Q27    = df['q27']
Q27_pt = get_binary_dist(Q27) # Made a suicide plan

