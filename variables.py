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


# Import datasets
df  = pd.read_csv('resources/XXHq.csv')
dfN = pd.read_csv('resources/XXHqn.csv') # Contains calculated variables

# // SUFFIX MEANINGS //
# _pt - probability table
# _cpt - conditional probabability table
# _new - recoded variable

# // DIATHESIS //
QN87    = dfN['qn87'] 
QN87_pt = get_binary_dist(QN87) # Homelessness during pandemic


QN94    = dfN['qn94']
QN94_pt = get_binary_dist(QN94) # Job loss

diathesis = [QN87, QN94]

# // MOTIVATIONAL STAGES //     
# Defeat
Q19  = df['q19'] # Sexual Violence
QN22 = dfN['qn22'] # Physical violence by a partner within past 12 months

QN22_new = get_recoded_var([Q19,QN22], "qn22_new") # Sexual/Dating Violence (collective)

Q23    = df['q23'] # Bullied at school
Q24    = df['q24'] # Electronically Bullied

Q23_new    = get_recoded_var([Q23,Q24], "q23_new") # Bullied (collective)
Q23_new_pt = get_binary_dist(Q23_new)

defeat = [QN22_new, Q23_new]

# Entrapment
QN85 = dfN['qn85']

QN93 = dfN['qn93']

# // OUTCOMES //
# Ideation
Q26 = df['q26'] # Considered Suicide

# Actualization
QN28 = dfN['qn28'] # Attempted suicide


# // MODERATORS // 
# Threats to Self Moderators
QN66     = dfN['qn66']
QN66_pt  = get_binary_dist(QN66) # Negative Perception of Weight

QN67        = dfN['qn67'] # Trying to lose weight
QN67_new    = get_recoded_var([QN66, QN67], "qn67_new") 
QN67_new_pt = get_binary_dist(QN67_new) # Negative Perception of Weight and Taking Action

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
QN12 = dfN['qn12'] # Carried weapon on school property
QN13 = dfN['qn13'] # Carried a gun

QN13_new    = get_recoded_var([QN12, QN13], "qn13_new") 
QN13_new_pt = get_binary_dist(QN13_new) # Carried a weapon

QN16    = dfN['qn16'] 
QN16_pt = get_binary_dist(QN16) # Was in a physical fight

Q18    = df['q18']
Q18_pt = get_binary_dist(Q18) # Saw physical violence in neighborhood

Q27    = df['q27']
Q27_pt = get_binary_dist(Q27) # Made a suicide plan

