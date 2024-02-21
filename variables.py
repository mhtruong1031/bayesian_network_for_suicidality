import pandas as pd

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
# _new - recoded variable

# // DIATHESIS //
QN87      = dfN['qn87'] # Homelessness during pandemic

QN94      = dfN['qn94'] # Job loss

diathesis = [QN87, QN94]

# // MOTIVATIONAL STAGES //   
  
# Defeat
Q19      = df['q19'] # Sexual Violence
QN22     = dfN['qn22'] # Physical violence by a partner within past 12 months
QN22_new = get_recoded_var([Q19,QN22], "qn22_new") # Sexual/Dating Violence (collective)

Q23      = df['q23'] # Bullied at school
Q24      = df['q24'] # Electronically Bullied
Q23_new  = get_recoded_var([Q23,Q24], "q23_new") # Bullied (collective)

defeat = [QN22_new, Q23_new]

# Entrapment
QN85       = dfN['qn85'] # Poor mental health (past 12 months)

QN93       = dfN['qn93'] # Poor mental health over COVID-19 pandemic

entrapment = [QN85, QN93]

# // OUTCOMES //

# Ideation
Q26      = df['q26'] # Considered Suicide

ideation = [Q26]

# Actualization
QN28     = dfN['qn28'] # Attempted suicide

behavior = [QN28]

# // MODERATORS // 

# Threats to Self Moderators
QN66     = dfN['qn66'] # Negative Perception of Weight

QN67     = dfN['qn67'] # Trying to lose weight
QN67_new = get_recoded_var([QN66, QN67], "qn67_new") # Negative Perception of Weight and Taking Action

Q98      = df['q98'] # Difficulty concentrating

QNILLICT = dfN['qnillict'] # Ever used selct illicit drugs

threats_to_self_moderators = [QN66, QN67_new, Q98, QNILLICT]

# Motivational Moderators
Q25                     = df['q25'] # Sad or Hopeless

QN96                    = dfN['qn96'] # Feel close to people at school

motivational_moderators = [Q25, QN96]

# Volitional Moderators
QN12     = dfN['qn12'] # Carried weapon on school property
QN13     = dfN['qn13'] # Carried a gun
QN13_new = get_recoded_var([QN12, QN13], "qn13_new") # Carried a weapon

QN16     = dfN['qn16'] # Was in a physical fight

Q18      = df['q18'] # Saw physical violence in neighborhood

Q27      = df['q27'] # Made a suicide plan

volitional_moderators = [QN13_new, QN16, Q18, Q27]

all_var       = pd.concat(diathesis+defeat+entrapment+ideation+behavior+threats_to_self_moderators+motivational_moderators+volitional_moderators, axis=1).dropna().reset_index()

training_idx  = round(len(all_var.index) * .80)
training_data = all_var.loc[[i for i in range(training_idx)]]
testing_data  = all_var.loc[[i for i in range(training_idx, len(all_var.index))]].reset_index()

# Relationship in the IMV Model
relationships = (
    (defeat, entrapment),
    (threats_to_self_moderators, entrapment),
    (entrapment, ideation),
    (motivational_moderators, ideation),
    (ideation, behavior),
    (volitional_moderators, behavior),
    (diathesis, behavior)
)