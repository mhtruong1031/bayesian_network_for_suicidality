from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork

# Pre-disposition
diathesis = Categorical()

# Motivational Stages
defeat     = Categorical()
entrapment = ConditionalCategorical()
motivational_stages = [defeat, entrapment]

# Outcomes
ideation      = ConditionalCategorical()
actualization = ConditionalCategorical()
outcomes = [ideation, actualization]

# Moderators
threats_to_self_moderators = Categorical()
motivational_moderators    = Categorical()
volitional_moderators      = Categorical()
moderators = [threats_to_self_moderators, motivational_moderators, volitional_moderators]

# Model init
nodes = [diathesis] + motivational_stages + outcomes + moderators
model = BayesianNetwork()
model.add_distributions(nodes)


