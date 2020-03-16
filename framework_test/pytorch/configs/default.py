config={
'project_name': 'project',
'workspace': 'workspace',
######################## USE LISTS FOR HPs(GRID SEARCH) ########################
'learning_rate': [0.001, 0.1],
'nr_epochs': [200, 1000],
'batch_size': [8,16],
########################## ENABLE THIS FOR HP SEARCH ###########################
#'metric': 'loss',
#'objective': 'minimize',
#'algorithm': 'bayes'
}