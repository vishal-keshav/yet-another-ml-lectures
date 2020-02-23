config={
'project_name': 'project',
'workspace': 'workspace',
######################## USE LISTS FOR HPs(GRID SEARCH) ########################
'learning_rate': [0.0001, 0.00001],
'nr_epochs': [100, 200],
'batch_size': [8, 16],
########################## ENABLE THIS FOR HP SEARCH ###########################
#'metric': 'loss',
#'objective': 'minimize',
#'algorithm': 'bayes'
}