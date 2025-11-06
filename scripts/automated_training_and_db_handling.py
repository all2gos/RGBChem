import pandas as pd

def places(test_mae:int, epoch:int):
    from scripts.params import BATCH_SIZE, CYCLE, DB, LEARNING_RATE, LOG_FILE, MATRIX_SIZE, MOMENTUM, PATH, PREDICTED_VALUE, SHUFFLE, TYPE_OF_IMAGE, PATIENCE, DELTA, MODEL, RANDOM_OR, MARGIN, RESIZE, DROPOUT

    print('Essential data, do not commit !!!')


    criteria = [DB, TYPE_OF_IMAGE, CYCLE, SHUFFLE, MODEL]
    criteria_names = ['DB', 'TYPE_OF_IMAGE', 'CYCLE', 'SHUFFLE', 'MODEL']
    def load_data(sheets_url):
        csv_url = sheets_url.replace("/edit?gid=", "/export?format=csv&gid=")
        return pd.read_csv(csv_url, index_col=0)
    
    df = load_data("https://docs.google.com/spreadsheets/d/19XoQ3WYAo8-CXQCRo5otEnLYAxfNyFSgy9LqvQ-he50/edit?gid=0#gid=0")
    df = pd.read_csv('/home/rstottko/RGBChem/models_db.csv')
    #df = df.reset_index()


    print(df.head())
    print(df.SHUFFLE.unique())
    df['epochs'] = pd.to_numeric(df['EPOCHS'], errors='coerce')


    for col in ['BATCH_SIZE','RESIZE','PATIENCE','LR','MOMENTUM', 'CYCLE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')



    for c in range(len(criteria)):
        #print(f"Value of {criteria_names[c]} for current run is {criteria[c]}")
        all_models = len(df[df[criteria_names[c]] == criteria[c]]) +1
        better_models = len(df[(df[criteria_names[c]] == criteria[c]) & (df['ACCURACY'] < test_mae)])
        print(f"Current model is --{better_models+1}/{all_models}-- out of models with {criteria_names[c]} = {criteria[c]}")


    print('All model with the same critical parameters:')
    print(df[
        (df[criteria_names[0]] == criteria[0]) &
        (df[criteria_names[1]] == criteria[1]) &
        (df[criteria_names[2]] == criteria[2]) &
        (df[criteria_names[3]] == criteria[3]) &
        (df[criteria_names[4]] == criteria[4])
    ][['ID', 'ACCURACY']])    


    new_row = {'ID':f"M{len(df)+1}",
               'DB':DB,
               'CYCLE': CYCLE,
               'N_DATA_POINTS': 'tba',
               'TYPE_OF_IMAGE': TYPE_OF_IMAGE,
               'SHUFFLE': SHUFFLE,
               'MODEL': MODEL,
               'EPOCHS': epoch,
               'BATCH_SIZE':BATCH_SIZE,
               'MARGIN':MARGIN,
               'RESIZE':RESIZE,
               'RANDOM_OR': RANDOM_OR,
               'LR': LEARNING_RATE,
               'MOMENTUM': MOMENTUM,
               'PATIENCE': PATIENCE,
               'PREDICTED_VALUE':PREDICTED_VALUE,
               'ACCURACY':test_mae,
               'ITER':5,
               'IMG': ''}
    print('---------------------------------------------------------')
    print(f"---This model has been identified as {new_row['ID']}---")
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print('---------------------------------------------------------')

    df.to_csv('models_db.csv', index=False)


    
