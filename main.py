from Utils import *
from Model import *
import sys
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def main():
    # print('start loading ')
    matchesDF1 = load(r"C:\Users\Asus\Downloads\recentMatches.csv")
    matchesDF2 = load(r"C:\Users\Asus\Downloads\recentMatches2.csv")
    matchesDF3 = load(r"C:\Users\Asus\Downloads\recentMatches3.csv")
    seed_everything(42)
    print('start preparing data')
    allMatchesDF = prepare_data(matchesDF1, matchesDF2, matchesDF3)
    allMatchesDF["result"] = allMatchesDF["res"].apply(convert_label)
    for i in allMatchesDF.columns[4:9]:
        allMatchesDF[i] = allMatchesDF[i].apply(convert_home_Last)
    for i in allMatchesDF.columns[9:14]:
        allMatchesDF[i] = allMatchesDF[i].apply(convert_away_Last)
    # allMatchesDF_augmented = data_augmentation (allMatchesDF)
    print('Patch extraction')
    teamPatches = patch_extractor(r"C:\Users\Asus\Downloads\teamPatches.json")
    full_columns = prepare_column_names(allMatchesDF)
    print('preparing columns')
    # allMatchesDF_extented,not_found =  extend_data (full_columns)
    HF = Home_Form(allMatchesDF[['homeLast1', 'homeLast2', 'homeLast3', 'homeLast4', 'homeLast5']])
    #  AF = Away_Form (allMatchesDF[['awayLast1','awayLast2','awayLast3','awayLast4','awayLast5']])
    allMatchesDF['Home_Form'] = HF
    # allMatchesDF['Away_Form'] =  AF
    pourcentage = victory_pourcentage(allMatchesDF[['homeWins', 'matchesPlayed']])
    pourcentage100 = [x * 100 for x in pourcentage]
    pourcentage_carre = [x ** 2 for x in pourcentage]
    allMatchesDF['pourcentage'] = pourcentage100
    allMatchesDF['pourcentage_carre'] = pourcentage_carre
    big_home = big_home_team(allMatchesDF[['homeTeam']])
    allMatchesDF['big_home'] = big_home
    # big_away = big_away_team (allMatchesDF[['awayTeam']])
    # allMatchesDF['big_away'] = big_away
    Attaque_home = force_attaque_home(allMatchesDF)
    for x in Attaque_home:
        allMatchesDF.loc[(allMatchesDF['homeTeam'] == x[0][0]) & (allMatchesDF['season'] == x[0][1]), 'Goal_scored_home'] = x[1]
    Attaque_away = force_attaque_away(allMatchesDF)
    for x in Attaque_away:
        allMatchesDF.loc[(allMatchesDF['homeTeam'] == x[0][0]) & (allMatchesDF['season'] == x[0][1]), 'Goal_scored_away'] = x[1]

    Defense_Home = force_defence_home(allMatchesDF)
    for x in Defense_Home:
        allMatchesDF.loc[(allMatchesDF['homeTeam'] == x[0][0]) & (allMatchesDF['season'] == x[0][1]), 'Goal_conceded_home'] = x[1]

    Defense_away = force_Defense_away(allMatchesDF)
    for x in Defense_away:
        allMatchesDF.loc[(allMatchesDF['homeTeam'] == x[0][0]) & (allMatchesDF['season'] == x[0][1]), 'Goal_conceded_away'] = x[1]

    Pts_home = force_home(allMatchesDF)
    for x in Pts_home:
        allMatchesDF.loc[(allMatchesDF['homeTeam'] == x[0][0]) & (allMatchesDF['season'] == x[0][1]), 'force_home'] = x[1]

    Pts_away = force_away(allMatchesDF)
    for x in Pts_away:
        allMatchesDF.loc[(allMatchesDF['homeTeam'] == x[0][0]) & (allMatchesDF['season'] == x[0][1]), 'force_away'] = x[1]

    allMatchesDF['difference'] = allMatchesDF['force_home'] - allMatchesDF['force_away']
    allMatchesDF['home_scoring_prob'] = allMatchesDF['Goal_scored_home'] - allMatchesDF['Goal_conceded_away']
    allMatchesDF['away_scoring_prob'] = allMatchesDF['Goal_scored_away'] - allMatchesDF['Goal_conceded_home']
    print('splitting data')
    train, test = splitting_data(allMatchesDF)
    features_columns_lgbm = features_utils_lgbm(train)
    X, y, skf = divide_train(train,features_columns_lgbm)
    print('train')
    models = StratifiedKFold_Train(X, y,skf,train,test,features_columns_lgbm)
    print('scoring')
    lgb_pred = predict(models, features_columns_lgbm,test)
    print ('the accuracy score is about ' , accuracy_score(lgb_pred,test['result']) )

if __name__ == "__main__":
    main()