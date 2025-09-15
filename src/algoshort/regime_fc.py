def hilo_alternation(hilo, dist= None, hurdle= None):
    i=0    
    while (np.sign(hilo.shift(1)) == np.sign(hilo)).any(): # runs until duplicates are eliminated

        # removes swing lows > swing highs
        hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo)) &  # hilo alternation test
                 (hilo.shift(1)<0) &  # previous datapoint:  high
                 (np.abs(hilo.shift(1)) < np.abs(hilo) )] = np.nan # high[-1] < low, eliminate low 

        hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo)) &  # hilo alternation
                 (hilo.shift(1)>0) &  # previous swing: low
                 (np.abs(hilo ) < hilo.shift(1))] = np.nan # swing high < swing low[-1]

        # alternation test: removes duplicate swings & keep extremes
        hilo.loc[(np.sign(hilo.shift(1)) == np.sign(hilo)) & # same sign
                 (hilo.shift(1) < hilo )] = np.nan # keep lower one

        hilo.loc[(np.sign(hilo.shift(-1)) == np.sign(hilo)) & # same sign, forward looking 
                 (hilo.shift(-1) < hilo )] = np.nan # keep forward one

        # removes noisy swings: distance test
        if pd.notnull(dist):
            hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo))&\
                 (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1)< hurdle)] = np.nan

        # reduce hilo after each pass
        hilo = hilo.dropna().copy() 
        i+=1
        if i == 4: # breaks infinite loop
            break 
        return hilo