import optuna


if __name__ == '__main__':
    study = optuna.load_study('hparam_search', storage='sqlite:///hparam_search.db')
    # print(f'Best result was {study.best_value} (trial {study.best_trial.number})')
    # print(f'Best hparams were {study.best_params}')
    print('Trials:')
    df = study.trials_dataframe(("number", "value", "datetime_complete", "state"))
    print(df)

    while True:
        trial_number = input('\nEnter a trial number for more details\n>>> ').strip()
        if len(trial_number) == 0:
            break
        trial = next(trial
                     for trial in study.trials
                     if trial.number == int(trial_number))  # type: optuna.structs.FrozenTrial

        for field in trial._ordered_fields:
            print(f'{field}: {getattr(trial, field)}')
