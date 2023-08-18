"""
*** OBJECTIVE ***
Data preprocessing for classification on Home Credit Risk assessment

*** CONTENTS ***

- Data leak prevention (many onehot encoded columns missing when test_set sample is small :
    Reason : not all categorical values represented.
    > Verification of preprocessed test_file columns in regard of preprocessed train_set columns
    > If missing and Onehot encoded column, population with 0

- Inf replaced by NA, and NA treatment
    > OneHotted columns : fillna with 0s
    > Delete very poorly filled columns
    > Remaining columns : fillna with median (median of train set if test set preprocessed)

*** CREDITS ***
Data source : https://www.kaggle.com/competitions/home-credit-default-risk
Code adapted from : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script
"""

import numpy as np
import pandas as pd
import gc
import time
import json
from contextlib import contextmanager
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, prefix_sep='_OHE_')
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# >>>  PREPROCESSING FUNCTIONS  <<<  #

# Preprocess application_x.csv
def application(df, nan_as_category=False):
    # Remove 4 applications with XNA (CODE_GENDER)
    df = df[df['CODE_GENDER'] != 'XNA']
    # Binary encoding of binary categorical features (0 or 1)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # One-Hot encoding for other categorical features
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Adding some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    gc.collect()
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(bureau, bb, nan_as_category=True):
    # Import et OneHot encode cat features
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    # Data agglomeration per current application
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_OHE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_OHE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(prev, nan_as_category=True):
    # OneHot encode
    prev, cat_cols = one_hot_encoder(prev, nan_as_category)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_OHE_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_OHE_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(pos, nan_as_category=True):
    # OneHot
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)

    # Aggregation features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(ins, nan_as_category=True):
    # OneHot
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(cc, nan_as_category=True):
    # OneHot
    cc, cat_cols = one_hot_encoder(cc, nan_as_category)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


# >>>  JOIN FUNCTION  <<<  #

# Limitation to useful sk_ids
def sk_ids_selection(data_path, file_name, sk_ids):
    file_import = pd.read_csv(data_path + file_name)
    file_limited = file_import.loc[file_import.SK_ID_CURR.isin(sk_ids)]
    return file_limited


# Import and file generation, from csv file of applications to evaluate
def import_and_file_generation(data_path='./data/', entry_file='application_train.csv'):
    # Import
    df = pd.read_csv(data_path + entry_file)
    sk_ids = df.SK_ID_CURR.tolist()
    print(f"Samples nb: {len(df)}")

    # Import other files and limit to useful SK_ID_CURR
    bureau = sk_ids_selection(data_path, 'bureau.csv', sk_ids)
    sk_bureau = bureau.SK_ID_BUREAU.tolist()
    bb_import = pd.read_csv(data_path + 'bureau_balance.csv')
    bb = bb_import.loc[bb_import.SK_ID_BUREAU.isin(sk_bureau)]
    prev = sk_ids_selection(data_path, 'previous_application.csv', sk_ids)
    pos = sk_ids_selection(data_path, 'POS_CASH_balance.csv', sk_ids)
    ins = sk_ids_selection(data_path, 'installments_payments.csv', sk_ids)
    cc = sk_ids_selection(data_path, 'credit_card_balance.csv', sk_ids)

    return df, bureau, bb, prev, pos, ins, cc


# Data leak Verification for preprocessed entry files that are not training files
def data_leak_verification(df, is_training_file=False):
    df_populated = df.copy()
    columns_remaining = []

    # If training file, record columns
    if is_training_file:
        train_columns = df.columns.tolist()
        with open('./refs/train_columns.json', 'w', encoding='utf-8') as f:
            json.dump(train_columns, f, ensure_ascii=False, indent=4)

    # If not training file, see what columns are missing compared to training columns
    else:
        f = open('./refs/train_columns.json')
        train_columns = json.load(f)
        # Remove TARGET column from train columns if present
        if 'TARGET' in train_columns:
            train_columns.remove('TARGET')
        # Isolate missing columns
        df_columns = df.columns.tolist()
        missing_columns = []
        for col in train_columns:
            if col not in df_columns:
                missing_columns.append(col)
        # Check if missing columns = categorical, then populate with 0s
        columns_populated = []
        print('...Missing columns :', len(missing_columns))
        if len(missing_columns) > 0:
            for col in missing_columns:
                if '_OHE_' in col:
                    populated_col = pd.Series(data=np.zeros(len(df)), name=col)
                    df_populated = pd.concat((df_populated, populated_col), axis=1)
                    columns_populated.append(col)
                else:
                    columns_remaining.append(col)
            print('...Missing columns populated :', len(columns_populated))
            print('...Missing columns remaining (not OHE) :', len(columns_remaining))
        else:
            print('...All clear, no missing columns')

    return df_populated, columns_remaining


# Global Feature engineering
def feature_engineering(data_path='../data/', entry_file='application_train.csv', is_training_file=False):
    print('---------------')
    print('PREPROCESSING')
    print('---------------')

    with timer('Data preprocessing'):
        df, bureau, bb, prev, pos, ins, cc = import_and_file_generation(data_path, entry_file)

        df = application(df, nan_as_category=False)

        bureau = bureau_and_balance(bureau, bb)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()

        prev = previous_applications(prev)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()

        pos = pos_cash(pos)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()

        ins = installments_payments(ins)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()

        cc = credit_card_balance(cc)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()

        print('Dataframe shape before Data leak verification :', df.shape)

        # DataLeak Verification
        df, columns_remaining = data_leak_verification(df, is_training_file)

        # Set index = SK_ID + sort columns
        df.set_index('SK_ID_CURR', inplace=True)
        df.sort_index(axis=1, ascending=True, inplace=True)

        print('Dataframe shape after Data leak correction :', df.shape)
        print('---------------')

    return df


# >>> INF & NA TREATMENT <<<

def separation_colonnes(df):
    colonnes = df.columns.tolist()
    cols_OHE, cols_autres = [], []
    for col in colonnes:
        if '_OHE_' in col:
            cols_OHE.append(col)
        else:
            cols_autres.append(col)
    return cols_OHE, cols_autres


# OneHot encoded columns
def fill_NA_colonnes_OHE(df, cols_OHE):
    for col in cols_OHE:
        df[col].fillna(value=0, inplace=True)
    return df


# Drop empty cols
def pourcentage_na(df, col) :
    return df[col].isna().sum()/len(df)


# --- if Train set
def remove_colonnes_NA_superieures_a(df, cols_autres, pc_NA_max=0.5) :
    # List columns having NA% > percentage and drop from dataframe
    cols_a_enlever = []
    for col in cols_autres :
        if pourcentage_na(df, col) > pc_NA_max:
            cols_a_enlever.append(col)
    df = df.drop(cols_a_enlever, axis=1)
    # Dump for further use on testing sets
    with open('./refs/unused_columns.json', 'w', encoding='utf-8') as f:
        json.dump(cols_a_enlever, f, ensure_ascii=False, indent=4)
    return df, cols_a_enlever


# --- if Test set
def remove_unused_columns(df):
    f = open('./refs/unused_columns.json')
    unused_columns = json.load(f)
    df = df.drop(unused_columns, axis=1)
    return df, unused_columns


# Replace remaining NA with median
# --- if Train set
def fill_NA_median(df, cols_autres):
    dict_median_values = {}
    for col in cols_autres:
        mediane = df[col].median()
        df[col].fillna(value=mediane, inplace=True)
        dict_median_values[col] = mediane
    with open('./refs/median_values.json', 'w', encoding='utf-8') as f:
        json.dump(dict_median_values, f, ensure_ascii=False, indent=4)
    return df


# --- if Test set
def fill_NA_median_from_dict(df, cols_autres):
    f = open('./refs/median_values.json')
    dict_median_values = json.load(f)
    for col in cols_autres:
        mediane = dict_median_values[col]
        df[col].fillna(value=mediane, inplace=True)
    return df


# GLOBAL NA treatment function
def NA_treatment(df, pc_NA_max=0.5, is_training_file=False):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Get columns per type
    cols_OHE, cols_autres = separation_colonnes(df)
    # Fill OHE
    df = fill_NA_colonnes_OHE(df, cols_OHE)
    if is_training_file:
        df, cols_removed = remove_colonnes_NA_superieures_a(df, cols_autres, pc_NA_max)
        cols_autres = set(cols_autres)-set(cols_removed)
        fill_NA_median(df, cols_autres)
        # Listing and export of final columns
        cols_final = df.columns.tolist()
        if 'TARGET' in cols_final :
            cols_final.remove('TARGET')
        with open('./refs/final_columns.json', 'w', encoding='utf-8') as f:
            json.dump(cols_final, f, ensure_ascii=False, indent=4)

    else:
        df, cols_removed = remove_unused_columns(df)
        cols_autres = set(cols_autres) - set(cols_removed)
        fill_NA_median_from_dict(df, cols_autres)
        # Set final columns to match train file
        f = open('./refs/final_columns.json')
        final_columns = json.load(f)
        df = df[final_columns]
    return df


# >>> GLOBAL PREPROCESSING <<<
def preprocess(data_path='./data/', entry_file='application_train.csv', is_training_file=False,
               with_NA_treatment=True, pc_NA_max=0.5):
    df = feature_engineering(data_path, entry_file, is_training_file)
    if with_NA_treatment:
        df = NA_treatment(df, pc_NA_max, is_training_file)
    return df