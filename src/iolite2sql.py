import sqlite3
import numpy as np
import pandas as pd
from thefuzz import fuzz, process
import warnings
import re

# %%


def yyyymm_validator(test_str):
    """
    validate a string in a yyyy-mm format

    Parameters
    ----------
    test_str : str
        string to validate.

    Returns
    -------
    bool
        valid or not

    """
    pattern = r'^\d{4}-\d{2}$'
    return bool(re.match(pattern, test_str))


def parseexcel(excel_path, run_date, run_number):
    """
    prepare a clean DataFrame to then match with the database

    Parameters
    ----------
    excel_path : str
        Path to an excel file with geochemical data exported from iolite4

    run_date : str
        yyyy-mm of the laser run date

    run_number : int
        integer of the the run number

    Returns
    -------
    df : pd.DataFrame
        DataFrame with cleaned and standardized column names

    """
    # load dataframe
    df = pd.read_excel(excel_path, sheet_name='Data')

    # rename first column
    df.rename({'Unnamed: 0': 'spot'}, axis=1, inplace=True)

    # drop any other empty columns (Unnamed)
    cols = list(df)
    idx_drop = np.atleast_1d(np.argwhere(
        ['Unnamed:' in x for x in cols]).squeeze())
    cols_drop = [cols[x] for x in idx_drop]
    df.drop(cols_drop, axis=1, inplace=True)

    # trim column names
    cols = list(df)
    cols_new = [col.replace('_ppm', '') for col in cols]
    cols_new = [col.replace('_mean', '') for col in cols_new]
    cols_new = [col.replace('Final ', '') for col in cols_new]
    cols_new = [col.replace('(prop)', '') for col in cols_new]
    cols_new = [col.replace('(int)', '') for col in cols_new]
    cols_new = [col.replace('Approx_', '') for col in cols_new]
    cols_new = [col.replace('_PPM', '') for col in cols_new]
    cols_new = [col.replace('_2SE', ' 2SE') for col in cols_new]

    # rename columns
    df.rename(columns=dict(zip(cols, cols_new)), inplace=True)

    assert yyyymm_validator(run_date), 'Run date must have yyyy-mm format.'

    df['run date'] = run_date
    df['run number'] = run_number

    return df

# %%


def parsespot(iolite_spot):
    """
    Take an iolite spot string and create a sample name str and a spot name
    str.

    Parameters
    ----------
    iolite_spot : str
        Spot name in iolite.

    Returns
    -------
    sample_name : str
        sample name
    spot_name : str
        spot name

    """
    # count number of underscores
    iolite_spot_split = iolite_spot.split('_')
    n_under = len(iolite_spot_split)

    # likely a standard
    if n_under == 2:
        sample_name, spot_name = iolite_spot_split[0], iolite_spot_split[1]
    # likely an unknown
    elif n_under >= 3:
        sample_name = iolite_spot_split[0:-2]
        spot_name = iolite_spot_split[-2:]

        # join with spaces
        sample_name = ' '.join(sample_name)

        # check for z in spot names and remove it if so
        if 'z' in spot_name[1]:
            spot_name[1] = spot_name[1].replace('z', '')
        # concatenate mount prefix and spot number
        spot_name = ''.join(spot_name)
    return sample_name, spot_name

# %%


class GeochemDB:
    """
    assumes a SQLite database with the schema describe in the package
    documentation.
    """

    def __init__(self, database_path):
        """
        create a GeochemDB instance

        Parameters
        ----------
        database_path : str
            Path to sqlite geochem database.

        Returns
        -------
        None.

        """
        self._database_path = database_path

    def matchrows_strings(self, table, names, column, score_threshold=96.0):
        """
        match to rows in a table based on a column in the row using strings

        Parameters
        ----------
        table : str
            name of table to do row matching in
        names : arraylike
            list of names to match in the table
        column : str
            name of column in table to do matching on


        Returns
        -------
        sample_matches : list
            closest matching sample names in database with scores exceeding the
            threshold

        """
        # connect to database
        con = sqlite3.connect(self._database_path)
        # read table
        table_df = pd.read_sql_query(f'SELECT * from {table}', con)
        con.close()
        # table rows to match to
        rows = table_df[column].values

        # matched names
        name_matches = []
        row_matches = []
        for name in names:
            # use fuzzy matching to get nearest match and score
            row_match, score = process.extractOne(name, rows)
            # must meet threshold
            if score >= score_threshold:
                name_matches.append(name)
                row_matches.append(row_match)

        row_match_dict = dict(zip(name_matches, row_matches))
        return row_match_dict

    def matchcolumns(self, table, df_cols, score_threshold=96.0):
        """
        Match columns of df to columns in the sqlite database

        Parameters
        ----------
        table : str
            Table whose columns to match
        df_cols : arraylike
            Columns to match to columns in sqlite database
        score_threshold : float
            thefuzz score that matching must exceed to be a match

        Returns
        -------
        col_match_dict : dictionary
            dictionary of matches where keys are df_cols and values are the sql
            columns for the matched table

        """
        # connect to database
        con = sqlite3.connect(self._database_path)
        cursor = con.cursor()
        # get table header
        res = cursor.execute(f'PRAGMA table_info("{table}")')
        columns_info = res.fetchall()
        # sqlite columns, not sure what the last 3 are
        cols_sq_df = pd.DataFrame(columns=['id',
                                           'name',
                                           'type',
                                           'c1',
                                           'c2',
                                           'c3'],
                                  data=columns_info)

        con.close()

        sq_cols = cols_sq_df['name'].values

        # matching columns dict lists (to make dict later)
        df_cols_matched = []
        sq_cols_matched = []
        # do matching
        for col in df_cols:
            sq_col_match, score = \
                process.extractOne(col, sq_cols)
            # must meet threshold
            if score >= score_threshold:
                df_cols_matched.append(col)
                sq_cols_matched.append(sq_col_match)

        col_match_dict = dict(zip(df_cols_matched, sq_cols_matched))
        return col_match_dict

    def getsamplespots(self, sample):
        # connect to database
        con = sqlite3.connect(self._database_path)
        # read table
        spots_df = pd.read_sql_query(
            f'SELECT spot from geochemistry where sample = "{sample}"', con)
        con.close()
        # warn if no spots
        if len(spots_df) == 0:
            warnings.warn('no spots for sample', UserWarning)
        return spots_df['spot'].values

    def matchsamples_df(self, df, score_threshold=96.0):
        """
        match samples in a DataFrame with a 'spot' column that is a list of
        iolite spot names to existing samples in the database

        Parameters
        ----------
        df : pd.DataFrame
            data from an iolite session, processed by parseexcel()

        Returns
        -------
        df_matched : pd.DataFrame
            df with rows corresponding to matched samples, adds a 'sample'
            column and the 'spot' column is renamed to just the spot name

        """
        # parse sample and spot names
        sample_names_parsed = []
        sample_names_matched = []
        spot_names = []
        for ii, spot in df['spot'].items():
            sample_name_parsed, spot_name = parsespot(spot)
            sample_names_parsed.append(sample_name_parsed)
            spot_names.append(spot_name)
        # let's just take unique names
        sample_names_parsed_unique = np.unique(np.array(sample_names_parsed))

        # now match parsed sample names against database
        sample_match_dict = self.matchrows_strings(
            'samples',
            sample_names_parsed_unique,
            'name',
            score_threshold=score_threshold)

        # check which sample_names were not matched
        sample_names_matched = list(sample_match_dict.keys())

        # samples with no matches
        samples_not_matched = set(sample_names_parsed) ^ \
            set(sample_names_matched)

        if len(samples_not_matched) > 0:
            warnings.warn('some sample names not matched', UserWarning)
            print(samples_not_matched)

        # keep only rows with matched samples (indexing into df)
        idx_samples = np.argwhere(np.array(
            [x in sample_names_matched for x in sample_names_parsed])).squeeze()
        df = df.iloc[idx_samples].copy()
        # add sample column, renamed
        sample_name_col = [sample_match_dict[sample_names_parsed[x]]
                           for x in idx_samples]
        df.loc[:, 'sample'] = sample_name_col
        # add spot name column
        df.loc[:, 'spot'] = [spot_names[x] for x in idx_samples]

        return df.copy()

    def matchruns(self, run_dates, run_numbers):
        """
        Generate run ids matching lists of run dates and run numbers

        Parameters
        ----------
        run_dates : arraylike
            list of run dates in yyyy-mm form.
        run_numbers : arraylike
            list of run numbers, equal length to run_dates.

        Returns
        -------
        run_ids : arraylike
            run id for each run_date and run_number

        """
        # connect to database
        con = sqlite3.connect(self._database_path)
        # read table
        runs_df = pd.read_sql_query('SELECT * from runs', con)
        con.close()

        run_ids = []
        for run_date, run_number in zip(run_dates, run_numbers):
            assert yyyymm_validator(
                run_date), 'Run date must have yyyy-mm format.'
            idx = (runs_df['date'] == run_date) & \
                (runs_df['run number'] == run_number)
            # if no matches, warn
            if np.sum(idx) == 0:
                raise Exception(f'No match for {run_date}: run {run_number}.')
            elif np.sum(idx) > 1:
                raise Exception(
                    'Multiple matches, please check database for duplicate runs.')
            run_ids.append(runs_df['runID'].values[idx].squeeze())

        return run_ids

    def matchruns_df(self, df):
        """
        use matchruns() to replace 'run date' and 'run number' columns with a 
        'run_id' column

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'run date' and 'run number' columns.

        Returns
        -------
        df_run_matched : pd.DataFrame
            same as df but without 'run date' and 'run number' columns and
            instead the 'runID' column.

        """
        # get unique runs
        runs_df = df[['run date', 'run number']].drop_duplicates()

        # get runIDs
        run_ids = self.matchruns(
            runs_df['run date'].values, runs_df['run number'].values)

        # set runIDs in df
        for ii, (_, group_df) in enumerate(df.groupby(['run date', 'run number'])):
            df.loc[group_df.index, 'runID'] = run_ids[ii]

        # drop run date and number dolumns
        df_run_matched = df.drop(['run date', 'run number'], axis=1)

        return df_run_matched

    def matchspots(self, df):
        """
        Three pieces of information are needed to match to spots in the
        database:
            1. sample name
            2. spot number
            3. runID
        This function returns indices into df of rows for which matching spots
        occur in the database.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with 'sample', 'spot', and 'runID' columns.

        Returns
        -------
        idx : array
            logical indices into df; True for matching spots.

        """
        # connect to database
        con = sqlite3.connect(self._database_path)

        # only necessary information
        df_spots = df[['sample', 'spot', 'runID']].copy()

        # add indexing column
        df_spots.loc[:, 'index'] = np.arange(len(df_spots))

        # runs, most efficient way to iterate
        runIDs = df['runID'].unique()

        # iterate over runs
        idx = np.zeros(len(df), dtype=bool)
        for runID in runIDs:
            cur_run_df = pd.read_sql_query(
                f'SELECT spot, sample from geochemistry where runID = {runID}', con)
            cur_df = df_spots[df_spots['runID'] == runID]

            # indices in df_spots of spots in the database
            cur_run_match_idx = pd.merge(cur_df,
                                         cur_run_df,
                                         on=['sample', 'spot'])['index'].values
            idx[cur_run_match_idx] = True

        # close connection
        con.close()

        return idx

    def updatespots(self, df):
        """
        Update matching spot measurements in the database for matching data
        columns. Does not attempt to add missing spot rows or chemistry
        columns.

        Parameters
        ----------
        df : pd.DataFrame
            must have the following columns: 'spot', 'run date', 'run number'
            ideally is output from parseexcel()

        Returns
        -------
        None.

        """
        # match samples in the df to those in the database
        df = self.matchsamples_df(df)

        # run ids
        df = self.matchruns_df(df)

        # indices of matching spots
        idx = self.matchspots(df)

        # if no spots, exception
        if np.sum(idx) == 0:
            raise Exception('No spots found to update.')

        # focus just on spots to update
        df = df.loc[idx]

        # match columns from df to sqlite (ignore spot matching cols)
        df_cols = list(set(['spot', 'sample', 'runID']) ^ set(list(df)))
        cols_match_dict = self.matchcolumns('geochemistry', df_cols)

        # if no matches, exception
        if len(cols_match_dict) == 0:
            raise Exception('No matching database columns.')

        # finally update data

    def addspots(self, df):
        """
        Add measurements for new spots, but don't add samples.

        Parameters
        ----------
        df : pd.DataFrame
            must have a 'spot' column. ideally is output from parseexcel()

        Returns
        -------
        None.

        """
        # match samples in the df to those in the database
        df = self.matchsamples(df)


# %% test data
df = parseexcel('data/2023-03_run-4_U-Pb.xlsx', '2023-03', 4)
database_path = '../geochem.db'

df_cols = list(df)

# %%

geochemdb = GeochemDB(database_path)

# %%
# simulate updatespots
df = geochemdb.matchsamples_df(df)
df = geochemdb.matchruns_df(df)
idx = geochemdb.matchspots(df)
df = df.loc[idx]
df_cols = list(set(['spot', 'sample', 'runID']) ^ set(list(df)))
cols_match_dict = geochemdb.matchcolumns('geochemistry', df_cols)
