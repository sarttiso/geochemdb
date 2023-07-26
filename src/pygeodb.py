import iolite_tools
import sqlite3
import numpy as np
import pandas as pd
from thefuzz import fuzz, process
import warnings
import re

# %%


# %%


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

    def matchrows_strings(self, table, names, column, score_threshold=98):
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
        idx : array (bool)
            logical indices of length len(names); true for each entry matched
            in the table
        sample_matches_dict : dict
            closest matching sample names in database with scores exceeding the
            threshold as values for keys being the provide matching sample
            names

        """
        # connect to database
        con = sqlite3.connect(self._database_path)
        # read table
        table_df = pd.read_sql_query(f'SELECT * from {table}', con)
        con.close()

        n_names = len(names)

        # if table is empty, return empty idx, sample_matches
        idx = np.zeros(n_names, dtype=bool)
        if len(table_df) == 0:
            return idx, {}

        # table rows to match to
        rows = table_df[column].values

        # matched names
        name_matches = []
        row_matches = []
        for ii, name in enumerate(names):
            # use fuzzy matching to get nearest match and score
            row_match, score = process.extractOne(name, rows)
            # must meet threshold
            if score >= score_threshold:
                name_matches.append(name)
                row_matches.append(row_match)
                idx[ii] = True

        row_match_dict = dict(zip(name_matches, row_matches))
        return idx, row_match_dict

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

    def matchsamples_df(self, df, score_threshold=96.0):
        """
        match samples in a DataFrame with a 'sample' column  to existing
        samples in the database

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a 'sample' column

        Returns
        -------
        df_matched : pd.DataFrame
            df with rows corresponding to matched samples

        """
        samples_unique = df['sample'].unique()
        # now match parsed sample names against database
        idx, sample_match_dict = self.matchrows_strings(
            'Samples',
            samples_unique,
            'name',
            score_threshold=score_threshold)

        # check which sample_names were not matched
        sample_names_matched = list(sample_match_dict.keys())

        # samples with no matches
        samples_not_matched = set(samples_unique) ^ \
            set(sample_names_matched)

        if len(samples_not_matched) > 0:
            warnings.warn('some sample names not matched', UserWarning)
            print(samples_not_matched)

        # keep only rows with matched samples (indexing into df)
        idx_samples = np.array(
            [x in sample_names_matched for x in df['sample']])
        df = df.iloc[idx_samples].copy()
        # add sample column, renamed
        df['sample'] = df['sample'].replace(sample_match_dict)

        return df.copy()

    def insert_row(self, table, columns, values):
        """


        Parameters
        ----------
        table : str
            name of table in which to insert row.
        columns : arraylike
            columns in table to insert new values for.
        values : arraylike
            same length as columns, values for those columns.

        Returns
        -------
        None.

        """
        assert len(columns) == len(values), 'Must have value for each column.'
        # connect to database
        con = sqlite3.connect(self._database_path)
        # create cursor
        cursor = con.cursor()

        # columns as string
        cols_str = ''
        for ii in range(len(columns)-1):
            cols_str = cols_str + columns[ii] + ', '
        cols_str = cols_str + columns[-1]

        # values ? string
        vals_str = ''
        for ii in range(len(values)-1):
            vals_str = vals_str + '?, '
        vals_str = vals_str + '?'

        # sql string
        sql = f'INSERT INTO {table} ({cols_str}) VALUES ({vals_str})'

        # execute sql
        cursor.execute(sql, values)

        # commit
        con.commit()

        # close connection
        con.close()

    def measurements_update(self, df):
        """
        Update matching spot measurements in the database for matching data
        columns. Does not attempt to add missing spot rows or chemistry
        columns.

        Parameters
        ----------
        df : pd.DataFrame
            must have minimally the following columns:
                analysis, quantity, mean, measurement_unit, uncertainty,
                uncertainty_unit
            optionally:
                reference_material

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

    def measurements_add(self, df_analyses, df_measurements):
        """
        Add measurements for new analyses, but don't add samples.

        Parameters
        ----------
        df_analyses : pd.DataFrame
            DataFrame suitable for reference against the Analyses table.    
            must have the following columns:
            analysis, aliquot, sample, date, insturment, technique, material    

        df_measurements : pd.DataFrame
            DataFram suitable for reference against the Measurements table
            must have have the following columns:
                analysis, quantity, mean, measurement_unit, uncertainty,
                uncertainty_unit

        Returns
        -------
        None.

        """
        # make sure that all analyses in df_measurements are also in df_analyses
        assert set(df_analyses['analysis'].unique().tolist()) == \
            set(df_measurements['analysis'].unique().tolist()), \
            """All analyses in df_analyses must be present in
                df_measurements, and vice versa."""

        # distinguish between existing and new analyses for existing samples
        idx, row_match_dict = \
            self.matchrows_strings('Analyses',
                                   df_analyses['analysis'].values,
                                   'name',
                                   score_threshold=99)

        # ignore existing analyses
        if np.sum(idx) > 0:
            warnings.warn('Existing analyses found, ignoring.')
            # remove from df_analyses
            df_analyses = df_analyses.iloc[~idx]
            # remove corresponding rows from df_measurements
            idx_measurements = df_measurements['analysis'].isin(
                list(row_match_dict.keys())).values
            df_measurements = df_measurements.iloc[~idx_measurements]

        # match samples
        df_analyses = self.matchsamples_df(df_analyses)
        # remove measurements with missing samples
        idx = df_measurements['analysis'].isin(df_analyses['analysis'])
        df_measurements = df_measurements.loc[idx]

        # create necessary aliquots
        df_aliquots = df_analyses.drop_duplicates('aliquot')
        idx, aliquot_match_dict =  \
            self.matchrows_strings('Aliquots',
                                   df_aliquots['aliquot'].values,
                                   'name')
        idx_create = np.argwhere(~idx)
        for idx in idx_create:
            cur_values = df_aliquots.iloc[idx][['aliquot',
                                                'material',
                                                'sample']].values.squeeze()
            self.insert_row('Aliquots',
                            ['name', 'material', 'sample'],
                            cur_values)

        # then add analyses
        cols_analyses = ['analysis', 'aliquot',
                         'date', 'instrument', 'technique']
        cols_Analyses = ['name', 'aliquot', 'date', 'instrument', 'technique']
        for ii, row in df_analyses[cols_analyses].iterrows():
            self.insert_row('Analyses', cols_Analyses, row.values)

        # then add measurements
        cols_meas = ['analysis', 'quantity', 'mean', 'measurement_unit',
                     'uncertainty', 'uncertainty_unit', 'reference_material']
        for ii, row in df_measurements[cols_meas].iterrows():
            self.insert_row('Measurements', cols_meas, row.values)

# %% test data


# df = iolite_tools.excel2measurements(['../../../python/iolite_tools/example_data/2023-03_run-5_trace.xlsx'],
#                                      ['2023-03'], [5], 'trace')
df = iolite_tools.excel2measurements(['../../../python/iolite_tools/example_data/2023-03_run-5_U-Pb.xlsx'],
                                     ['2023-03'], [5], 'U-Pb')
df_measurements = iolite_tools.measurements2sql(df, refmat='91500')
df_analyses = iolite_tools.analyses2sql(df, date='2023-03-17',
                                        instrument='Nu Plasma 3D',
                                        technique='LASS ICPMS',
                                        material='zircon')

database_path = '../geochem.db'

df_cols = list(df)

# %%

geochemdb = GeochemDB(database_path)

# %%
# simulate add_measurements
idx, row_match_dict = geochemdb.matchrows_strings('Analyses',
                                                  df_analyses['analysis'].values,
                                                  'name', score_threshold=99)


# %%
geochemdb.measurements_add(df_analyses, df_measurements)
