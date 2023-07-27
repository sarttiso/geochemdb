import iolite_tools
import sqlite3
import numpy as np
import pandas as pd
from thefuzz import fuzz, process
import warnings
import re

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
        self.con = sqlite3.connect(self._database_path)
        self.cursor = self.con.cursor()

    def __del__(self):
        """
        destructor, just want to close sqlite connection

        Returns
        -------
        None.

        """
        self.con.close()

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
        # read table
        table_df = pd.read_sql_query(f'SELECT * from {table}', self.con)

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

    def matchrows(self, table, values, columns):
        """
        exactly match rows in a table based on provided values

        Parameters
        ----------
        table : str
            name of the table to match rows into.
        values : arraylike
            array of values to match.
        columns : arraylike
            names of columns in table that contain values; must have same
            length as second dimension of values

        Returns
        -------
        idx : array (bool)
            logical indices of length len(names); true for each row in values
            matched in the table.

        """
        # make columns array
        columns = np.atleast_1d(columns)
        # make values column vector if necessary
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # make sure columns and values have same shapes
        assert len(columns) == values.shape[1], \
            'values.shape[1] should be same as number of comparison columns.'
        # read table
        table_df = pd.read_sql_query(f'SELECT * from {table}', self.con)

        table_arr = table_df[columns].values

        # indices of rows in values that are matched in table_arr
        idx = (values[:, None] == table_arr).all(2).any(1)

        return idx

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
        # get table header
        res = self.cursor.execute(f'PRAGMA table_info("{table}")')
        columns_info = res.fetchall()
        # sqlite columns, not sure what the last 3 are
        cols_sq_df = pd.DataFrame(columns=['id',
                                           'name',
                                           'type',
                                           'c1',
                                           'c2',
                                           'c3'],
                                  data=columns_info)

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
            print(f'Some sample names not matched:\n{samples_not_matched}')

        # keep only rows with matched samples (indexing into df)
        idx_samples = np.array(
            [x in sample_names_matched for x in df['sample']])
        df = df.iloc[idx_samples].copy()
        # add sample column, renamed
        df['sample'] = df['sample'].replace(sample_match_dict)

        return df.copy()

    def insert_rows(self, table, columns, values):
        """


        Parameters
        ----------
        table : str
            name of table in which to insert row.
        columns : arraylike
            columns in table to insert new values for.
        values : list
            must be a list of tuples

        Returns
        -------
        None.

        """
        assert len(columns) == len(values[0]),\
            'Must have value for each column.'

        # columns and values as string
        cols_str = ''
        vals_str = ''
        for ii in range(len(columns)-1):
            cols_str = cols_str + columns[ii] + ', '
            vals_str = vals_str + '?, '
        cols_str = cols_str + columns[-1]
        vals_str = vals_str + '?'

        # sql string
        sql = f'INSERT INTO {table} ({cols_str}) VALUES ({vals_str})'

        # execute sql
        self.cursor.executemany(sql, values)

        # commit
        self.con.commit()

    def update_rows(self, table,
                    match_columns, match_values,
                    update_columns, update_values):
        """
        Update columns in rows in a table based on values in matching columns.

        Parameters
        ----------
        table : str
            Name of table to update rows in.
        match_columns : arraylike
            Columns to do matching on.
        match_values : list
            List of tuples of values to match rows on in match_columns. Length
            of each tuple must be same as len(match_columns)
        update_columns : arraylike
            Columns for which to update values.
        update_values : list
            List of tuples with values to update in update_columns. Length of
            each tuple must be same as len(update_columns)

        Returns
        -------
        None.

        """
        assert len(match_columns) == len(match_values[0]),\
            'Must have match_value for each match_column.'
        assert len(update_columns) == len(update_values[0]),\
            'Must have update_value for each update_column.'
        assert len(update_values) == len(match_values), \
            'update_values and match_values must be same length.'

        # string for columns to update (SET)
        set_str = ''
        for ii in range(len(update_columns)-1):
            set_str = set_str + update_columns[ii] + ' = ?, '
        set_str = set_str + update_columns[-1] + ' = ?'

        # string for columns to match on (WHERE)
        where_str = ''
        for ii in range(len(match_columns)-1):
            where_str = where_str + match_columns[ii] + ' = ? AND '
        where_str = where_str + match_columns[-1] + ' = ?'

        # sql string
        sql = f'UPDATE {table} SET {set_str} WHERE {where_str}'

        # assemble values
        values = [update_value + match_value for
                  update_value, match_value in
                  zip(update_values, match_values)]

        # execute sql
        self.cursor.executemany(sql, values)

        # commit
        self.con.commit()

        return

    def measurements_update(self, df_measurements):
        """
        Update matching spot measurements in the Measurements table for
        matching analyses. Does not attempt to add aliquots, analyses, samples,
        or measurements

        Parameters
        ----------
        df : pd.DataFrame
            Ideally generated by iolite_tools.measurements2sql()
            must have minimally the following columns:
                analysis, quantity, mean, measurement_unit, uncertainty,
                uncertainty_unit
            optionally:
                reference_material

        Returns
        -------
        None.

        """
        # check for basic column structure
        cols_meas = ['analysis', 'quantity', 'mean', 'measurement_unit',
                     'uncertainty', 'uncertainty_unit', 'reference_material']
        assert set(cols_meas) <= set(list(df_measurements)), \
            'Missing columns in df_measurements.'

        # match measurements
        idx = self.matchrows('Measurements',
                             df_measurements[['analysis', 'quantity']].values,
                             ['analysis', 'quantity'])
        # if no matching measurements, stop
        if np.sum(idx) == 0:
            print('No existing measurements found.')
            return

        # keep matched measurements
        df_measurements = df_measurements.loc[idx]
        cols_match = ['analysis', 'quantity']
        match_values = df_measurements[cols_match].values.tolist()
        cols_update = ['mean', 'measurement_unit', 'uncertainty',
                       'uncertainty_unit', 'reference_material']
        update_values = df_measurements[cols_update].values.tolist()

        # update in database
        self.update_rows('Measurements',
                         cols_match,
                         match_values,
                         cols_update,
                         update_values)

        print('Updated:\n' +
              f'{len(df_measurements)} measurements')

    def measurements_add(self, df_measurements, df_analyses, df_aliquots):
        """
        Add measurements for new analyses, but don't add samples.

        Parameters
        ----------
        df_measurements : pd.DataFrame
            DataFram suitable for reference against the Measurements table
            must have have the following columns:
                analysis, quantity, mean, measurement_unit, uncertainty,
                uncertainty_unit
        df_analyses : pd.DataFrame
            DataFrame suitable for reference against the Analyses table.
            must have the following columns:
            analysis, aliquot, date, insturment, technique
        df_aliquots : pd.DataFrame
            DataFrame suitable for reference against the Aliquots table.
            must have the following columns:
            aliquot, sample, material

        Returns
        -------
        None.

        """
        # check for basic column structure
        cols_meas = ['analysis', 'quantity', 'mean', 'measurement_unit',
                     'uncertainty', 'uncertainty_unit', 'reference_material']
        assert set(cols_meas) <= set(list(df_measurements)), \
            'Missing columns in df_measurements.'
        cols_analyses = ['analysis', 'aliquot',
                         'date', 'instrument', 'technique']
        assert set(cols_analyses) <= set(list(df_analyses)), \
            'Missing columns in df_analyses.'
        cols_aliquots = ['aliquot', 'sample', 'material']
        assert set(cols_aliquots) <= set(list(df_aliquots)), \
            'Missing columns in df_aliquots.'

        # make sure that all analyses in df_measurements are also in
        # df_analyses
        assert set(df_analyses['analysis'].unique().tolist()) == \
            set(df_measurements['analysis'].unique().tolist()), \
            """All analyses in df_analyses must be present in
                df_measurements, and vice versa."""

        # make sure that all aliquots in df_analyses are also in df_aliquots
        assert set(df_analyses['aliquot'].unique().tolist()) == \
            set(df_aliquots['aliquot'].unique().tolist()), \
            """All aliquots in df_analyses must be present in
                df_aliquots, and vice versa."""

        # distinguish between existing and new measurements
        idx = self.matchrows('Measurements',
                             df_measurements[['analysis', 'quantity']].values,
                             ['analysis', 'quantity'])

        # if all measurements are already in the database, stop
        if np.sum(idx) == len(df_measurements):
            print('All measurements already in database, use ' +
                  'measurements_update() instead.')
            return

        # ignore existing measurements
        if np.sum(idx) > 0:
            print('Existing measurements found, ignoring.')
            # remove from df_measurements
            df_measurements = df_measurements.iloc[~idx]
            # keep only corresponding analyses
            analyses_unique = df_measurements['analysis'].unique()
            idx = df_analyses['analysis'].isin(analyses_unique).values
            df_analyses = df_analyses.iloc[idx]

        # match samples
        df_aliquots = self.matchsamples_df(df_aliquots)
        # remove analyses for aliquots with missing samples
        idx = df_analyses['aliquot'].isin(df_aliquots['aliquot'])
        df_analyses = df_analyses.loc[idx]
        # remove measurements with missing samples
        idx = df_measurements['analysis'].isin(df_analyses['analysis'])
        df_measurements = df_measurements.loc[idx]

        # create necessary aliquots
        idx_aliquots = ~self.matchrows('Aliquots',
                                       df_aliquots['aliquot'].values,
                                       'aliquot')
        if np.any(idx_aliquots):
            cur_values = df_aliquots.loc[idx_aliquots][cols_aliquots].values
            self.insert_rows('Aliquots',
                             cols_aliquots,
                             cur_values.tolist())

        # create necessary analyses
        idx_analyses = ~self.matchrows('Analyses',
                                       df_analyses['analysis'].values,
                                       'analysis')
        if np.any(idx_analyses):
            cur_values = df_analyses.loc[idx_analyses][cols_analyses].values
            self.insert_rows('Analyses',
                             cols_analyses,
                             cur_values.tolist())

        # then add measurements
        self.insert_rows('Measurements',
                         cols_meas,
                         df_measurements[cols_meas].values.tolist())

        print(f'Added:\n' +
              f'{np.sum(idx_aliquots)} aliquots,\n' +
              f'{np.sum(idx_analyses)} analyses,\n' +
              f'{len(df_measurements)} measurements')

# %% test data


df = iolite_tools.excel2measurements(['../../../python/iolite_tools/example_data/2023-03_run-5_trace.xlsx'],
                                     ['2023-03'], [5], 'trace')
# df = iolite_tools.excel2measurements(['../../../python/iolite_tools/example_data/2023-03_run-5_U-Pb.xlsx'],
#                                      ['2023-03'], [5], 'U-Pb')
df_measurements = iolite_tools.measurements2sql(df, refmat='91500')
df_analyses = iolite_tools.analyses2sql(df, date='2023-03-17',
                                        instrument='Nu Plasma 3D',
                                        technique='LASS ICPMS')
df_aliquots = iolite_tools.aliquots2sql(df, material='zircon')

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
geochemdb.measurements_add(df_measurements, df_analyses, df_aliquots)

# %%
geochemdb.measurements_update(df_measurements)
