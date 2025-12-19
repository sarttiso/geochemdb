"""Interface for interacting with a SQLite database containing geochemical data.

This module provides a class, GeochemDB, for interacting with a SQLite database, as well as helper functions for structuring and processing data from the database.
"""

import sqlite3
import numpy as np
import pandas as pd
from thefuzz import process


class GeochemDB:
    """
    assumes a SQLite database with the schema describe in the package
    documentation.
    """

    def __init__(self, database_path):
        """
        Initializes a GeochemDB instance.

        Parameters
        ----------
        database_path (str): Path to the SQLite database.

        Attributes
        ----------
        _database_path (str): Internal storage for the database path.
        con (sqlite3.Connection): SQLite connection object.
        """
        self._database_path = database_path
        self.con = sqlite3.connect(self._database_path)
        self._configure_connection()

    def _configure_connection(self):
        """
        Configure the SQLite connection for better performance.
        """
        cur = self.con.cursor()
        cur.execute('PRAGMA journal_mode=WAL;')
        cur.execute('PRAGMA synchronous=NORMAL;')
        cur.execute('PRAGMA foreign_keys=ON;')
        cur.close()

    def __del__(self):
        """
        destructor, just want to close sqlite connection
        """
        self.con.close()

    def matchrows_strings(self, table, names, column, score_threshold=98):
        """
        match to rows in a table based on a column in the row using strings

        Parameters
        ----------
        table : str
            name of the table to match rows into.
        names : arraylike
            list of names to match in the table
        column : str
            name of column in table to do matching on
        score_threshold : float
            thefuzz score that matching must exceed to be a match

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

        # more flexible input
        names = np.atleast_1d(names)
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
        with self.con:
            with self.con.cursor() as cursor:
                res = cursor.execute(f'PRAGMA table_info("{table}")')
        # res = self.cursor.execute(f'PRAGMA table_info("{table}")')
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
        Match samples in a DataFrame with a 'sample' column  to existing
        samples in the database

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with a 'sample' column

        Returns
        -------
        df_matched : pandas.DataFrame
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
            print(f'Sample names not matched:\n{samples_not_matched}')

        # keep only rows with matched samples (indexing into df)
        idx_samples = np.array(
            [x in sample_names_matched for x in df['sample']])
        df = df.iloc[idx_samples].copy()
        # add sample column, renamed
        df['sample'] = df['sample'].replace(sample_match_dict)

        return df.copy()
    
    def _diagnose_foreign_key_violation(self, table, columns, values):
        """
        Helper to identify which values caused a foreign key violation.

        Parameters
        ----------
        table : str
            name of table in which to insert row.
        columns : arraylike
            columns in table to insert new values for.
        values : list
            must be a list of tuples
        
        Raises
        ------
        ValueError
            with message indicating which foreign key(s) were violated and
            which values were missing.
        """
        cursor = self.con.cursor()
        # Get foreign keys: id, seq, table, from, to, on_update, on_delete, match
        fks = cursor.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        
        error_msgs = []
        
        for fk in fks:
            ref_table = fk[2]
            local_col = fk[3]
            ref_col = fk[4]
            
            # Only check if we are actually inserting into this column
            if local_col in columns:
                # Convert columns to list to ensure .index() works (e.g. if numpy array)
                col_idx = list(columns).index(local_col)
                
                # Extract values for this column from the input
                input_vals = set()
                for row in values:
                    val = row[col_idx]
                    if val is not None:
                        input_vals.add(val)
                
                if not input_vals:
                    continue
                
                input_vals_list = list(input_vals)
                # Chunking to be safe (SQLite limit is often 999 variables)
                chunk_size = 900 
                existing_vals = set()
                
                for i in range(0, len(input_vals_list), chunk_size):
                    chunk = input_vals_list[i:i + chunk_size]
                    placeholders = ','.join('?' for _ in chunk)
                    query = f"SELECT {ref_col} FROM {ref_table} WHERE {ref_col} IN ({placeholders})"
                    res = cursor.execute(query, chunk).fetchall()
                    for r in res:
                        existing_vals.add(r[0])
                
                missing = input_vals - existing_vals
                
                if missing:
                    # Limit output if too many missing
                    missing_list = sorted(list(missing))
                    missing_str = str(missing_list[:10])
                    if len(missing_list) > 10:
                        missing_str += f" ... and {len(missing_list) - 10} more"
                    
                    error_msgs.append(
                        f"Foreign key violation for table '{table}', column '{local_col}'. "
                        f"The following values are missing in referenced table '{ref_table}', column '{ref_col}': {missing_str}"
                    )
        
        if error_msgs:
            raise ValueError("\n".join(error_msgs))

    def insert_rows(self, table, columns, values):
        """
        Insert rows into table.

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
        try:
            with self.con:
                self.con.executemany(sql, values)
        except sqlite3.IntegrityError as e:
            if "FOREIGN KEY constraint failed" in str(e):
                self._diagnose_foreign_key_violation(table, columns, values)
            else:
                raise e


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
        with self.con:
            self.con.executemany(sql, values)

        return

    def measurements_update(self, df_measurements):
        """
        Update matching spot measurements in the Measurements table for
        matching analyses. Does not attempt to add aliquots, analyses, samples,
        or measurements

        Parameters
        ----------
        df : pandas.DataFrame
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

    def measurements_add(self, df_measurements, df_analyses, df_aliquots,
                         score_threshold=98):
        """
        Add measurements for new analyses, but don't add samples.

        Parameters
        ----------
        df_measurements : pandas.DataFrame
            DataFrame suitable for reference against the Measurements table
            must have have the following columns:
                analysis, quantity, mean, measurement_unit, uncertainty,
                uncertainty_unit
        df_analyses : pandas.DataFrame
            DataFrame suitable for reference against the Analyses table.
            must have the following columns:
            analysis, aliquot, date, insturment, technique
        df_aliquots : pandas.DataFrame
            DataFrame suitable for reference against the Aliquots table.
            must have the following columns:
            aliquot, sample, material
        score_threshold : int
            0-100, scoring threshold for matching sample names. defaults to 98 

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

        # match samples
        df_aliquots = self.matchsamples_df(df_aliquots, score_threshold=score_threshold)
        # if no matching samples, stop
        if len(df_aliquots) == 0:
            print('No samples matched.')
            return
        # remove analyses for aliquots with missing samples
        idx = df_analyses['aliquot'].isin(df_aliquots['aliquot'])
        df_analyses = df_analyses.loc[idx]
        # remove measurements with missing samples
        idx = df_measurements['analysis'].isin(df_analyses['analysis'])
        df_measurements = df_measurements.loc[idx]

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

        # create necessary aliquots
        idx_aliquots = ~self.matchrows('Aliquots',
                                       df_aliquots['aliquot'].values,
                                       'aliquot')
        if np.any(idx_aliquots):
            cur_values = df_aliquots.loc[idx_aliquots][cols_aliquots].values
            self.insert_rows('Aliquots',
                             cols_aliquots,
                             cur_values.tolist())
            print(f'Added: {np.sum(idx_aliquots)} aliquots.')

        # create necessary analyses
        idx_analyses = ~self.matchrows('Analyses',
                                       df_analyses['analysis'].values,
                                       'analysis')
        # TODO: check that instrument and technique are valid entries
        if np.any(idx_analyses):
            cur_values = df_analyses.loc[idx_analyses][cols_analyses].values
            self.insert_rows('Analyses',
                             cols_analyses,
                             cur_values.tolist())
            print(f'Added: {np.sum(idx_analyses)} analyses.')

        # verify that measurement quantity and unit are present in QuantitiesMeasurementUnits table
        QuantitiesMeasurementUnits_df = pd.read_sql_query('SELECT * from QuantitiesMeasurementUnits',
                                                          self.con)
        for ii, row in df_measurements[['quantity', 'measurement_unit']].iterrows():
            idx = (QuantitiesMeasurementUnits_df['quantity'] == row['quantity']) & \
                  (QuantitiesMeasurementUnits_df['measurement_unit'] == row['measurement_unit'])
            if not np.any(idx):
                raise ValueError(f"Quantity and measurement_unit pair not in QuantitiesMeasurementUnits table: {row['quantity']}, {row['measurement_unit']}")

        # then add measurements gracefully (close connection if error)
        self.insert_rows('Measurements',
                         cols_meas,
                         df_measurements[cols_meas].values.tolist())

        print(f'Added: {len(df_measurements)} measurements.')
    

    def measurements_by_sample(self, samples):
        """
        return a DataFrame with all measurements corresponding to the requested
        samples

        Parameters
        ----------
        samples : str or arraylike
            sample or samples for which to retrieve measurements

        Returns
        -------
        df : pandas.DataFrame
            all measurements associated with the sample.

        """
        samples = np.atleast_1d(samples)

        # get aliquots matching samples
        if len(samples) == 1:
            sql = f'SELECT aliquot, sample FROM Aliquots WHERE sample = "{samples[0]}"'
        else:
            sql = f'SELECT aliquot, sample FROM Aliquots WHERE sample in {tuple(samples)}'
        df_aliquots = pd.read_sql_query(sql, self.con)
        aliquots = tuple(df_aliquots['aliquot'].values)

        # then get matching analyses and measurements
        sql = f'SELECT analysis, aliquot FROM Analyses WHERE aliquot in {aliquots}'
        df_analyses = pd.read_sql_query(sql, self.con)
        analyses = tuple(df_analyses['analysis'].values)

        sql = f'SELECT * FROM Measurements WHERE analysis in {analyses}'
        df_measurements = pd.read_sql_query(sql, self.con)

        # add aliquot and sample information
        df_analyses = df_analyses.merge(df_aliquots,
                                        how='left',
                                        left_on='aliquot',
                                        right_on='aliquot')
        df_measurements = df_measurements.merge(df_analyses,
                                                how='left',
                                                left_on='analysis',
                                                right_on='analysis')

        return df_measurements

    def measurements_by_aliquot(self, aliquots):
        """
        Return a DataFrame with all measurements corresponding to the requested aliquots.

        Parameters
        ----------
        aliquots : str or arraylike
            aliquot(s) for which to retrieve measurements

        Returns
        -------
        df : pandas.DataFrame
            All measurements associated with the aliquot(s).

        """
        aliquots = np.atleast_1d(aliquots)

        # get samples matching aliquots
        if len(aliquots) == 1:
            sql = f'SELECT aliquot, sample FROM Aliquots WHERE aliquot = "{aliquots[0]}"'
        else:
            sql = f'SELECT aliquot, sample FROM Aliquots WHERE aliquot in {tuple(aliquots)}'
        df_aliquots = pd.read_sql_query(sql, self.con)
        aliquots = tuple(df_aliquots['aliquot'].values)

        # then get matching analyses
        if len(aliquots) == 1:
            sql = f'SELECT analysis, aliquot FROM Analyses WHERE aliquot = "{aliquots[0]}"'
        else:
            sql = f'SELECT analysis, aliquot FROM Analyses WHERE aliquot in {tuple(aliquots)}'
        df_analyses = pd.read_sql_query(sql, self.con)
        analyses = tuple(df_analyses['analysis'].values)

        sql = f'SELECT * FROM Measurements WHERE analysis in {analyses}'
        df_measurements = pd.read_sql_query(sql, self.con)

        # add aliquot and sample information
        df_analyses = df_analyses.merge(df_aliquots,
                                        how='left',
                                        left_on='aliquot',
                                        right_on='aliquot')
        df_measurements = df_measurements.merge(df_analyses,
                                                how='left',
                                                left_on='analysis',
                                                right_on='analysis')

        return df_measurements

    def get_samples(self):
        """List samples in the database.

        Parameters
        ----------
        None.

        Returns
        -------
        samples : array
            Array of sample names in the database.
        """
        sql = 'SELECT name FROM Samples'
        samples = pd.read_sql_query(sql, self.con)
        return samples['name'].values

    def get_aliquots(self):
        """List aliquots in the database.

        Parameters
        ----------
        None.

        Returns
        -------
        aliquots : array
            Array of aliquot names in the database.
        """
        sql = 'SELECT aliquot FROM Aliquots'
        aliquots = pd.read_sql_query(sql, self.con)
        return aliquots['aliquot'].values

    def get_aliquots_samples(self):
        """List samples and aliquots in the database.

        Parameters
        ----------
        None.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with columns 'sample' and 'aliquot'.
        """
        sql = 'SELECT sample, aliquot FROM Aliquots'
        df = pd.read_sql_query(sql, self.con)
        return df

def aliquot_average(df_measurements):
    """
    given a dataframe of measurements as generated by :py:meth:`GeochemDB.measurements_by_sample()` or :py:meth:`GeochemDB.measurements_by_aliquots()`, gather measurements by aliquot, averaging duplicate measurements. Assumes that duplicates have the same units.

    to do:
        implement more robust duplicate checking
        responsible uncertainty propagation

    Args:
        df_measurements (pd.DataFrame): Dataframe of measurements output by :py:meth:`GeochemDB.measurements_by_sample()`.
    Returns:
        pd.DataFrame: DataFrame with geochemical measurements averaged by aliquot.
    """
    df_aliquots = \
        df_measurements.pivot_table(columns=['quantity'],
                                    index=['aliquot', 'sample'],
                                    values=['mean', 'uncertainty'],
                                    aggfunc={'mean': 'mean',
                                             'uncertainty': 'max'})
    df_aliquots = df_aliquots.reorder_levels([1, 0], axis=1)

    return df_aliquots
