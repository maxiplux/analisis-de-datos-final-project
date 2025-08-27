import unittest
import io
import sys
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import tools

from tools import get_basic_info


class TestGetBasicInfo(unittest.TestCase):
    def setUp(self):
        # Create a small DataFrame with mixed dtypes
        self.df = pd.DataFrame({
            'age': [25, 32, 40, 28],                    # int
            'balance': [1000.50, 250.0, 0.0, 500.75],   # float
            'job': ['admin.', 'technician', 'blue-collar', 'services'],  # object
            'marital': pd.Series(['single', 'married', 'single', 'divorced'], dtype='category')  # category
        })

        # Patch tools.print_markdown to also print to stdout so tests can capture headings
        self._orig_print_markdown = tools.print_markdown
        tools.print_markdown = lambda s: print(s)

    def tearDown(self):
        # Restore original print_markdown after each test
        tools.print_markdown = self._orig_print_markdown

    def test_get_basic_info_returns_expected_summary(self):
        # Capture stdout to ensure print statements exist and are informative
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary = get_basic_info(self.df)
        output = buf.getvalue()

        # Basic structure checks
        self.assertIsInstance(summary, dict)
        for key in ['shape', 'memory_mb', 'numerical_cols', 'categorical_cols', 'dtypes']:
            self.assertIn(key, summary)

        # Shape
        self.assertEqual(summary['shape'], self.df.shape)

        # Memory usage should be a non-negative float
        self.assertIsInstance(summary['memory_mb'], (float, np.floating))
        self.assertGreaterEqual(summary['memory_mb'], 0.0)

        # Numerical columns: age, balance
        self.assertIsInstance(summary['numerical_cols'], list)
        self.assertEqual(set(summary['numerical_cols']), {'age', 'balance'})
        self.assertEqual(len(summary['numerical_cols']), 2)

        # Categorical columns: job (object), marital (category)
        self.assertIsInstance(summary['categorical_cols'], list)
        self.assertEqual(set(summary['categorical_cols']), {'job', 'marital'})
        self.assertEqual(len(summary['categorical_cols']), 2)

        # Dtypes mapping contains expected keys with appropriate dtype kinds
        dtypes = summary['dtypes']
        self.assertIn('age', dtypes)
        self.assertIn('balance', dtypes)
        self.assertIn('job', dtypes)
        self.assertIn('marital', dtypes)

        self.assertTrue(np.issubdtype(dtypes['age'], np.integer))
        self.assertTrue(np.issubdtype(dtypes['balance'], np.floating))
        # object dtype for 'job'
        self.assertEqual(str(dtypes['job']), 'object')
        # category dtype for 'marital'
        self.assertEqual(str(dtypes['marital']), 'category')

        # Check that some expected headings are printed
        self.assertIn('STEP 2: BASIC DATASET INFORMATION', output)
        self.assertIn('Data Type Distribution', output)
        self.assertIn('Column Classification', output)


class TestCheckDataQuality(unittest.TestCase):
    def setUp(self):
        # Construct a DataFrame that triggers various data quality findings
        self.df = pd.DataFrame({
            'age': [30, -1, 30, 40, 50],                 # includes a negative value (-1)
            'duration': [100, 200, 100, np.nan, 300],    # includes a NaN
            'campaign': [1, 2, 1, 1, 3],                 # numeric, no negatives
            'cat': ['Yes', 'yes', 'Yes', None, 'MAYBE'], # mixed case + a missing (None)
            'text': ['hello', '', 'hello', 'world', 'test']  # empty string in object column
        })

        # Patch tools.print_markdown to also print to stdout so tests can capture headings
        self._orig_print_markdown = tools.print_markdown
        tools.print_markdown = lambda s: print(s)

    def tearDown(self):
        # Restore original print_markdown after each test
        tools.print_markdown = self._orig_print_markdown

    def test_check_data_quality_reports_expected_issues(self):
        from tools import check_data_quality

        buf = io.StringIO()
        with redirect_stdout(buf):
            report = check_data_quality(self.df)
        output = buf.getvalue()

        # Structure of the report
        self.assertIsInstance(report, dict)
        self.assertIn('missing_values', report)
        self.assertIn('duplicates', report)
        self.assertIn('empty_strings', report)
        self.assertIn('consistency_issues', report)

        # Missing values: expect 1 in 'duration' and 1 in 'cat'
        mv = report['missing_values']
        self.assertIsInstance(mv, dict)
        self.assertEqual(mv.get('duration', 0), 1)
        self.assertEqual(mv.get('cat', 0), 1)
        self.assertEqual(mv.get('age', 0), 0)
        self.assertEqual(mv.get('campaign', 0), 0)
        self.assertEqual(mv.get('text', 0), 0)

        # Duplicates: the third row duplicates the first -> expect 1 duplicate
        self.assertEqual(report['duplicates'], 1)

        # Empty strings: one empty string in 'text'
        empty = report['empty_strings']
        self.assertEqual(empty, {'text': 1})

        # Consistency issues: mixed case in 'cat' + negative value in 'age'
        issues = report['consistency_issues']
        self.assertTrue(any('cat: Mixed case values detected' in s for s in issues))
        self.assertTrue(any('age: Contains negative values' in s for s in issues))

        # Check some key output headings and warnings are printed
        self.assertIn('DATA QUALITY ASSESSMENT', output)
        self.assertIn('Missing values detected', output)
        self.assertIn('Duplicate Rows Analysis', output)
        self.assertIn('Empty String Analysis', output)
        self.assertIn('Data Consistency Checks', output)


if __name__ == '__main__':
    unittest.main()
