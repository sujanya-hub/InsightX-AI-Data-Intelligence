from __future__ import annotations

import pandas as pd
import numpy as np


class QualityEngine:
    """
    Evaluates dataset quality metrics:
    - Missing value percentage
    - Duplicate row percentage
    - Outlier percentage (IQR method)
    - Overall quality score (0–100)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------------------
    def calculate_missing_percentage(self) -> float:
        total_cells = self.df.size
        if total_cells == 0:
            return 0.0

        missing_cells = self.df.isnull().sum().sum()
        return (missing_cells / total_cells) * 100

    # ------------------------------------------------------------------
    # Duplicate Rows
    # ------------------------------------------------------------------
    def calculate_duplicate_percentage(self) -> float:
        total_rows = len(self.df)
        if total_rows == 0:
            return 0.0

        duplicate_rows = self.df.duplicated().sum()
        return (duplicate_rows / total_rows) * 100

    # ------------------------------------------------------------------
    # Outliers (IQR method for numeric columns)
    # ------------------------------------------------------------------
    def calculate_outlier_percentage(self) -> float:
        numeric_df = self.df.select_dtypes(include=np.number)

        if numeric_df.empty:
            return 0.0

        total_values = numeric_df.size
        outlier_count = 0

        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = ((numeric_df[col] < lower_bound) | 
                        (numeric_df[col] > upper_bound)).sum()

            outlier_count += outliers

        return (outlier_count / total_values) * 100 if total_values > 0 else 0.0

    # ------------------------------------------------------------------
    # Overall Quality Score
    # ------------------------------------------------------------------
    def calculate_quality_score(self) -> float:
        """
        Weighted quality scoring model:
        - Missing penalty: 50%
        - Duplicate penalty: 30%
        - Outlier penalty: 20%
        """

        missing = self.calculate_missing_percentage()
        duplicates = self.calculate_duplicate_percentage()
        outliers = self.calculate_outlier_percentage()

        score = 100 - (0.5 * missing + 0.3 * duplicates + 0.2 * outliers)

        return max(score, 0.0)