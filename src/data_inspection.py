import pandas as pd


class DataInspection:
    """Utility class for inspecting a dataset before preprocessing."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def overview(self):
        """Display basic dataset information."""

        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print(f"Rows: {self.df.shape[0]:,}")
        print(f"Columns: {self.df.shape[1]:,}")

        print("\nColumn Names:")
        print(list(self.df.columns))

    def data_types(self):
        """Display column data types."""

        print("\n" + "=" * 60)
        print("DATA TYPES")
        print("=" * 60)

        print(self.df.dtypes)

    def missing_values(self):
        """Display standard missing values and percentages."""

        print("\n" + "=" * 60)
        print("MISSING VALUES")
        print("=" * 60)

        missing = pd.DataFrame({
            "missing_count": self.df.isnull().sum(),
            "missing_percentage": (
                self.df.isnull().mean() * 100
            ).round(2)
        })

        missing = missing[missing["missing_count"] > 0]

        if missing.empty:
            print("No standard missing values detected.")
        else:
            print(
                missing.sort_values(
                    "missing_count",
                    ascending=False
                )
            )

    def placeholder_values(self, placeholder="unknown"):
        """Inspect placeholder values such as 'unknown'."""

        print("\n" + "=" * 60)
        print(f"PLACEHOLDER VALUES: '{placeholder}'")
        print("=" * 60)

        categorical_columns = self.df.select_dtypes(
            include="object"
        ).columns

        results = []

        for column in categorical_columns:

            count = (self.df[column] == placeholder).sum()

            if count > 0:

                percentage = (count / len(self.df)) * 100

                results.append({
                    "column": column,
                    "count": count,
                    "percentage": round(percentage, 2)
                })

        if results:

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                "count",
                ascending=False
            )

            print(result_df)

            return result_df

        print(f"No '{placeholder}' values detected.")

        return pd.DataFrame()

    def duplicates(self):
        """Display duplicate record count."""

        print("\n" + "=" * 60)
        print("DUPLICATE RECORDS")
        print("=" * 60)

        count = self.df.duplicated().sum()

        print(f"Duplicate rows: {count:,}")

    def numerical_summary(self):
        """Display descriptive statistics for numerical columns."""

        print("\n" + "=" * 60)
        print("NUMERICAL SUMMARY")
        print("=" * 60)

        print(self.df.describe().T)

    def categorical_summary(self):
        """Display value counts for categorical columns."""

        print("\n" + "=" * 60)
        print("CATEGORICAL SUMMARY")
        print("=" * 60)

        categorical_columns = self.df.select_dtypes(
            include="object"
        ).columns

        for column in categorical_columns:

            print(f"\n{column}")
            print("-" * 40)

            print(
                self.df[column].value_counts(
                    dropna=False
                )
            )

    def target_distribution(self, target):
        """Display target class counts and percentages."""

        print("\n" + "=" * 60)
        print(f"TARGET DISTRIBUTION: {target}")
        print("=" * 60)

        counts = self.df[target].value_counts(
            dropna=False
        )

        percentages = (
            self.df[target]
            .value_counts(
                normalize=True,
                dropna=False
            )
            .mul(100)
            .round(2)
        )

        distribution = pd.DataFrame({
            "count": counts,
            "percentage": percentages
        })

        print(distribution)

        return distribution

    def inspect(self, target=None):
        """Run the complete dataset inspection."""

        self.overview()
        self.data_types()
        self.missing_values()
        self.placeholder_values()
        self.duplicates()
        self.numerical_summary()
        self.categorical_summary()

        if target:
            self.target_distribution(target)