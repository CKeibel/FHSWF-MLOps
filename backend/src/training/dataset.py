import pandas as pd
from sklearn.model_selection import train_test_split


class AdultIncomeData:
    def __init__(self, config: dict) -> None:
        self.target: str = config["columns"]["target"]
        self.features: list[str] = config["columns"]["features"]

        # Load initial data
        self.origin_data = pd.read_csv(
            config["path"],
            usecols=self.features + [self.target],
            na_values=" ?",
            header=0,
        )

        # Process the data
        self._process_data()

    def _process_data(self) -> None:
        """Process the origin_data to prepare for training."""
        # Clean data and prepare X, y
        df = self._clean_data(self.origin_data)
        X, y = self._prepare_features_target(df)

        # Detect feature types
        self.detect_feature_type(X)

        # Split data
        self._split_data(X, y)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by removing NAs and transforming target."""
        df = df.dropna()
        df[self.target] = df[self.target].apply(lambda x: 1 if x == ">50K" else 0)
        return df

    def _prepare_features_target(self, df: pd.DataFrame) -> tuple:
        """Split dataframe into features (X) and target (y)."""
        X = df.drop(self.target, axis=1)
        y = df[self.target]
        return X, y

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split data into training and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

    def detect_feature_type(self, df: pd.DataFrame) -> None:
        """Detect numerical and categorical features in the dataframe."""
        self.numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
        self.categorical_features = df.select_dtypes(include=["object"]).columns

    def get_categorical_features(self) -> list[str]:
        return self.categorical_features

    def get_nummerical_features(self) -> list[str]:
        return self.numerical_features

    def set_origin_data(self, df: pd.DataFrame) -> None:
        """
        Setter for origin_data that ensures all dependent data is updated.

        Args:
            df: New DataFrame to set as origin_data
        """
        self.origin_data = df
        self._process_data()

    def append_data(self, new_df: pd.DataFrame) -> None:
        """
        Append new data to the existing dataset while preserving structure.

        Args:
            new_df: DataFrame with new data to append
        """
        required_columns = self.features + [self.target]

        # Check if required columns exist in the new dataframe
        missing_columns = [col for col in required_columns if col not in new_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in new data: {missing_columns}")

        # Identify extra columns in the new dataframe
        extra_columns = [col for col in new_df.columns if col not in required_columns]
        if extra_columns:
            print(f"Warning: Ignoring extra columns in new data: {extra_columns}")

        # Check data types for required columns
        for col in required_columns:
            original_dtype = self.origin_data[col].dtype
            new_dtype = new_df[col].dtype

            # For numeric columns, ensure they're numeric
            if pd.api.types.is_numeric_dtype(original_dtype):
                if not pd.api.types.is_numeric_dtype(new_dtype):
                    raise ValueError(
                        f"Data type mismatch for column {col}: Expected numeric, got {new_dtype}"
                    )

            # For categorical/string columns, ensure they're similar
            elif pd.api.types.is_object_dtype(original_dtype):
                if not pd.api.types.is_object_dtype(
                    new_dtype
                ) and not pd.api.types.is_string_dtype(new_dtype):
                    raise ValueError(
                        f"Data type mismatch for column {col}: Expected object/string, got {new_dtype}"
                    )

        # Filter to only include required columns
        filtered_df = new_df[required_columns].copy()

        # Append to origin_data and reprocess
        combined_df = pd.concat([self.origin_data, filtered_df], ignore_index=True)
        self.set_origin_data(combined_df)
