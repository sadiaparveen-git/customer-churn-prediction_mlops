from typing import List, Tuple

import great_expectations as ge
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    """
    print("🔍 Starting data validation with Great Expectations...")

    # TotalCharges arrives as raw text with ~11 blank rows in this dataset;
    # coerce before the numeric checks below so they evaluate real numbers
    # instead of failing outright on a type mismatch. Doesn't mutate caller's df.
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Build an in-memory GX context + a batch backed directly by this DataFrame
    context = ge.get_context()
    data_source = context.data_sources.add_pandas("telco_pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="telco_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "telco_batch_definition"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = ge.ExpectationSuite(name="telco_validation_suite")

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")

    # Customer identifier must exist (required for business operations)
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="customerID"))
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToNotBeNull(column="customerID")
    )

    # Core demographic features
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="gender"))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="Partner"))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="Dependents"))

    # Service features (critical for churn analysis)
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="PhoneService"))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="InternetService"))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="Contract"))

    # Financial features (key churn predictors)
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="tenure"))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="MonthlyCharges"))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="TotalCharges"))

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    # Gender must be one of expected values (data integrity)
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="gender", value_set=["Male", "Female"]
        )
    )

    # Yes/No fields must have valid values
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="Partner", value_set=["Yes", "No"]
        )
    )
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="Dependents", value_set=["Yes", "No"]
        )
    )
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="PhoneService", value_set=["Yes", "No"]
        )
    )

    # Contract types must be valid (business constraint)
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="Contract",
            value_set=["Month-to-month", "One year", "Two year"],
        )
    )

    # Internet service types (business constraint)
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="InternetService", value_set=["DSL", "Fiber optic", "No"]
        )
    )

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="tenure", min_value=0, max_value=120
        )
    )

    # Monthly charges should be within reasonable business range
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="MonthlyCharges", min_value=0, max_value=200
        )
    )

    # Total charges should be non-negative (business logic)
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="TotalCharges", min_value=0
        )
    )

    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")

    # No missing values in critical numeric features
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToNotBeNull(column="tenure")
    )
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges")
    )

    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    suite.add_expectation(
        ge.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,  # Allow 5% exceptions for edge cases
        )
    )

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = batch.validate(suite)

    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = [
        r.expectation_config.type for r in results.results if not r.success
    ]

    total_checks = len(results.results)
    passed_checks = total_checks - len(failed_expectations)

    if results.success:
        print(
            f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful"
        )
    else:
        print(
            f"❌ Data validation FAILED: {len(failed_expectations)}/{total_checks} checks failed"
        )
        print(f"   Failed expectations: {failed_expectations}")

    return results.success, failed_expectations
