from pydantic import BaseModel, Field


class AdultIncomeFeatures(BaseModel):
    """Pydantic model for Adult Income dataset features."""

    # Numerical features with validation
    age: int = Field(..., ge=0, description="Age in years")
    educational_num: int = Field(
        ...,
        ge=0,
        alias="educational-num",
        description="Education level (numeric)",
    )
    capital_gain: int = Field(
        ..., ge=0, alias="capital-gain", description="Capital gains"
    )
    capital_loss: int = Field(
        ..., ge=0, alias="capital-loss", description="Capital losses"
    )
    hours_per_week: int = Field(
        ..., ge=0, alias="hours-per-week", description="Hours worked per week"
    )
    fnlwgt: int = Field(..., ge=0, description="Final weight (census weight)")

    # Categorical features
    workclass: str = Field(..., description="Type of employer")
    education: str = Field(..., description="Education level (text)")
    marital_status: str = Field(
        ..., alias="marital-status", description="Marital status"
    )
    occupation: str = Field(..., description="Type of job")
    relationship: str = Field(..., description="Relationship status")
    race: str = Field(..., description="Race")
    gender: str = Field(..., description="Gender")
    native_country: str = Field(
        ..., alias="native-country", description="Native country"
    )

    class Config:
        # Allow field name aliases to handle hyphens in column names
        allow_population_by_field_name = True


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    data: AdultIncomeFeatures


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: float
    model_version: str


class SetModelRequest(BaseModel):
    """Request model for setting model endpoint."""

    alias: str = Field(..., example="best", description="Model alias to set")


class ModelInfo(BaseModel):
    """Model information response model."""

    model_version: str
    alias: str
    run_id: str
    model_name: str
    model_type: str
    metrics: dict[str, float]
