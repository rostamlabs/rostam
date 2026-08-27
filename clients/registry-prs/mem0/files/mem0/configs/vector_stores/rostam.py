from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RostamConfig(BaseModel):
    collection_name: str = Field("mem0", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    url: str = Field("http://localhost:8080", description="Rostam server URL")
    api_key: Optional[str] = Field(None, description="API key for Rostam")
    metric: str = Field("cosine", description="Distance metric for vector similarity ('cosine', 'dot', or 'euclidean')")

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. "
                f"Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)
