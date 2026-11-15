from datetime import date as Date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class TaskSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=100)
    description: str | None = None
    date: Date
    status: Literal["Complete", "In progress", "Done"]


class TaskRecord(TaskSchema):
    id: int = Field(gt=0)


class TaskSummary(BaseModel):
    id: int = Field(gt=0)
    name: str
    date: Date
    status: Literal["Complete", "In progress", "Done"]


class TaskListResponse(BaseModel):
    tasks: list[TaskRecord]
    limit: int = Field(ge=1, le=100)


class TaskMutationResponse(BaseModel):
    message: str
    task: TaskRecord

class TaskUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=3, max_length=100)
    description: str | None = None
    date: Date | None = None
    status: Literal["Complete", "In progress", "Done"] | None = None

    @model_validator(mode="after")
    def validate_non_nullable_fields(self) -> "TaskUpdate":
        for field_name in ("name", "date", "status"):
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null")
        return self

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=3, ge=1, le=10)
    status: Literal["Complete", "In progress", "Done"] | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("La pregunta no puede estar vacía")

        return value.strip()


class ChatResponse(BaseModel):
    answer: str