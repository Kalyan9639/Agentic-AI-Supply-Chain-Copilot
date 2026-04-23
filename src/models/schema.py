"""Core data models using Pydantic."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """Represents a scraped news article."""

    id: str = Field(description="Unique identifier")
    title: str = Field(description="Article title")
    content: str = Field(description="Article content in Markdown")
    url: str = Field(description="Source URL")
    published_at: Optional[datetime] = Field(
        default=None, description="Publication date"
    )
    scraped_at: datetime = Field(default_factory=datetime.now)
    source: str = Field(description="Source website name")


class RiskAssessment(BaseModel):
    """AI-generated risk assessment for a news event."""

    risk_level: str = Field(
        description="Risk level: High, Medium, or Low",
        pattern="^(High|Medium|Low)$",
    )
    reasoning: str = Field(description="Why this affects the business")
    proposed_action: Optional[str] = Field(
        default=None,
        description="Recommended action to mitigate the risk",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="AI's confidence in this assessment (0-1)",
    )
    created_at: datetime = Field(default_factory=datetime.now)


class UserConfig(BaseModel):
    """User's business profile and rules."""

    business_name: str = Field(description="Name of the business")
    commodity: str = Field(description="What is traded (e.g., tomatoes, wheat)")
    region: str = Field(description="Primary operating region (e.g., Telangana)")
    rules: List[str] = Field(
        default_factory=list,
        description="Business rules (e.g., 'Never buy transport above Rs 30/kg')",
    )
    created_at: datetime = Field(default_factory=datetime.now)


class ActionProposal(BaseModel):
    """A proposed action from the agent with human approval status."""

    news_id: str = Field(description="ID of the news article that triggered this")
    assessment: RiskAssessment = Field(description="Risk assessment for the news")
    proposed_action: str = Field(description="What the agent proposes to do")
    status: str = Field(
        default="pending",
        pattern="^(pending|approved|rejected)$",
        description="Human approval status",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = Field(default=None)


class LogEntry(BaseModel):
    """Log entry for agent activities."""

    timestamp: datetime = Field(default_factory=datetime.now)
    level: str = Field(description="INFO, WARN, ERROR")
    message: str = Field(description="Log message")
    data: Optional[dict] = Field(default=None, description="Additional context data")
