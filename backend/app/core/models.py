from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class EDAOutput(BaseModel):
    shape: Dict[str, int]
    missing: Dict[str, int]
    columns: List[str] = Field(default_factory=list)

class ModelingBest(BaseModel):
    name: Optional[str] = None
    f1: Optional[float] = None
    acc: Optional[float] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    tuned_threshold: Optional[float] = None

class ModelingOutput(BaseModel):
    task: str
    leaderboard: List[Dict[str, Any]]
    best: ModelingBest
    features: Dict[str, int] = Field(default_factory=dict)
    selected_tools: List[str] = Field(default_factory=list)

class ExplainOutput(BaseModel):
    importances: Optional[List[float]] = None
    roc: Optional[str] = None
    pr: Optional[str] = None
    pdp: Optional[List[str]] = None

class ResultPayload(BaseModel):
    phase: Optional[str] = None
    eda: Dict[str, Any]
    modeling: Dict[str, Any]
    explain: Dict[str, Any]
    qa: Dict[str, Any]
    timings: Optional[Dict[str, Any]] = None

