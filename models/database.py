#This is using minio for storing model result
#This is a --- IGNORE ---
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import Column, Integer, Float, String, Datetime
db = SQLAlchemy()


class InferenceResult(db.Model):
    __tablename__ = 'inference_results'

    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    result_url = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(Datetime, nullable=False)

