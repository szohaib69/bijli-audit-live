from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
from datetime import datetime
import os


class BillRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    billing_month: str
    units_consumed: float
    total_amount_due: float
    discrepancy_flag: str
    structured_json: str  # Stores full extracted JSON as text
    image_filename: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)  # Stored bill image (media/)
    raw_ocr_text: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bijli_audit.db")
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    SQLModel.metadata.create_all(engine)
    # Lightweight migration: add new columns if the table already exists
    # from an older version of the app.
    _add_column_if_missing("billrecord", "image_filename")
    _add_column_if_missing("billrecord", "image_path")
    _add_column_if_missing("billrecord", "raw_ocr_text")


def _add_column_if_missing(table: str, column: str):
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        names = [row[1] for row in rows]
        if column not in names:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} VARCHAR")
            conn.commit()


def get_session():
    with Session(engine) as session:
        yield session