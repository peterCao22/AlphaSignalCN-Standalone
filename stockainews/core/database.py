"""数据库连接管理模块"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from .config import config
from loguru import logger


# 创建数据库引擎
engine = create_engine(
    config.database_url,
    pool_pre_ping=True,
    pool_size=config.database_pool_size,
    max_overflow=config.database_max_overflow,
    echo=config.debug,
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    获取数据库会话上下文管理器
    
    Usage:
        with get_session() as session:
            # 使用session进行数据库操作
            result = session.query(Model).all()
    
    Yields:
        Session: SQLAlchemy会话对象
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI依赖注入用的数据库会话生成器
    
    Usage:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    Yields:
        Session: SQLAlchemy会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

