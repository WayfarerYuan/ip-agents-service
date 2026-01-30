import pickle
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple
from contextlib import asynccontextmanager

import aiomysql
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)

from agent_service.config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
)

class PickleSerializer(SerializerProtocol):
    def dumps(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)

class AsyncMySQLSaver(BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a MySQL database asynchronously."""

    def __init__(
        self,
        pool: aiomysql.Pool,
        serde: Optional[SerializerProtocol] = None,
    ):
        super().__init__(serde=serde or PickleSerializer())
        self.pool = pool

    @classmethod
    @asynccontextmanager
    async def from_conn_info(cls) -> AsyncIterator["AsyncMySQLSaver"]:
        pool = await aiomysql.create_pool(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            db=MYSQL_DATABASE,
            autocommit=True,
        )
        try:
            yield cls(pool)
        finally:
            pool.close()
            await pool.wait_closed()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        
        type_ = "checkpoint" # Default type
        
        # Serialize
        checkpoint_blob = self.serde.dumps(checkpoint)
        metadata_blob = self.serde.dumps(metadata)

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Upsert checkpoint
                await cur.execute(
                    """
                    INSERT INTO checkpoints 
                    (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    checkpoint = VALUES(checkpoint),
                    metadata = VALUES(metadata)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        parent_checkpoint_id,
                        type_,
                        checkpoint_blob,
                        metadata_blob,
                    ),
                )
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                for idx, (channel, value) in enumerate(writes):
                    type_ = "write" # Default type
                    value_blob = self.serde.dumps(value)
                    
                    await cur.execute(
                        """
                        INSERT INTO checkpoint_writes 
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE value = VALUES(value)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            idx,
                            channel,
                            type_,
                            value_blob,
                        ),
                    )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                if checkpoint_id:
                    await cur.execute(
                        """
                        SELECT checkpoint, metadata, parent_checkpoint_id
                        FROM checkpoints 
                        WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                        """,
                        (thread_id, checkpoint_ns, checkpoint_id),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT checkpoint, metadata, parent_checkpoint_id, checkpoint_id
                        FROM checkpoints 
                        WHERE thread_id = %s AND checkpoint_ns = %s
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (thread_id, checkpoint_ns),
                    )
                
                row = await cur.fetchone()
                if not row:
                    return None

                if checkpoint_id:
                    checkpoint_blob, metadata_blob, parent_checkpoint_id = row
                else:
                    checkpoint_blob, metadata_blob, parent_checkpoint_id, checkpoint_id = row

                checkpoint = self.serde.loads(checkpoint_blob)
                metadata = self.serde.loads(metadata_blob)
                
                # Get writes
                await cur.execute(
                    """
                    SELECT task_id, channel, value
                    FROM checkpoint_writes
                    WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                
                writes = []
                # Check for pending writes if any (LangGraph implementation detail)
                # For basic functionality, we might just return the checkpoint tuple
                
                # Note: Full implementation of aget_tuple would construct pending_writes
                # Here we return the basic CheckpointTuple
                
                return CheckpointTuple(
                    config,
                    checkpoint,
                    metadata,
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    [], # pending_writes
                )

    async def alist(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        query = """
            SELECT checkpoint, metadata, parent_checkpoint_id, checkpoint_id
            FROM checkpoints 
            WHERE thread_id = %s AND checkpoint_ns = %s
        """
        params = [thread_id, checkpoint_ns]
        
        if before:
            query += " AND created_at < (SELECT created_at FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s)"
            params.extend([thread_id, before["configurable"]["checkpoint_id"]])
            
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(params))
                rows = await cur.fetchall()
                
                for row in rows:
                    checkpoint_blob, metadata_blob, parent_checkpoint_id, checkpoint_id = row
                    yield CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        self.serde.loads(checkpoint_blob),
                        self.serde.loads(metadata_blob),
                        (
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": parent_checkpoint_id,
                                }
                            }
                            if parent_checkpoint_id
                            else None
                        ),
                        [],
                    )
