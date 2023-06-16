# pylint: disable=invalid-name, missing-docstring, unspecified-encoding, line-too-long, too-few-public-methods

import logging
import os
import json
import datetime


class BaseConfig(object):
    DEBUG = False
    TESTING = False

    LOGGING_FORMAT = '{"source":"%(name)s","time":"%(asctime)s","severity":"%(levelname)s","message":"%(message)s"}'
    LOGGING_DATEFORMAT = "%Y-%m-%dT%H:%M:%S.0Z"
    LOGGING_LEVEL = logging.DEBUG

    CHUNKGRAPH_INSTANCE_ID = "pychunkedgraph"
    PROJECT_ID = os.environ.get("PROJECT_ID", None)
    CG_READ_ONLY = os.environ.get("CG_READ_ONLY", None) is not None
    PCG_GRAPH_IDS = os.environ.get("PCG_GRAPH_IDS", "").split(",")

    USE_REDIS_JOBS = False

    daf_credential_path = os.environ.get("DAF_CREDENTIALS", None)

    AUTH_TOKEN = None
    if daf_credential_path is not None:
        with open(daf_credential_path, "r") as f:
            AUTH_TOKEN = json.load(f)["token"]

    AUTH_SERVICE_NAMESPACE = "pychunkedgraph"
    VIRTUAL_TABLES = {
        "minnie65_public_v117": {
            "table_id": "minnie3_v1",
            "timestamp": datetime.datetime(
                year=2021,
                month=6,
                day=11,
                hour=8,
                minute=10,
                second=0,
                microsecond=253,
                tzinfo=datetime.timezone.utc,
            ),
        }
    }


class DevelopmentConfig(BaseConfig):
    """Development configuration."""

    USE_REDIS_JOBS = False
    DEBUG = True
    LOGGING_LEVEL = logging.ERROR


class DockerDevelopmentConfig(DevelopmentConfig):
    """Development configuration."""

    USE_REDIS_JOBS = True
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


class DeploymentWithRedisConfig(BaseConfig):
    """Deployment configuration with Redis."""

    USE_REDIS_JOBS = True
    REDIS_HOST = os.environ.get("REDIS_HOST")
    REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


class TestingConfig(BaseConfig):
    """Testing configuration."""

    TESTING = True
    USE_REDIS_JOBS = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False
