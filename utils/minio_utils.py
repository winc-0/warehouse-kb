import json
import logging

from minio import Minio

from config.minio_config import minio_config
from processor.import_processor.base import setup_logging

setup_logging()

# 惰性初始化：原来在 import 时就建连接、查 bucket、设策略。
# 一旦 MinIO 不可用（比如虚拟机时钟漂移导致 S3 签名失效），
# 每次 import 都要白等几次失败请求，而且进程会带着一个坏客户端跑到底。
_client = None


def _build_client() -> Minio:
    """建立连接，并确保 bucket 存在、公开读权限已开。"""
    client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        # 是否强制启用HTTPS的加密链接方式，False：使用http；True：使用https
        secure=False,
    )

    # 1. 判断 bucket 是否存在，不存在则创建
    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)

    # 2. 设置 bucket 的公开读权限
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{minio_config.bucket_name}/*",
            },
        ],
    }
    client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))
    return client


def get_minio_client():
    """
    返回可用的 MinIO 客户端。

    成功一次后缓存复用；失败则返回 None 并记日志，**下次调用会重试** ——
    MinIO 短暂不可用不会把进程钉死在一个坏状态上。
    """
    global _client
    if _client is not None:
        return _client

    try:
        _client = _build_client()
    except Exception as e:
        logging.error(f"MinIO连接失败，错误原因：{e}")
        return None
    return _client


if __name__ == '__main__':
    get_minio_client()
