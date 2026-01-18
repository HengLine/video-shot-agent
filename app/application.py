# -*- coding: utf-8 -*-
"""
@FileName: application.py
@Description: 应用程序主模块 - 负责初始化和配置整个应用
@Author: HengLine
@Time: 2025/10/6
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#
# 导入模型API路由器
from api.shot_api import app as shot_api
from api.index_api import app as index_api
from hengline.context_var import RequestContextMiddleware
from .proxy import router as proxy_router

from config.config import get_data_paths

async def app_startup():
    """
    应用启动时的初始化操作
    """
    # 在这里添加任何需要在应用启动时执行的初始化代码
    data_paths = get_data_paths()
    os.makedirs(data_paths["data_output"], exist_ok=True)
    os.makedirs(data_paths["data_input"], exist_ok=True)
    # os.makedirs(data_paths["model_cache"], exist_ok=True)
    # os.makedirs(data_paths["embedding_cache"], exist_ok=True)

    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    yield


# 创建FastAPI应用
app = FastAPI(
    title="剧本分镜智能体服务",
    description="一个能够将剧本智能拆分为短视频脚本单元的API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.include_router(proxy_router)

# 生产环境应限制为特定域名
cors_config = os.environ.get("APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8000", "*"]
    else:
        cors_config = cors_config.split(";")
else:
    cors_config = ["*"]

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 添加请求上下文中间件
app.add_middleware(RequestContextMiddleware)


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response


app.include_router(index_api, prefix="/api")
app.include_router(shot_api, prefix="/api")
