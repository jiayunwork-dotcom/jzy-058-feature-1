# 活性污泥 CSTR 稳态求解服务
# 镜像构建完成后容器一跑起来，服务即在 8000 端口对外可用。
FROM python:3.12-slim

# 容器内默认把 SQLite 数据库放到专用数据目录，可用 SLUDGE_DB_PATH 覆盖
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SLUDGE_DB_PATH=/data/sludge.db

WORKDIR /app

# 先装依赖，利用层缓存（含测试依赖，构建末尾用测试把镜像产物固定下来）
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# 拷贝应用与测试
COPY app ./app
COPY tests ./tests
COPY pytest.ini ./pytest.ini

# 构建时跑一遍完整测试：任一测试失败，pytest 非零退出直接中断镜像构建
RUN python -m pytest -q

# 非 root 运行；确保数据目录可写
RUN mkdir -p /data && chown -R nobody:nogroup /data /app
USER nobody

EXPOSE 8000

# 容器自带健康检查（用标准库，不依赖 curl）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json,urllib.request;r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3);assert json.load(r)['status']=='ok'" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
