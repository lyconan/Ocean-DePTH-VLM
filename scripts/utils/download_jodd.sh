#!/bin/bash
# JODD 数据集自动续传下载脚本
# 目标: 335MB (351,311,207 bytes)
# 服务器下载速率不稳定,需要多次续传

set -e

TARGET_URL="https://www.godac.jamstec.go.jp/jedi/j/dataset/download/JediOrganismDetectionDataset.zip"
TARGET_FILE="data/raw/jamstec/JediOrganismDetectionDataset.zip"
EXPECTED_SIZE=351311207  # bytes

MAX_RETRIES=50
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    CURRENT_SIZE=$(stat -c%s "$TARGET_FILE" 2>/dev/null || echo 0)
    PERCENT=$((CURRENT_SIZE * 100 / EXPECTED_SIZE))
    echo "[尝试 $RETRY/$MAX_RETRIES] 当前: $CURRENT_SIZE bytes ($PERCENT%)"
    
    if [ $CURRENT_SIZE -ge $EXPECTED_SIZE ]; then
        echo "✓ 下载完成! 文件大小: $CURRENT_SIZE bytes"
        exit 0
    fi
    
    # 用 curl 续传,超时30分钟
    curl -L -C - -o "$TARGET_FILE" "$TARGET_URL" \
        --connect-timeout 30 \
        --max-time 1800 \
        --retry 3 \
        --retry-delay 10 \
        -w "\nHTTP %{http_code}, Downloaded %{size_download} bytes, Speed %{speed_download} bytes/sec\n" \
        2>&1 || true
    
    NEW_SIZE=$(stat -c%s "$TARGET_FILE" 2>/dev/null || echo 0)
    echo "  下载后: $NEW_SIZE bytes ($((NEW_SIZE * 100 / EXPECTED_SIZE))%)"
    
    if [ $NEW_SIZE -eq $CURRENT_SIZE ]; then
        echo "  没有进展,等待60秒后重试..."
        sleep 60
    fi
    
    RETRY=$((RETRY + 1))
done

echo "达到最大重试次数,但文件可能仍在下载中"
echo "可手动运行此脚本继续续传"
