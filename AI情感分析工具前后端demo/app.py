# -*- coding: utf-8 -*-
"""
AI 情感分析工具 - 后端服务
使用 Flask 提供 Web 页面和情感分析 API，调用 DeepSeek V3 大模型。
"""

import json
import os
import urllib.error
import urllib.request
from flask import Flask, jsonify, request, send_from_directory

# ==================== 配置区域（请在此处修改） ====================

# DeepSeek API 密钥：在 https://platform.deepseek.com 注册并创建
# 建议通过环境变量 DEEPSEEK_API_KEY 设置，也可直接替换下面的字符串
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "输入你已充值好的deepseekAPI 密钥:sk-开头")

# DeepSeek API 地址（OpenAI 兼容格式）
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 使用的模型：deepseek-chat 对应 DeepSeek V3 对话模型
DEEPSEEK_MODEL = "deepseek-chat"

# ==================== Flask 应用初始化 ====================

# 获取当前文件所在目录，用于定位 index.html
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


def call_deepseek_sentiment(text: str) -> dict:
    """
    调用 DeepSeek API 对文本进行情感分析。

    参数:
        text: 待分析的文本内容

    返回:
        包含 sentiment（情感倾向）和 confidence（置信度 0-100）的字典

    异常:
        网络错误或 API 返回异常时抛出 Exception
    """
    # 系统提示词：要求模型以固定 JSON 格式返回结果，便于程序解析
    system_prompt = (
        "你是一个专业的中文情感分析助手。"
        "请分析用户输入文本的情感倾向，并给出置信度。"
        "必须严格以 JSON 格式回复，不要包含任何其他文字或 markdown 标记。"
        '格式示例：{"sentiment": "正面", "confidence": 85}'
        "其中 sentiment 只能是以下三个值之一：正面、负面、中性；"
        "confidence 是 0 到 100 之间的整数，表示你对判断的确信程度。"
    )

    user_prompt = f"请分析以下文本的情感倾向：\n\n{text}"

    # 构造请求体（OpenAI Chat Completions 兼容格式）
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,  # 较低温度，使输出更稳定
        "response_format": {"type": "json_object"},  # 强制 JSON 输出
    }

    # 将 payload 编码为 UTF-8 字节
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # 构造 HTTP 请求
    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # API 返回 4xx/5xx 错误
        error_body = e.read().decode("utf-8", errors="replace")
        raise Exception(f"DeepSeek API 请求失败 (HTTP {e.code}): {error_body}") from e
    except urllib.error.URLError as e:
        # 网络连接问题
        raise Exception(f"无法连接 DeepSeek API，请检查网络: {e.reason}") from e

    # 从响应中提取模型回复的文本内容
    try:
        content = response_data["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise Exception(f"解析 API 响应失败: {response_data}") from e

    # 校验并规范化返回字段
    sentiment = result.get("sentiment", "中性")
    if sentiment not in ("正面", "负面", "中性"):
        sentiment = "中性"

    confidence = result.get("confidence", 50)
    try:
        confidence = int(confidence)
        confidence = max(0, min(100, confidence))  # 限制在 0-100 范围
    except (TypeError, ValueError):
        confidence = 50

    return {"sentiment": sentiment, "confidence": confidence}


# ==================== 路由定义 ====================


@app.route("/")
def index():
    """返回前端页面 index.html"""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    情感分析接口
    接收 JSON: {"text": "待分析文本"}
    返回 JSON: {"sentiment": "正面/负面/中性", "confidence": 85}
    """
    # 检查 API 密钥是否已配置
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "在这里填入你的DeepSeek_API密钥":
        return jsonify({"error": "请先在 app.py 中配置 DeepSeek API 密钥，或设置环境变量 DEEPSEEK_API_KEY"}), 500

    # 解析请求体
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "请输入要分析的文本内容"}), 400

    text = data["text"].strip()

    # 限制文本长度，避免超出 API 限制或产生过高费用
    if len(text) > 5000:
        return jsonify({"error": "文本长度不能超过 5000 字"}), 400

    try:
        result = call_deepseek_sentiment(text)
        return jsonify(result)
    except Exception as e:
        # 将异常信息返回给前端，便于用户排查问题
        return jsonify({"error": str(e)}), 500


# ==================== 启动入口 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("  AI 情感分析工具已启动")
    print("  请在浏览器访问: http://127.0.0.1:5000")
    print("=" * 50)
    # debug=True 方便开发调试，生产环境请改为 False
    app.run(host="127.0.0.1", port=5000, debug=True)
