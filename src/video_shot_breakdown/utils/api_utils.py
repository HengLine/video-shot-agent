"""
@FileName: api_utils.py
@Description: 
@Author: HengLine
@Time: 2026/2/3 12:45
"""
import aiohttp


async def check_llm_provider(base_url: str, api_key: str):
    """检查提供商"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
                f"{base_url}/models",
                headers=headers,
                timeout=20
        ) as response:
            if response.status != 200:
                raise ConnectionError(f"OpenAI API returned status {response.status}")

            data = await response.json()
            if "data" not in data:
                raise ValueError("Invalid OpenAI API response")
