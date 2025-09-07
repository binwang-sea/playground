"""
Inference example for OpenAI's GPT-4o-mini model via GenAI Gateway.

Prerequisites:

pip install -U portkey-ai

References:
- https://platform.openai.com/docs/guides/prompt-engineering
- https://docs.portkey.ai/docs/provider-endpoints/chat
"""

import os

import portkey_ai

# NOTE: Fetch this from https://app.portkey.ai/api-keys
api_key = os.environ["PORTKEY_API_KEY"]

# NOTE: Fetch this from https://app.portkey.ai/virtual-keys
virtual_key = os.environ["PORTKEY_OPENAI_VIRTUAL_KEY"]
model_name = "gpt-5"

# https://doordash.atlassian.net/wiki/spaces/Eng/pages/3717661428/GenAI+Gateway+FAQs#Base-URLs
# - In your **laptop**, use `https://cybertron-service-gateway.doordash.team/v1`
# - In your **production microservice**, use `http://cybertron-service-gateway.service.prod.ddsd:8080/v1`
# - In your **Databricks notebook**, use `http://cybertron-service-gateway.svc.ddnw.net:8080/v1`
base_url = os.environ.get("PORTKEY_BASE_URL", "https://cybertron-service-gateway.doordash.team/v1")

client = portkey_ai.Portkey(
    base_url=base_url,
    api_key=api_key,
    virtual_key=virtual_key,
)

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of the United States?",
        },
    ],
)

print(completion.choices[0].message.content)
